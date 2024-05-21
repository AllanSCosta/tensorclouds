from collections import defaultdict
import pickle
import time
import haiku as hk
from typing import Callable, NamedTuple, Tuple, Dict, Any

import jax
from jax.tree_util import tree_reduce
import numpy as np
import optax
import functools

from ..losses import LossPipe
from .utils import inner_stack, clip_grads, inner_split

from wandb.sdk.wandb_run import Run
from torch.utils.data import DataLoader
from tqdm import tqdm
import jax.numpy as jnp


class TrainState(NamedTuple):
    params: Any
    opt_state: Any


def batch_dict(list_):
    keys = list_[0].keys()
    return {k: jnp.stack([d[k] for d in list_]) for k in keys if list_[0][k] is not None}


class Trainer:

    def __init__(
        self,
        model: hk.Module,
        learning_rate,
        losses: LossPipe,
        seed,
        dataset,
        num_epochs,
        batch_size,
        num_workers,
        save_every,
        validate_every,

        save_model: Callable,
        run: Run,

        single_datum: bool = False,
        single_batch: bool = False,
        train_only: bool = False,

        plot_pipe: Callable = None,
        plot_every: int = 1000,
        plot_model: Callable = None,
        # plot_metrics: MetricsPipe = None,

        load_weights: bool = False,
        sample_every: int = None,
        sample_model: Callable = None,
        sample_params: str = None,
        sample_plot: Callable = None,

        sample_batch_size=None,
        sample_metrics=None,

    ):
        self.model = model
        self.transform = hk.transform(lambda *args: model()(*args))

        self.optimizer = optax.adam(learning_rate, 0.9, 0.999)

        self.losses = losses
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.seed = seed
        self.save_every = save_every
        self.batch_size = batch_size
        self.num_workers = num_workers
        print(f"Batch Size: {self.batch_size}")
        self.save_model = save_model
        self.run = run
        self.max_grad = 1000.0
        self.loaders = {
            split: DataLoader(
                self.dataset.splits[split],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=lambda x: [x_.to_dict() for x_ in x],
            ) for split in self.dataset.splits
        }

        self.train_only = train_only
        self.single_batch = single_batch
        self.single_datum = single_datum

        if self.single_batch:
            print('[!!WARNING!!] using single batch')
            sample_batch = next(iter(self.loaders['train']))
            self.loaders = { 'train': [sample_batch] * 1000 }

        elif self.single_datum:
            print('[!!WARNING!!] using single datum')
            sample_batch = next(iter(self.loaders['train']))
            sample_datum = sample_batch[0]
            sample_batch = [sample_datum] * len(sample_batch)
            self.loaders = { 'train': [sample_batch] * 1000 }

        self.plot_pipe = plot_pipe
        self.plot_every = plot_every
        self.plot_model = plot_model
        # self.plot_metrics = plot_metrics

        self.validate_every = validate_every
        self.load_weights = load_weights

        self.sample_every = sample_every
         
        if sample_every:
            sample_model_transform = hk.transform(lambda *args: sample_model().sample(*args))
            @jax.jit
            def _sample_model(params, key, *args):
                return sample_model_transform.apply(params, key, *args)
            if sample_params:
                with open(sample_params, 'rb') as f:
                    self.sample_params = pickle.load(f)
            self.sample_model = _sample_model
            self.sample_plot = sample_plot
            self.sample_metrics = sample_metrics
            self.sample_conditional = True
            self.plot_mode = 'trajectory'

    def init(self):
        print("Initializing Model...")
        init_datum = self.dataset.splits['train'][0]

        rng_seq = hk.PRNGSequence(self.seed)
        init_rng = next(rng_seq)

        def _init(rng, datum):
            params = self.transform.init(rng, datum)
            opt_state = self.optimizer.init(params)
            return TrainState(
                params,
                opt_state,
            )
    
        train_state = jax.jit(_init)(init_rng, init_datum)
        num_params = hk.data_structures.tree_size(train_state.params)

        print(f"Model has {num_params:.3e} parameters")
        self.run.summary["NUM_PARAMS"] = num_params

        return rng_seq, train_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, params, keys, batch, step):
        def _apply_losses(params, rng_key, datum: Any, step: int):
            model_output = self.transform.apply(params, rng_key, datum, True)
            return self.losses(rng_key, model_output, datum, step)
        
        output, loss, metrics = jax.vmap(_apply_losses, in_axes=(None, 0, 0, None))(params, keys, batch, step)
        loss = jnp.where(jnp.isnan(loss), 0.0, loss)
        metrics = {k: v.mean() for k, v in metrics.items()}
        loss = loss.mean()

        return loss, (output, loss, metrics)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, rng, state, batch, step):
        grad, (output, loss, metrics) = jax.grad(
            lambda params, rng, batch, step: self.loss(params, rng, batch, step),
            has_aux=True,
        )(state.params, rng, batch, step)

        # reduce gradients & metrics
        loss = loss.mean()
        metrics = dict(loss=loss, **metrics)

        # clip gradients
        grad = clip_grads(grad, self.max_grad)

        # update parameters
        updates, opt_state = self.optimizer.update(grad, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        return output, TrainState(params, opt_state), metrics
    
    def run_sample(self, rng_seq, state, batch, step):
        keys = jax.random.split(next(rng_seq), min(9, len(batch)))

        if hasattr(self, 'sample_params'):
            params_ = {**state.params, **self.sample_params}
        else: 
            params_ = state.params

        batched = inner_stack(batch[:9])

        start = time.time()
        samples, trajectories = jax.vmap(
            self.sample_model, 
            in_axes=(None, 0, 0)
        )(params_, keys, batched)
        end = time.time()
        samples = inner_split(samples)
    
        sample_metrics = {}

        if step != 0: 
            sample_metrics.update({'sample_time': end - start})

        if self.plot_mode == 'samples':
            self.sample_plot(self.run, samples, None)

        elif self.plot_mode == 'trajectory':
            trajectories = [inner_split(traj) for traj in inner_split(trajectories)]
            self.sample_plot(self.run, trajectories, None)

        if self.sample_metrics:
            sample_metrics_ = defaultdict(list)
            for sample, batch in zip(samples, batch):
                sample_metrics = self.sample_metrics(sample, batch)
                for k, v in sample_metrics.items():
                    sample_metrics_[k].append(v)
            sample_metrics.update({k: np.mean(v) for k, v in sample_metrics_.items()})

        return sample_metrics

    def epoch(
        self,
        train_state,
        rng_seq,
        epoch,
    ) -> Tuple[Dict, Dict]:

        for split in self.loaders.keys():
            if (self.train_only and split != 'train') or (split != 'train' and epoch % self.validate_every != 0):
                continue
            
            loader = self.loaders[split]

            pbar = tqdm(loader, position=1, disable=False)
            pbar.set_description(f"[{self.run.name}] {split}@{epoch}")
            
            # batch_size = None
            epoch_metrics = defaultdict(list)

            for step, batch in enumerate(pbar):
                if len(batch) != self.batch_size:
                    continue
                total_step = epoch * len(loader) + step
         
                keys = jax.random.split(next(rng_seq), len(batch))
                batched = batch_dict(batch)

                output, new_train_state, metrics = self.update(
                    keys, train_state, batched, total_step
                )
                output = inner_split(output)
                pbar.set_postfix({"loss": f"{metrics['loss']:.3e}"})

                _param_has_nan = lambda agg, p: jnp.isnan(p).any() | agg
                has_nan = tree_reduce(_param_has_nan, new_train_state.params, initializer=False)
                # has_nan = jax.tree_util.tree_map(lambda x: jnp.isnan(x).any(), new_train_state.params)
                # for k, v in has_nan.items(): print(k, v.item())
                # if has_nan:
                    # breakpo'int()
                metrics.update(dict(has_nan=has_nan))

                if not has_nan and split == 'train':
                    train_state = new_train_state

                if (self.plot_pipe is not None) and split == 'train' and total_step % self.plot_every == 0:
                    self.plot_pipe(self.run, output, batch)
                
                if (self.sample_every is not None) and split == 'train' and total_step % self.sample_every == 0:
                    sample_metrics = self.run_sample(rng_seq, train_state, batch, total_step)
                    metrics.update(
                        {f'{k}': v for k, v in sample_metrics.items()}
                    )

                if split == 'train':
                    self.run.log(
                        {
                            **{f'{split}/{k}': float(v) 
                               for (k, v) in metrics.items()},
                            'step':total_step,
                        }
                    )    
                    if total_step % self.save_every == 0:
                        self.save_model(train_state.params)

                for k, v in metrics.items():
                    epoch_metrics[k].append(v)                

            self.run.log(
                {
                    **{ 
                        f'{split}/{k}_epoch': float(np.mean(v)) 
                        for (k, v) in epoch_metrics.items()
                    },
                    'epoch': epoch
                },
            )
            
        return train_state

    def train(self, params=None) -> Run:
        rng_seq, train_state = self.init()
        if params is not None:
            train_state = TrainState(params, train_state.opt_state)
        print("Starting Training Loop...")
        for epoch in tqdm(range(self.num_epochs), position=0):
            train_state = self.epoch(
                train_state=train_state,
                rng_seq=rng_seq,
                epoch=epoch,
            )