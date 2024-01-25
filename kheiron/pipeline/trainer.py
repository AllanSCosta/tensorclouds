from collections import defaultdict
import pickle
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

# from moleculib.metrics import MetricsPipe

class TrainState(NamedTuple):
    params: Any
    opt_state: Any

@functools.partial(jax.jit, static_argnums=(0,))
def _jit_forward(model, params, keys, batch):
    return inner_split(
        jax.vmap(model, in_axes=(None, 0, 0))(params, keys, inner_stack(batch))
    )

@functools.partial(jax.jit, static_argnums=(0,))
def _jit_metrics(f, outputs, batch):
    outputs, batch, metrics = jax.vmap(f, in_axes=(0,0))(inner_stack(outputs), inner_stack(batch))
    return inner_split(outputs), inner_split(batch), metrics

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
        evaluate_every,
        sample_batch_size,
        # sample_metrics: MetricsPipe,
        save_model: Callable,
        run: Run,
        single_batch: bool = False,
        
        plot_pipe: Callable = None,
        plot_every: int = 1000,
        plot_model: Callable = None,
        # plot_metrics: MetricsPipe = None,

        load_weights: bool = False,
        sample_every: int = None,
        sample_model: Callable = None,
        sample_plot: Callable = None,
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
                collate_fn=lambda x: x,
            ) for split in self.dataset.splits
        }

        self.single_batch = single_batch
        if self.single_batch:
            print('[!!WARNING!!] using single batch')
            sample_batch = next(iter(self.loaders['train']))
            self.loaders = { 'train': [sample_batch] * 1000 }

        self.plot_pipe = plot_pipe
        self.plot_every = plot_every
        self.plot_model = plot_model
        # self.plot_metrics = plot_metrics

        self.evaluate_every = evaluate_every
        self.load_weights = load_weights

        self.sample_every = sample_every
        self.sample_model = sample_model
        self.sample_plot = sample_plot

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
        print(f"Model has {num_params} parameters!")
        self.run.summary["NUM_PARAMS"] = num_params

        return rng_seq, train_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, params, rng, batch, step):
        rng_keys = jax.random.split(rng, len(batch))

        def _apply_losses(params, rng_key, datum: Any, step: int):
            model_output = self.transform.apply(params, rng_key, datum, True)
            return self.losses(rng_key, model_output, datum, step)
        
        output, loss, metrics = jax.vmap(_apply_losses, in_axes=(None, 0, 0, None))(params, rng_keys, inner_stack(batch), step)
        output = inner_split(output)

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
    
    def epoch(
        self,
        train_state,
        rng_seq,
        epoch,
    ) -> Tuple[Dict, Dict]:
        epoch_metrics = defaultdict(list)

        for split in self.dataset.splits:
            if split != 'train' and epoch % self.evaluate_every != 0:
                continue

            loader = self.loaders[split]

            pbar = tqdm(loader, position=1, disable=False)
            pbar.set_description(f"[{self.run.name}] {split}@{epoch}")
            
            batch_size = None

            for step, batch in enumerate(pbar):
                total_step = epoch * len(loader) + step

                # if batch_size is not None and len(batch) != batch_size:
                    # continue
                # if step == 0:
                    # batch_size = len(batch)

                output, new_train_state, metrics = self.update(
                    next(rng_seq), train_state, batch, total_step
                )
                pbar.set_postfix({"loss": f"{metrics['loss']:.3e}"})

                _param_has_nan = lambda agg, p: jnp.isnan(p).any() | agg
                has_nan = tree_reduce(_param_has_nan, new_train_state.params, initializer=False)
                metrics.update(dict(has_nan=has_nan))

                if not has_nan and split == 'train':
                    train_state = new_train_state

                if split == 'train' and total_step % self.save_every == 0:
                    self.save_model(train_state.params)

                if (self.plot_pipe is not None) and split == 'train' and total_step % self.plot_every == 0:
                    self.plot_pipe(self.run, output, batch)

                if (self.sample_every is not None) and split == 'train' and total_step % self.sample_every == 0:
                    keys = jax.random.split(next(rng_seq), 9)
                    samples = jax.vmap(self.sample_model, in_axes=(None, 0))(train_state.params, keys)
                    self.sample_plot(self.run, samples, None)

        for k, v in epoch_metrics.items():
            self.run.log(
                {k: float(np.mean(v))},
                step=epoch,
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