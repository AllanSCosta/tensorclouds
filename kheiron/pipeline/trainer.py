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

from moleculib.metrics import MetricsPipe
from moleculib.protein.datum import ProteinDatum

import os 

class TrainState(NamedTuple):
    params: Any
    opt_state: Any


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
        sample_metrics: MetricsPipe,
        save_model: Callable,
        run: Run,
        samples_path: str = None,
        source_hash: str = None,
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

        self.evaluate_every = evaluate_every
        self.samples_path = samples_path
        self.source_hash = source_hash
        
    def init(self):
        print("Initializing Model...")
        init_datum = self.dataset.splits['train'][0]
        rng_seq = hk.PRNGSequence(self.seed)
        init_rng = next(rng_seq)

        if self.source_hash is None:

            def _init(rng, datum):
                params = self.transform.init(rng, datum)
                opt_state = self.optimizer.init(params)
                return TrainState(
                    params,
                    opt_state,
                )
            train_state = jax.jit(_init)(init_rng, init_datum)
            
        else: 
            path = f'/mas/projects/molecularmachines/experiments/generative/allanc3/{self.source_hash}/'
            print('LOADING FROM PATH', path)
            files = os.listdir(path)
            # find last checkpoint file and load it
            checkpoint_files = [f for f in files if f.startswith("params")]
            checkpoint_files.sort(
                key=lambda x: -os.path.getmtime(os.path.join(path, x))
            )
            checkpoint_file = checkpoint_files[-1]
            with open(f'{path}{checkpoint_file}', 'rb') as f:
                params = pickle.load(f)
            opt_state = self.optimizer.init(params)

            train_state = TrainState(
                params=params,
                opt_state=opt_state,
            )

        return rng_seq, train_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, params, rng, batch, step):
        rng_keys = jax.random.split(rng, len(batch))

        def _apply_losses(params, rng_key, datum: ProteinDatum, step: int):
            model_output = self.transform.apply(params, rng_key, datum, True)
            return self.losses(rng_key, model_output, datum, step)

        output, loss, metrics = jax.vmap(_apply_losses, in_axes=(None, 0, 0, None))(
            params, rng_keys, inner_stack(batch), step
        )
        output = inner_split(output)

        loss = jnp.where(jnp.isnan(loss), 0.0, loss)
        metrics = {k: v.mean() for k, v in metrics.items()}
        loss = loss.mean()

        return loss, (output, loss, metrics)

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, rng, state, batch, step):
        grad, (_, loss, metrics) = jax.grad(
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

        return TrainState(params, opt_state), metrics
    
    def epoch(
        self,
        train_state,
        rng_seq,
        epoch,
    ) -> Tuple[Dict, Dict]:
        for split in self.dataset.splits:
            if split != 'train' and epoch % self.evaluate_every != 0:
                continue

            loader = self.loaders[split]

            pbar = tqdm(loader, position=1, disable=False)
            pbar.set_description(f"[{self.run.name}] {split}@{epoch}")
            epoch_metrics = defaultdict(list)
            
            for step, batch in enumerate(pbar):
                if split == 'train':
                    total_step = epoch * len(pbar) + step

                new_train_state, metrics = self.update(
                    next(rng_seq), train_state, batch, total_step
                )
                pbar.set_postfix({"loss": f"{metrics['loss']:.3e}"})

                _param_has_nan = lambda agg, p: jnp.isnan(p).any() | agg
                has_nan = tree_reduce(_param_has_nan, new_train_state.params, initializer=False)
                metrics.update(dict(has_nan=has_nan))

                for k, v in metrics.items():
                    epoch_metrics[k].append(float(v))

                if not has_nan and split == 'train':
                    train_state = new_train_state

                if split == 'train':
                    self.run.log(
                        {k: float(v) for (k, v) in metrics.items()},
                        step=total_step,
                    )

                if split == 'train' and total_step % self.save_every == 0:
                    self.save_model(train_state.params)
            
            for k, v in epoch_metrics.items():
                self.run.track(
                    {f'{split}/{k}': float(np.mean(v))},
                    step=total_step,
                )
                
        return train_state

    def train(self) -> Run:
        rng_seq, train_state = self.init()
        print("Starting Training Loop...")
        for epoch in tqdm(range(self.num_epochs), position=0):
            train_state = self.epoch(
                train_state=train_state,
                rng_seq=rng_seq,
                epoch=epoch,
            )
