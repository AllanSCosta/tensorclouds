from collections import defaultdict
import os
import pickle
import shutil
import time
from flax import linen as nn
from typing import Callable, NamedTuple, Tuple, Dict, Any

import random
import jax
from jax.tree_util import tree_reduce
import jax.numpy as jnp
import numpy as np
import optax
import functools

from .utils import inner_stack, clip_grads, inner_split

from wandb.sdk.wandb_run import Run
from torch.utils.data import DataLoader
import torch


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm




class TrainState(NamedTuple):
    params: Any
    opt_state: Any




def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *v: np.stack(v) if type(v[0]) != str else None, *trees)


def tree_unstack(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        learning_rate,
        losses,
        seed,
        dataset,
        num_epochs,
        batch_size,
        num_workers,
        save_every,
        validate_every,
        save_model: Callable = None,
        run: Run = None,
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
        # torch.multiprocessing.set_start_method('spawn')
        self.transform = model

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

        self.name = self.run.name if run else "trainer"
        self.max_grad = 1000.0
        self.loaders = {
            split: DataLoader(
                self.dataset.splits[split],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=lambda x: x,
                shuffle=True,
            ) for split in self.dataset.splits
        }

        if self.batch_size == None:
            self.batch_size = self.dataset.batch_size

        self.train_only = train_only
        self.single_batch = single_batch
        self.single_datum = single_datum

        if self.single_batch:
            print("[!!WARNING!!] using single batch")
            sample_batch = next(iter(self.loaders["train"]))
            self.loaders = {"train": [sample_batch] * 1000}

        elif self.single_datum:
            print("[!!WARNING!!] using single datum")
            sample_batch = next(iter(self.loaders["train"]))
            sample_datum = sample_batch[0]
            # while len(sample_datum.nuc_token[sample_datum.nuc_mask]) < 100:
            #     # Randomly select a new sample datum from the batch
            #     sample_datum = random.choice(sample_batch)
            with open('single_jul22.pkl', 'wb') as file:
                pickle.dump(sample_datum, file)
            print("sample_datum has been saved as a pickle file.")
            sample_batch = [sample_datum] * self.batch_size
            self.loaders = {"train": [sample_batch] * 1000}

        self.plot_pipe = plot_pipe
        self.plot_every = plot_every
        self.plot_model = plot_model
        # self.plot_metrics = plot_metrics

        self.validate_every = validate_every
        self.load_weights = load_weights

        self.sample_every = sample_every
        self.metrics = defaultdict(list)

        self.init()

    def init(self):
        print("Initializing Model...")
        init_datum = next(iter(self.loaders['train']))[0]
        init_datum = [init_datum.to_pytree()] if type(init_datum) != list else [d.to_pytree() for d in init_datum] 

        self.rng_seq = jax.random.key(self.seed)
        self.rng_seq, init_rng = jax.random.split(self.rng_seq)

        def _init(rng, *datum):
            param_rng, _ = jax.random.split(rng)
            params = self.transform.init(param_rng, *datum)["params"]
            opt_state = self.optimizer.init(params)
            return TrainState(
                params,
                opt_state,
            )

        clock = time.time()
        self.train_state = _init(init_rng, *init_datum)
        print("Init Time:", time.time() - clock)
        num_params = sum(
            x.size for x in jax.tree_util.tree_leaves(self.train_state.params)
        )

        print(f"Model has {num_params:.3e} parameters")
        if self.run:
            self.run.summary["NUM_PARAMS"] = num_params

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, params, keys, batch, step):

        def _apply_losses(rng_key, datum: Any):
            model_output = self.transform.apply(
                {"params": params}, *datum, True, rngs={"params": rng_key}
            )
            return self.losses(rng_key, model_output, datum, step)

        output, loss, metrics = jax.vmap(_apply_losses, in_axes=(0, 0))(keys, batch)

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

    def epoch(self, epoch) -> Tuple[Dict, Dict]:

        for split in self.loaders.keys():
            if (self.train_only and split != "train") or (
                split != "train" and epoch % self.validate_every != 0
            ):
                continue

            loader = self.loaders[split]

            pbar = tqdm(loader, position=1, disable=False)
            pbar.set_description(f"[{self.name}] {split}@{epoch}")

            # batch_size = None
            epoch_metrics = defaultdict(list)

            for step, data in enumerate(pbar):

                if len(data) != self.batch_size:
                    continue         
                       
                batch = tree_stack([[d.to_pytree()] if type(d)!= list else [d_.to_pytree() for d_ in d] for d in data])

                total_step = epoch * len(loader) + step

                self.rng_seq, subkey = jax.random.split(self.rng_seq)
                keys = jax.random.split(subkey, len(data))

                batched = batch
                output, new_train_state, step_metrics = self.update(
                    keys, self.train_state, batched, total_step
                )
                output = inner_split(output)
                pbar.set_postfix({"loss": f"{step_metrics['loss']:.3e}"})

                _param_has_nan = lambda agg, p: jnp.isnan(p).any() | agg
                has_nan = tree_reduce(
                    _param_has_nan, new_train_state.params, initializer=False
                )

                step_metrics.update(dict(has_nan=has_nan))

                if not has_nan and split == "train":
                    self.train_state = new_train_state

                if (
                    (self.plot_pipe is not None)
                    and split == "train"
                    and total_step % self.plot_every == 0
                ):
                    self.plot_pipe(self.run, output, batch)

                if (
                    (self.sample_every is not None)
                    and split == "train"
                    and total_step % self.sample_every == 0
                ):
                    sample_metrics = self.run_sample(
                        self.rng_seq, self.train_state, batch, total_step
                    )
                    step_metrics.update({f"{k}": v for k, v in sample_metrics.items()})

                if split == "train":
                    for k, v in step_metrics.items():
                        self.metrics[f"{split}/{k}"].append(float(v))
                    if self.run:
                        self.run.log(
                            {
                                **{
                                    f"{split}/{k}": float(v)
                                    for (k, v) in step_metrics.items()
                                },
                                "step": total_step,
                            }
                        )
                    if self.run and total_step % self.save_every == 0:
                        checkpoint_path = self.run.dir + '/checkpoints'
                        os.makedirs(checkpoint_path, exist_ok=True)
                        self.checkpoint_index = 0 # Currently no rule for checkpointing
                        with open(checkpoint_path + f"/params_{self.checkpoint_index}.npy", "wb") as file:
                            checkpoint = { 'params': jax.device_get(self.train_state.params) }
                            pickle.dump(checkpoint, file)



                for k, v in step_metrics.items():
                    epoch_metrics[k].append(v)

            for k, v in epoch_metrics.items():
                self.metrics[f"{split}/{k}_epoch"].append(float(np.mean(v)))
                if self.run:
                    self.run.log(
                        {
                            **{
                                f"{split}/{k}_epoch": float(np.mean(v))
                                for (k, v) in epoch_metrics.items()
                            },
                            "epoch": epoch,
                        },
                    )

    def train(self) -> Run:
        print("Training...")
        for epoch in tqdm(range(self.num_epochs), position=0):
            self.epoch(epoch=epoch)

    # def run_sample(self, rng_seq, state, batch, step):
    #     keys = jax.random.split(next(rng_seq), min(9, len(batch)))

    #     if hasattr(self, 'sample_params'):
    #         params_ = {**state.params, **self.sample_params}
    #     else:
    #         params_ = state.params

    #     batched = inner_stack(batch[:9])

    #     start = time.time()
    #     samples, trajectories = jax.vmap(
    #         self.sample_model,
    #         in_axes=(None, 0, 0)
    #     )(params_, keys, batched)
    #     end = time.time()
    #     samples = inner_split(samples)

    #     sample_metrics = {}
    #     if step != 0:
    #         sample_metrics.update({'sample_time': end - start})

    #     if self.plot_mode == 'samples':
    #         self.sample_plot(self.run, samples, None)

    #     elif self.plot_mode == 'trajectory':
    #         trajectories = [inner_split(traj) for traj in inner_split(trajectories)]
    #         self.sample_plot(self.run, trajectories, None)

    #     if self.sample_metrics:
    #         sample_metrics_ = defaultdict(list)
    #         for sample, batch in zip(samples, batch):
    #             sample_metrics = self.sample_metrics(sample, batch)
    #             for k, v in sample_metrics.items():
    #                 sample_metrics_[k].append(v)
    #         sample_metrics.update({k: np.mean(v) for k, v in sample_metrics_.items()})

    #     return sample_metrics
