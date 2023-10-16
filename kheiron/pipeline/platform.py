from datetime import datetime
import functools
import importlib
import json
import pickle
import sys

import numpy as np
import jax.numpy as jnp
import os
from pathlib import Path

import jax
import yaml

from hydra_zen import zen, to_yaml, instantiate, load_from_yaml

from wandb.sdk.wandb_run import Run

import builtins

builtins.bfloat16 = jnp.dtype("bfloat16").type

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export CUDA_VISIBLE_DEVICES=1


def _setup(cfg):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "" if cfg.env.device == "cpu" else str(cfg.env.device)
    )

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if cfg.env.preallocate else 'false'
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    jax.config.update("jax_debug_nans", cfg.env.debug_nans)
    jax.config.update("jax_disable_jit", cfg.env.disable_jit)

    np.random.seed(cfg.trainer.seed)


def _prepare_sbatch_script(cfg):
    script = "#!/bin/bash"
    script += f"#SBATCH --job-name={cfg['env']['name']}"
    script += f"#SBATCH --output={cfg['env']['platform_path']}/output.txt"
    script += f"#SBATCH --error={cfg['env']['platform_path']}/error.txt"
    script += f"#SBATCH --partition=compute"
    script += f"#SBATCH --nodes=1"
    script += f"#SBATCH --ntasks-per-node=1"
    script += f"#SBATCH --cpus-per-task={cfg['env']['cores']}"
    script += f"#SBATCH --mem={cfg['env']['memory']}G"
    script += f"source ~/.bashrc"
    script += f"mamba activate protein-u-net"
    script += f"cd {cfg['env']['platform_path']}/code/"
    script += f"python train.py --platform={cfg['env']['name']}"
    return script


class Platform:
    """
    Abstract launch platform for training and inference
    Stores snapshot of code along with concrete instantiation of config
    Platforms are able to issue instantiated trainers for training
    """

    def __init__(self, name: str, path: Path, run: Run):
        self.name = name
        self.path = path
        self.cfg_path = Path(f"{self.path}/config.yml")
        self.cfg = yaml.safe_load(self.cfg_path.read_text())
        self.run = run

        self.checkpoint_index = 0

    def new_trainer(self):
        print("Instantiating Training Pipeline...")
        print(json.dumps(yaml.safe_load(to_yaml(self.cfg)), indent=4))

        def _train_wrapper(trainer, env, zen_cfg) -> None:
            trainer_ = trainer(run=self.run, save_model=self.save_model)
            trainer_.train()

        train = zen(_train_wrapper, pre_call=_setup)
        train.validate(self.cfg)
        return functools.partial(train, _Zen__cfg=self.cfg)

    def save_model(self, model_params):
        self.checkpoint_index = 0
        with open(str(self.path / f"params_{self.checkpoint_index}.npy"), "wb") as file:
            picklable_params = jax.device_get(model_params)
            pickle.dump(picklable_params, file)
        self.checkpoint_index += 1

    def instantiate_model(self):
        instantiable_cfg = load_from_yaml(self.cfg_path)
        return instantiate(instantiable_cfg.trainer.model)

    def get_params(self, checkpoint_index=-1):
        files = os.listdir(self.path)
        # find last checkpoint file and load it
        checkpoint_files = [f for f in files if f.startswith("params")]
        checkpoint_files.sort(
            key=lambda x: -os.path.getmtime(os.path.join(self.path, x))
        )
        checkpoint_file = checkpoint_files[checkpoint_index]
        with open(str(self.path / checkpoint_file), "rb") as file:
            params = pickle.load(file)
        return params

    def _submit(self):
        # NOTE(Allan): currently deprecated
        script = _prepare_sbatch_script(self.cfg)
        print(script)
        with open(f"{self.path}/sbatch.sh", "w") as f:
            f.write(script)
        os.system(f"sbatch {self.path}/sbatch.sh")
        return
