import os

import numpy as np

from hydra_zen import builds, make_config

from moleculib.protein.transform import ProteinCrop, ProteinPad
from moleculib.protein.dataset import MonomerDataset

from kheiron.models import SequenceTransformer
from kheiron.losses import MaskedLanguageLoss, LossPipe
from kheiron.pipeline import Trainer
 

from typing import Dict


DEFAULTS = dict(
    seed=42,
    preallocate=False,
    device=0,
    debug_nans=False,
    disable_jit=False,

    data_path='/mas/projects/molecularmachines/db/PDB',
    sequence_length=64,
    max_sequence_length=1024,
    min_sequence_length=32,

    dim=128,
    num_heads=2,
    num_layers=6,
    attn_size=64,
    dropout_rate=0.1,
    widening_factor=2,        

    batch_size=16,
    learning_rate=1e-3,
    num_epochs=100,
    save_every=1000,
    
)

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def prepare_config(
    hparams=dotdict(DEFAULTS),
    substitutes: Dict = {},
    **kwargs,
):
    for k, v in substitutes.items():
        if k not in hparams:
            raise ValueError(f"hparam {k} has no default value")
        hparams[k] = v

    EnvCfg = make_config(
        preallocate=hparams.preallocate,
        device=hparams.device,
        debug_nans=hparams.debug_nans,
        disable_jit=hparams.disable_jit,
    )

    transform = [
        builds(ProteinCrop, crop_size=hparams.sequence_length),
        builds(ProteinPad, pad_size=hparams.sequence_length, random_position=False),
    ]
    
    DatasetCfg = builds(
        MonomerDataset,
        base_path=hparams.data_path,
        attrs="all",
        max_resolution=3.0,
        min_sequence_length=hparams.min_sequence_length,
        max_sequence_length=hparams.max_sequence_length,
        frac=1.0,
        transform=transform,
    )

    ModelCfg = builds(
        SequenceTransformer, 
        dim=hparams.dim,
        num_heads=hparams.num_heads,
        num_layers=hparams.num_layers,
        attn_size=hparams.attn_size,
        dropout_rate=hparams.dropout_rate,
        widening_factor=hparams.widening_factor,        
        zen_partial=True
    )

    loss_list = [
        builds(ResidueCrossEntropyLoss, weight=1.0, start_step=0),
    ]

    LossCfg = builds(LossPipe, loss_list=loss_list)

    TrainerCfg = builds(
        Trainer,
        learning_rate=hparams.learning_rate,
        model=ModelCfg,
        dataset=DatasetCfg,
        batch_size=hparams.batch_size,
        num_workers=(hparams.batch_size),
        losses=LossCfg,
        seed=hparams.seed,
        num_epochs=hparams.num_epochs,
        save_every=hparams.save_every,
        evaluate_every=hparams.evaluate_every,
        sample_batch_size=None,
        sample_metrics=None,
        zen_partial=True,
        source_hash=hparams.source_hash,
    )

    TrainCfg = make_config(
        trainer=TrainerCfg,
        env=EnvCfg,
    )
    return TrainCfg


if __name__ == '__main__':
    # using a temporary dir for demonstration, but use 
    # an actual path when running this
    from kheiron.pipeline.registry import Registry

    print('running example training')
    cfg = prepare_config()
    registry = Registry('test')

    platform = registry.new_platform(cfg)
    trainer = platform.new_trainer()
    trainer()





