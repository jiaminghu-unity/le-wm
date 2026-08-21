"""SCALE with a DISTILLED metric target: pixel LeWM training where L_obj aligns the
embedding's distance profile to the q-only model's learned geometry instead of raw q.

Motivation (family #13 + P4 attribution): the raw-q Euclidean metric is a BAD
planning metric in its pure form on Push-T (oracle cost 51.9; the no-SIGReg L_obj
arm that collapses onto it scores 39.4), while the q-only-input model's embedding of
the SAME 6 numbers -- shaped by prediction loss + SIGReg, eff-rank 38.6 -- plans at
71.2. This run swaps L_obj's alignment target from ||dq||^2 to ||d T(q)||^2, where
T = frozen q-only model (projector output, the planning-validated space). Teacher
output is NOT re-standardized: preserving its metric is the point (Pearson is
globally scale-free). The teacher appears only on the loss-target side; the student
checkpoint is a plain pixel JEPA, structurally identical to SCALE's.

Implementation: train.py is imported, not modified. The only intervention is a
forward wrapper that replaces batch["q"] with T(q) before lejepa_forward runs; the
aux branch would consume the replaced tensor, so aux.weight must be 0.

Teacher ckpt name comes from $TEACHER_CKPT (default: the canonical q-only run).

    usage: same hydra CLI as train.py, e.g.
      TEACHER_CKPT=lewm_q1_qinput_s3072/weights_epoch_10.pt \
      python train_qdistill.py experiment=c10_qdistill data=pusht seed=3072
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from qjepa import QJEPA
from train import lejepa_forward, validate_config
from module import SIGReg
from utils import (
    SaveCkptCallback,
    WithEpisodeIdx,
    get_column_normalizer,
    get_img_preprocessor,
    get_q_normalizer,
)

TEACHER_CKPT_DEFAULT = "lewm_q1_qinput_s3072/weights_epoch_10.pt"


def qdistill_forward(self, batch, stage, cfg, teacher):
    q = batch["q"].float()
    dev = q.device
    if next(teacher.parameters()).device != dev:
        teacher.to(dev)
    with torch.no_grad():
        flat = q.reshape(-1, q.size(-1))
        zt = teacher.projector(teacher.encoder(flat))
        batch["q"] = zt.reshape(*q.shape[:-1], zt.size(-1))
    return lejepa_forward(self, batch, stage, cfg)


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    validate_config(cfg)
    assert cfg.loss.aux.weight == 0, \
        "qdistill replaces batch['q'] with a 192-d teacher embedding; aux head must be off"

    teacher_name = os.environ.get("TEACHER_CKPT", TEACHER_CKPT_DEFAULT)
    teacher = swm.wm.utils.load_pretrained(teacher_name).eval()
    teacher.requires_grad_(False)
    assert isinstance(teacher, QJEPA), type(teacher)
    assert not bool((teacher.q_std == 1).all()), "teacher q_std buffer untrained"
    print(f"[qdistill] teacher = {teacher_name} (frozen; projector-output target)", flush=True)

    # ---- dataset (identical to train.py) ----
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    q_variant = cfg.loss.obj.get("q_variant", "pusht_state")
    q_stats_path = Path(
        swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets"),
        f"{dataset_name}.q_stats.{q_variant}.json",
    )
    transforms.append(get_q_normalizer(dataset, q_stats_path, q_variant))
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    dataset.transform = spt.data.transforms.Compose(*transforms)
    dataset = WithEpisodeIdx(dataset)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True,
                                        drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    # ---- model / optim (identical minus aux head, asserted off) ----
    world_model = hydra.utils.instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(qdistill_forward, cfg=cfg, teacher=teacher),
        optim=optimizers,
    )

    # ---- training (identical) ----
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        logger = CSVLogger(save_dir=str(run_dir), name="csv_logs")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1)],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == "__main__":
    run()
