"""Stage 2 of automatic planning-state discovery: SCALE trained with the gated
metric d_{Q,g*} discovered by qgate_stage1.py.

Exact-equivalence trick: lobj.obj_loss aligns ||dz||^2 with ||dq||^2 on the
STANDARDIZED q, so replacing batch["q"] with sqrt(g*) ⊙ q makes the alignment
target sum_k g*_k (dq_k)^2 = d_{Q,g*} exactly -- no change to the loss code.
The checkpoint is a plain pixel JEPA, structurally identical to every SCALE arm.

g* comes from $QGATE_JSON (a qgate_stage1 output file; its "g_star" dict is read
in dim order). With $QGATE_JSON unset the scale is identity -- that run IS the
SCALE-All control arm (ungated obj on the full 22-d config q).

train.py / utils.py untouched; cube_full_config is merged into the variant
registry in-process (train_cube_full.py precedent). aux must be 0 (batch["q"] is
rescaled; the aux head would regress the rescaled tensor).

    usage:  QGATE_JSON=/path/qgate_stage1_cube_lam0.01.json \
            python train_qgate2.py experiment=qgate_scale_cube seed=3072
            python train_qgate2.py experiment=qall_scale_cube seed=3072   # control
"""

import json
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

import q_cube_full
import q_cube_noise
import q_native_full
import q_ogb_multi
import q_pointmaze
import q_reacher_full
import q_tworoom
import utils

for _mod in (q_cube_full.Q_VARIANTS_CUBE_FULL, q_native_full.Q_VARIANTS_NATIVE,
             q_tworoom.Q_VARIANTS_TWOROOM, q_pointmaze.Q_VARIANTS_POINTMAZE,
             q_reacher_full.Q_VARIANTS_REACHER_FULL, q_ogb_multi.Q_VARIANTS_OGB_MULTI,
             q_cube_noise.Q_VARIANTS_CUBE_NOISE):
    clash = set(_mod) & set(utils.Q_VARIANTS)
    assert not clash, f"variant collision: {clash}"
    utils.Q_VARIANTS.update(_mod)

from train import lejepa_forward, validate_config  # noqa: E402
from module import SIGReg  # noqa: E402
from utils import (  # noqa: E402
    SaveCkptCallback,
    WithEpisodeIdx,
    get_column_normalizer,
    get_img_preprocessor,
    get_q_normalizer,
)


def _load_gate_scale():
    path = os.environ.get("QGATE_JSON")
    if not path:
        print("[qgate2] QGATE_JSON unset -> identity scale (SCALE-All control arm)", flush=True)
        return None
    payload = json.loads(Path(path).read_text())
    g = torch.tensor(list(payload["g_star"].values()), dtype=torch.float32)
    names = list(payload["g_star"].keys())
    print(f"[qgate2] g* from {path}: " +
          ", ".join(f"{n}={v:.2f}" for n, v in zip(names, g.tolist())), flush=True)
    return g.clamp_min(0).sqrt()  # sqrt: ||d(sqrt(g)q)||^2 == sum g_k dq_k^2


def qgate2_forward(self, batch, stage, cfg, scale):
    if scale is not None:
        s = scale.to(batch["q"].device)
        assert batch["q"].shape[-1] == s.numel(), (batch["q"].shape, s.numel())
        batch["q"] = batch["q"].float() * s
    return lejepa_forward(self, batch, stage, cfg)


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    validate_config(cfg)
    assert cfg.loss.aux.weight == 0, "qgate2 rescales batch['q']; aux head must be off"
    assert cfg.loss.obj.weight > 0, "qgate2 without L_obj is just the baseline"
    scale = _load_gate_scale()

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

    # ---- model / optim (identical to train.py) ----
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
        forward=partial(qgate2_forward, cfg=cfg, scale=scale),
        optim=optimizers,
    )

    # ---- training (identical to train_qdistill.py's proven tail) ----
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
