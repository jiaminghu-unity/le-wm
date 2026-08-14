"""Frozen-encoder ablation: retrain only the predictor on a fixed embedding space.

A standalone entry point so that train.py, its config, and every existing checkpoint
stay untouched. Everything except the model-construction step is imported from
train.py, so the two share one definition of the forward pass, the losses, the
dataset pipeline and the logging.

Why this experiment. In the original runs SIGReg, L_obj and the aux head all act on
emb = projector(encoder(x)), and the prediction MSE trains everything end to end, so
each arm's predictor grew up chasing a differently-moving representation. Freezing
encoder+projector removes that confound: all three arms then train a predictor from
scratch under the same objective, and the frozen space is the only thing that differs.

Writing A_m for a run from this file and orig_m for the co-trained model, the original
gap splits exactly:

    (orig_obj - orig_base) = (A_obj - A_base) + (delta_obj - delta_base)
                              \_ space _/       \_ predictor _/

with delta_m = orig_m - A_m. Both terms are measurable without any alignment map: the
space term compares two identically-trained predictors, and delta_m compares two
predictors on a bit-identical encoder, which makes even absolute rollerr comparable
(z_true, z_goal and the pairwise-distance scale are the same numbers).

    usage: python train_frozen.py experiment=fz_obj_cube
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

from module import MLP, SIGReg
from train import lejepa_forward, validate_config
from utils import (
    SaveCkptCallback,
    WithEpisodeIdx,
    get_column_normalizer,
    get_img_preprocessor,
    get_q_normalizer,
)


def validate_frozen(cfg):
    """Refuse the configurations that would silently not be this experiment."""
    if not cfg.get("init_encoder_from"):
        raise ValueError(
            "init_encoder_from is required: freezing a randomly initialised encoder "
            "would measure nothing.")
    for name in ("sigreg", "obj", "aux"):
        w = cfg.loss[name].weight
        if w != 0:
            raise ValueError(
                f"loss.{name}.weight must be 0 here (got {w}). It acts on "
                f"emb = projector(encoder(x)), which is frozen, so its gradient goes "
                f"nowhere -- it would still be computed, still enter the logged total, "
                f"and still cost time while training nothing, and it would stop the "
                f"three arms' predictor training from being identical.")


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_frozen")
def run(cfg):
    validate_config(cfg)
    validate_frozen(cfg)

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    # q must be built from RAW physical columns, so this runs before the
    # z-score normalizers below overwrite them in place
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
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    dataset = WithEpisodeIdx(dataset)  # L_obj pair sampling needs per-sample episode ids

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # ---------------------- the only thing this file adds ----------------------
    # Copy encoder+projector out of a trained checkpoint and hold them fixed.
    # Those two are exactly what z is (z = projector(encoder(x))) and exactly what
    # every regulariser acts on, so freezing both pins the embedding space bit for
    # bit. predictor, pred_proj and action_encoder stay freshly initialised: each arm
    # then trains its predictor from scratch under an IDENTICAL objective (prediction
    # MSE alone), leaving the frozen space as the single variable.
    src = swm.wm.utils.load_pretrained(cfg.init_encoder_from)
    world_model.encoder.load_state_dict(src.encoder.state_dict())
    world_model.projector.load_state_dict(src.projector.state_dict())
    del src
    print(f"[init] encoder+projector <- {cfg.init_encoder_from}; "
          f"predictor/pred_proj/action_encoder freshly initialised", flush=True)

    world_model.encoder.requires_grad_(False)
    world_model.projector.requires_grad_(False)
    n_frozen = sum(p.numel() for p in world_model.encoder.parameters()) + \
               sum(p.numel() for p in world_model.projector.parameters())
    n_train = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
    print(f"[freeze] encoder+projector frozen ({n_frozen/1e6:.2f}M params); "
          f"trainable {n_train/1e6:.2f}M (predictor + pred_proj + action_encoder)",
          flush=True)
    # ---------------------------------------------------------------------------

    # auxiliary q-regression head (probing-MLP architecture); lives on the
    # training wrapper, not inside JEPA — checkpoints stay identical to C1
    aux_head = None
    if cfg.loss.aux.weight > 0:
        import json as _json
        q_dim = len(_json.loads(Path(q_stats_path).read_text())["mean"])
        aux_head = MLP(
            input_dim=cfg.embed_dim, hidden_dim=cfg.loss.aux.hidden,
            output_dim=q_dim, norm_fn=None, act_fn=torch.nn.ReLU,
        )

    optimizers = {
        'model_opt': {
            "modules": 'model|aux_head' if aux_head is not None else 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module_kwargs = dict(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    if aux_head is not None:
        module_kwargs["aux_head"] = aux_head
    world_model = spt.Module(**module_kwargs)

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        # the monitoring in lejepa_forward (obj_rho, eff_rank, grad balance) has
        # to land somewhere even without wandb
        logger = CSVLogger(save_dir=str(run_dir), name="csv_logs")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return



if __name__ == "__main__":
    run()
