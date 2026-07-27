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

from lobj import obj_loss
from module import MLP, SIGReg
from utils import (
    SaveCkptCallback,
    WithEpisodeIdx,
    get_column_normalizer,
    get_img_preprocessor,
    get_q_normalizer,
)


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = [g.pow(2).sum() for g in grads if g is not None]
    if not sq:
        return torch.zeros((), device=loss.device)
    return torch.sqrt(torch.stack(sq).sum())


def latent_health(emb):
    """||z|| stats + effective rank of the batch covariance (collapse alarm)."""
    # bf16 autocast would downcast the matmul below and eigvalsh has no bf16 kernel
    with torch.autocast(device_type=emb.device.type, enabled=False):
        z = emb.detach().reshape(-1, emb.size(-1)).float()
        norms = z.norm(dim=-1)
        zc = z - z.mean(0, keepdim=True)
        cov = (zc.T @ zc) / max(z.size(0) - 1, 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0)
        p = eig / eig.sum().clamp_min(1e-12)
        eff_rank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum())
    return {"z_norm_mean": norms.mean(), "z_norm_std": norms.std(), "eff_rank": eff_rank}


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    lambd_obj = cfg.loss.obj.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    loss = output["pred_loss"]

    if lambd > 0:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        loss = loss + lambd * output["sigreg_loss"]

    # auxiliary q-regression head (information-injection control): same q, same
    # tensor as SIGReg/L_obj, but supervises DECODABILITY instead of geometry.
    # Head lives outside JEPA and is discarded at eval time.
    lambd_aux = cfg.loss.aux.weight
    if lambd_aux > 0:
        q_flat = batch["q"].reshape(-1, batch["q"].size(-1))
        q_hat = self.aux_head(emb.reshape(-1, emb.size(-1)))
        output["aux_loss"] = (q_hat - q_flat).pow(2).mean()
        loss = loss + lambd_aux * output["aux_loss"]

    # L_obj on the encoder-side embedding (same tensor SIGReg sees);
    # gradients never reach the predictor by construction
    if lambd_obj > 0:
        obj, rho, skipped = obj_loss(
            emb,
            batch["q"],
            batch["ep_idx"],
            num_pairs=cfg.loss.obj.num_pairs,
            within_frac=cfg.loss.obj.within_frac,
        )
        output["obj_loss"] = obj
        loss = loss + lambd_obj * obj
        if skipped:
            self._obj_skipped = getattr(self, "_obj_skipped", 0) + 1
        self.log(f"{stage}/obj_rho", rho, on_step=True, sync_dist=True)
        self.log(
            f"{stage}/obj_skipped",
            float(getattr(self, "_obj_skipped", 0)),
            on_step=True,
            sync_dist=True,
        )

    output["loss"] = loss

    # gradient balance between the encoder losses (train only, every N steps)
    if (
        (lambd_obj > 0 or lambd_aux > 0)
        and torch.is_grad_enabled()
        and self.global_step % cfg.log.grad_balance_every == 0
    ):
        enc_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        gn_pred = grad_norm(output["pred_loss"], enc_params)
        metrics = {f"{stage}/grad_pred_enc": gn_pred}
        if lambd_obj > 0:
            gn_obj = grad_norm(lambd_obj * output["obj_loss"], enc_params)
            metrics[f"{stage}/grad_obj_enc"] = gn_obj
            metrics[f"{stage}/grad_ratio_obj_pred"] = gn_obj / gn_pred.clamp_min(1e-12)
        if lambd_aux > 0:
            gn_aux = grad_norm(lambd_aux * output["aux_loss"], enc_params)
            metrics[f"{stage}/grad_aux_enc"] = gn_aux
            metrics[f"{stage}/grad_ratio_aux_pred"] = gn_aux / gn_pred.clamp_min(1e-12)
        self.log_dict(metrics, on_step=True, sync_dist=True)

    if self.global_step % cfg.log.latent_every == 0:
        with torch.no_grad():
            health = latent_health(emb)
        self.log_dict(
            {f"{stage}/{k}": v for k, v in health.items()}, on_step=True, sync_dist=True
        )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def validate_config(cfg):
    if cfg.loss.sigreg.weight > 0 and cfg.model.projector is None:
        raise ValueError(
            "SIGReg requires the MLP projector: a final-LayerNorm embedding lives on "
            "a norm-sqrt(D) shell, incompatible with the isotropic Gaussian target. "
            "This combination is forbidden by design (instructions §4)."
        )
    if cfg.loss.obj.type != "pearson":
        raise NotImplementedError(
            f"obj_loss_type={cfg.loss.obj.type!r} is a future ablation hook; "
            "only 'pearson' is implemented."
        )
    if cfg.loss.get("sigreg_on_pred", False):
        raise NotImplementedError(
            "sigreg_on_pred is a future ablation hook ('version B'); must stay false."
        )

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    validate_config(cfg)

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
