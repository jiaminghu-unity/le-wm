"""DINO-WM baseline training: frozen DINOv2 patch features + causal predictor, no q.

A standalone entry point; train.py, utils.py and every existing config/checkpoint are
untouched. The pipeline is deliberately identical to ours everywhere it can be -- same
lance datasets, same loader/frameskip/history, same 10 epochs, same seed -- so the only
variable against the LeWM arms is the model family. One honest caveat carried into the
reports: DINO-WM's paper trains ~100 epochs; under our 10-epoch budget it may be
undertrained, which biases AGAINST this baseline.

The forward is teacher-forced next-feature prediction: encode all T frames (backbone
frozen, features detached), feed the first T-1 through the block-causal predictor, and
take MSE against the NEXT frame's pixel-patch features -- predictions are compared on
the pixel part only (the action slice of the token is an input, not a target). The
causal mask makes all T-1 shifted comparisons valid at once.

    usage: python train_dinowm.py experiment=dw_pointmaze
"""

import sys
from functools import partial

import hydra
import torch
import torch.nn.functional as F

import train  # noqa: E402  (reuses its dataset pipeline via run.__wrapped__? no -- see below)


def dinowm_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    info = self.model.encode(batch)          # emb (B,T,P,d), pixels_emb (B,T,P,384)
    emb, px = info["emb"], info["pixels_emb"]
    preds = self.model.predict(emb[:, :-1])  # causal: preds[:,t] sees frames <= t
    pdim = px.shape[-1]
    out = {}
    out["pred_loss"] = F.mse_loss(preds[..., :pdim], px[:, 1:].detach())
    out["loss"] = out["pred_loss"]
    if stage == "fit":
        self.log("pred_loss", out["pred_loss"], prog_bar=True, sync_dist=True)
    return out


def _guard_argv():
    exp = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("experiment=")), None)
    if exp is None or not exp.startswith("dw_"):
        raise SystemExit(f"train_dinowm.py takes experiment=dw_*, got {exp!r}")


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # train.py's body builds the LeWM-specific module (SIGReg, aux head, lejepa_forward),
    # so it is not reusable here; this is a sibling entry with the same dataset pipeline.
    import os
    from pathlib import Path

    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.loggers import CSVLogger
    from omegaconf import OmegaConf, open_dict

    from utils import SaveCkptCallback, WithEpisodeIdx, get_column_normalizer, get_img_preprocessor

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
        # same runtime wiring lewm uses for its action encoder, applied to the Embedder
        cfg.model.extra_encoders.action.in_chans = (
            cfg.data.dataset.frameskip * dataset.get_dim("action"))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    dataset = WithEpisodeIdx(dataset)
    g = torch.Generator().manual_seed(cfg.seed)
    tr, va = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=g)
    train_dl = torch.utils.data.DataLoader(tr, **cfg.loader, shuffle=True, drop_last=True, generator=g)
    val_dl = torch.utils.data.DataLoader(va, **cfg.loader, shuffle=False, drop_last=False)

    model = hydra.utils.instantiate(cfg.model)
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    nf = sum(p.numel() for p in model.backbone.parameters())
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dinowm] backbone frozen ({nf/1e6:.1f}M); trainable {nt/1e6:.1f}M "
          f"(predictor + action embedder)", flush=True)
    # regression guard for the plain-dict bug: the action embedder's parameters must be
    # registered (trainable, device-tracked, saved), or the run is not DINO-WM
    assert any(n.startswith("extra_encoders.") for n, _ in model.named_parameters()), (
        "extra_encoders parameters are not registered on the module -- the action "
        "encoder would neither train nor be saved")

    module = spt.Module(
        model=model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={"model_opt": {"modules": "model", "optimizer": dict(cfg.optimizer),
                             "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                             "interval": "epoch"}},
    )
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[SaveCkptCallback(run_name=cfg.output_model_name,
                                                     cfg=cfg.model, epoch_interval=1)],
                         num_sanity_val_steps=1,
                         logger=CSVLogger(save_dir=str(run_dir), name="csv_logs"),
                         enable_checkpointing=True)
    ckpt = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    spt.Manager(trainer=trainer, module=module,
                data=spt.data.DataModule(train=train_dl, val=val_dl),
                ckpt_path=ckpt if ckpt.exists() else None)()


if __name__ == "__main__":
    _guard_argv()
    run()
