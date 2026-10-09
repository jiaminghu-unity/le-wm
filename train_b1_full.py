"""B1 full training — frozen per configs/p5_protocol_freeze.md (100k steps).
Resume-capable: picks up the latest outputs/b1/ckpt_last.pt automatically.
Run from ~/ContactWorld on the trainer VM.
"""
import json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from touchwm.lewm_adapter import MultiViewJEPA
from touchwm.data_adapter import MultiViewClipDataset

OUT = "outputs/b1"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"
TOTAL, BS, VAL_EVERY, LOG_EVERY, CKPT_EVERY = 100_000, 32, 2_500, 500, 10_000
torch.manual_seed(3072); np.random.seed(3072)

train_ds = MultiViewClipDataset("data/demo_data/insertion_usb",
                                "outputs/preflight/usb/splits.json", "train", clip_len=3)
val_ds = MultiViewClipDataset("data/demo_data/insertion_usb",
                              "outputs/preflight/usb/splits.json", "val", clip_len=6)
rng = np.random.RandomState(0)
VAL_IDX = rng.choice(len(val_ds), size=64, replace=False)
val_batch = None  # lazy-built once

dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=6,
                drop_last=True, persistent_workers=True, prefetch_factor=4)
model = MultiViewJEPA().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

step, best_val = 0, float("inf")
if os.path.exists(f"{OUT}/ckpt_last.pt"):
    st = torch.load(f"{OUT}/ckpt_last.pt", map_location=dev)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    step, best_val = st["step"], st.get("best_val", float("inf"))
    print(f"[resume] step {step} best_val {best_val}", flush=True)

def gnorm(mod):
    s = 0.0
    for p in mod.parameters():
        if p.grad is not None: s += p.grad.detach().pow(2).sum().item()
    return round(s ** 0.5, 4)

def build_val():
    items = [val_ds[int(i)] for i in VAL_IDX]
    return {k: torch.stack([it[k] for it in items]).to(dev)
            for k in ("front", "wrist", "action")}

@torch.no_grad()
def validate():
    global val_batch
    if val_batch is None: val_batch = build_val()
    model.eval()
    obs = {"front": val_batch["front"], "wrist": val_batch["wrist"]}
    z = model.encode(obs)                              # (64, 7, 192)
    res = {}
    # one-step teacher-forced
    pred = model.predict(z[:, :-1], val_batch["action"])
    res["val_pred_mse_h1"] = float((pred - z[:, 1:]).pow(2).mean())
    res["persist_mse_h1"] = float((z[:, :-1] - z[:, 1:]).pow(2).mean())
    # multi-step autoregressive from first 3 frames
    H = 3
    zz = z[:, :H].clone(); acts = val_batch["action"]
    for h, hor in enumerate(range(H, z.shape[1])):
        pr = model.predict(zz[:, -H:], acts[:, hor-H:hor])[:, -1:]
        zz = torch.cat([zz, pr], 1)
    for h in (1, 2, 4):                                 # horizons beyond context
        t = H - 1 + h
        if t < z.shape[1]:
            res[f"val_roll_mse_h{h}"] = float((zz[:, t] - z[:, t]).pow(2).mean())
            res[f"persist_roll_mse_h{h}"] = float((z[:, H-1] - z[:, t]).pow(2).mean())
    # collapse metrics
    Zf = z.reshape(-1, z.shape[-1])
    var = Zf.var(0); cov = torch.cov(Zf.T)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0); p = eig / eig.sum()
    res["latent_var_mean"] = float(var.mean()); res["latent_var_min"] = float(var.min())
    res["eff_rank"] = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    model.train()
    return res

log_f = open(f"{OUT}/train_log.jsonl", "a")
model.train(); t0 = time.time()
while step < TOTAL:
    for b in dl:
        obs = {"front": b["front"].to(dev, non_blocking=True),
               "wrist": b["wrist"].to(dev, non_blocking=True)}
        out = model.loss(obs, b["action"].to(dev))
        opt.zero_grad(); out["loss"].backward()
        rec_g = None
        if step % VAL_EVERY == 0:
            rec_g = {"g_vit": gnorm(model.mv_encoder.vit),
                     "g_ca_f": gnorm(model.mv_encoder.ca_f),
                     "g_ca_w": gnorm(model.mv_encoder.ca_w),
                     "g_pool": gnorm(model.mv_encoder.pool),
                     "g_proj": gnorm(model.projector),
                     "g_pred": gnorm(model.predictor),
                     "g_act": gnorm(model.action_encoder)}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
        if step % LOG_EVERY == 0:
            rec = {"step": step, "pred": round(float(out["pred_loss"]), 6),
                   "sigreg": round(float(out["sigreg_loss"]), 4),
                   "min": round((time.time() - t0) / 60, 1)}
            if rec_g: rec.update(rec_g)
            if step % VAL_EVERY == 0:
                v = validate(); rec.update({k: round(x, 6) for k, x in v.items()})
                if v["val_pred_mse_h1"] < best_val:
                    best_val = v["val_pred_mse_h1"]
                    torch.save({"model": model.state_dict(), "step": step,
                                "val": v}, f"{OUT}/ckpt_best.pt")
                    rec["best"] = True
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % CKPT_EVERY == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best_val": best_val}, f"{OUT}/ckpt_last.pt")
            torch.save({"model": model.state_dict(), "step": step},
                       f"{OUT}/ckpt_{step//1000}k.pt")
        if step >= TOTAL: break
torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
            "step": step, "best_val": best_val}, f"{OUT}/ckpt_last.pt")
print("B1 FULL TRAINING DONE", flush=True)
