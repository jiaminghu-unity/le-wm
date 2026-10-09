"""B3 / Ours full training (frozen B1 protocol + gated residual dynamics).
  python train_b3_ours.py --variant ours|b3
Identical parameter count across variants; 'ours' adds tactile-supervised gate CE
(teacher-forced routing on labeled transitions); 'b3' trains the same gate+residuals
end-to-end with no tactile supervision (capacity control). Resume-capable.
"""
import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from touchwm.mode_residual import GatedResidualJEPA
from touchwm.data_adapter import MultiViewClipDataset

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["b3", "ours"], required=True)
ap.add_argument("--steps", type=int, default=100_000)
args = ap.parse_args()
sup = args.variant == "ours"
OUT = f"outputs/{args.variant}"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"; BS, VAL_EVERY, LOG_EVERY, CKPT_EVERY = 32, 2_500, 500, 10_000
torch.manual_seed(3072); np.random.seed(3072)

# per-row mode lookup (-2 undefined, -1 ambiguous, 0..2 labels)
M = np.load("outputs/preflight/usb/provisional_modes.npz")
mode_row = np.full(10**6, -2, dtype=np.int64)
mode_row[M["t"]] = M["mode"]

class ClipWithModes(MultiViewClipDataset):
    def __getitem__(self, i):
        it = super().__getitem__(i)
        t0 = it["t0"]
        it["mode"] = torch.from_numpy(mode_row[t0:t0 + self.clip_len].copy())
        return it

train_ds = ClipWithModes("data/demo_data/insertion_usb",
                         "outputs/preflight/usb/splits.json", "train", clip_len=3)
val_ds = ClipWithModes("data/demo_data/insertion_usb",
                       "outputs/preflight/usb/splits.json", "val", clip_len=3)
rng = np.random.RandomState(0)
VAL_IDX = rng.choice(len(val_ds), size=64, replace=False)
val_batch = None

dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=6,
                drop_last=True, persistent_workers=True, prefetch_factor=4)
model = GatedResidualJEPA().to(dev)
print("params(M):", sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

step, best_val = 0, float("inf")
if os.path.exists(f"{OUT}/ckpt_last.pt"):
    st = torch.load(f"{OUT}/ckpt_last.pt", map_location=dev)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    step, best_val = st["step"], st.get("best_val", float("inf"))
    print(f"[resume] step {step}", flush=True)

@torch.no_grad()
def validate():
    global val_batch
    if val_batch is None:
        items = [val_ds[int(i)] for i in VAL_IDX]
        val_batch = {k: torch.stack([it[k] for it in items]).to(dev)
                     for k in ("front", "wrist", "action", "mode")}
    model.eval()
    z = model.encode({"front": val_batch["front"], "wrist": val_batch["wrist"]})
    pred, logits = model.predict_gated(z[:, :-1], val_batch["action"], None)  # 部署口径:软/自由门
    res = {"val_pred_mse_h1": float((pred - z[:, 1:]).pow(2).mean()),
           "persist_mse_h1": float((z[:, :-1] - z[:, 1:]).pow(2).mean())}
    lab = val_batch["mode"].reshape(-1); msk = lab >= 0
    if msk.any():
        res["val_gate_acc"] = float((logits.reshape(-1, 3).argmax(-1)[msk] == lab[msk]).float().mean())
    Zf = z.reshape(-1, z.shape[-1])
    cov = torch.cov(Zf.T); eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = eig / eig.sum()
    res["eff_rank"] = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    model.train(); return res

log_f = open(f"{OUT}/train_log.jsonl", "a")
model.train(); t0 = time.time()
while step < args.steps:
    for b in dl:
        obs = {"front": b["front"].to(dev, non_blocking=True),
               "wrist": b["wrist"].to(dev, non_blocking=True)}
        out = model.loss_gated(obs, b["action"].to(dev), b["mode"].to(dev),
                               supervise_gate=sup)
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1
        if step % LOG_EVERY == 0:
            rec = {"step": step, "pred": round(float(out["pred_loss"]), 6),
                   "sigreg": round(float(out["sigreg_loss"]), 4),
                   "min": round((time.time() - t0) / 60, 1)}
            for k in ("gate_ce", "gate_acc"):
                if k in out: rec[k] = round(float(out[k]), 4)
            if step % VAL_EVERY == 0:
                v = validate(); rec.update({k: round(x, 6) for k, x in v.items()})
                if v["val_pred_mse_h1"] < best_val:
                    best_val = v["val_pred_mse_h1"]
                    torch.save({"model": model.state_dict(), "step": step, "val": v},
                               f"{OUT}/ckpt_best.pt")
                    rec["best"] = True
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % CKPT_EVERY == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best_val": best_val}, f"{OUT}/ckpt_last.pt")
        if step >= args.steps: break
torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
            "step": step, "best_val": best_val}, f"{OUT}/ckpt_last.pt")
print(f"{args.variant.upper()} FULL TRAINING DONE", flush=True)
