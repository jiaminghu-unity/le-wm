"""D1 pilot: extract-once frozen DINO token caches, then train the fusion+predictor
on the caches (L4-sized; no A100). 15k-step pilot with the standard diagnostics.
  python d1_pilot.py extract   # once: writes outputs/d1/{front,wrist}_tokens.fp16.npy
  python d1_pilot.py train     # pilot training + val metrics
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn
import zarr

OUT = "outputs/d1"; os.makedirs(OUT, exist_ok=True)
dev = "cuda"
root = zarr.open("data/demo_data/insertion_usb", mode="r")
ends = np.asarray(root["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
N = int(ends[-1])
act = np.asarray(root["data/action"][:])
P_TOK, D_TOK = 256, 384

if sys.argv[1] == "extract":
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(dev)
    for p in model.parameters(): p.requires_grad_(False)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    for view in ("front", "wrist"):
        arr = root[f"data/{view}"]
        mm = np.lib.format.open_memmap(f"{OUT}/{view}_tokens.fp16.npy", mode="w+",
                                       dtype=np.float16, shape=(N, P_TOK, D_TOK))
        with torch.no_grad():
            for i in range(0, N, 32):
                x = torch.from_numpy(np.asarray(arr[i:i+32])).permute(0, 3, 1, 2).float()
                x = Fn.interpolate(x, size=(224, 224), mode="bilinear", antialias=True).to(dev)
                t = model.forward_features((x - MEAN) / STD)["x_norm_patchtokens"]
                mm[i:i+32] = t.half().cpu().numpy()
        mm.flush(); print(view, "tokens cached", flush=True)
    print("EXTRACT DONE", flush=True)
    sys.exit(0)

# ---- train ----
sys.path.insert(0, ".")
from touchwm.dino_mv import DinoMultiViewJEPA
torch.manual_seed(3072); np.random.seed(3072)
STEPS, BS, CLIP = 15_000, 32, 3
tok = {v: np.load(f"{OUT}/{v}_tokens.fp16.npy", mmap_mode="r") for v in ("front", "wrist")}
splits = json.load(open("outputs/preflight/usb/splits.json"))

def clip_index(eps):
    idx = []
    for ep in eps:
        for t0 in range(int(starts[ep]), int(ends[ep]) - CLIP):
            idx.append(t0)
    return np.array(idx)
tr_idx, va_idx = clip_index(splits["train"]), clip_index(splits["val"])
rng = np.random.RandomState(0)
VAL = va_idx[rng.choice(len(va_idx), 64, replace=False)]

def batch(t0s):
    f = torch.from_numpy(np.stack([tok["front"][t:t+CLIP+1] for t in t0s]))
    w = torch.from_numpy(np.stack([tok["wrist"][t:t+CLIP+1] for t in t0s]))
    a = torch.from_numpy(np.stack([act[t:t+CLIP] for t in t0s])).float()
    return f.to(dev), w.to(dev), a.to(dev)

model = DinoMultiViewJEPA().to(dev)
print("trainable params(M):", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
t0 = time.time(); logf = open(f"{OUT}/pilot_log.jsonl", "a"); best = 1e9
for step in range(1, STEPS + 1):
    sel = tr_idx[np.random.randint(0, len(tr_idx), BS)]
    f, w, a = batch(sel)
    out = model.loss(f, w, a)
    opt.zero_grad(); out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            fv, wv, av = batch(VAL)
            z = model.encode_tokens(fv, wv)
            pred = model.predict(z[:, :-1], av)
            v = float((pred - z[:, 1:]).pow(2).mean())
            pers = float((z[:, :-1] - z[:, 1:]).pow(2).mean())
            Zf = z.reshape(-1, z.shape[-1])
            cov = torch.cov(Zf.T); eig = torch.linalg.eigvalsh(cov).clamp(min=0)
            pr = eig / eig.sum()
            er = float(torch.exp(-(pr * (pr + 1e-12).log()).sum()))
        model.train()
        rec = {"step": step, "pred": round(float(out["pred_loss"]), 6),
               "sigreg": round(float(out["sigreg_loss"]), 4),
               "val_h1": round(v, 6), "persist_h1": round(pers, 6),
               "eff_rank": round(er, 2), "min": round((time.time()-t0)/60, 1)}
        print(json.dumps(rec), flush=True); logf.write(json.dumps(rec)+"\n"); logf.flush()
        if v < best:
            best = v
            torch.save({"model": model.state_dict(), "step": step}, f"{OUT}/pilot_best.pt")
torch.save({"model": model.state_dict(), "step": STEPS}, f"{OUT}/pilot_last.pt")
print("D1 PILOT DONE", flush=True)
