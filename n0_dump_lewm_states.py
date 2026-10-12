"""N0 row-7 helper: dump B1's per-row joint latent, history hidden state and
predicted next latent for all rows with a full 3-frame in-episode context.

Outputs outputs/n0_lewm_states.npz:
  rows    (N,)      absolute row index t
  z_joint (N,192)   encoder joint latent of frame t
  h_pred  (N,192)   ARPredictor output at last position (pre pred_proj), context
                    z_{t-2..t} + a_{t-2..t}  == the model's history summary
  z_hat   (N,192)   pred_proj(h_pred) == deployed prediction of z_{t+1}

Run on cw-l4-01: ~/d0venv/bin/python n0_dump_lewm_states.py
"""
import json
import numpy as np
import torch
import torch.nn.functional as Fn

dev = "cuda"
from touchwm.cw_eval_adapter import MVJepaWorldModel
wm = MVJepaWorldModel("outputs_b1_ckpt_100k.pt", variant="b1")
model = wm.model.eval()

import zarr
zr = zarr.open("data/demo_data/insertion_usb", mode="r")
ends = np.asarray(zr["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
acts = np.asarray(zr["data/action"][:], dtype=np.float32)
N_rows = acts.shape[0]

@torch.no_grad()
def enc(lo, hi, bs=24):
    zs = []
    for a in range(lo, hi, bs):
        b = min(a + bs, hi)
        f = torch.from_numpy(np.asarray(zr["data/front"][a:b], dtype=np.float32)).permute(0, 3, 1, 2).to(dev)
        w = torch.from_numpy(np.asarray(zr["data/wrist"][a:b], dtype=np.float32)).permute(0, 3, 1, 2).to(dev)
        if f.shape[-1] != 224:
            f = Fn.interpolate(f, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
            w = Fn.interpolate(w, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        zs.append(model.encode({"front": f[:, None], "wrist": w[:, None]})[:, 0])
    return torch.cat(zs)

rows_out, zj_out, hp_out, zh_out = [], [], [], []
with torch.no_grad():
    for e in range(len(ends)):
        s, en = int(starts[e]), int(ends[e])
        z_ep = enc(s, en)                                   # (n,192)
        taus = np.arange(s + 2, en)                          # t with full context
        for a0 in range(0, len(taus), 512):
            tb = taus[a0:a0 + 512]
            idx = torch.from_numpy(tb[:, None] + np.arange(-2, 1)[None] - s).to(dev)
            z_ctx = z_ep[idx]                                # (B,3,192)
            a_ctx = torch.from_numpy(acts[tb[:, None] + np.arange(-2, 1)[None]]).to(dev)
            a_emb = model.action_encoder(a_ctx)
            h = model.predictor(z_ctx, a_emb)                # (B,3,192)
            b_, t_, d_ = h.shape
            zh = model.pred_proj(h.reshape(b_ * t_, d_)).reshape(b_, t_, d_)
            rows_out.append(tb)
            zj_out.append(z_ep[torch.from_numpy(tb - s).to(dev)].cpu())
            hp_out.append(h[:, -1].cpu()); zh_out.append(zh[:, -1].cpu())
        if e % 40 == 0:
            print(f"ep {e}/201", flush=True)

np.savez_compressed("outputs/n0_lewm_states.npz",
                    rows=np.concatenate(rows_out),
                    z_joint=torch.cat(zj_out).numpy(),
                    h_pred=torch.cat(hp_out).numpy(),
                    z_hat=torch.cat(zh_out).numpy())
print("N0 LEWM STATES DONE", flush=True)
