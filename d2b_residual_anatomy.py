"""D2b — residual error-correction anatomy (closing item 3; no training).

For the needed one-step correction D* = z_{t+1} - P0(z,a), ask of each residual
branch R: does it point the right way (cos), is it the right size (norm ratio,
optimal global alpha), and does it generalize (train vs held-out episodes)?

Verdict logic:
  cos ~ 0 on train           -> never learned the correction (direction wrong)
  cos > 0 train, ~0 held-out -> generalization failure
  cos > 0 both, alpha << 1   -> magnitude mis-scaled
  mode-0 ||D*|| small        -> P0 already fits free motion (R0 = 0 justified)

Run on cw-l4-01 (d0venv): ~/d0venv/bin/python d2b_residual_anatomy.py --variant ours
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["ours", "b3"], required=True)
ap.add_argument("--ckpt-path", default="")
ap.add_argument("--data-root", default="data/demo_data/insertion_usb")
ap.add_argument("--splits", default="outputs/preflight/usb/splits.json")
ap.add_argument("--modes", default="outputs/preflight/usb/provisional_modes.npz")
ap.add_argument("--train-eps", type=int, default=30, help="train episodes to subsample")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--enc-bs", type=int, default=24)
args = ap.parse_args()
dev = "cuda"
ck = args.ckpt_path or f"{args.variant}_ckpt_best.pt"
H = 3

from touchwm.mode_residual import GatedResidualJEPA
model = GatedResidualJEPA().to(dev)
st = torch.load(ck, map_location=dev)
model.load_state_dict(st.get("model", st)); model.eval()
print(f"[d2b] {args.variant} step={st.get('step')}", flush=True)

import zarr
zr = zarr.open(args.data_root, mode="r")
ends = np.asarray(zr["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
splits = json.load(open(args.splits))
rng = np.random.RandomState(0)
train_eps = sorted(rng.choice(sorted(splits["train"]), size=min(args.train_eps, len(splits["train"])), replace=False).tolist())
held_eps = sorted(list(splits["val"]) + list(splits["test"]))

mode_row = np.full(10**6, -2, dtype=np.int64)
M = np.load(args.modes)
mode_row[M["t"]] = M["mode"]
acts_glob = np.asarray(zr["data/action"][:], dtype=np.float32)

@torch.no_grad()
def encode_rows(lo, hi):
    zs = []
    for a in range(lo, hi, args.enc_bs):
        b = min(a + args.enc_bs, hi)
        f = torch.from_numpy(np.asarray(zr["data/front"][a:b], dtype=np.float32)).permute(0, 3, 1, 2).to(dev)
        w = torch.from_numpy(np.asarray(zr["data/wrist"][a:b], dtype=np.float32)).permute(0, 3, 1, 2).to(dev)
        if f.shape[-1] != 224:
            f = Fn.interpolate(f, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
            w = Fn.interpolate(w, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        zs.append(model.encode({"front": f[:, None], "wrist": w[:, None]})[:, 0])
    return torch.cat(zs)

@torch.no_grad()
def analyze(eps, tag):
    rows_lab, rows = [], {k: [] for k in
        ("lab", "dstar_n", "cos_R1", "cos_R2", "cos_hard", "cos_soft", "cos_oracle",
         "n_R1", "n_R2", "dot_R1", "dot_R2", "nn_R1", "nn_R2",
         "err0", "err_hard", "err_soft", "err_oracle")}
    for e in eps:
        s, en = int(starts[e]), int(ends[e])
        if en - s < H + 2: continue
        z_ep = encode_rows(s, en)                           # (n,D)
        # positions tau (abs) with context [tau-2, tau] and target tau+1 inside episode
        taus = np.arange(s + H - 1, en - 1, args.stride)
        if len(taus) == 0: continue
        for a0 in range(0, len(taus), 256):
            tb = taus[a0:a0 + 256]
            idx = tb[:, None] + np.arange(-(H - 1), 2)[None]     # (B,4): tau-2..tau+1
            z_w = z_ep[torch.from_numpy(idx - s).to(z_ep.device)]  # (B,4,D)
            a_ctx = torch.from_numpy(acts_glob[tb[:, None] + np.arange(-(H - 1), 1)[None]]).to(dev)  # (B,3,A)
            z_ctx = z_w[:, :H]
            a_emb = model.action_encoder(a_ctx)
            base = model.predictor(z_ctx, a_emb)
            b_, t_, d_ = base.shape
            base = model.pred_proj(base.reshape(b_ * t_, d_)).reshape(b_, t_, d_)[:, -1]
            res = model._residual_stack(z_ctx, a_emb)[:, -1]     # (B,K,D)
            logits = model.gate_logits(z_ctx, a_emb)[:, -1]
            dstar = z_w[:, H] - base                              # (B,D)
            lab = torch.from_numpy(mode_row[tb]).to(dev)

            R1, R2 = res[:, 1], res[:, 2]
            Rh = res.gather(1, logits.argmax(-1)[:, None, None].expand(-1, 1, d_)).squeeze(1)
            Rs = (torch.softmax(logits, -1).unsqueeze(-1) * res).sum(1)
            Ro = res.gather(1, lab.clamp(min=0)[:, None, None].expand(-1, 1, d_)).squeeze(1)

            cos = lambda a, b: Fn.cosine_similarity(a, b, dim=-1)
            rows["lab"].append(lab.cpu()); rows["dstar_n"].append(dstar.norm(dim=-1).cpu())
            rows["cos_R1"].append(cos(R1, dstar).cpu()); rows["cos_R2"].append(cos(R2, dstar).cpu())
            rows["cos_hard"].append(cos(Rh, dstar).cpu()); rows["cos_soft"].append(cos(Rs, dstar).cpu())
            rows["cos_oracle"].append(cos(Ro, dstar).cpu())
            rows["n_R1"].append(R1.norm(dim=-1).cpu()); rows["n_R2"].append(R2.norm(dim=-1).cpu())
            rows["dot_R1"].append((R1 * dstar).sum(-1).cpu()); rows["dot_R2"].append((R2 * dstar).sum(-1).cpu())
            rows["nn_R1"].append(R1.pow(2).sum(-1).cpu()); rows["nn_R2"].append(R2.pow(2).sum(-1).cpu())
            rows["err0"].append(dstar.pow(2).mean(-1).cpu())
            rows["err_hard"].append((dstar - Rh).pow(2).mean(-1).cpu())
            rows["err_soft"].append((dstar - Rs).pow(2).mean(-1).cpu())
            rows["err_oracle"].append((dstar - Ro).pow(2).mean(-1).cpu())
    R = {k: torch.cat(v).numpy() for k, v in rows.items()}
    out = {"n": int(len(R["lab"]))}
    for m in (-9, 0, 1, 2):     # -9 = all
        sel = np.ones_like(R["lab"], bool) if m == -9 else (R["lab"] == m)
        if not sel.any(): continue
        g = {"n": int(sel.sum()), "dstar_norm": float(R["dstar_n"][sel].mean())}
        for k in ("cos_R1", "cos_R2", "cos_hard", "cos_soft", "cos_oracle",
                  "n_R1", "n_R2", "err0", "err_hard", "err_soft", "err_oracle"):
            g[k] = float(R[k][sel].mean())
        for b in ("R1", "R2"):   # optimal global alpha per group
            nn = R[f"nn_{b}"][sel].sum()
            g[f"alpha_{b}"] = float(R[f"dot_{b}"][sel].sum() / (nn + 1e-12))
        out["all" if m == -9 else f"mode{m}"] = g
    print(f"[d2b] {tag}: n={out['n']}", flush=True)
    return out

res = {"variant": args.variant,
       "train": analyze(train_eps, "train"),
       "heldout": analyze(held_eps, "heldout")}
out_path = f"outputs/d2b_residual_anatomy_{args.variant}.json"
json.dump(res, open(out_path, "w"), indent=2)
print(json.dumps(res, indent=1)[:2500], flush=True)
print("D2B ANATOMY DONE", flush=True)
