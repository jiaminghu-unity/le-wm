"""D2 — routing & residual ablation on existing Ours/B3 checkpoints (no training).

Same checkpoint, four inference paths on held-out episodes (val+test):
  base   : P0 only (residual removed; NOTE: P0 here is the jointly-trained base
           predictor inside Ours/B3, NOT the independently trained B1 checkpoint)
  soft   : P0 + sum_k p_k R_k   (gate-probability routing, B3's training mode)
  hard   : P0 + R_argmax        (deployed mode)
  oracle : P0 + R_{m_true}      (true transition label; offline labeled steps only,
           unlabeled AR steps fall back to hard and are masked out of oracle stats)

Reports (JSON):
  - H1/H3/H6 AR prediction MSE per routing, grouped free/contact/transition
  - gate macro-F1 / per-mode recall / confusion on held-out labeled steps
  - routing agreement: argmax gate on predicted vs true latents at each AR step
  - R1/R2 norms, usage frequency, cosine(R1,R2) — are the experts distinct?

Run on cw-l4-01 (d0venv, no Isaac needed):
  ~/d0venv/bin/python d2_routing_audit.py --variant ours --ckpt-path ours_ckpt_best.pt
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["ours", "b3"], required=True)
ap.add_argument("--ckpt-path", required=True)
ap.add_argument("--data-root", default="data/demo_data/insertion_usb")
ap.add_argument("--splits", default="outputs/preflight/usb/splits.json")
ap.add_argument("--modes", default="outputs/preflight/usb/provisional_modes.npz")
ap.add_argument("--horizon", type=int, default=6)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--enc-bs", type=int, default=24)
ap.add_argument("--out", default="")
args = ap.parse_args()
dev = "cuda"
T = args.horizon
H = 3                      # training context length
W = H + T                  # frames per window: context 3 + predicted 6 -> need 9
out_path = args.out or f"outputs/d2_routing_audit_{args.variant}.json"

from touchwm.mode_residual import GatedResidualJEPA
model = GatedResidualJEPA().to(dev)
st = torch.load(args.ckpt_path, map_location=dev)
model.load_state_dict(st.get("model", st))
model.eval()
print(f"[d2] loaded {args.variant} step={st.get('step')}", flush=True)

import zarr
zr = zarr.open(args.data_root, mode="r")
ends = np.asarray(zr["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
splits = json.load(open(args.splits))
held = sorted(list(splits["val"]) + list(splits["test"]))
print(f"[d2] held-out episodes: {len(held)}", flush=True)

mode_row = np.full(10**6, -2, dtype=np.int64)
M = np.load(args.modes)
mode_row[M["t"]] = M["mode"]

@torch.no_grad()
def encode_rows(lo, hi):
    """Encode zarr rows [lo,hi) -> (n, D) latents, chunked."""
    zs = []
    for a in range(lo, hi, args.enc_bs):
        b = min(a + args.enc_bs, hi)
        f = np.asarray(zr["data/front"][a:b], dtype=np.float32)
        w = np.asarray(zr["data/wrist"][a:b], dtype=np.float32)
        f = torch.from_numpy(f).permute(0, 3, 1, 2).to(dev)
        w = torch.from_numpy(w).permute(0, 3, 1, 2).to(dev)
        if f.shape[-1] != 224:
            f = Fn.interpolate(f, size=(224, 224), mode="bilinear",
                               align_corners=False, antialias=True)
            w = Fn.interpolate(w, size=(224, 224), mode="bilinear",
                               align_corners=False, antialias=True)
        z = model.encode({"front": f[:, None], "wrist": w[:, None]})[:, 0]
        zs.append(z)
    return torch.cat(zs)

@torch.no_grad()
def step_components(z_ctx, a_ctx):
    """One AR step: returns base pred, residual stack, gate logits (last position)."""
    a_emb = model.action_encoder(a_ctx)
    base = model.predictor(z_ctx, a_emb)
    b_, t_, d_ = base.shape
    base = model.pred_proj(base.reshape(b_ * t_, d_)).reshape(b_, t_, d_)[:, -1]
    res = model._residual_stack(z_ctx, a_emb)[:, -1]        # (B,K,D)
    logits = model.gate_logits(z_ctx, a_emb)[:, -1]          # (B,K)
    return base, res, logits

# ---------- collect windows ----------
wins = []           # (t_abs of first context frame)
for e in held:
    s, en = int(starts[e]), int(ends[e])
    for t0 in range(s, en - W, args.stride):
        wins.append(t0)
wins = np.asarray(wins)
print(f"[d2] windows: {len(wins)}", flush=True)

acts_glob = np.asarray(zr["data/action"][:], dtype=np.float32)

# encode each held-out episode once, keep per-row latents in a dict on GPU
z_rows = {}
with torch.no_grad():
    for e in held:
        s, en = int(starts[e]), int(ends[e])
        z_rows[e] = encode_rows(s, en)
row2ep = np.zeros(10**6, dtype=np.int64)
for e in held:
    row2ep[int(starts[e]):int(ends[e])] = e
def z_at(rows):
    """rows: (B,W) absolute -> (B,W,D)"""
    out = []
    for r in rows:
        e = int(row2ep[r[0]])
        out.append(z_rows[e][r - int(starts[e])])
    return torch.stack(out)

ROUTES = ["base", "soft", "hard", "oracle"]
B = 256
err = {r: {h: [] for h in (1, 3, 6)} for r in ROUTES}        # squared err per window
agree_pred_true = [[] for _ in range(T)]                      # hard-route agreement
mode_of_step = [[] for _ in range(T)]                         # label of each predicted transition
r_norms, r_cos, hard_pick = [], [], []
gate_logits_true, gate_labels_true = [], []

with torch.no_grad():
    for a0 in range(0, len(wins), B):
        wb = wins[a0:a0 + B]
        rows = wb[:, None] + np.arange(W)[None]              # (B,W)
        z_true = z_at(rows)                                   # (B,W,D)
        acts = torch.from_numpy(acts_glob[wb[:, None] + np.arange(W - 1)[None]]).to(dev)
        labs = mode_row[wb[:, None] + np.arange(W - 1)[None]]  # label of transition tau->tau+1

        # gate quality on TRUE latents (all context-capable positions 2..W-2)
        a_emb_full = model.action_encoder(acts)
        lg_full = model.gate_logits(z_true[:, :-1], a_emb_full)   # (B,W-1,K)
        gate_logits_true.append(lg_full[:, 2:].reshape(-1, 3).cpu())
        gate_labels_true.append(torch.from_numpy(labs[:, 2:]).reshape(-1))

        # R character on true latents
        res_full = model._residual_stack(z_true[:, :-1], a_emb_full)  # (B,W-1,K,D)
        r1, r2 = res_full[..., 1, :], res_full[..., 2, :]
        r_norms.append(torch.stack([r1.norm(dim=-1).mean(), r2.norm(dim=-1).mean()]).cpu())
        r_cos.append(Fn.cosine_similarity(r1.reshape(-1, 192), r2.reshape(-1, 192), dim=-1).mean().cpu())
        hard_pick.append(lg_full.argmax(-1).reshape(-1).cpu())

        # AR rollouts per routing
        n = len(wb)
        for route in ROUTES:
            z_ctx = z_true[:, :H].clone()
            a_ctx = acts[:, :H].clone()
            for s in range(T):
                base, res, logits = step_components(z_ctx, a_ctx)
                if route == "base":
                    delta = torch.zeros_like(base)
                elif route == "soft":
                    delta = (torch.softmax(logits, -1).unsqueeze(-1) * res).sum(-2)
                elif route == "hard":
                    delta = res.gather(-2, logits.argmax(-1)[:, None, None].expand(-1, 1, 192)).squeeze(-2)
                else:  # oracle with hard fallback on unlabeled
                    lab_s = torch.from_numpy(labs[:, H - 1 + s]).to(dev)
                    idx = torch.where(lab_s >= 0, lab_s, logits.argmax(-1))
                    delta = res.gather(-2, idx[:, None, None].expand(-1, 1, 192)).squeeze(-2)
                pred = (base + delta)
                if route == "hard":
                    # routing agreement: same step computed on TRUE context
                    zt = z_true[:, s:s + H]
                    _, _, lg_t = step_components(zt, a_ctx)
                    agree_pred_true[s].append((logits.argmax(-1) == lg_t.argmax(-1)).float().cpu())
                    mode_of_step[s].append(torch.from_numpy(labs[:, H - 1 + s]))
                e2 = (pred - z_true[:, H + s]).pow(2).mean(-1)
                if s + 1 in (1, 3, 6):
                    lab0 = labs[:, H - 1].astype(np.float32)
                    lab1 = labs[:, H].astype(np.float32)
                    trans = ((labs[:, H - 1] >= 0) & (labs[:, H] >= 0)
                             & (labs[:, H - 1] != labs[:, H])).astype(np.float32)
                    err[route][s + 1].append(torch.stack(
                        [e2.cpu(), torch.from_numpy(lab0), torch.from_numpy(trans)]))
                z_ctx = torch.cat([z_ctx[:, 1:], pred[:, None]], dim=1)
                if s + 1 < T:
                    a_ctx = torch.cat([a_ctx[:, 1:], acts[:, H + s][:, None]], dim=1)

# ---------- aggregate ----------
def group_stats(stacked):
    e, lab, tr = stacked[0].numpy(), stacked[1].numpy(), stacked[2].numpy()
    out = {"all": float(e.mean())}
    out["free"] = float(e[lab == 0].mean()) if (lab == 0).any() else None
    out["contact"] = float(e[(lab == 1) | (lab == 2)].mean()) if ((lab == 1) | (lab == 2)).any() else None
    out["transition"] = float(e[tr == 1].mean()) if (tr == 1).any() else None
    return out

res = {"variant": args.variant, "n_windows": int(len(wins)), "horizon": T}
for route in ROUTES:
    res[route] = {f"H{h}": group_stats(torch.cat(err[route][h], dim=1)) for h in (1, 3, 6)}

# per-mode grouping at H1 (label of first predicted transition)
labs_h1 = torch.cat(err["base"][1], dim=1)[1].numpy()
for route in ROUTES:
    e_h1 = torch.cat(err[route][1], dim=1)[0].numpy()
    res[route]["H1_by_mode"] = {str(m): float(e_h1[labs_h1 == m].mean())
                                for m in (0, 1, 2) if (labs_h1 == m).any()}

# gate quality
from sklearn.metrics import f1_score, confusion_matrix, recall_score
lg = torch.cat(gate_logits_true).numpy(); gl = torch.cat(gate_labels_true).numpy()
msk = gl >= 0
pred_m = lg[msk].argmax(-1); true_m = gl[msk]
res["gate"] = {
    "n_labeled": int(msk.sum()),
    "macro_f1": float(f1_score(true_m, pred_m, average="macro")),
    "per_mode_recall": {str(k): float(r) for k, r in
                        zip((0, 1, 2), recall_score(true_m, pred_m, labels=[0, 1, 2], average=None, zero_division=0))},
    "confusion": confusion_matrix(true_m, pred_m, labels=[0, 1, 2]).tolist(),
}

# routing agreement predicted-vs-true latents per AR step, split by mode
res["routing_agreement"] = {}
for s in range(T):
    ag = torch.cat(agree_pred_true[s]).numpy()
    mo = torch.cat(mode_of_step[s]).numpy()
    d = {"all": float(ag.mean())}
    for m in (0, 1, 2):
        if (mo == m).any(): d[f"mode{m}"] = float(ag[mo == m].mean())
    res["routing_agreement"][f"step{s+1}"] = d

# residual character
rn = torch.stack(r_norms).mean(0)
hp = torch.cat(hard_pick).numpy()
res["residuals"] = {
    "norm_R1": float(rn[0]), "norm_R2": float(rn[1]),
    "cos_R1_R2": float(torch.stack(r_cos).mean()),
    "hard_usage": {str(k): float((hp == k).mean()) for k in (0, 1, 2)},
}

json.dump(res, open(out_path, "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "n_windows"}, indent=1)[:3000], flush=True)
print("D2 ROUTING AUDIT DONE", flush=True)
