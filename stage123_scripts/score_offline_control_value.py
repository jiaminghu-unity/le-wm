#!/usr/bin/env python3
"""Stage 3: offline control-value proxy C_t^offline (proposal §8.1) + formal wrist P_t.

PROXY ONLY (user directive 2026-10-07): this is never to be called tactile criticality;
the strict version is C^sim from simulator branch rollouts (Stage 4), and gold-region
selection / mining / THN training are frozen until closed-loop wrist vs wrist+TacFF SR
exists.

BC probes (ridge primary, episode-level 5-fold cross-fit; MLP capacity check optional):
    vision-only:    f_V(z_t)            -> a_t
    vision+tactile: f_VT(z_t, r_cur_t)  -> a_t      (r_cur: PCA of T_t only, no future)
    C_t^offline = (l_V - l_VT) / (l_V + eps)

Also recomputes the FORMAL Stage-2 P_t on wrist features with the ridge scorer and
saves it as vision_predictability_wrist_ridge.npz (the gold-region input, when unfrozen).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touch_guided.probes import ridge_crossfit  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--art", default="artifacts/usb")
p.add_argument("--features", default="wrist_features.npy")
p.add_argument("--lam", type=float, default=1000.0)
p.add_argument("--eps", type=float, default=1e-6)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

art = Path(args.art)
d = np.load(art / "tactile_descriptors.npz")
z = np.load(art / args.features)
rows, ep, prog = d["rows"], d["ep_ids"], d["progress"]
r_tau, r_cur, action, dT = d["r_tau"], d["r_cur"], d["action"], d["dT_norm"]
zt = z[rows]

# ---- Stage 3: BC probes ----
xv = zt.astype(np.float32)
xvt = np.concatenate([zt, r_cur], 1).astype(np.float32)
lV = ridge_crossfit(xv, action, ep, seed=args.seed, lam=args.lam)
lVT = ridge_crossfit(xvt, action, ep, seed=args.seed, lam=args.lam)
C = (lV - lVT) / (lV + args.eps)

# ---- formal Stage-2 P_t on wrist, ridge scorer ----
x0 = np.concatenate([action, prog[:, None]], 1).astype(np.float32)
xP = np.concatenate([zt, action, prog[:, None]], 1).astype(np.float32)
e0 = ridge_crossfit(x0, r_tau, ep, seed=args.seed, lam=args.lam)
eV = ridge_crossfit(xP, r_tau, ep, seed=args.seed, lam=args.lam)
P = (e0 - eV) / (e0 + args.eps)

np.savez_compressed(art / "offline_control_value.npz",
                    rows=rows, ep_ids=ep, C=C.astype(np.float32), lV=lV, lVT=lVT,
                    progress=prog, dT_norm=dT)
np.savez_compressed(art / "vision_predictability_wrist_ridge.npz",
                    rows=rows, ep_ids=ep, P=P.astype(np.float32), e0=e0, eV=eV,
                    progress=prog, dT_norm=dT)

q = lambda a, x: float(np.quantile(a, x))
print(f"[C_offline 代理] mean={C.mean():+.4f} median={np.median(C):+.4f} C>0={np.mean(C>0):.3f} "
      f"分位75/90={q(C,.75):+.3f}/{q(C,.9):+.3f}")
print(f"[BC误差] 仅视觉 {lV.mean():.4f} -> +触觉 {lVT.mean():.4f}(整体降幅 {(1-lVT.mean()/lV.mean()):+.2%})")
print(f"[P_t wrist ridge 正式版] P>0={np.mean(P>0):.3f} median={np.median(P):+.4f} 整体增益={(1-eV.mean()/e0.mean()):+.2%}")
rho = np.corrcoef(np.argsort(np.argsort(P)), np.argsort(np.argsort(C)))[0, 1]
print(f"[P×C] Spearman={rho:.3f}  右上象限占比(双>75分位)={np.mean((P>q(P,.75))&(C>q(C,.75))):.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plots = art / "plots"; plots.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(C, bins=80); axes[0].axvline(0, c="r", lw=1); axes[0].set_title("C_offline (proxy)")
sc = axes[1].scatter(P, C, s=2, alpha=.2, c=np.log1p(dT)); axes[1].set_xlabel("P_t"); axes[1].set_ylabel("C_offline")
axes[1].set_title("P vs C (color=log||dT||)"); fig.colorbar(sc, ax=axes[1])
axes[2].scatter(prog, C, s=2, alpha=.2); axes[2].set_title("C_offline vs progress")
fig.tight_layout(); fig.savefig(plots / "control_value_wrist.png", dpi=130)
print("[plots]", plots / "control_value_wrist.png")
