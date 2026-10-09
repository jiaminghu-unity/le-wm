#!/usr/bin/env python3
"""Stage 2: vision predictability P_t (proposal §7).

Probe A (nuisance): [a_t, q_t] -> r_tau.   Probe B (vision): [z_t, a_t, q_t] -> r_tau.
Both cross-fitted 5-fold at episode level. P_t = (e0 - eV) / (e0 + eps).

Outputs artifacts/usb/vision_predictability.npz and plots/ diagnostics.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touch_guided.probes import crossfit_regress  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--art", default="artifacts/usb")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eps", type=float, default=1e-6)
p.add_argument("--features", default="vision_features.npy")
p.add_argument("--tag", default="")
args = p.parse_args()

art = Path(args.art)
d = np.load(art / "tactile_descriptors.npz")
z = np.load(art / args.features)
rows, ep_ids, prog = d["rows"], d["ep_ids"], d["progress"]
r_tau, action, dT = d["r_tau"], d["action"], d["dT_norm"]
zt = z[rows]

x0 = np.concatenate([action, prog[:, None]], 1).astype(np.float32)
xv = np.concatenate([zt, action, prog[:, None]], 1).astype(np.float32)

print(f"[probe A 仅动作+相位] in={x0.shape[1]}")
e0, _ = crossfit_regress(x0, r_tau, ep_ids, args.folds, args.seed, args.steps)
print(f"[probe B 视觉+动作+相位] in={xv.shape[1]}")
eV, _ = crossfit_regress(xv, r_tau, ep_ids, args.folds, args.seed, args.steps)

P = (e0 - eV) / (e0 + args.eps)
np.savez_compressed(art / f"vision_predictability{args.tag}.npz",
                    rows=rows, ep_ids=ep_ids, P=P.astype(np.float32),
                    e0=e0, eV=eV, progress=prog, dT_norm=dT)

q = lambda a, x: float(np.quantile(a, x))
print(f"[P_t] mean={P.mean():.4f} median={np.median(P):.4f} "
      f"P>0 比例={np.mean(P > 0):.3f}  分位 25/75/90={q(P,.25):.3f}/{q(P,.75):.3f}/{q(P,.9):.3f}")
print(f"[误差] e0 mean={e0.mean():.3f}  eV mean={eV.mean():.3f}  整体降幅={(1 - eV.mean()/e0.mean()):.3%}")

# required diagnostics (§7)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plots = art / "plots"; plots.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(P, bins=80); axes[0].set_title("P_t distribution"); axes[0].axvline(0, c="r", lw=1)
axes[1].scatter(prog, P, s=2, alpha=.2); axes[1].set_title("P_t vs episode progress")
axes[2].scatter(dT, P, s=2, alpha=.2); axes[2].set_xscale("log"); axes[2].set_title("P_t vs ||dT||")
fig.tight_layout(); fig.savefig(plots / f"predictability{args.tag}.png", dpi=130)
hi = rows[np.argsort(P)[-8:]]; lo = rows[np.argsort(P)[:8]]
np.save(plots / f"highP_rows{args.tag}.npy", hi); np.save(plots / f"lowP_rows{args.tag}.npy", lo)
print("[plots] saved", plots / f"predictability{args.tag}.png", "| high/low 样例行已存")
