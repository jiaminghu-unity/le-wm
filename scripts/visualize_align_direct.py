"""Direct view of what L_obj actually changes.

The 2-D PCA scatter hides the effect: PC1+PC2 hold only 4-8% of the variance
(SIGReg flattens the spectrum by design), the dense core overplots, and an
outlier-stretched colour scale washes the gradient out. rho = 0.86 vs 0.15 is a
huge difference that the scatter simply does not render.

So plot the quantity rho actually measures: the latent coordinate against q-PC1,
one point per frame. Row 1 does it for the single best-aligned latent PC; row 2
does the honest whole-space version — the best linear read-out of q-PC1 from all
192 dims, fitted on half the frames and scored on the other half.

Reads eval_results/pca_cache.npz produced by visualize_pca_grid.py, so no GPU and
no re-encoding.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

C = np.load("eval_results/pca_cache.npz")
TASKS = ["Push-T", "Reacher", "Cube"]
COLS = {"Push-T": ["baseline (C1)", "+L_obj λ=0.1 (C3)", "+aux w=0.3 (C5)"],
        "Reacher": ["baseline (R1)", "+L_obj λ=0.15 (R2)", "+aux w=0.4 (R5)"],
        "Cube": ["baseline (K1)", "+L_obj λ=0.1 (K2)", "+aux w=0.1 (K4)"]}
COLOR = {0: "#2a78d6", 1: "#eb6834", 2: "#e87ba4"}
TEXT = "#3d3d3c"

fig, axes = plt.subplots(2, 9, figsize=(26, 7.2), dpi=150)
summary = []

for r_i, mode in enumerate(["best PC", "full 192-d read-out"]):
    for t_i, task in enumerate(TASKS):
        q = C[f"{task}|qpc1"]
        for m_i, label in enumerate(COLS[task]):
            ax = axes[r_i, t_i * 3 + m_i]
            z = C[f"{task}|{label}|z"]
            zc = z - z.mean(0)
            if mode == "best PC":
                U, S, _ = np.linalg.svd(zc, full_matrices=False)
                pcs = U[:, :2] * S[:2]
                k = int(np.argmax([abs(spearmanr(pcs[:, j], q).statistic) for j in (0, 1)]))
                x = pcs[:, k]
                rho = abs(spearmanr(x, q).statistic)
            else:
                # ridge read-out, fit on even frames, scored on odd ones
                tr, te = np.arange(0, len(q), 2), np.arange(1, len(q), 2)
                A = zc[tr]
                w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ q[tr])
                x = zc @ w
                rho = abs(spearmanr(x[te], q[te]).statistic)
            xs = (x - x.mean()) / x.std()
            ax.scatter(xs, q, s=2, alpha=0.4, linewidths=0, color=COLOR[m_i])
            short = label.split(" (")[0]
            ax.set_title(f"{task}\n{short}   ρ={rho:.2f}", fontsize=8.5, color=TEXT)
            ax.set_xlabel("latent coord (std)", fontsize=7, color=TEXT)
            if m_i == 0:
                ax.set_ylabel(f"q-PC1   [{mode}]", fontsize=7.5, color=TEXT)
            ax.tick_params(labelsize=6)
            summary.append((mode, task, label, rho))

fig.suptitle("What L_obj changes: latent alignment to physical state, plotted directly\n"
             "top = single best principal component · bottom = best linear read-out "
             "of q-PC1 from all 192 dims (held-out frames)",
             fontsize=11, color=TEXT)
out = Path("eval_results/viz_align_direct.png")
fig.savefig(out, facecolor="white", bbox_inches="tight")
print("wrote", out)
print()
print(f"{'mode':22s}{'task':10s}{'model':22s}{'rho':>7s}")
for mode, task, label, rho in summary:
    print(f"{mode:22s}{task:10s}{label:22s}{rho:7.3f}")
