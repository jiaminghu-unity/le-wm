"""Plot the budget sweep: SR vs planning compute, one line per config.

x = predictor forwards per replan (candidates x iterations x horizon, log scale)
y = success rate over the 50 paired episodes, 95% CI via bootstrap (1000 reps).

Colors: dataviz reference categorical palette slots 1-4 in fixed order
(pre-validated adjacent-pair CVD DeltaE >= 8 on light surface).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZON = 5
TIER_ORDER = ["T1", "T2", "T3", "T4", "T5"]
# fixed identity -> color assignment (never recycled/reordered)
SERIES = {
    "c1": ("#2a78d6", "C1 baseline (SIGReg)"),
    "c2p": ("#eb6834", "C2p L_obj + projector"),
    "c2": ("#1baf7a", "C2 L_obj vanilla (LN)"),
    "c3": ("#eda100", "C3 SIGReg + L_obj"),
    "c3_l02": ("#e87ba4", "C3 SIGReg + L_obj (l=0.2)"),
    "r1": ("#2a78d6", "R1 baseline (SIGReg)"),
    "r2": ("#eb6834", "R2 PAEP joints"),
    "r3": ("#1baf7a", "R3 PAEP joints+finger"),
}
BOOT_REPS = 1000
BOOT_SEED = 0
TEXT = "#3d3d3c"
MUTED = "#6f6e66"


def bootstrap_band(success, reps=BOOT_REPS, seed=BOOT_SEED):
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(success), size=(reps, len(success)))
    srs = success[idx].mean(axis=1) * 100.0
    return np.percentile(srs, 2.5), np.percentile(srs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_results/results.csv")
    ap.add_argument("--out", default="eval_results/budget_sweep.png")
    ap.add_argument("--title", default="Push-T planning success vs CEM budget (50 paired episodes, 95% bootstrap CI)")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)

    tier_x = {}
    endpoints = []
    for config in [c for c in SERIES if c in df.config.unique()]:
        color, label = SERIES[config]
        xs, ys, los, his = [], [], [], []
        for tier in TIER_ORDER:
            grp = df[(df.config == config) & (df.tier == tier)]
            if not len(grp):
                continue
            x = int(grp.candidates.iloc[0]) * int(grp.iterations.iloc[0]) * HORIZON
            tier_x[tier] = x
            succ = grp.sort_values("episode_id")["success"].to_numpy()
            lo, hi = bootstrap_band(succ)
            xs.append(x)
            ys.append(100.0 * succ.mean())
            los.append(lo)
            his.append(hi)
        ax.fill_between(xs, los, his, color=color, alpha=0.13, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
        endpoints.append((config, xs[0], ys[0]))

    # direct labels at the T1 end, pushed apart when lines end close together
    endpoints.sort(key=lambda e: e[2])
    offsets = [-3.0] * len(endpoints)
    for i in range(1, len(endpoints)):
        gap = (endpoints[i][2] + offsets[i]) - (endpoints[i - 1][2] + offsets[i - 1])
        if gap < 10.0:
            offsets[i] += 10.0 - gap
    for (config, x0, y0), dy in zip(endpoints, offsets):
        ax.annotate(config.upper(), (x0, y0), textcoords="offset points",
                    xytext=(8, dy), color=TEXT, fontsize=9, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks([tier_x[t] for t in TIER_ORDER if t in tier_x])
    ax.set_xticklabels(
        [f"{t}\n{tier_x[t]:,}" for t in TIER_ORDER if t in tier_x], fontsize=8.5, color=TEXT
    )
    ax.set_ylim(0, 102)
    ax.set_xlabel("Predictor forwards per replan (candidates x iterations x horizon)",
                  fontsize=9.5, color=TEXT)
    ax.set_ylabel("Success rate (%)", fontsize=9.5, color=TEXT)
    ax.set_title(args.title, fontsize=10.5, color=TEXT, pad=12)
    ax.grid(True, which="major", axis="y", color="#e8e7e0", linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c9c8bf")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=TEXT)
    ax.margins(x=0.09)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white", bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
