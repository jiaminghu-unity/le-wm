"""viz_pca_angle-style panels for Reacher and OGBench-Cube: top-2 PCA of z,
coloured by one physical quantity at a time.

Quantities are chosen by what each task's success criterion actually scores, plus
the interaction variables the planner has to control:

  Reacher  success = per-joint |qpos - target_qpos| < threshold
           -> shoulder angle, wrist angle (both cyclic -> twilight colormap),
              finger x, finger y
  Cube     success = ||block_pos - target_pos|| <= 0.04 m   (position only)
           -> block x, y, z  (z is the lift height: the 3-D-specific dimension),
              effector z, gripper opening, effector yaw (cyclic)

Reads the z cache written by visualize_pca_grid.py, so no GPU and no re-encoding —
except the extra physical columns, which are pulled straight from the dataset.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_worldmodel as swm  # noqa: E402
from scripts.visualize_pca_grid import (  # noqa: E402
    N_WINDOWS, SPLIT_SEED, TEST_EPISODE_FRAC, W, load_frames,
)

TEXT = "#3d3d3c"

SPECS = {
    "Reacher": dict(
        dataset="reacher.h5",
        raw=("qpos", "finger_pos"),
        models={"R1 baseline": "baseline (R1)", "R2 L_obj λ=.15": "+L_obj λ=0.15 (R2)",
                "R5 aux w=.4": "+aux w=0.4 (R5)"},
        cache_key="Reacher",
        quantities=[
            ("shoulder angle", lambda c: np.mod(c["qpos"][:, 0] + np.pi, 2*np.pi) - np.pi, "twilight"),
            ("wrist angle",    lambda c: np.mod(c["qpos"][:, 1] + np.pi, 2*np.pi) - np.pi, "twilight"),
            ("finger x",       lambda c: c["finger_pos"][:, 0], "viridis"),
            ("finger y",       lambda c: c["finger_pos"][:, 1], "viridis"),
        ]),
    "Cube": dict(
        dataset="ogbench/cube_single_expert.lance",
        raw=("privileged_block_0_pos", "proprio_effector_pos",
             "proprio_gripper_opening", "proprio_effector_yaw"),
        models={"K1 baseline": "baseline (K1)", "K2 L_obj λ=.1": "+L_obj λ=0.1 (K2)",
                "K4 aux w=.1": "+aux w=0.1 (K4)"},
        cache_key="Cube",
        quantities=[
            ("block x",         lambda c: c["privileged_block_0_pos"][:, 0], "viridis"),
            ("block y",         lambda c: c["privileged_block_0_pos"][:, 1], "viridis"),
            ("block z (lift)",  lambda c: c["privileged_block_0_pos"][:, 2], "magma"),
            ("effector z",      lambda c: c["proprio_effector_pos"][:, 2], "magma"),
            ("gripper opening", lambda c: c["proprio_gripper_opening"][:, 0], "cividis"),
            ("effector yaw",    lambda c: np.mod(2*c["proprio_effector_yaw"][:, 0] + np.pi,
                                                 2*np.pi) - np.pi, "twilight"),
        ]),
}


def sampled_rows(ds, g):
    """Exactly the window sampling visualize_pca_grid.py used, so rows line up
    one-for-one with the cached z."""
    n_ep = len(ds.lengths)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(ds.lengths), np.asarray(ds.offsets)
    elig = [e for e in test_eps if lengths[e] > W]
    picks = g.choice(elig, N_WINDOWS, replace=len(elig) < N_WINDOWS)
    starts = np.array([g.integers(0, lengths[e] - W + 1) for e in picks])
    g0 = offsets[picks] + starts
    kept, last = [], -1
    for i in np.argsort(g0):
        if g0[i] > last:
            kept.append(g0[i]); last = g0[i] + W - 1
    return np.concatenate([k + np.arange(W) for k in kept])


def main():
    C = np.load("eval_results/pca_cache.npz")
    # fail in the first second if a cache key is misspelled, rather than after a
    # multi-GB dataset pull
    missing = [f"{sp['cache_key']}|{k}|z" for sp in SPECS.values() for k in sp["models"].values()
               if f"{sp['cache_key']}|{k}|z" not in C]
    if missing:
        raise KeyError(f"cache keys absent: {missing}\navailable: {sorted(C.files)}")
    for task, spec in SPECS.items():
        ds = swm.data.load_dataset(spec["dataset"], keys_to_load=["pixels", *spec["raw"]])
        rows = sampled_rows(ds, np.random.default_rng(SPLIT_SEED))
        _, cols = load_frames(ds, rows, spec["raw"], "cpu")
        qs = spec["quantities"]
        models = list(spec["models"].items())

        fig, axes = plt.subplots(len(qs), len(models),
                                 figsize=(4.1 * len(models), 3.7 * len(qs)), dpi=150)
        print(f"\n=== {task} (n={len(rows)} frames) ===")
        print(f"{'quantity':18s}" + "".join(f"{m:>18s}" for m, _ in models))
        for qi, (qname, fn, cmap) in enumerate(qs):
            val = np.asarray(fn(cols), dtype=np.float64)
            line = f"{qname:18s}"
            for mi, (mlabel, ckey) in enumerate(models):
                z = C[f"{spec['cache_key']}|{ckey}|z"]
                zc = z - z.mean(0)
                U, S, _ = np.linalg.svd(zc, full_matrices=False)
                xy = U[:, :2] * S[:2]
                rho = max(abs(spearmanr(xy[:, 0], val).statistic),
                          abs(spearmanr(xy[:, 1], val).statistic))
                ax = axes[qi, mi]
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, cmap=cmap, s=5,
                                alpha=0.6, edgecolors="none")
                ax.set_title(f"{mlabel}  —  {qname}", fontsize=8.5, color=TEXT)
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.03, 0.97, f"|ρ| = {rho:.2f}", transform=ax.transAxes,
                        va="top", fontsize=8, color=TEXT)
                if mi == len(models) - 1:
                    fig.colorbar(sc, ax=ax, fraction=0.05, label=qname)
                line += f"{rho:18.3f}"
            print(line)
        fig.suptitle(f"{task} — top-2 PCA of z, coloured by physical quantity "
                     f"(|ρ| = best Spearman of PC1/PC2 with that quantity)",
                     fontsize=11, color=TEXT)
        out = Path(f"eval_results/viz_pca_physical_{task.lower()}.png")
        fig.tight_layout()
        fig.savefig(out, facecolor="white", bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
