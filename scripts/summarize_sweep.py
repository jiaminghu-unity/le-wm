"""Summarize budget-sweep results: per-(config, tier) SR + paired stats vs C1.

Reads results.csv (one row per config x tier x episode), emits summary.csv with:
  SR, bootstrap 95% CI (resample the 50 episodes, 1000 reps),
  mean env_steps of successful episodes, mean wallclock per plan,
  McNemar exact p-value vs C1 at the same tier (paired on episode_id).

Also runs the sanity checks from the task spec:
  * every row carries the same episodes_pusht_50.json hash,
  * SR is weakly monotone non-increasing from T1 -> T5 per config (small
    noise tolerated, big inversions flagged as suspected seeding bugs).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

TIER_ORDER = ["T1", "T2", "T3", "T4", "T5"]
BOOT_REPS = 1000
BOOT_SEED = 0
MONOTONE_TOLERANCE = 6.0  # percentage points of SR; ~noise floor for n=50


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Exact McNemar p-value on paired binary outcomes (a = baseline)."""
    disc_ab = int(((a == 1) & (b == 0)).sum())
    disc_ba = int(((a == 0) & (b == 1)).sum())
    n = disc_ab + disc_ba
    if n == 0:
        return 1.0
    return binomtest(disc_ab, n=n, p=0.5).pvalue


def bootstrap_ci(success: np.ndarray, reps=BOOT_REPS, seed=BOOT_SEED):
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(success), size=(reps, len(success)))
    srs = success[idx].mean(axis=1) * 100.0
    return np.percentile(srs, 2.5), np.percentile(srs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_results/results.csv")
    ap.add_argument("--out", default="eval_results/summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.results)

    hashes = df["episodes_hash"].unique()
    assert len(hashes) == 1, f"MIXED EPISODE SETS in results: {hashes}"
    print(f"episodes_hash consistent: {hashes[0]}")

    rows = []
    for (config, tier), grp in df.groupby(["config", "tier"]):
        grp = grp.sort_values("episode_id")
        succ = grp["success"].to_numpy()
        assert len(succ) == succ.shape[0] == grp["episode_id"].nunique(), (
            f"{config}/{tier}: duplicate or missing episodes"
        )
        lo, hi = bootstrap_ci(succ)
        row = {
            "config": config,
            "tier": tier,
            "candidates": grp["candidates"].iloc[0],
            "iterations": grp["iterations"].iloc[0],
            "elites": grp["elites"].iloc[0],
            "n_episodes": len(succ),
            "sr": 100.0 * succ.mean(),
            "sr_ci_lo": round(lo, 1),
            "sr_ci_hi": round(hi, 1),
            "mean_steps_success": round(
                float(grp.loc[grp.success == 1, "env_steps_used"].mean()), 1
            ) if succ.any() else np.nan,
            "mean_plan_ms": round(float(grp["wallclock_per_plan_ms"].mean()), 1),
        }
        # paired McNemar vs C1 at same tier
        c1 = df[(df.config == "c1") & (df.tier == tier)].sort_values("episode_id")
        if config != "c1" and len(c1) == len(grp):
            a = c1["success"].to_numpy()
            assert (c1["episode_id"].to_numpy() == grp["episode_id"].to_numpy()).all()
            assert (c1["cem_seed"].to_numpy() == grp["cem_seed"].to_numpy()).all(), (
                f"{config}/{tier}: CEM seeds differ from C1 — pairing broken!"
            )
            row["mcnemar_p_vs_c1"] = round(mcnemar_exact(a, grp["success"].to_numpy()), 4)
        else:
            row["mcnemar_p_vs_c1"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out["tier_rank"] = out["tier"].map({t: i for i, t in enumerate(TIER_ORDER)})
    out = out.sort_values(["config", "tier_rank"]).drop(columns="tier_rank")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))

    # monotonicity check T1 -> T5
    for config, grp in out.groupby("config"):
        grp = grp.set_index("tier").reindex(TIER_ORDER).dropna(subset=["sr"])
        srs = grp["sr"].to_numpy()
        for i in range(1, len(srs)):
            rise = srs[i] - srs[: i].min()
            if rise > MONOTONE_TOLERANCE:
                print(
                    f"WARNING {config}: SR rises {rise:.1f}pp at {grp.index[i]} "
                    f"— exceeds noise tolerance, check seeding"
                )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
