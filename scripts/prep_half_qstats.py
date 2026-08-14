"""Compute a reduced-q variant's q_stats once, and prove it is the full variant's
stats restricted to the surviving coordinates.

Run before training so that both arms of a task (obj and aux) read byte-identical
normalization from a file, exactly as scripts/make_cube_qstats.py did for the
original runs -- two runs recomputing independently would almost certainly agree,
but "almost certainly" is not the protocol.

The check that matters is the second one. If the reduced q's per-component mean/std
differed from the full q's, the retained coordinates would be scaled differently than
in the original experiment and a change in results could not be attributed to the
withheld coordinates alone. Since every reduced variant is a strict coordinate subset
(q_half.HALF_OF), the stats must match exactly on the kept indices, and this script
fails loudly if they do not.

    usage: prep_half_qstats.py <dataset_name> <half_variant> [full_stats.json]

The output path is derived exactly as train.py:177 derives it, from STABLEWM_HOME --
including the subdirectory in a name like ogbench/cube_single_expert.lance, whose stats
file therefore lives under datasets/ogbench/. Writing it anywhere else means training
silently recomputes instead of reading the file this script verified.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402

import q_half  # noqa: E402
import utils  # noqa: E402

utils.Q_VARIANTS.update(q_half.Q_VARIANTS_HALF)

DATASET_NAME = sys.argv[1]          # e.g. ogbench/cube_single_expert.lance
VARIANT = sys.argv[2]               # e.g. cube_effector_only
FULL_STATS = Path(sys.argv[3]) if len(sys.argv) > 3 else None

full_variant, dim_full, dim_half, kept = q_half.HALF_OF[VARIANT]

# same two lines train.py uses, so the file lands where training will look for it
DS_DIR = Path(swm.data.utils.get_cache_dir(None, sub_folder="datasets"))
out = DS_DIR / f"{DATASET_NAME}.q_stats.{VARIANT}.json"
out.parent.mkdir(parents=True, exist_ok=True)

dataset = swm.data.load_dataset(DATASET_NAME, transform=None, cache_dir=None)

# ---- 1. builders agree on real data -------------------------------------------
half_fn, half_srcs, _ = utils.Q_VARIANTS[VARIANT]
full_fn, full_srcs, _ = utils.Q_VARIANTS[full_variant]
cols = {s: torch.from_numpy(np.asarray(dataset.get_col_data(s)))
        for s in set(half_srcs) | set(full_srcs)}
mask = torch.ones(next(iter(cols.values())).size(0), dtype=torch.bool)
for c in cols.values():
    mask &= ~torch.isnan(c).any(dim=1)
q_half_v = half_fn(*[cols[s][mask] for s in half_srcs])
q_full_v = full_fn(*[cols[s][mask] for s in full_srcs])
assert q_half_v.shape[-1] == dim_half, f"{VARIANT} gave {q_half_v.shape[-1]}d, want {dim_half}"
assert q_full_v.shape[-1] == dim_full, f"{full_variant} gave {q_full_v.shape[-1]}d, want {dim_full}"
d = (q_half_v - q_full_v[:, list(kept)]).abs().max().item()
assert d == 0.0, f"{VARIANT} is not the {kept} subset of {full_variant}: max|diff|={d}"
print(f"[check] {VARIANT} == {full_variant}[{list(kept)}] exactly, "
      f"{dim_full}d -> {dim_half}d over {int(mask.sum())} frames", flush=True)

# ---- 2. write the stats through the real code path ----------------------------
utils.get_q_normalizer(dataset, out, VARIANT)
stats = json.loads(out.read_text())
print(f"[stats] {out.name}: dim={len(stats['mean'])}", flush=True)
print("  mean", np.round(stats["mean"], 4).tolist())
print("  std ", np.round(stats["std"], 4).tolist())

# ---- 3. and that they are the full variant's stats, restricted ----------------
if FULL_STATS and FULL_STATS.exists():
    f = json.loads(FULL_STATS.read_text())
    for key in ("mean", "std"):
        got = np.asarray(stats[key], dtype=np.float64)
        want = np.asarray(f[key], dtype=np.float64)[list(kept)]
        m = float(np.abs(got - want).max())
        # both are float32 sums over the same values in the same order; they agree
        # bit for bit in practice, and a real mismatch means the wrong columns
        assert m < 1e-6, f"{key} differs from {full_variant}[{list(kept)}]: max|diff|={m}"
    print(f"[check] stats == {FULL_STATS.name} restricted to {list(kept)}", flush=True)
else:
    print(f"[warn] full-variant stats not supplied; skipped the restriction check",
          flush=True)

print(f"QSTATS OK {out}")
