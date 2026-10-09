# Provisional tactile dynamics-mode definition — insertion_usb (P2, 2026-10-09)

All statistics are computed per valid within-episode transition (t → t+1), and all
thresholds are percentiles of the **training episodes only** (splits.json, seed 3072).

## Signals
- `A_t` — raw tactile activity: mean cell magnitude of TacFF (10×14×3).
- `D_t` — temporal change: mean cell magnitude of `T_{t+1} − T_t`.
- `A_rel_t` — baseline-relative activity: mean cell magnitude of `T_t − B`, where `B` is
  the median TacFF over the most stable training frames (bottom-20% `D_t`). Rationale:
  the gripper already holds the plug, so zero force ≠ no contact; `B` captures the
  "settled grasp" field.
- `R_t` — spatial redistribution: L2 distance between unit-normalized flattened fields at
  t and t+1, defined only when both norms exceed the train 5th-percentile norm floor
  (guards unstable directions of near-zero fields; otherwise `R_t := 0`).
- Patch geometry (diagnostic): active cells above the train stable-frame 95th-percentile
  noise floor; area, weighted center, center shift.

## Labels (fit on train; applied everywhere)
| mode | name | rule |
|---|---|---|
| 2 | contact transition / redistribution | `D_t ≥ p85(D)` **or** (`R_t ≥ p85(R)` and R valid) |
| 0 | approx. free motion | not mode 2 and `A_rel_t ≤ p40(A_rel)` |
| 1 | sustained constrained contact | not mode 2 and `A_rel_t ≥ p60(A_rel)` |
| −1 | ambiguous | the p40–p60 band (no forced assignment) |

No temporal smoothing is applied: at 10 Hz single-frame events can be real (per proposal §7.4).

## Validation highlights (full numbers: mode_class_distribution.json)
- Distribution (13,470 transitions): mode0 5,128 / mode1 3,563 / mode2 2,246 / ambiguous 2,533 (18.8%).
- Coverage: every mode appears in ≥185 of 201 episodes — no rare-episode class.
- Motion response (key evidence that modes differ dynamically, not just in force):
  at matched command magnitude (quartile bins, smallest 20% of commands excluded), free motion moves the EE **1.7–2.0× further than sustained contact (geo-mean 1.8×)**, consistent across bins; the raw mean-gain figures (45.9/11.4/27.4) are inflated by near-zero-command outliers and kept only as diagnostics;
  plug displacement per step 0.0030 → 0.0018 → 0.0011.
- Phase confound (to be controlled in P3/P4): mean episode progress 0.31 / 0.64 / 0.73
  for modes 0/1/2 — modes are **not** independent of task phase.
- Action convention check: corr(action_xyz, Δee) = 0.88 / 0.90 / 0.62.
- Threshold sensitivity: mode2 share 16.7% at base; 13.6% / 24.5% at D-threshold ±10 pts —
  moderate; conclusions below should be re-checked at ±10 pts (done in P4).

## Known limitations
1. Single global baseline `B`; per-grasp variation is not modeled (flagged for revision if
   P3 probes show baseline drift dominates).
2. Mode 2 mixes "first contact", "re-seating", and "release"; sub-structure is not claimed.
3. The TacFF heat strips are rendered on one global scale (p99.9) so brightness is
   comparable across modes.
