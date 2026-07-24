# PAEP Reacher — Round-1 Report (2026-07-22)

## 0. Recon findings (Step 0)

- Dataset `quentinll/lewm-reacher` → `reacher.h5` (92 GB raw): 10,000 episodes × 201 steps.
  Columns all as expected: `pixels(224²) action(2) qpos(2) qvel(2) finger_pos(2) target_pos(2)`
  (+`observation(6)` unused by models). Matches paper App. D (10k episodes, SAC-collected, 10 epochs).
- **Angle units: radians confirmed** by full scan — shoulder qpos[0] ∈ [−8.20, 7.91]
  (unbounded, accumulates past ±π as spec expected), wrist qpos[1] ∈ [−2.947, 2.943].
- Env wiring verified: `action_repeat=2` lives inside `DMControlWrapper` (separate from
  data frameskip 5, not double-counted); `render_target=0` → target invisible in pixels
  (justifies excluding `target_pos` from q); success = per-joint |qpos − target_qpos| < threshold.

## 1. Deviations from spec (declared, none silent)

1. **λ_sig = 0.09** (repo default) instead of the spec table's 0.1 — same precedent as Push-T
   ("以仓库为准", REPRODUCE.md).
2. **epochs = 10** (paper App. E; repo yaml default 100 is misleading).
3. **Training data converted to lance** (JPEG q95). Post-hoc check of upstream commit
   `519fa42` shows the author trains Push-T from lance but **Reacher from raw h5** —
   so lance is a real (small) deviation on Reacher only. Candidate contributor to the
   absolute SR gap vs the paper; does not affect paired comparisons.

## 2. Health gates (epoch 0, automated)

All three runs passed: z_norm ≈ 13.5–13.6 (√192 shell), no eff-rank collapse
(R1 59.7 / R2 55.2 / R3 51.9), grad_ratio single-digit (R2 0.45 / R3 0.46),
obj_skipped = 0 everywhere. R3 launched automatically after R2's gate cleared.
R2's val_pred (0.027) ≈ R1's (0.029): the spec's torque-control concern did not materialize.

## 3. Sanity gate

R1 @ T1 on 50 episodes: **94.0%** vs published ≈86 — above the bar; exact binomial CI
[83.5, 98.7] covers 86. Full-set (n=250) T1 = 81.6% (CI [76.8, 86.0]).
Paper protocol (App. F, Table 5 caption): published numbers are means over
**3 training seeds × 50 episodes**; ours are 1 seed × 250 episodes — a 3–5pp absolute
offset is within the combined noise (paper's own seed-σ reaches ±6.5 in Table 6).
Determinism check: the 50 sanity episodes reproduced bit-identically inside the 250-run.

## 4. Headline table (n=250 paired episodes, hash cc13963c0bac)

| Tier (fwd/replan) | R1 base | R2 joints | R3 +finger | R2 vs R1 p | R3 vs R1 p |
|---|---|---|---|---|---|
| T1 (45,000) | 81.6 | 82.0 | 82.8 | 1.000 | 0.780 |
| T2 (11,250) | 79.2 | 83.6 | 82.0 | 0.207 | 0.427 |
| T3 (2,500)  | 67.6 | **74.4** | **74.0** | 0.060 | 0.089 |
| T4 (500)    | 65.2 | 69.6 | 67.6 | 0.207 | 0.561 |
| T5 (150)    | 57.2 | 58.8 | **65.2** | 0.683 | **0.019** (vs R2: 0.040) |

Figure: `budget_sweep_reacher.png`. Per-cell details: `summary_reacher.csv`, per-episode: `results_reacher.csv`.

**Reading:**
- All 10 PAEP cells ≥ baseline (direction fully consistent with Push-T's inverted-U);
  magnitudes ≈ half of Push-T — expected, Reacher's budget cliff is intrinsically shallow
  (baseline only drops 81.6→57.2 across a 300× budget cut vs Push-T's 92→35).
- Zero interference at full budget (T1: +0.4/+1.2).
- **R3's finger term wins the extreme-low-budget tier**: T5 65.2 vs 57.2 (p=0.019),
  also beating joints-only R2 (p=0.040) — end-effector distance in q gives the planner
  a more task-aligned gradient when search is nearly absent.
- λ_obj = 0.1 transferred from Push-T without re-tuning, as the spec mandated.
  No tier is significant at Bonferroni level on its own; n=250/1-seed limits power
  given the ~5pp effect size (Push-T needed the same n for ~9pp).

## 5. Probing (`probing_reacher.csv`)

| target | R1 lin/mlp r | R2 lin/mlp r | R3 lin/mlp r |
|---|---|---|---|
| joints (cos/sin) | .9995/.9997 | .9997/.9997 | .9996/.9997 |
| finger xy | .9989/.9998 | .9994/.9998 | .9996/.9998 |
| **qvel** | **.003/.001** | **.003/.000** | **.007/.002** |

All configs decode pose near-perfectly. The **velocity probe is ≈0 for every config
including the baseline**: a single-frame encoder cannot represent velocity (no motion
blur in dm_control renders). The non-circularity check therefore passes trivially —
L_obj squeezed nothing out, but the probe cannot differentiate configs on this
architecture; velocity lives only in the predictor's multi-frame context.

## 6. Follow-up in flight

- `lewm_r2_reacher_paep_l015_s3072` (λ_obj = 0.15, joints_only) training since 23:42 UTC —
  dose-response point between the Push-T findings (0.1 best at T3; 0.2 trades T3 for T4/T5).
