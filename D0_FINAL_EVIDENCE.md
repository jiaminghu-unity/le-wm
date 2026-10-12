# D0 — Final Evidence Fixation (ContactWorld USB, 2026-10-12)

This document freezes the evidence base for the Dynamics closure decision. Every
closed-loop number quoted anywhere else is superseded by the tables here.

## 1. Protocol identity across arms (verified)

All arms were evaluated by the same command template (only `--variant`/`--ckpt-path`
differ): official ContactWorld evaluator, goal offsets 12/24/36/48, CEM 100
candidates × 4 iterations × top-8, 100 envs, ≤15 steps, stop-on-success, joint
success = plug AND ee within 1 cm / 15°.

Per-offset SHA-256 hashes of the full `current_info`/`goal_info` tensors at the
first plan call are **byte-identical across arms** (e.g. offset 12:
`init c8cbac12c81b7b31 / goal 827f99beb8a20042` for every arm). Identical initial
states, goals, and success geometry ⇒ per-env pairing is valid.

## 2. Determinism (verified twice)

Re-running any arm reproduces its per-offset success counts **bit-for-bit**
(20/20 cells on the trace rerun; previously 12/12 on fix3pe). All paired
statistics below therefore reflect model differences only.

## 3. Gate label time-indexing (verified)

`provisional_modes.npz: mode[t]` is defined as the mode of transition t→t+1
(preflight docs: "inputs are strictly time-t quantities; the label is the
tactile-derived mode of transition t→t+1"). Training pairs it with (z_t, a_t)
and prediction target z_{t+1} (`ClipWithModes`, `loss_gated`): alignment correct,
causally available at deployment.

## 4. Authoritative closed-loop table (fixed action-history context)

| arm | 12/24/36/48 (/100) | avg SR | notes |
|---|---|---|---|
| **B1 fused vision-only** | 53/26/3/1 | **20.8%** | plain LeWM-style predictor |
| Ours-soft | 47/18/3/0 | 17.0% | tactile CE, soft deploy |
| B3-soft | 46/17/3/0 | 16.5% | no CE, soft deploy (= training mode) |
| DINO (D2) | 45/17/2/0 | 16.0% | frozen DINOv2 backbone |
| Ours-hard | 47/14/2/0 | 15.8% | tactile CE, argmax deploy |
| wrist-only | 42/14/1/0 | 14.2% | |
| B3-hard | 30/7/5/0 | 10.5% | train/deploy routing mismatch |
| front-only | 28/6/0/0 | 8.5% | |
| DUET dual-target | 25/6/1/0 | 8.0% | different cost definition |

Official pretrained front reference: 29.3%.

## 5. Paired per-episode statistics (pooled 400 envs, McNemar exact)

| pair | successes | diff [95% CI] | discordant | p |
|---|---|---|---|---|
| B1 vs Ours-hard | 83 vs 63 | **+5.0pp** [+1.7,+8.2] | 33 vs 13 | **0.0045** |
| B1 vs Ours-soft | 83 vs 68 | **+3.7pp** [+0.7,+6.7] | 27 vs 12 | **0.024** |
| B1 vs B3-soft | 83 vs 66 | **+4.2pp** [+0.8,+7.7] | 34 vs 17 | **0.024** |
| Ours-hard vs B3-hard | 63 vs 42 | +5.3pp [+1.8,+8.8] | 37 vs 16 | 0.0055 |
| **Ours-soft vs B3-soft** | 68 vs 66 | **+0.5pp** [−2.7,+3.5] | 20 vs 18 | **0.87** |
| Ours hard vs soft | 63 vs 68 | −1.3pp [−3.7,+0.8] | 7 vs 12 | 0.36 |
| B3 soft vs hard | 66 vs 42 | **+6.0pp** [+3.0,+8.8] | 31 vs 7 | **0.0001** |

Readings: (i) B1 significantly beats every multi-branch arm under pairing;
(ii) the old "tactile rescues multi-branch" effect (+5.3pp, hard routing)
disappears under fair routing (+0.5pp, p=0.87) and is fully accounted for by
B3's train/deploy routing mismatch (+6.0pp, p=1e-4).

## 6. T6 sequence diagnostics (80 states × 100 six-step sequences, offset 12)

| | Spearman vs physical | top-8 overlap | top-1 regret (best-step) | success-pick* |
|---|---|---|---|---|
| B1 predicted cost | +0.47 | 0.31 | 0.342 | 69% |
| B1 oracle cost | +0.64 | 0.42 | 0.197 | 89% |
| DINO predicted | +0.50 | 0.31 | 0.334 | 74% |
| DINO oracle | +0.61 | 0.38 | 0.221 | 74% |

*among the 35/80 states where some candidate reaches official success geometry
(outcome ≤ 1), fraction where the cost's argmin candidate reaches it.

Contact grouping (B1 pred Spearman): free +0.53, contact +0.45, near-transition
+0.50, no-transition +0.44 — **no contact-specific deficit**; multi-step
degradation is generic compounding. One-step reference (proper 3-frame context):
pred ρ +0.69 > oracle +0.52.

## 7. Voided-data registry

- All zero-padded-context closed-loop SR (every number before 2026-10-12):
  superseded; raw dirs retained for audit.
- `dinofix/b1fix` "first fixed re-eval": VOID — monkeypatch alias loop disabled
  the fix; reproduced buggy numbers bit-for-bit (`*_VOID_zeropad`).
- Smoke-era Test F (wrong npz target array), B1 ckpt selection at T=6
  (pos-emb trained at T=3): both previously voided and re-done.

Data: `results/closedloop_fix2/`, `results/per_env/`, `results/rank_diag/`,
`results/d2_audit/` on this branch; GCS mirror
`gs://prism-training-us/contactworld/outputs/`.
