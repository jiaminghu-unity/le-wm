# Contact-conditioned Planning Regret — Decisive Motivation Test (2026-10-12)

The single approved diagnostic for the Tactile-Guided Representation track.
Reuses the T6 executed rollouts (80 states × 100 six-step sequences, offset 12,
B1 fixed-context); no training. Regret per state:
r(s) = J_real(argmin-cost sequence) − min_A J_real(A), best-step outcome in
official success geometry; normalized form divides by the candidate outcome
spread. 64/80 states carry a mode label; 5 states' 3-frame context crosses an
episode boundary (flagged, results unchanged without them).

## Q1 — Is B1's selection regret concentrated in contact-related states?

| group | n | norm regret (pred) | norm regret (oracle) | success-possible: pred missed / oracle missed | mean phase |
|---|---|---|---|---|---|
| free (m=0) | 27 | 0.273 | 0.158 | 2/6 · 1/6 | 0.29 |
| contact (m=1,2) | 37 | 0.379 | 0.257 | 7/24 · 3/24 | 0.59 |
| near-transition | 46 | 0.392 | 0.247 | 9/29 · 4/29 | 0.57 |
| no-transition | 34 | 0.227 | 0.140 | 2/6 · 0/6 | 0.26 |

Directionally yes — but contact states sit systematically later in the task
(phase 0.59 vs 0.29), and after phase control the contact effect is **not
significant**: OLS contact coefficient +0.093 (permutation p=0.16),
Mann-Whitney p=0.09. **Suggestive at best; the pre-registered bar ("明显更高")
is not met.**

## Q2 — Do the failure states coincide with wrist-recoverable info the latent lost?

Out-of-fold (5-fold, episode-grouped) probe gaps at each initial state:
mode-probability gap = P_wrist(true mode) − P_latent(true mode);
motion-error gap = latent motion-prediction error − wrist motion-prediction error.

| gap | n | Spearman vs norm regret | contact-only | gap in hi-regret half vs lo |
|---|---|---|---|---|
| mode-prob gap | 60 | **ρ = −0.04 (p=0.74)** | −0.13 (p=0.46) | **+0.048 vs +0.157** |
| motion-err gap | 75 | +0.16 (p=0.16) | +0.13 (p=0.46) | +0.050 vs −0.018 |

**No.** The information the joint latent loses (which is real — the gap is
positive on average) does **not** localize to the states where B1 picks bad
actions; in the mode channel the association runs the *opposite* way (high-regret
states show a *smaller* wrist-over-latent advantage). The chain
"contact info lost ⇒ wrong action" fails its direct test.

## Q3 — Does the oracle cost pick better in the trouble spots?

| group | norm regret pred | oracle | Δ | oracle missed / success-possible |
|---|---|---|---|---|
| contact | 0.379 | 0.257 | +0.122 | 3/24 |
| near-transition | 0.392 | 0.247 | +0.145 | 4/29 |
| hi-regret half | 0.546 | 0.306 | +0.240 | 4/24 |

**Yes, strongly.** With the *same* latent space and the *same* MSE goal cost,
substituting the real next-frame encoding recovers most of the regret exactly
where B1 fails (oracle misses 3/24 success-possible contact states vs pred 7/24).
The failures are dominated by **generic multi-step prediction error**, not by
latent cost geometry and not by contact cues missing from the cost space.

## Decision (per the pre-registered stop condition)

> 如果接触相关状态没有明显更高的物理动作选择损失……停止 ContactWorld USB 上的
> 新 tactile representation 训练。

- Q1: contact elevation not significant after phase control.
- Q2: lost-but-recoverable contact information does not predict decision failure
  (mode channel anti-correlates).
- Q3: the same latent geometry suffices when prediction is right.

**The stop condition triggers: close the USB/RGB tactile-representation line.**
Contact-relevant information in this task is, at the current evidence level,
not decision-relevant information. The binding constraint for the 20.8% → 29.3%
gap is multi-step prediction fidelity (and, secondarily, cost slack), which is a
general world-model problem, not a tactile one.

### Caveats
- n = 80 states (64 labeled): Q1 has limited power; its direction is positive.
  Q2's anti-correlation, not Q1's p-value, is the decisive blocker.
- Probes are linear and measured at the initial state only, not along rollouts.
- This is a diagnostic association, not a causal counterfactual.
- Consistent with: DINO retention fix (+0.032 probe gain) → SR unchanged;
  official Wrist+TacFF vs Wrist-only ≈ +0.17pp on USB/RGB; predictability ×
  control-value overlap ≈ independence (6% vs 6.25%).

### What remains open (not approved, for discussion)
- Task/modality choice: official tactile gains concentrate in point-cloud
  tasks (loose-lid +29pp etc.) — a different deployment question from USB/RGB.
- The general (non-tactile) problem this diagnosis exposes: multi-step
  compounding and demo-scale, candidates for any future direction.
