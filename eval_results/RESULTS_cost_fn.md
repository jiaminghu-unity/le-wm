# Planner cost function: L1 and cosine against squared L2

The shipped planning cost is `||z_hat - z_goal||^2` over the terminal step only
(`stable_worldmodel/wm/lewm/lewm.py`, `LeWM.criterion`). These rounds rescore the
SAME checkpoints with `||.||_1` and with cosine distance. No retraining: criterion is
reached only through `get_cost`, which is planning-only.

Only cem and icem are valid arms and that was fixed before running: both select by
rank alone (`topk(costs, largest=False)`), whereas mppi weights by
`softmax(-(cost-min)/0.5)` without rescaling for spread and gd descends the cost
gradient at a fixed lr, so a change under those two would not be attributable to the
cost's shape. Each run asserts its cost numerically before evaluating -- a wiring
mistake would otherwise file one variant's numbers under another's name.

The comparison is exactly paired against the squared-L2 results: same checkpoints,
same episode sets, same `cem_seed = crc32("episode_id|tier")`, same tiers, same code
path.

## How large an intervention each variant is

Measured before the sweeps, on the same 300 candidates per start (`scripts/probe_latent_geometry.py`). tau=1 would mean the variant reproduces the
shipped ranking exactly and could not measure anything. `norm%` is the share of
Var(cost) carried by the `||z_hat||^2` term, which is what the dot product drops
and cosine divides out.

| task | arm | tau vs L1 | tau vs cos | tau vs dot | ovl@30 cos | norm% |
|---|---|---|---|---|---|---|
| Push-T | base | 0.874 | 0.775 | 0.712 | 0.813 | 16.3 |
| Push-T | obj | 0.907 | 0.710 | 0.655 | 0.823 | 22.0 |
| Push-T | aux | 0.880 | 0.769 | 0.738 | 0.847 | 15.8 |
| Reacher | base | 0.870 | 0.901 | 0.901 | 0.985 | 1.6 |
| Reacher | obj | 0.879 | 0.906 | 0.900 | 0.993 | 1.6 |
| Reacher | aux | 0.868 | 0.895 | 0.896 | 0.980 | 1.7 |
| OGBench Cube | base | 0.874 | 0.767 | 0.725 | 0.837 | 17.7 |
| OGBench Cube | obj | 0.919 | 0.747 | 0.654 | 0.832 | 29.7 |
| OGBench Cube | aux | 0.889 | 0.820 | 0.753 | 0.888 | 17.2 |

L1 changes only ~12% of the pairwise ordering, so **the L1 round was**
**underpowered by construction** -- a fact that should have been measured before
spending the jobs, not after. Cosine is a 1.5-2x stronger perturbation on Push-T
and Cube and near-identity on Reacher (98% of the same 30 elites).

## L1  (sum |z_hat - z_goal|)

**Push-T / cem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 70.50 | 70.53 | +0.03 | 1.000 | 145/144 | 1.000 |
| L_obj | 73.87 | 74.80 | +0.93 | 0.062 | 137/109 | 0.085 |
| aux q-head | 74.90 | 74.37 | -0.53 | 0.625 | 120/136 | 0.349 |

Difference-in-differences vs baseline: L_obj +0.90 (p=0.406); aux q-head -0.57 (p=1.000)

**Push-T / icem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 63.90 | 63.77 | -0.13 | 0.625 | 108/112 | 0.840 |
| L_obj | 68.20 | 67.40 | -0.80 | 0.625 | 77/101 | 0.084 |
| aux q-head | 69.30 | 68.03 | -1.27 | 0.188 | 78/116 | 0.008* |

Difference-in-differences vs baseline: L_obj -0.67 (p=0.656); aux q-head -1.13 (p=0.250)

**Reacher / cem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 70.57 | 70.07 | -0.50 | 0.688 | 261/276 | 0.546 |
| L_obj | 71.53 | 71.17 | -0.37 | 0.625 | 238/249 | 0.650 |
| aux q-head | 70.43 | 71.10 | +0.67 | 0.344 | 258/238 | 0.394 |

Difference-in-differences vs baseline: L_obj +0.13 (p=0.750); aux q-head +1.17 (p=0.438)

**Reacher / icem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 78.03 | 77.27 | -0.77 | 0.062 | 145/168 | 0.214 |
| L_obj | 81.13 | 81.63 | +0.50 | 0.688 | 132/117 | 0.375 |
| aux q-head | 80.17 | 79.80 | -0.37 | 0.844 | 124/135 | 0.534 |

Difference-in-differences vs baseline: L_obj +1.27 (p=0.312); aux q-head +0.40 (p=0.688)

**OGBench Cube / cem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 61.37 | 61.13 | -0.23 | 0.688 | 113/120 | 0.694 |
| L_obj | 64.17 | 64.13 | -0.03 | 0.688 | 118/119 | 1.000 |
| aux q-head | 63.50 | 63.67 | +0.17 | 1.000 | 104/99 | 0.779 |

Difference-in-differences vs baseline: L_obj +0.20 (p=0.688); aux q-head +0.40 (p=0.688)

**OGBench Cube / icem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 65.23 | 65.50 | +0.27 | 0.469 | 51/43 | 0.470 |
| L_obj | 70.50 | 70.17 | -0.33 | 0.250 | 31/41 | 0.289 |
| aux q-head | 68.60 | 67.97 | -0.63 | 0.250 | 33/52 | 0.050 |

Difference-in-differences vs baseline: L_obj -0.60 (p=0.094); aux q-head -0.90 (p=0.188)

## cosine distance

**Push-T / cem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 70.50 | 70.40 | -0.10 | 0.781 | 155/158 | 0.910 |
| L_obj | 73.87 | 72.87 | -1.00 | 0.062 | 139/169 | 0.098 |
| aux q-head | 74.90 | 74.87 | -0.03 | 1.000 | 134/135 | 1.000 |

Difference-in-differences vs baseline: L_obj -0.90 (p=0.438); aux q-head +0.07 (p=1.000)

**Push-T / icem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 63.90 | 64.60 | +0.70 | 0.250 | 120/99 | 0.176 |
| L_obj | 68.20 | 67.47 | -0.73 | 0.219 | 115/137 | 0.186 |
| aux q-head | 69.30 | 69.77 | +0.47 | 0.688 | 106/92 | 0.356 |

Difference-in-differences vs baseline: L_obj -1.43 (p=0.031*); aux q-head -0.23 (p=0.688)

**Reacher / cem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 70.57 | 70.03 | -0.53 | 0.156 | 199/215 | 0.461 |
| L_obj | 71.53 | 71.17 | -0.37 | 0.594 | 171/182 | 0.595 |
| aux q-head | 70.43 | 70.10 | -0.33 | 0.312 | 196/206 | 0.654 |

Difference-in-differences vs baseline: L_obj +0.17 (p=0.812); aux q-head +0.20 (p=0.844)

**Reacher / icem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 78.03 | 77.67 | -0.37 | 0.438 | 110/121 | 0.511 |
| L_obj | 81.13 | 80.37 | -0.77 | 0.156 | 72/95 | 0.088 |
| aux q-head | 80.17 | 79.80 | -0.37 | 0.469 | 96/107 | 0.483 |

Difference-in-differences vs baseline: L_obj -0.40 (p=0.500); aux q-head +0.00 (p=0.938)

**OGBench Cube / cem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 61.37 | 61.20 | -0.17 | 0.812 | 110/115 | 0.790 |
| L_obj | 64.17 | 64.57 | +0.40 | 0.469 | 130/118 | 0.485 |
| aux q-head | 63.50 | 63.47 | -0.03 | 0.844 | 112/113 | 1.000 |

Difference-in-differences vs baseline: L_obj +0.57 (p=0.344); aux q-head +0.13 (p=0.562)

**OGBench Cube / icem** — 5 tiers x 6 seeds

| arm | squared L2 | variant | delta | p | flips up/down | McNemar |
|---|---|---|---|---|---|---|
| baseline | 65.23 | 64.80 | -0.43 | 0.406 | 41/54 | 0.218 |
| L_obj | 70.50 | 70.57 | +0.07 | 0.594 | 52/50 | 0.921 |
| aux q-head | 68.60 | 68.43 | -0.17 | 0.688 | 43/48 | 0.675 |

Difference-in-differences vs baseline: L_obj +0.50 (p=0.344); aux q-head +0.27 (p=0.625)

## Reading

- **Neither variant moves SR.** Across both rounds the largest change is 1.00 pp, and
  the significant cells are at the rate chance produces at this number of tests.
  Isolated cells did appear -- Push-T's aux under L1, Reacher's baseline under L1,
  Push-T's obj under cosine -- but none replicated on another task and they point at
  different arms.
- One mechanism was proposed and then refuted by its own prediction. Push-T's obj fell
  under cosine on both solvers, and `norm%` said obj raises the `||z_hat||^2` share
  (22.0 vs baseline 16.3), so Cube -- where the excess is twice as large (29.7 vs
  17.7) -- should have fallen further. It did not move (+0.40 / +0.07, and the
  difference-in-differences is positive). The Push-T cells are noise.
- The structural reason is in the solver, not the cost: CEM executes the **mean of
  its 30 elites** (`cem.py:271`), so swapping a few elites barely moves the action.
  That damps any cost change, and cosine keeps 81-89% of the same elites.

**What this does not show.** That the cost function is irrelevant in general -- only
that perturbations of this size, under a planner that averages its elites, do not
reach SR.
