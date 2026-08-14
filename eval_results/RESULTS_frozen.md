# Frozen-encoder ablation

Does an arm's advantage live in the **representation** or in the predictor it was
co-trained with? In the original runs SIGReg, `L_obj` and the aux head all act on
`emb = projector(encoder(x))`, and the prediction MSE carries no stop-gradient, so
each arm's predictor grew up chasing a differently-moving representation. Freezing
encoder+projector removes that confound: all three arms then train a predictor from
scratch under the same objective, and the frozen space is the only thing that differs.

**What this design cannot do.** All three frozen arms train their predictor
identically, so the predictor is not an independently manipulated variable. The
"predictor term" below is a residual. What is measurable is whether an advantage
*transfers* to a freshly trained predictor, not which predictor is better.

Frozen: 6.29M parameters held fixed, 11.74M trainable (predictor + pred_proj +
action_encoder), 10 epochs, one training seed (3072) per arm.

## Push-T

50 starts x 300 candidates, headline k=30 (= CEM T1's 300/30).

**End-to-end (original models)**

| arm | rollerr | tau | ovl@30 | erank@30 | sigma | ICC | cmp_noise |
|---|---|---|---|---|---|---|---|
| base | 0.0317 | 0.7395 | 0.724 | 0.0970 | 0.0653 | 0.220 | 0.0816 |
| obj | 0.0244 | 0.7978 | 0.759 | 0.0835 | 0.0559 | 0.208 | 0.0703 |
| aux | 0.0251 | 0.7633 | 0.764 | 0.0860 | 0.0570 | 0.198 | 0.0722 |

**Frozen encoder+projector**

| arm | rollerr | tau | ovl@30 | erank@30 | sigma | ICC | cmp_noise |
|---|---|---|---|---|---|---|---|
| base | 0.0294 | 0.7493 | 0.725 | 0.0949 | 0.0648 | 0.226 | 0.0806 |
| obj | 0.0223 | 0.8047 | 0.763 | 0.0889 | 0.0549 | 0.189 | 0.0699 |
| aux | 0.0237 | 0.7740 | 0.767 | 0.0828 | 0.0550 | 0.190 | 0.0700 |

**tau decomposition** — the original gap splits as
`(orig_m - orig_base) = (A_m - A_base) + (delta_m - delta_base)` with
`delta_m = orig_m - A_m`; the first term is the space, the second the residual.

| contrast | end-to-end | frozen (space) | space share | residual |
|---|---|---|---|---|
| obj - base | +0.0583 | +0.0554 | 95% | +0.0029 |
| aux - base | +0.0239 | +0.0248 | 104% | -0.0009 |

**Within the frozen series, per-start paired Wilcoxon vs baseline** (n=50, no co-training confound):

| arm | rollerr | tau | ovl@30 |
|---|---|---|---|
| obj - base | -0.0071 (p=0.000*) | +0.0554 (p=0.000*) | +0.0373 (p=0.002*) |
| aux - base | -0.0057 (p=0.000*) | +0.0248 (p=0.001*) | +0.0413 (p=0.000*) |

## Reacher

64 starts x 300 candidates, headline k=30 (= CEM T1's 300/30).

**End-to-end (original models)**

| arm | rollerr | tau | ovl@30 | erank@30 | sigma | ICC | cmp_noise |
|---|---|---|---|---|---|---|---|
| base | 0.1824 | 0.5976 | 0.647 | 0.1849 | 0.2121 | 0.096 | 0.2852 |
| obj | 0.1586 | 0.6221 | 0.659 | 0.2038 | 0.2049 | 0.066 | 0.2800 |
| aux | 0.1709 | 0.6084 | 0.657 | 0.1808 | 0.2043 | 0.064 | 0.2795 |

**Frozen encoder+projector**

| arm | rollerr | tau | ovl@30 | erank@30 | sigma | ICC | cmp_noise |
|---|---|---|---|---|---|---|---|
| base | 0.1809 | 0.5990 | 0.647 | 0.1858 | 0.2112 | 0.096 | 0.2840 |
| obj | 0.1585 | 0.6232 | 0.662 | 0.2014 | 0.2052 | 0.066 | 0.2804 |
| aux | 0.1714 | 0.6091 | 0.659 | 0.1810 | 0.2049 | 0.066 | 0.2801 |

**tau decomposition** — the original gap splits as
`(orig_m - orig_base) = (A_m - A_base) + (delta_m - delta_base)` with
`delta_m = orig_m - A_m`; the first term is the space, the second the residual.

| contrast | end-to-end | frozen (space) | space share | residual |
|---|---|---|---|---|
| obj - base | +0.0244 | +0.0241 | 99% | +0.0003 |
| aux - base | +0.0107 | +0.0101 | 94% | +0.0006 |

**Within the frozen series, per-start paired Wilcoxon vs baseline** (n=64, no co-training confound):

| arm | rollerr | tau | ovl@30 |
|---|---|---|---|
| obj - base | -0.0224 (p=0.000*) | +0.0241 (p=0.001*) | +0.0151 (p=0.005*) |
| aux - base | -0.0095 (p=0.000*) | +0.0101 (p=0.018*) | +0.0120 (p=0.015*) |

## OGBench Cube

64 starts x 300 candidates, headline k=30 (= CEM T1's 300/30).

**End-to-end (original models)**

| arm | rollerr | tau | ovl@30 | erank@30 | sigma | ICC | cmp_noise |
|---|---|---|---|---|---|---|---|
| base | 0.2654 | 0.3987 | 0.397 | 0.2378 | 0.1856 | 0.224 | 0.2312 |
| obj | 0.2675 | 0.4476 | 0.374 | 0.2168 | 0.2026 | 0.303 | 0.2392 |
| aux | 0.2109 | 0.4362 | 0.404 | 0.2223 | 0.1722 | 0.262 | 0.2092 |

**Frozen encoder+projector**

| arm | rollerr | tau | ovl@30 | erank@30 | sigma | ICC | cmp_noise |
|---|---|---|---|---|---|---|---|
| base | 0.2935 | 0.3797 | 0.386 | 0.2451 | 0.1954 | 0.254 | 0.2388 |
| obj | 0.2794 | 0.4226 | 0.360 | 0.2309 | 0.2103 | 0.323 | 0.2447 |
| aux | 0.2318 | 0.4110 | 0.385 | 0.2384 | 0.1830 | 0.301 | 0.2163 |

**tau decomposition** — the original gap splits as
`(orig_m - orig_base) = (A_m - A_base) + (delta_m - delta_base)` with
`delta_m = orig_m - A_m`; the first term is the space, the second the residual.

| contrast | end-to-end | frozen (space) | space share | residual |
|---|---|---|---|---|
| obj - base | +0.0489 | +0.0429 | 88% | +0.0060 |
| aux - base | +0.0375 | +0.0313 | 84% | +0.0062 |

**Within the frozen series, per-start paired Wilcoxon vs baseline** (n=64, no co-training confound):

| arm | rollerr | tau | ovl@30 |
|---|---|---|---|
| obj - base | -0.0141 (p=0.069) | +0.0429 (p=0.083) | -0.0266 (p=0.147) |
| aux - base | -0.0617 (p=0.000*) | +0.0313 (p=0.060) | -0.0016 (p=0.961) |

## Reading

- On Push-T and Reacher the frozen and end-to-end series agree almost cell for cell
  (Reacher's tau gap: +0.0244 end-to-end, +0.0241 frozen). The advantage there is a
  transferable property of the representation.
- Cube is the exception: its obj arm reaches significance on no frozen metric, and
  the end-to-end and frozen tau p-values straddle 0.05 with point estimates 0.006
  apart -- both should be read as borderline, not as two series disagreeing.
- rollerr is comparable in absolute terms between the two series because the frozen
  models' encoder+projector are bit-identical to the originals, so `z_true`, `z_goal`
  and the pairwise-distance scale are the same numbers. Cross-arm comparisons always
  divide by each model's own mean pairwise distance.

**Limit.** One training seed per arm. Per-start pairing measures consistency across
starts for a fixed model, not reproducibility across retraining.
