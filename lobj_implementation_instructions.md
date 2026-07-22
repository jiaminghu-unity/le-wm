# Implementation Instructions: Object-Pose Metric Regularization on LeWM (Push-T)

**Target repo:** https://github.com/lucas-maes/le-wm (official LeWM codebase, built on `stable-worldmodel` + `stable-pretraining`; core model in `jepa.py`)
**Paper:** LeWorldModel, arXiv:2603.19312
**Task:** Push-T only (for now). 4 training configurations, sharing one codebase via config flags.

> 中文注释:本文档给 coding agent 执行。所有"DEFAULT"标记的决定可以直接采用;所有"ASK USER"标记的点,实现前先来问。

---

## 0. Ground rules (apply to everything below)

1. **Do not modify** anything not explicitly listed here: optimizer, LR schedule, batch size, epochs, augmentation, ViT/predictor architecture hyperparams, SIGReg internals, CEM/eval parameters — all stay at the repo's Push-T defaults. The four configs must differ **only** in the flags defined in §4. (中文:除本文档明确要求的改动外,一律沿用官方 repo 默认值,保证四个配置之间只差目标变量。)
2. All four configs must run from the **same commit** of your modified codebase, switched purely by config file. No per-config code branches.
3. Reproduce first, modify second: Config ① (vanilla LeWM) must reach its acceptance bar (§8) before any other config is trained.

---

## 1. Repo reconnaissance (do this first, report back)

Before writing code, inspect the repo and report:

1. Where the encoder embedding `z` is produced (ViT CLS → projector). Identify the exact tensor SIGReg is applied to.
2. Whether the Push-T dataset used by the repo already stores per-frame ground-truth state (pusher x,y; T-block x,y; T-block angle). The DINO-WM-lineage datasets usually do. **If states are absent, dataset must be regenerated with states logged — ASK USER before doing this.**
3. How the dataloader batches sub-trajectories: confirm shapes `obs (B, T, 3, 224, 224)`, `actions (B, T, A_block)`, with `B=128, T=4`, frame-skip 5. Confirm whether episode IDs / frame indices are available per sample (needed for pair sampling, §3.3). If not available, thread them through the dataloader.
4. The exact eval protocol for Push-T (success criterion, #episodes, CEM params). **Use it unchanged for all configs.**

---

## 2. The physical-state vector q (new data plumbing)

**Confirmed: the dataset stores per-frame ground-truth state.** Column mapping:

| dataset column | role |
|---|---|
| `agent_x`, `agent_y` | pusher position → q[0:2] |
| `block_x`, `block_y` | T-block position → q[2:4] |
| `block_angle` | T-block angle → q[4:6] as (cos, sin) |
| `agent_vx`, `agent_vy` | **not used** (future ablation; leave accessible) |
| `action_x`, `action_y` | actions — belong to the action-block pipeline, never enter q |

**Verify angle units before anything else:** scan `block_angle` over the whole dataset; range ≈ [−π, π] or [0, 2π] ⇒ radians (expected); range ≈ [0, 360] ⇒ degrees, convert first. A unit mistake here fails silently. (中文:先全量扫一遍角度取值范围确认单位,弄错不会报错但几何全错。)

Note the raw scales: positions are in pixels (hundreds), cos/sin are O(1) — a ~2-order-of-magnitude gap. This is exactly what the dataset-level standardization below absorbs; do not skip it.

Per frame, build:

```
q_raw = [pusher_x, pusher_y, block_x, block_y, cos(block_theta), sin(block_theta)]   # 6-dim
```

- Angle enters **only** as (cos θ, sin θ). Never use raw θ anywhere. (中文:角度必须用 cos/sin,避免 359°/1° 回绕问题。)
- **Normalization:** compute per-component mean and std **once over the entire training set**, save to a small JSON artifact, and standardize: `q = (q_raw - mean) / std`. Use the same saved stats everywhere (training, probing, diagnostics). Do not recompute per batch.
- The dataloader must return `q (B, T, 6)` aligned with `obs`, plus `episode_id (B,)`.

---

## 3. The new loss: L_obj (Pearson distance-profile alignment)

### 3.1 Which tensor it applies to

L_obj is computed on the **same embedding tensor `Z` that SIGReg uses** (i.e., the encoder-side embedding after whatever projection the config specifies — see §4). Never on predictor outputs. It therefore backpropagates into the encoder (+projector if present) but, by construction, **never touches the predictor** — do not add any stop-gradients or optimizer groups; standard joint backprop on the total loss is correct. (中文:L_obj 打在 encoder 输出 z 上,和 SIGReg 同一位置;梯度天然到不了 predictor,不需要任何手动路由。)

### 3.2 Exact computation

Inputs per training step: `Z (B, T, D)` reshaped to `(N, D)` with `N = B*T`; matching `Q (N, 6)`; `episode_id (N,)`.

```python
def obj_loss(Z, Q, episode_id, K=4096, eps=1e-6):
    i, j = sample_pairs(episode_id, K)            # see 3.3
    x = ((Z[i] - Z[j]) ** 2).sum(-1)              # squared L2. 中文:平方距离,禁止开根号(sqrt 在 0 处梯度爆炸)
    y = ((Q[i] - Q[j]) ** 2).sum(-1)              # Q requires_grad=False (it's data)
    if x.std() < eps or y.std() < eps:            # degenerate batch guard
        return Z.sum() * 0.0                      # zero loss, keep graph valid; log a warning counter
    x_t = (x - x.mean()) / (x.std() + eps)
    y_t = (y - y.mean()) / (y.std() + eps)
    rho = (x_t * y_t).mean()                      # Pearson correlation
    return 1.0 - rho                              # in [0, 2]
```

Hard requirements:

1. Use **squared** distances (no sqrt).
2. Do **not** detach mean/std — gradients must flow through the standardization (this is what makes the loss scale-invariant end to end).
3. `y` (pose distances) is a constant w.r.t. parameters; `x` carries all gradients.
4. The degenerate-batch guard must return a connected zero (as shown) so DDP/graph bookkeeping stays intact, and increment a logged counter `obj_loss_skipped`.

### 3.3 Pair sampling (stratified)

`sample_pairs(episode_id, K)` returns K index pairs from the N=512 flattened embeddings:

- **50% within-episode pairs:** both indices from the same sub-trajectory sample (each sample offers C(4,2)=6 frame pairs). Sample uniformly among all such pairs in the batch. These teach local/fine geometry. (中文:同轨迹近距离对,教局部几何——规划最后一厘米靠它。)
- **50% cross-episode pairs:** two indices with different `episode_id`, uniform random. These teach global layout.
- Resample fresh pairs every training step. `K=4096` DEFAULT; expose as config.

### 3.4 Total loss per config

```
L = L_pred + lambda_sig * SIGReg(Z) + lambda_obj * obj_loss(Z, Q)
```

with `lambda_sig`, `lambda_obj` set per config (§4). `L_pred` is the repo's unmodified teacher-forced next-embedding MSE.

---

## 4. The four configurations

| Config | Encoder head | SIGReg (`lambda_sig`) | L_obj (`lambda_obj`) | Purpose |
|---|---|---|---|---|
| **C1** `lewm_baseline` | ViT CLS → MLP+BN projector (repo default) | 0.1 (repo default) | 0 | Reproduce paper (96±3 SR) |
| **C2p** `obj_projector` | same as C1 | 0 | swept, then fixed | Loss swap, architecture held fixed |
| **C2** `obj_vanilla` | **no projector**: z = ViT CLS after its native final LayerNorm | 0 | same as C2p | Architecture simplification on top of C2p |
| **C3** `sig_plus_obj` | same as C1 | 0.1 | same as C2p | Additivity / interference test |

Config-specific notes:

- **C2 (no projector):** remove the encoder-side MLP+BN projector entirely; `z` is the LN'd CLS token (D=192). The predictor's output must live in the same space as its target: **replace the predictor-side projector with a plain LayerNorm on the predictor output** (so predictions can reach the LN shell that targets live on). DEFAULT — flag any repo detail that makes this awkward and ASK USER. (中文:C2 里砍掉两个投影头;predictor 输出端补一个 LN,让预测和目标住同一个球壳上。)
- **C2/C2p/C3 must never combine SIGReg with a final-LayerNorm head** — SIGReg requires the projector (LN locks embeddings on a norm-√D shell, which is incompatible with a Gaussian target). This is why the table has no "SIGReg + vanilla" cell; assert this combination is unreachable in config validation.
- In C2p and C3, L_obj is applied to the projector output (same tensor as SIGReg). In C2, to the LN'd CLS.

---

## 5. Hyperparameter sweep plan (execution order)

1. Train **C1**, 1 seed. Must pass acceptance (§8) before proceeding.
2. Sweep **C2p** with `lambda_obj ∈ {0.01, 0.1, 1.0}`, 1 seed each. Pick the best by planning success rate. Tie-break by probing (§7).
3. Train final **C1, C2p, C2, C3** × **3 seeds** each, with `lambda_obj` fixed from step 2.
4. Total ≈ 12 full runs + 3 sweep runs. Each run ≈ repo's Push-T default (10 epochs, single GPU, a few hours).

---

## 6. Logging (every run, non-negotiable)

Per training step (or every 50 steps):

1. `loss/pred`, `loss/sigreg` (if active), `loss/obj` (if active), `loss/total`
2. `obj/rho` (the raw Pearson value), `obj/skipped_count`
3. **Gradient balance:** every 200 steps, compute `grad_norm(L_pred, encoder_params)` and `grad_norm(lambda_obj * L_obj, encoder_params)` separately (two extra backward passes on a held copy, or `autograd.grad` with `retain_graph=True`). Log their ratio. Healthy range: within 3× of each other. (中文:配平看梯度不看 loss 读数。)
4. **Latent health:** every epoch, histogram of `||z||` over one batch (C2 should show a spike at √192≈13.86 — that is expected and fine under LN; C1/C2p/C3 should show a spread), plus effective rank of the batch covariance of z (`exp(entropy of normalized eigenvalues)`). Effective rank collapsing toward ~1 = collapse alarm.

---

## 7. Evaluation (identical for all four configs)

1. **Planning success rate:** use the repo's Push-T eval unchanged (CEM 300 candidates / 30 iters / top-30 / σ=1 / horizon 5 blocks; execute full sequence then replan; budget and success criterion exactly as repo). Report mean±std over the 3 seeds.
2. **Probing suite (new script):** freeze encoder; on held-out frames fit (a) linear probe, (b) 2-layer MLP probe (hidden 256, ReLU) from z to each of: pusher (x,y), block (x,y), block (cos θ, sin θ). Report test MSE and Pearson r per target per probe type. Train/test split by episode, not by frame.
3. **Cost-quality diagnostic (new script):** on ~500 held-out frame pairs (t, goal) sampled like the eval protocol (goal = 25 env steps ahead in the same episode), compute Pearson and Spearman correlation between `||z_t − z_g||²` and `||q_t − q_g||²`. This directly measures whether the planning cost improved. (中文:这个诊断就是论文主张的直接证据——latent 距离是否更像物理距离。)

---

## 8. Acceptance criteria & sanity checks

1. **C1 gate:** success rate ≥ 93% (paper: 96.0 ± 2.83). If below after 1 seed, STOP; debug data/eval pipeline before training anything else. Common failure points: frame-skip mismatch, action-block packing, eval goal sampling.
2. **Unit test for `obj_loss`:** (a) `Z` proportional to `Q`-derived coordinates → loss ≈ 0; (b) shuffled `Z` rows → loss ≈ 1; (c) constant `Z` → guard triggers, returns 0 with warning. Include these three as an actual pytest.
3. **Unit test for pair sampling:** verify ~50/50 stratification and no self-pairs (i ≠ j).
4. **C2/C2p/C3 collapse alarm:** if effective rank < 10 or `obj/rho` plateaus below 0.3 in the first epoch, stop and report rather than burning the full run.

---

## 9. Pinned decisions (do not re-litigate; 中文:已拍板)

1. q = pusher + block, 6-dim, cos/sin angle, dataset-level standardization.
2. ℓ = 1 − Pearson on squared distances. (Absolute matching and rank-hinge variants are FUTURE ablations — leave hooks: `obj_loss_type: pearson` in config.)
3. Eval protocol = official repo, unchanged.
4. Optimizer/schedule = official repo defaults, identical across configs.
5. L_obj on encoder-side z only; no SIGReg on predictor outputs (a "version B" future ablation — leave a config hook `sigreg_on_pred: false`).

## 10. ASK USER before deciding

1. ~~If the dataset lacks per-frame ground-truth states~~ — RESOLVED: states confirmed present (see §2 column mapping).
2. If the repo's projector/embedding wiring differs materially from §4's assumptions (e.g., SIGReg applied pre-projector, or no predictor-side projector exists).
3. Any deviation needed from repo defaults for memory/compute reasons.
4. If `block_angle` turns out to be in degrees, or the column semantics differ from §2's mapping.
