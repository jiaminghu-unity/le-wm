# Tactile-Guided Multi-View LeWM — Preflight (ContactWorld / USB insertion)

Pre-training feasibility study (Oct 2026 proposal): a vision-only multi-view JEPA
(front + wrist RGB, DUET-style bidirectional cross-attention, original LeWM causal
Transformer predictor + SIGReg) with a tactile-supervised dynamics-mode gate used
only at training time. This branch is independent of the `reproduce` branch
(Auto-SCALE / OGBench work); it shares the repo for hosting only.

## Layout
- `touchwm/` — model code
  - `lewm_modules.py` — verbatim copies of the official LeWM blocks (ARPredictor,
    SIGReg, Embedder, MLP, attention blocks)
  - `multiview_encoder.py` — shared ViT-Tiny (192d, patch 14) + bidirectional
    cross-attention + 8-query pooling → one joint latent
  - `lewm_adapter.py` — MultiViewJEPA: encode / predict / rollout / loss
    (pred-MSE + 0.09·SIGReg, end-to-end, no EMA, no stop-grad)
  - `data_adapter.py` — episode-aware Zarr clip dataset (never crosses episode
    boundaries; terminal action dropped; 256→224; frameskip 1)
- `preflight/` — P0–P4 scripts
  - `audit_dataset.py` (P0: manifest, alignment invariants, canonical split seed 3072)
  - `extract_tactile_events.py` (P2: tactile statistics, provisional modes, examples)
  - `mode_definition.md` (P2: label rules + validation highlights)
  - `smoke_multiview.py` (P1: shapes / gradients / overfit / collapse / view
    ablation / 6-step rollout)

## Status (2026-10-09)
- P0 ✅ 201 eps / 13,671 frames / 13,470 transitions; split 141/30/30 (seed 3072).
- P2 ✅ modes {free, sustained contact, transition, ambiguous}; at matched command magnitude, the same command moves the end effector 1.7–2.0× further in free motion than in sustained contact (geo-mean 1.8×; the raw mean-gain 4× was inflated by near-zero-command outliers).
- P1 ✅ 5/6 (Isaac-sim smoke pending on the render-capable L4 VM; A100 cannot render).

All run artifacts mirror to `gs://prism-training-us/contactworld/outputs/preflight/`.
Results board: Tactile Hard-Negative Board (V8).

## Deviations from official repos (logged per proposal §11.3)
- D1: ViT-Tiny re-implemented at identical spec instead of wrapping HF ViT
  (both train from scratch; avoids the HF dependency on bare VMs).
- D2: the projector MLP operates on the fused multi-view latent, not a CLS token.
