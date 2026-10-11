# P0 Synchronization & Split Report (insertion_usb)
Date: 2026-10-09 · dataset: data/demo_data/insertion_usb · audit: preflight/audit_dataset.py

## Episode structure
- 201 episodes, 13,671 frames, 13,470 valid within-episode transitions (matches proposal §3.1).
- episode_ends strictly monotonic; lengths 37–112 (mean 68.0).

## Alignment invariants (verified)
- All data arrays share first axis length 13,671 == episode_ends[-1]: action, front, wrist,
  tactile_force_field_right, tactile_depth_right, tactile_rgb_right, poses, dof, pointcloud.
- Row t is one synchronized sample across all keys -> both cameras use identical indices by construction.
- Transition rule: (O_t, a_t) -> O_{t+1} with t, t+1 inside [start, end) of one episode;
  exactly 200 boundary pairs are excluded. frameskip=1.
- Terminal action of each episode is dropped by the adapter (K obs -> K-1 actions used).

## Image preprocessing
- Stored: 256x256x3 float32 HWC in [0,1] (front frame0 min 0.0 max 1.0; wrist 0.086–1.0).
- For LeWM ViT-Tiny: resize 256->224 (antialias), keep [0,1] float; identical transform for
  both views; no geometric augmentation in preflight.

## TacFF
- shape (10,14,3) per frame; values approx ±0.003, std 4e-4, 0 NaN / 0 Inf.
- Thresholds in P2 must be fitted on TRAIN episodes only, in these native units.

## Canonical split (seed 3072, episode-level, 70/15/15)
- train 141 / val 30 / test 30 episodes -> outputs/preflight/usb/splits.json.
- Prior-era splits.json was lost with the old VMs; this file is the new canonical split and
  is mirrored to gs://prism-training-us/contactworld/outputs/preflight/usb/.
- Official paper-protocol baselines (§3.3) remain labeled as paper-protocol, not this split.
