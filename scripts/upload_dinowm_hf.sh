#!/usr/bin/env bash
# Upload the five DINO-WM baseline checkpoints from GCS to a HuggingFace model repo.
#   usage: HF_TOKEN=hf_xxx ./scripts/upload_dinowm_hf.sh <hf-username-or-org> [repo-name]
#
# Layout on the Hub: one repo, one subfolder per task, each holding
# weights_epoch_10.pt + config.json (the pair swm.wm.utils.load_pretrained expects).
# Re-running is idempotent: upload_folder overwrites in place.
set -euo pipefail

OWNER="${1:?hf username or org}"
REPO="${2:-lewm-dinowm-baselines}"
: "${HF_TOKEN:?set HF_TOKEN to a write token}"
BUCKET=gs://prism-training-us/le-wm

command -v python3 >/dev/null
python3 -c "import huggingface_hub" 2>/dev/null || pip install -q -U huggingface_hub

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
for t in pusht reacher cube tworoom pointmaze; do
  mkdir -p "$STAGE/$t"
  gcloud storage cp "$BUCKET/ckpts_dinowm/dinowm_${t}_s3072/weights_epoch_10.pt" \
                    "$BUCKET/ckpts_dinowm/dinowm_${t}_s3072/config.json" "$STAGE/$t/"
done

cat > "$STAGE/README.md" <<'MD'
---
license: mit
tags: [world-model, dino-wm, jepa, planning]
---
# DINO-WM baselines for the le-wm study

Five DINO-WM world models (frozen DINOv2-small patch encoder + block-causal
predictor, action Embedder tiled per patch), trained with the SAME pipeline as the
LeWM arms they are compared against: 10 epochs on the lance-format datasets, no
proprio, no q supervision, batch 32, seed 3072.

| folder | task | dataset |
|---|---|---|
| pusht | Push-T | pusht_expert_train |
| reacher | Reacher | reacher |
| cube | OGBench Cube (single, expert) | cube_single_expert |
| tworoom | two-room | lewm-tworooms |
| pointmaze | PointMaze UMaze | DINO-WM's released point_maze |

Each folder holds `weights_epoch_10.pt` + `config.json`; load with
`stable_worldmodel.wm.utils.load_pretrained` (the config's `_target_` is the
`DinoWM` wrapper in the le-wm repo, which registers `extra_encoders` as an
`nn.ModuleDict`). Training/eval code: the le-wm repo, `train_dinowm.py` and
`config/train/experiment/dw_*.yaml`.
MD

python3 - "$OWNER/$REPO" "$STAGE" <<'PY'
import sys

from huggingface_hub import HfApi

repo, stage = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=stage, repo_type="model",
                  commit_message="DINO-WM baselines: 5 tasks, same 10-epoch lance pipeline as LeWM arms")
print(f"uploaded -> https://huggingface.co/{repo}")
PY
