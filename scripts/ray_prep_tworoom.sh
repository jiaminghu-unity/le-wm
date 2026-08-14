#!/usr/bin/env bash
# Stage the two-room dataset: fetch from HuggingFace, report its real schema, convert to
# lance, upload both to GCS.
#   usage: ray_prep_tworoom.sh
#
# two-room is the fourth LeWM environment and the only one this study has not covered.
# It differs from the other three in kind: no physics engine at all -- the env renders
# 224x224 frames from torch directly (stable_worldmodel/envs/two_room/env.py), the state
# is agent xy + target xy + up to three door centres, and the action is a 2-d velocity.
#
# The HF repo ships tworoom.tar.zst, not a bare file, so swm.data.load_dataset's HF branch
# rejects it outright ("expected a top-level *.lance directory or *.h5/*.hdf5 file").
# The archive is downloaded and unpacked here, the same way ray_p5.sh unpacks the cube one.
#
# The schema is PRINTED rather than assumed. Guessing column names cost a wasted round on
# cube (infos use slashes, dataset columns use underscores), so the q variant for this task
# is written only after this job reports what the columns are actually called.
#
# Nothing existing is touched: new GCS names (tworoom.h5, tworoom.lance) under the same
# datasets/ prefix, and no checkpoint or result path is written.
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
HF_REPO=quentinll/lewm-tworooms
ARCH_URL="https://huggingface.co/datasets/$HF_REPO/resolve/main/tworoom.tar.zst"

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd libgl1 libglib2.0-0 libxcb1 \
  libsm6 libxext6 libxrender1
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets

LOG="$SSD/prep_tworoom.log"
: > "$LOG"

# ---- fetch and unpack ----
echo "[fetch] $ARCH_URL" | tee -a "$LOG"
if ! find "$DS" -maxdepth 2 -name 'tworoom*.h5' -o -maxdepth 2 -name 'tworoom*.hdf5' | grep -q .; then
  time curl -fL --retry 3 "$ARCH_URL" | zstd -dc --long=31 | tar -xf - -C "$DS"
fi
echo "[stage] what landed under $DS:" | tee -a "$LOG"
find "$DS" -maxdepth 2 -iname '*tworoom*' -printf '  %p  %s bytes\n' 2>/dev/null | head -20 | tee -a "$LOG"
H5=$(find "$DS" -maxdepth 2 -type f \( -iname 'tworoom*.h5' -o -iname 'tworoom*.hdf5' \) | head -1)
if [ -z "$H5" ]; then
  echo "FATAL: no tworoom h5 after extraction. Full listing:" >&2
  find "$DS" -maxdepth 3 | head -40 >&2
  exit 1
fi
echo "[stage] using $H5" | tee -a "$LOG"

cp -n "$H5" "$DS/tworoom.h5" 2>/dev/null || true
gcloud storage cp "$DS/tworoom.h5" "$BUCKET/datasets/tworoom.h5"
echo "[upload] tworoom.h5 -> $BUCKET/datasets/" | tee -a "$LOG"

# ---- schema, printed rather than assumed ----
python - "$H5" <<'PY' 2>&1 | tee -a "$LOG"
import sys

import hdf5plugin  # noqa: F401
import numpy as np

import stable_worldmodel as swm

ds = swm.data.load_dataset(sys.argv[1])
print(f"[schema] rows: {len(ds)}")
cols = sorted(ds.column_names) if hasattr(ds, "column_names") else None
print(f"[schema] columns: {cols}")
SKIP = {"pixels"}  # 920809 x 224x224x3 ~ 138 GB; get_col_data would materialise it all
for c in cols or []:
    if c in SKIP:
        print(f"  {c:32s} skipped (image column)")
        continue
    try:
        a = np.asarray(ds.get_col_data(c))
    except Exception as e:
        print(f"  {c:32s} <unavailable: {type(e).__name__}>")
        continue
    line = f"  {c:32s} {str(a.dtype):9s} shape {str(a.shape):22s}"
    if a.dtype.kind in "fiu" and a.ndim <= 2 and a.size:
        f = a.reshape(len(a), -1).astype(np.float64)
        keep = f[~np.isnan(f).any(axis=1)]
        if len(keep):
            line += (f" min {np.round(keep.min(0), 2)!s:30.30s}"
                     f" max {np.round(keep.max(0), 2)!s:30.30s}")
    print(line)
for k in ("ep_idx", "episode_idx", "episode_id", "traj_id"):
    if cols and k in cols:
        v = np.asarray(ds.get_col_data(k)).reshape(-1).astype(int)
        print(f"[episodes] {k}: {len(np.unique(v))} unique, "
              f"first lengths {np.bincount(v)[:5].tolist()}")
print("SCHEMA OK")
PY
grep -q "^SCHEMA OK" "$LOG" || { echo "FATAL: schema probe failed" >&2; exit 1; }

# ---- lance, the format the other three tasks trained on ----
OUT="$DS/tworoom.lance"
if [ ! -d "$OUT" ]; then
  echo "[convert] -> tworoom.lance (JPEG q95, writer default)" | tee -a "$LOG"
  python - "$DS/tworoom.h5" "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import sys

import hdf5plugin  # noqa: F401

from stable_worldmodel.data import convert

convert(sys.argv[1], sys.argv[2], dest_format="lance")
PY
fi
du -sh "$OUT" | tee -a "$LOG"
gcloud storage rsync -r "$OUT" "$BUCKET/datasets/tworoom.lance"
gcloud storage cp "$LOG" "$BUCKET/datasets/" || true
echo "TWOROOM PREP DONE"
