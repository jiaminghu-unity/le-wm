#!/usr/bin/env bash
# Stage DINO-WM's PointMaze (UMaze) dataset: download from OSF, convert to h5 and lance,
# upload both to GCS.
#   usage: ray_prep_pointmaze.sh
#
# WHY THIS DATASET. It is the only ready-made source of a 224x224 pixel maze: OGBench has
# no visual-pointmaze (verified — the .npz 404s while the state version is 200), and its
# visual-antmaze is 64x64, which would not match the resolution every other task here uses.
# DINO-WM's release is 224x224, the same as ours, so no resampling deviation is introduced.
#
# WHAT IS IN IT (measured, not assumed):
#   states.pth       (2000, 100, 4) float64   (x, y, vx, vy)
#   actions.pth      (2000, 100, 2) float64   in [-1, 1]
#   seq_lengths.pth  (2000,)                  all exactly 100
#   obses/episode_NNN.pth (100,224,224,3) uint8 THWC, 2000 files, 30 GB raw
#                        (names are zero-padded to 3 digits below 1000)
# 2000 rollouts x 100 steps = 200,000 frames. DINO-WM describe these as fully random
# trajectories, not expert ones, which is a real risk for this task: if no arm can plan in
# it, a ceiling/floor effect will hide any difference between arms. That is reported, not
# worked around.
#
# The h5 writer chunks one frame per chunk and does not compress, so the h5 is ~30 GB. That
# chunking is the right shape for this pipeline: the cube dataset's (100,224,224,3) chunks
# caused a 25x read amplification because a 4-frame clip decompressed 100 frames.
#
# Nothing existing is touched: new GCS names (pointmaze.h5, pointmaze.lance).
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
OSF_VIEW=a56a296ce3b24cceaf408383a175ce28
OSF_FILE_ID=678ac918567274d368282c6d   # datasets/point_maze.zip on osf.io/bmw48
ZIP_URL="https://files.osf.io/v1/resources/bmw48/providers/osfstorage/${OSF_FILE_ID}?view_only=${OSF_VIEW}"

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
echo "[env] $(hostname), free=$(df -h --output=avail "$SSD" | tail -1 | tr -d ' ')"

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

LOG="$SSD/prep_pointmaze.log"; : > "$LOG"

ZIP="$DS/point_maze.zip"
if [ ! -f "$ZIP" ]; then
  echo "[fetch] OSF point_maze.zip (0.72 GB)" | tee -a "$LOG"
  time curl -fL --retry 3 -o "$ZIP" "$ZIP_URL"
fi
ls -l "$ZIP" | awk '{printf "  %.2f GB\n",$5/1e9}' | tee -a "$LOG"

H5="$DS/pointmaze.h5"
# "File exists" is NOT the same as "conversion finished". A crashed earlier run left a 4 KB
# stub h5 on this worker's SSD, this guard treated it as done, skipped the conversion, and
# the read-back then failed on a missing 'ep_len'. So validate the CONTENT: the file must
# carry ep_len and the full row count, otherwise it is discarded and rebuilt.
h5_complete() {
  [ -f "$H5" ] || return 1
  python - "$H5" <<'PYCHK' >/dev/null 2>&1
import sys

import h5py

with h5py.File(sys.argv[1], "r") as f:
    assert "ep_len" in f and "ep_offset" in f, "missing episode index"
    assert f["ep_len"].shape[0] == 2000, f["ep_len"].shape
    assert f["pixels"].shape[0] == 200000, f["pixels"].shape
PYCHK
}
if h5_complete; then
  echo "[convert] h5 already complete, reusing" | tee -a "$LOG"
else
  [ -f "$H5" ] && { echo "[convert] discarding incomplete h5 ($(du -h "$H5" | cut -f1))" | tee -a "$LOG"; rm -f "$H5"; }
  echo "[convert] zip -> h5" | tee -a "$LOG"
  time python - "$ZIP" "$H5" <<'PY' 2>&1 | tee -a "$LOG"
import io
import re
import sys
import zipfile

import numpy as np
import torch

from stable_worldmodel.data.formats.hdf5 import HDF5Writer

zip_path, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(zip_path)
L = lambda n: torch.load(io.BytesIO(z.read(f"point_maze/{n}")), map_location="cpu",
                         weights_only=False)
S = torch.as_tensor(L("states.pth")).numpy()
A = torch.as_tensor(L("actions.pth")).numpy()
SL = torch.as_tensor(L("seq_lengths.pth")).numpy()
n_ep = len(S)
assert S.shape[1:] == (100, 4) and A.shape[1:] == (100, 2), (S.shape, A.shape)
assert (SL == 100).all(), f"expected fixed length 100, got {np.unique(SL)}"
print(f"[convert] {n_ep} episodes x 100 steps, state 4-d, action 2-d", flush=True)

with HDF5Writer(out, mode="overwrite") as w:
    # Episode filenames are zero-padded to THREE digits up to 999 and then plain
    # ("episode_000.pth" … "episode_999.pth", "episode_1000.pth" …), so building the name
    # with f"episode_{e}.pth" misses every episode below 1000. Index the archive's own
    # entry list instead of reconstructing names.
    by_idx = {}
    for name in z.namelist():
        m = re.search(r"obses/episode_(\d+)\.pth$", name)
        if m:
            by_idx[int(m.group(1))] = name
    missing = [e for e in range(n_ep) if e not in by_idx]
    assert not missing, f"archive lacks episodes {missing[:10]} (of {len(missing)})"
    assert len(by_idx) == n_ep, f"{len(by_idx)} obs files vs {n_ep} state rows"

    for e in range(n_ep):
        px = torch.load(io.BytesIO(z.read(by_idx[e])),
                        map_location="cpu", weights_only=False)
        px = torch.as_tensor(px).numpy()
        assert px.shape == (100, 224, 224, 3) and px.dtype == np.uint8, (px.shape, px.dtype)
        st = S[e].astype(np.float32)
        w.write_episode({
            "pixels": px,
            "action": A[e].astype(np.float32),
            # state carries velocity too: the eval protocol needs the full state to reset
            # the simulator, even though q uses position only
            "state": st,
            # dedicated 2-d position column, mirroring two-room's pos_agent, so the q
            # builder reads a named column and the dimension assertion means something
            "pos": st[:, :2].copy(),
            "ep_idx": np.full(100, e, dtype=np.int32),
            "step_idx": np.arange(100, dtype=np.int64),
        })
        if (e + 1) % 200 == 0:
            print(f"[convert] {e + 1}/{n_ep} episodes", flush=True)
print("H5 OK")
PY
  grep -q "^H5 OK" "$LOG" || { echo "FATAL: h5 conversion failed" >&2; exit 1; }
fi
du -sh "$H5" | tee -a "$LOG"

# ---- read the h5 back through the same path training uses, and print the schema ----
python - "$H5" <<'PY' 2>&1 | tee -a "$LOG"
import sys

import hdf5plugin  # noqa: F401
import numpy as np

import stable_worldmodel as swm

ds = swm.data.load_dataset(sys.argv[1])
cols = sorted(ds.column_names)
print(f"[schema] rows {len(ds)}  columns {cols}")
for c in cols:
    if c == "pixels":
        print(f"  {c:10s} skipped (image column; 200k x 224x224x3 = 30 GB)")
        continue
    a = np.asarray(ds.get_col_data(c))
    line = f"  {c:10s} {str(a.dtype):9s} {str(a.shape):18s}"
    if a.dtype.kind in "fiu" and a.ndim <= 2:
        f = a.reshape(len(a), -1).astype(np.float64)
        line += f" min {np.round(f.min(0), 3)!s:26.26s} max {np.round(f.max(0), 3)!s:26.26s}"
    print(line)
v = np.asarray(ds.get_col_data("ep_idx")).reshape(-1)
print(f"[episodes] {len(np.unique(v))} unique, lengths {np.bincount(v)[:3].tolist()}")
print("SCHEMA OK")
PY
grep -q "^SCHEMA OK" "$LOG" || { echo "FATAL: schema read-back failed" >&2; exit 1; }

gcloud storage cp "$H5" "$BUCKET/datasets/pointmaze.h5"
echo "[upload] pointmaze.h5" | tee -a "$LOG"

# ---- lance, the format every other task trained on (JPEG q95, writer default) ----
OUT="$DS/pointmaze.lance"
# same discipline for lance: a partial directory must not pass as finished
lance_complete() {
  [ -d "$OUT" ] || return 1
  python - "$OUT" <<'PYCHK' >/dev/null 2>&1
import sys

import stable_worldmodel as swm

ds = swm.data.load_dataset(sys.argv[1])
assert len(ds) == 200000, len(ds)
PYCHK
}
if lance_complete; then
  echo "[convert] lance already complete, reusing" | tee -a "$LOG"
else
  [ -d "$OUT" ] && { echo "[convert] discarding incomplete lance" | tee -a "$LOG"; rm -rf "$OUT"; }
  echo "[convert] h5 -> lance" | tee -a "$LOG"
  time python - "$H5" "$OUT" <<'PY' 2>&1 | tee -a "$LOG"
import sys

import hdf5plugin  # noqa: F401

from stable_worldmodel.data import convert

convert(sys.argv[1], sys.argv[2], dest_format="lance")
PY
fi
du -sh "$OUT" | tee -a "$LOG"
gcloud storage rsync -r "$OUT" "$BUCKET/datasets/pointmaze.lance"
gcloud storage cp "$LOG" "$BUCKET/datasets/" || true
echo "POINTMAZE PREP DONE"
