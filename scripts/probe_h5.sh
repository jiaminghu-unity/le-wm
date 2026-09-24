#!/usr/bin/env bash
# Print the column schema of a dataset h5; self-sufficient on cold workers
# (falls back to system python + pip h5py, fetches the h5 from GCS if not cached).
#   usage: probe_h5.sh <filename.h5> [gcs-uri]
set -euo pipefail
NAME="${1:?h5 name}"; SRC="${2:-}"
source /mnt/disks/ssd0/.venv/bin/activate 2>/dev/null || {
  python3 -m pip install -q --user h5py hdf5plugin 2>/dev/null || pip install -q h5py hdf5plugin
}
H5=$(find /mnt/disks/ssd0 /tmp -name "$NAME" 2>/dev/null | head -1 || true)
if [ -z "$H5" ] && [ -n "$SRC" ]; then
  H5="/tmp/$NAME"; gcloud storage cp "$SRC" "$H5"
fi
[ -n "$H5" ] || { echo "NO_H5"; exit 0; }
python3 - "$H5" <<'PY' 2>/dev/null || python - "$H5" <<'PY2'
import sys, h5py, hdf5plugin
with h5py.File(sys.argv[1], "r") as f:
    for k in f: print(k, f[k].shape, f[k].dtype)
PY
import sys, h5py, hdf5plugin
with h5py.File(sys.argv[1], "r") as f:
    for k in f: print(k, f[k].shape, f[k].dtype)
PY2
