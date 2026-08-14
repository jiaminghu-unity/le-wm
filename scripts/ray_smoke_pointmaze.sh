#!/usr/bin/env bash
# PointMaze evaluation smoke: install the env stack, generate the episode sets, run the
# eval-path gate, then ONE cell.
#   usage: [GATE_ONLY=1] ray_smoke_pointmaze.sh
#
# Written fresh, not derived by string replacement from another task's script -- the
# tworoom->pointmaze derivation produced four silent-miss failures in one day.
#
# One cell only: the sweep is not started until a human has seen this number. What the
# earlier gates proved is that the env reproduces dataset pixels (raw env MAE 0.039, and
# mujoco_py coexists with the py3.10 eval stack); what remains unproven is the end-to-end
# path -- adapter + callables + planner -- and whether baseline SR lands somewhere
# plausible rather than 0% or 100%, either of which would mean protocol error, not model
# quality. The dataset is fully RANDOM trajectories (DINO-WM's description), so a low
# baseline would not by itself be alarming; a zero one would.
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
TS_REPO=https://raw.githubusercontent.com/agentic-learning-ai-lab/temporal-straightening/main
SEEDS="101 102 103 104 105 106"

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS" "$SSD/eps"
LOG="$SSD/smoke_pointmaze.log"; : > "$LOG"
echo "[env] $(hostname), free=$(df -h --output=avail "$SSD" | tail -1 | tr -d ' ')" | tee -a "$LOG"

# ---- system deps: the standard eval set PLUS the mujoco_py compile/render chain ----
sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd curl patchelf \
  libosmesa6-dev libglew-dev libgl1-mesa-dev libglfw3 \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1

if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  mkdir -p "$HOME/.mujoco"
  curl -fsSL https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz | tar -xz -C "$HOME/.mujoco"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

# ---- the standard py3.10 eval venv, plus mujoco-py (compatibility proven by
#      ray_test_pm310.sh: MAE 0.043 with numpy 2.2.6 in this exact stack) ----
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scikit-learn
uv pip install -q 'cython==0.29.37' 'gym==0.23.1' 'glfw==2.7.0'
uv pip install -q --no-build-isolation 'mujoco-py==2.1.2.14' 2>&1 | tail -3 | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import mujoco_py
print("[mujoco_py] OK:", mujoco_py.__version__)
PY
grep -q "\[mujoco_py\] OK" "$LOG" || { echo "FATAL: mujoco_py unusable" >&2; exit 1; }

# ---- vendored upstream env + shims (same files the gates validated) ----
export PMENV_DIR="$SSD/pmenv"
mkdir -p "$PMENV_DIR/env/pointmaze" "$PMENV_DIR/d4rl"
for f in __init__.py maze_model.py point_maze_wrapper.py dynamic_mjc.py; do
  [ -f "$PMENV_DIR/env/pointmaze/$f" ] || \
    curl -fsSL "$TS_REPO/env/pointmaze/$f" -o "$PMENV_DIR/env/pointmaze/$f"
done
touch "$PMENV_DIR/env/__init__.py" "$PMENV_DIR/d4rl/__init__.py"
cat > "$PMENV_DIR/utils.py" <<'PY'
def aggregate_dct(dcts):
    import numpy as np
    if not dcts:
        return {}
    if not isinstance(dcts[0], dict):
        return np.stack(dcts)
    return {k: np.stack([d[k] for d in dcts]) for k in dcts[0]}
PY
cat > "$PMENV_DIR/d4rl/offline_env.py" <<'PY'
class OfflineEnv:
    """Stand-in: MazeEnv only inherits this and forwards kwargs; get_dataset is never called."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
PY

H5="$DS/pointmaze.h5"
[ -f "$H5" ] || { echo "[data] fetching pointmaze.h5 (30 GB)"; time gcloud storage cp "$BUCKET/datasets/pointmaze.h5" "$H5"; }

# ---- episode sets, the same convention every other task uses ----
for S in $SEEDS; do
  OUT="$SSD/eps/episodes_pointmaze_s${S}_100.json"
  if gcloud storage ls "$BUCKET/eval_sets/episodes_pointmaze_s${S}_100.json" >/dev/null 2>&1; then
    gcloud storage cp "$BUCKET/eval_sets/episodes_pointmaze_s${S}_100.json" "$OUT"
  else
    python scripts/gen_episodes.py --num 100 --seed "$S" --dataset pointmaze \
      --env-seed-base $(( 40000 + (S - 101) * 10000 )) --out "$OUT" 2>&1 | tee -a "$LOG"
    gcloud storage cp "$OUT" "$BUCKET/eval_sets/"
  fi
done

# ---- the eval-path gate: World + callables + adapter, FATAL ----
echo "[gate] eval-path scene reconstruction" | tee -a "$LOG"
set +e
python scripts/check_render_pointmaze.py 12 \
  --episodes "$SSD/eps/episodes_pointmaze_s101_100.json" --max-mae 3.0 2>&1 | tee -a "$LOG"
grc=${PIPESTATUS[0]}
set -e
gcloud storage cp "$LOG" "$BUCKET/eval/smoke_pointmaze.log" || true
[ "$grc" = 0 ] || { echo "FATAL: eval-path gate failed; not running any cell" >&2; exit 1; }

if [ "${GATE_ONLY:-0}" = "1" ]; then
  echo "POINTMAZE GATE ONLY DONE"; exit 0
fi

# ---- one cell: baseline, cem, seed 101. Checkpoint name resolved from GCS by prefix. ----
CK=$(gcloud storage ls "$BUCKET/ckpts_pointmaze/" | sed 's|.*ckpts_pointmaze/||;s|/$||' \
     | grep -E "^lewm_p1_pointmaze" | head -1)
[ -n "$CK" ] || { echo "FATAL: no baseline checkpoint under ckpts_pointmaze/" >&2; exit 1; }
mkdir -p "$STABLEWM_HOME/checkpoints/$CK"
gcloud storage cp "$BUCKET/ckpts_pointmaze/$CK/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$CK/"
gcloud storage cp "$BUCKET/ckpts_pointmaze/$CK/config.json" "$STABLEWM_HOME/checkpoints/$CK/" || true

OUT="final_pointmaze_p1_cem_s101.csv"
echo "[run] pointmaze p1 cem seed=101" | tee -a "$LOG"
set +e
python scripts/budget_sweep_pointmaze.py \
  --env pointmaze --solver cem --config p1 "$CK/weights_epoch_10.pt" \
  --tiers T1 T2 T3 T4 T5 \
  --episodes-json "$SSD/eps/episodes_pointmaze_s101_100.json" \
  --out "$SSD/$OUT" > "$SSD/run_$OUT.log" 2>&1
rc=$?
set -e
grep -E "^\[preset\]|=== p1 @" "$SSD/run_$OUT.log" | tee -a "$LOG"
tail -20 "$SSD/run_$OUT.log" | tee -a "$LOG"
[ -f "$SSD/$OUT" ] && gcloud storage cp "$SSD/$OUT" "$BUCKET/final_eval_pointmaze/$OUT"
gcloud storage cp "$LOG" "$BUCKET/eval/smoke_pointmaze.log" || true
gcloud storage cp "$SSD/run_$OUT.log" "$BUCKET/final_eval_pointmaze/logs/" || true
echo "[done] rc=$rc"
exit $rc
