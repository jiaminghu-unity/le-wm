#!/usr/bin/env bash
# q-input (QJEPA) eval for the self-collected OGBench multi-object tasks.
#   usage: ray_eval_qinput_ogbmulti.sh <task> <cfgname> <ckpt_dir> <solver> <seed> [seed...]
#     task: cube_double | scene
# Presets/dataset dispatch live in scripts/ogbmulti_preset.py (lance-only datasets);
# episode sets episodes_<task>_s<seed>_100.json must already be in eval_sets/
# (ray_gen_ogbmulti_sets.sh). Outputs to final_eval_ogbmulti/, all new files.
set -euo pipefail

TASK="${1:?task}"; CFG="${2:?cfgname}"; CKPT_DIR="${3:?ckpt dir}"; SOLVER="${4:?solver}"; shift 4
SEEDS=("$@")
BUCKET=gs://prism-training-us/le-wm
OUTP="$BUCKET/final_eval_ogbmulti"

case "$TASK" in
  cube_double) LANCE=cube_double_play.lance ;;
  scene)       LANCE=scene_play.lance ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets/ogbench"
mkdir -p "$DS" "$STABLEWM_HOME/checkpoints/$CKPT_DIR" "$SSD/eps"
echo "[env] $TASK/$CFG/$SOLVER on $(hostname), free=$(df -h --output=avail "$SSD"|tail -1|tr -d ' ')"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
  libegl1 libegl-mesa0 libgles2 libglvnd0 libopengl0 libosmesa6 libosmesa6-dev
sudo apt-get install -y -q libnvidia-gl-580-server || true
sudo usermod -aG render "$(id -un)" 2>/dev/null || true
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

# ---- dataset (lance dir rsync) ----
if [ ! -d "$DS/$LANCE" ]; then
  echo "[data] rsync $LANCE"
  time gcloud storage rsync -r "$BUCKET/datasets/ogbench/$LANCE" "$DS/$LANCE"
fi

# ---- checkpoint ----
gcloud storage cp "$BUCKET/ckpts/$CKPT_DIR/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$CKPT_DIR/"
gcloud storage cp "$BUCKET/ckpts/$CKPT_DIR/config.json" "$STABLEWM_HOME/checkpoints/$CKPT_DIR/" || true

# ---- mujoco GL ----
GL=egl
for g in egl osmesa; do
  if MUJOCO_GL=$g PYOPENGL_PLATFORM=$g python - <<'PY' >/dev/null 2>&1
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); assert r.render().shape == (64, 64, 3)
PY
  then GL=$g; break; fi
done
export MUJOCO_GL="$GL" PYOPENGL_PLATFORM="$GL"
echo "[gl] $GL"

# ---- preflight gate, FATAL: dataset + World + reset callables must round-trip ----
python - "$TASK" <<'PY'
import sys
sys.path.insert(0, ".")
task = sys.argv[1]
from scripts import budget_sweep, ogbmulti_preset
ogbmulti_preset.register(budget_sweep.ENV_PRESETS)
ogbmulti_preset.install_lance_dispatch(budget_sweep)
import stable_worldmodel as swm
from pathlib import Path
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal
p = budget_sweep.ENV_PRESETS[task]
ds = swm.data.HDF5Dataset(p["dataset"], keys_to_cache=p["process_cols"],
                          cache_dir=Path(swm.data.utils.get_cache_dir()),
                          keys_to_load=p["keys_to_load"])
init, goal, _ = _extract_init_goal(ds, [0], [10], 25)
world = swm.World(env_name=p["env_name"], num_envs=1, image_shape=(224, 224),
                  max_episode_steps=100, **p["env_kwargs"])
world.reset(seed=[30000])
_apply_callables(world.envs, p["callables"], init, goal)
import numpy as np
obs, r, term, trunc, info = world.step(np.zeros_like(world.action_space.sample()))
keys = sorted(k for k in (info[0] if isinstance(info, (list, tuple)) else info) if not str(k).startswith("goal"))
print("[gate] step ok; info keys:", keys[:24], flush=True)
print("[gate] PASS", flush=True)
PY

RC=0
for S in "${SEEDS[@]}"; do
  EPSNAME="episodes_${TASK}_s${S}_100.json"
  OUT="final_${TASK}_${CFG}_${SOLVER}_s${S}.csv"
  if gcloud storage ls "$OUTP/$OUT" >/dev/null 2>&1; then
    echo "[skip] $OUT already in GCS"; continue
  fi
  gcloud storage cp "$BUCKET/eval_sets/$EPSNAME" "$SSD/eps/"
  echo "[run] $TASK $CFG $SOLVER seed=$S"
  set +e
  python scripts/budget_sweep_qinput_any.py \
    --env "$TASK" --solver "$SOLVER" \
    --config "$CFG" "$CKPT_DIR/weights_epoch_10.pt" \
    --tiers T1 T2 T3 T4 T5 \
    --episodes-json "$SSD/eps/$EPSNAME" \
    --out "$SSD/$OUT" 2>&1 | tail -40
  rc=${PIPESTATUS[0]}; set -e
  if [ "$rc" -ne 0 ]; then RC=$rc
  elif [ -f "$SSD/$OUT" ]; then
    # 只有跑成功且非空才上传:空表头文件曾把 done-check 堵死(pointmaze 教训)
    [ "$(wc -c < "$SSD/$OUT")" -gt 1000 ] && gcloud storage cp "$SSD/$OUT" "$OUTP/$OUT" \
      || echo "[warn] $OUT too small, NOT uploaded"
  fi
done
echo "[done] rc=$RC -> $OUTP/final_${TASK}_${CFG}_${SOLVER}_s*.csv"
exit $RC
