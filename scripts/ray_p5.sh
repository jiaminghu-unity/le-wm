#!/usr/bin/env bash
# P5 ranking-noise analysis (no physical channel).
#   usage: ray_p5.sh <task> [--starts N --cands N]
# Renders live frames, so it needs the same working GL backend and the same
# fidelity gate as the evaluation sweep: z_true comes from rendered pixels, and a
# software-rendering fallback would corrupt the geometry channel it is measuring.
set -euo pipefail

TASK="${1:?task}"; shift || true
EXTRA=("$@")
BUCKET=gs://prism-training-us/le-wm

case "$TASK" in
  pusht)   H5NAME=pusht_expert_train.h5; SRC="$BUCKET/datasets/pusht_expert_train.h5"; SUB=""
           MODELS=("base:lewm_c1_s3072/weights_epoch_10.pt"
                   "obj:lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt"
                   "aux:lewm_c5_qhead0.3_s3072/weights_epoch_10.pt"
                   "combo:lewm_c6_o01a03_s3072/weights_epoch_10.pt") ;;
  reacher) H5NAME=reacher.h5; SRC="$BUCKET/datasets/reacher.h5"; SUB=""
           MODELS=("base:lewm_r1_reacher_s3072/weights_epoch_10.pt"
                   "obj:lewm_r2_reacher_paep_l015_s3072/weights_epoch_10.pt"
                   "aux:lewm_r5_qhead0.4_s3072/weights_epoch_10.pt") ;;
  cube)    H5NAME=cube_single_expert.h5; SRC="$BUCKET/datasets/ogbench/cube_single_expert.tar.zst"; SUB="ogbench"
           MODELS=("base:lewm_k1_cube_s3072/weights_epoch_10.pt"
                   "obj:lewm_k2_cube_obj_eff0.1_s3072/weights_epoch_10.pt"
                   "aux:lewm_k4_cube_qhead_eff0.1_s3072/weights_epoch_10.pt"
                   "combo:lewm_k6_cube_combo_o0.1a0.1_s3072/weights_epoch_10.pt") ;;
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
DS="$STABLEWM_HOME/datasets${SUB:+/$SUB}"
mkdir -p "$DS"

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

H5="$DS/$H5NAME"
if [ ! -f "$H5" ]; then
  if [ "$TASK" = cube ]; then
    time gcloud storage cat "$SRC" | zstd -dc --long=31 | tar -xf - -C "$DS"
  else
    time gcloud storage cp "$SRC" "$H5"
  fi
fi

# FROZEN=1 swaps in the frozen-encoder ablation's models. Their encoder+projector are
# bit-identical to the originals, so z_true, z_goal and the pairwise-distance scale are
# literally the same numbers -- which is what makes rollerr comparable in absolute
# terms between a frozen run and its co-trained counterpart.
CKROOT="$BUCKET/ckpts"
if [ "${FROZEN:-0}" = "1" ]; then
  CKROOT="$BUCKET/ckpts_frozen"
  MODELS=("base:lewm_fz_base_${TASK}_s3072/weights_epoch_10.pt"
          "obj:lewm_fz_obj_${TASK}_s3072/weights_epoch_10.pt"
          "aux:lewm_fz_aux_${TASK}_s3072/weights_epoch_10.pt")
fi
# HALF=1 adds the reduced-q models ALONGSIDE the full-q ones, five models in one job.
# They must share a job: the env rollouts (the expensive part) are model-independent, so
# every model is scored on the SAME candidate actions from the SAME starts, making
# obj vs obj_h a paired comparison per start rather than two runs compared by their means.
if [ "${HALF:-0}" = "1" ]; then
  MODELS+=("obj_h:lewm_hq_obj_${TASK}_s3072/weights_epoch_10.pt"
           "aux_h:lewm_hq_aux_${TASK}_s3072/weights_epoch_10.pt")
fi
for spec in "${MODELS[@]}"; do
  d="${spec#*:}"; d="${d%%/*}"
  # the reduced-q checkpoints live under their own prefix, so the root is inferred from
  # the name rather than being one global value (no existing run is named lewm_hq_*)
  root="$CKROOT"
  case "$d" in lewm_hq_*) root="$BUCKET/ckpts_half" ;; esac
  mkdir -p "$STABLEWM_HOME/checkpoints/$d"
  gcloud storage cp "$root/$d/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$d/"
  gcloud storage cp "$root/$d/config.json" "$STABLEWM_HOME/checkpoints/$d/" || true
done

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
# The gate writes into the SAME log as the results. It used to run before the tee, so
# its output only reached the Ray job log while the uploaded .log carried the numbers —
# which is how a FAILED cube gate sat unnoticed under a table of results.
# And it is FATAL here, not a warning: z_true is the encoding of a rendered frame, so a
# renderer that disagrees with the training data does not bias these numbers, it
# invalidates them.
LOG="$SSD/p5_${TASK}${P5TAG:-}.log"
: > "$LOG"
if [ "$TASK" != pusht ]; then
  if ! python scripts/check_render_fidelity.py "$TASK" 8 --max-mae 3.0 2>&1 | tee -a "$LOG"; then
    echo "FATAL: render fidelity gate failed — refusing to produce rollerr/tau on frames \
the encoder was not trained on" | tee -a "$LOG"
    gcloud storage cp "$LOG" "$BUCKET/eval/" || true
    exit 1
  fi
fi

python scripts/p5_rank_noise.py "$TASK" "${MODELS[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
for f in eval_results/p5_"$TASK"*.json; do [ -f "$f" ] && gcloud storage cp "$f" "$BUCKET/eval/" || true; done
gcloud storage cp "$LOG" "$BUCKET/eval/"
echo "P5 DONE $TASK"
