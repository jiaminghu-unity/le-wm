#!/usr/bin/env bash
# Mirror the rebuttal checkpoints (training seeds 1024/1025) from GCS to the GitHub branch
# rebuttal-ckpts, in the same layout as the existing model-ckpts branch:
#   <gcs prefix>/<run>/{config.json, weights_epoch_10.pt}
# Idempotent: only runs not yet in the branch are fetched; commits and pushes every 12 runs
# (~0.9 GB) to stay under GitHub's per-push limit. Uses its own small repo so the working tree
# the Ray jobs are packaged from is never touched.
export PATH=/usr/local/bin:/snap/bin:/usr/bin:/bin:$PATH
WT=/home/jiaming.hu/le-wm-ckpts; B=gs://prism-training-us/le-wm; BR=rebuttal-ckpts
URL=git@github.com:jiaminghu-unity/le-wm.git
exec 9>/home/jiaming.hu/.rebuttal_ckpt_push.lock; flock -n 9 || { echo "already running"; exit 0; }
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*"; }
if [ ! -d "$WT/.git" ]; then
  mkdir -p "$WT" && cd "$WT" && git init -q && git remote add origin "$URL"
  git config user.name "Jiaming Hu"; git config user.email "jiaming.hu@unity3d.com"
  git config core.sshCommand "ssh -i $HOME/.ssh/le_wm_deploy -o IdentitiesOnly=yes"
  if git ls-remote --exit-code origin "refs/heads/$BR" >/dev/null 2>&1; then
    git fetch -q --depth 1 origin "$BR" && git checkout -q -b "$BR" FETCH_HEAD
  else
    git checkout -q -b "$BR"
    cat > README.md <<'MD'
# SCALE rebuttal checkpoints

Models trained for the rebuttal round (training seeds 1024 and 1025), 10 epochs each.
Layout mirrors the GCS bucket and the `model-ckpts` branch: `<prefix>/<run>/weights_epoch_10.pt`
plus `config.json`. Code, configs and results are on the `rebuttal` branch
(`scripts/rebuttal_specs/*.spec` lists every run and the command that trained it;
`eval_results/rebuttal_mirror/SUMMARY.md` and `DIAG.md` hold the numbers).

- `ckpts/` Push-T, Reacher, Cube · `ckpts_tworoom/` Two-Room · `ckpts_pointmaze/` PointMaze
- `lewm_c1|r1_reacher|k1_cube|t1_tworoom|p1_pointmaze` LeWM · `c3|r2|k2|t2|p2` SCALE · `c5|r5|k4|t5|p5` Aux
- `lewm_lpwmr_<task>_{base,scale}` LpWM and LpWM+SCALE
- `*_hl2_<b>` / `*_hl2pilot_<b>` Aux with an L2 penalty of strength b on the readout
- `lewm_rq_<task>_{high,low}` SCALE on the high- or low-recoverability subset of the task state
MD
    git add README.md && git commit -q -m "rebuttal checkpoints: layout"
  fi
fi
cd "$WT" || exit 1
n=0; total=0
flush(){ git diff --cached --quiet && return 0
  git commit -q -m "rebuttal ckpts: $1 runs" && for i in 1 2 3; do timeout 1800 git push -q origin "$BR" && { log "pushed $1 runs"; return 0; }; log "push failed, retry $i"; sleep 20; done; return 1; }
for d in ckpts ckpts_tworoom ckpts_pointmaze; do
  for run in $(timeout 120 gcloud storage ls "$B/$d/*_s102[45]/weights_epoch_10.pt" 2>/dev/null | sed 's#.*/\([^/]*\)/weights_epoch_10.pt#\1#'); do
    [ -f "$d/$run/weights_epoch_10.pt" ] && continue
    mkdir -p "$d/$run"
    if timeout 600 gcloud storage cp -q "$B/$d/$run/weights_epoch_10.pt" "$d/$run/" 2>/dev/null; then
      timeout 120 gcloud storage cp -q "$B/$d/$run/config.json" "$d/$run/" 2>/dev/null || true
      git add "$d/$run"; n=$((n+1)); total=$((total+1))
      [ "$n" -ge 12 ] && { flush "$n" || exit 1; n=0; }
    else rm -rf "${d:?}/$run"; log "download failed: $run"; fi
  done
done
flush "$n"
log "done: $total new runs this pass; branch has $(git ls-files | grep -c weights_epoch_10.pt) checkpoints"
