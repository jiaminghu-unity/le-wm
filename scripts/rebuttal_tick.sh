#!/usr/bin/env bash
# One monitoring pass for the rebuttal round: status, chain liveness, mirror results and
# diagnostics, submit missing diagnostics, refresh tables, keep the checkpoint mirror going,
# commit and push. Read-only towards the workers (never ssh into them -- see launchers' RemoveIPC note).
export PATH=/usr/local/bin:/snap/bin:/usr/bin:/bin:$PATH
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /home/jiaming.hu/le-wm || exit 1
bash scripts/rebuttal_status.sh "${1:-55}" 2>&1 | grep -v -E "^grab log|have=|stopped on user|^chain log|^\[11-" | head -14
for c in all_seeds1024 recov_seeds1024 hl2_pilot hl2_seeds1024; do
  a=$(pgrep -f "^flock -n -o /home/jiaming.hu/.rebuttal_$c.lock" >/dev/null && echo alive || echo "not running")
  d=$([ -f eval_results/$c.DONE ] && echo DONE || echo "")
  echo "$c: $a $d | $(grep -E 'status:|COMPLETE' eval_results/$c.log | tail -1 | cut -c17-90)"
done
echo "running trainings:"; python3 scripts/rebuttal_progress.py
B=gs://prism-training-us/le-wm; M=eval_results/rebuttal_mirror
for d in final_eval final_eval_tworoom final_eval_pointmaze; do timeout 300 gcloud storage cp -n "$B/$d/final_*r102[45]_*.csv" $M/$d/ >/dev/null 2>&1; done
timeout 120 gcloud storage cp -n "$B/eval/rebuttal_diag/*.json" $M/diag/ >/dev/null 2>&1
echo "mirrored CSVs: $(find $M -name '*.csv' | wc -l)  diag JSONs: $(ls $M/diag | wc -l)"
bash scripts/run_rebuttal_diag.sh 2>&1 | tail -5
python3 scripts/rebuttal_summary.py | sed -n '6p'; python3 scripts/rebuttal_diag_summary.py >/dev/null; python3 scripts/rebuttal_extra_tables.py >/dev/null
echo "extra 40GB VMs running: $(timeout 90 gcloud compute instances list --filter='labels.purpose=scale-rebuttal AND status=RUNNING' --format='value(name)' 2>/dev/null | wc -l)"
pgrep -f "push_rebuttal_ckpts" >/dev/null || { setsid nohup bash scripts/push_rebuttal_ckpts.sh >> eval_results/ckpt_push.out 2>&1 < /dev/null & disown; }
echo "ckpt mirror: $(grep -E 'pushed|done:|failed' eval_results/ckpt_push.out | tail -1 | cut -c1-110) | on branch: $(cd /home/jiaming.hu/le-wm-ckpts 2>/dev/null && git ls-files | grep -c weights_epoch_10.pt)"
git add eval_results/rebuttal_mirror eval_results/*.log scripts 2>/dev/null
git commit -q -m "Rebuttal: refresh result mirror, diagnostics and chain logs

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>" 2>/dev/null
timeout 300 git push -q origin rebuttal 2>&1 | tail -1
echo "pushed: $(git log --oneline -1 | cut -c1-70)"
