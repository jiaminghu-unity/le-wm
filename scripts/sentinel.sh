#!/usr/bin/env bash
# 哨兵:每 5 分钟全面体检所有在跑的工作负载;发现异常或里程碑达成就 exit
# (退出会触发通知,把 Claude 叫醒来处理)。健康时静默循环。
# usage: sentinel.sh <max_rounds>   (默认 96 轮 = 8 小时,到点也退出汇报一次)
B=gs://prism-training-us/le-wm
MAX=${1:-96}
LOGDIR=/workspace/le-wm/eval_results
snap_caps(){ grep -hc "attempt cap" "$LOGDIR"/ogbmulti_eval.log "$LOGDIR"/qinput_full.log \
  "$LOGDIR"/pointmaze_retest.log "$LOGDIR"/retest1024.log 2>/dev/null | paste -sd+ | bc; }
CAPS0=$(snap_caps)

for i in $(seq 1 "$MAX"); do
  sleep 300
  BAD=""
  # 1) ray dashboard 活着吗
  curl -sf -m 10 http://127.0.0.1:8265/api/jobs/ >/dev/null 2>&1 || BAD="ray dashboard 不响应"
  # 2) 预期存活的链子(有未完成工作的才算)
  if [ -z "$BAD" ]; then
    N=$(gcloud storage ls "$B/final_eval_ogbmulti/final_*_s10?.csv" 2>/dev/null | wc -l)
    if [ "$N" -lt 24 ] && ! ps -eo cmd | grep -q "[b]ash scripts/run_ogbmulti_eval_chain.sh"; then
      BAD="ogbmulti 评测链死了(csv=$N/24)"
    fi
    [ "$N" -ge 24 ] && { echo "MILESTONE: ogbmulti 24/24 全齐"; exit 0; }
  fi
  # 3) 新增 attempt cap(说明有格子连败触帽)
  if [ -z "$BAD" ]; then
    CAPS=$(snap_caps)
    [ "${CAPS:-0}" -gt "$(( ${CAPS0:-0} + 8 ))" ] && BAD="日志新增大量 attempt cap($CAPS0 -> $CAPS)"
  fi
  # 4) 最近 15 分钟内的 FAILED 任务
  if [ -z "$BAD" ]; then
    F=$(python3 -c "
import json,urllib.request,time
jobs=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/',timeout=20))
f=[j for j in jobs if j.get('status')=='FAILED' and (j.get('start_time') or 0)/1000>time.time()-900]
print(len(f), f[0]['submission_id'] if f else '')" 2>/dev/null)
    set -- $F
    [ "${1:-0}" -gt 0 ] && BAD="最近15分钟有 $1 个 FAILED 任务(如 ${2:-?})"
  fi
  if [ -n "$BAD" ]; then echo "ANOMALY: $BAD"; exit 1; fi
  echo "[$(date -u '+%H:%M')] healthy round $i: ogbmulti csv=$N/24, caps=$CAPS"
done
echo "SENTINEL TIMEOUT: $MAX 轮健康,例行汇报"
exit 0
