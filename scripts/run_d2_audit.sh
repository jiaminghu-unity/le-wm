#!/bin/bash
# D2 离线路由审计(无 Isaac,d0venv),与评测队列并行;OOM 则整体重试一次
cd ~/ContactWorld
for V in ours b3; do
  CK=${V}_ckpt_best.pt
  ~/d0venv/bin/python d2_routing_audit.py --variant $V --ckpt-path $CK \
    >> ~/d2_audit.log 2>&1 && echo "$V D2 AUDIT OK" >> ~/d0d2_status.log || echo "$V D2 AUDIT FAIL" >> ~/d0d2_status.log
done
gcloud storage cp outputs/d2_routing_audit_*.json gs://prism-training-us/contactworld/outputs/ >/dev/null 2>&1
echo "D2 AUDIT COMPLETE" >> ~/d0d2_status.log
