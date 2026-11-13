#!/usr/bin/env python3
"""Last progress line of every RUNNING rebuttal training (from `ray job logs`, no ssh).
A training that has run for a while with no `[Epoch x/10] step` line, or whose log ends in a
shared-memory error, is hung (see launchers' RemoveIPC note) and should be `ray job stop`ped."""
import json, re, subprocess, time, urllib.request
d = json.load(urllib.request.urlopen("http://127.0.0.1:8265/api/jobs/", timeout=30))
now = time.time() * 1000
for j in d:
    e = j.get("entrypoint") or ""
    if j["status"] == "RUNNING" and "ray_train_" in e and ("102" in e or "ray_train_qnative" in e):
        out = subprocess.run(["ray", "job", "logs", j["submission_id"]], capture_output=True, text=True, timeout=180).stdout.replace("\r", "\n")
        prog = [l for l in out.splitlines() if re.search(r"\[Epoch \d+/10\] step", l)]
        shm = "unable to open shared memory" in out or "sem_unlink" in out
        m = re.search(r"output_model_name=(\S+)", e)
        name = m.group(1) if m else re.sub(r"^.*bash scripts/ray_train_\w+\.sh ", "", e)[:44]
        age = (now - j["start_time"]) / 3.6e6
        flag = " <-- HUNG? (shared-memory error in log)" if shm else (" <-- no progress line" if not prog and age > 1.0 else "")
        print(f"  {age:5.1f}h {name:44s} {prog[-1].strip()[:42] if prog else '(setting up)'}{flag}  [{j['submission_id']}]")
