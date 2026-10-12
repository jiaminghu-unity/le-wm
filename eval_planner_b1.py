"""Official ContactWorld evaluator with our multi-view model spliced in.
Monkey-patches eval_planner.build_model; everything else (env, CEM, metrics,
success definition, action penalties) is the untouched official code path.

  python eval_planner_b1.py --variant b1 --ckpt-path outputs/b1/ckpt_100k.pt \
      --data-root data/demo_data/insertion_usb --isaacgym-cfg-name isaacgym_config_usb \
      --output-dir logs/planning/b1/step_12 --vision-key front --vision-type image \
      --use-tactile --tactile-key wrist --image-size 224 --history-size 3 \
      --horizon 6 --candidates 100 --topk 8 --iterations 4 --num-envs 100 \
      --goal-offset-steps 12 --max-steps 15 --stop-on-success --reg-loss-type vc
"""
import sys

VARIANT = "b1"
if "--variant" in sys.argv:
    i = sys.argv.index("--variant")
    VARIANT = sys.argv[i + 1]
    del sys.argv[i:i + 2]

import eval_planner as ep
from touchwm.cw_eval_adapter import MVJepaWorldModel

_ckpt = sys.argv[sys.argv.index("--ckpt-path") + 1]

def _build_model(cfg, *a, **kw):
    if VARIANT == "dino":
        from touchwm.dino_eval_adapter import DinoWorldModel
        return DinoWorldModel(_ckpt)
    return MVJepaWorldModel(_ckpt, variant=VARIANT)

_WM = {}
def _build_model_tracked(cfg, *a, **kw):
    m = _build_model(cfg, *a, **kw)
    _WM["m"] = m
    return m

# --- per-env trace: wrap the official per-step metrics fn, dump at exit (D0) ---
import atexit, hashlib, json as _json, os as _os
import numpy as _np
_PER_ENV = {"steps": []}
_orig_epm = ep.end_pose_metrics_joint
def _epm_tracked(*a, **kw):
    m = _orig_epm(*a, **kw)
    try:
        _PER_ENV["steps"].append({k: _np.asarray(v).copy() for k, v in m.items()})
    except Exception:
        pass
    return m
ep.end_pose_metrics_joint = _epm_tracked

def _dump_per_env():
    try:
        if not _PER_ENV["steps"] or "--output-dir" not in sys.argv:
            return
        out = sys.argv[sys.argv.index("--output-dir") + 1]
        n_env = max(_np.asarray(s["success"]).shape[0] for s in _PER_ENV["steps"])
        steps = [s for s in _PER_ENV["steps"] if _np.asarray(s["success"]).shape[0] == n_env]
        arrs = {k: _np.stack([_np.asarray(s[k]) for s in steps]) for k in steps[0]}
        _np.savez_compressed(_os.path.join(out, "per_env_trace.npz"), **arrs)
        meta = {k: _PER_ENV[k] for k in ("goal_hash", "init_hash") if k in _PER_ENV}
        if meta:
            _json.dump(meta, open(_os.path.join(out, "trace_meta.json"), "w"))
        print(f"[trace] per_env_trace.npz: {arrs['success'].shape}", flush=True)
    except Exception as e:
        print("[trace] dump failed:", e, flush=True)
atexit.register(_dump_per_env)

def _hash_info(info):
    h = hashlib.sha256()
    for k in sorted(info):
        v = info[k]
        try:
            h.update(_np.asarray(v.cpu() if hasattr(v, "cpu") else v).tobytes())
        except Exception:
            pass
    return h.hexdigest()[:16]

_orig_planner = ep.BatchedCEMPlanner
class _TrackedPlanner(_orig_planner):
    _n_plans = 0

    def plan(self, current_info, goal_info, prev_action=None):
        import torch as _t
        if _TrackedPlanner._n_plans == 0:
            _PER_ENV["goal_hash"] = _hash_info(goal_info)
            _PER_ENV["init_hash"] = _hash_info(current_info)
            print(f"[trace] init_hash={_PER_ENV['init_hash']} goal_hash={_PER_ENV['goal_hash']}", flush=True)
        wm = _WM.get("m")
        if wm is None:
            # 静默回退 = 本次事故根源;缺模型引用直接拒跑
            raise RuntimeError("[tracked] _WM empty: build_model patch not engaged, refusing to eval with zero-padded action history")
        H = getattr(wm, "history_size", None) or getattr(getattr(wm, "model", None), "history_size", 3)
        hist = getattr(wm, "_act_hist", None)
        targets = [wm] + ([wm.model] if hasattr(wm, "model") else [])
        if prev_action is None:
            for t_ in targets: t_._act_hist = None   # 回合开始:零填充回退
        else:
            pa = _t.as_tensor(prev_action).float()
            if pa.ndim == 1: pa = pa[None]
            pa = pa[:, None]               # (B,1,A)
            if hist is None:
                hist = _t.zeros(pa.shape[0], H, pa.shape[-1])
            newh = _t.cat([hist[:, 1:], pa], dim=1)
            for t_ in targets: t_._act_hist = newh
            if _TrackedPlanner._n_plans < 5:
                print(f"[tracked] plan#{_TrackedPlanner._n_plans} act_hist_norm={newh.norm().item():.4f}", flush=True)
        _TrackedPlanner._n_plans += 1
        return super().plan(current_info, goal_info, prev_action=prev_action)
ep.BatchedCEMPlanner = _TrackedPlanner

ep.build_model = _build_model_tracked

if __name__ == "__main__":
    ep.main()
