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

ep.build_model = _build_model
# some versions import build_model into main's namespace at module import time;
# patch any alias defensively
for name in dir(ep):
    if name == "build_model":
        setattr(ep, name, _build_model)

if __name__ == "__main__":
    ep.main()
