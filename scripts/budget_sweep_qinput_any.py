"""budget_sweep for q-input (QJEPA) models on REACHER and CUBE: planning consumes
env state keys, not pixels. Companion to budget_sweep_qinput.py (pusht-only).

Per task the eval-path q is built from the same sources the training q_variant used:
  reacher : q = build_q_reacher_joints(infos['qpos'][..., :2])          (4-d)
  cube    : q = build_q_cube_effector(proprio_effector_pos,
                 proprio_effector_yaw, proprio_gripper_opening,
                 privileged_block_0_pos)                                 (9-d)
normalized with the checkpoint's own q_mean/q_std buffers. Implementation is a
re-bless of the loaded model's class overriding encode's eval path only ("q" in
info still takes the training path); JEPA.get_cost's goal_* -> bare-name remap
feeds the goal side automatically.

Naming caveat handled here: env INFOS use slash-namespaced keys (e.g.
'privileged/block_0_pos') while DATASET goal columns use underscores
('goal_privileged_block_0_pos' -> remapped to 'privileged_block_0_pos'), so every
key is resolved against both spellings. Cube additionally needs the effector
columns loaded so _extract_init_goal produces their goal_* twins: the preset is
FORKED here with an extended keys_to_load (budget_sweep's own dict is untouched).

    usage: budget_sweep_qinput_any.py --env {reacher|cube} [budget_sweep args]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from einops import rearrange  # noqa: E402

import stable_worldmodel as swm  # noqa: E402

from scripts import budget_sweep  # noqa: E402
from qjepa import QJEPA  # noqa: E402
from utils import build_q_cube_effector, build_q_reacher_joints  # noqa: E402


def _get(info, names):
    for n in names:
        if n in info:
            return info[n].float()
    raise KeyError(f"none of {names} in info; available: {sorted(info.keys())}")


def _q_reacher(info):
    return build_q_reacher_joints(_get(info, ["qpos"])[..., :2])


def _q_cube(info):
    eff = _get(info, ["proprio/effector_pos", "proprio_effector_pos"])
    yaw = _get(info, ["proprio/effector_yaw", "proprio_effector_yaw"])
    grip = _get(info, ["proprio/gripper_opening", "proprio_gripper_opening"])
    blk = _get(info, ["privileged/block_0_pos", "privileged_block_0_pos"])
    return build_q_cube_effector(eff, yaw, grip, blk)


Q_BUILDERS = {"reacher": _q_reacher, "cube": _q_cube}


def main():
    env = sys.argv[sys.argv.index("--env") + 1]
    assert env in Q_BUILDERS, f"--env must be one of {sorted(Q_BUILDERS)}"
    build_q = Q_BUILDERS[env]

    if env == "cube":
        preset = dict(budget_sweep.ENV_PRESETS["cube"])
        preset["keys_to_load"] = list(preset["keys_to_load"]) + [
            "proprio_effector_pos", "proprio_effector_yaw", "proprio_gripper_opening",
        ]
        budget_sweep.ENV_PRESETS = dict(budget_sweep.ENV_PRESETS)
        budget_sweep.ENV_PRESETS["cube"] = preset
        print("[qinput-any] cube preset forked: effector columns added to keys_to_load", flush=True)

    _orig_load = swm.wm.utils.load_pretrained

    def load_and_bless(ckpt, *a, **kw):
        model = _orig_load(ckpt, *a, **kw)
        assert isinstance(model, QJEPA), f"expected QJEPA, got {type(model)}"
        assert not bool((model.q_std == 1).all()), "q_std buffer untrained"
        logged = {"done": False}

        class QInputEval(type(model)):
            def encode(self, info):
                if "q" in info:
                    return super().encode(info)
                if not logged["done"]:
                    print(f"[qinput-any] encode keys: {sorted(k for k in info if not k.startswith('goal'))}",
                          flush=True)
                    logged["done"] = True
                x = build_q(info)
                x = (x - self.q_mean) / self.q_std
                b = x.size(0)
                flat = rearrange(x, "b t ... -> (b t) ...")
                emb = self.projector(self.encoder(flat))
                info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
                if "action" in info:
                    info["act_emb"] = self.action_encoder(info["action"])
                return info

        model.__class__ = QInputEval
        print(f"[qinput-any] QJEPA re-blessed for {env}; q_mean[:2]={model.q_mean[:2].tolist()}", flush=True)
        return model

    swm.wm.utils.load_pretrained = load_and_bless
    budget_sweep.main()


if __name__ == "__main__":
    main()
