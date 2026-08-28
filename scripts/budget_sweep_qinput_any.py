"""budget_sweep for q-input (QJEPA) models on reacher/cube/tworoom/pointmaze:
planning consumes env state keys, not pixels. The q builder is AUTO-SELECTED by
(env, q_dim-of-checkpoint), so subset-q and native-full-q variants of the same task
need no flags: reacher 4d joints; cube 9d effector / 22d full-config; tworoom 2d
agent pos; pointmaze 4d native state (x,y,vx,vy).

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


from q_cube_full import build_q_cube_full  # noqa: E402


def _q_cube_full(info):
    return build_q_cube_full(
        _get(info, ["proprio/effector_pos", "proprio_effector_pos"]),
        _get(info, ["proprio/effector_yaw", "proprio_effector_yaw"]),
        _get(info, ["proprio/gripper_opening", "proprio_gripper_opening"]),
        _get(info, ["proprio/gripper_contact", "proprio_gripper_contact"]),
        _get(info, ["proprio/joint_pos", "proprio_joint_pos"]),
        _get(info, ["privileged/block_0_pos", "privileged_block_0_pos"]),
        _get(info, ["privileged/block_0_yaw", "privileged_block_0_yaw"]),
    )


def _q_tworoom(info):
    return _get(info, ["proprio", "pos_agent"])[..., :2]


def _q_pointmaze(info):
    return _get(info, ["state"])[..., :4]


# (env, q_dim) -> builder;q_dim 从 ckpt 的 q_mean 读出,变体免旗标
Q_BUILDERS = {
    ("reacher", 4): _q_reacher,
    ("cube", 9): _q_cube,
    ("cube", 22): _q_cube_full,
    ("tworoom", 2): _q_tworoom,
    ("pointmaze", 4): _q_pointmaze,
}


def main():
    env = sys.argv[sys.argv.index("--env") + 1]
    envs_known = sorted({e for e, _ in Q_BUILDERS})
    assert env in envs_known, f"--env must be one of {envs_known}"

    if env == "tworoom":
        from scripts.tworoom_preset import register
        register(budget_sweep.ENV_PRESETS)
    elif env == "pointmaze":
        # 本文件顶部已 `from utils import ...`,sys.modules['utils'] 被 le-wm 占住;
        # pmenv 的 point_maze_wrapper 还要 `from utils import aggregate_dct`,给缓存
        # 模块注入垫片(与 ray_eval_pointmaze.sh 写进 pmenv/utils.py 的实现一致)。
        import utils as _lewm_utils
        if not hasattr(_lewm_utils, "aggregate_dct"):
            import numpy as _np

            def _aggregate_dct(dcts):
                if not dcts:
                    return {}
                if not isinstance(dcts[0], dict):
                    return _np.stack(dcts)
                return {k: _np.stack([d[k] for d in dcts]) for k in dcts[0]}

            _lewm_utils.aggregate_dct = _aggregate_dct
        from scripts.pointmaze_preset import register
        register(budget_sweep.ENV_PRESETS)

    # q 构建输入必须是 RAW 值:去掉这些列上的 StandardScaler
    _RAW = {"tworoom": ["proprio", "pos_agent"], "pointmaze": ["state", "pos"]}
    if env in _RAW:
        _orig_bp = budget_sweep.build_process

        def build_process_raw(dataset, cols, _orig=_orig_bp, _drop=_RAW[env]):
            process = _orig(dataset, cols)
            for k in _drop:
                process.pop(k, None); process.pop(f"goal_{k}", None)
            print(f"[qinput-any] raw cols for {env}: dropped scalers {_drop}", flush=True)
            return process

        budget_sweep.build_process = build_process_raw

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
        qdim = int(model.q_mean.numel())
        key = (env, qdim)
        assert key in Q_BUILDERS, f"no q builder for {key}; have {sorted(Q_BUILDERS)}"
        build_q = Q_BUILDERS[key]
        print(f"[qinput-any] builder = ({env}, {qdim}d)", flush=True)
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
