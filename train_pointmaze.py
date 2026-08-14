"""PointMaze training: three arms, end to end.

A standalone entry point. Nothing on disk changes: utils.py, train.py, lewm.yaml and every
existing config and checkpoint are untouched. The single intervention is merging
q_pointmaze's variant into the in-process copy of utils.Q_VARIANTS -- a new key, nothing
overwritten -- before delegating to train.py's body.

    usage: python train_pointmaze.py experiment=p2_pointmaze_obj

The hydra decorator is redeclared here rather than calling train.run directly: train.run is
@hydra.main decorated and hydra resolves config_path relative to the module the decorated
function belongs to, so a cross-module call turns "./config/train" into the module path
"..config.train" and dies. Declaring it here makes the path resolve exactly as when train.py
runs directly. Same fix train_half.py and train_tworoom.py carry.
"""

import sys

import hydra

import q_pointmaze
import utils

_clash = set(q_pointmaze.Q_VARIANTS_POINTMAZE) & set(utils.Q_VARIANTS)
if _clash:
    raise RuntimeError(
        f"pointmaze variant names collide with existing ones: {sorted(_clash)}. "
        f"Refusing to shadow a variant the earlier experiments were trained on."
    )
utils.Q_VARIANTS.update(q_pointmaze.Q_VARIANTS_POINTMAZE)
print(f"[q_pointmaze] registered: {sorted(q_pointmaze.Q_VARIANTS_POINTMAZE)}", flush=True)

import train  # noqa: E402  -- after the registry update, by design


def _guard_argv():
    exp = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("experiment=")), None)
    if exp is None or not exp.startswith("p"):
        raise SystemExit(
            f"train_pointmaze.py takes a pointmaze experiment (experiment=p*), got {exp!r}."
        )


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    body = getattr(train.run, "__wrapped__", train.run)
    return body(cfg)


if __name__ == "__main__":
    _guard_argv()
    run()
