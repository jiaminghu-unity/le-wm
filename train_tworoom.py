"""Two-room training: registers the two-room q variant, then runs train.py unchanged.

Standalone entry point, same pattern as train_half.py. Nothing on disk changes:
utils.py, train.py, lewm.yaml and every existing config and checkpoint are untouched.
The single mutation is in-process -- utils.Q_VARIANTS gains one key, `tworoom_agent`,
and no existing variant's builder, sources or unit check is affected.

Everything else (forward pass, losses, dataloader, optimiser, logging) is literally
train.py's, so the two-room arms differ from the pusht/reacher/cube arms only in the
dataset and the q definition -- which is what makes the four tasks comparable.

    usage: python train_tworoom.py experiment=t2_tworoom_obj

Own hydra entry: train.run cannot be called as a plain function from here, because
hydra resolves config_path relative to the module the decorated function belongs to,
so a cross-module call turns "./config/train" into the module path "..config.train"
and dies (that is exactly how the reduced-q round first failed).
"""

import sys

import hydra

import q_tworoom
import utils

_clash = set(q_tworoom.Q_VARIANTS_TWOROOM) & set(utils.Q_VARIANTS)
if _clash:
    raise RuntimeError(
        f"two-room variant names collide with existing ones: {sorted(_clash)}. "
        f"Refusing to shadow a variant the other tasks were trained on."
    )
utils.Q_VARIANTS.update(q_tworoom.Q_VARIANTS_TWOROOM)
print(f"[q_tworoom] registered: {sorted(q_tworoom.Q_VARIANTS_TWOROOM)}", flush=True)

import train  # noqa: E402  -- after the registry update, by design


def _guard_argv():
    exp = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("experiment=")), None)
    if exp is None or not exp.startswith("t"):
        raise SystemExit(
            f"train_tworoom.py takes a two-room experiment (experiment=t*), got {exp!r}."
        )


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    body = getattr(train.run, "__wrapped__", train.run)
    return body(cfg)


if __name__ == "__main__":
    _guard_argv()
    run()
