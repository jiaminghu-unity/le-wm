"""Reduced-q ablation: retrain obj and aux end to end with roughly half of q withheld.

A standalone entry point. It changes nothing on disk: utils.py, train.py, every
existing config and every existing checkpoint are untouched. The single thing it does
before delegating to train.run is to merge the new variants from q_half into the
in-process copy of utils.Q_VARIANTS -- new dictionary keys only, so no existing
variant's builder, sources or unit check can be affected, and the existing
<dataset>.q_stats.<variant>.json artifacts are never read or rewritten (each new
variant gets its own file under its own name).

Everything else -- the forward pass, the losses, the dataset pipeline, the optimiser,
the logging -- is literally train.py's, so the reduced-q runs differ from the original
obj/aux runs in exactly one respect: which components of q the loss is handed.

    usage: python train_half.py experiment=hq_obj_cube

The baseline arm is deliberately absent: it never consumes q, so halving q cannot
change it and the existing baseline checkpoints remain the right comparison.
"""

import sys

import hydra

import q_half
import utils

# --- the only mutation, and it is in-process: add keys, overwrite nothing ---
_clash = set(q_half.Q_VARIANTS_HALF) & set(utils.Q_VARIANTS)
if _clash:
    raise RuntimeError(
        f"reduced-q variant names collide with existing ones: {sorted(_clash)}. "
        f"Refusing to shadow a variant the original experiments were trained on."
    )
utils.Q_VARIANTS.update(q_half.Q_VARIANTS_HALF)
print(f"[q_half] registered reduced-q variants: {sorted(q_half.Q_VARIANTS_HALF)}",
      flush=True)

import train  # noqa: E402  -- imported after the registry update, by design


def _guard_argv():
    """This entry point exists only for the reduced-q variants. Running a full-q
    experiment through it would train a duplicate of a model that already exists and
    quietly overwrite its checkpoint directory, so require an hq_* experiment."""
    exp = next((a.split("=", 1)[1] for a in sys.argv[1:]
                if a.startswith("experiment=")), None)
    if exp is None or not exp.startswith("hq_"):
        raise SystemExit(
            f"train_half.py takes a reduced-q experiment (experiment=hq_*), got "
            f"{exp!r}. Full-q runs belong to train.py -- using this file for them "
            f"would retrain an existing model over its own checkpoint."
        )


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    """Own hydra entry, then train.py's body verbatim.

    train.run cannot be called as a plain function from here: it is @hydra.main
    decorated, and hydra resolves config_path relative to the module the decorated
    function belongs to, so a cross-module call turns "./config/train" into the module
    path "..config.train" and dies. Declaring the decorator in THIS file makes the
    path resolve exactly as it does when train.py is run directly -- same directory,
    same config, same composition.

    hydra.main uses functools.wraps, so __wrapped__ is train.run's undecorated body;
    the fallback passes cfg through the wrapper, which hydra also honours (a non-None
    argument means "skip initialisation, just run the task").
    """
    body = getattr(train.run, "__wrapped__", train.run)
    return body(cfg)


if __name__ == "__main__":
    _guard_argv()
    run()
