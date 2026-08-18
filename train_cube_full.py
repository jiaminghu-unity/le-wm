"""Full-config-q ablation entry point: train the aux arm with the 22-d
cube_full_config q (see q_cube_full.py). Same isolation contract as train_half.py:
in-process registry merge only, nothing on disk changes, existing q_stats artifacts
are never touched (the new variant persists under its own name).

    usage: python train_cube_full.py experiment=aux21_cube
"""

import sys

import hydra

import q_cube_full
import utils

# --- the only mutation, and it is in-process: add keys, overwrite nothing ---
_clash = set(q_cube_full.Q_VARIANTS_CUBE_FULL) & set(utils.Q_VARIANTS)
if _clash:
    raise RuntimeError(
        f"full-config-q variant names collide with existing ones: {sorted(_clash)}. "
        f"Refusing to shadow a variant the original experiments were trained on."
    )
utils.Q_VARIANTS.update(q_cube_full.Q_VARIANTS_CUBE_FULL)
print(f"[q_cube_full] registered full-config-q variants: {sorted(q_cube_full.Q_VARIANTS_CUBE_FULL)}",
      flush=True)

import train  # noqa: E402  -- imported after the registry update, by design


def _guard_argv():
    """This entry point exists only for the full-config-q variants. Running a full-q
    experiment through it would train a duplicate of a model that already exists and
    quietly overwrite its checkpoint directory, so require an aux21_* experiment."""
    exp = next((a.split("=", 1)[1] for a in sys.argv[1:]
                if a.startswith("experiment=")), None)
    if exp is None or not exp.startswith("aux21_"):
        raise SystemExit(
            f"train_half.py takes a full-config-q experiment (experiment=aux21_*), got "
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
