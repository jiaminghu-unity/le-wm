"""train.py with extra q variants registered at runtime, for the q-only-input and
OGB multi-object experiments:

  q_native_full   pusht_state_native (8-d incl. velocities),
                  pointmaze_state_native (4-d x/y/vx/vy)
  q_cube_full     cube_full_config (22-d)
  q_tworoom       tworoom_agent (2-d)
  q_pointmaze     pointmaze_pos (2-d)
  q_ogb_multi     cube_double_full (27-d), scene_full (26-d)

utils.py / train.py are never modified. Hydra entry follows train_cube_full.py's
pattern: hydra resolves config_path relative to the module DECLARING @hydra.main,
so a cross-module call of train.run turns "./config/train" into module path
"..config.train" and dies -- the decorator must live in this file, with the body
delegated to train.run.__wrapped__.
"""

import hydra

import q_cube_full
import q_native_full
import q_ogb_multi
import q_pointmaze
import q_reacher_full
import q_tworoom
import utils

_ALL = (q_cube_full.Q_VARIANTS_CUBE_FULL, q_native_full.Q_VARIANTS_NATIVE,
        q_tworoom.Q_VARIANTS_TWOROOM, q_pointmaze.Q_VARIANTS_POINTMAZE,
        q_reacher_full.Q_VARIANTS_REACHER_FULL,
        q_ogb_multi.Q_VARIANTS_OGB_MULTI)
for mod in _ALL:
    clash = set(mod) & set(utils.Q_VARIANTS)
    assert not clash, f"variant name collision: {sorted(clash)}"
    utils.Q_VARIANTS.update(mod)
print(f"[qnative] variants registered: {sorted(k for m in _ALL for k in m)}", flush=True)

import train  # noqa: E402  -- imported after the registry update, by design


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    body = getattr(train.run, "__wrapped__", train.run)
    return body(cfg)


if __name__ == "__main__":
    run()
