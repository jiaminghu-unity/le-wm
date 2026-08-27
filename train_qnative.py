"""train.py with the native-full q variants registered at runtime:
q_native_full (pusht 8-d incl. velocities) + q_cube_full (cube 22-d full config).
The reacher 6-d joints+finger variant already ships in utils.Q_VARIANTS.
utils.py / train.py are never modified.
"""

import q_cube_full
import q_native_full
import utils

for mod in (q_cube_full.Q_VARIANTS_CUBE_FULL, q_native_full.Q_VARIANTS_NATIVE):
    clash = set(mod) & set(utils.Q_VARIANTS)
    assert not clash, f"variant name collision: {sorted(clash)}"
    utils.Q_VARIANTS.update(mod)
print(f"[qnative] variants registered: "
      f"{sorted(set(q_cube_full.Q_VARIANTS_CUBE_FULL) | set(q_native_full.Q_VARIANTS_NATIVE))}", flush=True)

from train import run  # noqa: E402

if __name__ == "__main__":
    run()
