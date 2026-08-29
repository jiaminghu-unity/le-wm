"""Pixel-model budget_sweep for the self-collected OGB multi-object tasks
(cube_double / scene): registers the ogbmulti presets + the lance dataset
dispatch, then runs the stock sweep. The q-input counterpart re-blesses QJEPA in
budget_sweep_qinput_any.py; pixel JEPA models need no wrapper.

    usage: budget_sweep_ogbmulti.py --env {cube_double|scene} [budget_sweep args]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import budget_sweep, ogbmulti_preset  # noqa: E402


def main():
    env = sys.argv[sys.argv.index("--env") + 1]
    assert env in ("cube_double", "scene"), env
    ogbmulti_preset.register(budget_sweep.ENV_PRESETS)
    ogbmulti_preset.install_lance_dispatch(budget_sweep)
    budget_sweep.main()


if __name__ == "__main__":
    main()
