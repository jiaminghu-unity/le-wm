"""budget_sweep for PointMaze. Same arguments, same output, same code path.

A wrapper, not a fork: budget_sweep.py is imported and its main() called unchanged, so
the episode protocol, tiers, seeding and CSV schema are literally the same code. The only
intervention is registering the pointmaze preset (and its gymnasium env) into
budget_sweep.ENV_PRESETS before argparse builds its --env choices from that dict.

    usage: PMENV_DIR=/path/to/vendored/env budget_sweep_pointmaze.py --env pointmaze ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import budget_sweep  # noqa: E402
from scripts.pointmaze_preset import register  # noqa: E402

if __name__ == "__main__":
    register(budget_sweep.ENV_PRESETS)
    budget_sweep.main()
