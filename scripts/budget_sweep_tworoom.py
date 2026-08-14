"""budget_sweep for two-room. Same arguments, same output, same code path.

A wrapper, not a fork: budget_sweep.py is imported and its main() called unchanged, so the
episode protocol, tiers, seeding, render gate and CSV schema are literally the same code.
The only intervention is registering the two-room preset into budget_sweep.ENV_PRESETS
before argparse builds its --env choices from that dict.

    usage: budget_sweep_tworoom.py --env tworoom --solver cem --config t1 <ckpt> ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import budget_sweep  # noqa: E402
from scripts.tworoom_preset import register  # noqa: E402

if __name__ == "__main__":
    register(budget_sweep.ENV_PRESETS)
    budget_sweep.main()
