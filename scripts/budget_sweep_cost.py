"""budget_sweep with the planner cost selected by --cost. Same arguments otherwise.

A wrapper, not a fork: budget_sweep.py is imported and its main() called unchanged, so
the episode protocol, tiers, seeding, render gate and CSV schema are literally the same
code. The single intervention is on swm.wm.utils.load_pretrained -- the one place
budget_sweep obtains a model (budget_sweep.py:270) -- which re-blesses the loaded
instance into the chosen cost subclass. Weights, config and every other method are
untouched.

    usage: budget_sweep_cost.py --cost dot [budget_sweep.py's own arguments]

--cost is consumed here and removed from argv before budget_sweep sees it, so
budget_sweep's parser stays exactly as it is.

Only cem and icem are accepted: mppi's fixed softmax temperature and gd's fixed
learning rate both react to the cost magnitude, and dot/cos change magnitude far
more than L1 did. See cost_variants.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402

from scripts import budget_sweep  # noqa: E402
from scripts.cost_variants import COSTS, make_cost_class, verify  # noqa: E402

_ALLOWED_SOLVERS = {"cem", "icem"}


def _take_cost():
    """Pull --cost out of argv so budget_sweep's parser never sees it."""
    argv = sys.argv[1:]
    if "--cost" not in argv:
        raise SystemExit(f"--cost is required; one of {sorted(COSTS)}")
    i = argv.index("--cost")
    try:
        name = argv[i + 1]
    except IndexError:
        raise SystemExit("--cost needs a value")
    if name not in COSTS:
        raise SystemExit(f"unknown cost {name!r}; have {sorted(COSTS)}")
    del argv[i:i + 2]
    slv = argv[argv.index("--solver") + 1] if "--solver" in argv else "cem"
    if slv not in _ALLOWED_SOLVERS:
        raise SystemExit(
            f"solver {slv!r} is not valid for a cost comparison. Only "
            f"{sorted(_ALLOWED_SOLVERS)} rank by cost alone; mppi rescales through a "
            f"fixed softmax temperature and gd descends the cost gradient at a fixed "
            f"lr, so both would change for reasons other than the cost's shape."
        )
    sys.argv = [sys.argv[0]] + argv
    return name


_orig_load = swm.wm.utils.load_pretrained


def _make_loader(name):
    def _load(*args, **kwargs):
        model = _orig_load(*args, **kwargs)
        model.__class__ = make_cost_class(type(model), name)
        verify(model, name)
        return model
    return _load


if __name__ == "__main__":
    cost = _take_cost()
    swm.wm.utils.load_pretrained = _make_loader(cost)
    budget_sweep.main()
