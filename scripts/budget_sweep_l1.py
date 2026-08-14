"""budget_sweep with the planner cost switched to L1. Same arguments, same output.

A wrapper, not a fork: budget_sweep.py is imported and its main() is called
unchanged, so the episode protocol, the tiers, the seeding, the render gate and the
CSV schema are literally the same code. The single intervention is on
swm.wm.utils.load_pretrained -- the one place budget_sweep obtains a model
(budget_sweep.py:270) -- which now re-blesses the loaded instance into the L1
subclass. The weights, the config and every other method are untouched.

Patching that attribute reaches budget_sweep because both modules hold a reference
to the same stable_worldmodel module object; there is no second copy to miss.

    usage: exactly budget_sweep.py's, e.g.
      python scripts/budget_sweep_l1.py --env cube --solver cem --config k1_l1 \
          <ckpt>/weights_epoch_10.pt --tiers T1 T2 T3 T4 T5 \
          --episodes-json ... --out ...

Only cem and icem are accepted: mppi's softmax temperature and gd's fixed learning
rate both react to the L1/L2 magnitude difference, so a change under those solvers
would not be attributable to the norm. See l1_cost.py for the details.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402

from scripts import budget_sweep  # noqa: E402
from scripts.l1_cost import make_l1_class, verify  # noqa: E402

_ALLOWED = {"cem", "icem"}


def _guard_argv():
    argv = sys.argv[1:]
    if "--solver" in argv:
        slv = argv[argv.index("--solver") + 1]
    else:
        slv = "cem"  # budget_sweep's default
    if slv not in _ALLOWED:
        raise SystemExit(
            f"solver {slv!r} is not valid for the L1-cost comparison. Only "
            f"{sorted(_ALLOWED)} rank by cost alone; mppi rescales through a fixed "
            f"softmax temperature and gd descends the cost gradient at a fixed lr, "
            f"so both would change for reasons other than the norm."
        )


_orig_load = swm.wm.utils.load_pretrained


def _load_l1(*args, **kwargs):
    model = _orig_load(*args, **kwargs)
    model.__class__ = make_l1_class(type(model))
    verify(model)
    return model


if __name__ == "__main__":
    _guard_argv()
    swm.wm.utils.load_pretrained = _load_l1
    budget_sweep.main()
