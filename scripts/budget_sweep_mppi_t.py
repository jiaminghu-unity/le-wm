"""budget_sweep with an explicit MPPI temperature. Same protocol, same CSV schema.

WHY. MPPISolver weights candidates by softmax(-(cost - min)/T) with NO scale
normalization, and the two model families put their costs on scales ~5 orders of
magnitude apart (LeWM: squared error SUMMED over 192 dims, typical candidate gaps
O(10-100); DINO-WM: squared error MEANED over 98304 dims, gaps O(0.01-0.1)). At the
repo default T=0.5 the LeWM arms therefore run a degenerate argmax-MPPI while
DINO-WM runs the intended smooth-averaging MPPI -- the mppi column's cross-model
comparison is confounded by an arbitrary unit choice. This wrapper re-runs mppi at
a temperature matched to the LeWM family's cost scale.

A wrapper, not a fork: budget_sweep.py's main() runs unchanged; the only
intervention is substituting a subclass of MPPISolver whose default temperature
comes from the MPPI_T environment variable, plus registering the tworoom/pointmaze
presets so all five tasks are reachable.

    usage: MPPI_T=<float> budget_sweep_mppi_t.py --env <task> --solver mppi ...
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402

from scripts import budget_sweep  # noqa: E402

T = float(os.environ["MPPI_T"])
_Orig = swm.solver.MPPISolver


class MPPISolverT(_Orig):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("temperature", T)
        super().__init__(*args, **kwargs)
        assert self.temperature == T, (self.temperature, T)
        print(f"[mppi_t] temperature={self.temperature}", flush=True)


if __name__ == "__main__":
    swm.solver.MPPISolver = MPPISolverT
    env = None
    for i, a in enumerate(sys.argv):
        if a == "--env":
            env = sys.argv[i + 1]
    if env == "tworoom":
        from scripts.tworoom_preset import register
        register(budget_sweep.ENV_PRESETS)
    elif env == "pointmaze":
        from scripts.pointmaze_preset import register
        register(budget_sweep.ENV_PRESETS)
    budget_sweep.main()
