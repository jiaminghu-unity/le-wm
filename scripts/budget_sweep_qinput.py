"""budget_sweep for the q-input model (QJEPA): planning consumes env `state`, not
pixels.

Single intervention beyond stock budget_sweep: the eval pipeline StandardScaler-
processes `state`/`goal_state`, but QJEPA.encode expects RAW state (it applies its
own persisted q-normalization internally, matching training). So build_process is
wrapped to drop the state scalers. Everything else -- episodes, tiers, seeds, CEM
plumbing, goal-snapshot re-injection -- is stock.

    usage: budget_sweep_qinput.py [budget_sweep args]   (pusht only)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402

from scripts import budget_sweep  # noqa: E402
from qjepa import QJEPA  # noqa: E402  (import also makes the class resolvable for hydra)


def main():
    assert "--env" not in sys.argv or sys.argv[sys.argv.index("--env") + 1] == "pusht", \
        "q-input arm is pusht-only (q_stats/build_q_raw are pusht conventions)"

    _orig_bp = budget_sweep.build_process

    def build_process_no_state(dataset, cols):
        process = _orig_bp(dataset, cols)
        for k in ("state", "goal_state"):
            process.pop(k, None)
        print(f"[qinput] state scalers dropped; process keys = {sorted(process)}", flush=True)
        return process

    _orig_load = swm.wm.utils.load_pretrained

    def load_and_check(ckpt, *a, **kw):
        model = _orig_load(ckpt, *a, **kw)
        assert isinstance(model, QJEPA), f"expected QJEPA, got {type(model)}"
        std = model.q_std.detach()
        assert not bool((std == 1).all()), "q_std buffer still at init -- stats never trained in"
        print(f"[qinput] QJEPA loaded; q_mean[:2]={model.q_mean[:2].tolist()}", flush=True)
        return model

    budget_sweep.build_process = build_process_no_state
    swm.wm.utils.load_pretrained = load_and_check
    budget_sweep.main()


if __name__ == "__main__":
    main()
