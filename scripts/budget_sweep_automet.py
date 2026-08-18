"""budget_sweep with the AutoMetric learned Mahalanobis planner cost.

A wrapper in the cost_variants mold: budget_sweep.main() runs unchanged; the single
intervention re-blesses the loaded model so its criterion computes

    J(a) = (z_hat_H - z_g)^T  M  (z_hat_H - z_g)

with M the trace-normalized metric learned by automet_train.py from trajectory
temporal ordering alone (tr(M)=D; M=I recovers the shipped cost exactly). The
metric file is passed via --metric and verified at load: a synthetic probe with a
known M-distance must round-trip, and tr(M) must equal D.

Only cem and icem are accepted, same rationale as the cost family: they rank by
cost alone, while mppi/gd react to cost magnitude.

    usage: budget_sweep_automet.py --metric <automet_*.pt> [budget_sweep args]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import stable_worldmodel as swm  # noqa: E402

from scripts import budget_sweep  # noqa: E402
from scripts.cost_variants import _terminal  # noqa: E402

_ALLOWED_SOLVERS = {"cem", "icem"}


def _take_metric():
    argv = sys.argv[1:]
    if "--metric" not in argv:
        raise SystemExit("--metric <automet_*.pt> is required")
    i = argv.index("--metric")
    path = argv[i + 1]
    del argv[i:i + 2]
    slv = argv[argv.index("--solver") + 1] if "--solver" in argv else "cem"
    if slv not in _ALLOWED_SOLVERS:
        raise SystemExit(f"solver {slv!r} invalid for a cost comparison; use cem/icem")
    sys.argv = [sys.argv[0]] + argv
    return path


def main():
    path = _take_metric()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    M = blob["M"].float()
    D = M.shape[0]
    tr = float(M.diagonal().sum())
    assert abs(tr - D) < 1e-3 * D, f"metric not trace-normalized: tr={tr}, D={D}"
    print(f"[automet] loaded {path}: D={D}, tr(M)={tr:.2f}, "
          f"task={blob.get('task')}, ckpt={blob.get('ckpt')}", flush=True)

    _orig_load = swm.wm.utils.load_pretrained

    def load_and_bless(ckpt, *a, **kw):
        model = _orig_load(ckpt, *a, **kw)
        Mdev = {}

        class AutoMetricCost(type(model)):
            def criterion(self, info_dict):
                p, g, nd = _terminal(info_dict)
                d = (p - g).squeeze(-2)          # (B, S, D)
                dev = d.device
                if dev not in Mdev:
                    Mdev[dev] = M.to(dev)
                return ((d @ Mdev[dev]) * d).sum(-1)

        model.__class__ = AutoMetricCost
        # verify: with d = ones, cost must equal sum(M) exactly
        ones = torch.ones(1, 1, 1, D)
        got = float(model.criterion({"predicted_emb": ones * 2.0,
                                     "goal_emb": ones}).item())
        want = float(M.sum())
        assert abs(got - want) < 1e-2 * abs(want) + 1e-6, (got, want)
        print(f"[automet] criterion verified: {got:g} == sum(M) {want:g}", flush=True)
        return model

    env = None
    for i, a2 in enumerate(sys.argv):
        if a2 == "--env":
            env = sys.argv[i + 1]
    if env == "tworoom":
        from scripts.tworoom_preset import register
        register(budget_sweep.ENV_PRESETS)
    elif env == "pointmaze":
        from scripts.pointmaze_preset import register
        register(budget_sweep.ENV_PRESETS)
    swm.wm.utils.load_pretrained = load_and_bless
    budget_sweep.main()


if __name__ == "__main__":
    main()
