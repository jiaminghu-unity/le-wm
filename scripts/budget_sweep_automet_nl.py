"""budget_sweep with the NONLINEAR AutoMetric planner cost:
J(a) = ||phi(z_hat_H) - phi(z_g)||^2, phi loaded from --metric (automet_train_nl.py).
Same wrapper contract as budget_sweep_automet.py; cem/icem only. Verified at load:
with the residual MLP, phi(z) != z in general, but the criterion must equal the
manually computed embedded distance on a random probe.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import stable_worldmodel as swm  # noqa: E402

from scripts import budget_sweep  # noqa: E402
from scripts.cost_variants import _terminal  # noqa: E402
from scripts.automet_train_nl import Phi  # noqa: E402

_ALLOWED = {"cem", "icem"}


def _take_metric():
    argv = sys.argv[1:]
    i = argv.index("--metric")
    path = argv[i + 1]
    del argv[i:i + 2]
    slv = argv[argv.index("--solver") + 1] if "--solver" in argv else "cem"
    if slv not in _ALLOWED:
        raise SystemExit(f"solver {slv!r} invalid; use cem/icem")
    sys.argv = [sys.argv[0]] + argv
    return path


def main():
    path = _take_metric()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob.get("kind") == "nonlinear", blob.get("kind")
    D = blob["D"]
    phi_cpu = Phi(D, blob["hidden"])
    phi_cpu.load_state_dict(blob["phi_state"])
    phi_cpu.eval()
    print(f"[automet-nl] loaded {path}: D={D}, task={blob.get('task')}", flush=True)

    _orig_load = swm.wm.utils.load_pretrained

    def load_and_bless(ckpt, *a, **kw):
        model = _orig_load(ckpt, *a, **kw)
        phis = {}

        class NLCost(type(model)):
            def criterion(self, info_dict):
                p, g, nd = _terminal(info_dict)
                dev = p.device
                if dev not in phis:
                    phis[dev] = Phi(D, blob["hidden"]).to(dev)
                    phis[dev].load_state_dict(blob["phi_state"])
                    phis[dev].eval()
                with torch.no_grad():
                    ep = phis[dev](p.squeeze(-2).float())
                    eg = phis[dev](g.squeeze(-2).float())
                return (ep - eg).pow(2).sum(-1)

        model.__class__ = NLCost
        probe_p = torch.randn(1, 1, 1, D)
        probe_g = torch.randn(1, 1, 1, D)
        got = float(model.criterion({"predicted_emb": probe_p, "goal_emb": probe_g}).item())
        with torch.no_grad():
            want = float((phi_cpu(probe_p.squeeze(-2)) - phi_cpu(probe_g.squeeze(-2))).pow(2).sum(-1).item())
        assert abs(got - want) < 1e-3 * max(abs(want), 1.0), (got, want)
        print(f"[automet-nl] criterion verified: {got:g} == {want:g}", flush=True)
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
