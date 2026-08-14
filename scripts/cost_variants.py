"""Alternative planning costs, chosen so the contrasts decompose exactly.

Squared L2 expands, and the goal is the same for every candidate at a given start:

    ||z_hat - g||^2 = ||z_hat||^2 - 2<z_hat, g> + ||g||^2
                      \\_ norm _/   \\_ align _/   \\_ const _/

Ranking by the shipped cost is ranking by (norm + align). So:

    l2    the shipped cost, kept here as the reference implementation
    dot   -<z_hat, g>            drops the norm term EXACTLY
    cos   -<z_hat, g>/||z_hat||  divides by the norm instead of dropping it
    l1    sum |z_hat - g|        a different norm shape; the change cannot be
                                 attributed to any single term (already measured;
                                 see l1_cost.py, left untouched as the provenance
                                 of those results)

dot vs l2 is therefore an exact attribution: the whole difference is the
||z_hat||^2 term. That term is not small where it matters -- 16-22% of the cost
variance on Push-T and 17-30% on Cube -- and scripts/probe_latent_geometry.py
measured that L_obj raises its share well above baseline on both tasks (22.0 and
29.7 against 16.3 and 17.7) while the aux arm leaves it alone (15.8, 17.2). Hence
the pre-registered prediction: dropping the term should cost the obj arm more than
baseline or aux, and more on Cube than on Push-T.

Reacher is deliberately excluded: there the term is 1.6% for EVERY arm, the dot
ranking agrees with the shipped one at tau 0.90 and keeps 95.5% of the same 30
elites, so the sweep could not measure anything.

Validity, same as l1_cost.py: criterion() is reached only through get_cost(), which
is planning-only -- training computes its own pred_loss in train.py:lejepa_forward.
Nothing about any checkpoint changes; the same weights are scored differently.

Only cem and icem are valid arms. Both select by rank alone. mppi weights by
softmax(-(cost - min)/temperature) at a fixed temperature and gd descends the cost
gradient at a fixed lr, so both react to the magnitude change -- and dot and cos
change magnitude far more than L1 did (dot is not even non-negative).
"""

import torch


def _terminal(info_dict):
    """Predicted and goal embeddings at the terminal step, sliced exactly as the
    shipped criterion slices them (stable_worldmodel/wm/lewm/lewm.py)."""
    pred_emb = info_dict["predicted_emb"]
    goal_emb = info_dict["goal_emb"]
    goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
    return pred_emb[..., -1:, :], goal_emb[..., -1:, :].detach(), pred_emb.ndim


def _cost_l2(info_dict):
    p, g, nd = _terminal(info_dict)
    return (p - g).pow(2).sum(dim=tuple(range(2, nd)))


def _cost_l1(info_dict):
    p, g, nd = _terminal(info_dict)
    return (p - g).abs().sum(dim=tuple(range(2, nd)))


def _cost_dot(info_dict):
    p, g, nd = _terminal(info_dict)
    return -(p * g).sum(dim=tuple(range(2, nd)))


def _cost_cos(info_dict):
    p, g, nd = _terminal(info_dict)
    dims = tuple(range(2, nd))
    # ||g|| is constant across candidates at a start, so dividing by ||p|| alone would
    # already be rank-equivalent; the explicit ||g|| keeps logged values in [-1, 1]
    num = (p * g).sum(dim=dims)
    den = p.pow(2).sum(dim=dims).sqrt() * g.pow(2).sum(dim=dims).sqrt()
    return -num / den.clamp_min(1e-12)


COSTS = {"l2": _cost_l2, "l1": _cost_l1, "dot": _cost_dot, "cos": _cost_cos}

# Expected value of each variant on a fixed probe input, for the runtime proof below.
# p = ones, g = delta over d dims: l2 = d*(1-delta)^2, l1 = d*|1-delta|,
# dot = -d*delta, cos = -1 (the two vectors are parallel). p = zeros is NOT usable --
# it sends dot to 0 for every variant and could not tell dot from cos.
_D, _DELTA = 8, 3.0
_EXPECT = {"l2": _D * (1 - _DELTA) ** 2, "l1": _D * abs(1 - _DELTA),
           "dot": -_D * _DELTA, "cos": -1.0}


def make_cost_class(cls, name):
    """A subclass of the loaded world model whose planner cost is `name`."""
    if name not in COSTS:
        raise ValueError(f"unknown cost {name!r}; have {sorted(COSTS)}")
    fn = COSTS[name]

    class _Cost(cls):
        _cost_norm = name

        def criterion(self, info_dict: dict):
            return fn(info_dict)

    _Cost.__name__ = f"{cls.__name__}_{name}cost"
    return _Cost


def verify(model, name):
    """Prove numerically which cost this model scores with.

    A print is not enough: a wiring mistake that left another variant in place would
    produce a full run of plausible numbers filed under the wrong name. So feed
    criterion an input whose answer is known under EVERY variant and require this one --
    and report which other variant it matched if it failed, since that names the bug.
    """
    p = torch.ones(1, 1, 1, _D)
    g = torch.full((1, 1, 1, _D), _DELTA)
    got = float(model.criterion({"predicted_emb": p, "goal_emb": g}).item())
    want = _EXPECT[name]
    if abs(got - want) > 1e-4:
        matched = [k for k, v in _EXPECT.items() if abs(got - v) < 1e-4]
        raise SystemExit(
            f"FATAL: planner cost is not {name!r}. criterion returned {got:g}, "
            f"expected {want:g}"
            + (f" -- it matches {matched} instead, so the wrong variant was selected"
               if matched else " -- it matches no known variant")
        )
    others = {k: v for k, v in _EXPECT.items() if k != name}
    print(f"[cost] {name} verified: criterion -> {got:g}; "
          + ", ".join(f"{k} would be {v:g}" for k, v in sorted(others.items())),
          flush=True)
