"""Swap the planner's cost from squared L2 to L1, without touching anything else.

The shipped cost (stable_worldmodel/wm/lewm/lewm.py, LeWM.criterion) is

    F.mse_loss(pred, goal, reduction='none').sum(...)   ==  ||z_hat - z_goal||^2

i.e. a plain squared L2 over the latent, on the TERMINAL step only. This module
returns the same thing with abs() in place of the square:

    (pred - goal).abs().sum(...)                        ==  ||z_hat - z_goal||_1

Why this is worth measuring. L_obj is literally
1 - Pearson(||dz||^2, ||dq||^2): the quantity it calibrates is the SQUARED latent
distance, and the planner's cost is that same squared distance. The two match
exactly. Under L1 the planner reads a quantity L_obj never optimised, so if
L_obj's advantage is "it calibrated that particular norm" it should shrink more
than the aux arm's, whose q-head shapes decodability rather than distance
geometry. Either outcome is informative, and it is the first measurement with a
chance of separating obj from aux -- SR and P5 both leave them statistically
indistinguishable.

Scope, deliberately narrow: criterion() is reached only through get_cost(), which
is planning-only (training computes its own pred_loss in train.py:lejepa_forward).
So nothing about any checkpoint or any training path changes -- the same weights
are simply scored differently.

Only CEM and iCEM are valid arms for this comparison:
  * cem/icem  torch.topk(costs, largest=False) -- pure RANKING, no scale sensitivity
  * mppi      softmax(-(cost - min)/0.5) -- subtracts the min but does not rescale
              by spread, so the L1/L2 magnitude difference silently re-tunes the
              temperature; a change there is not attributable to the norm
  * gd        cost.sum().backward() at lr=1.0 -- L2's gradient is 2*(z_hat-z_goal)
              and vanishes near the goal, L1's is sign(.) with constant magnitude
              and no derivative at 0; the optimiser dynamics change, not just the
              ranking
"""

import torch


def make_l1_class(cls):
    """A subclass of the loaded world model whose planner cost is L1.

    Built per concrete class so this works whatever load_pretrained returns, and
    so `type(model).__mro__` still contains the original -- everything except
    criterion is inherited untouched.
    """

    class _L1(cls):
        # marker for the runtime assertion below and for the log gate in ray_eval_l1.sh
        _cost_norm = "L1"

        def criterion(self, info_dict: dict):
            pred_emb = info_dict["predicted_emb"]
            goal_emb = info_dict["goal_emb"]
            # same two lines as the shipped criterion: broadcast the goal over the
            # rollout, keep the terminal step only
            goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
            return (pred_emb[..., -1:, :] - goal_emb[..., -1:, :].detach()).abs().sum(
                dim=tuple(range(2, pred_emb.ndim))
            )

    _L1.__name__ = cls.__name__ + "_L1cost"
    return _L1


def verify(model):
    """Prove numerically that this model scores with L1, not the square.

    A print is not enough: a wiring mistake that left the shipped criterion in
    place would produce a full run of plausible-looking L2 numbers filed under an
    l1 name. So feed criterion a hand-built info_dict whose answer is known under
    both norms and require the L1 one.
    """
    d, delta = 8, 3.0
    pred = torch.zeros(1, 1, 1, d)
    goal = torch.full((1, 1, 1, d), delta)
    got = float(model.criterion({"predicted_emb": pred, "goal_emb": goal}).item())
    want_l1, want_l2 = d * abs(delta), d * delta ** 2
    if abs(got - want_l1) > 1e-4:
        raise SystemExit(
            f"FATAL: planner cost is not L1. criterion returned {got:g}; "
            f"L1 would be {want_l1:g} and squared L2 {want_l2:g}"
            + (" -- this is the shipped squared cost, the patch did not take"
               if abs(got - want_l2) < 1e-4 else "")
        )
    print(f"[cost] L1 verified: sum|dz| over {d} dims at delta={delta} -> {got:g} "
          f"(squared L2 would be {want_l2:g})", flush=True)
