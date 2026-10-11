"""Closed-loop adapter: plug MultiViewJEPA / GatedResidualJEPA into ContactWorld's
official BatchedCEMPlanner with ZERO planner changes.

Trick (documented, no tactile anywhere): the planner's second observation channel
(`use_tactile` + `tactile_key`) is reused to carry the WRIST camera. Run with
  --vision-key front --use-tactile --tactile-key wrist
so current_info/goal_info reach get_cost with both RGB streams.
Cost = original latent MSE on the last step (or discounted sum, per flags).
"""
import torch
from torch import nn
import torch.nn.functional as Fn


class MVJepaWorldModel(nn.Module):
    vision_key = "front"
    tactile_key = "wrist"       # second CAMERA, not tactile (see module docstring)
    use_tactile = True

    def __init__(self, ckpt_path, variant="b1", device="cuda"):
        super().__init__()
        if variant == "b1":
            from touchwm.lewm_adapter import MultiViewJEPA
            self.model = MultiViewJEPA()
        elif variant in ("front", "wrist"):
            from touchwm.front_only import FrontOnlyJEPA
            self.model = FrontOnlyJEPA(view=variant)
        elif variant == "ours2":
            from touchwm.mode_residual import GatedResidualJEPA
            self.model = GatedResidualJEPA(wrist_skip=True)
        else:
            from touchwm.mode_residual import GatedResidualJEPA
            self.model = GatedResidualJEPA()
        st = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(st["model"] if "model" in st else st)
        self.model.to(device).eval()
        self.variant = variant
        print(f"[adapter] loaded {variant} ckpt step={st.get('step')}", flush=True)

    @staticmethod
    def _prep(x):
        """[B,T,C,H,W] (or [B,C,H,W]) float -> resized 224, [0,1]."""
        if x.ndim == 4: x = x[:, None]
        x = x.float()
        if x.max() > 1.5: x = x / 255.0
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        if x.shape[-1] != 224:
            x = Fn.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
        return x.reshape(B, T, 3, 224, 224)

    @torch.no_grad()
    def get_cost(self, current_info, action_candidates, goal_info,
                 sum_all_diffs=False, discount=1.0):
        dev = action_candidates.device
        cur = {"front": self._prep(current_info[self.vision_key].to(dev)),
               "wrist": self._prep(current_info[self.tactile_key].to(dev))}
        goal = {"front": self._prep(goal_info[self.vision_key].to(dev)),
                "wrist": self._prep(goal_info[self.tactile_key].to(dev))}
        z_goal = self.model.encode(goal)[:, -1]                  # (B,D)
        fut = self.model.rollout(cur, action_candidates)         # (B,S,T,D)
        if not sum_all_diffs:
            return (fut[:, :, -1] - z_goal[:, None]).pow(2).sum(-1)
        w = discount ** torch.arange(fut.shape[2], device=dev).float()
        d = (fut - z_goal[:, None, None]).pow(2).sum(-1)         # (B,S,T)
        return (d * w[None, None]).sum(-1)
