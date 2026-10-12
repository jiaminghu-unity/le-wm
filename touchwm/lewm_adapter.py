"""Multi-view LeWM adapter — proposal §4.2 / §6.

Wires MultiViewEncoder into the original LeWM training objective:
  L = ||pred - z_next||^2 (mean) + 0.09 * SIGReg(emb)
with the original ARPredictor (depth 6, heads 16, mlp 2048, AdaLN-zero action
conditioning), action Embedder, and projector/pred_proj MLPs. End-to-end gradients;
no EMA teacher, no stop-grad target (per proposal §4.2).

Deviations logged:
  D1 — ViT re-implementation (see multiview_encoder.py).
  D2 — projector operates on the fused multi-view latent instead of a CLS token.
"""
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .lewm_modules import SIGReg, ARPredictor, Embedder, MLP
from .multiview_encoder import MultiViewEncoder


class MultiViewJEPA(nn.Module):
    def __init__(self, action_dim=6, latent_dim=192, history_size=3,
                 sigreg_weight=0.09, pred_depth=6, pred_heads=16, pred_mlp=2048,
                 pred_dropout=0.1, vit_depth=12, max_frames=64, wrist_skip=False):
        super().__init__()
        self.history_size = history_size
        self.sigreg_weight = sigreg_weight
        self.mv_encoder = MultiViewEncoder(latent_dim=latent_dim, vit_depth=vit_depth, wrist_skip=wrist_skip)
        self.projector = MLP(latent_dim, 2048, latent_dim,
                             norm_fn=nn.BatchNorm1d)      # as in official config
        self.pred_proj = MLP(latent_dim, 2048, latent_dim, norm_fn=nn.BatchNorm1d)
        self.predictor = ARPredictor(num_frames=max_frames, depth=pred_depth,
                                     heads=pred_heads, mlp_dim=pred_mlp,
                                     input_dim=latent_dim, hidden_dim=latent_dim,
                                     output_dim=latent_dim, dropout=pred_dropout)
        self.action_encoder = Embedder(input_dim=action_dim, smoothed_dim=action_dim,
                                       emb_dim=latent_dim)
        self.sigreg = SIGReg()

    # ---- §6.1 encoder API ----
    def encode(self, obs):
        """obs: {'front': (B,T,3,224,224), 'wrist': ...} -> z (B,T,D)."""
        z = self.mv_encoder.encode(obs)
        B, T, D = z.shape
        z = self.projector(z.reshape(B * T, D)).reshape(B, T, D)
        return z

    # ---- §6.2 prediction API ----
    def predict(self, z_history, action_history):
        """z_history (B,T,D), action_history (B,T,A) -> pred (B,T,D)."""
        act_emb = self.action_encoder(action_history)
        preds = self.predictor(z_history, act_emb)
        B, T, D = preds.shape
        preds = self.pred_proj(preds.reshape(B * T, D)).reshape(B, T, D)
        return preds

    def loss(self, obs, actions):
        """obs over T+? frames; actions (B, T-1, A). Teacher-forced next-step loss.
        Returns dict with total/pred/sigreg losses and the embeddings."""
        z = self.encode(obs)                              # (B, T, D)
        preds = self.predict(z[:, :-1], actions)          # predict z_{t+1}
        tgt = z[:, 1:]                                    # end-to-end: NO stop-grad
        pred_loss = (preds - tgt).pow(2).mean()
        sig = self.sigreg(z.transpose(0, 1))              # (T,B,D) convention
        total = pred_loss + self.sigreg_weight * sig
        return {"loss": total, "pred_loss": pred_loss.detach(),
                "sigreg_loss": sig.detach(), "emb": z}

    # ---- §6.2 rollout API (deployment-style, latent autoregression) ----
    @torch.no_grad()
    def rollout(self, current_obs, candidate_actions):
        """current_obs: {'front': (B,H,3,224,224), 'wrist': ...} H=history frames.
        candidate_actions: (B, S, T, A) — S candidate sequences, T future steps.
        Returns predicted future latents (B, S, T, D) in the joint visual space."""
        B, S, T, A = candidate_actions.shape
        z0 = self.encode(current_obs)                     # (B, H, D)
        H = z0.size(1)
        z = z0.unsqueeze(1).expand(B, S, H, -1)
        z = rearrange(z, "b s h d -> (b s) h d").clone()
        acts = rearrange(candidate_actions, "b s t a -> (b s) t a")
        hist_act = acts[:, :1] * 0                        # zero-pad action history
        hist_act = hist_act.expand(-1, H, -1).clone()
        _ah = getattr(self, "_act_hist", None)   # 真实已执行动作历史 (B,H,A)
        if _ah is not None:
            _ah = _ah.to(acts.device).float()
            hist_act = _ah.unsqueeze(1).expand(B, S, H, -1).reshape(B * S, H, -1).clone()
        outs = []
        for t in range(T):
            hist_act = torch.cat([hist_act[:, 1:], acts[:, t:t + 1]], dim=1)
            zh = z[:, -self.history_size:]
            ah = hist_act[:, -self.history_size:]
            pred = self.predict(zh, ah)[:, -1:]
            z = torch.cat([z, pred], dim=1)
            outs.append(pred)
        fut = torch.cat(outs, dim=1)                      # (BS, T, D)
        return rearrange(fut, "(b s) t d -> b s t d", b=B)

    @staticmethod
    def cem_cost(pred_future, z_goal):
        """Original latent cost C(A) = ||ẑ_{t+H} − z_g||² (last step, sum over D)."""
        last = pred_future[:, :, -1, :]                   # (B, S, D)
        return (last - z_goal.unsqueeze(1)).pow(2).sum(-1)
