"""Experiment C — DUET-style dual-target diagnostic (pure vision, no tactile).

Keeps TWO per-view latents and TWO prediction targets instead of one pooled joint
latent: shared ViT + bidirectional cross-attention (identical to B1), then
PER-VIEW query pooling -> zF, zW (192 each); two independent LeWM predictors
P_f, P_w. Loss = per-view next-step MSE + 0.09·SIGReg on each view.

Diagnostic model only. If run through CEM, the goal cost becomes the SUM of the
two per-view latent MSEs — a DIFFERENT cost definition from B1's single-space MSE;
any SR from it must be labeled as such (per the analysis memo).

API-compatible with the training script: encode() returns the concatenation
[zF; zW] (B,T,384); predict() returns concatenated per-view predictions.
"""
import torch
from torch import nn
from einops import rearrange

from .multiview_encoder import PatchViT, CrossAttnBlock, QueryPool
from .lewm_modules import SIGReg, ARPredictor, Embedder, MLP


class DuetDiagJEPA(nn.Module):
    def __init__(self, action_dim=6, dim=192, history_size=3, sigreg_weight=0.09,
                 max_frames=64):
        super().__init__()
        self.history_size = history_size
        self.sigreg_weight = sigreg_weight
        self.dim = dim
        self.vit = PatchViT(dim=dim)
        self.view_emb = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        self.ca_f = CrossAttnBlock(dim, 4, 4)
        self.ca_w = CrossAttnBlock(dim, 4, 4)
        self.pool_f = QueryPool(dim, 8)
        self.pool_w = QueryPool(dim, 8)
        self.proj_f = nn.Linear(8 * dim, dim)
        self.proj_w = nn.Linear(8 * dim, dim)
        self.projector_f = MLP(dim, 2048, dim, norm_fn=nn.BatchNorm1d)
        self.projector_w = MLP(dim, 2048, dim, norm_fn=nn.BatchNorm1d)
        mk = lambda: ARPredictor(num_frames=max_frames, depth=6, heads=16,
                                 mlp_dim=2048, input_dim=dim, hidden_dim=dim,
                                 output_dim=dim, dropout=0.1)
        self.pred_f, self.pred_w = mk(), mk()
        self.pp_f = MLP(dim, 2048, dim, norm_fn=nn.BatchNorm1d)
        self.pp_w = MLP(dim, 2048, dim, norm_fn=nn.BatchNorm1d)
        self.action_encoder = Embedder(input_dim=action_dim, smoothed_dim=action_dim,
                                       emb_dim=dim)
        self.sigreg = SIGReg()

    def _views(self, obs):
        f, w = obs["front"], obs["wrist"]
        B, T = f.shape[:2]
        Ft = self.vit(rearrange(f, "b t c h w -> (b t) c h w").float()) + self.view_emb[0][None]
        Wt = self.vit(rearrange(w, "b t c h w -> (b t) c h w").float()) + self.view_emb[1][None]
        Fp = self.ca_f(Ft, Wt); Wp = self.ca_w(Wt, Ft)      # pre-update both sides
        zF = self.proj_f(self.pool_f(Fp).flatten(1))
        zW = self.proj_w(self.pool_w(Wp).flatten(1))
        zF = self.projector_f(zF); zW = self.projector_w(zW)
        zF = rearrange(zF, "(b t) d -> b t d", b=B)
        zW = rearrange(zW, "(b t) d -> b t d", b=B)
        return zF, zW

    def encode(self, obs):
        zF, zW = self._views(obs)
        return torch.cat([zF, zW], dim=-1)                   # (B,T,384)

    def _split(self, z):
        return z[..., :self.dim], z[..., self.dim:]

    def predict(self, z_history, action_history):
        zF, zW = self._split(z_history)
        a = self.action_encoder(action_history)
        pf = self.pred_f(zF, a); pw = self.pred_w(zW, a)
        B, T, D = pf.shape
        pf = self.pp_f(pf.reshape(B * T, D)).reshape(B, T, D)
        pw = self.pp_w(pw.reshape(B * T, D)).reshape(B, T, D)
        return torch.cat([pf, pw], dim=-1)

    def loss(self, obs, actions):
        z = self.encode(obs)
        pred = self.predict(z[:, :-1], actions)
        tgt = z[:, 1:]
        zF, zW = self._split(z)
        pF, pW = self._split(pred)
        tF, tW = self._split(tgt)
        pred_loss = (pF - tF).pow(2).mean() + (pW - tW).pow(2).mean()
        sig = 0.5 * (self.sigreg(zF.transpose(0, 1)) + self.sigreg(zW.transpose(0, 1)))
        total = pred_loss + self.sigreg_weight * sig
        return {"loss": total, "pred_loss": pred_loss.detach(),
                "sigreg_loss": sig.detach(), "emb": z}

    @torch.no_grad()
    def rollout(self, current_obs, candidate_actions):
        B, S, T, A = candidate_actions.shape
        z0 = self.encode(current_obs)
        H = z0.size(1)
        z = rearrange(z0.unsqueeze(1).expand(B, S, H, -1), "b s h d -> (b s) h d").clone()
        acts = rearrange(candidate_actions, "b s t a -> (b s) t a")
        hist_act = (acts[:, :1] * 0).expand(-1, H, -1).clone()
        _ah = getattr(self, "_act_hist", None)   # 真实已执行动作历史 (B,H,A)
        if _ah is not None:
            _ah = _ah.to(acts.device).float()
            hist_act = _ah.unsqueeze(1).expand(B, S, H, -1).reshape(B * S, H, -1).clone()
        outs = []
        for t in range(T):
            hist_act = torch.cat([hist_act[:, 1:], acts[:, t:t + 1]], dim=1)
            pred = self.predict(z[:, -self.history_size:],
                                hist_act[:, -self.history_size:])[:, -1:]
            z = torch.cat([z, pred], dim=1)
            outs.append(pred)
        fut = torch.cat(outs, dim=1)
        return rearrange(fut, "(b s) t d -> b s t d", b=B)

    @staticmethod
    def cem_cost(pred_future, z_goal):
        """SUM of per-view last-step MSEs — different definition from B1; label it."""
        last = pred_future[:, :, -1, :]
        d = (last - z_goal.unsqueeze(1)).pow(2)
        return d[..., :192].sum(-1) + d[..., 192:].sum(-1)
