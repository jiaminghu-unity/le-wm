"""Gated residual dynamics on top of the frozen-architecture LeWM predictor.

Ours (tactile-supervised):  ẑ_{t+1} = P0(z,a) + R_{m}(z,a),  R_0 = 0,
  gate G(z_t, act_emb_t) -> 3 logits trained with CE against tactile-derived mode
  labels (train-time only); routing is teacher-forced on labeled transitions and
  soft (gate-prob weighted) on unlabeled/ambiguous ones; argmax at deployment.
B3 (capacity control): identical parameters, no CE, always soft routing.
"""
import torch
from torch import nn

from .lewm_adapter import MultiViewJEPA


class ResidualBranch(nn.Module):
    def __init__(self, dim=192, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, hidden),
                                 nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, a_emb):
        return self.net(torch.cat([z, a_emb], dim=-1))


class GatedResidualJEPA(MultiViewJEPA):
    """Adds gate + per-mode residuals; mode 0 residual is identically zero."""

    def __init__(self, num_modes=3, gate_hidden=256, res_hidden=512, **kw):
        super().__init__(**kw)
        d = 192
        self.num_modes = num_modes
        self.gate = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, gate_hidden),
                                  nn.GELU(), nn.Linear(gate_hidden, num_modes))
        self.residuals = nn.ModuleList([ResidualBranch(d, res_hidden)
                                        for _ in range(num_modes - 1)])  # modes 1..K-1

    def gate_logits(self, z, a_emb):
        return self.gate(torch.cat([z, a_emb], dim=-1))

    def _residual_stack(self, z, a_emb):
        """(B,T,D) -> (B,T,K,D) with mode-0 residual = 0."""
        zero = torch.zeros_like(z)
        outs = [zero] + [r(z, a_emb) for r in self.residuals]
        return torch.stack(outs, dim=-2)

    def predict_gated(self, z_hist, act_hist, mode_labels=None):
        """Returns (pred, gate_logits). mode_labels (B,T) with -1/-2 = unlabeled:
        labeled steps route through their true branch (teacher forcing), unlabeled
        steps use gate-probability soft routing. mode_labels=None => all soft (B3/eval)."""
        a_emb = self.action_encoder(act_hist)
        base = self.predictor(z_hist, a_emb)
        B, T, D = base.shape
        base = self.pred_proj(base.reshape(B * T, D)).reshape(B, T, D)
        res = self._residual_stack(z_hist, a_emb)              # (B,T,K,D)
        logits = self.gate_logits(z_hist, a_emb)               # (B,T,K)
        soft = torch.softmax(logits, dim=-1)
        soft_res = (soft.unsqueeze(-1) * res).sum(-2)
        if mode_labels is None:
            chosen = soft_res
        else:
            lab = mode_labels.clamp(min=0)
            hard = torch.gather(res, -2, lab[..., None, None].expand(-1, -1, 1, D)).squeeze(-2)
            use_hard = (mode_labels >= 0).unsqueeze(-1).float()
            chosen = use_hard * hard + (1 - use_hard) * soft_res
        return base + chosen, logits

    @torch.no_grad()
    def rollout(self, current_obs, candidate_actions):
        """Deployment rollout: argmax gate, vision-only (same signature as B1)."""
        from einops import rearrange
        B, S, T, A = candidate_actions.shape
        z0 = self.encode(current_obs)
        H = z0.size(1)
        z = rearrange(z0.unsqueeze(1).expand(B, S, H, -1), "b s h d -> (b s) h d").clone()
        acts = rearrange(candidate_actions, "b s t a -> (b s) t a")
        hist_act = (acts[:, :1] * 0).expand(-1, H, -1).clone()
        outs = []
        for t in range(T):
            hist_act = torch.cat([hist_act[:, 1:], acts[:, t:t + 1]], dim=1)
            zh = z[:, -self.history_size:]; ah = hist_act[:, -self.history_size:]
            a_emb = self.action_encoder(ah)
            base = self.predictor(zh, a_emb)
            b_, t_, d_ = base.shape
            base = self.pred_proj(base.reshape(b_ * t_, d_)).reshape(b_, t_, d_)
            res = self._residual_stack(zh, a_emb)
            hard = torch.argmax(self.gate_logits(zh, a_emb), dim=-1)
            pick = torch.gather(res, -2, hard[..., None, None].expand(-1, -1, 1, d_)).squeeze(-2)
            pred = (base + pick)[:, -1:]
            z = torch.cat([z, pred], dim=1)
            outs.append(pred)
        fut = torch.cat(outs, dim=1)
        return rearrange(fut, "(b s) t d -> b s t d", b=B)

    def loss_gated(self, obs, actions, mode_labels, supervise_gate: bool,
                   gate_weight=0.05):
        z = self.encode(obs)
        pred, logits = self.predict_gated(
            z[:, :-1], actions, mode_labels if supervise_gate else None)
        tgt = z[:, 1:]
        pred_loss = (pred - tgt).pow(2).mean()
        sig = self.sigreg(z.transpose(0, 1))
        total = pred_loss + self.sigreg_weight * sig
        out = {"pred_loss": pred_loss.detach(), "sigreg_loss": sig.detach()}
        if supervise_gate:
            lab = mode_labels.reshape(-1)
            lg = logits.reshape(-1, self.num_modes)
            msk = lab >= 0
            if msk.any():
                ce = nn.functional.cross_entropy(lg[msk], lab[msk])
                total = total + gate_weight * ce
                out["gate_ce"] = ce.detach()
                out["gate_acc"] = (lg[msk].argmax(-1) == lab[msk]).float().mean().detach()
        out["loss"] = total
        return out
