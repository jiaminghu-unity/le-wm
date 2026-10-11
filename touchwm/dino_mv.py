"""D1 — Frozen-DINO multi-view LeWM.

Backbone: frozen DINOv2 ViT-S/14 patch tokens (precomputed cache, fp16, 256x384
per view per frame; DINOv3 weights are license-gated, documented fallback).
Trainable: linear adapters 384->192, the EXISTING bidirectional cross-attention,
8 shared queries, projection to the 192-d joint latent, projector MLP, original
ARPredictor + action Embedder. Losses/CEM identical to B1.
"""
import torch
from torch import nn
from einops import rearrange

from .multiview_encoder import CrossAttnBlock, QueryPool
from .lewm_modules import SIGReg, ARPredictor, Embedder, MLP


class DinoFusionEncoder(nn.Module):
    def __init__(self, token_dim=384, dim=192, num_queries=8, latent_dim=192):
        super().__init__()
        self.adapt_f = nn.Linear(token_dim, dim)
        self.adapt_w = nn.Linear(token_dim, dim)
        self.view_emb = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        self.ca_f = CrossAttnBlock(dim, 4, 4)
        self.ca_w = CrossAttnBlock(dim, 4, 4)
        self.pool = QueryPool(dim, num_queries)
        self.project = nn.Linear(num_queries * dim, latent_dim)

    def forward(self, tok_f, tok_w):
        """tok_*: (N, P, 384) frozen DINO patch tokens -> z (N, latent)."""
        Ft = self.adapt_f(tok_f.float()) + self.view_emb[0][None]
        Wt = self.adapt_w(tok_w.float()) + self.view_emb[1][None]
        Fp = self.ca_f(Ft, Wt); Wp = self.ca_w(Wt, Ft)
        U = self.pool(torch.cat([Fp, Wp], dim=1))
        return self.project(U.flatten(1)), Wp.mean(1), Fp.mean(1)


class DinoMultiViewJEPA(nn.Module):
    """Same training objective / predictor as B1; consumes cached DINO tokens."""

    def __init__(self, action_dim=6, latent_dim=192, history_size=3,
                 sigreg_weight=0.09, max_frames=64):
        super().__init__()
        self.history_size = history_size
        self.sigreg_weight = sigreg_weight
        self.encoder = DinoFusionEncoder(latent_dim=latent_dim)
        self.projector = MLP(latent_dim, 2048, latent_dim, norm_fn=nn.BatchNorm1d)
        self.pred_proj = MLP(latent_dim, 2048, latent_dim, norm_fn=nn.BatchNorm1d)
        self.predictor = ARPredictor(num_frames=max_frames, depth=6, heads=16,
                                     mlp_dim=2048, input_dim=latent_dim,
                                     hidden_dim=latent_dim, output_dim=latent_dim,
                                     dropout=0.1)
        self.action_encoder = Embedder(input_dim=action_dim, smoothed_dim=action_dim,
                                       emb_dim=latent_dim)
        self.sigreg = SIGReg()

    def encode_tokens(self, tok_f, tok_w):
        """tok_*: (B, T, P, 384) -> z (B, T, D)."""
        B, T = tok_f.shape[:2]
        z, _, _ = self.encoder(rearrange(tok_f, "b t p d -> (b t) p d"),
                               rearrange(tok_w, "b t p d -> (b t) p d"))
        z = self.projector(z)
        return rearrange(z, "(b t) d -> b t d", b=B)

    def predict(self, z_history, action_history):
        a = self.action_encoder(action_history)
        p = self.predictor(z_history, a)
        B, T, D = p.shape
        return self.pred_proj(p.reshape(B * T, D)).reshape(B, T, D)

    def loss(self, tok_f, tok_w, actions):
        z = self.encode_tokens(tok_f, tok_w)
        pred = self.predict(z[:, :-1], actions)
        pred_loss = (pred - z[:, 1:]).pow(2).mean()
        sig = self.sigreg(z.transpose(0, 1))
        return {"loss": pred_loss + self.sigreg_weight * sig,
                "pred_loss": pred_loss.detach(), "sigreg_loss": sig.detach(), "emb": z}
