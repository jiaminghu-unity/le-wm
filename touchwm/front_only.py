"""Matched front-only LeWM control (review R3): identical ViT / query pooling /
predictor / SIGReg / training protocol, but a single camera and no cross-attention.
Separates 'cost of multi-view fusion' from 'value of tactile supervision'."""
import torch
from torch import nn
from einops import rearrange

from .multiview_encoder import PatchViT, QueryPool
from .lewm_adapter import MultiViewJEPA


class FrontOnlyEncoder(nn.Module):
    def __init__(self, dim=192, num_queries=8, latent_dim=192, view="front"):
        super().__init__()
        self.view = view
        self.vit = PatchViT(dim=dim)
        self.pool = QueryPool(dim, num_queries)
        self.project = nn.Linear(num_queries * dim, latent_dim)
        self.latent_dim = latent_dim

    def encode(self, obs):
        f = obs[self.view]
        B, T = f.shape[:2]
        tok = self.vit(rearrange(f, "b t c h w -> (b t) c h w").float())
        z = self.project(self.pool(tok).flatten(1))
        return rearrange(z, "(b t) d -> b t d", b=B)


class FrontOnlyJEPA(MultiViewJEPA):
    """Same losses/predictor as B1; encoder sees the front camera only."""

    def __init__(self, view="front", **kw):
        super().__init__(**kw)
        self.mv_encoder = FrontOnlyEncoder(view=view)

    def encode(self, obs):
        z = self.mv_encoder.encode(obs)
        B, T, D = z.shape
        return self.projector(z.reshape(B * T, D)).reshape(B, T, D)
