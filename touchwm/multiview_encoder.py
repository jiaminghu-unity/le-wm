"""Multi-view (front + wrist) encoder — proposal §4.1 / §6.1.

- LeWM-style ViT-Tiny (dim 192, patch 14, depth 12, heads 3), trained from scratch,
  returning PATCH tokens (no CLS pooling — cross-view attention needs patch tokens).
  Deviation from official LeWM: the official repo wraps HuggingFace's ViT
  (stable_pretraining vit_hf). We re-implement the same tiny spec with the LeWM
  Block/Attention modules to avoid the HF dependency; both are trained from scratch,
  so no pretrained weights are lost. Logged as deviation D1.
- Shared ViT weights across views + separate learned view embeddings.
- One bidirectional cross-attention block per direction (4 heads, MLP 4x), computed
  from PRE-UPDATE features in both directions (order independence).
- 8 learned shared-state queries read [F'; W'] -> flatten -> project to latent_dim.
"""
import torch
from torch import nn
from einops import rearrange

from .lewm_modules import Block, FeedForward


class PatchViT(nn.Module):
    """ViT-Tiny over 224x224, patch 14 -> 16x16=256 patch tokens, dim 192."""

    def __init__(self, image_size=224, patch_size=14, dim=192, depth=12, heads=3,
                 mlp_ratio=4, dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.blocks = nn.ModuleList([
            Block(dim, heads, dim // heads, dim * mlp_ratio, dropout, causal=False)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (N, 3, H, W) -> (N, P, dim) patch tokens."""
        x = self.patch_embed(x)                      # (N, dim, h, w)
        x = rearrange(x, "n d h w -> n (h w) d")
        x = x + self.pos_embedding
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class CrossAttnBlock(nn.Module):
    """Pre-norm cross attention (Q=x, K/V=y) + FFN, residual."""

    def __init__(self, dim=192, heads=4, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(dim, dim * mlp_ratio, dropout)

    def forward(self, x, y):
        q = self.norm_q(x); kv = self.norm_kv(y)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + out
        x = x + self.ffn(x)
        return x


class QueryPool(nn.Module):
    """8 learned queries cross-attend to the concatenated conditioned tokens."""

    def __init__(self, dim=192, num_queries=8, heads=4, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.block = CrossAttnBlock(dim, heads, 4, dropout)

    def forward(self, tokens):
        q = self.queries.expand(tokens.size(0), -1, -1)
        return self.block(q, tokens)                 # (N, num_queries, dim)


class MultiViewEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=14, dim=192, vit_depth=12,
                 vit_heads=3, ca_heads=4, ca_layers=1, num_queries=8,
                 latent_dim=192, dropout=0.0, wrist_skip=False):
        super().__init__()
        self.vit = PatchViT(image_size, patch_size, dim, vit_depth, vit_heads,
                            dropout=dropout)        # shared weights across views
        self.view_emb = nn.Parameter(torch.randn(2, 1, 1, dim) * 0.02)
        self.ca_f = nn.ModuleList([CrossAttnBlock(dim, ca_heads, 4, dropout)
                                   for _ in range(ca_layers)])
        self.ca_w = nn.ModuleList([CrossAttnBlock(dim, ca_heads, 4, dropout)
                                   for _ in range(ca_layers)])
        self.pool = QueryPool(dim, num_queries, ca_heads, dropout)
        self.wrist_skip = wrist_skip
        in_dim = num_queries * dim + (dim if wrist_skip else 0)
        self.project = nn.Linear(in_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, front, wrist):
        """front/wrist: (N, 3, H, W) -> z: (N, latent_dim)."""
        F_tok = self.vit(front) + self.view_emb[0]
        W_tok = self.vit(wrist) + self.view_emb[1]
        Fp, Wp = F_tok, W_tok
        for caf, caw in zip(self.ca_f, self.ca_w):
            # pre-update features on both sides (order independence)
            Fp2 = caf(Fp, W_tok)
            Wp2 = caw(Wp, F_tok)
            Fp, Wp = Fp2, Wp2
        U = self.pool(torch.cat([Fp, Wp], dim=1))    # (N, Q, dim)
        flat = U.flatten(1)
        if self.wrist_skip:                           # R1/R2: 腕 token 均值直通,
            flat = torch.cat([flat, Wp.mean(1)], 1)   # 接触信息按构造进入 latent
        return self.project(flat)                     # (N, latent_dim)

    def encode(self, obs):
        """obs: {'front': (B,T,3,H,W), 'wrist': (B,T,3,H,W)} -> (B,T,D)."""
        f, w = obs["front"], obs["wrist"]
        B, T = f.shape[:2]
        z = self.forward(rearrange(f, "b t c h w -> (b t) c h w").float(),
                         rearrange(w, "b t c h w -> (b t) c h w").float())
        return rearrange(z, "(b t) d -> b t d", b=B)
