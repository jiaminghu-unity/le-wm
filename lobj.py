"""L_obj: Pearson distance-profile alignment between latent and physical-state geometry.

Applied to the encoder-side embedding Z (the same tensor SIGReg sees) — never
to predictor outputs, so gradients reach the encoder (+projector) only.
"""

import torch


def sample_pairs(episode_id, num_frames, num_pairs, within_frac=0.5, generator=None):
    """Stratified index-pair sampling over flattened (B*T) frame embeddings.

    episode_id: (B,) episode index per batch sample; flattened index = b*T + t
    num_frames: T, frames per sub-trajectory sample
    Returns (i, j) long tensors, i != j guaranteed. First ~within_frac*num_pairs
    pairs are within-sample (same sub-trajectory), the rest cross-episode.
    Cross-episode pairs may come up short if the batch lacks episode diversity.
    """
    device = episode_id.device
    B = episode_id.numel()
    T = num_frames
    N = B * T
    n_within = int(num_pairs * within_frac)
    n_cross = num_pairs - n_within

    # within-sample pairs: uniform over the B * C(T,2) frame pairs in the batch
    frame_pairs = torch.combinations(torch.arange(T, device=device), r=2)
    b = torch.randint(B, (n_within,), device=device, generator=generator)
    p = torch.randint(frame_pairs.size(0), (n_within,), device=device, generator=generator)
    wi = b * T + frame_pairs[p, 0]
    wj = b * T + frame_pairs[p, 1]

    # cross-episode pairs: uniform random with rejection on matching episode_id
    ep_flat = episode_id.repeat_interleave(T)
    ci = wi.new_empty(0)
    cj = wj.new_empty(0)
    for _ in range(10):
        if ci.numel() >= n_cross:
            break
        m = 2 * (n_cross - ci.numel())
        a = torch.randint(N, (m,), device=device, generator=generator)
        c = torch.randint(N, (m,), device=device, generator=generator)
        keep = ep_flat[a] != ep_flat[c]
        ci = torch.cat([ci, a[keep]])
        cj = torch.cat([cj, c[keep]])
    ci, cj = ci[:n_cross], cj[:n_cross]

    return torch.cat([wi, ci]), torch.cat([wj, cj])


def obj_loss(Z, Q, episode_id, num_pairs=4096, within_frac=0.5, eps=1e-6, generator=None):
    """1 - Pearson(squared latent distances, squared pose distances) over sampled pairs.

    Z: (B, T, D) encoder-side embeddings
    Q: (B, T, 6) standardized pose vectors (constant w.r.t. parameters)
    episode_id: (B,)
    Returns (loss, rho, skipped): rho is the detached Pearson value; skipped is
    True when the degenerate-batch guard fired (loss is a connected zero).
    """
    B, T, D = Z.shape
    z = Z.reshape(B * T, D).float()
    q = Q.reshape(B * T, -1).detach().float()

    i, j = sample_pairs(episode_id, T, num_pairs, within_frac, generator)
    x = (z[i] - z[j]).pow(2).sum(-1)  # squared L2 — no sqrt (gradient blows up at 0)
    y = (q[i] - q[j]).pow(2).sum(-1)

    if x.std() < eps or y.std() < eps:
        # connected zero keeps the graph valid for DDP/backward bookkeeping
        return z.sum() * 0.0, torch.zeros((), device=z.device), True

    # mean/std stay in the graph: scale-invariance must be end to end
    x_t = (x - x.mean()) / (x.std() + eps)
    y_t = (y - y.mean()) / (y.std() + eps)
    rho = (x_t * y_t).mean()
    return 1.0 - rho, rho.detach(), False
