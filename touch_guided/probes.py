"""Stage 2/3 probes (proposal §7, §8.1): small MLP regressors with episode-level
cross-fitting. A transition is never scored by a probe trained on its own episode.
"""
import numpy as np
import torch
import torch.nn as nn


class ProbeMLP(nn.Module):
    """2 hidden layers, width 256, GELU, LayerNorm — fixed across probe variants so
    nuisance-only and vision-conditioned probes have comparable capacity."""

    def __init__(self, d_in, d_out, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, d_out),
        )

    def forward(self, x):
        return self.net(x)


def kfold_episodes(ep_ids: np.ndarray, n_folds: int, seed: int):
    """Yield (fold, train_mask, test_mask) over transitions, split by EPISODE."""
    eps = np.unique(ep_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(eps)
    folds = np.array_split(order, n_folds)
    for i, test_eps in enumerate(folds):
        test = np.isin(ep_ids, test_eps)
        yield i, ~test, test


def crossfit_regress(X, Y, ep_ids, n_folds=5, seed=0, steps=3000, batch=1024,
                     lr=1e-3, wd=1e-4, device="cpu", log=print):
    """Train one probe per fold on the fold-train episodes, predict the held-out
    episodes. Inputs standardized with fold-train statistics. Returns per-sample
    squared error ||y - yhat||^2 (and the predictions)."""
    N, d_out = Y.shape
    err = np.full(N, np.nan, np.float32)
    pred = np.full((N, d_out), np.nan, np.float32)
    for fold, tr, te in kfold_episodes(ep_ids, n_folds, seed):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        torch.manual_seed(seed * 100 + fold)
        net = ProbeMLP(X.shape[1], d_out).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        Xt = torch.from_numpy(Xn[tr]).float().to(device)
        Yt = torch.from_numpy(Y[tr]).float().to(device)
        n_tr = Xt.shape[0]
        g = torch.Generator().manual_seed(seed * 1000 + fold)
        for s in range(steps):
            idx = torch.randint(0, n_tr, (min(batch, n_tr),), generator=g)
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            yh = net(torch.from_numpy(Xn[te]).float().to(device)).cpu().numpy()
        pred[te] = yh
        err[te] = ((Y[te] - yh) ** 2).sum(1)
        log(f"  fold {fold}: n_test={te.sum()} mse={err[te].mean():.4f}")
    assert not np.isnan(err).any()
    return err, pred


def ridge_crossfit(X, Y, ep_ids, n_folds=5, seed=0, lam=1000.0):
    """Closed-form ridge with episode-level cross-fitting. Primary scorer for the
    probe stages: with 181 train episodes, the 512-d-input MLP overfits across
    episodes (documented deviation from the proposal's MLP-only recipe); ridge is
    deterministic and better-behaved. Returns per-sample squared error."""
    Y = Y.reshape(len(Y), -1)
    err = np.full(len(X), np.nan, np.float32)
    for _, tr, te in kfold_episodes(ep_ids, n_folds, seed):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-3
        Xn = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
        W = np.linalg.solve(Xn[tr].T @ Xn[tr] + lam * np.eye(Xn.shape[1]), Xn[tr].T @ Y[tr])
        err[te] = ((Y[te] - Xn[te] @ W) ** 2).sum(1)
    assert not np.isnan(err).any()
    return err
