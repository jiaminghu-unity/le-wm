"""Stage 1: tactile consequence descriptors (proposal §6).

For every valid within-episode transition (t, t+1):

    X_t = [vec(T_t), vec(T_{t+1} - T_t)]   (10*14*3 * 2 = 840 dims)

z-scored per coordinate with statistics fitted on TRAIN episodes only, then PCA
(train-episode fit) to d_tau dims. The current-state-only descriptor used by the
offline control-value probe (§8.1) is produced the same way from vec(T_t) alone.

Episode-level split everywhere: never split by frames (§21).
"""
import json
from pathlib import Path

import numpy as np
import zarr
from sklearn.decomposition import PCA


def make_episode_split(n_episodes: int, train_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_episodes)
    n_train = int(round(train_ratio * n_episodes))
    return np.sort(order[:n_train]).tolist(), np.sort(order[n_train:]).tolist()


def episode_table(episode_ends: np.ndarray):
    """row ranges per episode: returns (offsets, lengths)."""
    offsets = np.concatenate([[0], episode_ends[:-1]])
    lengths = episode_ends - offsets
    return offsets, lengths


def valid_transitions(episode_ends: np.ndarray):
    """Global row ids t such that t and t+1 are in the same episode.

    Returns (rows, ep_ids, progress) where progress = step/(len-1)."""
    offsets, lengths = episode_table(episode_ends)
    rows, eps, prog = [], [], []
    for e, (o, L) in enumerate(zip(offsets, lengths)):
        idx = np.arange(o, o + L - 1)
        rows.append(idx)
        eps.append(np.full(L - 1, e))
        prog.append((idx - o) / max(L - 1, 1))
    return np.concatenate(rows), np.concatenate(eps), np.concatenate(prog)


class Standardizer:
    def __init__(self, mean, std, eps=1e-6):
        self.mean, self.std, self.eps = mean, std, eps

    def __call__(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}


def fit_descriptors(tacff: np.ndarray, rows, ep_ids, train_eps, d_tau: int, seed: int):
    """Returns (r_tau [N,d_tau], r_cur [N,d_cur], fit artifacts dict)."""
    T_t = tacff[rows].reshape(len(rows), -1)
    dT = tacff[rows + 1].reshape(len(rows), -1) - T_t
    X = np.concatenate([T_t, dT], axis=1).astype(np.float64)

    train_mask = np.isin(ep_ids, train_eps)
    std_full = Standardizer(X[train_mask].mean(0), X[train_mask].std(0))
    pca_full = PCA(n_components=d_tau, random_state=seed).fit(std_full(X[train_mask]))
    r_tau = pca_full.transform(std_full(X)).astype(np.float32)

    std_cur = Standardizer(T_t[train_mask].mean(0), T_t[train_mask].std(0))
    pca_cur = PCA(n_components=d_tau, random_state=seed).fit(std_cur(T_t[train_mask]))
    r_cur = pca_cur.transform(std_cur(T_t)).astype(np.float32)

    fit = {
        "norm_full": std_full.to_dict(), "norm_cur": std_cur.to_dict(),
        "evr_full": pca_full.explained_variance_ratio_.tolist(),
        "evr_cur": pca_cur.explained_variance_ratio_.tolist(),
    }
    return r_tau, r_cur, (pca_full, pca_cur), fit
