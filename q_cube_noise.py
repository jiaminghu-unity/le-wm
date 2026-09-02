"""Stress-test q variant: the 22-d cube full-config q padded with 78 iid standard
Gaussian dims (total 100). The noise is resampled at every data-load, so the
padded dims carry zero mutual information with anything -- the L_obj alignment
target becomes  sum_22 dq_k^2 + chi2-noise(78),  probing how much irrelevant-dim
dilution the Pearson objective tolerates before the metric signal drowns.

Registered at runtime by train_qnative.py; utils.Q_VARIANTS untouched.
"""

import torch

from q_cube_full import Q_VARIANTS_CUBE_FULL

_build22, _COLS, _UNIT = Q_VARIANTS_CUBE_FULL["cube_full_config"]

N_NOISE = 78


def build_q_cube_noise100(*cols):
    q22 = _build22(*cols)
    noise = torch.randn(*q22.shape[:-1], N_NOISE, dtype=q22.dtype, device=q22.device)
    return torch.cat([q22, noise], dim=-1)


def build_q_cube_alien48(*cols):
    """22-d cube full-config q + 26-d ALIEN block (real scene trajectories,
    episode-paired at data-build time; stored as the dataset column alien_q)."""
    q22 = _build22(*cols[:-1])
    alien = cols[-1]
    alien = alien.reshape(*alien.shape[:-1], -1)[..., :26]
    return torch.cat([q22, alien.to(q22.dtype)], dim=-1)


Q_VARIANTS_CUBE_NOISE = {
    "cube_full_noise100": (build_q_cube_noise100, list(_COLS), _UNIT),
    "cube_alien48": (build_q_cube_alien48, list(_COLS) + ["alien_q"], _UNIT),
}
