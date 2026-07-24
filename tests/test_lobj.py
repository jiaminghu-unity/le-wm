import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobj import obj_loss, sample_pairs  # noqa: E402


def test_proportional_z_gives_zero_loss():
    # Z an exact scaling of Q -> squared distances perfectly correlated -> loss ~ 0
    g = torch.Generator().manual_seed(0)
    B, T = 32, 4
    Q = torch.randn(B, T, 6, generator=g)
    Z = 3.7 * Q
    loss, rho, skipped = obj_loss(Z, Q, torch.arange(B), num_pairs=2048, generator=g)
    assert not skipped
    assert loss.item() < 1e-3
    assert rho.item() > 0.999


def test_shuffled_z_gives_loss_near_one():
    g = torch.Generator().manual_seed(0)
    B, T = 32, 4
    Q = torch.randn(B, T, 6, generator=g)
    perm = torch.randperm(B * T, generator=g)
    Z = Q.reshape(B * T, 6)[perm].reshape(B, T, 6)
    loss, rho, skipped = obj_loss(Z, Q, torch.arange(B), num_pairs=4096, generator=g)
    assert not skipped
    assert abs(loss.item() - 1.0) < 0.2


def test_constant_z_triggers_guard():
    B, T = 8, 4
    Z = torch.ones(B, T, 16, requires_grad=True)
    Q = torch.randn(B, T, 6)
    loss, rho, skipped = obj_loss(Z, Q, torch.arange(B))
    assert skipped
    assert loss.item() == 0.0
    loss.backward()  # connected zero: graph must stay valid
    assert Z.grad is not None


def test_pair_sampling_stratified_no_self_pairs():
    g = torch.Generator().manual_seed(0)
    B, T, K = 16, 4, 1000
    # duplicate episode ids across samples: cross-episode sampling must reject those
    ep = torch.arange(B) // 2
    i, j = sample_pairs(ep, T, K, generator=g)
    assert (i != j).all()

    n_within = K // 2
    same_sample = (i // T) == (j // T)
    assert same_sample[:n_within].all()  # first half within-sample by construction

    ep_flat = ep.repeat_interleave(T)
    assert (ep_flat[i[n_within:]] != ep_flat[j[n_within:]]).all()

    frac_within = same_sample.float().mean().item()
    assert abs(frac_within - 0.5) <= 0.02


def test_reacher_joint_q_wraps_correctly():
    from utils import build_q_reacher_joints, build_q_reacher_joints_finger

    qpos = torch.tensor([[0.0, torch.pi / 2]])
    q = build_q_reacher_joints(qpos)
    assert torch.allclose(q, torch.tensor([[1.0, 0.0, 0.0, 1.0]]), atol=1e-6)

    # unbounded shoulder: theta and theta + 2pi must map to the same q
    a = build_q_reacher_joints(torch.tensor([[1.3, 0.4]]))
    b = build_q_reacher_joints(torch.tensor([[1.3 + 2 * torch.pi, 0.4]]))
    assert torch.allclose(a, b, atol=1e-5)

    # finger enters raw (positions are not periodic)
    finger = torch.tensor([[0.12, -0.05]])
    q6 = build_q_reacher_joints_finger(qpos, finger)
    assert q6.shape == (1, 6)
    assert torch.allclose(q6[:, 4:], finger)


def test_q_normalizer_dict_transform():
    from utils import QNormalizer, build_q_reacher_joints

    sample = {"qpos": torch.zeros(4, 2), "other": 1}
    tf = QNormalizer(build_q_reacher_joints, ["qpos"],
                     mean=torch.zeros(4), std=torch.ones(4))
    out = tf(sample)
    assert out["q"].shape == (4, 4)
    assert out["other"] == 1  # dict passthrough


def test_pair_sampling_single_episode_degrades_gracefully():
    # cross-episode pairs impossible -> only within pairs come back, no hang
    g = torch.Generator().manual_seed(0)
    B, T, K = 8, 4, 100
    ep = torch.zeros(B, dtype=torch.long)
    i, j = sample_pairs(ep, T, K, generator=g)
    assert i.numel() == K // 2
    assert (i != j).all()
    assert ((i // T) == (j // T)).all()
