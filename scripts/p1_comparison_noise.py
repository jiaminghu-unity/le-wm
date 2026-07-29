"""P1: comparative-noise measurement (the true denominator of the planner's SNR).

From each of S start states, sample N candidate action sequences. Each candidate
gets (a) a model-imagined cost and (b) a ground-truth cost: execute the same
actions in the simulator, encode the resulting final frame with the SAME model,
and measure distance to the goal embedding. Per-candidate cost error
eps = cost_model - cost_true. Report per model (costs normalized by the model's
pairwise-distance scale):

  (i)   marginal noise:      sigma = std(eps)
  (ii)  common-mode share:   rho_corr = Var_s(mean_i eps) / Var_total (ICC)
  (iii) comparison noise:    sqrt(2 * sigma^2 * (1 - rho_corr))

Pre-registration: C5's (iii) < C1's despite (i) being equal.
Environment rollouts are model-independent and shared across models.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402

MODELS = {
    "c1": "lewm_c1_s3072/weights_epoch_10.pt",
    "c3": "lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt",
    "c5_w03": "lewm_c5_qhead0.3_s3072/weights_epoch_10.pt",
}
N_STARTS = 20
N_CAND = 64
HORIZON = 5
ACTION_BLOCK = 5
GOAL_OFFSET = 25
CALLABLES = [
    {"method": "_set_state", "args": {"state": {"value": "state"}}},
    {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
]


def img_tf():
    return transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(224)])


@torch.no_grad()
def encode_frames(model, frames, device, bs=128):
    zs = []
    for i in range(0, len(frames), bs):
        x = frames[i:i + bs].to(device)
        out = model.encoder(x, interpolate_pos_encoding=True)
        zs.append(model.projector(out.last_hidden_state[:, 0]).float().cpu())
    return torch.cat(zs)


@torch.no_grad()
def imagined_terminal(model, z0, act_blocks, device):
    """Roll the predictor over HORIZON action blocks starting from a single frame,
    mirroring JEPA.rollout mechanics (context grows to 3)."""
    S = act_blocks.size(0)
    ctx = z0.expand(S, 1, -1).clone().to(device)
    for k in range(HORIZON):
        c = ctx[:, -3:]
        blocks = act_blocks[:, max(0, k + 1 - c.size(1)): k + 1][:, -c.size(1):]
        pred = model.predict(c, model.action_encoder(blocks))[:, -1:]
        ctx = torch.cat([ctx, pred], dim=1)
    return ctx[:, -1]


def main():
    device = "cuda"
    tf = img_tf()
    eps_path = Path("scripts/episodes_pusht_50.json")
    episodes = json.loads(eps_path.read_text())["episodes"][:N_STARTS]

    dataset = swm.data.HDF5Dataset(
        "pusht_expert_train", keys_to_cache=["action", "proprio", "state"],
        cache_dir=Path(swm.data.utils.get_cache_dir()))
    scaler = preprocessing.StandardScaler()
    act = dataset.get_col_data("action")
    scaler.fit(act[~np.isnan(act).any(axis=1)])

    world = swm.World(env_name="swm/PushT-v1", num_envs=1, image_shape=(224, 224),
                      max_episode_steps=10_000)
    g = torch.Generator().manual_seed(7)
    cands = torch.randn(N_STARTS, N_CAND, HORIZON, ACTION_BLOCK * 2, generator=g)

    start_frames, goal_frames, final_frames = [], [], []  # final: (S, N, C,H,W)
    for si, ep in enumerate(episodes):
        init_state, goal_state, _ = _extract_init_goal(
            dataset, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
        start_frames.append(tf(init_state["pixels"][0].astype(np.uint8)))   # already HWC
        goal_frames.append(tf(goal_state["goal"][0].astype(np.uint8)))
        merged = {**init_state, **goal_state}
        env_init = {k: v[0] for k, v in merged.items()}
        per_cand = []
        for ci in range(N_CAND):
            world.reset(seed=[ep["env_seed"]])
            _apply_callables(world.envs.envs[0].unwrapped, CALLABLES, env_init)
            raw = scaler.inverse_transform(
                cands[si, ci].reshape(HORIZON * ACTION_BLOCK, 2).numpy())
            for a in raw:
                world.envs.step(a[None].astype(np.float32))
            per_cand.append(tf(np.asarray(world.infos["pixels"][0][-1] if world.infos["pixels"][0].ndim > 3
                                          else world.infos["pixels"][0])))
        final_frames.append(torch.stack(per_cand))
        print(f"env rollouts: start {si+1}/{N_STARTS} done", flush=True)
    world.close()

    starts = torch.stack(start_frames)
    goals = torch.stack(goal_frames)
    finals = torch.stack(final_frames)  # (S, N, C, H, W)

    print(f"\n{'model':8s} {'(i) sigma':>10s} {'(ii) rho':>10s} {'(iii) cmp-noise':>16s}")
    for name, ckpt in MODELS.items():
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z_start = encode_frames(model, starts, device)
        z_goal = encode_frames(model, goals, device)
        z_final = encode_frames(model, finals.reshape(-1, *finals.shape[2:]), device)
        z_final = z_final.reshape(N_STARTS, N_CAND, -1)

        flat = z_final.reshape(-1, z_final.size(-1))
        ii = torch.randint(0, flat.size(0), (20000,)); jj = torch.randint(0, flat.size(0), (20000,))
        keep = ii != jj
        scale = (flat[ii[keep]] - flat[jj[keep]]).pow(2).sum(-1).mean().item()

        eps = np.zeros((N_STARTS, N_CAND))
        z_hats = torch.zeros_like(z_final)
        for si in range(N_STARTS):
            blocks = cands[si].to(device)
            z_hat = imagined_terminal(model, z_start[si:si+1].to(device), blocks, device).cpu()
            z_hats[si] = z_hat
            cost_model = (z_hat - z_goal[si]).pow(2).sum(-1).numpy() / scale
            cost_true = (z_final[si] - z_goal[si]).pow(2).sum(-1).numpy() / scale
            eps[si] = cost_model - cost_true

        sigma = eps.std()
        var_common = eps.mean(axis=1).var()
        rho = var_common / eps.var()
        cmp_noise = np.sqrt(2 * sigma**2 * (1 - rho))
        print(f"{name:8s} {sigma:10.4f} {rho:10.3f} {cmp_noise:16.4f}")
        np.savez(f"eval_results/p1_cache_{name}.npz", eps=eps,
                 z_hat=z_hats.numpy(), z_final=z_final.numpy(),
                 z_goal=z_goal.numpy(), z_start=z_start.numpy(), scale=scale)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
