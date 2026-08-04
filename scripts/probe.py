"""Probing suite (instructions §7.2, paper-style physical understanding on Push-T).

Freeze the encoder; fit (a) a linear probe and (b) a 2-layer MLP probe
(hidden 256, ReLU) from the embedding z — the SAME tensor the losses see
(ViT CLS -> projector; Identity projector for the vanilla config) — to each of:
pusher (x,y), block (x,y), block (cos t, sin t).

Targets are standardized with the training-time q_stats artifact so numbers are
comparable across configs. Train/test frames come from DISJOINT episodes.
Reports test MSE and mean per-dim Pearson r per target per probe type.

Usage: python scripts/probe.py --config c1 lewm_c1_s3072/weights_epoch_10.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_pretraining as spt
import stable_worldmodel as swm
from utils import build_q_raw

SPLIT_SEED = 0
TEST_EPISODE_FRAC = 0.1
N_TRAIN_FRAMES = 20000
N_TEST_FRAMES = 5000
ENCODE_BS = 512
MLP_HIDDEN = 256
MLP_EPOCHS = 30
MLP_BS = 1024
MLP_LR = 1e-3
TARGETS = {"pusher_xy": slice(0, 2), "block_xy": slice(2, 4), "block_angle": slice(4, 6)}

# Reacher targets (spec Step 4): joints as cos/sin ONLY (raw qpos is ill-posed —
# unbounded shoulder), finger position as reference, and the qvel probe as the
# non-circularity check (qvel is absent from q in BOTH training variants).
REACHER_TARGETS = {"joints_cossin": slice(0, 4), "finger_xy": slice(4, 6), "qvel": slice(6, 8)}

# Cube targets: the 9-dim q L_obj aligns to / the aux head regresses, split into the
# groups that matter for the task, plus joint_vel as the non-circularity check
# (velocity is absent from q in every cube config, exactly as on Reacher).
CUBE_TARGETS = {"effector_xyz": slice(0, 3), "effector_yaw_cossin": slice(3, 5),
                "gripper_opening": slice(5, 6), "block_xyz": slice(6, 9),
                "joint_vel": slice(9, 15)}


def load_frames(dataset, rows, device, cols=("state",)):
    """Decode pixels + raw physical columns for the given dataset rows."""
    imagenet = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(imagenet["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(imagenet["std"], device=device).view(1, 3, 1, 1)
    pix_list = []
    col_lists = {c: [] for c in cols}
    for i in range(0, len(rows), ENCODE_BS):
        chunk = rows[i : i + ENCODE_BS].tolist()
        batch = dataset.get_row_data(chunk)
        for c in cols:
            col_lists[c].append(np.asarray(batch[c], dtype=np.float32))
        blobs = batch["pixels"].tolist()
        pix = dataset._decode_images(blobs).to(device).float() / 255.0
        pix = (pix - mean) / std
        pix_list.append(pix.cpu())
    return pix_list, {c: np.concatenate(v) for c, v in col_lists.items()}


@torch.no_grad()
def encode(model, pix_list, device):
    zs = []
    for pix in pix_list:
        out = model.encoder(pix.to(device), interpolate_pos_encoding=True)
        z = model.projector(out.last_hidden_state[:, 0])
        zs.append(z.float().cpu())
    return torch.cat(zs)


def pearson_per_dim(pred, target):
    px = pred - pred.mean(0, keepdim=True)
    py = target - target.mean(0, keepdim=True)
    r = (px * py).mean(0) / (px.std(0) * py.std(0) + 1e-8)
    return r.mean().item()


def fit_linear(z_tr, y_tr):
    x = torch.cat([z_tr, torch.ones(len(z_tr), 1)], dim=1)
    lam = 1e-4 * torch.eye(x.size(1))
    w = torch.linalg.solve(x.T @ x + lam, x.T @ y_tr)
    return lambda z: torch.cat([z, torch.ones(len(z), 1)], dim=1) @ w


def fit_mlp(z_tr, y_tr, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(z_tr.size(1), MLP_HIDDEN), nn.ReLU(), nn.Linear(MLP_HIDDEN, y_tr.size(1))
    ).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=MLP_LR)
    z_gpu, y_gpu = z_tr.cuda(), y_tr.cuda()
    for _ in range(MLP_EPOCHS):
        perm = torch.randperm(len(z_gpu), device="cuda")
        for j in range(0, len(perm), MLP_BS):
            idx = perm[j : j + MLP_BS]
            opt.zero_grad()
            loss = (net(z_gpu[idx]) - y_gpu[idx]).pow(2).mean()
            loss.backward()
            opt.step()
    net.eval()

    @torch.no_grad()
    def predict(z):
        return net(z.cuda()).cpu()

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs=2, required=True, metavar=("NAME", "CKPT"))
    ap.add_argument("--env", default="pusht", choices=["pusht", "reacher", "cube"])
    ap.add_argument("--out", default="eval_results/probing.csv")
    args = ap.parse_args()
    name, ckpt = args.config
    device = "cuda"

    if args.env == "pusht":
        dataset = swm.data.load_dataset(
            "pusht_expert_train.lance", keys_to_load=["pixels", "state"]
        )
        cols = ("state",)
        targets = TARGETS
        stats = json.loads(
            Path(
                swm.data.utils.get_cache_dir(sub_folder="datasets"),
                "pusht_expert_train.lance.q_stats.pusht_state.json",
            ).read_text()
        )
        q_mean = torch.tensor(stats["mean"])
        q_std = torch.tensor(stats["std"])
    elif args.env == "cube":
        CUBE_COLS = ("proprio_effector_pos", "proprio_effector_yaw",
                     "proprio_gripper_opening", "privileged_block_0_pos",
                     "proprio_joint_vel")
        dataset = swm.data.load_dataset(
            "ogbench/cube_single_expert.lance", keys_to_load=["pixels", *CUBE_COLS]
        )
        cols = CUBE_COLS
        targets = CUBE_TARGETS
        q_mean = q_std = None  # standardized on the probe train split (shared across configs)
    else:
        dataset = swm.data.load_dataset(
            "reacher.lance", keys_to_load=["pixels", "qpos", "finger_pos", "qvel"]
        )
        cols = ("qpos", "finger_pos", "qvel")
        targets = REACHER_TARGETS
        q_mean = q_std = None  # standardized on the probe train split (shared across configs)

    # episode-disjoint split, deterministic
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    n_test_ep = int(n_ep * TEST_EPISODE_FRAC)
    test_eps = set(perm[:n_test_ep].tolist())
    ep_of_row = np.repeat(np.arange(n_ep), dataset.lengths)
    all_rows = np.arange(len(ep_of_row))
    test_pool = all_rows[np.isin(ep_of_row, list(test_eps))]
    train_pool = all_rows[~np.isin(ep_of_row, list(test_eps))]
    train_rows = np.sort(g.choice(train_pool, N_TRAIN_FRAMES, replace=False))
    test_rows = np.sort(g.choice(test_pool, N_TEST_FRAMES, replace=False))

    model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
    model.requires_grad_(False)

    print(f"[{name}] encoding {len(train_rows)}+{len(test_rows)} frames...", flush=True)
    pix_tr, cols_tr = load_frames(dataset, train_rows, device, cols)
    pix_te, cols_te = load_frames(dataset, test_rows, device, cols)
    z_tr = encode(model, pix_tr, device)
    z_te = encode(model, pix_te, device)

    if args.env == "pusht":
        y_tr = (build_q_raw(torch.from_numpy(cols_tr["state"])) - q_mean) / q_std
        y_te = (build_q_raw(torch.from_numpy(cols_te["state"])) - q_mean) / q_std
    elif args.env == "cube":
        from utils import build_q_cube_effector

        def targets_raw(c):
            # same 9-dim q the loss uses, then joint_vel appended as the control
            q = build_q_cube_effector(
                torch.from_numpy(c["proprio_effector_pos"]),
                torch.from_numpy(c["proprio_effector_yaw"]),
                torch.from_numpy(c["proprio_gripper_opening"]),
                torch.from_numpy(c["privileged_block_0_pos"]),
            )
            return torch.cat([q, torch.from_numpy(c["proprio_joint_vel"])], dim=-1)

        y_tr_raw, y_te_raw = targets_raw(cols_tr), targets_raw(cols_te)
        t_mean, t_std = y_tr_raw.mean(0), y_tr_raw.std(0).clamp_min(1e-8)
        y_tr = (y_tr_raw - t_mean) / t_std
        y_te = (y_te_raw - t_mean) / t_std
    else:
        from utils import build_q_reacher_joints

        def targets_raw(c):
            return torch.cat(
                [build_q_reacher_joints(torch.from_numpy(c["qpos"])),
                 torch.from_numpy(c["finger_pos"]), torch.from_numpy(c["qvel"])], dim=-1
            )

        y_tr_raw, y_te_raw = targets_raw(cols_tr), targets_raw(cols_te)
        # standardize on the (deterministic) probe train split — identical
        # across configs, so MSE is comparable; Pearson r is scale-free anyway
        t_mean, t_std = y_tr_raw.mean(0), y_tr_raw.std(0).clamp_min(1e-8)
        y_tr = (y_tr_raw - t_mean) / t_std
        y_te = (y_te_raw - t_mean) / t_std

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["config", "probe", "target", "test_mse", "pearson_r", "n_train", "n_test"]
        )
        if write_header:
            writer.writeheader()
        for probe_name, fitter in [("linear", fit_linear), ("mlp", fit_mlp)]:
            for tname, sl in targets.items():
                predict = fitter(z_tr, y_tr[:, sl])
                pred = predict(z_te)
                mse = (pred - y_te[:, sl]).pow(2).mean().item()
                r = pearson_per_dim(pred, y_te[:, sl])
                writer.writerow(
                    {
                        "config": name, "probe": probe_name, "target": tname,
                        "test_mse": round(mse, 5), "pearson_r": round(r, 4),
                        "n_train": len(train_rows), "n_test": len(test_rows),
                    }
                )
                print(f"[{name}] {probe_name:6s} {tname:11s} mse={mse:.5f} r={r:.4f}", flush=True)


if __name__ == "__main__":
    main()
