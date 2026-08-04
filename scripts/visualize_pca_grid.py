"""3x3 PCA panel: top-2 principal components of the encoder embedding z, for the
three finally-selected models on each of the three tasks.

    rows    = Push-T / Reacher / OGBench-Cube
    columns = baseline (SIGReg only) / +L_obj / +aux q-head
    colour  = first principal component of that task's standardized q

Colouring by q-PC1 makes the three rows directly comparable despite q having
different dimensions per task (6 / 4 / 9), and turns the panel into a direct read
of "does the latent's leading 2-D plane organise by physical state?".
Each panel is annotated with the variance explained by PC1/PC2 and with
|Spearman(latent PC1, q-PC1)|.

Works from either lance (JPEG blobs) or raw h5 (uint8 arrays) — Push-T and Reacher
are staged as h5, cube as lance, and converting them just for this figure would
cost an hour for nothing.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from utils import build_q_cube_effector, build_q_raw, build_q_reacher_joints  # noqa: E402

SPLIT_SEED = 0
TEST_EPISODE_FRAC = 0.1
N_WINDOWS = 600      # contiguous runs sampled per task
W = 8                # window length for the latent-norm smoothness panel (A/D)
BS = 250
TEXT = "#3d3d3c"

TASKS = {
    "Push-T": dict(
        dataset="pusht_expert_train.h5", cols=("state",),
        q=lambda c: build_q_raw(torch.from_numpy(c["state"])),
        models={"baseline (C1)": "lewm_c1_s3072",
                "+L_obj λ=0.1 (C3)": "lewm_c3_sig_obj0.1_s3072",
                "+aux w=0.3 (C5)": "lewm_c5_qhead0.3_s3072"}),
    "Reacher": dict(
        dataset="reacher.h5", cols=("qpos",),
        q=lambda c: build_q_reacher_joints(torch.from_numpy(c["qpos"])),
        models={"baseline (R1)": "lewm_r1_reacher_s3072",
                "+L_obj λ=0.15 (R2)": "lewm_r2_reacher_paep_l015_s3072",
                "+aux w=0.4 (R5)": "lewm_r5_qhead0.4_s3072"}),
    "Cube": dict(
        dataset="ogbench/cube_single_expert.lance",
        cols=("proprio_effector_pos", "proprio_effector_yaw",
              "proprio_gripper_opening", "privileged_block_0_pos"),
        q=lambda c: build_q_cube_effector(*[torch.from_numpy(c[k]) for k in
            ("proprio_effector_pos", "proprio_effector_yaw",
             "proprio_gripper_opening", "privileged_block_0_pos")]),
        models={"baseline (K1)": "lewm_k1_cube_s3072",
                "+L_obj λ=0.1 (K2)": "lewm_k2_cube_obj_eff0.1_s3072",
                "+aux w=0.1 (K4)": "lewm_k4_cube_qhead_eff0.1_s3072"}),
}


def load_frames(dataset, rows, cols, device):
    """Format-agnostic: lance stores pixels as JPEG blobs, h5 as raw uint8 arrays."""
    st = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(st["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(st["std"], device=device).view(1, 3, 1, 1)
    pix, out = [], {c: [] for c in cols}
    for i in range(0, len(rows), BS):
        chunk = rows[i : i + BS].tolist()
        batch = dataset.get_row_data(chunk)
        for c in cols:
            a = np.asarray(batch[c], dtype=np.float32)
            out[c].append(a if a.ndim > 1 else a[:, None])
        a = np.asarray(batch["pixels"])
        # Decide by DTYPE, not container: lance hands back an object-dtype array of
        # JPEG blobs (still an ndarray), h5 hands back real uint8 NHWC pixels.
        if a.dtype == object or a.dtype.kind in ("S", "O", "U"):
            t = dataset._decode_images(a.tolist()).to(device).float() / 255.0
        else:
            if a.ndim == 4 and a.shape[-1] == 3:      # NHWC -> NCHW
                a = a.transpose(0, 3, 1, 2)
            t = torch.from_numpy(np.ascontiguousarray(a)).to(device).float() / 255.0
        pix.append(((t - mean) / std).cpu())
    return pix, {c: np.concatenate(v) for c, v in out.items()}


def encode(model, pix_list, device):
    zs = []
    for p in pix_list:
        o = model.encoder(p.to(device), interpolate_pos_encoding=True)
        zs.append(model.projector(o.last_hidden_state[:, 0]).float().cpu())
    return torch.cat(zs).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    figs, axs = {}, {}
    for key in ("qpc1", "progress", "norms"):
        figs[key], axs[key] = plt.subplots(3, 3, figsize=(14.5, 13.5), dpi=150)

    CACHE = {}
    for r, (tname, spec) in enumerate(TASKS.items()):
        ds = swm.data.load_dataset(spec["dataset"], keys_to_load=["pixels", *spec["cols"]])
        n_ep = len(ds.lengths)
        g = np.random.default_rng(SPLIT_SEED)
        perm = g.permutation(n_ep)
        test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
        lengths, offsets = np.asarray(ds.lengths), np.asarray(ds.offsets)
        # Sample CONTIGUOUS runs of W frames: the norm panel needs consecutive
        # frames (it measures short-timescale latent jitter), and the same frames
        # serve the PCA scatter, so one data pass covers both.
        elig = [e for e in test_eps if lengths[e] > W]
        picks = g.choice(elig, N_WINDOWS, replace=len(elig) < N_WINDOWS)
        starts = np.array([g.integers(0, lengths[e] - W + 1) for e in picks])
        g0 = offsets[picks] + starts
        # h5py fancy indexing demands strictly increasing indices: sort windows by
        # global start and drop any that overlaps its predecessor, which keeps the
        # flattened row list monotone while preserving the (n_win, W) grouping.
        srt = np.argsort(g0)
        kept, last = [], -1
        for i in srt:
            if g0[i] > last:
                kept.append(g0[i]); last = g0[i] + W - 1
        n_win = len(kept)
        rows = np.concatenate([k + np.arange(W) for k in kept])
        pix, cols = load_frames(ds, rows, spec["cols"], device)

        # normalized within-episode progress for each sampled frame
        ep_of_row = np.searchsorted(offsets, rows, side="right") - 1
        progress = (rows - offsets[ep_of_row]) / np.maximum(lengths[ep_of_row] - 1, 1)

        q = spec["q"](cols).numpy()
        qs = (q - q.mean(0)) / q.std(0).clip(1e-8)
        qc = qs - qs.mean(0)
        qpc1 = qc @ np.linalg.svd(qc, full_matrices=False)[2][0]
        print(f"[{tname}] windows={n_win} frames={len(rows)} q_dim={q.shape[1]}", flush=True)

        for c, (label, ckpt) in enumerate(spec["models"].items()):
            m = swm.wm.utils.load_pretrained(f"{ckpt}/weights_epoch_10.pt").to(device).eval()
            m.requires_grad_(False)
            z = encode(m, pix, device)
            del m
            if device == "cuda":
                torch.cuda.empty_cache()

            zc = z - z.mean(0)
            U, S, _ = np.linalg.svd(zc, full_matrices=False)
            pcs = U[:, :2] * S[:2]
            ev = (S ** 2) / (S ** 2).sum()
            rho = max(abs(spearmanr(pcs[:, 0], qpc1).statistic),
                      abs(spearmanr(pcs[:, 1], qpc1).statistic))

            rho_p = max(abs(spearmanr(pcs[:, 0], progress).statistic),
                        abs(spearmanr(pcs[:, 1], progress).statistic))
            for key, col, cmap, cl, extra in [
                    ("qpc1", qpc1, "viridis", "q-PC1", f"|ρ(PC, q-PC1)| = {rho:.2f}"),
                    ("progress", progress, "coolwarm", "task progress",
                     f"|ρ(PC, progress)| = {rho_p:.2f}")]:
                ax = axs[key][r, c]
                sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=col, s=3, cmap=cmap,
                                alpha=0.75, linewidths=0)
                ax.set_title(f"{tname} — {label}", fontsize=9, color=TEXT)
                ax.set_xlabel(f"PC 1 ({100*ev[0]:.1f}%)", fontsize=8, color=TEXT)
                ax.set_ylabel(f"PC 2 ({100*ev[1]:.1f}%)", fontsize=8, color=TEXT)
                ax.tick_params(labelsize=6)
                ax.text(0.03, 0.97, f"{extra}\nPC1+PC2 = {100*(ev[0]+ev[1]):.1f}%",
                        transform=ax.transAxes, va="top", fontsize=7, color=TEXT)
                if c == 2:
                    figs[key].colorbar(sc, ax=ax, fraction=0.046, label=cl)

            # --- A/D-style panel: frame vs window-mean latent norms ---
            fn = np.linalg.norm(z, axis=1)
            zw = z.reshape(n_win, W, -1).mean(1)
            wn = np.linalg.norm(zw, axis=1)
            ref = fn.mean()
            axn = axs["norms"][r, c]
            axn.hist(fn / ref, bins=60, density=True, alpha=0.65, color="#eb6834",
                     label="$\\|z\\|$ (frames)")
            axn.hist(wn / ref, bins=60, density=True, alpha=0.65, color="#4a4a4a",
                     label=f"$\\|\\bar z_{{win}}\\|$ (W={W})")
            axn.set_title(f"{tname} — {label}", fontsize=9, color=TEXT)
            axn.set_xlabel("norm / mean frame norm", fontsize=8, color=TEXT)
            axn.set_ylabel("density", fontsize=8, color=TEXT)
            axn.tick_params(labelsize=6)
            # ratio of the two means: 1.0 = perfectly smooth in time, lower = more jitter
            axn.text(0.03, 0.97, f"mean ratio = {wn.mean()/ref:.3f}",
                     transform=axn.transAxes, va="top", fontsize=7, color=TEXT)
            if c == 0:
                axn.legend(frameon=False, fontsize=6.5, labelcolor=TEXT)
            print(f"   {label:22s} window/frame norm ratio = {wn.mean()/ref:.4f}", flush=True)
            CACHE[f"{tname}|{label}|z"] = z.astype(np.float32)
            CACHE[f"{tname}|qpc1"] = qpc1.astype(np.float32)
            CACHE[f"{tname}|progress"] = progress.astype(np.float32)
            print(f"   {label:22s} PC1={100*ev[0]:5.1f}% PC2={100*ev[1]:5.1f}% "
                  f"rho_q={rho:.3f} rho_progress={rho_p:.3f}", flush=True)
        del pix

    Path("eval_results").mkdir(exist_ok=True)
    np.savez_compressed("eval_results/pca_cache.npz", **CACHE)
    print("wrote eval_results/pca_cache.npz  (z / q-PC1 / progress per task+model)")
    for key, title, fname in [
            ("qpc1", "Top-2 PCA of encoder embedding z — colour = first PC of the task's q",
             "viz_pca_grid_q.png"),
            ("progress", "Top-2 PCA of encoder embedding z — colour = normalized task progress",
             "viz_pca_grid_progress.png"),
            ("norms", f"Latent-norm distributions — frames vs {W}-frame window means "
                      "(both / mean frame norm)", "viz_latent_norms.png")]:
        figs[key].suptitle(title, fontsize=11, color=TEXT)
        out = Path("eval_results") / fname
        figs[key].savefig(out, facecolor="white", bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
