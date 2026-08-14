"""Decide, before spending GPU-days, which alternative planning costs can differ.

Squared L2 expands exactly, and the goal is constant across candidates:

    ||z_hat - g||^2 = ||z_hat||^2 - 2<z_hat, g> + ||g||^2
                      \_ norm _/   \_ align _/   \_ const _/

So ranking candidates by the shipped cost is ranking by (norm + align), ranking by
the dot product -<z_hat, g> drops the norm term exactly, and cosine divides by
||z_hat|| instead of dropping it. This script measures how much each term actually
contributes, so the follow-up sweeps are only run where they can move the ranking:

  * share of Var(cost) attributable to the norm term. Near zero means the dot
    product would reproduce the shipped ranking and the sweep would measure nothing.
  * Kendall tau between the shipped ranking and each alternative, per start. This is
    the direct answer -- tau near 1.0 means the planner keeps the same elites.
  * ||mean(z)|| against the per-dimension spread. A large mean offset makes
    <z_hat, g> dominated by that common component, which would make the dot product
    degenerate rather than informative; then it has to be centred first.
  * per-dimension std spread and participation ratio, i.e. how anisotropic z is.
    The projector's BatchNorm sits on the HIDDEN layer (module.py:229-235) and the
    output is a bare Linear, so nothing forces isotropy -- and near-isotropy would
    also explain why L1 and squared L2 rank almost identically.
  * whether ||z_hat|| shrinks for candidates the predictor is unsure about. MSE
    training without a stop-gradient pulls uncertain predictions toward the mean,
    which shrinks ||z_hat||; since the norm term enters the cost POSITIVELY, that
    would hand a lower cost to exactly the candidates the model understands least.
    Measured without any simulator: dispersion across the candidate set at a start
    stands in for uncertainty.

Model-side only: no environment rollouts, no rendering, so this costs minutes.

    usage: probe_latent_geometry.py <task> <label>:<ckpt> [...] [--starts N] [--cands N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import hdf5plugin  # noqa: F401,E402
import stable_worldmodel as swm  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from stable_worldmodel.world.world import _extract_init_goal  # noqa: E402

from scripts.budget_sweep import ENV_PRESETS  # noqa: E402
from scripts.p4_bottleneck import (  # noqa: E402
    ACTION_BLOCK, CAND_SEED, EPISODES, GOAL_OFFSET, HORIZON,
    encode_frames, imagined_terminal, img_tf,
)

K_ELITE = 30  # CEM T1 keeps 30 of 300


def elite_overlap(c_a, c_b, k=K_ELITE):
    ia, ib = np.argsort(c_a)[:k], np.argsort(c_b)[:k]
    return len(set(ia) & set(ib)) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(ENV_PRESETS))
    ap.add_argument("models", nargs="+", help="label:ckpt")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--cands", type=int, default=300)
    args = ap.parse_args()
    Path("eval_results").mkdir(parents=True, exist_ok=True)

    task, preset = args.task, ENV_PRESETS[args.task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = img_tf()

    episodes = json.loads(Path(EPISODES[task]).read_text())["episodes"][: args.starts]
    ds_kwargs = {"keys_to_load": preset["keys_to_load"]} if preset.get("keys_to_load") else {}
    dataset = swm.data.HDF5Dataset(preset["dataset"], keys_to_cache=preset["process_cols"],
                                   cache_dir=Path(swm.data.utils.get_cache_dir()), **ds_kwargs)
    act = dataset.get_col_data("action")
    scaler = preprocessing.StandardScaler()
    scaler.fit(act[~np.isnan(act).any(axis=1)])
    adim = act.shape[1]

    start_frames, goal_frames = [], []
    for ep in episodes:
        init_state, goal_state, _ = _extract_init_goal(
            dataset, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
        start_frames.append(tf(np.asarray(init_state["pixels"][0]).astype(np.uint8)))
        gk = next((k for k in ("goal", "goal_pixels") if k in goal_state),
                  next((k for k in goal_state if "pixel" in k), None))
        goal_frames.append(tf(np.asarray(goal_state[gk][0]).astype(np.uint8)))
    starts = torch.stack(start_frames)
    goals = torch.stack(goal_frames)

    # the same candidate draw p5 uses, so the two analyses describe one action set
    g = torch.Generator().manual_seed(CAND_SEED)
    cands = torch.randn(args.starts, args.cands, HORIZON, ACTION_BLOCK * adim, generator=g)

    rows = []
    for spec in args.models:
        label, ckpt = spec.split(":", 1)
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z_start = encode_frames(model, starts, device)
        z_goal = encode_frames(model, goals, device)

        zh = []
        for si in range(args.starts):
            zh.append(imagined_terminal(model, z_start[si:si + 1].to(device),
                                        cands[si].to(device), device).cpu())
        Z = torch.stack(zh)                       # (starts, cands, D)
        D = Z.size(-1)

        flat = Z.reshape(-1, D).double()
        mu = flat.mean(0)
        sd = flat.std(0)
        # participation ratio of the per-dim variances: D if isotropic, 1 if a single
        # dimension owns all the variance
        v = (sd ** 2)
        pr = float(v.sum() ** 2 / (v ** 2).sum())
        mean_norm_ratio = float(mu.norm() / sd.mean() / np.sqrt(D))

        taus_dot, taus_cos, taus_l1, taus_wht = [], [], [], []
        ovl_dot, ovl_cos = [], []
        norm_share, corr_disp = [], []
        for si in range(args.starts):
            z = Z[si].double()
            gg = z_goal[si].cpu().double()
            l2 = (z - gg).pow(2).sum(-1).numpy()
            nrm = z.pow(2).sum(-1).numpy()
            algn = (-2 * (z @ gg)).numpy()
            dot = (-(z @ gg)).numpy()
            cos = (-(z @ gg) / z.norm(dim=-1)).numpy()
            l1 = (z - gg).abs().sum(-1).numpy()
            wht = ((z - gg) / sd).pow(2).sum(-1).numpy()

            # exact variance decomposition of the shipped cost across candidates
            # Var(norm + align) = Var(norm) + Var(align) + 2Cov; report the norm term's
            # own share plus its covariance share, which is what makes it matter
            vn, va = np.var(nrm), np.var(algn)
            cv = np.cov(nrm, algn)[0, 1]
            norm_share.append(float((vn + cv) / (vn + va + 2 * cv)))

            taus_dot.append(kendalltau(l2, dot).statistic)
            taus_cos.append(kendalltau(l2, cos).statistic)
            taus_l1.append(kendalltau(l2, l1).statistic)
            taus_wht.append(kendalltau(l2, wht).statistic)
            ovl_dot.append(elite_overlap(l2, dot))
            ovl_cos.append(elite_overlap(l2, cos))

            # shrinkage: does ||z_hat|| fall for candidates far from the candidate-set
            # centroid (a simulator-free stand-in for "the predictor is unsure here")?
            disp = (z - z.mean(0)).pow(2).sum(-1).numpy()
            corr_disp.append(float(np.corrcoef(np.sqrt(nrm), disp)[0, 1]))

        row = {
            "label": label, "D": D,
            "mean_offset_ratio": mean_norm_ratio,
            "sd_min": float(sd.min()), "sd_med": float(sd.median()), "sd_max": float(sd.max()),
            "participation_ratio": pr,
            "norm_term_share": float(np.mean(norm_share)),
            "tau_vs_dot": float(np.mean(taus_dot)), "tau_vs_cos": float(np.mean(taus_cos)),
            "tau_vs_l1": float(np.mean(taus_l1)), "tau_vs_whiten": float(np.mean(taus_wht)),
            "ovl30_dot": float(np.mean(ovl_dot)), "ovl30_cos": float(np.mean(ovl_cos)),
            "corr_norm_dispersion": float(np.mean(corr_disp)),
        }
        rows.append(row)
        print(f"  {label:8s} done", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'=' * 112}\nlatent geometry — {task}  "
          f"({args.starts} starts x {args.cands} candidates, model-side only)\n{'=' * 112}")
    print(f"{'model':8s}{'D':>5s}{'|mu|/sd':>9s}{'sd min':>9s}{'sd med':>9s}{'sd max':>9s}"
          f"{'PR/D':>7s}{'norm%':>8s}")
    for r in rows:
        print(f"{r['label']:8s}{r['D']:5d}{r['mean_offset_ratio']:9.2f}{r['sd_min']:9.4f}"
              f"{r['sd_med']:9.4f}{r['sd_max']:9.4f}{r['participation_ratio'] / r['D']:7.2f}"
              f"{100 * r['norm_term_share']:8.1f}")
    print(f"\n与现有平方 L2 排序的一致性（tau=1 表示排序完全相同，那个变体测不出东西）:")
    print(f"{'model':8s}{'vs dot':>9s}{'vs cos':>9s}{'vs L1':>9s}{'vs whiten':>11s}"
          f"{'ovl@30 dot':>12s}{'ovl@30 cos':>12s}{'corr(|z|,disp)':>16s}")
    for r in rows:
        print(f"{r['label']:8s}{r['tau_vs_dot']:9.3f}{r['tau_vs_cos']:9.3f}{r['tau_vs_l1']:9.3f}"
              f"{r['tau_vs_whiten']:11.3f}{r['ovl30_dot']:12.3f}{r['ovl30_cos']:12.3f}"
              f"{r['corr_norm_dispersion']:16.3f}")

    out = f"eval_results/latgeom_{task}.json"
    Path(out).write_text(json.dumps(
        {"task": task, "starts": args.starts, "cands": args.cands, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
