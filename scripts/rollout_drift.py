"""Open-loop rollout drift diagnostic (Push-T, held-out windows).

For each model: encode 8 true frames (frameskip 5: 3 context + 5 rollout),
roll the predictor autoregressively with TRUE expert actions, and measure
||z_hat_k - z_k||^2 at each horizon step. Errors are reported normalized by
the model's own mean pairwise distance (the 'signal scale'), giving the
noise-to-signal ratio the planner actually faces when ranking candidates.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, encode, load_frames  # noqa: E402

MODELS = {
    "c1": "lewm_c1_s3072/weights_epoch_10.pt",
    "c3": "lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt",
    "c5_w02": "lewm_c5_qhead0.2_s3072/weights_epoch_10.pt",
    "c6": "lewm_c6_combo_s3072/weights_epoch_10.pt",
}
M = 256          # windows
FRAMESKIP = 5
CTX = 3
K = 5            # open-loop steps
SPAN = (CTX + K) * FRAMESKIP + 1


@torch.no_grad()
def main():
    device = "cuda"
    dataset = swm.data.load_dataset(
        "pusht_expert_train.lance", keys_to_load=["pixels", "state", "action"]
    )
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)

    # action normalization stats, same convention as training
    act_col = np.asarray(dataset.get_col_data("action"), dtype=np.float64)
    ok = ~np.isnan(act_col).any(axis=1)
    a_mean, a_std = act_col[ok].mean(0), act_col[ok].std(0)

    valid = test_eps[lengths[test_eps] > SPAN + 1]
    eps = g.choice(valid, M, replace=True)
    starts = g.integers(0, lengths[eps] - SPAN - 1)

    # frame rows at frameskip; action blocks between consecutive frames
    frame_rows = np.concatenate(
        [offsets[e] + s + np.arange(CTX + K) * FRAMESKIP for e, s in zip(eps, starts)]
    )
    uniq, inverse = np.unique(frame_rows, return_inverse=True)
    pix, _ = load_frames(dataset, uniq, device)

    act_blocks = np.zeros((M, CTX + K - 1, FRAMESKIP * 2), dtype=np.float32)
    for m, (e, s) in enumerate(zip(eps, starts)):
        base = offsets[e] + s
        for t in range(CTX + K - 1):
            rows = act_col[base + t * FRAMESKIP: base + (t + 1) * FRAMESKIP]
            rows = (np.nan_to_num(rows) - a_mean) / a_std
            act_blocks[m, t] = rows.reshape(-1)
    act_blocks = torch.from_numpy(act_blocks).float().to(device)

    print(f"{'model':8s} {'1step_tf':>9s} " + "".join(f"{'ol_k'+str(k):>9s}" for k in range(1, K + 1))
          + f" {'scale':>9s}  (errors /scale = noise-to-signal)")
    for name, ckpt in MODELS.items():
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z = encode(model, pix, device)[torch.from_numpy(inverse)].reshape(M, CTX + K, -1).to(device)

        scale = 0.0  # mean pairwise squared distance among all encoded frames (signal scale)
        flat = z.reshape(-1, z.size(-1))
        ii = torch.randint(0, flat.size(0), (20000,), device=device)
        jj = torch.randint(0, flat.size(0), (20000,), device=device)
        keep = ii != jj
        scale = (flat[ii[keep]] - flat[jj[keep]]).pow(2).sum(-1).mean().item()

        # teacher-forced 1-step
        pred_tf = model.predict(z[:, :CTX], model.action_encoder(act_blocks[:, :CTX]))
        err_tf = (pred_tf[:, -1] - z[:, CTX]).pow(2).sum(-1).mean().item()

        # open-loop autoregressive
        ctx = z[:, :CTX].clone()
        errs = []
        for k in range(K):
            blocks = act_blocks[:, k: k + CTX]
            pred = model.predict(ctx, model.action_encoder(blocks))[:, -1:]
            errs.append((pred[:, 0] - z[:, CTX + k]).pow(2).sum(-1).mean().item())
            ctx = torch.cat([ctx[:, 1:], pred], dim=1)

        print(f"{name:8s} {err_tf/scale:9.4f} " + "".join(f"{e/scale:9.4f}" for e in errs)
              + f" {scale:9.2f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
