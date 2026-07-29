"""P4: linear probes on the predictor's internal layers.

Teacher-forced prediction on held-out windows (3 context frames + 1 target,
frameskip 5). Hook every transformer block in the predictor and probe the
last-token hidden state for physics of the PREDICTED step:

  block_speed   ||block_xy(t+1) - block_xy(t)||       (did/如何 the block move)
  block_angvel  angle change via cos/sin difference    (rotation dynamics)
  pusher_speed  ||pusher_xy(t+1) - pusher_xy(t)||      (action-driven, control)
  contact_dist  pusher-block distance at target frame  (contact geometry)
  block_xy      block position at target frame (2d)    (pose decodability)

Ridge probes, half fit / half held-out R^2. Layer 0 = projected input, then
one row per transformer block. Pre-registration: C5's intermediates carry more
contact/velocity signal than C1's.
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
    "c5_w03": "lewm_c5_qhead0.3_s3072/weights_epoch_10.pt",
}
M = 4096
FRAMESKIP = 5
CTX = 3
SPAN = (CTX + 1) * FRAMESKIP + 1


def ridge_r2(H, y):
    n = len(H)
    tr, va = slice(0, n // 2), slice(n // 2, n)
    Hc, yc = H[tr] - H[tr].mean(0), y[tr] - y[tr].mean(0)
    W = np.linalg.solve(Hc.T @ Hc + 1e-2 * np.eye(H.shape[1]), Hc.T @ yc)
    pred = (H[va] - H[tr].mean(0)) @ W + y[tr].mean(0)
    return 1 - ((y[va] - pred) ** 2).sum() / ((y[va] - y[va].mean(0)) ** 2).sum()


@torch.no_grad()
def main():
    device = "cuda"
    dataset = swm.data.load_dataset(
        "pusht_expert_train.lance", keys_to_load=["pixels", "state", "action"])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)

    act_col = np.asarray(dataset.get_col_data("action"), dtype=np.float64)
    ok = ~np.isnan(act_col).any(axis=1)
    a_mean, a_std = act_col[ok].mean(0), act_col[ok].std(0)
    state_col = np.asarray(dataset.get_col_data("state"), dtype=np.float64)

    valid = test_eps[lengths[test_eps] > SPAN + 1]
    eps = g.choice(valid, M, replace=True)
    starts = g.integers(0, lengths[eps] - SPAN - 1)

    frame_rows = np.concatenate(
        [offsets[e] + s + np.arange(CTX + 1) * FRAMESKIP for e, s in zip(eps, starts)])
    uniq, inverse = np.unique(frame_rows, return_inverse=True)
    pix, _ = load_frames(dataset, uniq, device)

    act_blocks = np.zeros((M, CTX, FRAMESKIP * 2), dtype=np.float32)
    targets = {k: np.zeros(M) for k in
               ["block_speed", "block_angvel", "pusher_speed", "contact_dist"]}
    targets["block_xy"] = np.zeros((M, 2))
    for m, (e, s) in enumerate(zip(eps, starts)):
        base = offsets[e] + s
        for t in range(CTX):
            rows = act_col[base + t * FRAMESKIP: base + (t + 1) * FRAMESKIP]
            act_blocks[m, t] = ((np.nan_to_num(rows) - a_mean) / a_std).reshape(-1)
        s_prev = state_col[base + CTX * FRAMESKIP - 1]   # env-step before target
        s_tgt = state_col[base + CTX * FRAMESKIP]        # target frame
        targets["block_speed"][m] = np.linalg.norm(s_tgt[2:4] - s_prev[2:4])
        targets["block_angvel"][m] = np.linalg.norm(
            [np.cos(s_tgt[4]) - np.cos(s_prev[4]), np.sin(s_tgt[4]) - np.sin(s_prev[4])])
        targets["pusher_speed"][m] = np.linalg.norm(s_tgt[:2] - s_prev[:2])
        targets["contact_dist"][m] = np.linalg.norm(s_tgt[:2] - s_tgt[2:4])
        targets["block_xy"][m] = s_tgt[2:4]
    acts = torch.from_numpy(act_blocks).float().to(device)

    names = list(targets)
    for name, ckpt in MODELS.items():
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z = encode(model, pix, device)[torch.from_numpy(inverse)].reshape(M, CTX + 1, -1)

        hiddens = []  # list over layers of (M, hidden)
        feats = {}

        def mk_hook(idx):
            def hook(mod, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                feats.setdefault(idx, []).append(x[:, -1].float().cpu())
            return hook

        hs = [model.predictor.transformer.input_proj.register_forward_hook(mk_hook(0))]
        for li, blk in enumerate(model.predictor.transformer.layers):
            hs.append(blk.register_forward_hook(mk_hook(li + 1)))

        bs = 512
        for i in range(0, M, bs):
            model.predict(z[i:i + bs, :CTX].to(device), model.action_encoder(acts[i:i + bs]))
        for h in hs:
            h.remove()
        n_layers = len(feats)
        hiddens = [torch.cat(feats[i]).numpy().astype(np.float64) for i in range(n_layers)]

        print(f"\n=== {name} (rows: in-proj + {n_layers-1} blocks) ===")
        print(f"{'layer':>6s}" + "".join(f"{t:>14s}" for t in names))
        for li, H in enumerate(hiddens):
            r2s = [ridge_r2(H, targets[t].reshape(len(H), -1)) for t in names]
            print(f"{li:>6d}" + "".join(f"{r:14.3f}" for r in r2s))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
