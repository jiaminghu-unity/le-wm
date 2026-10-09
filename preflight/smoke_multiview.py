"""P1 — Multi-view LeWM engineering smoke test (proposal §6).

Tests A (shapes), B (gradients), C (small-batch overfit), D (collapse metrics),
E (view ablations), G (6-step rollout). Test H (Isaac evaluator) runs separately on
the render-capable machine; Test F (wrist info retention) follows with the probes.

  python -m preflight.smoke_multiview --data-root data/demo_data/insertion_usb \
      --splits outputs/preflight/usb/splits.json --output-dir outputs/preflight/usb
"""
import argparse, json, os, time
import numpy as np
import torch

from touchwm.lewm_adapter import MultiViewJEPA
from touchwm.data_adapter import MultiViewClipDataset


def finite(x):
    return bool(torch.isfinite(x).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--overfit-clips", type=int, default=24)
    ap.add_argument("--overfit-steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = MultiViewJEPA().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    report = {"device": dev, "n_params": n_params}

    ds = MultiViewClipDataset(args.data_root, args.splits, "train", clip_len=3)

    def batch_of(idxs, clip_len=None):
        items = [ds[i % len(ds)] for i in idxs]
        b = {k: torch.stack([it[k] for it in items]).to(dev)
             for k in ("front", "wrist", "action")}
        return b

    # ---- Test A: shapes ----
    A = {}
    for bs, T in [(1, 3), (4, 3), (4, 8)]:
        ds_t = MultiViewClipDataset(args.data_root, args.splits, "train", clip_len=T)
        items = [ds_t[(i * 37) % len(ds_t)] for i in range(bs)]
        b = {k: torch.stack([it[k] for it in items]).to(dev) for k in ("front", "wrist", "action")}
        with torch.no_grad():
            z = model.encode({"front": b["front"], "wrist": b["wrist"]})
            pred = model.predict(z[:, :-1], b["action"])
        A[f"bs{bs}_T{T}"] = {"z": list(z.shape), "pred": list(pred.shape),
                             "finite": finite(z) and finite(pred)}
    report["test_A_shapes"] = A

    # ---- Test B: gradients ----
    b = batch_of(list(range(0, args.batch * 101, 101)))
    out = model.loss({"front": b["front"], "wrist": b["wrist"]}, b["action"])
    model.zero_grad(); out["loss"].backward()
    def gnorm(mod):
        s = 0.0
        for p in mod.parameters():
            if p.grad is not None: s += p.grad.detach().pow(2).sum().item()
        return s ** 0.5
    gB = {"vit": gnorm(model.mv_encoder.vit),
          "cross_attn_f": gnorm(model.mv_encoder.ca_f),
          "cross_attn_w": gnorm(model.mv_encoder.ca_w),
          "query_pool": gnorm(model.mv_encoder.pool),
          "projector": gnorm(model.projector),
          "predictor": gnorm(model.predictor),
          "action_encoder": gnorm(model.action_encoder)}
    gB["all_nonzero_finite"] = all(v > 0 and np.isfinite(v) for v in gB.values())
    report["test_B_gradients"] = gB
    model.zero_grad()

    # ---- Test C: small-batch overfit ----
    idxs = list(range(0, args.overfit_clips * 173, 173))[:args.overfit_clips]
    fixed = batch_of(idxs)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    model.train()
    hist = []
    t0 = time.time()
    for step in range(args.overfit_steps):
        sel = torch.randperm(len(idxs))[:args.batch]
        ob = {k: fixed[k][sel] for k in fixed}
        out = model.loss({"front": ob["front"], "wrist": ob["wrist"]}, ob["action"])
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == args.overfit_steps - 1:
            hist.append({"step": step, "loss": float(out["loss"]),
                         "pred": float(out["pred_loss"]), "sigreg": float(out["sigreg_loss"])})
    report["test_C_overfit"] = {
        "steps": args.overfit_steps, "seconds": round(time.time() - t0, 1),
        "first": hist[0], "last": hist[-1],
        "pred_loss_drop_ratio": hist[0]["pred"] / max(hist[-1]["pred"], 1e-9),
        "history": hist}

    # ---- Test D: collapse metrics ----
    model.eval()
    with torch.no_grad():
        zb = []
        for s in range(0, 64 * 211, 211):
            it = ds[s % len(ds)]
            zb.append(model.encode({"front": it["front"][None].to(dev),
                                    "wrist": it["wrist"][None].to(dev)})[0])
        Z = torch.cat(zb, 0)                      # (64*T, D)
        var = Z.var(0)
        cov = torch.cov(Z.T)
        eig = torch.linalg.eigvalsh(cov).clamp(min=0)
        p = eig / eig.sum()
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
        pd = torch.cdist(Z[:128], Z[:128])
        report["test_D_collapse"] = {
            "latent_var_min": float(var.min()), "latent_var_mean": float(var.mean()),
            "effective_rank": eff_rank, "dim": Z.shape[1],
            "mean_pairwise_dist": float(pd[pd > 0].mean()),
            "sigreg_eval": float(model.sigreg(Z[None]))}

    # ---- Test E: both views matter ----
    with torch.no_grad():
        n = args.batch
        base_idx = list(range(0, n * 307, 307))
        swap_idx = list(range(5000, 5000 + n * 101, 101))
        b0 = batch_of(base_idx); bs_ = batch_of(swap_idx)
        obs = lambda f, w: {"front": f, "wrist": w}
        z_cc = model.encode(obs(b0["front"], b0["wrist"]))
        z_cw = model.encode(obs(b0["front"], bs_["wrist"]))       # wrist swapped
        z_fc = model.encode(obs(bs_["front"], b0["wrist"]))       # front swapped
        z_fm = model.encode(obs(b0["front"], torch.zeros_like(b0["wrist"])))
        z_wm = model.encode(obs(torch.zeros_like(b0["front"]), b0["wrist"]))
        d = lambda a: float((a - z_cc).norm(dim=-1).mean())
        base_scale = float(z_cc.norm(dim=-1).mean())
        pe = {}
        for name, zz in [("swap_wrist", z_cw), ("swap_front", z_fc),
                         ("mask_wrist", z_fm), ("mask_front", z_wm)]:
            pred = model.predict(zz[:, :-1], b0["action"])
            pe[name] = {"latent_delta": d(zz),
                        "pred_err_vs_true_next": float((pred - z_cc[:, 1:]).pow(2).mean())}
        pred0 = model.predict(z_cc[:, :-1], b0["action"])
        report["test_E_view_ablation"] = {
            "latent_norm_scale": base_scale,
            "pred_err_clean": float((pred0 - z_cc[:, 1:]).pow(2).mean()), **pe,
            "both_views_change_latent": pe["swap_wrist"]["latent_delta"] > 1e-3 and
                                        pe["swap_front"]["latent_delta"] > 1e-3}

    # ---- Test G: 6-step rollout ----
    with torch.no_grad():
        it = ds[0]
        cur = {"front": it["front"][None, :3].to(dev), "wrist": it["wrist"][None, :3].to(dev)}
        cand = torch.randn(1, 5, 6, 6, device=dev) * 0.1
        fut = model.rollout(cur, cand)
        fut2 = model.rollout(cur, cand + 0.5)
        report["test_G_rollout"] = {
            "shape": list(fut.shape), "finite": finite(fut),
            "latent_norm_max": float(fut.norm(dim=-1).max()),
            "action_sensitivity": float((fut - fut2).abs().mean()),
            "cost_example": model.cem_cost(fut, fut[:, 0, -1]).tolist()}

    with open(os.path.join(args.output_dir, "smoke_training_log.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "test_C_overfit"} |
                     {"test_C_overfit": {kk: report["test_C_overfit"][kk]
                      for kk in ("first", "last", "pred_loss_drop_ratio", "seconds")}}, indent=2))


if __name__ == "__main__":
    main()
