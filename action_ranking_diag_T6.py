"""Sequence-level (horizon-6) oracle-vs-predicted ranking diagnostic on USB.

Extends action_ranking_diag.py from 1-step to full 6-step action SEQUENCES:
for N dataset states, tile across K envs, execute K sampled 6-step sequences,
then rank by
  real outcome (final & best-step)  <->  C_oracle = ||E(o_{t+6})-z_g||^2
                                    <->  C_pred  = ||ẑ_{t+6}-z_g||^2 (6-step AR rollout)
Isolates multi-step compounding (pred collapses, oracle holds) vs long-horizon
cost geometry (oracle collapses too) vs CEM optimization (both hold, SR still low).

Run inside the Isaac conda env on cw-l4-01:
  python action_ranking_diag_T6.py --variant b1 --ckpt-path outputs_b1_ckpt_100k.pt \
      --num-states 40 --k 100 --horizon 6 --goal-offset-steps 12
"""
import eval_planner as ep   # imports isaacgym first (must precede torch)
import argparse, json
import numpy as np
import torch

from action_ranking_diag import quat_angle_deg, to_chw, spearman, pairwise_acc


def main():
    ap = argparse.ArgumentParser()
    # ours/b3 excluded: their deployment dynamics is base+argmax-gate residual
    # (mode_residual.rollout), not plain predict(); add a hard-gate step before
    # extending to them or the diag silently tests the wrong dynamics.
    ap.add_argument("--variant", choices=["dino", "b1", "front", "wrist", "duet"], required=True)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--data-root", default="data/demo_data/insertion_usb")
    ap.add_argument("--isaacgym-cfg-name", default="isaacgym_config_usb")
    ap.add_argument("--num-states", type=int, default=40)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--goal-offset-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-raw", action="store_true",
                    help="dump per-state raw arrays (outcomes, costs, per-step outcomes, mode context)")
    args, _ = ap.parse_known_args()
    dev = "cuda"
    T = args.horizon
    out_path = args.out or f"outputs/rank_diag_T{T}_{args.variant}.json"

    if args.variant == "dino":
        from touchwm.dino_eval_adapter import DinoWorldModel
        wm = DinoWorldModel(args.ckpt_path)
        enc = lambda f, w: wm._encode_views(torch.as_tensor(np.ascontiguousarray(f)).float().to(dev),
                                            torch.as_tensor(np.ascontiguousarray(w)).float().to(dev))
        predict = wm.model.predict
    else:
        from touchwm.cw_eval_adapter import MVJepaWorldModel
        wm = MVJepaWorldModel(args.ckpt_path, variant=args.variant)
        def enc(f, w):
            f = torch.as_tensor(np.ascontiguousarray(f)).float().to(dev)
            w = torch.as_tensor(np.ascontiguousarray(w)).float().to(dev)
            return wm.model.encode({"front": wm._prep(f), "wrist": wm._prep(w)})
        predict = wm.model.predict

    class A: pass
    ea = A()
    ea.data_root = args.data_root; ea.vision_key = "front"; ea.vision_type = "image"
    ea.use_tactile = True; ea.tactile_key = "wrist"; ea.image_size = 224
    ea.num_envs = args.k; ea.isaacgym_cfg_name = args.isaacgym_cfg_name
    ea.seed = args.seed; ea.headless = True
    from dataset.zarr_dataset import ZarrDataset
    dataset = ZarrDataset(root=args.data_root, frameskip=1, num_steps=1,
                          keys_to_load=["front", "wrist", "action", "dof_pos", "dof_vel",
                                        "plug_pos", "plug_quat", "socket_pos_gt",
                                        "socket_quat", "ee_pos", "ee_quat"],
                          keys_to_cache=["action", "dof_pos", "dof_vel", "plug_pos",
                                         "plug_quat", "socket_pos_gt", "socket_quat",
                                         "ee_pos", "ee_quat"])
    all_actions = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    a_mean = all_actions.mean(0); a_std = all_actions.std(0) + 1e-6

    batch = ep.sample_init_goal_segments(dataset=dataset, num_goals=args.num_states,
                                         goal_offset_steps=args.goal_offset_steps,
                                         seed=args.seed, vision_key="front",
                                         tactile_key="wrist")
    import zarr as _z
    _zr = _z.open(args.data_root, mode="r")
    _fr_glob = _zr["data/front"]; _wr_glob = _zr["data/wrist"]
    _ac_glob = np.asarray(_zr["data/action"][:])

    # mode context: provisional transition-mode labels (-2 undefined, -1 ambiguous)
    mode_row = np.full(10**6, -2, dtype=np.int64)
    try:
        M = np.load("outputs/preflight/usb/provisional_modes.npz")
        mode_row[M["t"]] = M["mode"]
    except Exception as e:
        print("[warn] no mode labels:", e, flush=True)

    env = ep.make_env(ea)
    if hasattr(env, "seed"): env.seed(ep.safe_env_seed(args.seed))
    _ = env.reset()

    rng = np.random.RandomState(args.seed)
    rows = []
    raw = {k: [] for k in ("state", "t_abs", "outcome_final", "outcome_best",
                           "oc_steps", "c_oracle", "c_pred", "mode_window")}
    for i in range(args.num_states):
        tile = {k: np.repeat(np.asarray(v)[i:i+1], args.k, axis=0)
                for k, v in batch.items() if hasattr(v, "__len__")}
        ep.reset_env_to_dataset_state(env, tile)
        # K independent T-step sequences from demo stats (same prior as CEM init)
        seqs = np.clip(a_mean + a_std * rng.randn(args.k, T, 6), -1.0, 1.0).astype(np.float32)

        gp, gq = tile["goal_plug_pos"], tile["goal_plug_quat"]
        gep, geq = tile["goal_ee_pos"], tile["goal_ee_quat"]
        def step_outcome(obs, info):
            def grab(name):
                for src in (obs, info):
                    for k in src:
                        if k == name or k.endswith(name): return np.asarray(src[k])
                raise KeyError(name)
            plug_err = np.linalg.norm(grab("plug_pos") - gp, axis=1)
            ee_err = np.linalg.norm(grab("ee_pos") - gep, axis=1)
            plug_ang = quat_angle_deg(grab("plug_quat"), gq)
            ee_ang = quat_angle_deg(grab("ee_quat"), geq)
            oc = np.maximum.reduce([plug_err/0.01, plug_ang/15.0, ee_err/0.01, ee_ang/15.0])
            return oc, grab("front"), grab("wrist")

        best = np.full(args.k, np.inf)
        oc_steps = []
        for s in range(T):
            obs, _, _, info = env.step(seqs[:, s])
            oc, f_last, w_last = step_outcome(obs, info)
            best = np.minimum(best, oc)
            oc_steps.append(oc.copy())
        outcome_final, outcome_best = oc, best

        t_abs = int(np.asarray(batch["init_row_idx"])[i])
        if t_abs - 2 < 0:
            print(f"state {i}: t_abs<2, skipped", flush=True); continue
        with torch.no_grad():
            fr3 = to_chw(np.asarray(_fr_glob[t_abs-2:t_abs+1]))[None]
            wr3 = to_chw(np.asarray(_wr_glob[t_abs-2:t_abs+1]))[None]
            z_hist = enc(fr3, wr3)                                   # (1,3,D)
            a_hist = torch.as_tensor(_ac_glob[t_abs-2:t_abs], device=dev).float()[None]  # (1,2,A)
            gi = ep.build_goal_info_numpy(tile, type("x", (), {"vision_key": "front",
                 "use_tactile": True, "tactile_key": "wrist"}))
            z_g = enc(to_chw(gi["front"][0:1]), to_chw(gi["wrist"][0:1]))[:, -1]

            cand = torch.as_tensor(seqs, device=dev).float()          # (K,T,A)
            D = z_hist.shape[-1]
            z_ctx = z_hist.expand(args.k, 3, D).clone()
            a_ctx = torch.cat([a_hist.expand(args.k, 2, -1), cand[:, 0:1]], dim=1)
            for s in range(T):
                z_next = predict(z_ctx, a_ctx)[:, -1:]
                z_ctx = torch.cat([z_ctx[:, 1:], z_next], dim=1)
                if s + 1 < T:
                    a_ctx = torch.cat([a_ctx[:, 1:], cand[:, s+1:s+2]], dim=1)
            c_pred = (z_ctx[:, -1] - z_g).pow(2).sum(-1).cpu().numpy()

            # oracle: encode the REAL final frames after executing each sequence
            z_fin = enc(to_chw(f_last)[:, None], to_chw(w_last)[:, None])[:, -1]
            c_oracle = (z_fin - z_g).pow(2).sum(-1).cpu().numpy()

        row = {"state": i}
        for tag, oc_v in (("final", outcome_final), ("best", outcome_best)):
            row[f"sp_{tag}_oracle"] = spearman(oc_v, c_oracle)
            row[f"sp_{tag}_pred"] = spearman(oc_v, c_pred)
            row[f"pa_{tag}_oracle"] = pairwise_acc(oc_v, c_oracle, rng=rng)
            row[f"pa_{tag}_pred"] = pairwise_acc(oc_v, c_pred, rng=rng)
            row[f"regret_{tag}_oracle"] = float(oc_v[int(np.argmin(c_oracle))] - oc_v.min())
            row[f"regret_{tag}_pred"] = float(oc_v[int(np.argmin(c_pred))] - oc_v.min())
        row["sp_oracle_pred"] = spearman(c_oracle, c_pred)
        row["outcome_spread_final"] = float(outcome_final.max() - outcome_final.min())
        # mode context of the initial dataset state: labels of transitions t-2 .. t+5
        mw = mode_row[max(t_abs - 2, 0):t_abs + T].tolist()
        row["mode_t"] = int(mode_row[t_abs])
        lab = [m for m in mw if m >= 0]
        row["near_transition"] = int(len(set(lab)) > 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.save_raw:
            raw["state"].append(i); raw["t_abs"].append(t_abs)
            raw["outcome_final"].append(outcome_final.copy())
            raw["outcome_best"].append(outcome_best.copy())
            raw["oc_steps"].append(np.stack(oc_steps))
            raw["c_oracle"].append(c_oracle.copy()); raw["c_pred"].append(c_pred.copy())
            raw["mode_window"].append(np.asarray(mw, dtype=np.int64))

    agg = {}
    for k in rows[0]:
        if k == "state": continue
        v = np.array([r[k] for r in rows])
        agg[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)}
    out = {"variant": args.variant, "horizon": T, "n_states": len(rows), "k": args.k,
           "goal_offset": args.goal_offset_steps, "per_state": rows, "aggregate": agg}
    json.dump(out, open(out_path, "w"), indent=2)
    if args.save_raw:
        np.savez_compressed(out_path.replace(".json", "_raw.npz"),
                            **{k: np.asarray(v) for k, v in raw.items()})
        print("RAW saved:", out_path.replace(".json", "_raw.npz"), flush=True)
    print("AGG", json.dumps(agg, indent=1), flush=True)
    print("T6 RANKING DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
