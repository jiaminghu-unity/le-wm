"""Oracle-vs-predicted action-ranking diagnostic on USB (directive: freeze everything).

For N fixed dataset states: tile each across K Isaac envs, execute K sampled actions
for ONE step, then compare three rankings over the K candidates:
  real outcome  <->  C_oracle = ||E(o_{t+1})-z_g||^2  <->  C_pred = ||ẑ_{t+1}-z_g||^2
Outcome (lower=better) = max(plug_pos/1cm, plug_ang/15deg, ee_pos/1cm, ee_ang/15deg)
to the goal state (official joint-success geometry), plus plug_pos alone.

Run inside the Isaac conda env on cw-l4-01:
  python action_ranking_diag.py --variant dino --ckpt-path outputs/d1/d2_best.pt \
      --num-states 40 --goal-offset-steps 12 [official eval args...]
"""
import eval_planner as ep   # imports isaacgym first (must precede torch)
import argparse, json
import numpy as np
import torch


def quat_angle_deg(q1, q2):
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-12)
    dot = np.abs((q1 * q2).sum(-1).clip(-1, 1))
    return np.degrees(2 * np.arccos(dot))


def to_chw(x):
    """(..., H, W, 3) -> (..., 3, H, W); passthrough if already channel-first."""
    x = np.asarray(x)
    if x.shape[-1] == 3 and (x.ndim < 3 or x.shape[-3] != 3):
        x = np.moveaxis(x, -1, -3)
    return x


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ca, cb = ra - ra.mean(), rb - rb.mean()
    return float((ca * cb).sum() / (np.sqrt((ca**2).sum() * (cb**2).sum()) + 1e-12))


def pairwise_acc(outcome, cost, n_pairs=2000, rng=None):
    rng = rng or np.random.RandomState(0)
    K = len(outcome); ok = tot = 0
    for _ in range(n_pairs):
        i, j = rng.randint(0, K, 2)
        if outcome[i] == outcome[j]: continue
        tot += 1
        if (outcome[i] < outcome[j]) == (cost[i] < cost[j]): ok += 1
    return ok / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["dino", "b1", "front", "wrist", "duet"], required=True)
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--data-root", default="data/demo_data/insertion_usb")
    ap.add_argument("--isaacgym-cfg-name", default="isaacgym_config_usb")
    ap.add_argument("--num-states", type=int, default=40)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--goal-offset-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/action_ranking_diag.json")
    args, _ = ap.parse_known_args()
    dev = "cuda"

    # model adapter (frozen, both-camera encode + one-step predict)
    if args.variant == "dino":
        from touchwm.dino_eval_adapter import DinoWorldModel
        wm = DinoWorldModel(args.ckpt_path)
        enc = lambda f, w: wm._encode_views(torch.as_tensor(np.ascontiguousarray(f)).float().to("cuda"), torch.as_tensor(np.ascontiguousarray(w)).float().to("cuda"))
        predict = wm.model.predict
    else:
        from touchwm.cw_eval_adapter import MVJepaWorldModel
        wm = MVJepaWorldModel(args.ckpt_path, variant=args.variant)
        def enc(f, w):
            f = torch.as_tensor(np.ascontiguousarray(f)).float().to(dev)
            w = torch.as_tensor(np.ascontiguousarray(w)).float().to(dev)
            return wm.model.encode({"front": wm._prep(f), "wrist": wm._prep(w)})
        predict = wm.model.predict

    # dataset + candidate (state, goal) segments, env with K parallel copies
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
    global _fr_glob, _wr_glob, _ac_glob, _ends_glob, _starts_glob
    import zarr as _z
    _zr = _z.open(args.data_root, mode="r")
    _fr_glob = _zr["data/front"]; _wr_glob = _zr["data/wrist"]
    _ac_glob = np.asarray(_zr["data/action"][:])
    _ends_glob = np.asarray(_zr["meta/episode_ends"][:]).astype(int)
    _starts_glob = np.concatenate([[0], _ends_glob[:-1]])

    env = ep.make_env(ea)
    if hasattr(env, "seed"): env.seed(ep.safe_env_seed(args.seed))
    _ = env.reset()

    rng = np.random.RandomState(args.seed)
    rows = []
    for i in range(args.num_states):
        tile = {k: np.repeat(np.asarray(v)[i:i+1], args.k, axis=0)
                for k, v in batch.items() if hasattr(v, "__len__")}
        ep.reset_env_to_dataset_state(env, tile)
        acts = np.clip(a_mean + a_std * rng.randn(args.k, 6), -1.0, 1.0).astype(np.float32)
        obs, _, _, info = env.step(acts)
        if i == 0:
            print("obs keys:", list(obs.keys()), "| info keys:", list(info.keys()), flush=True)
        def grab(name):
            for src in (obs, info):
                for k in src:
                    if k == name or k.endswith(name): return np.asarray(src[k])
            raise KeyError(name)
        plug_p, plug_q = grab("plug_pos"), grab("plug_quat")
        ee_p, ee_q = grab("ee_pos"), grab("ee_quat")
        gp, gq = tile["goal_plug_pos"], tile["goal_plug_quat"]
        gep, geq = tile["goal_ee_pos"], tile["goal_ee_quat"]
        plug_err = np.linalg.norm(plug_p - gp, axis=1)
        ee_err = np.linalg.norm(ee_p - gep, axis=1)
        plug_ang = quat_angle_deg(plug_q, gq); ee_ang = quat_angle_deg(ee_q, geq)
        outcome = np.maximum.reduce([plug_err/0.01, plug_ang/15.0, ee_err/0.01, ee_ang/15.0])

        # frames: 3-frame context t-2..t from the dataset (training-matched history)
        ei, si = int(batch["ep_idx"][i]) if "ep_idx" in batch else None, None
        t_abs = int(batch["start_abs"][i]) if "start_abs" in batch else None
        f_t = to_chw(np.asarray(tile["front"])[0:1]); w_t = to_chw(np.asarray(tile["wrist"])[0:1])
        f_g = np.asarray(tile["goal_front"] if "goal_front" in tile else tile["front_goal"])[0:1] \
              if ("goal_front" in tile or "front_goal" in tile) else None
        with torch.no_grad():
            z_t = enc(f_t[:, None] if f_t.ndim == 4 else f_t, w_t[:, None] if w_t.ndim == 4 else w_t)
            hist_ok = False
            t_abs = int(np.asarray(batch["init_row_idx"])[i])
            ep_start = int(_starts_glob[np.searchsorted(_ends_glob, t_abs, side="right")] if False else 0)
            if t_abs - 2 >= 0:
                fr3 = to_chw(np.asarray(_fr_glob[t_abs-2:t_abs+1]))[None]
                wr3 = to_chw(np.asarray(_wr_glob[t_abs-2:t_abs+1]))[None]
                z_hist = enc(fr3, wr3)
                a_hist = torch.as_tensor(_ac_glob[t_abs-2:t_abs], device=dev).float()[None]
                hist_ok = True
            if f_g is None:
                # goal obs delivered via build_goal_info_numpy convention
                gi = ep.build_goal_info_numpy(tile, type("x", (), {"vision_key": "front",
                     "use_tactile": True, "tactile_key": "wrist"}))
                z_g = enc(to_chw(gi["front"][0:1]), to_chw(gi["wrist"][0:1]))[:, -1]
            else:
                z_g = enc(f_g[:, None], np.asarray(tile["goal_wrist"])[0:1][:, None])[:, -1]
            f1 = to_chw(grab("front")); w1 = to_chw(grab("wrist"))
            z1 = enc(torch.as_tensor(f1)[:, None] if f1.ndim == 4 else torch.as_tensor(f1),
                     torch.as_tensor(w1)[:, None] if w1.ndim == 4 else torch.as_tensor(w1))
            c_oracle = (z1[:, -1] - z_g).pow(2).sum(-1).cpu().numpy()
            if hist_ok:
                zt_k = z_hist.expand(args.k, 3, -1)
                a_hist_k = torch.cat([a_hist.expand(args.k, 2, -1),
                                      torch.as_tensor(acts, device=dev).float()[:, None]], dim=1)
                z_pred = predict(zt_k, a_hist_k)[:, -1]
            else:
                zt_k = z_t[:, -1:].expand(args.k, 1, -1)
                a_k = torch.as_tensor(acts, device=dev).float()[:, None]
                z_pred = predict(zt_k, a_k)[:, -1]
            c_pred = (z_pred - z_g).pow(2).sum(-1).cpu().numpy()
            halves = {}
            if args.variant == "duet":
                for nm, sl in (("f", slice(0, 192)), ("w", slice(192, 384))):
                    halves[f"c_oracle_{nm}"] = (z1[:, -1, sl] - z_g[:, sl]).pow(2).sum(-1).cpu().numpy()
                    halves[f"c_pred_{nm}"] = (z_pred[:, sl] - z_g[:, sl]).pow(2).sum(-1).cpu().numpy()

        row = {"state": i,
               "sp_outcome_oracle": spearman(outcome, c_oracle),
               "sp_outcome_pred": spearman(outcome, c_pred),
               "sp_oracle_pred": spearman(c_oracle, c_pred),
               "pa_oracle": pairwise_acc(outcome, c_oracle, rng=rng),
               "pa_pred": pairwise_acc(outcome, c_pred, rng=rng),
               "regret_oracle": float(outcome[int(np.argmin(c_oracle))] - outcome.min()),
               "regret_pred": float(outcome[int(np.argmin(c_pred))] - outcome.min()),
               "outcome_spread": float(outcome.max() - outcome.min())}
        if args.variant == "duet":
            for nm in ("f", "w"):
                row[f"sp_outcome_oracle_{nm}"] = spearman(outcome, halves[f"c_oracle_{nm}"])
                row[f"sp_outcome_pred_{nm}"] = spearman(outcome, halves[f"c_pred_{nm}"])
        rows.append(row)
        print(json.dumps(row), flush=True)

    agg = {}
    for k in rows[0]:
        if k == "state": continue
        v = np.array([r[k] for r in rows])
        agg[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)}
    out = {"variant": args.variant, "n_states": args.num_states, "k": args.k,
           "goal_offset": args.goal_offset_steps, "per_state": rows, "aggregate": agg}
    json.dump(out, open(args.out, "w"), indent=2)
    print("AGG", json.dumps(agg, indent=1), flush=True)
    print("RANKING DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
