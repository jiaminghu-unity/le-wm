"""P2 — Tactile dynamics-mode label feasibility (proposal §7).

Computes per-transition tactile statistics (A, D, A_rel, R, patch geometry),
motion-response features, fits thresholds on TRAIN episodes only, assigns
provisional modes {0 free, 1 sustained contact, 2 transition, -1 ambiguous},
and emits quantitative checks + example strips.

  python -m preflight.extract_tactile_events \
      --data-root data/demo_data/insertion_usb \
      --splits outputs/preflight/usb/splits.json \
      --output-dir outputs/preflight/usb
"""
import argparse, json, os
import numpy as np
import zarr


def quat_angle(q1, q2):
    """Geodesic angle (rad) between unit quaternions, batch."""
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-12)
    dot = np.abs(np.sum(q1 * q2, axis=-1).clip(-1, 1))
    return 2 * np.arccos(dot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-examples", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    root = zarr.open(args.data_root, mode="r")
    T = np.asarray(root["data/tactile_force_field_right"][:])      # [N,10,14,3]
    act = np.asarray(root["data/action"][:])                        # [N,6]
    plug = np.asarray(root["data/plug_pos"][:])
    ee = np.asarray(root["data/ee_pos"][:])
    sock = np.asarray(root["data/socket_pos_gt"][:])
    plug_q = np.asarray(root["data/plug_quat"][:])
    ee_q = np.asarray(root["data/ee_quat"][:])
    ends = np.asarray(root["meta/episode_ends"][:]).astype(int)
    starts = np.concatenate([[0], ends[:-1]])
    n_ep = len(ends)
    splits = json.load(open(args.splits))
    train_eps = set(splits["train"])

    # ---- per-transition index sets (never cross boundaries) ----
    rows = []   # (ep, t) with t, t+1 in-episode
    for ep in range(n_ep):
        for t in range(starts[ep], ends[ep] - 1):
            rows.append((ep, t))
    rows = np.array(rows)
    ep_of, t_of = rows[:, 0], rows[:, 1]
    is_train = np.isin(ep_of, list(train_eps))

    cell = np.linalg.norm(T, axis=-1)                # [N,10,14] per-cell magnitude
    flatT = T.reshape(len(T), -1)                    # [N,420]

    # A: raw activity;  D: temporal change (within episode via row pairs)
    A = cell.mean(axis=(1, 2))
    D_pair = np.linalg.norm(T[t_of + 1] - T[t_of], axis=-1).mean(axis=(1, 2))

    # B: baseline reference field from stable TRAIN frames (lowest-D quintile)
    d_thr = np.quantile(D_pair[is_train], 0.2)
    stable_train_frames = t_of[is_train & (D_pair <= d_thr)]
    B = np.median(T[stable_train_frames], axis=0)    # [10,14,3]
    A_rel = np.linalg.norm(T - B[None], axis=-1).mean(axis=(1, 2))

    # R: spatial redistribution with near-zero-norm guard
    eps = 1e-6
    n0 = np.linalg.norm(flatT[t_of], axis=1)
    n1 = np.linalg.norm(flatT[t_of + 1], axis=1)
    R_pair = np.linalg.norm(flatT[t_of + 1] / (n1[:, None] + eps)
                            - flatT[t_of] / (n0[:, None] + eps), axis=1)
    norm_floor = np.quantile(np.concatenate([n0[is_train], n1[is_train]]), 0.05)
    R_valid = (n0 > norm_floor) & (n1 > norm_floor)
    R_pair = np.where(R_valid, R_pair, 0.0)

    # contact patch geometry (active cells over train noise floor)
    noise_thr = np.quantile(cell[stable_train_frames].ravel(), 0.95)
    active = cell > noise_thr                        # [N,10,14]
    area = active.mean(axis=(1, 2))
    yy, xx = np.meshgrid(np.arange(10), np.arange(14), indexing="ij")
    w = cell + 1e-9
    cy = (w * yy).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
    cx = (w * xx).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
    center_shift = np.sqrt((cy[t_of + 1] - cy[t_of]) ** 2 + (cx[t_of + 1] - cx[t_of]) ** 2)

    # ---- motion-response features (§7.3, diagnostics only) ----
    d_plug = np.linalg.norm(plug[t_of + 1] - plug[t_of], axis=1)
    d_ee = np.linalg.norm(ee[t_of + 1] - ee[t_of], axis=1)
    rel0 = plug[t_of] - sock[t_of]; rel1 = plug[t_of + 1] - sock[t_of + 1]
    d_rel = np.linalg.norm(rel1 - rel0, axis=1)
    rot_plug = quat_angle(plug_q[t_of + 1], plug_q[t_of])
    rot_ee = quat_angle(ee_q[t_of + 1], ee_q[t_of])
    a_mag = np.linalg.norm(act[t_of], axis=1)
    # action convention check: corr(action_xyz, d_ee_vec)
    dee_vec = ee[t_of + 1] - ee[t_of]
    corr = [float(np.corrcoef(act[t_of][:, i], dee_vec[:, i])[0, 1]) for i in range(3)]
    # motion-per-command ratio (response gain), guarded
    gain = d_ee / (np.linalg.norm(act[t_of][:, :3], axis=1) + 1e-6)

    # ---- provisional modes (§7.4), thresholds fitted on TRAIN only ----
    q = lambda x, p: float(np.quantile(x[is_train], p))
    TH = {
        "A_rel_hi": q(A_rel[t_of], 0.60),
        "A_rel_lo": q(A_rel[t_of], 0.40),
        "D_hi":     q(D_pair, 0.85),
        "R_hi":     float(np.quantile(R_pair[R_valid & is_train], 0.85))
                    if (R_valid & is_train).any() else float(np.quantile(R_pair[is_train], 0.85)),
        "noise_thr": float(noise_thr), "norm_floor": float(norm_floor), "stable_D_q20": float(d_thr),
    }
    a_rel_t = A_rel[t_of]
    hiD = (D_pair >= TH["D_hi"]) | ((R_pair >= TH["R_hi"]) & R_valid)
    mode = np.full(len(rows), -1, dtype=int)
    mode[hiD] = 2
    mode[(~hiD) & (a_rel_t <= TH["A_rel_lo"])] = 0
    mode[(~hiD) & (a_rel_t >= TH["A_rel_hi"])] = 1
    # remainder stays -1 (ambiguous band between lo/hi)

    np.savez(os.path.join(args.output_dir, "tactile_event_features.npz"),
             ep=ep_of, t=t_of, A=A[t_of], A_rel=a_rel_t, D=D_pair, R=R_pair,
             R_valid=R_valid, area=area[t_of], center_shift=center_shift,
             d_plug=d_plug, d_ee=d_ee, d_rel=d_rel, rot_plug=rot_plug, rot_ee=rot_ee,
             a_mag=a_mag, gain=gain, is_train=is_train, baseline=B)
    np.savez(os.path.join(args.output_dir, "provisional_modes.npz"),
             ep=ep_of, t=t_of, mode=mode, thresholds=json.dumps(TH))

    # ---- quantitative checks (§7.6) ----
    rep = {"action_xyz_vs_dee_corr": corr, "thresholds": TH}
    for name, msk in [("all", np.ones(len(rows), bool)), ("train", is_train),
                      ("heldout", ~is_train)]:
        mm = mode[msk]
        dist = {str(k): int((mm == k).sum()) for k in (-1, 0, 1, 2)}
        rep[f"class_distribution_{name}"] = dist
    rep["episodes_per_class"] = {str(k): int(len(np.unique(ep_of[mode == k]))) for k in (-1, 0, 1, 2)}
    # durations & transitions (run-lengths within episodes)
    durs = {k: [] for k in (-1, 0, 1, 2)}; trans = {}
    for ep in range(n_ep):
        m = mode[ep_of == ep]
        if not len(m): continue
        s = 0
        for i in range(1, len(m) + 1):
            if i == len(m) or m[i] != m[s]:
                durs[int(m[s])].append(i - s)
                if i < len(m):
                    key = f"{m[s]}->{m[i]}"; trans[key] = trans.get(key, 0) + 1
                s = i
    rep["mode_duration_mean"] = {str(k): (float(np.mean(v)) if v else None) for k, v in durs.items()}
    rep["mode_duration_p90"] = {str(k): (float(np.quantile(v, 0.9)) if v else None) for k, v in durs.items()}
    rep["transition_counts"] = dict(sorted(trans.items(), key=lambda kv: -kv[1]))
    # progress correlation & motion relationship
    prog = (t_of - starts[ep_of]) / (ends[ep_of] - starts[ep_of] - 1).clip(min=1)
    rep["mean_progress_by_mode"] = {str(k): float(prog[mode == k].mean()) for k in (-1, 0, 1, 2) if (mode == k).any()}
    for feat, arr in [("d_plug", d_plug), ("d_ee", d_ee), ("d_rel", d_rel),
                      ("gain", gain), ("a_mag", a_mag), ("rot_plug", rot_plug)]:
        rep[f"mean_{feat}_by_mode"] = {str(k): float(arr[mode == k].mean()) for k in (-1, 0, 1, 2) if (mode == k).any()}
    # threshold sensitivity: shift each percentile ±10pts, report class share drift
    sens = {}
    for dp in (-0.10, 0.10):
        th_d = q(D_pair, min(max(0.85 + dp, 0.01), 0.99))
        hiD2 = (D_pair >= th_d) | ((R_pair >= TH["R_hi"]) & R_valid)
        sens[f"D_hi{dp:+.2f}"] = float(hiD2.mean())
    rep["sensitivity_mode2_share"] = {"base": float(hiD.mean()), **sens}
    with open(os.path.join(args.output_dir, "mode_class_distribution.json"), "w") as f:
        json.dump(rep, f, indent=2)

    # ---- example strips per mode (§7.5) ----
    try:
        from PIL import Image
        front = root["data/front"]; wrist = root["data/wrist"]
        exdir = os.path.join(args.output_dir, "mode_examples"); os.makedirs(exdir, exist_ok=True)
        rng = np.random.RandomState(0)
        global_hi = float(np.quantile(cell, 0.999)) + 1e-9   # one scale for all strips
        meta = {"tacff_heatmap_scale_p999": global_hi}
        for k in (0, 1, 2, -1):
            idx = np.where(mode == k)[0]
            # spread examples across distinct episodes
            chosen, seen_ep = [], set()
            for i in rng.permutation(len(idx)):
                e = ep_of[idx[i]]
                if e in seen_ep and len(seen_ep) < 25: continue
                chosen.append(idx[i]); seen_ep.add(e)
                if len(chosen) >= args.n_examples: break
            meta[str(k)] = []
            for j, ridx in enumerate(chosen[:args.n_examples]):
                t = int(t_of[ridx])
                fr = (np.asarray(front[t]) * 255).astype(np.uint8)
                wr = (np.asarray(wrist[t]) * 255).astype(np.uint8)
                tac = cell[t]; tacn = (255 * np.clip(tac / global_hi, 0, 1)).astype(np.uint8)
                tac_img = np.asarray(Image.fromarray(tacn).resize((256, 256), Image.NEAREST))
                strip = np.concatenate([fr, wr, np.stack([tac_img] * 3, -1)], axis=1)
                Image.fromarray(strip).save(os.path.join(exdir, f"mode{k}_{j:02d}_ep{ep_of[ridx]}_t{t}.png"))
                meta[str(k)].append({"ep": int(ep_of[ridx]), "t": t,
                                     "A_rel": float(a_rel_t[ridx]), "D": float(D_pair[ridx]),
                                     "R": float(R_pair[ridx]), "d_plug": float(d_plug[ridx]),
                                     "d_ee": float(d_ee[ridx]), "action": act[t].round(4).tolist()})
        with open(os.path.join(exdir, "examples_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:
        print("example export failed:", repr(exc))

    print(json.dumps({k: rep[k] for k in
                      ["class_distribution_all", "episodes_per_class", "mean_progress_by_mode",
                       "mean_d_plug_by_mode", "mean_gain_by_mode", "action_xyz_vs_dee_corr",
                       "sensitivity_mode2_share"]}, indent=2))


if __name__ == "__main__":
    main()
