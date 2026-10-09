"""P0 — Dataset, synchronization, and episode-split audit (proposal §5).

Audits the raw Zarr store directly (no silent aliases), verifies temporal
alignment invariants, emits the canonical episode-level split, and exports
sample sequences for eyeballing.

Run on the machine that holds the dataset:
  python -m preflight.audit_dataset \
      --data-root data/demo_data/insertion_usb \
      --task insertion_usb \
      --output-dir outputs/preflight/usb \
      --make-splits --seed 3072
"""
import argparse, json, os, sys
import numpy as np

try:
    import zarr
except ImportError:
    sys.exit("zarr is required (pip install zarr)")


def walk_arrays(group, prefix=""):
    out = {}
    for k, v in group.items():
        path = f"{prefix}{k}"
        if hasattr(v, "shape"):
            out[path] = v
        else:
            out.update(walk_arrays(v, prefix=path + "/"))
    return out


def stats(arr, sample_cap=2_000_000):
    """Finite-aware stats on a (possibly subsampled) flat view."""
    n = int(np.prod(arr.shape))
    if n == 0:
        return {"empty": True}
    if n > sample_cap:
        # even stride subsample along axis 0 to bound memory
        step = max(1, arr.shape[0] // max(1, sample_cap // int(np.prod(arr.shape[1:]) or 1)))
        x = np.asarray(arr[::step])
    else:
        x = np.asarray(arr[:])
    xf = x.astype(np.float64, copy=False).ravel()
    finite = np.isfinite(xf)
    return {
        "min": float(np.min(xf[finite])) if finite.any() else None,
        "max": float(np.max(xf[finite])) if finite.any() else None,
        "mean": float(np.mean(xf[finite])) if finite.any() else None,
        "std": float(np.std(xf[finite])) if finite.any() else None,
        "nan": int(np.isnan(xf).sum()),
        "inf": int(np.isinf(xf).sum()),
        "subsampled": n > sample_cap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--n-sample-seqs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    root = zarr.open(args.data_root, mode="r")
    arrays = walk_arrays(root)

    manifest = {"task": args.task, "data_root": args.data_root, "arrays": {}}
    for path, a in sorted(arrays.items()):
        entry = {"shape": list(a.shape), "dtype": str(a.dtype)}
        # cheap stats for small/medium arrays; skip videos' full stats but keep range probe
        if "front" in path or "wrist" in path:
            probe = np.asarray(a[0]) if a.shape[0] else np.zeros(1)
            entry["frame0_min"] = float(probe.min()); entry["frame0_max"] = float(probe.max())
        else:
            entry["stats"] = stats(a)
        manifest["arrays"][path] = entry

    # ---- episode structure ----
    ee_key = next((k for k in arrays if k.endswith("episode_ends")), None)
    assert ee_key, f"meta/episode_ends not found; keys={list(arrays)}"
    episode_ends = np.asarray(arrays[ee_key][:]).astype(int)
    n_ep = len(episode_ends)
    starts = np.concatenate([[0], episode_ends[:-1]])
    lengths = episode_ends - starts
    n_frames = int(episode_ends[-1])
    n_transitions = int((lengths - 1).sum())
    manifest["episodes"] = {
        "count": n_ep,
        "total_frames": n_frames,
        "valid_transitions": n_transitions,
        "len_min": int(lengths.min()), "len_max": int(lengths.max()),
        "len_mean": float(lengths.mean()),
        "monotonic_ends": bool(np.all(np.diff(episode_ends) > 0)),
    }

    # ---- alignment invariants (§5.2) ----
    align = {}
    data_keys = [k for k in arrays if k.split("/")[-1] in
                 ("action", "front", "wrist") or "tactile" in k]
    lens = {k: arrays[k].shape[0] for k in data_keys}
    align["first_axis_lengths"] = lens
    align["all_lengths_equal"] = len(set(lens.values())) == 1
    align["lengths_match_episode_ends"] = all(v == n_frames for v in lens.values())
    # Episode boundary rule: a transition (t -> t+1) is valid iff both indices
    # fall inside one episode; count how many raw consecutive pairs would cross.
    crossing = n_ep - 1  # by construction: one crossing pair per boundary
    align["boundary_crossing_pairs_excluded"] = crossing
    align["note"] = ("Both cameras and tactile share the first axis with action; "
                     "per-row sampling therefore uses identical indices across keys. "
                     "Transitions must be generated within [start, end) per episode "
                     "so no pair crosses episode boundaries.")
    manifest["alignment"] = align

    # ---- required-key checklist for USB (§5.1) ----
    want = ["front", "wrist", "action", "tactile_force_field_right", "dof_pos",
            "dof_vel", "ee_pos", "plug_pos", "socket_pos_gt", "plug_quat", "socket_quat"]
    have = {w: any(k.split("/")[-1] == w for k in arrays) for w in want}
    manifest["required_keys_present"] = have

    with open(os.path.join(args.output_dir, "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- canonical episode-level split (§5.3) ----
    if args.make_splits:
        rng = np.random.RandomState(args.seed)
        order = rng.permutation(n_ep)
        n_tr = int(round(args.train_frac * n_ep))
        n_va = int(round(args.val_frac * n_ep))
        splits = {
            "seed": args.seed,
            "train": sorted(order[:n_tr].tolist()),
            "val": sorted(order[n_tr:n_tr + n_va].tolist()),
            "test": sorted(order[n_tr + n_va:].tolist()),
        }
        assert not (set(splits["train"]) & set(splits["val"]) & set(splits["test"]))
        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == n_ep
        with open(os.path.join(args.output_dir, "splits.json"), "w") as f:
            json.dump(splits, f, indent=2)

    # ---- sample sequences for visual inspection (§5.2 deliverable) ----
    seq_dir = os.path.join(args.output_dir, "sample_sequences")
    os.makedirs(seq_dir, exist_ok=True)
    try:
        from PIL import Image
        front = arrays[next(k for k in arrays if k.split("/")[-1] == "front")]
        wrist = arrays[next(k for k in arrays if k.split("/")[-1] == "wrist")]
        action = arrays[next(k for k in arrays if k.split("/")[-1] == "action")]
        rng = np.random.RandomState(0)
        for ep in rng.choice(n_ep, size=min(args.n_sample_seqs, n_ep), replace=False):
            s, e = int(starts[ep]), int(episode_ends[ep])
            t0 = s + (e - s) // 2
            rows = []
            for t in range(t0, min(t0 + 3, e)):
                fr, wr = np.asarray(front[t]), np.asarray(wrist[t])
                if fr.ndim == 3 and fr.shape[0] in (1, 3):  # CHW -> HWC
                    fr, wr = fr.transpose(1, 2, 0), wr.transpose(1, 2, 0)
                if fr.dtype != np.uint8:
                    fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
                    wr = (np.clip(wr, 0, 1) * 255).astype(np.uint8)
                rows.append(np.concatenate([fr, wr], axis=1))
            img = np.concatenate(rows, axis=0)
            Image.fromarray(img).save(os.path.join(seq_dir, f"ep{ep}_t{t0}.png"))
            acts = np.asarray(action[t0:min(t0 + 3, e)]).tolist()
            with open(os.path.join(seq_dir, f"ep{ep}_t{t0}_actions.json"), "w") as f:
                json.dump(acts, f)
    except Exception as exc:  # sample export is best-effort; audit numbers above are not
        with open(os.path.join(seq_dir, "EXPORT_FAILED.txt"), "w") as f:
            f.write(repr(exc))

    print(json.dumps({
        "episodes": manifest["episodes"],
        "required_keys_present": have,
        "alignment_ok": align["all_lengths_equal"] and align["lengths_match_episode_ends"],
    }, indent=2))


if __name__ == "__main__":
    main()
