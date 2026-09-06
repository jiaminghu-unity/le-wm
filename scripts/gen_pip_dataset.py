"""Build cube_pip.lance: cube_single_expert with (a) the paired scene episode's
frame RENDERED into the bottom-left 48x48 corner of every pixels frame, and
(b) the matching alien_q 26-d column (state of exactly what the thumbnail shows).

Pairing is IDENTICAL to gen_alien_column.py (rng seed 7, cube ep k <- scene ep
perm[k mod M], frame t <- scene frame t mod len), so the existing 48-d ALIEN
Stage-1 gates remain valid for this dataset.

Corner choice: bottom-left 48px, picked from a 400-pair max-motion heatmap over
the full dataset (max frame-diff 39/255 there vs 120+ for 64px corners) so the
overlay never occludes task content; the pip-baseline arm is the empirical
guard for that claim.

    usage: gen_pip_dataset.py --cube <cube.lance> --scene <scene.lance> \
                              --out <cube_pip.lance> [--seed 7]
"""

import argparse
import io
from concurrent.futures import ProcessPoolExecutor

import numpy as np

PATCH = 48
OY, OX = 224 - PATCH, 0  # bottom-left
JPEG_Q = 90


def scene_q26(cols):
    """Numpy mirror of the scene 26-d q builder (same as qgate_stage1)."""
    def flat(k):
        v = cols[k]
        return v.reshape(*v.shape[:-1], -1)[..., :1]
    yaw = cols["proprio/effector_yaw"]
    psi2 = 2.0 * yaw.reshape(*yaw.shape[:-1], -1)[..., :1]
    jp = cols["proprio/joint_pos"]
    joints = jp.reshape(*jp.shape[:-1], -1)[..., :5]
    byaw = cols["privileged/block_0_yaw"]
    th4 = 4.0 * byaw.reshape(*byaw.shape[:-1], -1)[..., :1]
    return np.concatenate([
        cols["proprio/effector_pos"][..., :3],
        np.cos(psi2), np.sin(psi2),
        flat("proprio/gripper_opening"), flat("proprio/gripper_contact"),
        np.cos(joints), np.sin(joints),
        cols["privileged/block_0_pos"][..., :3], np.cos(th4), np.sin(th4),
        flat("privileged/drawer_pos"), flat("privileged/window_pos"),
        flat("privileged/button_0_state"), flat("privileged/button_1_state"),
    ], axis=-1).astype(np.float32)


SCENE_COLS = ["proprio/effector_pos", "proprio/effector_yaw", "proprio/gripper_opening",
              "proprio/gripper_contact", "proprio/joint_pos",
              "privileged/block_0_pos", "privileged/block_0_yaw",
              "privileged/drawer_pos", "privileged/window_pos",
              "privileged/button_0_state", "privileged/button_1_state"]


def _decode_resize(raw):
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(im.resize((PATCH, PATCH), Image.BILINEAR), dtype=np.uint8)


def _paste_encode(args):
    from PIL import Image
    raw, thumb = args
    im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8).copy()
    im[OY:OY + PATCH, OX:OX + PATCH] = thumb
    buf = io.BytesIO()
    Image.fromarray(im).save(buf, format="JPEG", quality=JPEG_Q)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    import lance
    import pyarrow as pa

    # ---- scene: q26 + resized thumbnails, episode slices ----
    sds = lance.dataset(args.scene)
    s_ep = np.concatenate([b.column("episode_idx").to_numpy()
                           for b in sds.to_batches(columns=["episode_idx"])])
    cols = {}
    for c in SCENE_COLS:
        arrs = [np.stack(b.column(c).to_numpy(zero_copy_only=False))
                for b in sds.to_batches(columns=[c])]
        v = np.concatenate(arrs).astype(np.float64)
        cols[c] = v if v.ndim > 1 else v[:, None]
    sq = scene_q26(cols)
    s_ids = np.unique(s_ep)
    s_slices = [np.nonzero(s_ep == e)[0] for e in s_ids]
    M = len(s_slices)
    print(f"[pip] scene q26 ok: {M} eps, {len(s_ep)} frames", flush=True)

    thumbs = np.empty((len(s_ep), PATCH, PATCH, 3), dtype=np.uint8)
    with ProcessPoolExecutor(args.workers) as ex:
        off = 0
        for b in sds.to_batches(columns=["pixels"], batch_size=4096):
            raws = b.column("pixels").to_pylist()
            for i, t in enumerate(ex.map(_decode_resize, raws, chunksize=64)):
                thumbs[off + i] = t
            off += len(raws)
            if off % 40960 == 0:
                print(f"[pip] scene thumbs {off}/{len(s_ep)}", flush=True)
    print("[pip] scene thumbs done", flush=True)

    # ---- cube: pairing ----
    cds = lance.dataset(args.cube)
    c_ep = np.concatenate([b.column("episode_idx").to_numpy()
                           for b in cds.to_batches(columns=["episode_idx"])])
    c_ids = np.unique(c_ep)
    ep_pos = np.searchsorted(c_ep, c_ids)  # episode-major verified by gen_alien run
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(M)
    # per-row: paired scene GLOBAL frame row
    pair_row = np.empty(len(c_ep), dtype=np.int64)
    for k, e in enumerate(c_ids):
        lo = ep_pos[k]
        hi = ep_pos[k + 1] if k + 1 < len(c_ids) else len(c_ep)
        srow = s_slices[perm[k % M]]
        pair_row[lo:hi] = srow[np.arange(hi - lo) % len(srow)]
    alien = sq[pair_row].astype(np.float32)
    print(f"[pip] pairing done: {len(c_ids)} cube eps", flush=True)

    # ---- streaming rewrite: replace pixels, append alien_q ----
    schema = cds.schema
    fields = list(schema)
    out_schema = pa.schema(fields + [pa.field("alien_q", pa.list_(pa.float32(), 26))],
                           metadata=schema.metadata)
    ex = ProcessPoolExecutor(args.workers)
    state = {"off": 0}

    def batches():
        for b in cds.to_batches(batch_size=2048):
            off = state["off"]; m = b.num_rows
            raws = b.column(b.schema.get_field_index("pixels")).to_pylist()
            th = thumbs[pair_row[off:off + m]]
            new_pix = list(ex.map(_paste_encode, zip(raws, th), chunksize=32))
            arrays = []
            for f in fields:
                if f.name == "pixels":
                    arrays.append(pa.array(new_pix, type=f.type))
                else:
                    arrays.append(b.column(b.schema.get_field_index(f.name)))
            arrays.append(pa.array(list(alien[off:off + m]),
                                   type=pa.list_(pa.float32(), 26)))
            state["off"] = off + m
            if (off // 2048) % 50 == 0:
                print(f"[pip] rewrite {off}/{len(c_ep)}", flush=True)
            yield pa.record_batch(arrays, schema=out_schema)

    lance.write_dataset(batches(), args.out, schema=out_schema,
                        max_rows_per_file=200000)
    ex.shutdown()

    # ---- verify ----
    ods = lance.dataset(args.out)
    assert ods.count_rows() == len(c_ep)
    t = ods.take([0, len(c_ep) // 2, len(c_ep) - 1], columns=["pixels", "alien_q"])
    from PIL import Image
    for raw in t.column("pixels").to_pylist():
        im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        assert im.shape == (224, 224, 3)
        assert im[OY:OY + PATCH, OX:OX + PATCH].std() > 1.0  # thumbnail present
    a = np.stack(t.column("alien_q").to_numpy(zero_copy_only=False))
    assert a.shape[-1] == 26 and np.isfinite(a).all()
    print(f"[pip] verified: {ods.count_rows()} rows, thumbnail + alien_q ok", flush=True)


if __name__ == "__main__":
    main()
