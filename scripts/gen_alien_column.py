"""Build the ALIEN-q column for the structured-noise experiment (2026-09-02):
cube task, q = [cube 22-d ; scene 26-d taken from REAL scene trajectories].

For cube episode i we attach scene episode (i mod M); cube frame t reads scene
frame (t mod len_scene_ep). This preserves the scene dims' temporal structure --
the whole point: unlike iid noise they are smooth in time, so their pairwise
distances correlate with |t1-t2| exactly like cube's own q distances do, giving
the ungated Pearson metric a spurious-alignment channel that iid noise lacks.

Writes the (N_frames, 26) float32 column into a LOCAL COPY of
cube_single_expert.lance via lance merge on (episode_idx, step_idx); the caller
uploads the merged dataset as datasets/ogbench/cube_alien.lance.

    usage: gen_alien_column.py --cube <cube.lance> --scene <scene.lance> [--seed 7]
"""

import argparse

import numpy as np


def scene_q26(cols):
    """Numpy mirror of the scene 26-d builder (same as qgate_stage1)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True, help="LOCAL cube lance dir (will be merged in place)")
    ap.add_argument("--scene", required=True, help="scene lance dir")
    ap.add_argument("--seed", type=int, default=7, help="scene-episode pairing shuffle seed")
    args = ap.parse_args()

    import lance
    import pyarrow as pa
    from stable_worldmodel.data.formats.lance import LanceDataset

    sc = LanceDataset(path=args.scene, keys_to_load=SCENE_COLS)
    s_ep = np.asarray(sc.get_col_data("episode_idx")).reshape(-1)
    cols = {c: np.asarray(sc.get_col_data(c), dtype=np.float64) for c in SCENE_COLS}
    for c in list(cols):
        if cols[c].ndim == 1:
            cols[c] = cols[c][:, None]
    sq = scene_q26(cols)                                  # (Ns, 26)
    # per-episode frame slices of the scene data
    s_ids = np.unique(s_ep)
    s_slices = [np.nonzero(s_ep == e)[0] for e in s_ids]
    M = len(s_slices)
    print(f"[alien] scene: {M} episodes, {len(s_ep)} frames, q26 ok", flush=True)

    cb = LanceDataset(path=args.cube, keys_to_load=["action"])
    c_ep = np.asarray(cb.get_col_data("episode_idx")).reshape(-1)
    c_st = np.asarray(cb.get_col_data("step_idx")).reshape(-1)
    order = np.lexsort((c_st, c_ep))
    assert (order == np.arange(len(c_ep))).all(), "cube lance not episode-major"
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(M)                              # fixed random pairing
    alien = np.empty((len(c_ep), 26), dtype=np.float32)
    c_ids = np.unique(c_ep)
    for k, e in enumerate(c_ids):
        rows = np.nonzero(c_ep == e)[0]
        srow = s_slices[perm[k % M]]
        alien[rows] = sq[srow[np.arange(len(rows)) % len(srow)]]
    print(f"[alien] cube: {len(c_ids)} episodes, {len(c_ep)} frames filled", flush=True)

    tbl = pa.table({
        "episode_idx": pa.array(c_ep),
        "step_idx": pa.array(c_st),
        "alien_q": pa.array(list(alien), type=pa.list_(pa.float32(), 26)),
    })
    try:
        ds3 = lance.dataset(args.cube)
        ds3.merge(tbl, left_on=["episode_idx", "step_idx"],
                  right_on=["episode_idx", "step_idx"])
        print("[alien] lance multi-key merge ok", flush=True)
    except Exception as e:
        print(f"[alien] multi-key merge unsupported ({e}); falling back to add_columns", flush=True)
        state = {"off": 0}
        def udf(batch):
            off = state["off"]; m = len(batch)
            out = pa.array(list(alien[off:off + m]), type=pa.list_(pa.float32(), 26))
            state["off"] = off + m
            return pa.record_batch([out], names=["alien_q"])
        ds3 = lance.dataset(args.cube)
        ds3.add_columns(udf, read_columns=["episode_idx"])
        print("[alien] add_columns ok", flush=True)

    chk = LanceDataset(path=args.cube, keys_to_load=["alien_q"])
    a = np.asarray(chk.get_col_data("alien_q"))
    assert a.shape == (len(c_ep), 26), a.shape
    assert np.isfinite(a).all() and a.std() > 0.01
    print(f"[alien] verified: shape {a.shape}, std {a.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
