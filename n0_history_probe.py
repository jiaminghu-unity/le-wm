"""N0 — does visual/motor HISTORY add recoverable contact information? (no training)

Frozen P3 features (official-encoder wrist/front 512-d per row), identical
episode-level split (canonical seed 3072), linear probes throughout.
Common sample set: rows t with t-11 >= ep_start and t+1 < ep_end, so every
input row scores the SAME samples. History never reads past t; the candidate
action a_t is the action about to be executed.

Target A: mode of transition t->t+1 (0/1/2; ambiguous -1 excluded) —
          multinomial logistic; macro-F1, per-class F1, mode-2 AUPRC, 95% CI.
Target B: next-step motion response y=[dee_pos, dplug_pos] (6-d) — ridge;
          held-out R^2 overall and grouped by mode / transition-adjacency.

  python3 n0_history_probe.py --rows act_now act_h12 wrist_now wrist_h3 wrist_h6 wrist_h12 fw_h6
  python3 n0_history_probe.py --rows lewm_h lewm_zhat z_joint --lewm-npz outputs/n0_lewm_states.npz
"""
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", required=True)
ap.add_argument("--lewm-npz", default="")
ap.add_argument("--out", default="outputs/n0_history_probe.json")
args = ap.parse_args()

import zarr
zr = zarr.open("data/demo_data/insertion_usb", mode="r")
ends = np.asarray(zr["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
N = int(ends[-1])
ep_of = np.zeros(N, dtype=int)
for e in range(len(ends)):
    ep_of[starts[e]:ends[e]] = e
phase = np.zeros(N, dtype=np.float32)
for e in range(len(ends)):
    L = ends[e] - starts[e]
    phase[starts[e]:ends[e]] = np.arange(L, dtype=np.float32) / max(L - 1, 1)

W = np.load("artifacts/usb/wrist_features.npy")
F = np.load("artifacts/usb/vision_features.npy")
A = np.load("artifacts/usb/action.npy")
M = np.load("cluster_bundle/outputs/preflight/usb/provisional_modes.npz")
mode_row = np.full(N, -2, dtype=np.int64)
mode_row[M["t"]] = M["mode"]
ee = np.asarray(zr["data/ee_pos"][:], dtype=np.float32)
plug = np.asarray(zr["data/plug_pos"][:], dtype=np.float32)

splits = json.load(open("artifacts/usb/splits_canonical.json"))

# common sample set
ok = np.zeros(N, dtype=bool)
for e in range(len(ends)):
    s, en = starts[e], ends[e]
    ok[s + 11:en - 1] = True
T_IDX = np.where(ok)[0]
lab = mode_row[T_IDX]
nxt = mode_row[np.minimum(T_IDX + 1, N - 1)]
trans_adj = (lab >= 0) & (nxt >= 0) & (lab != nxt)
y_motion = np.concatenate([ee[T_IDX + 1] - ee[T_IDX], plug[T_IDX + 1] - plug[T_IDX]], axis=1)
ep_t = ep_of[T_IDX]
in_tr = np.isin(ep_t, splits["train"]); in_va = np.isin(ep_t, splits["val"]); in_te = np.isin(ep_t, splits["test"])
print(f"common set: {len(T_IDX)} rows | train {in_tr.sum()} val {in_va.sum()} test {in_te.sum()}", flush=True)

def hist(feat, k):
    return np.concatenate([feat[T_IDX - j] for j in range(k - 1, -1, -1)], axis=1)

def build(row):
    ph = phase[T_IDX][:, None]
    if row == "act_now":    return np.concatenate([A[T_IDX], ph], 1)
    if row == "act_h12":    return np.concatenate([hist(A, 12), ph], 1)
    if row == "wrist_now":  return np.concatenate([W[T_IDX], A[T_IDX], ph], 1)
    if row == "wrist_h3":   return np.concatenate([hist(W, 3), hist(A, 3), ph], 1)
    if row == "wrist_h6":   return np.concatenate([hist(W, 6), hist(A, 6), ph], 1)
    if row == "wrist_h12":  return np.concatenate([hist(W, 12), hist(A, 12), ph], 1)
    if row == "fw_h6":      return np.concatenate([hist(F, 6), hist(W, 6), hist(A, 6), ph], 1)
    if row in ("lewm_h", "lewm_zhat", "z_joint"):
        D = np.load(args.lewm_npz)
        pos = {int(r): i for i, r in enumerate(D["rows"])}
        sel = np.array([pos[int(t)] for t in T_IDX])
        key = {"lewm_h": "h_pred", "lewm_zhat": "z_hat", "z_joint": "z_joint"}[row]
        return np.concatenate([D[key][sel], A[T_IDX], ph], 1)
    raise ValueError(row)

res = {}
for row in args.rows:
    X = build(row).astype(np.float32)
    sc = StandardScaler().fit(X[in_tr])
    Xs = sc.transform(X)
    out = {"dim": int(X.shape[1])}

    # --- target A: mode classification (ambiguous excluded) ---
    m_tr = in_tr & (lab >= 0); m_va = in_va & (lab >= 0); m_te = in_te & (lab >= 0)
    best = None
    for C in (0.05, 0.5):
        clf = LogisticRegression(max_iter=2000, C=C, n_jobs=-1)
        clf.fit(Xs[m_tr], lab[m_tr])
        f1v = f1_score(lab[m_va], clf.predict(Xs[m_va]), average="macro")
        if best is None or f1v > best[0]: best = (f1v, C, clf)
    f1v, C, clf = best
    pred = clf.predict(Xs[m_te]); prob = clf.predict_proba(Xs[m_te])
    yte = lab[m_te]
    out["mode"] = {
        "C": C, "val_macro_f1": round(float(f1v), 4),
        "test_macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "per_class_f1": {str(k): round(float(f), 4) for k, f in
                         zip((0, 1, 2), f1_score(yte, pred, labels=[0, 1, 2], average=None, zero_division=0))},
    }
    cls = list(clf.classes_)
    if 2 in cls:
        out["mode"]["mode2_auprc"] = round(float(average_precision_score(
            (yte == 2).astype(int), prob[:, cls.index(2)])), 4)
    rngb = np.random.RandomState(0); boots = []
    for _ in range(400):
        bi = rngb.randint(0, len(yte), len(yte))
        boots.append(f1_score(yte[bi], pred[bi], average="macro"))
    out["mode"]["macro_f1_ci95"] = [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)]

    # --- target B: next-step motion response (all rows) ---
    mu, sd = y_motion[in_tr].mean(0), y_motion[in_tr].std(0) + 1e-9
    Y = (y_motion - mu) / sd
    bestr = None
    for al in (1.0, 10.0, 100.0, 1000.0):
        rg = Ridge(alpha=al).fit(Xs[in_tr], Y[in_tr])
        r2v = 1 - ((Y[in_va] - rg.predict(Xs[in_va])) ** 2).mean() / (Y[in_va] ** 2).mean()
        if bestr is None or r2v > bestr[0]: bestr = (r2v, al, rg)
    r2v, al, rg = bestr
    P = rg.predict(Xs[in_te]); E = (Y[in_te] - P) ** 2
    r2 = lambda sel: round(float(1 - E[sel].mean() / (Y[in_te][sel] ** 2).mean()), 4) if sel.any() else None
    allsel = np.ones(in_te.sum(), bool)
    lab_te, tr_te = lab[in_te], trans_adj[in_te]
    out["motion"] = {"alpha": al, "val_r2": round(float(r2v), 4), "test_r2": r2(allsel),
                     "test_r2_ee": round(float(1 - E[:, :3].mean() / (Y[in_te][:, :3] ** 2).mean()), 4),
                     "test_r2_plug": round(float(1 - E[:, 3:].mean() / (Y[in_te][:, 3:] ** 2).mean()), 4),
                     "by_mode": {str(m): r2(lab_te == m) for m in (0, 1, 2)},
                     "transition_adjacent": r2(tr_te), "non_transition": r2((lab_te >= 0) & ~tr_te)}
    res[row] = out
    print(row, json.dumps(out), flush=True)

import os
os.makedirs("outputs", exist_ok=True)
prev = {}
if os.path.exists(args.out):
    prev = json.load(open(args.out))
prev.update(res)
json.dump(prev, open(args.out, "w"), indent=2)
print("N0 PROBE DONE", flush=True)
