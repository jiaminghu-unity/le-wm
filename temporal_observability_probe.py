"""Temporal / action-conditioned tactile observability probes (directive, no training).

Identical split & protocols (ridge alpha=1000 / logistic, episode-held-out), frozen
DINO wrist features from the D0 cache. Probes:

  CURRENT tactile state (target r_cur — current-frame descriptor):
    A  single wrist frame            feat = W[t]
    B  wrist frames t-3..t           feat = [W[t-3..t]]
    C  + actions a_{t-3..t-1}        feat = [W[t-3..t], a[t-3..t-1]]

  FUTURE tactile consequence (inputs end at t, plus the action about to be taken):
    D  (W[t-3..t], a[t-3..t-1], a_t) -> dT_norm (|T_{t+1}-T_t|, scalar R^2)
                                      -> next contact mode (macro-F1)
                                      -> r_tau (joint [T_t, dT] descriptor, R^2)
  Controls: action(+phase) nuisance for every target. Bonus row: both-views temporal.

Run on cw-l4-01: ~/d0venv/bin/python temporal_observability_probe.py
"""
import json
import numpy as np
import zarr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

OUT = "outputs/preflight/usb"
root = zarr.open("data/demo_data/insertion_usb", mode="r")
ends = np.asarray(root["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
N = int(ends[-1]); act = np.asarray(root["data/action"][:])

F = np.load(f"{OUT}/d0_dino_features.npz")
W = F["wrist_mean"]; FR = F["front_mean"]
D_ = np.load("artifacts_usb_tactile_descriptors.npz")
r_tau, r_cur = D_["r_tau"], D_["r_cur"]
drows = D_["rows"].astype(int); dT = D_["dT_norm"]
M = np.load(f"{OUT}/provisional_modes.npz")
mode_row = np.full(N, -2, np.int64); mode_row[M["t"]] = M["mode"]
splits = json.load(open(f"{OUT}/splits.json"))

ep_of = np.searchsorted(ends, drows, side="right")
prog = (drows - starts[ep_of]) / np.maximum(ends[ep_of] - starts[ep_of] - 1, 1)
# full 4-frame window must stay inside the episode
ok = drows - 3 >= starts[ep_of]
rows = drows[ok]; ep = ep_of[ok]; pg = prog[ok]
y_cur = r_cur[ok]; y_tau = r_tau[ok]; y_dT = dT[ok]
y_mode = mode_row[rows]
tr = np.isin(ep, splits["train"]); te = np.isin(ep, splits["test"])
print(f"usable transitions: {len(rows)} (dropped {int((~ok).sum())} at episode starts)")

def stack(feats, idx_offsets):
    return np.concatenate([feats[rows + o] for o in idx_offsets], axis=1)

X_single = W[rows]
X_temp = stack(W, [-3, -2, -1, 0])
A_hist = np.concatenate([act[rows + o] for o in (-3, -2, -1)], axis=1)
X_temp_act = np.concatenate([X_temp, A_hist], axis=1)
a_t = act[rows]
X_cons = np.concatenate([X_temp, A_hist, a_t], axis=1)
X_both_temp = np.concatenate([stack(W, [-3, -2, -1, 0]), stack(FR, [-3, -2, -1, 0]),
                              A_hist, a_t], axis=1)
X_nuis_cur = np.column_stack([a_t, pg])
X_nuis_fut = np.column_stack([A_hist, a_t, pg])

def ridge_r2(X, Y):
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    r = Ridge(alpha=1000.0).fit(Xs[tr], Y[tr])
    p = r.predict(Xs[te]); y = Y[te]
    ss_tot = ((y - y.mean(0)) ** 2).sum()
    return round(float(1 - ((y - p) ** 2).sum() / ss_tot), 4)

def mode_f1(X):
    m = y_mode; keep = m >= 0
    ktr = tr & keep; kte = te & keep
    sc = StandardScaler().fit(X[ktr]); Xs = sc.transform(X)
    best = None
    for C in (0.01, 0.1, 1.0):
        clf = LogisticRegression(max_iter=2000, C=C).fit(Xs[ktr], m[ktr])
        # small val: last 20% of train episodes
        v = f1_score(m[ktr], clf.predict(Xs[ktr]), average="macro")
        if best is None or v > best[0]: best = (v, clf)
    return round(float(f1_score(m[kte], best[1].predict(Xs[kte]), average="macro")), 4)

report = {"n": int(len(rows))}
report["controls"] = {
    "cur_r2_nuisance": ridge_r2(X_nuis_cur, y_cur),
    "fut_dT_r2_nuisance": ridge_r2(X_nuis_fut, y_dT.reshape(-1, 1)),
    "fut_tau_r2_nuisance": ridge_r2(X_nuis_fut, y_tau),
    "fut_mode_f1_nuisance": mode_f1(X_nuis_fut)}
print("controls", report["controls"], flush=True)

report["A_single_frame_cur"] = {"r2_rcur": ridge_r2(X_single, y_cur), "dims": X_single.shape[1]}
report["B_temporal_cur"] = {"r2_rcur": ridge_r2(X_temp, y_cur), "dims": X_temp.shape[1]}
report["C_temporal_actions_cur"] = {"r2_rcur": ridge_r2(X_temp_act, y_cur), "dims": X_temp_act.shape[1]}
for k in ("A_single_frame_cur", "B_temporal_cur", "C_temporal_actions_cur"):
    print(k, report[k], flush=True)

report["D_consequence"] = {
    "dims": X_cons.shape[1],
    "r2_dT_next": ridge_r2(X_cons, y_dT.reshape(-1, 1)),
    "r2_rtau": ridge_r2(X_cons, y_tau),
    "f1_next_mode": mode_f1(X_cons)}
report["E_bothviews_consequence"] = {
    "dims": X_both_temp.shape[1],
    "r2_dT_next": ridge_r2(X_both_temp, y_dT.reshape(-1, 1)),
    "r2_rtau": ridge_r2(X_both_temp, y_tau),
    "f1_next_mode": mode_f1(X_both_temp)}
print("D", report["D_consequence"], flush=True)
print("E", report["E_bothviews_consequence"], flush=True)

json.dump(report, open(f"{OUT}/temporal_observability.json", "w"), indent=2)
print("TEMPORAL PROBE DONE", flush=True)
