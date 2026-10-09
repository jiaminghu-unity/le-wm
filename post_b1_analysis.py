"""Post-B1 analysis (directive §3/§4/§6) — runs automatically when B1 finishes.

Test F: tactile-descriptor probes at A=legacy wrist, B=B1 pre-pool wrist tokens,
        C=B1 joint latent — each as (feat, a_t, phase) -> descriptor, with
        incremental R^2 over the (a_t, phase) nuisance probe, fraction of positive
        per-episode gains, episode bootstrap.
M5:     mode probes — action+phase / legacy front+a / legacy wrist+a / B1 latent+a.
WM:     mode-conditioned 1-step & multi-step (h=2,4,6) latent error vs persistence,
        plus action-magnitude x phase bin-matched comparisons.
Run from ~/ContactWorld (trainer VM) after outputs/b1/ckpt_best.pt exists.
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
import zarr

from touchwm.lewm_adapter import MultiViewJEPA

dev = "cuda"; OUT = "outputs/b1"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

model = MultiViewJEPA().to(dev)
ck = torch.load(f"{OUT}/ckpt_best.pt", map_location=dev)
model.load_state_dict(ck["model"]); model.eval()
print("loaded ckpt_best step", ck.get("step"), flush=True)

root = zarr.open("data/demo_data/insertion_usb", mode="r")
act = np.asarray(root["data/action"][:])
ends = np.asarray(root["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
N = int(ends[-1])
front_z = root["data/front"]; wrist_z = root["data/wrist"]
splits = json.load(open("outputs/preflight/usb/splits.json"))

# ---- encode all frames: joint latent Z and pre-pool wrist mean W1 ----
Z = np.zeros((N, 192), np.float32); W1 = np.zeros((N, 192), np.float32)
enc = model.mv_encoder
with torch.no_grad():
    for i in range(0, N, 64):
        f = torch.from_numpy(np.asarray(front_z[i:i+64])).permute(0,3,1,2).float()
        w = torch.from_numpy(np.asarray(wrist_z[i:i+64])).permute(0,3,1,2).float()
        f = Fn.interpolate(f, size=(224,224), mode="bilinear", antialias=True).to(dev)
        w = Fn.interpolate(w, size=(224,224), mode="bilinear", antialias=True).to(dev)
        Ft = enc.vit(f) + enc.view_emb[0]; Wt = enc.vit(w) + enc.view_emb[1]
        Fp, Wp = Ft, Wt
        for caf, caw in zip(enc.ca_f, enc.ca_w):
            Fp2 = caf(Fp, Wt); Wp2 = caw(Wp, Ft); Fp, Wp = Fp2, Wp2
        U = enc.pool(torch.cat([Fp, Wp], 1))
        z = enc.project(U.flatten(1))
        B_, T_, D_ = 1, 1, 192
        zp = model.projector(z)
        Z[i:i+64] = zp.cpu().numpy()
        W1[i:i+64] = Wp.mean(1).cpu().numpy()
np.save(f"{OUT}/b1_joint_latent.npy", Z); np.save(f"{OUT}/b1_prepool_wrist.npy", W1)
print("encoded all frames", flush=True)

ep_of_row = np.searchsorted(ends, np.arange(N), side="right")
prog_row = (np.arange(N) - starts[ep_of_row]) / np.maximum(ends[ep_of_row]-starts[ep_of_row]-1, 1)
D = np.load("artifacts_usb_tactile_descriptors.npz"); desc = D[sorted(D.files)[0]]
Wleg = np.load("artifacts_usb_wrist_features.npy")
Fleg = np.load("artifacts_usb_front_features.npy") if os.path.exists("artifacts_usb_front_features.npy") else None
rows = np.arange(min(len(desc), N))
tr = np.isin(ep_of_row[rows], splits["train"]); te = np.isin(ep_of_row[rows], splits["test"])
report = {"ckpt_step": int(ck.get("step", -1))}

def ridge_probe(X):
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    best = None
    for al in (100.0, 1000.0, 10000.0):
        r = Ridge(alpha=al).fit(Xs[tr], desc[rows][tr])
        # alpha selected on train-internal split: last 20% of train rows
        cut = int(tr.sum()*0.8)
        idx = np.where(tr)[0]
        r2v = 1 - ((desc[rows][idx[cut:]] - r.predict(Xs[idx[cut:]]))**2).sum() / \
                  ((desc[rows][idx[cut:]] - desc[rows][idx[:cut]].mean(0))**2).sum()
        if best is None or r2v > best[0]: best = (r2v, al, r)
    _, al, r = best
    pred = r.predict(Xs[te])
    y = desc[rows][te]
    r2 = float(1 - ((y-pred)**2).sum() / ((y-y.mean(0))**2).sum())
    # per-episode positive-gain fraction computed by caller (needs baseline preds)
    return r2, al, pred

base_X = np.column_stack([act[rows], prog_row[rows]])
r2_base, _, pred_base = ridge_probe(base_X)
report["testF_nuisance_action_phase_r2"] = round(r2_base, 4)
y_te = desc[rows][te]; ep_te = ep_of_row[rows][te]
def per_ep_gain(pred):
    gains = []
    for e in np.unique(ep_te):
        m = ep_te == e
        ssr = ((y_te[m]-pred[m])**2).sum(); ssb = ((y_te[m]-pred_base[m])**2).sum()
        gains.append(ssb - ssr)
    g = np.array(gains)
    return float((g > 0).mean()), g

rng = np.random.RandomState(0)
for name, X in [("A_legacy_wrist", np.column_stack([Wleg[rows], base_X])),
                ("B_b1_prepool_wrist", np.column_stack([W1[rows], base_X])),
                ("C_b1_joint_latent", np.column_stack([Z[rows], base_X]))]:
    r2, al, pred = ridge_probe(X)
    frac, g = per_ep_gain(pred)
    boots = [np.mean(g[rng.randint(0, len(g), len(g))] > 0) for _ in range(300)]
    report[f"testF_{name}"] = {
        "r2": round(r2, 4), "alpha": al, "dims": X.shape[1],
        "incremental_r2_over_nuisance": round(r2 - r2_base, 4),
        "episode_positive_gain_frac": round(frac, 3),
        "frac_ci95": [round(float(np.percentile(boots, 2.5)), 3),
                      round(float(np.percentile(boots, 97.5)), 3)]}
    print("testF", name, report[f"testF_{name}"], flush=True)

# ---- M5 mode probes ----
M = np.load("outputs/preflight/usb/provisional_modes.npz")
mode, t_of, ep_of = M["mode"], M["t"], M["ep"]
keep = mode >= 0
m_, t_, e_ = mode[keep], t_of[keep], ep_of[keep]
ktr = np.isin(e_, splits["train"]); kva = np.isin(e_, splits["val"]); kte = np.isin(e_, splits["test"])
prog_t = prog_row[t_]
def mode_probe(X):
    sc = StandardScaler().fit(X[ktr]); Xs = sc.transform(X)
    best = None
    for C in (0.01, 0.1, 1.0):
        clf = LogisticRegression(max_iter=2000, C=C).fit(Xs[ktr], m_[ktr])
        v = f1_score(m_[kva], clf.predict(Xs[kva]), average="macro")
        if best is None or v > best[0]: best = (v, clf)
    clf = best[1]
    pred = clf.predict(Xs[kte]); proba = clf.predict_proba(Xs[kte])
    cls = list(clf.classes_)
    return {"macro_f1": round(f1_score(m_[kte], pred, average="macro"), 4),
            "mode2_auprc": round(float(average_precision_score((m_[kte]==2).astype(int),
                                                               proba[:, cls.index(2)])), 4),
            "confusion": confusion_matrix(m_[kte], pred, labels=[0,1,2]).tolist()}
inputs = {"m5_action_phase": np.column_stack([act[t_], prog_t]),
          "m5_legacy_wrist_action": np.column_stack([Wleg[t_], act[t_]]),
          "m5_b1_latent_action": np.column_stack([Z[t_], act[t_]]),
          "m5_b1_prepool_wrist_action": np.column_stack([W1[t_], act[t_]])}
if Fleg is not None:
    inputs["m5_legacy_front_action"] = np.column_stack([Fleg[t_], act[t_]])
for name, X in inputs.items():
    report[name] = mode_probe(X)
    print(name, report[name], flush=True)

# ---- WM mode-conditioned errors (1-step + rollout h=2,4,6) vs persistence ----
H = 3
res = {m: {h: {"pred": [], "pers": []} for h in (1, 2, 4, 6)} for m in (0, 1, 2)}
with torch.no_grad():
    for mi, ti, ei, is_te in zip(m_, t_, e_, kte):
        if not is_te: continue
        s, e_end = int(starts[ei]), int(ends[ei])
        t0 = max(int(ti) - H + 1, s)
        hist = list(range(t0, int(ti) + 1))
        if len(hist) < 2 or int(ti) + 6 >= e_end: continue
        zz = torch.from_numpy(Z[hist][None]).to(dev)
        for h in (1, 2, 4, 6):
            zcur = zz.clone()
            for k in range(h):
                a_idx = list(range(t0, t0 + zcur.shape[1]))   # actions aligned to frames
                aw = torch.from_numpy(act[a_idx][None]).float().to(dev)
                pr = model.predict(zcur[:, -H:], aw[:, -H:])[:, -1:]
                zcur = torch.cat([zcur, pr], 1)
            tgt = Z[int(ti)+h]
            res[int(mi)][h]["pred"].append(float(((zcur[0, -1].cpu().numpy()-tgt)**2).mean()))
            res[int(mi)][h]["pers"].append(float(((Z[int(ti)]-tgt)**2).mean()))
report["wm_mode_errors"] = {
    str(m): {f"h{h}": {"n": len(v["pred"]),
                       "pred_mse": round(float(np.mean(v["pred"])), 6),
                       "pers_mse": round(float(np.mean(v["pers"])), 6),
                       "ratio": round(float(np.mean(v["pred"])/max(np.mean(v["pers"]),1e-12)), 3)}
             for h, v in hs.items() if v["pred"]}
    for m, hs in res.items()}
print(json.dumps(report["wm_mode_errors"], indent=1), flush=True)

json.dump(report, open(f"{OUT}/post_b1_analysis.json", "w"), indent=2)
print("POST B1 ANALYSIS DONE", flush=True)
