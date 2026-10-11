"""Experiment B — where exactly is the wrist contact information lost? (no retraining)

Probes FOUR feature points of the trained B1 encoder with IDENTICAL protocols:
  P1  W_pre   : wrist ViT tokens mean, BEFORE cross-attention
  P2  W_cross : wrist tokens mean, AFTER cross-attention (pre-pooling)
  P3  z_pool  : pooled joint latent (before projector MLP)
  P4  z_joint : final joint latent (after projector; what predictor/CEM see)
plus front counterparts (F_pre, F_cross) for reference.

Probe 1 (regression): ridge alpha=1000, target = PCA-32 tactile descriptor r_tau,
  (feat, a_t, phase) inputs, episode-level split, incremental R^2 over nuisance.
Probe 2 (modes): multinomial logistic, target = validated contact mode (t -> t+1).
Pair check: for test transitions with similar FRONT features but different modes,
  measure wrist/joint latent separation (does the joint space still distinguish
  contact situations the front view cannot?).

Run on a GPU box from ~/ContactWorld:
  python exp_b_probe_layers.py outputs_b1_ckpt_100k.pt
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import zarr

from touchwm.lewm_adapter import MultiViewJEPA

CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs_b1_ckpt_100k.pt"
dev = "cuda"
torch.manual_seed(0)
model = MultiViewJEPA().to(dev)
st = torch.load(CKPT, map_location=dev)
model.load_state_dict(st["model"] if "model" in st else st)
model.eval()
print("ckpt step", st.get("step"), flush=True)

root = zarr.open("data/demo_data/insertion_usb", mode="r")
act = np.asarray(root["data/action"][:])
ends = np.asarray(root["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
N = int(ends[-1])
front_z, wrist_z = root["data/front"], root["data/wrist"]

P = {k: np.zeros((N, 192), np.float32)
     for k in ("W_pre", "W_cross", "F_pre", "F_cross", "z_pool", "z_joint")}
enc = model.mv_encoder
with torch.no_grad():
    for i in range(0, N, 64):
        f = torch.from_numpy(np.asarray(front_z[i:i+64])).permute(0,3,1,2).float()
        w = torch.from_numpy(np.asarray(wrist_z[i:i+64])).permute(0,3,1,2).float()
        f = Fn.interpolate(f, size=(224,224), mode="bilinear", antialias=True).to(dev)
        w = Fn.interpolate(w, size=(224,224), mode="bilinear", antialias=True).to(dev)
        Ft = enc.vit(f) + enc.view_emb[0]; Wt = enc.vit(w) + enc.view_emb[1]
        P["F_pre"][i:i+64] = Ft.mean(1).cpu().numpy()
        P["W_pre"][i:i+64] = Wt.mean(1).cpu().numpy()
        Fp, Wp = Ft, Wt
        for caf, caw in zip(enc.ca_f, enc.ca_w):
            Fp2 = caf(Fp, Wt); Wp2 = caw(Wp, Ft); Fp, Wp = Fp2, Wp2
        P["F_cross"][i:i+64] = Fp.mean(1).cpu().numpy()
        P["W_cross"][i:i+64] = Wp.mean(1).cpu().numpy()
        U = enc.pool(torch.cat([Fp, Wp], 1))
        zp = enc.project(U.flatten(1))
        P["z_pool"][i:i+64] = zp.cpu().numpy()
        P["z_joint"][i:i+64] = model.projector(zp).cpu().numpy()
print("encoded 6 probe points", flush=True)

splits = json.load(open("outputs/preflight/usb/splits.json"))
D = np.load("artifacts_usb_tactile_descriptors.npz")
desc, drows = D["r_tau"], D["rows"].astype(int)
ep_of_row = np.searchsorted(ends, np.arange(N), side="right")
prog_row = (np.arange(N) - starts[ep_of_row]) / np.maximum(ends[ep_of_row]-starts[ep_of_row]-1, 1)
tr = np.isin(ep_of_row[drows], splits["train"]); te = np.isin(ep_of_row[drows], splits["test"])
base = np.column_stack([act[drows], prog_row[drows]])

def ridge_r2(X):
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    r = Ridge(alpha=1000.0).fit(Xs[tr], desc[tr])
    p = r.predict(Xs[te]); y = desc[te]
    return float(1 - ((y-p)**2).sum() / ((y-y.mean(0))**2).sum())
r2_base = ridge_r2(base)

M = np.load("outputs/preflight/usb/provisional_modes.npz")
mode, t_of, ep_of = M["mode"], M["t"], M["ep"]
keep = mode >= 0
m_, t_, e_ = mode[keep], t_of[keep], ep_of[keep]
ktr = np.isin(e_, splits["train"]); kva = np.isin(e_, splits["val"]); kte = np.isin(e_, splits["test"])

def mode_f1(X):
    sc = StandardScaler().fit(X[ktr]); Xs = sc.transform(X)
    best = None
    for C in (0.01, 0.1, 1.0):
        clf = LogisticRegression(max_iter=2000, C=C).fit(Xs[ktr], m_[ktr])
        v = f1_score(m_[kva], clf.predict(Xs[kva]), average="macro")
        if best is None or v > best[0]: best = (v, clf)
    return float(f1_score(m_[kte], best[1].predict(Xs[kte]), average="macro"))

report = {"ckpt_step": int(st.get("step", -1)), "nuisance_r2": round(r2_base, 4)}
for k in ("F_pre", "F_cross", "W_pre", "W_cross", "z_pool", "z_joint"):
    r2 = ridge_r2(np.column_stack([P[k][drows], base]))
    f1 = mode_f1(np.column_stack([P[k][t_], act[t_]]))
    report[k] = {"tactile_r2": round(r2, 4),
                 "tactile_r2_incr": round(r2 - r2_base, 4),
                 "mode_macro_f1": round(f1, 4)}
    print(k, report[k], flush=True)

# ---- pair check: front-similar, mode-different test transitions ----
rng = np.random.RandomState(0)
tes = np.where(kte)[0]
Fn_ = P["F_pre"][t_[tes]]; Wn = P["W_cross"][t_[tes]]; Zn = P["z_joint"][t_[tes]]
Fn_ = Fn_/ (np.linalg.norm(Fn_,axis=1,keepdims=True)+1e-9)
sims = []
for _ in range(4000):
    i, j = rng.randint(0, len(tes), 2)
    if m_[tes[i]] == m_[tes[j]]: continue
    fs = float(Fn_[i] @ Fn_[j])
    if fs < 0.995: continue           # front 几乎相同
    wd = float(np.linalg.norm(Wn[i]-Wn[j]))
    zd = float(np.linalg.norm(Zn[i]-Zn[j]))
    sims.append((fs, wd, zd))
if sims:
    a = np.array(sims)
    # 随机同模式对照
    ctrl = []
    for _ in range(4000):
        i, j = rng.randint(0, len(tes), 2)
        if m_[tes[i]] != m_[tes[j]]: continue
        if float(Fn_[i] @ Fn_[j]) < 0.995: continue
        ctrl.append((float(np.linalg.norm(Wn[i]-Wn[j])), float(np.linalg.norm(Zn[i]-Zn[j]))))
    c = np.array(ctrl) if ctrl else np.zeros((1,2))
    report["pair_check"] = {
        "n_pairs_diff_mode_front_similar": len(a),
        "wrist_cross_dist_diffmode_vs_samemode": [round(float(a[:,1].mean()),3), round(float(c[:,0].mean()),3)],
        "joint_dist_diffmode_vs_samemode": [round(float(a[:,2].mean()),3), round(float(c[:,1].mean()),3)]}
    print("pair_check", report["pair_check"], flush=True)

json.dump(report, open("outputs/preflight/usb/exp_b_probe_layers.json", "w"), indent=2)
print("EXP B DONE", flush=True)
