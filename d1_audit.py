"""D1 audit — tactile/mode information at four stages of the frozen-DINO fusion:
  S1 raw DINO wrist tokens (mean)   [from cache]
  S2 adapted wrist (after linear adapter + view emb, mean)
  S3 wrist after cross-attention (mean)
  S4a pooled joint (before projector) / S4b final joint latent (after projector)
Same ridge/logistic protocols as exp_b. Run on cw-l4-01: ~/d0venv/bin/python d1_audit.py
"""
import json
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import zarr, sys
sys.path.insert(0, ".")
from touchwm.dino_mv import DinoMultiViewJEPA
from einops import rearrange

dev = "cuda"; OUT = "outputs/d1"
root = zarr.open("data/demo_data/insertion_usb", mode="r")
ends = np.asarray(root["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
N = int(ends[-1]); act = np.asarray(root["data/action"][:])
tok = {v: np.load(f"{OUT}/{v}_tokens.fp16.npy", mmap_mode="r") for v in ("front", "wrist")}

model = DinoMultiViewJEPA().to(dev)
st = torch.load(f"{OUT}/pilot_best.pt", map_location=dev)
model.load_state_dict(st["model"]); model.eval()
enc = model.encoder
print("pilot ckpt step", st["step"], flush=True)

S = {k: np.zeros((N, 192), np.float32) for k in ("S2_adapted_w", "S3_cross_w", "S4a_pool", "S4b_joint")}
S1 = np.zeros((N, 384), np.float32)
with torch.no_grad():
    for i in range(0, N, 128):
        tf = torch.from_numpy(np.asarray(tok["front"][i:i+128])).to(dev)
        tw = torch.from_numpy(np.asarray(tok["wrist"][i:i+128])).to(dev)
        S1[i:i+128] = tw.float().mean(1).cpu().numpy()
        Ft = enc.adapt_f(tf.float()) + enc.view_emb[0][None]
        Wt = enc.adapt_w(tw.float()) + enc.view_emb[1][None]
        S["S2_adapted_w"][i:i+128] = Wt.mean(1).cpu().numpy()
        Fp = enc.ca_f(Ft, Wt); Wp = enc.ca_w(Wt, Ft)
        S["S3_cross_w"][i:i+128] = Wp.mean(1).cpu().numpy()
        U = enc.pool(torch.cat([Fp, Wp], dim=1))
        zp = enc.project(U.flatten(1))
        S["S4a_pool"][i:i+128] = zp.cpu().numpy()
        S["S4b_joint"][i:i+128] = model.projector(zp).cpu().numpy()
print("encoded stages", flush=True)

splits = json.load(open("outputs/preflight/usb/splits.json"))
D_ = np.load("artifacts_usb_tactile_descriptors.npz")
desc, drows = D_["r_tau"], D_["rows"].astype(int)
ep_of_row = np.searchsorted(ends, np.arange(N), side="right")
prog_row = (np.arange(N) - starts[ep_of_row]) / np.maximum(ends[ep_of_row]-starts[ep_of_row]-1, 1)
tr = np.isin(ep_of_row[drows], splits["train"]); te = np.isin(ep_of_row[drows], splits["test"])
base = np.column_stack([act[drows], prog_row[drows]])

def ridge_r2(X):
    sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X)
    r = Ridge(alpha=1000.0).fit(Xs[tr], desc[tr])
    p = r.predict(Xs[te]); y = desc[te]
    return float(1 - ((y-p)**2).sum() / ((y-y.mean(0))**2).sum())
r2b = ridge_r2(base)

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

report = {"pilot_step": int(st["step"]), "nuisance_r2": round(r2b, 4)}
stages = [("S1_dino_wrist", S1)] + list(S.items())
for k, X in stages:
    r2 = ridge_r2(np.column_stack([X[drows], base]))
    f1 = mode_f1(np.column_stack([X[t_], act[t_]]))
    report[k] = {"tactile_r2_incr": round(r2 - r2b, 4), "mode_macro_f1": round(f1, 4)}
    print(k, report[k], flush=True)
json.dump(report, open(f"{OUT}/d1_audit.json", "w"), indent=2)
print("D1 AUDIT DONE", flush=True)
