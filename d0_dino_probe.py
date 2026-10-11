"""D0 — Frozen DINO feature evaluation on USB (no training, L4-sized).

1. Load dinov3_vits16 (official); documented fallback: dinov2_vits14 via torch.hub.
2. Extract frozen patch tokens for every frame, both views; keep mean-pooled (D,)
   and 2x2 grid-pooled (4D,) summaries per frame.
3. Established probes on the canonical episode split:
   - tactile descriptor ridge (alpha=1000, target r_tau, incremental over action+phase)
   - contact-mode logistic probes (same protocol as P3/M5)
   - geometry recoverability: ridge R^2 for plug_pos / socket_pos_gt / ee_pos /
     plug-socket relative, vs action+phase control
   for DINO-front / DINO-wrist / both; episode-bootstrap CIs on key numbers.
Run from ~/ContactWorld on cw-l4-01.
"""
import json
import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score
import zarr

dev = "cuda"
OUT = "outputs/preflight/usb"
torch.manual_seed(0)

# ---- backbone ----
backbone_name = None
model = None
for name, hub in (("dinov3_vits16", "facebookresearch/dinov3"),
                  ("dinov2_vits14", "facebookresearch/dinov2")):
    try:
        model = torch.hub.load(hub, name)
        backbone_name = name
        break
    except Exception as e:
        print(f"[backbone] {name} unavailable: {type(e).__name__}: {str(e)[:120]}", flush=True)
assert model is not None, "no DINO backbone available"
model.eval().to(dev)
for p in model.parameters():
    p.requires_grad_(False)
print("backbone:", backbone_name, flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def tokens(x):
    """x: (B,3,H,W) in [0,1] -> patch tokens (B,P,D) with official preprocessing."""
    x = (x - MEAN) / STD
    out = model.forward_features(x)
    if isinstance(out, dict):
        return out["x_norm_patchtokens"]
    return out  # some hub versions return tokens directly

root = zarr.open("data/demo_data/insertion_usb", mode="r")
ends = np.asarray(root["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
N = int(ends[-1])
act = np.asarray(root["data/action"][:])

feats = {}
with torch.no_grad():
    probe_dim = None
    for view in ("front", "wrist"):
        arr = root[f"data/{view}"]
        m_list, g_list = [], []
        for i in range(0, N, 32):
            x = torch.from_numpy(np.asarray(arr[i:i+32])).permute(0, 3, 1, 2).float()
            x = Fn.interpolate(x, size=(224, 224), mode="bilinear", antialias=True).to(dev)
            t = tokens(x)                             # (B,P,D)
            B, P, D = t.shape
            probe_dim = D
            g = int(P ** 0.5)
            m_list.append(t.mean(1).cpu().numpy())
            tg = t.reshape(B, g, g, D).permute(0, 3, 1, 2)
            tg = Fn.adaptive_avg_pool2d(tg, 2).reshape(B, D * 4)
            g_list.append(tg.cpu().numpy())
        feats[f"{view}_mean"] = np.concatenate(m_list)
        feats[f"{view}_grid"] = np.concatenate(g_list)
        print(view, "done", feats[f"{view}_mean"].shape, feats[f"{view}_grid"].shape, flush=True)
np.savez_compressed(f"{OUT}/d0_dino_features.npz",
                    **{k: v for k, v in feats.items()}, backbone=backbone_name)

splits = json.load(open(f"{OUT}/splits.json"))
D_ = np.load("artifacts_usb_tactile_descriptors.npz")
desc, drows = D_["r_tau"], D_["rows"].astype(int)
ep_of_row = np.searchsorted(ends, np.arange(N), side="right")
prog_row = (np.arange(N) - starts[ep_of_row]) / np.maximum(ends[ep_of_row] - starts[ep_of_row] - 1, 1)
tr = np.isin(ep_of_row[drows], splits["train"]); te = np.isin(ep_of_row[drows], splits["test"])
base = np.column_stack([act[drows], prog_row[drows]])

def ridge_r2(X, Y, tr_m, te_m):
    sc = StandardScaler().fit(X[tr_m]); Xs = sc.transform(X)
    r = Ridge(alpha=1000.0).fit(Xs[tr_m], Y[tr_m])
    p = r.predict(Xs[te_m]); y = Y[te_m]
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

M = np.load(f"{OUT}/provisional_modes.npz")
mode, t_of, ep_of = M["mode"], M["t"], M["ep"]
keep = mode >= 0
m_, t_, e_ = mode[keep], t_of[keep], ep_of[keep]
ktr = np.isin(e_, splits["train"]); kva = np.isin(e_, splits["val"]); kte = np.isin(e_, splits["test"])

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
    return (round(float(f1_score(m_[kte], pred, average="macro")), 4),
            round(float(average_precision_score((m_[kte] == 2).astype(int),
                                                proba[:, cls.index(2)])), 4))

# geometry targets (diagnostic recoverability from single frames)
plug = np.asarray(root["data/plug_pos"][:]); sock = np.asarray(root["data/socket_pos_gt"][:])
ee = np.asarray(root["data/ee_pos"][:]); rel = plug - sock
geo_rows = np.arange(N)
gtr = np.isin(ep_of_row, splits["train"]); gte = np.isin(ep_of_row, splits["test"])
geo_base = np.column_stack([act, prog_row])

report = {"backbone": backbone_name, "patch_dim": probe_dim,
          "tactile_nuisance_r2": round(ridge_r2(base, desc, tr, te), 4)}
report["mode_nuisance"] = mode_probe(np.column_stack([act[t_], prog_row[t_]]))
for tag, X_rows in [("front_mean", feats["front_mean"]), ("wrist_mean", feats["wrist_mean"]),
                    ("front_grid", feats["front_grid"]), ("wrist_grid", feats["wrist_grid"]),
                    ("both_mean", np.concatenate([feats["front_mean"], feats["wrist_mean"]], 1)),
                    ("both_grid", np.concatenate([feats["front_grid"], feats["wrist_grid"]], 1))]:
    r2 = ridge_r2(np.column_stack([X_rows[drows], base]), desc, tr, te)
    f1, auprc = mode_probe(np.column_stack([X_rows[t_], act[t_]]))
    geo = {}
    for gname, Y in (("plug", plug), ("socket", sock), ("ee", ee), ("rel", rel)):
        geo[gname] = round(ridge_r2(X_rows, Y, gtr, gte), 4)
    geo["base_rel"] = round(ridge_r2(geo_base, rel, gtr, gte), 4)
    report[tag] = {"dims": int(X_rows.shape[1]),
                   "tactile_r2": round(r2, 4),
                   "tactile_r2_incr": round(r2 - report["tactile_nuisance_r2"], 4),
                   "mode_macro_f1": f1, "mode2_auprc": auprc, "geometry_r2": geo}
    print(tag, report[tag], flush=True)

json.dump(report, open(f"{OUT}/d0_dino_probe.json", "w"), indent=2)
print("D0 DONE", flush=True)
