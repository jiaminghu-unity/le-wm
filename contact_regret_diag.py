"""Contact-conditioned Planning Regret (approved single diagnostic; no training).

Reuses T6 raw rollouts. Per initial state s with ~100 executed 6-step sequences:
  r(s) = J_real(A_picked_by_cost) - min_A J_real(A)        (best-step outcome)
Groups by initial contact state / nearby contact events, controls for phase,
and answers:
  Q1  is B1's selection regret concentrated in contact-related states?
  Q2  in high-regret states, do wrist features hold info the joint latent lost?
      (out-of-fold probe-error gap, correlated with regret)
  Q3  does the oracle cost (real next-frame encoding) pick better there?
      (separates prediction error from latent cost geometry)

Run locally:  python3 contact_regret_diag.py --variant b1
"""
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import mannwhitneyu, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="b1", choices=["b1", "dino"])
ap.add_argument("--raw-dir", default="/workspace/lewm-tactile/results/rank_diag")
ap.add_argument("--lewm-npz", default="outputs/n0_lewm_states.npz")
args = ap.parse_args()

D = np.load(f"{args.raw_dir}/rank_diag_T6raw_{args.variant}_raw.npz")
best, oc = D["outcome_best"], D["outcome_final"]
co, cp, t_abs = D["c_oracle"], D["c_pred"], D["t_abs"].astype(int)
mw = D["mode_window"]
S, K = best.shape

import zarr
zr = zarr.open("data/demo_data/insertion_usb", mode="r")
ends = np.asarray(zr["meta/episode_ends"][:]).astype(int)
starts = np.concatenate([[0], ends[:-1]])
N = int(ends[-1])
ep_of = np.zeros(N, int)
for e in range(len(ends)):
    ep_of[starts[e]:ends[e]] = e
M = np.load("cluster_bundle/outputs/preflight/usb/provisional_modes.npz")
mode_row = np.full(N, -2, np.int64)
mode_row[M["t"]] = M["mode"]

eps = ep_of[t_abs]
phase = (t_abs - starts[eps]) / np.maximum(ends[eps] - starts[eps] - 1, 1)
cross_boundary = (t_abs - 2) < starts[eps]          # 3-frame context leaks previous episode
mode_t = mode_row[t_abs]
contact = (mode_t == 1) | (mode_t == 2)
free = mode_t == 0
lab_rows = [[m for m in row if m >= 0] for row in mw]
near_tr = np.array([len(set(r)) > 1 for r in lab_rows])

r_pred = np.array([best[i, cp[i].argmin()] - best[i].min() for i in range(S)])
r_orac = np.array([best[i, co[i].argmin()] - best[i].min() for i in range(S)])
spread = best.max(1) - best.min(1)
r_pred_n, r_orac_n = r_pred / (spread + 1e-9), r_orac / (spread + 1e-9)
succ_possible = best.min(1) <= 1.0
miss_pred = succ_possible & (np.array([best[i, cp[i].argmin()] for i in range(S)]) > 1.0)
miss_orac = succ_possible & (np.array([best[i, co[i].argmin()] for i in range(S)]) > 1.0)

print(f"=== {args.variant}: S={S}, labeled {int((mode_t>=0).sum())}, "
      f"context-crosses-episode-boundary {int(cross_boundary.sum())} ===")

def grp(name, sel):
    sel = sel & (mode_t >= -2)
    if sel.sum() == 0: return
    print(f"  {name:16s} n={sel.sum():3d} | regret pred {r_pred[sel].mean():.3f} "
          f"(norm {r_pred_n[sel].mean():.3f}) oracle {r_orac[sel].mean():.3f} "
          f"(norm {r_orac_n[sel].mean():.3f}) | succ-poss {int((succ_possible&sel).sum()):2d} "
          f"miss pred {int((miss_pred&sel).sum()):2d} oracle {int((miss_orac&sel).sum()):2d} "
          f"| phase {phase[sel].mean():.2f}")

print("-- Q1: regret by contact grouping --")
grp("all", np.ones(S, bool))
grp("free (m=0)", free)
grp("contact (m=1,2)", contact)
grp("near-transition", near_tr)
grp("no-transition", ~near_tr)

lab_ok = mode_t >= 0
u, p_mw = mannwhitneyu(r_pred_n[contact], r_pred_n[free])
print(f"  Mann-Whitney contact vs free (norm regret): p={p_mw:.3f}")
# phase-controlled OLS: regret_n ~ 1 + phase + contact   (labeled states)
X = np.stack([np.ones(lab_ok.sum()), phase[lab_ok], contact[lab_ok].astype(float)], 1)
y = r_pred_n[lab_ok]
beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
rng = np.random.RandomState(0)
perm = [np.linalg.lstsq(np.stack([X[:, 0], X[:, 1], rng.permutation(X[:, 2])], 1),
                        y, rcond=None)[0][2] for _ in range(2000)]
p_perm = float((np.abs(perm) >= abs(beta[2])).mean())
print(f"  OLS norm-regret ~ phase + contact: contact coef {beta[2]:+.3f} "
      f"(se {se[2]:.3f}, permutation p={p_perm:.3f}); phase coef {beta[1]:+.3f}")

# -- Q2: out-of-fold info-gap (wrist vs joint latent) vs regret --
L = np.load(args.lewm_npz)
pass  # pos built after valid_next filter below
Wf = np.load("artifacts/usb/wrist_features.npy")
A = np.load("artifacts/usb/action.npy")
ee = np.asarray(zr["data/ee_pos"][:], np.float32)
plug = np.asarray(zr["data/plug_pos"][:], np.float32)

rows_all = L["rows"].astype(int)
valid_next = rows_all + 1 < ends[ep_of[rows_all]]
rows_all = rows_all[valid_next]
Lz = L["z_joint"][valid_next]
lab_all = mode_row[rows_all]
grp_ep = ep_of[rows_all]
y_mot = np.concatenate([ee[rows_all + 1] - ee[rows_all], plug[rows_all + 1] - plug[rows_all]], 1)

def oof_gap():
    Xw = np.concatenate([Wf[rows_all], A[rows_all]], 1)
    Xz = np.concatenate([Lz, A[rows_all]], 1)
    gap_mode = np.full(len(rows_all), np.nan)
    gap_mot = np.full(len(rows_all), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(Xw, groups=grp_ep):
        trl = tr[(lab_all[tr] >= 0)]
        mu, sd = y_mot[tr].mean(0), y_mot[tr].std(0) + 1e-9
        for X, out in ((Xw, "w"), (Xz, "z")):
            sc = StandardScaler().fit(X[tr])
            cl = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(X[trl]), lab_all[trl])
            pr = cl.predict_proba(sc.transform(X[te]))
            cls = list(cl.classes_)
            pc = np.array([pr[i, cls.index(l)] if l in cls else np.nan
                           for i, l in enumerate(lab_all[te])])
            rg = Ridge(alpha=100.0).fit(sc.transform(X[tr]), (y_mot[tr] - mu) / sd)
            me = (((y_mot[te] - mu) / sd - rg.predict(sc.transform(X[te]))) ** 2).mean(1)
            if out == "w": pw, mw_ = pc, me
            else: pz, mz_ = pc, me
        gap_mode[te] = pw - pz          # >0: wrist knows the mode better than latent
        gap_mot[te] = mz_ - mw_         # >0: latent loses motion-relevant info
    return gap_mode, gap_mot

gm, gt = oof_gap()
pos = {int(r): i for i, r in enumerate(rows_all)}
idx = np.array([pos.get(int(t), -1) for t in t_abs])
gm_s = np.where(idx >= 0, gm[np.maximum(idx, 0)], np.nan)
gt_s = np.where(idx >= 0, gt[np.maximum(idx, 0)], np.nan)

print("-- Q2: info-gap (wrist minus joint-latent, out-of-fold) vs regret --")
for nm, g in (("mode-prob gap", gm_s), ("motion-err gap", gt_s)):
    ok = ~np.isnan(g)
    rho, p = spearmanr(g[ok], r_pred_n[ok])
    rho_c, p_c = (spearmanr(g[ok & contact], r_pred_n[ok & contact])
                  if (ok & contact).sum() > 5 else (np.nan, np.nan))
    hi = r_pred_n > np.median(r_pred_n)
    print(f"  {nm:15s} n={ok.sum():3d} | corr with norm-regret rho={rho:+.3f} p={p:.3f} "
          f"| contact-only rho={rho_c:+.3f} p={p_c:.3f} "
          f"| gap hi-regret {np.nanmean(g[ok & hi]):+.4f} vs lo {np.nanmean(g[ok & ~hi]):+.4f}")

print("-- Q3: oracle vs pred regret in trouble spots --")
for nm, sel3 in (("contact", contact), ("near-transition", near_tr),
                 ("hi-regret half", r_pred_n > np.median(r_pred_n))):
    d = r_pred_n[sel3] - r_orac_n[sel3]
    print(f"  {nm:16s} n={sel3.sum():3d} | norm-regret pred {r_pred_n[sel3].mean():.3f} "
          f"oracle {r_orac_n[sel3].mean():.3f} (pred-oracle {d.mean():+.3f}) "
          f"| oracle miss {int((miss_orac&sel3).sum())}/{int((succ_possible&sel3).sum())}")
