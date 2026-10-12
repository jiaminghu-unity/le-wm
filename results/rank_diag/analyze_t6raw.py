"""D1 analysis over T6 raw dumps: Top-K overlap, success-flag regret,
contact-mode grouping, per-step divergence. Pure numpy, no GPU."""
import json
import numpy as np

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ca, cb = ra - ra.mean(), rb - rb.mean()
    return float((ca * cb).sum() / (np.sqrt((ca ** 2).sum() * (cb ** 2).sum()) + 1e-12))

for V in ("b1", "dino"):
    D = np.load(f"rank_diag_T6raw_{V}_raw.npz")
    oc, best = D["outcome_final"], D["outcome_best"]          # (S,K)
    co, cp = D["c_oracle"], D["c_pred"]
    steps = D["oc_steps"]                                      # (S,T,K)
    mw = D["mode_window"]                                      # (S,8) transitions t-2..t+5
    S, K = oc.shape
    mode_t = mw[:, 2]                                          # label of transition t->t+1
    lab_rows = [[m for m in row if m >= 0] for row in mw]
    near_tr = np.array([len(set(r)) > 1 for r in lab_rows])

    def topk_overlap(cost, out, k):
        o = 0
        for i in range(S):
            o += len(set(np.argsort(cost[i])[:k]) & set(np.argsort(out[i])[:k])) / k
        return o / S

    def stats(sel, tag):
        if sel.sum() == 0:
            return
        sp_p = np.mean([spearman(oc[i], cp[i]) for i in np.where(sel)[0]])
        sp_o = np.mean([spearman(oc[i], co[i]) for i in np.where(sel)[0]])
        rg_p = np.mean([best[i, np.argmin(cp[i])] - best[i].min() for i in np.where(sel)[0]])
        rg_o = np.mean([best[i, np.argmin(co[i])] - best[i].min() for i in np.where(sel)[0]])
        # official-success translation: among states where SOME candidate reaches
        # outcome<=1 (success geometrically possible), does the cost's pick reach it?
        poss = np.array([best[i].min() <= 1.0 for i in range(S)]) & sel
        hit_p = np.mean([best[i, np.argmin(cp[i])] <= 1.0 for i in np.where(poss)[0]]) if poss.any() else float("nan")
        hit_o = np.mean([best[i, np.argmin(co[i])] <= 1.0 for i in np.where(poss)[0]]) if poss.any() else float("nan")
        print(f"  {tag:16s} n={sel.sum():3d} | sp pred {sp_p:+.3f} oracle {sp_o:+.3f} | "
              f"regret pred {rg_p:.3f} oracle {rg_o:.3f} | succ-possible {int(poss.sum())}: "
              f"picked pred {hit_p:.2f} oracle {hit_o:.2f}")

    print(f"=== {V} (S={S}, K={K}) ===")
    for k in (8, 10, 20):
        print(f"  top{k} overlap with TRUE outcome: pred {topk_overlap(cp, best, k):.3f} "
              f"oracle {topk_overlap(co, best, k):.3f} | pred-vs-oracle {topk_overlap(cp, co, k):.3f}")
    stats(np.ones(S, bool), "all")
    stats(mode_t == 0, "mode0 free")
    stats((mode_t == 1) | (mode_t == 2), "mode1/2 contact")
    stats(near_tr, "near-transition")
    stats(~near_tr, "no-transition")
    # per executed step: where does pred ranking lose the real outcome?
    row = []
    for s in range(steps.shape[1]):
        row.append(np.mean([spearman(steps[i, s], cp[i]) for i in range(S)]))
    print("  sp(pred cost, outcome@step s):", " ".join(f"{x:+.2f}" for x in row))
