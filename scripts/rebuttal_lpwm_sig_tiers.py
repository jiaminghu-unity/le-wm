#!/usr/bin/env python3
"""LpWM vs LpWM+SCALE per planning-budget tier (T1-T5), pooled over the training seeds whose runs
did not collapse, with paired significance over test episodes. English table for the rebuttal.
Writes eval_results/rebuttal_mirror/LPWM_SIG_EN.md.

Seeds: a training seed is used for a task only if both arms trained normally at that seed, so
Push-T uses seed 1024 only (LpWM+SCALE seed 1025 collapsed); the other tasks use 1024 and 1025.
Per tier: unit = test episode (600 = 6 sets x 100, same episodes and same planner noise for both
models); per-episode difference = mean over the used training seeds of (success with SCALE -
success without); 95% t-interval over episodes; p = exact two-sided sign-flip permutation test.
Pure standard library.
"""
import collections, csv, math, os, statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
D = {"pusht": "final_eval", "reacher": "final_eval", "cube": "final_eval",
     "tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
ALL_SEEDS = ("1024", "1025")
EXCLUDE = {("pusht", "1025")}
ES = [str(e) for e in range(101, 107)]
TIERS = ["T1", "T2", "T3", "T4", "T5"]


def load(task, lab, seed, sol):
    out, keys = {}, []
    for es in ES:
        for r in csv.DictReader(open(f"{R}/{D[task]}/final_{task}_{lab}r{seed}_{sol}_s{es}.csv")):
            out[(es, r["traj_id"], r["start_idx"], r["tier"])] = int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(keys) == 3000 and len(out) == 3000, (task, lab, seed, sol)
    return out, keys


def perm_p(ints):
    obs = abs(sum(ints))
    dist = {0: 1.0}
    for a in ints:
        a = abs(a)
        if a == 0:
            continue
        nd = collections.defaultdict(float)
        for s, p in dist.items():
            nd[s + a] += p / 2
            nd[s - a] += p / 2
        dist = nd
    return min(1.0, sum(p for s, p in dist.items() if abs(s) >= obs - 1e-9))


def fp(p):
    return "<0.0001" if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}")


def sg(x):
    return f"{x:+.1f}".replace("-", "−")


rows = {}
for task in D:
    seeds = [s for s in ALL_SEEDS if (task, s) not in EXCLUDE]
    for sol, sname in SOLS:
        base, scale, ref = {}, {}, None
        for s in seeds:
            base[s], kb = load(task, "lpb", s, sol)
            scale[s], ks = load(task, "lps", s, sol)
            assert kb == ks and (ref is None or kb == ref), (task, sol, s)
            ref = kb
        for t in TIERS:
            eps = [k for k in base[seeds[0]] if k[3] == t]
            assert len(eps) == 600
            cl = collections.defaultdict(list)  # unique test episode -> differences, one per (occurrence, seed)
            for k in eps:
                for s in seeds:
                    cl[k[1:3]].append(scale[s][k] - base[s][k])
            n = len(cl)
            d = [100 * st.mean(v) for v in cl.values()]
            ints = [round(st.mean(v) * 2 * len(seeds)) for v in cl.values()]
            assert all(abs(i - st.mean(v) * 2 * len(seeds)) < 1e-9 for i, v in zip(ints, cl.values()))
            mb = 100 * st.mean(base[s][k] for s in seeds for k in eps)
            ms = 100 * st.mean(scale[s][k] for s in seeds for k in eps)
            md, se = st.mean(d), st.stdev(d) / math.sqrt(n)
            rows[(task, sname, t)] = (mb, ms, ms - mb, md - 1.9639 * se, md + 1.9639 * se, perm_p(ints))

out = ["**LpWM vs. LpWM+SCALE: planning success rate (%) at each planning-budget tier.**",
       "Both models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, "
       "so outcomes are paired per episode. Budgets (candidates × iterations): T1 300×30, T2 150×15, T3 50×10, "
       "T4 20×5, T5 10×3. Δ = LpWM+SCALE − LpWM in percentage points; the 95% CI is a paired t-interval over "
       "episodes; p is from an exact two-sided paired sign-flip permutation test (unadjusted). Reacher, Cube, Two-Room and "
       "PointMaze average two training seeds (1024, 1025); Push-T uses seed 1024 only, because the LpWM+SCALE "
       "run at seed 1025 collapsed during training and that seed is excluded for both models.", ""]
for _, sname in SOLS:
    out += [f"**{sname} planner**", "", "| Task | Tier | LpWM | LpWM+SCALE | Δ | 95% CI | p-value |",
            "|---|---|---|---|---|---|---|"]
    for task in D:
        for t in TIERS:
            mb, ms, d, lo, hi, p = rows[(task, sname, t)]
            out.append(f"| {TITLE[task]} | {t} | {mb:.1f} | {ms:.1f} | {sg(d)} | [{sg(lo)}, {sg(hi)}] | {fp(p)} |")
    out.append("")
p = os.path.join(R, "LPWM_SIG_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
sig = [(k, v) for k, v in rows.items() if v[5] < 0.05]
print("p<0.05:", len(sig), "of", len(rows), "| positive:", sum(1 for k, v in sig if v[2] > 0), "| negative:",
      [(k, round(v[2], 1), fp(v[5])) for k, v in sig if v[2] < 0])
