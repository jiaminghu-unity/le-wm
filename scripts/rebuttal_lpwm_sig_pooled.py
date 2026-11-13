#!/usr/bin/env python3
"""LpWM vs LpWM+SCALE pooled over the training seeds whose runs did not collapse, with
significance over test episodes. Writes eval_results/rebuttal_mirror/LPWM_SIG.md.

A training seed is used for a task only if BOTH arms trained normally at that seed (same-seed
pairs), so Push-T uses seed 1024 only (LpWM+SCALE seed 1025 collapsed) and the other four tasks
use seeds 1024 and 1025.

Design: all models are evaluated on the same 600 test episodes (6 sets x 100), each at the same
five budget tiers with the same planner random seed, so every outcome is paired.
  * unit of analysis = test episode; per-episode difference = mean over the used training seeds
    and the five tiers of (success with SCALE - success without), in percentage points
  * 95% t-interval over episodes; p = exact two-sided sign-flip permutation test
  * Holm correction over the 10 tests (5 tasks x 2 planners)
  * cross-check: paired t-test over the six evaluation sets (5 degrees of freedom)
Pure standard library (the head node has no numpy / scipy).
"""
import collections, csv, math, os, statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
D = {"pusht": "final_eval", "reacher": "final_eval", "cube": "final_eval",
     "tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
ALL_SEEDS = ("1024", "1025")
EXCLUDE = {("pusht", "1025")}  # LpWM+SCALE collapsed at this seed (6.8 / 8.1)
ES = [str(e) for e in range(101, 107)]
TIERS = ["T1", "T2", "T3", "T4", "T5"]
T975_5DF = 2.5706


def load(task, lab, seed, sol):
    """-> ({(es, traj, start): tiers solved (0..5)}, pairing keys)"""
    out, keys = collections.defaultdict(int), []
    cnt = collections.Counter()
    for es in ES:
        for r in csv.DictReader(open(f"{R}/{D[task]}/final_{task}_{lab}r{seed}_{sol}_s{es}.csv")):
            k = (es, r["traj_id"], r["start_idx"])
            out[k] += int(r["success"])
            cnt[k] += 1
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(keys) == 3000 and set(cnt.values()) == {5}, (task, lab, seed, sol)
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


def t5_p(t):
    th = math.atan(abs(t) / math.sqrt(5))
    return max(0.0, 1 - (2 / math.pi) * (th + math.sin(th) * (math.cos(th) + (2 / 3) * math.cos(th) ** 3)))


def fp(p):
    return "<0.0001" if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}")


main = []
for task in D:
    seeds = [s for s in ALL_SEEDS if (task, s) not in EXCLUDE]
    for sol, sname in SOLS:
        base, scale, ref = {}, {}, None
        for s in seeds:
            base[s], kb = load(task, "lpb", s, sol)
            scale[s], ks = load(task, "lps", s, sol)
            assert kb == ks and (ref is None or kb == ref), (task, sol, s)  # same episodes / tiers / planner seeds
            ref = kb
        eps = list(base[seeds[0]])
        cl = collections.defaultdict(list)  # unique test episode -> tier-count differences, one per (occurrence, seed)
        for k in eps:
            for s in seeds:
                cl[k[1:]].append(scale[s][k] - base[s][k])
        n = len(cl)
        d = [100 * st.mean(v) / 5 for v in cl.values()]
        ints = [round(st.mean(v) * 2 * len(seeds)) for v in cl.values()]
        assert all(abs(i - st.mean(v) * 2 * len(seeds)) < 1e-9 for i, v in zip(ints, cl.values()))
        mb = st.mean(100 * base[s][k] / 5 for s in seeds for k in eps)
        ms = st.mean(100 * scale[s][k] / 5 for s in seeds for k in eps)
        md, se = st.mean(d), st.stdev(d) / math.sqrt(n)
        ds = [st.mean(100 * (scale[s][k] - base[s][k]) / 5 for s in seeds for k in eps if k[0] == e) for e in ES]
        t6 = st.mean(ds) / (st.stdev(ds) / math.sqrt(6))
        per_seed = [st.mean(100 * (scale[s][k] - base[s][k]) / 5 for k in eps) for s in seeds]
        main.append(dict(task=task, sol=sname, seeds=seeds, n=n, lp=mb, sc=ms, d=ms - mb, md=md,
                         lo=md - 1.9639 * se, hi=md + 1.9639 * se, p=perm_p(ints), p6=t5_p(t6),
                         lo6=st.mean(ds) - T975_5DF * st.stdev(ds) / math.sqrt(6),
                         hi6=st.mean(ds) + T975_5DF * st.stdev(ds) / math.sqrt(6), per_seed=per_seed))
order = sorted(range(len(main)), key=lambda i: main[i]["p"])
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(main) - rank) * main[i]["p"]))
    main[i]["holm"] = run


def verdict(m):
    if m["holm"] < 0.05:
        return "显著提升" if m["md"] > 0 else "显著下降"
    return "不显著"


out = ["# LpWM 与 LpWM+SCALE：剔除崩溃种子后的合并结果与显著性", "",
       "使用的训练种子：Push-T 只用 1024（LpWM+SCALE 的种子 1025 训练崩溃，同种子的 LpWM 也不计入）；",
       "Reacher、Cube、Two-Room、PointMaze 用 1024 和 1025 两个种子的平均。",
       "所有模型在同样的 600 个测试回合、同样的 5 个预算档、同样的规划器随机数下评测，逐回合配对。",
       "每回合差值 = 对所用训练种子和 5 个预算档平均的（LpWM+SCALE − LpWM），单位百分点。",
       "95% 区间是对回合的 t 区间；p 值是精确的符号翻转置换检验（双侧）；Holm 校正覆盖 10 个检验；",
       "显著 = 校正后 p < 0.05。", "",
       "| 任务 | 规划器 | LpWM | LpWM+SCALE | 差值 | 95% 区间 | p 值 | Holm 校正后 p | 结论 |",
       "|---|---|---|---|---|---|---|---|---|"]
for m in main:
    out.append(f"| {TITLE[m['task']]} | {m['sol']} | {m['lp']:.1f} | {m['sc']:.1f} | {m['d']:+.1f} | "
               f"[{m['lo']:+.1f}, {m['hi']:+.1f}] | {fp(m['p'])} | {fp(m['holm'])} | {verdict(m)} |")
out += ["", "## 各训练种子单独的差值与交叉检验", "",
        "交叉检验把 6 个评测组当作 6 个样本做配对 t 检验（自由度 5），更保守。", "",
        "| 任务 | 规划器 | 所用种子 | 各种子的差值 | 六组 t 检验的 95% 区间 | 六组 t 检验 p 值 |", "|---|---|---|---|---|---|"]
for m in main:
    out.append(f"| {TITLE[m['task']]} | {m['sol']} | {'、'.join(m['seeds'])} | "
               + " / ".join(f"{x:+.1f}" for x in m["per_seed"])
               + f" | [{m['lo6']:+.1f}, {m['hi6']:+.1f}] | {fp(m['p6'])} |")
out.append("")
p = os.path.join(R, "LPWM_SIG.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
