#!/usr/bin/env python3
"""LpWM vs LpWM+SCALE at ONE training seed (default 1024, the seed with no collapsed run), with
significance over test episodes. Writes eval_results/rebuttal_mirror/LPWM_s<seed>.md.

Design: both models are evaluated on the same 600 test episodes (6 sets x 100), each at the same
five budget tiers with the same planner random seed, so every outcome is paired.
  * unit of analysis = test episode; score = % of the five tiers solved
  * difference = mean of (score with SCALE - score without); 95% t-interval over episodes
  * p-value = exact two-sided sign-flip permutation test on the per-episode differences
  * Holm correction over the 10 tests (5 tasks x 2 planners)
  * per tier: exact two-sided McNemar test on the discordant episodes
  * cross-check: paired t-test over the six evaluation sets (5 degrees of freedom)
Pure standard library (the head node has no numpy / scipy).
"""
import collections, csv, json, math, os, statistics as st, sys

SEED = sys.argv[1] if len(sys.argv) > 1 else "1024"
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
D = {"pusht": "final_eval", "reacher": "final_eval", "cube": "final_eval",
     "tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
ES = [str(e) for e in range(101, 107)]
TIERS = ["T1", "T2", "T3", "T4", "T5"]
T975_5DF = 2.5706


def load(task, lab, sol):
    """-> {(es, traj, start): {tier: 0/1}} with a pairing key per row"""
    out = collections.defaultdict(dict)
    keys = []
    for es in ES:
        for r in csv.DictReader(open(f"{R}/{D[task]}/final_{task}_{lab}r{SEED}_{sol}_s{es}.csv")):
            out[(es, r["traj_id"], r["start_idx"])][r["tier"]] = int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    return out, keys


def perm_p(ints):
    """exact two-sided sign-flip test; ints = per-episode differences as integers"""
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


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def t5_p(t):
    th = math.atan(abs(t) / math.sqrt(5))
    a = (2 / math.pi) * (th + math.sin(th) * (math.cos(th) + (2 / 3) * math.cos(th) ** 3))
    return max(0.0, 1 - a)


def fp(p):
    return "<0.0001" if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}")


main, tiers, diag = [], {}, {}
for task in D:
    for sol, sname in SOLS:
        b, kb = load(task, "lpb", sol)
        s, ks = load(task, "lps", sol)
        assert kb == ks and len(kb) == 3000, (task, sol)  # same episodes, tiers and planner seeds
        # unit = unique test episode (a start/goal pair drawn twice counts once)
        cl = collections.defaultdict(list)
        for k in b:
            assert sorted(b[k]) == TIERS and sorted(s[k]) == TIERS
            cl[k[1:]].append((sum(s[k].values()) - sum(b[k].values()), sum(b[k].values()), sum(s[k].values())))
        n = len(cl)
        # integer differences (x number of repeats) keep the permutation test exact
        ints = [round(st.mean(x[0] for x in v) * 2) for v in cl.values()]  # units of 0.5 tier
        d = [100 * st.mean(x[0] for x in v) / 5 for v in cl.values()]
        mb = st.mean(100 * sum(v.values()) / 5 for v in b.values())
        ms = st.mean(100 * sum(v.values()) / 5 for v in s.values())
        md, se = st.mean(d), st.stdev(d) / math.sqrt(n)
        tcrit = 1.9639  # t(0.975, 599)
        # six-set cross-check
        ds = [st.mean(100 * (sum(s[k].values()) - sum(b[k].values())) / 5 for k in b if k[0] == e) for e in ES]
        t6 = st.mean(ds) / (st.stdev(ds) / math.sqrt(6)) if st.stdev(ds) > 0 else float("inf")
        main.append(dict(task=task, sol=sname, n=n, lp=mb, sc=ms, d=ms - mb, md=md, lo=md - tcrit * se, hi=md + tcrit * se,
                         p=perm_p(ints), win=sum(x > 0 for x in d), lose=sum(x < 0 for x in d),
                         p6=t5_p(t6), lo6=st.mean(ds) - T975_5DF * st.stdev(ds) / math.sqrt(6),
                         hi6=st.mean(ds) + T975_5DF * st.stdev(ds) / math.sqrt(6)))
        for t in TIERS:
            bb = sum(1 for k in b if s[k][t] == 1 and b[k][t] == 0)
            cc = sum(1 for k in b if s[k][t] == 0 and b[k][t] == 1)
            tiers[(task, sname, t)] = (100 * st.mean(v[t] for v in b.values()), 100 * st.mean(v[t] for v in s.values()),
                                       bb, cc, mcnemar(bb, cc))
    for arm, tag in (("LpWM", "base"), ("LpWM+SCALE", "scale")):
        diag[(task, arm)] = json.load(open(f"{R}/diag/{task}__lewm_lpwmr_{task}_{tag}_s{SEED}.json"))
# Holm over the 10 main tests
order = sorted(range(len(main)), key=lambda i: main[i]["p"])
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(main) - rank) * main[i]["p"]))
    main[i]["holm"] = run


def verdict(m):
    if m["holm"] < 0.05:
        return "显著提升" if m["md"] > 0 else "显著下降"
    return "不显著"


out = [f"# LpWM 与 LpWM+SCALE：训练种子 {SEED}，含显著性检验", "",
       "两个模型在同样的 600 个测试回合（6 组 × 100）、同样的 5 个预算档、同样的规划器随机数下评测，逐回合配对。",
       "分析单位是测试回合，每回合得分 = 5 个预算档里成功的比例。差值 = LpWM+SCALE − LpWM（百分点）。",
       "95% 区间是对回合的 t 区间；p 值是对每回合差值做的精确符号翻转置换检验（双侧）；",
       "Holm 校正覆盖 10 个检验（5 个任务 × 2 种规划器）；显著 = 校正后 p < 0.05。", "",
       "## 总表", "",
       "| 任务 | 规划器 | LpWM | LpWM+SCALE | 差值 | 95% 区间 | p 值 | Holm 校正后 p | 结论 |",
       "|---|---|---|---|---|---|---|---|---|"]
for m in main:
    out.append(f"| {TITLE[m['task']]} | {m['sol']} | {m['lp']:.1f} | {m['sc']:.1f} | {m['d']:+.1f} | "
               f"[{m['lo']:+.1f}, {m['hi']:+.1f}] | {fp(m['p'])} | {fp(m['holm'])} | {verdict(m)} |")
out += ["", "## 配对细节与交叉检验", "",
        "“更好 / 更差的回合数”= 加 SCALE 后五档成功数更多 / 更少的测试回合数。",
        "交叉检验把 6 个评测组当作 6 个独立样本做配对 t 检验（自由度 5），更保守。", "",
        "| 任务 | 规划器 | 回合数 | 更好的回合数 | 更差的回合数 | 六组 t 检验的 95% 区间 | 六组 t 检验 p 值 |",
        "|---|---|---|---|---|---|---|"]
for m in main:
    out.append(f"| {TITLE[m['task']]} | {m['sol']} | {m['n']} | {m['win']} | {m['lose']} | "
               f"[{m['lo6']:+.1f}, {m['hi6']:+.1f}] | {fp(m['p6'])} |")
out += ["", "## 各预算档", "",
        "每格：LpWM → LpWM+SCALE（差值；p）。p 是该档 600 个配对回合上的精确 McNemar 检验（双侧，未做多重校正）。", ""]
for _, sname in SOLS:
    out += [f"### {sname}", "", "| 任务 | T1 | T2 | T3 | T4 | T5 |", "|---|---|---|---|---|---|"]
    for task in D:
        cells = []
        for t in TIERS:
            lb, ls, _, _, p = tiers[(task, sname, t)]
            cells.append(f"{lb:.1f} → {ls:.1f}（{ls - lb:+.1f}；p{'' if p < 1e-4 else '='}{fp(p)}）")
        out.append(f"| {TITLE[task]} | " + " | ".join(cells) + " |")
    out.append("")
out += ["## 表征诊断（留出 1500 个样本）", "",
        "| 任务 | 模型 | 距离对齐 Spearman | 距离对齐 Pearson | 探针 R² 平均 | 各状态分量探针 R² | 前 2 / 4 / 8 主成分方差占比 |",
        "|---|---|---|---|---|---|---|"]
for (task, arm), d in diag.items():
    out.append(f"| {TITLE[task]} | {arm} | {d['rho']['full_q']['spearman']:.2f} | {d['rho']['full_q']['pearson']:.2f} | "
               f"{d['full_r2_mean']:.3f} | " + "，".join(f"{q} {x:.2f}" for q, x in zip(d["qdims"], d["full_r2"])) + " | "
               + " / ".join(f"{d['spectrum_top'][k]:.2f}" for k in ("2", "4", "8")) + " |")
out.append("")
p = os.path.join(R, f"LPWM_s{SEED}.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
