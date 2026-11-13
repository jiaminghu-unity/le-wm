#!/usr/bin/env python3
"""Full detail for the rebuttal LpWM / LpWM+SCALE runs (training seeds 1024, 1025). Writes LPWM.md.

Per task: success by budget tier, by evaluation set, paired differences, and every diagnostic
stored by rebuttal_diag.py. Collapsed runs listed in EXCLUDE are left out of all tables.
"""
import collections, csv, json, os, statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
D = {"pusht": "final_eval", "reacher": "final_eval", "cube": "final_eval",
     "tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
RECIPE = {"pusht": "官方 Push-T 配方：正则权重 0.5，rms_norm 关，TemporalJaccard 关；SCALE 权重 0.1",
          "reacher": "无官方配方，沿用 Cube 配方：正则权重 10，rms_norm 开，TemporalJaccard 0.1；SCALE 权重 0.15",
          "cube": "官方 OGBench-Cube 配方：正则权重 10，rms_norm 开，TemporalJaccard 0.1；SCALE 权重 0.1",
          "tworoom": "无官方配方，沿用 Cube 配方：正则权重 10，rms_norm 开，TemporalJaccard 0.1；SCALE 权重 0.1",
          "pointmaze": "无官方配方，沿用 Cube 配方：正则权重 10，rms_norm 开，TemporalJaccard 0.1；SCALE 权重 0.1"}
ARMS = (("LpWM", "lpb", "base"), ("LpWM+SCALE", "lps", "scale"))
SEEDS = ("1024", "1025")
SOLS = ("cem", "icem")
ES = [str(e) for e in range(101, 107)]
TIERS = ["T1", "T2", "T3", "T4", "T5"]
EXCLUDE = {("pusht", "LpWM+SCALE", "1025")}  # collapsed (6.8 / 8.1)


def load(task, lab, seed, sol):
    """-> {es: {tier: success %}}"""
    out = {}
    for es in ES:
        rows = list(csv.DictReader(open(f"{R}/{D[task]}/final_{task}_{lab}r{seed}_{sol}_s{es}.csv")))
        t = collections.defaultdict(list)
        for r in rows:
            t[r["tier"]].append(int(r["success"]))
        assert sorted(t) == TIERS and all(len(v) == 100 for v in t.values()), (task, lab, seed, sol, es)
        out[es] = {k: 100 * sum(v) / len(v) for k, v in t.items()}
    return out


def pair(a, b, signed=False):
    f = "{:+.1f}" if signed else "{:.1f}"
    return f.format(a) + " / " + f.format(b)


out = ["# LpWM 与 LpWM+SCALE 全部明细（训练种子 1024、1025）", "",
       "已剔除：Push-T 的 LpWM+SCALE 种子 1025（训练崩溃，成功率 6.8 / 8.1）。", ""]
for task in D:
    ev = {(a, s, sol): load(task, lab, s, sol) for a, lab, _ in ARMS for s in SEEDS for sol in SOLS
          if (task, a, s) not in EXCLUDE}
    tier = {k: {t: st.mean(v[e][t] for e in ES) for t in TIERS} for k, v in ev.items()}
    allm = {k: st.mean(tier[k].values()) for k in tier}
    out += [f"## {TITLE[task]}", "", f"配方：{RECIPE[task]}。", "",
            "### 各预算档成功率（%，CEM / iCEM，6 组 × 100 回合平均）", "",
            "| 模型 | 种子 | T1 | T2 | T3 | T4 | T5 | 五档平均 |", "|---|---|---|---|---|---|---|---|"]
    for a, _, _ in ARMS:
        for s in SEEDS:
            if (task, a, s) in EXCLUDE:
                continue
            c, i = (a, s, "cem"), (a, s, "icem")
            out.append(f"| {a} | {s} | " + " | ".join(pair(tier[c][t], tier[i][t]) for t in TIERS)
                       + f" | {pair(allm[c], allm[i])} |")
    for s in SEEDS:
        if (task, "LpWM+SCALE", s) in EXCLUDE:
            continue
        d = {sol: {t: tier[("LpWM+SCALE", s, sol)][t] - tier[("LpWM", s, sol)][t] for t in TIERS} for sol in SOLS}
        out.append(f"| 差值（加 SCALE − 不加） | {s} | " + " | ".join(pair(d["cem"][t], d["icem"][t], True) for t in TIERS)
                   + f" | {pair(st.mean(d['cem'].values()), st.mean(d['icem'].values()), True)} |")
    out += ["", "### 各评测组成功率（%，CEM / iCEM，每组 100 回合 × 5 档平均）", "",
            "| 模型 | 种子 | " + " | ".join(ES) + " | 组间标准差 |", "|---|---|" + "---|" * (len(ES) + 1)]
    for a, _, _ in ARMS:
        for s in SEEDS:
            if (task, a, s) in EXCLUDE:
                continue
            pe = {sol: [st.mean(ev[(a, s, sol)][e].values()) for e in ES] for sol in SOLS}
            out.append(f"| {a} | {s} | " + " | ".join(pair(x, y) for x, y in zip(pe["cem"], pe["icem"]))
                       + f" | {pair(st.stdev(pe['cem']), st.stdev(pe['icem']))} |")
    dg = {}
    for a, _, tag in ARMS:
        for s in SEEDS:
            if (task, a, s) in EXCLUDE:
                continue
            dg[(a, s)] = json.load(open(f"{R}/diag/{task}__lewm_lpwmr_{task}_{tag}_s{s}.json"))
    any_d = next(iter(dg.values()))
    subs = [k for k in ("high_q", "low_q") if k in any_d["rho"]]
    head = ["模型", "种子", "距离对齐 Spearman", "距离对齐 Pearson"]
    head += [{"high_q": "对齐（高可恢复分量）", "low_q": "对齐（低可恢复分量）"}[k] for k in subs]
    head += ["探针 R² 平均", "前 2 / 4 / 8 主成分方差占比", "只用前 2 / 4 / 8 主成分的探针 R² 平均"]
    out += ["", f"### 表征诊断（留出数据 {any_d['n']} 个样本，隐表示 {any_d['rank']} 维）", "",
            "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (a, s), d in dg.items():
        row = [a, s, f"{d['rho']['full_q']['spearman']:.2f}", f"{d['rho']['full_q']['pearson']:.2f}"]
        row += [f"{d['rho'][k]['spearman']:.2f}" for k in subs]
        row += [f"{d['full_r2_mean']:.3f}",
                " / ".join(f"{d['spectrum_top'][k]:.2f}" for k in ("2", "4", "8")),
                " / ".join(f"{st.mean(d['topk_r2'][k]):.2f}" for k in ("2", "4", "8"))]
        out.append("| " + " | ".join(row) + " |")
    out += ["", "### 各状态分量的探针 R²", "",
            "| 模型 | 种子 | " + " | ".join(any_d["qdims"]) + " |", "|---|---|" + "---|" * len(any_d["qdims"])]
    for (a, s), d in dg.items():
        out.append(f"| {a} | {s} | " + " | ".join(f"{x:.2f}" for x in d["full_r2"]) + " |")
    out.append("")
p = os.path.join(R, "LPWM.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
