#!/usr/bin/env python3
"""English rebuttal tables: SCALE vs constrained-regression baselines (reviewer 9CWa W1/Q1).
Baselines = the paper's Aux arm (auxiliary state-regression head) without and with an L2 penalty
on the readout weights, aux term = lambda_aux * (MSE + beta * ||W||^2), beta in {0.01, 0.1, 1.0}.
Five tasks, training seeds 1024 and 1025 (averaged). Writes eval_results/rebuttal_mirror/CREG_TABLES_EN.md.

Table E (wide, one per planner): success by budget tier T1-T5 and their mean.
Table F: per task / planner / baseline, tier-averaged success of the baseline and of SCALE with
difference (SCALE - baseline), 95% CI and p-value. Unit = test episode (600 = 6 sets x 100, same
episodes and planner noise for every model); per-episode difference = mean over the two seeds and
five tiers; 95% t-interval over episodes; exact two-sided sign-flip test. Pure standard library.
"""
import collections, csv, math, os, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
D = {"pusht": "final_eval", "reacher": "final_eval", "cube": "final_eval",
     "tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
#            LeWM   SCALE  Aux
LAB = {"pusht": ("c1", "c3", "c5"), "reacher": ("r1", "r2", "r5"), "cube": ("k1", "k2", "k4"),
       "tworoom": ("t1", "t2", "t5"), "pointmaze": ("p1", "p2", "p5")}
BETAS = ("0.01", "0.1", "1.0")
BASELINES = ["Aux"] + [f"Aux + L2 (β = {b})" for b in BETAS]
METHODS = ["LeWM"] + BASELINES + ["SCALE"]
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
SEEDS = ("1024", "1025")
ES = [str(e) for e in range(101, 107)]
TIERS = ["T1", "T2", "T3", "T4", "T5"]


def labels(task):
    lw, sc, ax = LAB[task]
    out = {"LeWM": f"{lw}r", "SCALE": f"{sc}r", "Aux": f"{ax}r"}
    for b in BETAS:
        out[f"Aux + L2 (β = {b})"] = f"{ax}h{b}r"
    return out


def load(task, lab, seed, sol):
    out, keys = {}, []
    for es in ES:
        rows = list(csv.DictReader(open(f"{R}/{D[task]}/final_{task}_{lab}{seed}_{sol}_s{es}.csv")))
        for r in rows:
            out[(es, r["traj_id"], r["start_idx"], r["tier"])] = int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(keys) == 3000 and len(out) == 3000, (task, lab, seed, sol, len(keys), len(out))
    return out, keys, rows[0]["checkpoint_path"]


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


def pct(vals):
    vals = list(vals)
    return Fraction(100 * sum(vals), len(vals))


def r1(fr, signed=False):
    d = (Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return (f"{d:+}" if signed else f"{d}").replace("-", "−")


tier, avg, stat, ckpts = {}, {}, {}, set()
for task in D:
    lab = labels(task)
    for sol, sname in SOLS:
        data, ref = {}, None
        for name in METHODS:
            data[name] = {}
            for s in SEEDS:
                data[name][s], k, ck = load(task, lab[name], s, sol)
                ckpts.add((task, name, s, ck.split("/")[0]))
                assert ref is None or k == ref, (task, sol, name, s)  # same episodes / tiers / planner seeds
                ref = k
        keys = list(data["LeWM"][SEEDS[0]])
        for name in METHODS:
            for t in TIERS:
                tier[(task, sname, name, t)] = pct(data[name][s][k] for s in SEEDS for k in keys if k[3] == t)
            avg[(task, sname, name)] = pct(data[name][s][k] for s in SEEDS for k in keys)
        for name in BASELINES:
            occ = collections.defaultdict(lambda: collections.defaultdict(int))
            for k in keys:
                for s in SEEDS:
                    occ[(k[1], k[2])][(k[0], s)] += data["SCALE"][s][k] - data[name][s][k]
            cl = [list(x.values()) for x in occ.values()]  # per unique test episode
            d = [100 * st.mean(x) / 5 for x in cl]
            ints = [round(st.mean(x) * 2 * len(SEEDS)) for x in cl]
            assert all(abs(i - st.mean(x) * 2 * len(SEEDS)) < 1e-9 for i, x in zip(ints, cl))
            md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
            stat[(task, sname, name)] = (avg[(task, sname, "SCALE")] - avg[(task, sname, name)],
                                         md - 1.9639 * se, md + 1.9639 * se, perm_p(ints))

defn = ("Aux is the paper's auxiliary state-regression baseline (a linear readout of the state from the latent, "
        "trained with MSE). The constrained variants add an L2 penalty on the readout weights, "
        "λ_aux · (MSE + β · ‖W‖²), with β ∈ {0.01, 0.1, 1.0}. All numbers average two training seeds (1024, 1025).")
out = ["**Table E. SCALE vs. constrained-regression baselines: planning success rate (%) by planning-budget tier.** "
       + defn + " Each entry is the mean over 600 test episodes (6 sets × 100). Budgets (candidates × iterations): "
       "T1 300×30, T2 150×15, T3 50×10, T4 20×5, T5 10×3; Avg. is the mean over the five tiers.", ""]
for _, sname in SOLS:
    out += [f"**{sname} planner**", "", "| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |", "|---|---|---|---|---|---|---|---|"]
    for task in D:
        for name in METHODS:
            out.append(f"| {TITLE[task]} | {name} | " + " | ".join(r1(tier[(task, sname, name, t)]) for t in TIERS)
                       + f" | {r1(avg[(task, sname, name)])} |")
    out.append("")
out += ["**Table F. SCALE against each regression baseline: success rate (%) averaged over the five budget tiers, "
        "with paired significance tests.** All models are evaluated on the same 600 test episodes with identical "
        "planner noise, so outcomes are paired per episode. Δ = SCALE − baseline in percentage points (positive = "
        "SCALE better); the 95% CI is a paired t-interval over episodes; p is from an exact two-sided paired "
        "sign-flip permutation test. All numbers average two training seeds (1024, 1025).", "",
        "| Task | Planner | Baseline | Baseline success | SCALE | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|---|"]
for task in D:
    for _, sname in SOLS:
        for name in BASELINES:
            d, lo, hi, p = stat[(task, sname, name)]
            out.append(f"| {TITLE[task]} | {sname} | {name} | {r1(avg[(task, sname, name)])} | "
                       f"{r1(avg[(task, sname, 'SCALE')])} | {r1(d, True)} | [{sg(lo)}, {sg(hi)}] | {fp(p)} |")
out.append("")
# Table F, short form: success only
out += ["**Table F (short). SCALE and each regression baseline: success rate (%) averaged over the five budget tiers.** "
        "All numbers average two training seeds (1024, 1025).", "",
        "| Task | Planner | Baseline | Baseline success | SCALE |", "|---|---|---|---|---|"]
for task in D:
    for _, sname in SOLS:
        for name in BASELINES:
            out.append(f"| {TITLE[task]} | {sname} | {name} | {r1(avg[(task, sname, name)])} | "
                       f"{r1(avg[(task, sname, 'SCALE')])} |")
out.append("")
# Table G: held-out distance alignment of every method (mean of the two training seeds)
import json
runs = {(t, n, s): r for t, n, s, r in ckpts}
align = {}
for task in D:
    for name in METHODS:
        v = [json.load(open(f"{R}/diag/{task}__{runs[(task, name, s)]}.json"))["rho"]["full_q"]["spearman"] for s in SEEDS]
        align[(task, name)] = (st.mean(v), v)
out += ["**Table G. Distance alignment of the learned representation.** Spearman correlation between latent "
        "distance and ground-truth state distance on held-out data (1,500 samples); higher means latent distances "
        "track state distances more closely. Mean of two training seeds (1024, 1025). At β = 1.0 on Push-T, Reacher "
        "and Cube the penalty drives the readout weights to zero, so the regression term no longer shapes the "
        "representation.", "",
        "| Task | LeWM | Aux | Aux + L2 (β = 0.01) | Aux + L2 (β = 0.1) | Aux + L2 (β = 1.0) | SCALE |",
        "|---|---|---|---|---|---|---|"]
for task in D:
    out.append(f"| {TITLE[task]} | " + " | ".join(f"{align[(task, n)][0]:.2f}" for n in METHODS) + " |")
out.append("")
p = os.path.join(R, "CREG_TABLES_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
print("checkpoints used:")
for c in sorted(ckpts):
    print("  ", *c)
sig = [(k, v) for k, v in stat.items() if v[3] < 0.05]
print("Table F: p<0.05 in", len(sig), "of", len(stat), "| SCALE better:", sum(1 for k, v in sig if v[0] > 0),
      "| SCALE worse:", sum(1 for k, v in sig if v[0] < 0))
