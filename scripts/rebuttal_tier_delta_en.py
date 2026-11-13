#!/usr/bin/env python3
"""SCALE vs LeWM separately for each planning-budget tier (T1 = largest budget ... T5 = smallest).
Runs = the two of the submission + NEW_SEEDS (default here: 1025). Paired over the 600 test episodes
(per-episode difference averaged over the runs), 95% t-interval, exact sign-flip test; p printed as
"<0.05" or the number; means and delta in the paper's display convention (delta = difference of the
printed means). Writes eval_results/rebuttal_mirror/TIER_DELTA_<tag>_EN.md (long table + wide summary).
"""
import collections, contextlib, io, math, os, statistics as st, sys
from decimal import Decimal
from fractions import Fraction

os.environ.setdefault("NEW_SEEDS", "1025")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import rebuttal_sig_4seeds_en as S


def fp(p):
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}" if round(p, 3) > 0.05 else f"{p:.4f}"


def disp(x):
    return f"{float(x):.1f}"


def dlt(a, b):
    d = Decimal(disp(a)) - Decimal(disp(b))
    return f"{d:+}".replace("-", "−")


def stats(cl):
    den = 1
    for x in cl:
        den = den * x.denominator // math.gcd(den, x.denominator)
    d = [100 * float(x) for x in cl]
    md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    tq = 1.9639 if len(d) < 1500 else 1.9608
    return md, md - tq * se, md + tq * se, S.perm_p([int(x * den) for x in cl])


res, pooled = {}, collections.defaultdict(list)
for task in S.ARMS:
    seeds = S.PAPER_SEEDS[task] + S.NEW_SEEDS
    for sol, sname in S.SOLS:
        da, db, ref = {}, {}, None
        for s in seeds:
            da[s], k = S.load(task, "SCALE", s, sol)
            db[s], k2 = S.load(task, "LeWM", s, sol)
            assert k == k2 and (ref is None or k == ref)
            ref = k
        ident = {(k[0], k[2]): (k[3], k[4]) for k in ref}
        for t in S.TIERS:
            occ = collections.defaultdict(list)
            for (es, ep), uid in ident.items():
                occ[uid].append(Fraction(sum(da[s][(es, ep, t)] - db[s][(es, ep, t)] for s in seeds), len(seeds)))
            cl = [sum(v) / len(v) for v in occ.values()]
            a = S.pct(da[s][k] for s in seeds for k in da[s] if k[2] == t)
            b = S.pct(db[s][k] for s in seeds for k in db[s] if k[2] == t)
            res[(task, sname, t)] = (b, a) + stats(cl)
            pooled[(sname, t)] += cl

long = [f"**SCALE vs. LeWM by planning-budget tier (T1 = largest budget, T5 = smallest): success rate (%) with paired "
        f"significance tests ({S.WORD} runs per method).** Δ = SCALE − LeWM in percentage points. Both methods are "
        "evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise; the 95% CI is a paired "
        "t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.", "",
        "| Task | Planner | Tier | LeWM | SCALE | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|---|"]
for task in S.ARMS:
    for _, sname in S.SOLS:
        for t in S.TIERS:
            b, a, md, lo, hi, p = res[(task, sname, t)]
            long.append(f"| {S.TITLE[task]} | {sname} | {t} | {disp(b)} | {disp(a)} | {dlt(a, b)} | [{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
wide = ["", "**Δ = SCALE − LeWM by tier (percentage points); \\* marks p < 0.05.** The last rows average the five tasks "
        "(the test pools the episodes of all tasks).", "",
        "| Task | Planner | T1 | T2 | T3 | T4 | T5 |", "|---|---|---|---|---|---|---|"]
for task in S.ARMS:
    for _, sname in S.SOLS:
        cells = []
        for t in S.TIERS:
            b, a, md, lo, hi, p = res[(task, sname, t)]
            cells.append(dlt(a, b) + ("\\*" if p < 0.05 else ""))
        wide.append(f"| {S.TITLE[task]} | {sname} | " + " | ".join(cells) + " |")
summ = []
for _, sname in S.SOLS:
    cells, sig = [], []
    for t in S.TIERS:
        md, lo, hi, p = stats(pooled[(sname, t)])
        mean5 = st.mean(float(res[(task, sname, t)][1] - res[(task, sname, t)][0]) for task in S.ARMS)
        cells.append(f"{mean5:+.1f}".replace("-", "−") + ("\\*" if p < 0.05 else ""))
        sig.append(sum(1 for task in S.ARMS if res[(task, sname, t)][5] < 0.05 and res[(task, sname, t)][2] > 0))
    wide.append(f"| Mean of five tasks | {sname} | " + " | ".join(cells) + " |")
    summ.append(f"| Tasks with significant gain | {sname} | " + " | ".join(f"{x}/5" for x in sig) + " |")
wide += summ
path = os.path.join(S.R, "rebuttal_mirror", f"TIER_DELTA_{S.TAG}_EN.md")
open(path, "w").write("\n".join(long + wide) + "\n")
print("\n".join(wide[1:]))
