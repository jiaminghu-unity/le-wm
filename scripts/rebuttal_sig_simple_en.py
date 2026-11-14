#!/usr/bin/env python3
"""Method A vs method B (default SCALE vs LeWM), pooled over training runs, in the plain rebuttal format
(no per-run columns):   Task | Planner | B | A | delta | 95% CI | p-value
p is printed as "<0.05" when significant and as the number otherwise (user, 2026-11-13).
Runs = the two pooled in the paper + NEW_SEEDS (default 1024,1025; e.g. NEW_SEEDS=1025).
METHOD_A / METHOD_B choose the pair among LeWM, SCALE, Aux (e.g. METHOD_A=Aux for Aux vs LeWM).
Paired over the 600 test episodes: per-episode difference averaged over runs and the five tiers, 95%
t-interval, exact sign-flip test (same statistics as rebuttal_sig_4seeds_en.py, checked below).
Writes eval_results/rebuttal_mirror/SIG_SIMPLE_<tag>_EN.md (SCALE vs LeWM) or SIG_SIMPLE_<A>_vs_<B>_<tag>_EN.md.
"""
import collections, contextlib, io, math, os, statistics as st, sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import rebuttal_sig_4seeds_en as S
A, B = os.environ.get("METHOD_A", "SCALE"), os.environ.get("METHOD_B", "LeWM")
assert {A, B} <= {"LeWM", "SCALE", "Aux"} and A != B


def fp(p):
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}" if round(p, 3) > 0.05 else f"{p:.4f}"


def pooled(task, sol, a, b):
    seeds = S.PAPER_SEEDS[task] + S.NEW_SEEDS
    da, db, ref = {}, {}, None
    for s in seeds:
        da[s], k = S.load(task, a, s, sol)
        db[s], k2 = S.load(task, b, s, sol)
        assert k == k2 and (ref is None or k == ref)
        ref = k
    ident = {(k[0], k[2]): (k[3], k[4]) for k in ref}
    occ = collections.defaultdict(list)
    for (es, ep), uid in ident.items():
        occ[uid].append(Fraction(sum(da[s][(es, ep, t)] - db[s][(es, ep, t)] for s in seeds for t in S.TIERS), 5 * len(seeds)))
    cl = [sum(v) / len(v) for v in occ.values()]
    den = 1
    for x in cl:
        den = den * x.denominator // math.gcd(den, x.denominator)
    d = [100 * float(x) for x in cl]
    md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    ma = S.pct(v for s in seeds for v in da[s].values()); mb = S.pct(v for s in seeds for v in db[s].values())
    pos = sum(1 for s in seeds if sum(da[s].values()) > sum(db[s].values()))
    return mb, ma, ma - mb, md - 1.9639 * se, md + 1.9639 * se, S.perm_p([int(x * den) for x in cl]), pos, len(seeds)


desc = {"SCALE": "SCALE", "LeWM": "LeWM", "Aux": "Aux (the auxiliary state-regression control)"}
out = [f"**{A} vs. {B}: success rate (%) averaged over the five budget tiers, with paired significance tests "
       f"({S.WORD} runs per method).** Each method averages {S.WORD} independently trained runs"
       + (f": the two runs of the submission and {S.NEWTXT}" if S.NEW_SEEDS else "") + f". Δ = {A} − {B} in percentage "
       "points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so "
       "outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact "
       "two-sided paired sign-flip permutation test.", "",
       f"| Task | Planner | {B} | {A} | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|"]
extra = []
for task in S.ARMS:
    for sol, sname in S.SOLS:
        mb, ma, d, lo, hi, p, pos, n = pooled(task, sol, A, B)
        if (A, B) == ("SCALE", "LeWM"):      # must reproduce the reference implementation exactly
            rd, rlo, rhi, rp, rpos = S.comp[(task, sname, "LeWM")]
            assert rd == d and abs(rlo - lo) < 1e-9 and abs(rhi - hi) < 1e-9 and abs(rp - p) < 1e-12, (task, sname)
        if S.NEW_SEEDS:
            cells = (S.r1(mb), S.r1(ma), S.r1(d, True))
        else:   # only the submission's two runs: print exactly as the paper does (its Tables 13-17 and Sec. 6.2)
            from decimal import Decimal
            fa, fb = f"{float(ma):.1f}", f"{float(mb):.1f}"
            cells = (fb, fa, f"{Decimal(fa) - Decimal(fb):+}".replace("-", "−"))
        out.append(f"| {S.TITLE[task]} | {sname} | {cells[0]} | {cells[1]} | {cells[2]} | [{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
        extra.append(f"  {S.TITLE[task]:9s} {sname:4s} runs with {A} ahead: {pos}/{n}   exact p = {S.fp(p)}")
name = f"SIG_SIMPLE_{S.TAG}_EN.md" if (A, B) == ("SCALE", "LeWM") else f"SIG_SIMPLE_{A}_vs_{B}_{S.TAG}_EN.md"
open(os.path.join(S.R, "rebuttal_mirror", name), "w").write("\n".join(out) + "\n")
print("\n".join(out[2:]))
print("\n".join(extra))
