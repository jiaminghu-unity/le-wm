#!/usr/bin/env python3
"""Seed-level results in two levels (user, 2026-11-13): (1) the two runs of the submission pooled exactly as
in the paper, (2) the new run (seed 1025) on its own. SCALE vs LeWM, tier-averaged success, paired over the
600 test episodes (95% t-interval, exact sign-flip test); p printed as "<0.05" or the number.
Level 1 reproduces the Overall column of the paper's Tables 13-17. NEW_SEED selects the new run (default 1025).
Writes eval_results/rebuttal_mirror/SEEDLEVEL2_PAPER_VS_<seed>_EN.md.
"""
import collections, contextlib, io, math, os, statistics as st, sys
from fractions import Fraction

NEW = os.environ.get("NEW_SEED", "1025")
os.environ["NEW_SEEDS"] = ""                      # the significance module then holds the paper's two runs only
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import rebuttal_sig_4seeds_en as S


def fp(p):
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}" if round(p, 3) > 0.05 else f"{p:.4f}"


def within_seed(task, sol, a, b, seed):
    da, ka = S.load(task, a, seed, sol)
    db, kb = S.load(task, b, seed, sol)
    assert ka == kb
    ref = {(k[0], k[2]): (k[3], k[4]) for k in ka}
    per = collections.defaultdict(int)
    for (es, ep, tier), v in da.items():
        per[(es, ep)] += v - db[(es, ep, tier)]
    occ = collections.defaultdict(list)
    for k, v in per.items():
        occ[ref[k]].append(v)
    cl = [Fraction(sum(v), len(v)) for v in occ.values()]
    den = 1
    for x in cl:
        den = den * x.denominator // math.gcd(den, x.denominator)
    d = [100 * float(x) / 5 for x in cl]
    md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    sa, sb = S.pct(da.values()), S.pct(db.values())
    return sa, sb, sa - sb, md - 1.9639 * se, md + 1.9639 * se, S.perm_p([int(x * den) for x in cl])


out = ["**Seed-level results, SCALE vs. LeWM: success rate (%) averaged over the five budget tiers.** "
       "\"Submission\" pools the two training runs of the original submission, as in the paper; \"New run\" is one "
       "additional independently trained run with a new seed, reported on its own. Δ = SCALE − LeWM in percentage "
       "points. Within each training seed both methods are evaluated on the same 600 test episodes (6 sets × 100) with "
       "identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes "
       "and p is from an exact two-sided paired sign-flip permutation test.", "",
       "| Task | Planner | Training runs | LeWM | SCALE | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|---|"]
for task in S.ARMS:
    for sol, sname in S.SOLS:
        d, lo, hi, p, pos = S.comp[(task, sname, "LeWM")]
        out.append(f"| {S.TITLE[task]} | {sname} | Submission (2 runs) | {S.r1(S.avg[(task, sname, 'LeWM')])} | "
                   f"{S.r1(S.avg[(task, sname, 'SCALE')])} | {S.r1(d, True)} | [{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
        sa, sb, d, lo, hi, p = within_seed(task, sol, "SCALE", "LeWM", NEW)
        out.append(f"| {S.TITLE[task]} | {sname} | New run | {S.r1(sb)} | {S.r1(sa)} | {S.r1(d, True)} | "
                   f"[{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
path = os.path.join(S.R, "rebuttal_mirror", f"SEEDLEVEL2_PAPER_VS_{NEW}_EN.md")
open(path, "w").write("\n".join(out) + "\n")
print("\n".join(out[2:]))
