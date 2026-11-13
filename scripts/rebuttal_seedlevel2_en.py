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


# Display convention of the paper (user, 2026-11-13): means printed with plain one-decimal float formatting, and the
# difference taken between the two PRINTED means -- so rows that repeat the paper's numbers are identical to its
# Tables 13-17 and to the gains quoted in its Sec. 6.2, and every delta equals the difference of the two columns
# next to it. The confidence interval and p-value are unchanged (computed from the exact per-episode differences).
def disp(x):
    return f"{float(x):.1f}"


def dlt(a, b):
    from decimal import Decimal
    d = Decimal(disp(a)) - Decimal(disp(b))
    return f"{d:+}".replace("-", "−")


# TIERSEL=T1,T2 restricts the average to those budget tiers (default: all five); only with SPLIT=1.
SEL = tuple(x for x in os.environ.get("TIERSEL", ",".join(S.TIERS)).split(",") if x)
assert set(SEL) <= set(S.TIERS)


def within_seed(task, sol, a, b, seed):
    da, ka = S.load(task, a, seed, sol)
    db, kb = S.load(task, b, seed, sol)
    assert ka == kb
    da = {k: v for k, v in da.items() if k[2] in SEL}
    db = {k: v for k, v in db.items() if k[2] in SEL}
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
    d = [100 * float(x) / len(SEL) for x in cl]
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
SPLIT = os.environ.get("SPLIT", "") == "1"       # SPLIT=1: the two submission runs on separate rows as well
assert SPLIT or len(SEL) == 5, "tier selection is implemented for SPLIT=1 only"
if SPLIT:
    out[0] = ("**Seed-level results, SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, separately "
              "for each independently trained run.** Runs 1 and 2 are the two training runs of the original submission; "
              "the new run is one additional independently trained run with a new seed. Δ = SCALE − LeWM in percentage "
              "points. Within each run both methods share the training seed and are evaluated on the same 600 test "
              "episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a "
              "paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.")
for task in S.ARMS:
    for sol, sname in S.SOLS:
        if SPLIT:
            for i, seed in enumerate(S.PAPER_SEEDS[task]):
                sa, sb, d, lo, hi, p = within_seed(task, sol, "SCALE", "LeWM", seed)
                out.append(f"| {S.TITLE[task]} | {sname} | Submission, run {i + 1} | {disp(sb)} | {disp(sa)} | {dlt(sa, sb)} | "
                           f"[{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
        else:
            d, lo, hi, p, pos = S.comp[(task, sname, "LeWM")]
            out.append(f"| {S.TITLE[task]} | {sname} | Submission (2 runs) | {disp(S.avg[(task, sname, 'LeWM')])} | "
                       f"{disp(S.avg[(task, sname, 'SCALE')])} | {dlt(S.avg[(task, sname, 'SCALE')], S.avg[(task, sname, 'LeWM')])} | "
                       f"[{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
        sa, sb, d, lo, hi, p = within_seed(task, sol, "SCALE", "LeWM", NEW)
        out.append(f"| {S.TITLE[task]} | {sname} | New run | {disp(sb)} | {disp(sa)} | {dlt(sa, sb)} | "
                   f"[{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
if len(SEL) < 5:
    which = {("T1", "T2"): "averaged over the two largest planning budgets (tiers T1 and T2)",
             ("T1",): "at the largest planning budget (tier T1)",
             ("T2",): "at the second-largest planning budget (tier T2)"}.get(SEL, "averaged over tiers " + ", ".join(SEL))
    out[0] = out[0].replace("averaged over the five budget tiers", which)
    assert which in out[0]
path = os.path.join(S.R, "rebuttal_mirror", f"SEEDLEVEL{3 if SPLIT else 2}_PAPER_VS_{NEW}{'_' + ''.join(SEL) if len(SEL) < 5 else ''}_EN.md")
print("seeds behind the submission runs:", dict(S.PAPER_SEEDS))
open(path, "w").write("\n".join(out) + "\n")
print("\n".join(out[2:]))
