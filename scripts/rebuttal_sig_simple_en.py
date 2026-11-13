#!/usr/bin/env python3
"""SCALE vs LeWM, pooled over training runs, in the plain rebuttal format (no per-run columns):
    Task | Planner | LeWM | SCALE | delta | 95% CI | p-value
p is printed as "<0.05" when significant and as the number otherwise (user, 2026-11-13).
Runs = the two pooled in the paper + NEW_SEEDS (default 1024,1025; e.g. NEW_SEEDS=1025).
Statistics are those of rebuttal_sig_4seeds_en.py (paired over the 600 test episodes, per-episode
difference averaged over runs and the five tiers, exact sign-flip test). Writes
eval_results/rebuttal_mirror/SIG_SIMPLE_<tag>_EN.md.
"""
import contextlib, io, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import rebuttal_sig_4seeds_en as S


def fp(p):
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}" if round(p, 3) > 0.05 else f"{p:.4f}"


out = [f"**SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests "
       f"({S.WORD} runs per method).** Each method averages {S.WORD} independently trained runs"
       + (f": the two runs of the submission and {S.NEWTXT}" if S.NEW_SEEDS else "") + ". Δ = SCALE − LeWM in percentage "
       "points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so "
       "outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact "
       "two-sided paired sign-flip permutation test.", "",
       "| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|"]
for task in S.ARMS:
    for _, sname in S.SOLS:
        d, lo, hi, p, pos = S.comp[(task, sname, "LeWM")]
        out.append(f"| {S.TITLE[task]} | {sname} | {S.r1(S.avg[(task, sname, 'LeWM')])} | {S.r1(S.avg[(task, sname, 'SCALE')])} | "
                   f"{S.r1(d, True)} | [{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
path = os.path.join(S.R, "rebuttal_mirror", f"SIG_SIMPLE_{S.TAG}_EN.md")
open(path, "w").write("\n".join(out) + "\n")
print("\n".join(out[2:]))
