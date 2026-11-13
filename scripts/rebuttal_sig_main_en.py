#!/usr/bin/env python3
"""Significance of SCALE vs LeWM over ALL training seeds, and paper seeds vs new seeds.
Writes eval_results/rebuttal_mirror/SIG_MAIN_EN.md (English tables) and prints a consistency check.

Training seeds: 3072, 3073, 3074, 1024, 1025 (Two-Room has no 3074). The paper's Tables 13-17 pool
3072+3073 (PointMaze: 3072+3074); 1024 and 1025 were trained for the rebuttal.

Table J (across training seeds): unit = training seed; within-seed difference of tier-averaged
success (same 600 test episodes, same planner noise); 95% t-interval and paired t-test over seeds.
Table K (across test episodes): unit = test episode (600); per-episode difference = mean over all
seeds and the five tiers; 95% t-interval over episodes; exact two-sided sign-flip permutation test.
File selection as in rebuttal_pooled_stats.py (canonical directories, GCS mirrors first).
Pure standard library.
"""
import collections, csv, glob, math, os, re, statistics as st
from decimal import Decimal, ROUND_HALF_UP

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results")
ARMS = {"pusht": {"LeWM": "lewm_c1", "SCALE": "lewm_c3_sig_obj0.1", "Aux": "lewm_c5_qhead0.3"},
        "reacher": {"LeWM": "lewm_r1_reacher", "SCALE": "lewm_r2_reacher_paep_l015", "Aux": "lewm_r5_qhead0.4"},
        "cube": {"LeWM": "lewm_k1_cube", "SCALE": "lewm_k2_cube_obj_eff0.1", "Aux": "lewm_k4_cube_qhead_eff0.1"},
        "tworoom": {"LeWM": "lewm_t1_tworoom", "SCALE": "lewm_t2_tworoom_obj0.1", "Aux": "lewm_t5_tworoom_qhead0.1"},
        "pointmaze": {"LeWM": "lewm_p1_pointmaze", "SCALE": "lewm_p2_pointmaze", "Aux": "lewm_p5_pointmaze"}}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
SEEDS = ["3072", "3073", "3074", "1024", "1025"]
PAPER_SEEDS = {"pusht": ("3072", "3073"), "reacher": ("3072", "3073"), "cube": ("3072", "3073"),
               "tworoom": ("3072", "3073"), "pointmaze": ("3072", "3074")}
NEW_SEEDS = ("1024", "1025")
# Overall column of the paper's Tables 13-17: (LeWM, SCALE, Aux)
PAPER = {"pusht": {"cem": (70.7, 73.1, 74.3), "icem": (63.9, 68.3, 69.1)},
         "reacher": {"cem": (71.4, 71.7, 71.4), "icem": (78.8, 81.4, 80.0)},
         "cube": {"cem": (60.9, 64.1, 63.3), "icem": (64.3, 70.8, 68.4)},
         "tworoom": {"cem": (77.5, 89.2, 77.7), "icem": (82.5, 92.7, 83.3)},
         "pointmaze": {"cem": (83.5, 84.9, 81.9), "icem": (80.9, 83.4, 78.7)}}
ES = ["101", "102", "103", "104", "105", "106"]
TIERS = ["T1", "T2", "T3", "T4", "T5"]
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
BAD = ("l1", "cos", "cost", "FILTERED", "broken", "mppi", "frozen", "half", "automet")
PRIO = ("rebuttal_mirror", "sd_mirror/final_eval", "wsweep_mirror", "r73_mirror")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # by degrees of freedom
stems = {v: (t, a) for t, d in ARMS.items() for a, v in d.items()}

paths = collections.defaultdict(dict)  # cell -> {directory: path}
for f in glob.glob(os.path.join(R, "**", "final_*.csv"), recursive=True):
    d = os.path.relpath(os.path.dirname(f), R)
    if d == "." or any(b in d for b in BAD):
        continue
    m = re.match(r"final_(pusht|reacher|cube|tworoom|pointmaze)_(.+)_(cem|icem)_s(\d+)\.csv$", os.path.basename(f))
    if not m or m.group(4) not in ES:
        continue
    with open(f) as fh:
        first = next(csv.DictReader(fh), None)
    if not first:
        continue
    mm = re.match(r"(.+)_s(\d{4})/", first["checkpoint_path"])
    if not mm or mm.group(1) not in stems or stems[mm.group(1)][0] != m.group(1) or mm.group(2) not in SEEDS:
        continue
    paths[(m.group(1), stems[mm.group(1)][1], mm.group(2), m.group(3), m.group(4))][d] = f


def pick(cell):
    d = paths[cell]
    for p in PRIO:
        for k in sorted(d):
            if k.startswith(p):
                return d[k]
    return d[sorted(d)[0]]


def load(task, arm, seed, sol):
    """-> ({(es, episode_id): tiers solved 0..5}, pairing keys) or None if the seed is missing"""
    out, keys = collections.defaultdict(int), []
    for es in ES:
        cell = (task, arm, seed, sol, es)
        if cell not in paths:
            return None
        rows = list(csv.DictReader(open(pick(cell))))
        t = collections.Counter(r["tier"] for r in rows)
        if sorted(t) != TIERS or any(v != 100 for v in t.values()):
            return None
        for r in rows:
            out[(es, r["episode_id"])] += int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    return out, sorted(keys)


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


def t_p(t, df):
    """two-sided p-value of Student's t (Abramowitz & Stegun 26.7.3-4)"""
    th = math.atan(abs(t) / math.sqrt(df))
    s, c = math.sin(th), math.cos(th)
    if df % 2 == 0:
        term, tot = 1.0, 1.0
        for k in range(2, df, 2):
            term *= (k - 1) / k * c * c
            tot += term
        a = s * tot
    else:
        tot, term = (c if df > 1 else 0.0), c
        for k in range(3, df, 2):
            term *= (k - 1) / k * c * c
            tot += term
        a = (2 / math.pi) * (th + s * tot)
    return max(0.0, 1 - a)


assert abs(t_p(2.776, 4) - 0.05) < 2e-4 and abs(t_p(3.182, 3) - 0.05) < 2e-4 and abs(t_p(12.706, 1) - 0.05) < 2e-4


def fp(p):
    return "<0.0001" if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}")


def r1(x, signed=False):
    d = Decimal(repr(round(x, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return (f"{d:+}" if signed else f"{d}").replace("-", "−")


J, K, check = [], [], []
for task in ARMS:
    for sol, sname in SOLS:
        data, ref = {}, None
        for arm in ("LeWM", "SCALE", "Aux"):
            for s in SEEDS:
                got = load(task, arm, s, sol)
                if got is None:
                    continue
                data[(arm, s)], k = got
                assert ref is None or k == ref, (task, sol, arm, s)  # identical episodes / tiers / planner seeds
                ref = k
        seeds = [s for s in SEEDS if all((a, s) in data for a in ("LeWM", "SCALE", "Aux"))]
        eps = sorted(data[("LeWM", seeds[0])])
        sm = {(a, s): 100 * sum(data[(a, s)].values()) / (5 * len(eps)) for a in ("LeWM", "SCALE", "Aux") for s in seeds}
        # consistency: paper seed pair vs the paper's printed numbers, and vs the two new seeds
        pp = [st.mean(sm[(a, s)] for s in PAPER_SEEDS[task]) for a in ("LeWM", "SCALE", "Aux")]
        nn = [st.mean(sm[(a, s)] for s in NEW_SEEDS) for a in ("LeWM", "SCALE", "Aux")]
        check.append((task, sname, PAPER[task][sol], pp, nn))
        # Table J: across training seeds
        d = [sm[("SCALE", s)] - sm[("LeWM", s)] for s in seeds]
        n = len(d)
        md, se = st.mean(d), st.stdev(d) / math.sqrt(n)
        J.append((task, sname, n, st.mean(sm[("LeWM", s)] for s in seeds), st.mean(sm[("SCALE", s)] for s in seeds), md,
                  md - T975[n - 1] * se, md + T975[n - 1] * se, t_p(md / se, n - 1), sum(x > 0 for x in d), d, seeds))
        # Table K: across test episodes (unique start/goal = (traj, start); a duplicate draw is averaged)
        ident = {(k[0], k[2]): (k[3], k[4]) for k in ref}
        cl = collections.defaultdict(list)
        for e in eps:
            cl[ident[e]].append(sum(data[("SCALE", s)][e] - data[("LeWM", s)][e] for s in seeds))
        de = [100 * st.mean(v) / (5 * n) for v in cl.values()]
        ints = [round(st.mean(v) * 2) for v in cl.values()]
        assert all(abs(i - st.mean(v) * 2) < 1e-9 for i, v in zip(ints, cl.values()))
        me, see = st.mean(de), st.stdev(de) / math.sqrt(len(de))
        K.append((task, sname, n, me - 1.9639 * see, me + 1.9639 * see, perm_p(ints)))

seedtxt = ("Five training seeds per method (four on Two-Room): the two runs pooled in the paper, one additional "
           "earlier run, and two new runs (seeds 1024 and 1025).")
out = ["**Table J. SCALE vs. LeWM across training seeds: success rate (%) averaged over the five budget tiers and "
       "600 test episodes.** " + seedtxt + " Δ = SCALE − LeWM, computed within each training seed on the same test "
       "episodes with identical planner noise; the 95% CI is a t-interval over training seeds and p is from a "
       "two-sided paired t-test over training seeds.", "",
       "| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value | Seeds with Δ > 0 |", "|---|---|---|---|---|---|---|---|"]
for (task, sname, n, lw, sc, md, lo, hi, p, pos, d, seeds) in J:
    out.append(f"| {TITLE[task]} | {sname} | {r1(lw)} | {r1(sc)} | {r1(md, True)} | [{r1(lo, True)}, {r1(hi, True)}] | "
               f"{fp(p)} | {pos}/{n} |")
out += ["", "**Table K. SCALE vs. LeWM across test episodes, all training seeds pooled.** Same runs as Table J. Each "
        "of the 600 test episodes is evaluated with both methods under identical planner noise; the per-episode "
        "difference is averaged over the training seeds and the five budget tiers. The 95% CI is a paired t-interval "
        "over episodes and p is from an exact two-sided paired sign-flip permutation test.", "",
        "| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|"]
for (task, sname, n, lw, sc, md, *_), (_, _, _, lo, hi, p) in zip(J, K):
    out.append(f"| {TITLE[task]} | {sname} | {r1(lw)} | {r1(sc)} | {r1(md, True)} | [{r1(lo, True)}, {r1(hi, True)}] | {fp(p)} |")
out.append("")
p = os.path.join(R, "rebuttal_mirror", "SIG_MAIN_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
print("per-seed SCALE − LeWM:")
for (task, sname, n, lw, sc, md, lo, hi, pv, pos, d, seeds) in J:
    print(f"  {TITLE[task]:9s} {sname:4s} " + "  ".join(f"{s}:{x:+.1f}" for s, x in zip(seeds, d)))
print("consistency (LeWM / SCALE / Aux): paper printed | recomputed from the paper's seed pair | new seeds 1024+1025 | gains SCALE−LeWM paper -> new")
for task, sname, printed, pp, nn in check:
    print(f"  {TITLE[task]:9s} {sname:4s} " + " / ".join(f"{x:.1f}" for x in printed) + " | " + " / ".join(f"{x:.2f}" for x in pp)
          + " | " + " / ".join(f"{x:.1f}" for x in nn) + f" | {pp[1] - pp[0]:+.1f} -> {nn[1] - nn[0]:+.1f}"
          + f" | Aux−LeWM {pp[2] - pp[0]:+.1f} -> {nn[2] - nn[0]:+.1f} | SCALE−Aux {pp[1] - pp[2]:+.1f} -> {nn[1] - nn[2]:+.1f}")
