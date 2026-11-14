#!/usr/bin/env python3
"""English rebuttal tables, all five planning budgets: LeWM, Aux, Aux + L2 readout penalty, the
state-based world model and SWIM (SCALE with learned state weights) on the four multi-object tasks
of the Auto-SCALE (SWIM) grid, every method trained at the same two seeds (1024, 1025).
Writes eval_results/rebuttal_mirror/OGBMULTI_5TIER_EN.md. Supersedes the T1-only, seed-mismatched
AUXL2_OGBMULTI_EN.md.

Layout (the format settled for this rebuttal): one results table per planner, rows = task x method,
columns = T1 ... T5 and Avg.; then comparison tables Task | Planner | A | B | delta | 95% CI | p.

Statistics (pure stdlib):
    unit = test episode (a start/goal that occurs in two evaluation sets is averaged);
    per-episode value = success averaged over the method's training runs and the five tiers;
    delta = mean per-episode difference; 95% paired t-interval over episodes;
    p = exact two-sided paired sign-flip permutation test over the per-episode differences.
A comparison uses only the training seeds for which BOTH sides have all five tiers under both planners.
Runs with missing files are skipped and listed at the end of the printed output.
"""
import collections, csv, math, os, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from functools import reduce

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results")
OLD_DIR = os.path.join(R, "sd_mirror", "final_eval_ogbmulti")
NEW_DIR = os.path.join(R, "rebuttal_mirror", "final_eval_ogbmulti")
TASKS = {"cube_double": ("cubedouble", "Cube-Double"), "cube_triple": ("cubetriple", "Cube-Triple"),
         "cube_quadruple": ("cubequadruple", "Cube-Quadruple"), "scene": ("scene", "Scene")}
TIERS = ("T1", "T2", "T3", "T4", "T5")
NEW_SEEDS = ("1024", "1025")
OLD_SEEDS = ("3072", "3073", "3074")  # the earlier runs that were evaluated at all five tiers
BETAS = ("0.01", "0.1", "1.0")
LEWM, SWIM, STATE, SCALEF = "LeWM", "SWIM", "State-based", "SCALE (full state)"


def _oldlab(lab, s):
    return lab if s == "3072" else f"{lab}r{s[2:]}"


# method -> seed -> (dir, labels whose files together hold the five tiers, run name of the checkpoint)
def spec(method, s, stem):
    new = s in NEW_SEEDS
    if method == LEWM:
        return (NEW_DIR, [f"baser{s}"], f"lewm_{stem}_base_s{s}") if new else (OLD_DIR, [_oldlab("base", s)], f"lewm_{stem}_base_s{s}")
    if method == SWIM:
        return (NEW_DIR, [f"g03r{s}"], f"lewm_{stem}_scale_gate03_s{s}") if new else (OLD_DIR, [_oldlab("g03", s)], f"lewm_{stem}_scale_gate03_s{s}")
    if method == SCALEF:
        return None if new else (OLD_DIR, [_oldlab("obj", s)], f"lewm_{stem}_obj0.1_s{s}")
    if not new:
        return None
    if method == "Aux":
        return NEW_DIR, [f"auxr{s}", f"t25auxr{s}"], f"lewm_{stem}_aux0.1_s{s}"
    if method == STATE:
        return NEW_DIR, [f"qfr{s}", f"t25qfr{s}"], f"lewm_qinput_{stem}_s{s}"
    b = AUXB[method]
    return NEW_DIR, [f"auxh{b}r{s}", f"t25auxh{b}r{s}"], f"lewm_{stem}_aux0.1_hl2_{b}_s{s}"


AUXB = {f"Aux + L2 (β = {'1' if b == '1.0' else b})": b for b in BETAS}
MAIN = [LEWM, "Aux"] + list(AUXB) + [STATE, SWIM]          # the seed-matched methods (1024, 1025)
ALLM = MAIN + [SCALEF]
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
ES = [str(e) for e in range(101, 107)]
missing = []


def load(task, method, seed, sol):
    """-> ({tier: {(eval set, traj_id, start_idx): success}}, reference rows) or None if incomplete"""
    sp = spec(method, seed, TASKS[task][0])
    if sp is None:
        return None
    d, labs, want = sp
    out, ref = {t: {} for t in TIERS}, []
    for es in ES:
        per = collections.defaultdict(list)
        for lab in labs:
            f = os.path.join(d, f"final_{task}_{lab}_{sol}_s{es}.csv")
            if not os.path.exists(f):
                missing.append(os.path.relpath(f, R))
                return None
            for r in csv.DictReader(open(f)):
                assert r["checkpoint_path"].split("/")[0] == want, (f, r["checkpoint_path"], want)
                per[r["tier"]].append(r)
        for t in TIERS:
            rows = per.get(t, [])
            if not rows:
                missing.append(f"{task} {method} s{seed} {sol} set {es}: no {t} rows")
                return None
            assert len(rows) == 100, (task, method, seed, sol, es, t, len(rows))
            for r in rows:
                out[t][(es, r["traj_id"], r["start_idx"])] = int(r["success"])
                ref.append((t, es, r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert all(len(out[t]) == 600 for t in TIERS), (task, method, seed, sol)
    return out, sorted(ref)


def perm_p(ints):
    g = reduce(math.gcd, ints, 0) or 1
    ints = [a // g for a in ints]
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
    """the user's p-value style: '<0.05' when significant, otherwise the number"""
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}" if round(p, 3) > 0.05 else f"{p:.4f}"


def sg(x):
    return f"{x:+.1f}".replace("-", "−")


def r1(fr, signed=False):
    d = (Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return (f"{d:+}" if signed else f"{d}").replace("-", "−")


def lcm(a, b):
    return a * b // math.gcd(a, b)


# data[(task, sol)][method] = {seed: {tier: {episode key: success}}}
data = {}
for task in TASKS:
    for sol, _ in SOLS:
        ref = None
        data[(task, sol)] = {}
        for method in ALLM:
            got = {}
            for s in NEW_SEEDS + OLD_SEEDS:
                x = load(task, method, s, sol)
                if x is None:
                    continue
                got[s], k = x
                assert ref is None or k == ref, (task, sol, method, s)  # same episodes and planner noise, every tier
                ref = k
            data[(task, sol)][method] = got


def tier_sr(task, sol, method, seeds):
    """{tier: success rate in %} over the given seeds, or None unless every seed is complete"""
    got = data[(task, sol)][method]
    if any(s not in got for s in seeds):
        return None
    return {t: Fraction(100 * sum(v for s in seeds for v in got[s][t].values()), 600 * len(seeds)) for t in TIERS}


def epi_mean(task, sol, method, seeds):
    """per episode key: success averaged over the seeds and the five tiers (Fraction)"""
    got = data[(task, sol)][method]
    keys = got[seeds[0]][TIERS[0]].keys()
    return {k: Fraction(sum(got[s][t][k] for s in seeds for t in TIERS), len(seeds) * len(TIERS)) for k in keys}


def compare(a, b, tasks, pool):
    """A - B, five-tier average, pooled over the episodes of `tasks`; seeds = those of `pool` complete for both.
    -> {planner: (srA, srB, diff, lo, hi, p, n episodes, {task: seeds used})}"""
    out = {}
    for sol, sname in SOLS:
        cl, ea_all, eb_all, used = [], [], [], {}
        for t in tasks:
            # a training seed counts only if both methods have it complete for both planners
            seeds = [s for s in pool if all(s in data[(t, so)][m] for so, _ in SOLS for m in (a, b))]
            if not seeds:
                break
            used[t] = seeds
            ea, eb = epi_mean(t, sol, a, seeds), epi_mean(t, sol, b, seeds)
            occ = collections.defaultdict(list)
            for k in ea:
                occ[(t, k[1], k[2])].append(ea[k] - eb[k])  # unique start/goal within the task
            cl += [sum(v) / len(v) for v in occ.values()]
            ea_all += list(ea.values()); eb_all += list(eb.values())
        else:
            den = reduce(lcm, (x.denominator for x in cl), 1)
            ints = [int(x * den) for x in cl]
            d = [100 * float(x) for x in cl]
            md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
            tq = 1.9639 if len(d) < 1500 else 1.9610  # t quantile, df ~ 600 / ~ 2400
            sa, sb = 100 * sum(ea_all) / len(ea_all), 100 * sum(eb_all) / len(eb_all)
            out[sname] = (sa, sb, sa - sb, md - tq * se, md + tq * se, perm_p(ints), len(cl), used)
    return out


GROUPS = [(v[1], [t]) for t, v in TASKS.items()] + [("All four", list(TASKS))]
have = [m for m in MAIN if any(tier_sr(t, s, m, NEW_SEEDS) for t in TASKS for s, _ in SOLS)]
defn = ("LeWM is the image-only baseline. Aux adds a head that regresses the full simulator state from the latent "
        "(weight 0.1); Aux + L2 adds a penalty β·‖W‖² on that head's weights inside the same term. State-based is LeWM "
        "with the image encoder replaced by an MLP over the full simulator state (27-d, 32-d, 37-d and 26-d for the "
        "four tasks); it sees no images. SWIM is SCALE with per-dimension state weights learned from an inverse "
        "(action-prediction) model. Every method was trained twice, with the same two training seeds (1024 and 1025).")
out = ["**Table P. LeWM, Aux, Aux + L2, a state-based world model and SWIM on four multi-object manipulation tasks: "
       "planning success rate (%) at the five planning budgets (T1 largest … T5 smallest) and their average.** " + defn
       + " Each entry averages the two training runs over 600 test episodes (6 sets × 100); every model sees the same "
       "episodes and planner noise.", ""]
for sol, sname in SOLS:
    out += [f"**{sname} planner**", "", "| Task | Method | " + " | ".join(TIERS) + " | Avg. |", "|---|---|" + "---|" * (len(TIERS) + 1)]
    tot = {m: [] for m in have}
    for task, (_, tname) in TASKS.items():
        for m in have:
            x = tier_sr(task, sol, m, NEW_SEEDS)
            if x is None:
                out.append(f"| {tname} | {m} | " + " | ".join("–" for _ in TIERS) + " | – |")
                tot[m] = None
                continue
            out.append(f"| {tname} | {m} | " + " | ".join(r1(x[t]) for t in TIERS) + f" | {r1(sum(x.values()) / len(TIERS))} |")
            if tot[m] is not None:
                tot[m].append(x)
    for m in have:
        if tot[m] is None:
            continue
        x = {t: sum(y[t] for y in tot[m]) / len(tot[m]) for t in TIERS}
        out.append(f"| Mean of four tasks | {m} | " + " | ".join(r1(x[t]) for t in TIERS) + f" | {r1(sum(x.values()) / len(TIERS))} |")
    out.append("")

stat = ("Success is averaged over the five planning budgets and the two training runs (seeds 1024 and 1025 on both "
        "sides). Outcomes are paired per test episode; the 95% CI is a paired t-interval over episodes and p is from an "
        "exact two-sided paired sign-flip permutation test. \"All four\" pools the episodes of the four tasks.")


def cmp_table(title, pairs, pool=NEW_SEEDS, colA="A", colB="B"):
    rows = [title, "", f"| Task | Planner | {colA} | {colB} | {colA} SR | {colB} SR | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|---|---|"]
    n0 = len(rows)
    for tname, tasks in GROUPS:
        for _, sname in SOLS:
            for a, b in pairs:
                c = compare(a, b, tasks, pool).get(sname)
                if c is None:
                    continue
                sa, sb, d, lo, hi, p, n, used = c
                rows.append(f"| {tname} | {sname} | {a} | {b} | {r1(sa)} | {r1(sb)} | {r1(d, True)} | [{sg(lo)}, {sg(hi)}] | {fp(p)} |")
    return rows + [""] if len(rows) > n0 else []


others = [m for m in have if m != LEWM]
out += cmp_table("**Table Q. Each method against LeWM, with paired significance tests.** Δ = A − LeWM in percentage "
                 "points (positive = A better than LeWM). " + stat, [(m, LEWM) for m in others])
out += cmp_table("**Table R. SWIM against Aux, Aux + L2 and the state-based world model, with paired significance "
                 "tests.** Δ = SWIM − B in percentage points (positive = SWIM better). " + stat,
                 [(SWIM, m) for m in have if m not in (SWIM, LEWM)])

# all training runs of LeWM and SWIM that were evaluated at the five tiers (the two new seeds + the earlier ones)
ALL5 = NEW_SEEDS + OLD_SEEDS
c5 = {g: compare(SWIM, LEWM, tasks, ALL5) for g, tasks in GROUPS}
if any(c5.values()):
    used = {}
    for g, tasks in GROUPS[:-1]:
        for sname, c in c5[g].items():
            used.setdefault(g, set()).update(c[7][tasks[0]])
    runs = "; ".join(f"{g}: {', '.join(sorted(v))}" for g, v in used.items())
    t = cmp_table("**Table S. SWIM against LeWM over every training run that was evaluated at all five planning budgets.** "
                  "Δ = SWIM − LeWM in percentage points. Same statistics as Table Q, with success averaged over these "
                  "training seeds (the same seeds on both sides) — " + runs + ".", [(SWIM, LEWM)], ALL5)
    out += t

p = os.path.join(R, "rebuttal_mirror", "OGBMULTI_5TIER_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
print("reference (not in the file): earlier training seeds, five-tier average")
for a, b in [(SCALEF, LEWM), (SWIM, SCALEF)]:
    for tname, tasks in GROUPS:
        for sname, c in compare(a, b, tasks, OLD_SEEDS).items():
            print(f"  {a} − {b:18s} {tname:14s} {sname:4s} {r1(c[1])} -> {r1(c[0])}  {r1(c[2], True)} [{sg(c[3])}, {sg(c[4])}] "
                  f"p={c[5]:.4f} n={c[6]} seeds={ {k: ','.join(v) for k, v in c[7].items()} }")
print("reference: per-run five-tier averages of the seed-matched methods (task, planner, method: seed -> %)")
for task, (_, tname) in TASKS.items():
    for sol, sname in SOLS:
        for m in have:
            cells = []
            for s in NEW_SEEDS:
                x = tier_sr(task, sol, m, (s,))
                cells.append(f"{s}: {r1(sum(x.values()) / len(TIERS)) if x else '–'}")
            print(f"  {tname:14s} {sname:4s} {m:22s} " + "  ".join(cells))
if missing:
    u = sorted(set(missing))
    print(f"missing: {len(u)} (first: {u[0]})")
