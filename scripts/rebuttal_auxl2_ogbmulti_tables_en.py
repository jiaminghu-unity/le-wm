#!/usr/bin/env python3
"""English rebuttal tables: Aux and Aux + L2 readout penalty on the four multi-object tasks of the
Auto-SCALE (SWIM) grid, next to LeWM, SCALE on the full state, and SWIM (SCALE with learned state
weights). Writes eval_results/rebuttal_mirror/AUXL2_OGBMULTI_EN.md.

Protocol of that grid: T1 planning budget only (300 candidates x 30 iterations), CEM and iCEM,
six evaluation sets x 100 episodes, the same episodes and planner noise for every model.

Training runs differ by method: LeWM / SCALE / SWIM exist at seeds 3072-3076, the Aux family was
trained at seeds 1024 and 1025 (SEEDS below; override the first group with OLD_SEEDS=3072,3073).
Comparisons are therefore paired over test episodes, not over training seeds:
    unit = test episode (a start/goal that occurs in two sets is averaged);
    per-episode difference = mean over A's runs - mean over B's runs;
    95% t-interval over episodes; exact two-sided sign-flip permutation test. Pure stdlib.
Methods or runs with missing files are skipped and listed at the end of the printed output.
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
OLD_SEEDS = tuple(os.environ.get("OLD_SEEDS", "3072,3073,3074,3075,3076").split(","))
NEW_SEEDS = ("1024", "1025")
BETAS = ("0.01", "0.1", "1.0")
SWIM = "SWIM"
# method -> (dir, seeds, label(seed), run(stem, seed))
def _old(lab, run):
    return (OLD_DIR, OLD_SEEDS, lambda s: lab if s == "3072" else f"{lab}r{s[2:]}", run)
METHODS = collections.OrderedDict()
METHODS["LeWM"] = _old("base", lambda stem, s: f"lewm_{stem}_base_s{s}")
METHODS["Aux"] = (NEW_DIR, NEW_SEEDS, lambda s: f"auxr{s}", lambda stem, s: f"lewm_{stem}_aux0.1_s{s}")
for b in BETAS:
    METHODS[f"Aux + L2 (β = {b.rstrip('0').rstrip('.') if b != '1.0' else '1'})"] = (
        NEW_DIR, NEW_SEEDS, lambda s, b=b: f"auxh{b}r{s}", lambda stem, s, b=b: f"lewm_{stem}_aux0.1_hl2_{b}_s{s}")
METHODS["SCALE (full state)"] = _old("obj", lambda stem, s: f"lewm_{stem}_obj0.1_s{s}")
METHODS[SWIM] = _old("g03", lambda stem, s: f"lewm_{stem}_scale_gate03_s{s}")
AUXFAM = [m for m in METHODS if m.startswith("Aux")]
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
ES = [str(e) for e in range(101, 107)]
missing = []


def load(task, method, seed, sol):
    """-> {(eval set, traj_id, start_idx): success} for the T1 tier, or None if a file is missing"""
    d, _, lab, run = METHODS[method]
    out, ref = {}, []
    for es in ES:
        f = os.path.join(d, f"final_{task}_{lab(seed)}_{sol}_s{es}.csv")
        if not os.path.exists(f):
            missing.append(os.path.relpath(f, R))
            return None
        rows = [r for r in csv.DictReader(open(f)) if r["tier"] == "T1"]
        want = run(TASKS[task][0], seed)
        assert rows and rows[0]["checkpoint_path"].split("/")[0] == want, (f, rows[0]["checkpoint_path"] if rows else None, want)
        assert len(rows) == 100, (f, len(rows))
        for r in rows:
            out[(es, r["traj_id"], r["start_idx"])] = int(r["success"])
            ref.append((es, r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(out) == 600, (task, method, seed, sol, len(out))
    return out, ref


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


def r1(fr, signed=False):
    d = (Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return (f"{d:+}" if signed else f"{d}").replace("-", "−")


def lcm(a, b):
    return a * b // math.gcd(a, b)


# data[(task, sol)][method] = {seed: {episode key: success}}
data, sr, nruns = {}, {}, {}
for task in TASKS:
    for sol, _ in SOLS:
        ref = None
        data[(task, sol)] = {}
        for method, (_, seeds, _, _) in METHODS.items():
            got = {}
            for s in seeds:
                x = load(task, method, s, sol)
                if x is None:
                    continue
                got[s], k = x
                assert ref is None or k == ref, (task, sol, method, s)  # same episodes and planner seeds
                ref = k
            data[(task, sol)][method] = got
            nruns[(task, sol, method)] = len(got)
            if got:
                vals = [v for d in got.values() for v in d.values()]
                sr[(task, sol, method)] = Fraction(100 * sum(vals), len(vals))


def epi_mean(task, sol, method):
    """per episode key: mean success over the method's runs (Fraction)"""
    got = data[(task, sol)][method]
    keys = next(iter(got.values())).keys()
    return {k: Fraction(sum(d[k] for d in got.values()), len(got)) for k in keys}


def compare(a, b, tasks):
    """A - B over the given tasks (pooled over their episodes) for each planner -> {sol: (srA, srB, diff, lo, hi, p)}"""
    out = {}
    for sol, sname in SOLS:
        if any(not data[(t, sol)][a] or not data[(t, sol)][b] for t in tasks):
            continue
        cl, ea_all, eb_all = [], [], []
        for t in tasks:
            ea, eb = epi_mean(t, sol, a), epi_mean(t, sol, b)
            occ = collections.defaultdict(list)
            for k in ea:
                occ[(t, k[1], k[2])].append(ea[k] - eb[k])  # unique start/goal within the task
            cl += [sum(v) / len(v) for v in occ.values()]
            ea_all += list(ea.values()); eb_all += list(eb.values())
        den = reduce(lcm, (x.denominator for x in cl), 1)
        ints = [int(x * den) for x in cl]
        d = [100 * float(x) for x in cl]
        md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
        tq = 1.9639 if len(d) < 1500 else 1.9610  # t quantile, df ~ 600 / ~ 2400
        sa, sb = 100 * sum(ea_all) / len(ea_all), 100 * sum(eb_all) / len(eb_all)
        out[sname] = (sa, sb, sa - sb, md - tq * se, md + tq * se, perm_p(ints), len(cl))
    return out


have = [m for m in METHODS if any(nruns[(t, s, m)] for t in TASKS for s, _ in SOLS)]
runs_txt = {m: "/".join(sorted({str(nruns[(t, s, m)]) for t in TASKS for s, _ in SOLS})) for m in have}
defn = ("LeWM is the image-only baseline. Aux adds a head that regresses the full simulator state from the latent "
        "(weight 0.1); Aux + L2 adds a penalty β·‖W‖² on that head's weights inside the same term. SCALE (full state) "
        "aligns latent distances with distances in the full simulator state. SWIM is SCALE with per-dimension state "
        "weights learned from an inverse (action-prediction) model. LeWM, SCALE and SWIM: training seeds "
        + ", ".join(OLD_SEEDS) + "; Aux and Aux + L2: training seeds 1024 and 1025.")
out = ["**Table P. Aux and Aux + L2 on four multi-object manipulation tasks: planning success rate (%) at the "
       "largest planning budget (T1: 300 candidates × 30 iterations).** " + defn + " Each entry averages the "
       "method's training runs over 600 test episodes (6 sets × 100); every model sees the same episodes and "
       "planner noise.", ""]
for _, sname in SOLS:
    sol = sname.lower()
    out += [f"**{sname} planner**", "", "| Method | Runs | " + " | ".join(v[1] for v in TASKS.values()) + " | Mean |",
            "|---|---|" + "---|" * (len(TASKS) + 1)]
    for m in have:
        cells = [sr.get((t, sol, m)) for t in TASKS]
        mean = sum(cells) / len(cells) if all(c is not None for c in cells) else None
        out.append(f"| {m} | {runs_txt[m]} | " + " | ".join(r1(c) if c is not None else "–" for c in cells)
                   + f" | {r1(mean) if mean is not None else '–'} |")
    out.append("")

for tab, a in (("Q", SWIM), ("R", "SCALE (full state)")):
    out += [f"**Table {tab}. {a} against Aux and Aux + L2, with paired significance tests.** Δ = {a} − baseline in "
            "percentage points (positive = " + a + " better). Outcomes are paired per test episode; each method's "
            "per-episode outcome is first averaged over its training runs. The 95% CI is a paired t-interval over "
            "episodes and p is from an exact two-sided paired sign-flip permutation test. The training seeds of the "
            "two sides differ (see Table P), so the test covers test-episode variation only. \"All four\" pools the "
            "episodes of the four tasks.", "",
            f"| Task | Planner | Baseline | Baseline SR | {a} SR | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|---|"]
    for tname, tasks in [(v[1], [t]) for t, v in TASKS.items()] + [("All four", list(TASKS))]:
        for _, sname in SOLS:
            for b in AUXFAM:
                c = compare(a, b, tasks).get(sname)
                if c is None:
                    continue
                sa, sb, d, lo, hi, p, n = c
                out.append(f"| {tname} | {sname} | {b} | {r1(sb)} | {r1(sa)} | {r1(d, True)} | [{sg(lo)}, {sg(hi)}] | {fp(p)} |")
    out.append("")

p = os.path.join(R, "rebuttal_mirror", "AUXL2_OGBMULTI_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
print("reference (not in the file): SWIM and SCALE (full state) against LeWM, and Aux family against LeWM")
for a, b in [(SWIM, "LeWM"), ("SCALE (full state)", "LeWM"), (SWIM, "SCALE (full state)")] + [(m, "LeWM") for m in AUXFAM]:
    for tname, tasks in [(v[1], [t]) for t, v in TASKS.items()] + [("All four", list(TASKS))]:
        for sname, c in compare(a, b, tasks).items():
            print(f"  {a} − {b:18s} {tname:14s} {sname:4s} {r1(c[1])} -> {r1(c[0])}  {r1(c[2], True)} [{sg(c[3])}, {sg(c[4])}] p={fp(c[5])} n={c[6]}")
if missing:
    print(f"missing files: {len(missing)} (first: {missing[0]})")
