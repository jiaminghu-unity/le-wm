#!/usr/bin/env python3
"""English rebuttal tables for the component ablation (reviewer 9CWa Q2): SCALE supervised on all
state components, on the high-recoverability subset only, or on the low-recoverability subset
only, vs the LeWM baseline. Push-T and Cube, training seeds 1024 and 1025 (averaged).
Writes eval_results/rebuttal_mirror/RECOV_TABLES_EN.md.

Table A (wide, one per planner): success by budget tier T1-T5 and their mean.
Table B: per task / planner / SCALE variant, tier-averaged success vs LeWM with difference,
95% CI and p-value. Unit = test episode (600 = 6 sets x 100, same episodes and planner noise for
every model); per-episode difference = mean over the two seeds and five tiers of (success of the
variant - success of LeWM); 95% t-interval over episodes; exact two-sided sign-flip test.
Pure standard library.
"""
import collections, csv, math, os, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
TITLE = {"pusht": "Push-T", "cube": "Cube"}
BASE = {"pusht": "c1r", "cube": "k1r"}
VARIANTS = {"pusht": (("All", "c3r"), ("High-recoverability only", "c3rhr"), ("Low-recoverability only", "c3rlr")),
            "cube": (("All", "k2r"), ("High-recoverability only", "k2rhr"), ("Low-recoverability only", "k2rlr"))}
METHOD = {"All": "SCALE (all components)", "High-recoverability only": "SCALE (high-recoverability only)",
          "Low-recoverability only": "SCALE (low-recoverability only)"}
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
SEEDS = ("1024", "1025")
ES = [str(e) for e in range(101, 107)]
TIERS = ["T1", "T2", "T3", "T4", "T5"]


def load(task, lab, seed, sol):
    out, keys = {}, []
    for es in ES:
        for r in csv.DictReader(open(f"{R}/final_eval/final_{task}_{lab}{seed}_{sol}_s{es}.csv")):
            out[(es, r["traj_id"], r["start_idx"], r["tier"])] = int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(keys) == 3000 and len(out) == 3000, (task, lab, seed, sol)
    return out, keys


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


tier, avg, stat = {}, {}, {}
for task in TITLE:
    for sol, sname in SOLS:
        data, ref = {}, None
        for name, lab in (("LeWM", BASE[task]),) + tuple((METHOD[v], l) for v, l in VARIANTS[task]):
            data[name] = {}
            for s in SEEDS:
                data[name][s], k = load(task, lab, s, sol)
                assert ref is None or k == ref, (task, sol, name, s)  # same episodes / tiers / planner seeds
                ref = k
        keys = list(data["LeWM"][SEEDS[0]])
        for name in data:
            for t in TIERS:
                tier[(task, sname, name, t)] = pct(data[name][s][k] for s in SEEDS for k in keys if k[3] == t)
            avg[(task, sname, name)] = pct(data[name][s][k] for s in SEEDS for k in keys)
        for v, _ in VARIANTS[task]:
            name = METHOD[v]
            occ = collections.defaultdict(lambda: collections.defaultdict(int))
            for k in keys:
                for s in SEEDS:
                    occ[(k[1], k[2])][(k[0], s)] += data[name][s][k] - data["LeWM"][s][k]
            cl = [list(x.values()) for x in occ.values()]  # per unique test episode
            d = [100 * st.mean(x) / 5 for x in cl]
            ints = [round(st.mean(x) * 2 * len(SEEDS)) for x in cl]
            assert all(abs(i - st.mean(x) * 2 * len(SEEDS)) < 1e-9 for i, x in zip(ints, cl))
            md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
            stat[(task, sname, v)] = (avg[(task, sname, name)] - avg[(task, sname, "LeWM")],
                                      md - 1.9639 * se, md + 1.9639 * se, perm_p(ints))

split = ("The state components are split by how well the LeWM baseline already encodes them (held-out linear-probe "
         "R²). Push-T: high = block position (x, y); low = agent position (x, y) and block angle (cos θ, sin θ). "
         "Cube: high = end-effector position and block position (x, y, z each); low = end-effector yaw "
         "(cos 2ψ, sin 2ψ) and gripper opening. All numbers average two training seeds (1024, 1025).")
out = ["**Table C. Component ablation: planning success rate (%) when the SCALE loss supervises all state "
       "components, only the high-recoverability components, or only the low-recoverability components.** "
       + split + " Each entry is the mean over 600 test episodes (6 sets × 100). Budgets (candidates × iterations): "
       "T1 300×30, T2 150×15, T3 50×10, T4 20×5, T5 10×3; Avg. is the mean over the five tiers.", ""]
names = ["LeWM"] + [METHOD[v] for v, _ in VARIANTS["pusht"]]
for _, sname in SOLS:
    out += [f"**{sname} planner**", "", "| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |", "|---|---|---|---|---|---|---|---|"]
    for task in TITLE:
        for name in names:
            out.append(f"| {TITLE[task]} | {name} | " + " | ".join(r1(tier[(task, sname, name, t)]) for t in TIERS)
                       + f" | {r1(avg[(task, sname, name)])} |")
    out.append("")
out += ["**Table D. Component ablation: success rate (%) averaged over the five budget tiers, each SCALE variant "
        "against the LeWM baseline, with paired significance tests.** All models are evaluated on the same 600 test "
        "episodes with identical planner noise, so outcomes are paired per episode. Δ = SCALE variant − LeWM in "
        "percentage points; the 95% CI is a paired t-interval over episodes; p is from an exact two-sided paired "
        "sign-flip permutation test. All numbers average two training seeds (1024, 1025).", "",
        "| Task | Planner | Supervised components | LeWM | SCALE | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|---|"]
for task in TITLE:
    for _, sname in SOLS:
        for v, _ in VARIANTS[task]:
            d, lo, hi, p = stat[(task, sname, v)]
            out.append(f"| {TITLE[task]} | {sname} | {v} | {r1(avg[(task, sname, 'LeWM')])} | "
                       f"{r1(avg[(task, sname, METHOD[v])])} | {r1(d, True)} | [{sg(lo)}, {sg(hi)}] | {fp(p)} |")
out.append("")
p = os.path.join(R, "RECOV_TABLES_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
