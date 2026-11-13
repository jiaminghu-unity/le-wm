#!/usr/bin/env python3
"""English rebuttal tables: state-based world model vs LeWM and SCALE (reviewer 3Tuk, W6).
Training seeds 1024 and 1025 only (averaged). Writes eval_results/rebuttal_mirror/STATE_TABLES_EN.md.

State-based model = LeWM with the image encoder replaced by an MLP over the complete simulator
state (qjepa.QJEPA); predictor, regulariser and latent-space planning unchanged.

Table H (wide, one per planner): success by budget tier T1-T5 and their mean.
Table I: per task / planner, tier-averaged success of the state-based model and of SCALE with
difference (SCALE - state-based), 95% CI and p-value. Unit = test episode (600 = 6 sets x 100,
same episodes and planner noise for every model); per-episode difference = mean over the two
seeds and five tiers; 95% t-interval over episodes; exact two-sided sign-flip test. Pure stdlib.
"""
import collections, csv, math, os, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
D = {"pusht": "final_eval", "reacher": "final_eval", "cube": "final_eval",
     "tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
LAB = {"pusht": ("c1r", "c3r"), "reacher": ("r1r", "r2r"), "cube": ("k1r", "k2r"),
       "tworoom": ("t1r", "t2r"), "pointmaze": ("p1r", "p2r")}  # LeWM, SCALE
STATE_RUN = {"pusht": "lewm_qinput_full_pusht", "reacher": "lewm_qinput_native_reacher", "cube": "lewm_qinput_full_cube",
             "tworoom": "lewm_qinput_tworoom", "pointmaze": "lewm_qinput_pointmaze"}
METHODS = ["LeWM", "SCALE", "State-based"]
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
SEEDS = ("1024", "1025")
ES = [str(e) for e in range(101, 107)]
TIERS = ["T1", "T2", "T3", "T4", "T5"]


def load(task, lab, seed, sol, expect_run=None):
    out, keys = {}, []
    for es in ES:
        rows = list(csv.DictReader(open(f"{R}/{D[task]}/final_{task}_{lab}{seed}_{sol}_s{es}.csv")))
        if expect_run:
            assert rows[0]["checkpoint_path"].split("/")[0] == f"{expect_run}_s{seed}", rows[0]["checkpoint_path"]
        for r in rows:
            out[(es, r["traj_id"], r["start_idx"], r["tier"])] = int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(keys) == 3000 and len(out) == 3000, (task, lab, seed, sol, len(keys), len(out))
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


tier, avg, stat, per_seed = {}, {}, {}, {}
for task in D:
    lw, sc = LAB[task]
    for sol, sname in SOLS:
        data, ref = {}, None
        for name, lab, run in (("LeWM", lw, None), ("SCALE", sc, None), ("State-based", "qfr", STATE_RUN[task])):
            data[name] = {}
            for s in SEEDS:
                data[name][s], k = load(task, lab, s, sol, run)
                assert ref is None or k == ref, (task, sol, name, s)  # same episodes / tiers / planner seeds
                ref = k
        keys = list(data["LeWM"][SEEDS[0]])
        for name in METHODS:
            for t in TIERS:
                tier[(task, sname, name, t)] = pct(data[name][s][k] for s in SEEDS for k in keys if k[3] == t)
            avg[(task, sname, name)] = pct(data[name][s][k] for s in SEEDS for k in keys)
            per_seed[(task, sname, name)] = [pct(data[name][s][k] for k in keys) for s in SEEDS]
        for other in ("SCALE", "LeWM"):
            occ = collections.defaultdict(lambda: collections.defaultdict(int))
            for k in keys:
                for s in SEEDS:
                    occ[(k[1], k[2])][(k[0], s)] += data[other][s][k] - data["State-based"][s][k]
            cl = [list(x.values()) for x in occ.values()]  # per unique test episode
            d = [100 * st.mean(x) / 5 for x in cl]
            ints = [round(st.mean(x) * 2 * len(SEEDS)) for x in cl]
            assert all(abs(i - st.mean(x) * 2 * len(SEEDS)) < 1e-9 for i, x in zip(ints, cl))
            md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
            stat[(task, sname, other)] = (avg[(task, sname, other)] - avg[(task, sname, "State-based")],
                                          md - 1.9639 * se, md + 1.9639 * se, perm_p(ints))

defn = ("The state-based model is LeWM with the image encoder replaced by an MLP over the complete simulator state "
        "(Push-T: agent and block position, block angle, agent velocity, 8-d; Reacher: joint angles, fingertip "
        "position, joint velocities, 8-d; Cube: end-effector pose, gripper, arm joints, block position and yaw, 22-d; "
        "Two-Room: agent position, 2-d; PointMaze: position and velocity, 4-d); the predictor, the regulariser and "
        "latent-space planning are unchanged. LeWM and SCALE take images. All numbers average two training seeds "
        "(1024, 1025).")
out = ["**Table H. State-based world model vs. image-based LeWM and SCALE: planning success rate (%) by "
       "planning-budget tier.** " + defn + " Each entry is the mean over 600 test episodes (6 sets × 100). Budgets "
       "(candidates × iterations): T1 300×30, T2 150×15, T3 50×10, T4 20×5, T5 10×3; Avg. is the mean over the five "
       "tiers.", ""]
for _, sname in SOLS:
    out += [f"**{sname} planner**", "", "| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |", "|---|---|---|---|---|---|---|---|"]
    for task in D:
        for name in METHODS:
            out.append(f"| {TITLE[task]} | {name} | " + " | ".join(r1(tier[(task, sname, name, t)]) for t in TIERS)
                       + f" | {r1(avg[(task, sname, name)])} |")
    out.append("")
out += ["**Table I. SCALE against the state-based world model: success rate (%) averaged over the five budget tiers, "
        "with paired significance tests.** Both models are evaluated on the same 600 test episodes with identical "
        "planner noise, so outcomes are paired per episode. Δ = SCALE − state-based in percentage points (positive = "
        "SCALE better); the 95% CI is a paired t-interval over episodes; p is from an exact two-sided paired "
        "sign-flip permutation test. All numbers average two training seeds (1024, 1025).", "",
        "| Task | Planner | State-based | SCALE | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|"]
for task in D:
    for _, sname in SOLS:
        d, lo, hi, p = stat[(task, sname, "SCALE")]
        out.append(f"| {TITLE[task]} | {sname} | {r1(avg[(task, sname, 'State-based')])} | "
                   f"{r1(avg[(task, sname, 'SCALE')])} | {r1(d, True)} | [{sg(lo)}, {sg(hi)}] | {fp(p)} |")
out.append("")
p = os.path.join(R, "STATE_TABLES_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
print("LeWM minus state-based (for reference):")
for task in D:
    for _, sname in SOLS:
        d, lo, hi, pv = stat[(task, sname, "LeWM")]
        print(f"  {TITLE[task]:9s} {sname:4s} LeWM {r1(avg[(task, sname, 'LeWM')])}  State {r1(avg[(task, sname, 'State-based')])}  "
              f"diff {r1(d, True)} [{sg(lo)}, {sg(hi)}] p={fp(pv)}")
print("per-seed tier-averaged success (1024 / 1025):")
for task in D:
    for _, sname in SOLS:
        print("  ", TITLE[task], sname, {n: " / ".join(r1(x) for x in per_seed[(task, sname, n)]) for n in METHODS})
