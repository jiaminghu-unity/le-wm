#!/usr/bin/env python3
"""State-based world model against LeWM, SCALE and DINO-WM on the five tasks of the paper (reviewer 3Tuk W6).
Extends rebuttal_state_tables_en.py (which only compared SCALE with the state-based model).
Writes eval_results/rebuttal_mirror/STATE_TABLES2_EN.md.

Runs: state-based, LeWM, SCALE -> training seeds 1024 and 1025; DINO-WM -> the two runs of the submission
(seeds 3072, 3073; it was never trained at 1024 / 1025). All models are evaluated on the same 600 test episodes
with identical planner noise, five budget tiers.
Table H2 (wide, one per planner): success by tier and their mean, four methods.
Table I2: each image-based method against the state-based model: tier-averaged success, delta = method - state-based,
95% paired t-interval over episodes, exact two-sided sign-flip permutation p ("<0.05" or the value).
"""
import collections, contextlib, csv, io, math, os, statistics as st, sys
from fractions import Fraction

os.environ["NEW_SEEDS"] = "1024,1025"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import rebuttal_sig_4seeds_en as S
M = os.path.join(S.R, "rebuttal_mirror")
D = {"pusht": "final_eval", "reacher": "final_eval", "cube": "final_eval", "tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}
STATE_RUN = {"pusht": "lewm_qinput_full_pusht", "reacher": "lewm_qinput_native_reacher", "cube": "lewm_qinput_full_cube",
             "tworoom": "lewm_qinput_tworoom", "pointmaze": "lewm_qinput_pointmaze"}
NEW, DINO = ("1024", "1025"), ("3072", "3073")
METHODS = ["LeWM", "SCALE", "DINO-WM", "State-based"]
SEEDS = {"LeWM": NEW, "SCALE": NEW, "DINO-WM": DINO, "State-based": NEW}


def fp(p):
    if p < 0.05:
        return "<0.05"
    return f"{p:.3f}" if round(p, 3) > 0.05 else f"{p:.4f}"


def load_state(task, seed, sol):
    out, keys = {}, []
    for es in S.ES:
        rows = list(csv.DictReader(open(f"{M}/{D[task]}/final_{task}_qfr{seed}_{sol}_s{es}.csv")))
        assert rows[0]["checkpoint_path"].split("/")[0] == f"{STATE_RUN[task]}_s{seed}", rows[0]["checkpoint_path"]
        for r in rows:
            out[(es, r["episode_id"], r["tier"])] = int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(out) == 3000
    return out, sorted(keys)


tier, avg, stat = {}, {}, {}
for task in S.ARMS:
    for sol, sname in S.SOLS:
        data, ref = {}, None
        for m in METHODS:
            for s in SEEDS[m]:
                data[(m, s)], k = load_state(task, s, sol) if m == "State-based" else S.load(task, m, s, sol)
                assert ref is None or k == ref, (task, sol, m, s)   # same episodes, tiers and planner seeds
                ref = k
        keys = sorted(data[("LeWM", NEW[0])])
        for m in METHODS:
            for t in S.TIERS:
                tier[(task, sname, m, t)] = S.pct(data[(m, s)][k] for s in SEEDS[m] for k in keys if k[2] == t)
            avg[(task, sname, m)] = S.pct(data[(m, s)][k] for s in SEEDS[m] for k in keys)
        ident = {(k[0], k[2]): (k[3], k[4]) for k in ref}
        for m in ("LeWM", "SCALE", "DINO-WM"):
            occ = collections.defaultdict(list)
            for (es, ep), uid in ident.items():
                a = Fraction(sum(data[(m, s)][(es, ep, t)] for s in SEEDS[m] for t in S.TIERS), 5 * len(SEEDS[m]))
                b = Fraction(sum(data[("State-based", s)][(es, ep, t)] for s in NEW for t in S.TIERS), 5 * len(NEW))
                occ[uid].append(a - b)
            cl = [sum(v) / len(v) for v in occ.values()]
            den = 1
            for x in cl:
                den = den * x.denominator // math.gcd(den, x.denominator)
            d = [100 * float(x) for x in cl]
            md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
            stat[(task, sname, m)] = (avg[(task, sname, m)] - avg[(task, sname, "State-based")], md - 1.9639 * se,
                                      md + 1.9639 * se, S.perm_p([int(x * den) for x in cl]))

defn = ("The state-based model is LeWM with the image encoder replaced by an MLP over the complete simulator state "
        "(Push-T 8-d, Reacher 8-d, Cube 22-d, Two-Room 2-d, PointMaze 4-d); the predictor, the regulariser and "
        "latent-space planning are unchanged. LeWM, SCALE and DINO-WM take images. State-based, LeWM and SCALE average "
        "two training seeds; DINO-WM averages the two runs of the submission.")
out = ["**Table H2. State-based world model vs. image-based LeWM, SCALE and DINO-WM: planning success rate (%) by "
       "planning-budget tier.** " + defn + " Each entry is the mean over 600 test episodes (6 sets × 100) per run; "
       "Avg. is the mean over the five tiers.", ""]
for _, sname in S.SOLS:
    out += [f"**{sname} planner**", "", "| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |", "|---|---|---|---|---|---|---|---|"]
    for task in S.ARMS:
        for m in METHODS:
            out.append(f"| {S.TITLE[task]} | {m} | " + " | ".join(S.r1(tier[(task, sname, m, t)]) for t in S.TIERS)
                       + f" | {S.r1(avg[(task, sname, m)])} |")
    out.append("")
out += ["**Table I2. Image-based world models against the state-based world model: success rate (%) averaged over the "
        "five budget tiers, with paired significance tests.** All models are evaluated on the same 600 test episodes "
        "with identical planner noise, so outcomes are paired per episode. Δ = method − state-based in percentage points "
        "(positive = the image-based method is better); the 95% CI is a paired t-interval over episodes; p is from an "
        "exact two-sided paired sign-flip permutation test. State-based, LeWM and SCALE average two training seeds; "
        "DINO-WM averages the two runs of the submission.", "",
        "| Task | Planner | Method | State-based | Method success | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|---|"]
for task in S.ARMS:
    for _, sname in S.SOLS:
        for m in ("LeWM", "SCALE", "DINO-WM"):
            d, lo, hi, p = stat[(task, sname, m)]
            out.append(f"| {S.TITLE[task]} | {sname} | {m} | {S.r1(avg[(task, sname, 'State-based')])} | {S.r1(avg[(task, sname, m)])} | "
                       f"{S.r1(d, True)} | [{S.sg(lo)}, {S.sg(hi)}] | {fp(p)} |")
out.append("")
open(os.path.join(M, "STATE_TABLES2_EN.md"), "w").write("\n".join(out))
print("\n".join(out[out.index(next(l for l in out if l.startswith('**Table I2'))):]))
print("exact p of the SCALE rows (must equal Table I):", [(S.TITLE[t], sn, S.fp(stat[(t, sn, 'SCALE')][3])) for t in S.ARMS for _, sn in S.SOLS][:4])
