"""Regenerate eval_results/RESULTS_multiseed_sr.md from the final_eval CSVs.

The first version was written by hand, so adding episode seeds meant editing a dozen
numbers scattered across three tables and a summary — which is how cube stayed at 3
seeds after its 4th-6th had landed. Everything here is derived from the CSVs on disk:
seeds are discovered, not declared, so the document cannot drift from the data again.

Views, coarse to fine:
  per-seed SR by arm        one absolute SR per (seed, arm) — the sanity layer
  per-seed SR by solver     the same split by solver, so a solver-specific seed
                            oddity is visible
  per cell (seed-average)   every seed's episodes pooled per (solver, tier), which is
                            what the paired McNemar runs on
  by solver                 mean over a solver's 5 tiers, with sigma from the per-seed
                            aggregate restricted to that solver
  over all 20 cells         effect size, sign count, significance count, range
  per-seed contrasts        one contrast value per seed, giving SD / SE / sigma
  additivity                combo against obj + aux, on the seeds combo actually has

Combo joins the main tables only where it has the same seeds as the other arms.
Pooling requires every arm to be present for a seed to count, so an arm with fewer
seeds would silently shrink the whole task; when that happens combo is dropped from
the main tables and reported on its own instead.

    usage: make_multiseed_report.py [--out eval_results/RESULTS_multiseed_sr.md]
"""

import argparse
import csv
import glob
import re
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

SOLVERS = ["cem", "icem", "mppi", "gd"]
TIERS = ["T1", "T2", "T3", "T4", "T5"]
CORE = ["base", "obj", "aux"]

ARMS = {
    "pusht": {"base": ("c1", "lewm_c1_s3072"),
              "obj": ("c3_l01", "lewm_c3_sig_obj0.1_s3072"),
              "aux": ("c5_l03", "lewm_c5_qhead0.3_s3072"),
              "combo": ("c6_o01a03", "lewm_c6_o01a03_s3072")},
    "reacher": {"base": ("r1", "lewm_r1_reacher_s3072"),
                "obj": ("r2_l015", "lewm_r2_reacher_paep_l015_s3072"),
                "aux": ("r5_l04", "lewm_r5_qhead0.4_s3072")},
    "cube": {"base": ("k1", "lewm_k1_cube_s3072"),
             "obj": ("k2", "lewm_k2_cube_obj_eff0.1_s3072"),
             "aux": ("k4", "lewm_k4_cube_qhead_eff0.1_s3072"),
             "combo": ("k6", "lewm_k6_cube_combo_o0.1a0.1_s3072")},
}
QDIM = {"pusht": 6, "reacher": 4, "cube": 9}
QDESC = {"pusht": "6-d: pusher xy, block xy, cos/sin(block angle)",
         "reacher": "4-d: cos/sin of the two joint angles",
         "cube": "9-d: effector xyz, cos/sin(2·yaw), gripper opening, block xyz"}
DOSE = {"pusht": "obj λ=0.1 / aux w=0.3 / combo 0.1+0.3",
        "reacher": "obj λ=0.15 / aux w=0.4 (no combo arm)",
        "cube": "obj λ=0.1 / aux w=0.1 / combo 0.1+0.1"}


def mcnemar(x, y):
    b = sum(1 for e in x if x[e] == 1 and y[e] == 0)
    c = sum(1 for e in x if x[e] == 0 and y[e] == 1)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)


def load(task):
    """(config, solver, seed) -> {tier: {episode_id: success}}."""
    D = {}
    for f in glob.glob(f"eval_results/final/final_{task}_*_s1??.csv"):
        m = re.match(rf".*final_{task}_(.+)_({'|'.join(SOLVERS)})_s(\d+)\.csv$", f)
        if not m:
            continue
        cfg, slv, seed = m.groups()
        d = D.setdefault((cfg, slv, seed), {})
        for r in csv.DictReader(open(f)):
            d.setdefault(r["tier"], {})[int(r["episode_id"])] = int(r["success"])
    return D


def complete_seeds(D, cfgs):
    """Seeds where every given config has all 4 solvers x 5 tiers."""
    return [s for s in sorted({k[2] for k in D})
            if all((c, slv, s) in D and len(D[(c, slv, s)]) == len(TIERS)
                   for c in cfgs for slv in SOLVERS)]


def homogeneity_check(D, cfg_base, seeds, task, tol=8.0):
    """Refuse to pool episode seeds whose baseline difficulty does not match.

    Cube's s104-s106 were first drawn with gen_episodes_cube.py, whose non-trivial-goal
    filter is hardcoded, while s101-s103 came from the unfiltered gen_episodes.py.
    Roughly a third of unfiltered cube episodes are already solved at t=0 -- the goal is
    the cube's own position 25 steps later and the expert spends the early part of each
    episode merely reaching -- so the filter removed exactly the free wins and baseline
    SR fell 21-29 pp on every arm and every solver. Pooling averaged two difficulty
    regimes and cube stopped matching the earlier reproduction. tol is generous (8 pp)
    so ordinary noise passes while a protocol change cannot.
    """
    per = {}
    for sd in seeds:
        per[sd] = float(np.mean([100 * np.mean(list(D[(cfg_base, slv, sd)][t].values()))
                                 for slv in SOLVERS for t in TIERS]))
    med = float(np.median(list(per.values())))
    bad = {sd: v for sd, v in per.items() if abs(v - med) > tol}
    if bad:
        detail = "  ".join(f"s{sd}={v:.1f}" for sd, v in sorted(per.items()))
        off = ", ".join(f"s{k} ({v:.1f}, {v - med:+.1f})" for k, v in sorted(bad.items()))
        raise SystemExit(
            f"\nFATAL [{task}]: baseline SR differs across episode seeds by more than "
            f"{tol:.0f} pp, so they were probably not drawn the same way.\n"
            f"  per-seed baseline SR: {detail}\n  median {med:.1f}, offending: {off}\n"
            f"  gen_episodes.py is unfiltered, gen_episodes_cube.py filters on block\n"
            f"  displacement, and the two are not interchangeable.\n")
    return per


def pooled(D, cfg_of, seeds, labels, solvers=SOLVERS):
    """Per (solver,tier), every seed's episodes concatenated with unique ids."""
    out = {}
    for slv in solvers:
        for t in TIERS:
            cols = {L: {} for L in labels}
            for si, s in enumerate(seeds):
                for L in labels:
                    for e, v in D[(cfg_of[L], slv, s)][t].items():
                        cols[L][si * 1000 + e] = v
            eps = sorted(set.intersection(*[set(cols[L]) for L in labels]))
            out[(slv, t)] = {L: {e: cols[L][e] for e in eps} for L in labels}
    return out


def seed_sr(D, cfg_of, seed, label, solvers=SOLVERS):
    """Absolute SR for one (seed, arm), averaged over the given solvers' tiers."""
    return float(np.mean([100 * np.mean(list(D[(cfg_of[label], slv, seed)][t].values()))
                          for slv in solvers for t in TIERS]))


def per_seed(D, cfg_of, seeds, a, b, solvers=SOLVERS):
    """One contrast value per seed: mean of (a-b) over the given solvers' tiers."""
    vals = []
    for s in seeds:
        cells = []
        for slv in solvers:
            for t in TIERS:
                ea, eb = D[(cfg_of[a], slv, s)][t], D[(cfg_of[b], slv, s)][t]
                eps = sorted(set(ea) & set(eb))
                cells.append(100 * (np.mean([ea[e] for e in eps])
                                    - np.mean([eb[e] for e in eps])))
        vals.append(np.mean(cells))
    return np.array(vals)


def stats(v):
    if len(v) < 2:
        return (v.mean() if len(v) else np.nan), np.nan, np.nan, np.nan
    m, sd = v.mean(), v.std(ddof=1)
    se = sd / np.sqrt(len(v))
    return m, sd, se, (abs(m) / se if se > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_results/RESULTS_multiseed_sr.md")
    args = ap.parse_args()

    out = []
    W = out.append
    W("# Multi-seed success rates — baseline vs L_obj vs aux q-head vs combo\n")
    W(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by "
      "`scripts/make_multiseed_report.py` from "
      "`eval_results/final/final_<task>_<config>_<solver>_s<seed>.csv`. "
      "Seeds are discovered from the files present, not hard-coded.\n")

    W("## Protocol\n")
    W("| | |")
    W("|---|---|")
    W("| Training | 10 epochs, 1 GPU, seed **3072**, `weights_epoch_10.pt` — a SINGLE "
      "training seed, see Limitations |")
    W("| Planner | HORIZON=5, RECEDING_HORIZON=5, ACTION_BLOCK=5, EVAL_BUDGET=50 env "
      "steps, GOAL_OFFSET=25 |")
    W("| Tiers | sampling: T1 300/30, T2 150/15, T3 50/10, T4 20/5, T5 10/3 "
      "(candidates/iterations), elites = max(round(0.1·cand), 2) |")
    W("| | gradient: T1 100/90, T2 75/30, T3 50/10, T4 20/5, T5 10/3, AdamW lr=0.1 |")
    W("| | rollout evaluations per replan are matched across families: "
      "9000 / 2250 / 500 / 100 / 30 |")
    W("| Planner noise | `cem_seed = crc32(\"<episode_id>\\|<tier>\")` — identical across "
      "configs, so every comparison is paired |")
    W("| Episodes | 100 per seed, drawn without replacement by `gen_episodes.py` "
      "(unfiltered); sets pre-registered, every drawn seed reported |")
    W("| Statistics | exact paired McNemar per cell; per-seed aggregate for SD/SE/σ |\n")

    W("### Arms\n")
    W("| task | q_dim | q | dose | base | obj | aux | combo |")
    W("|---|---|---|---|---|---|---|---|")
    for task, arms in ARMS.items():
        cb = f"`{arms['combo'][1]}`" if "combo" in arms else "—"
        W(f"| {task} | {QDIM[task]} | {QDESC[task]} | {DOSE[task]} | "
          f"`{arms['base'][1]}` | `{arms['obj'][1]}` | `{arms['aux'][1]}` | {cb} |")
    W("")

    cross, cross_solver, cross_seed = [], [], []
    for task, arms in ARMS.items():
        D = load(task)
        cfg_of = {L: arms[L][0] for L in arms}
        core_seeds = complete_seeds(D, [cfg_of[L] for L in CORE])
        labels = list(CORE)
        note = ""
        if "combo" in arms:
            all_seeds = complete_seeds(D, [cfg_of[L] for L in arms])
            if len(all_seeds) == len(core_seeds):
                labels = CORE + ["combo"]
            else:
                note = (f"\nCombo is excluded from this task's main tables: it has "
                        f"{len(all_seeds)} complete seeds against the other arms' "
                        f"{len(core_seeds)}, and pooling needs every arm present per "
                        f"seed, so including it would shrink the whole task. It is "
                        f"reported under *Additivity* on its own seeds.\n")
        seeds = core_seeds
        cmps = [("obj", "base"), ("aux", "base")]
        if "combo" in labels:
            cmps.append(("combo", "base"))
        cmps.append(("obj", "aux"))
        per_base = homogeneity_check(D, cfg_of["base"], seeds, task)
        P = pooled(D, cfg_of, seeds, labels)
        n_eps = len(next(iter(P.values()))["base"])

        W(f"\n## {task}  (q_dim={QDIM[task]}, {len(seeds)} episode seeds: "
          f"{', '.join('s' + s for s in seeds)})\n")
        if note:
            W(note)

        # ---------------------------------------------------------- per-seed SR
        W("### Per-seed SR by arm\n")
        W("Absolute SR, each value averaged over that seed's 20 solver×tier cells "
          "(100 episodes each). This is the layer that makes a seed drawn under a "
          "different protocol visible instead of averaged away.\n")
        W("| seed | " + " | ".join(labels) + " | "
          + " | ".join(f"{a}−{b}" for a, b in cmps) + " |")
        W("|" + "---|" * (1 + len(labels) + len(cmps)))
        srs = {L: [] for L in labels}
        for sd in seeds:
            vals = {L: seed_sr(D, cfg_of, sd, L) for L in labels}
            for L in labels:
                srs[L].append(vals[L])
            W(f"| s{sd} | " + " | ".join(f"{vals[L]:.1f}" for L in labels) + " | "
              + " | ".join(f"{vals[a] - vals[b]:+.2f}" for a, b in cmps) + " |")
        W("| **mean** | " + " | ".join(f"**{np.mean(srs[L]):.1f}**" for L in labels)
          + " | " + " | ".join(f"**{np.mean(srs[a]) - np.mean(srs[b]):+.2f}**"
                               for a, b in cmps) + " |")
        W("| SD | " + " | ".join(f"{np.std(srs[L], ddof=1):.1f}" for L in labels)
          + " | " + " | ".join("" for _ in cmps) + " |")
        for L in labels:
            cross_seed.append((task, L, np.mean(srs[L]), np.std(srs[L], ddof=1)))

        W("\n### Per-seed SR by solver\n")
        W("The same absolute SR split by solver — each value is the mean over that "
          "solver's 5 tiers for one seed.\n")
        W("| solver | seed | " + " | ".join(labels) + " |")
        W("|" + "---|" * (2 + len(labels)))
        for slv in SOLVERS:
            for sd in seeds:
                W(f"| {slv} | s{sd} | "
                  + " | ".join(f"{seed_sr(D, cfg_of, sd, L, [slv]):.1f}" for L in labels)
                  + " |")
            W(f"| **{slv} mean** | — | "
              + " | ".join(f"**{np.mean([seed_sr(D, cfg_of, sd, L, [slv]) for sd in seeds]):.1f}**"
                           for L in labels) + " |")

        # ------------------------------------------------------- pooled per cell
        W("\n### Per cell (seed-average)\n")
        W(f"Every seed's episodes pooled — {n_eps} per cell. `\\*` = paired exact "
          "McNemar p<0.05. Pooling and averaging the per-seed SRs coincide here "
          "because every seed contributes the same 100 episodes.\n")
        W("| solver | tier | n | " + " | ".join(labels) + " | "
          + " | ".join(f"{a}−{b}" for a, b in cmps) + " |")
        W("|" + "---|" * (3 + len(labels) + len(cmps)))
        for slv in SOLVERS:
            for t in TIERS:
                C = P[(slv, t)]
                sr = {k: 100 * np.mean(list(v.values())) for k, v in C.items()}
                cells = []
                for a, b in cmps:
                    star = "\\*" if mcnemar(C[a], C[b]) < 0.05 else ""
                    cells.append(f"{sr[a] - sr[b]:+.1f}{star}")
                W(f"| {slv} | {t} | {len(C['base'])} | "
                  + " | ".join(f"{sr[k]:.1f}" for k in labels) + " | "
                  + " | ".join(cells) + " |")

        # ---------------------------------------------------------- by solver
        W("\n### By solver\n")
        W("SR is the mean over that solver's 5 tiers. σ comes from the per-seed "
          "aggregate restricted to the same solver, so it is episode-sampling variance "
          "within that solver.\n")
        W("| solver | " + " | ".join(labels) + " | "
          + " | ".join(f"{a}−{b} | σ" for a, b in cmps) + " | obj−base sig |")
        W("|" + "---|" * (1 + len(labels) + 2 * len(cmps) + 1))
        for slv in SOLVERS:
            srow = {L: np.mean([100 * np.mean(list(P[(slv, t)][L].values())) for t in TIERS])
                    for L in labels}
            cells = []
            for a, b in cmps:
                m, _, _, sg = stats(per_seed(D, cfg_of, seeds, a, b, [slv]))
                cells.append(f"**{m:+.2f}** | {sg:.1f}")
                cross_solver.append((task, slv, a, b, m, sg))
            sig = sum(1 for t in TIERS
                      if mcnemar(P[(slv, t)]["obj"], P[(slv, t)]["base"]) < 0.05)
            W(f"| {slv} | " + " | ".join(f"{srow[L]:.1f}" for L in labels) + " | "
              + " | ".join(cells) + f" | {sig}/5 |")

        # ------------------------------------------------- over all 20 cells
        W("\n### Over all 20 solver×tier cells\n")
        W("| contrast | mean | cells >0 | cells sig | range |")
        W("|---|---|---|---|---|")
        for a, b in cmps:
            v, sig = [], 0
            for C in P.values():
                v.append(100 * (np.mean(list(C[a].values())) - np.mean(list(C[b].values()))))
                if mcnemar(C[a], C[b]) < 0.05:
                    sig += 1
            W(f"| {a}−{b} | **{np.mean(v):+.2f}** | {sum(1 for x in v if x > 0)}/{len(v)} "
              f"| {sig}/{len(v)} | {min(v):+.1f} … {max(v):+.1f} |")

        # -------------------------------------------------- per-seed contrasts
        W("\n### Per-seed contrasts\n")
        W("Each value = mean over that seed's 20 solver×tier cells. σ = |mean| / SE, "
          "and it quantifies **episode-sampling variance only**.\n")
        W("| contrast | " + " | ".join("s" + s for s in seeds) + " | mean | SD | SE | σ |")
        W("|" + "---|" * (len(seeds) + 5))
        for a, b in cmps:
            v = per_seed(D, cfg_of, seeds, a, b)
            m, sd, se, sg = stats(v)
            W(f"| {a}−{b} | " + " | ".join(f"{x:+.2f}" for x in v)
              + f" | **{m:+.2f}** | {sd:.2f} | {se:.2f} | {sg:.1f} |")
            cross.append((task, a, b, m, sg, len(seeds)))

        # ------------------------------------------------------- additivity
        if "combo" in arms:
            cs = complete_seeds(D, [cfg_of[L] for L in arms])
            ob = per_seed(D, cfg_of, cs, "obj", "base")
            ab = per_seed(D, cfg_of, cs, "aux", "base")
            cb = per_seed(D, cfg_of, cs, "combo", "base")
            Pc = pooled(D, cfg_of, cs, CORE + ["combo"])
            best_gain, n_sig_best, n_pos_best = [], 0, 0
            for C in Pc.values():
                sr = {k: 100 * np.mean(list(v.values())) for k, v in C.items()}
                best = "obj" if sr["obj"] >= sr["aux"] else "aux"
                d = sr["combo"] - sr[best]
                best_gain.append(d)
                if d > 0:
                    n_pos_best += 1
                if mcnemar(C["combo"], C[best]) < 0.05:
                    n_sig_best += 1
            W(f"\n### Additivity  ({len(cs)} seeds: {', '.join('s' + s for s in cs)})\n")
            W("| quantity | value |")
            W("|---|---|")
            W(f"| obj−base | {ob.mean():+.2f} |")
            W(f"| aux−base | {ab.mean():+.2f} |")
            W(f"| sum, if the two losses stacked | {ob.mean() + ab.mean():+.2f} |")
            W(f"| combo−base, measured | **{cb.mean():+.2f}** |")
            W(f"| shortfall | **{cb.mean() - (ob.mean() + ab.mean()):+.2f}** pp |")
            W(f"| fraction of the expected gain realised | "
              f"**{100 * cb.mean() / (ob.mean() + ab.mean()):.0f}%** |")
            W(f"| combo − whichever single loss is better, per cell | "
              f"{np.mean(best_gain):+.2f} ({n_pos_best}/{len(best_gain)} positive, "
              f"{n_sig_best}/{len(best_gain)} significant) |")
            W("\nCombo lands near the better single loss rather than near their sum. "
              "Since obj and aux both act on the same channel — candidate ranking, with "
              "obj roughly twice the effect (see the P4/P5 diagnostics) — they compete "
              "for the same gain, so the absence of stacking is what that mechanism "
              "predicts.\n")

    # ------------------------------------------------------------ cross-task
    W("\n## Cross-task summary\n")
    W("| task | q_dim | seeds | obj−base | σ | aux−base | σ | obj−aux | σ |")
    W("|---|---|---|---|---|---|---|---|---|")
    for task in ARMS:
        row = {(a, b): (m, s) for t_, a, b, m, s, _ in cross if t_ == task}
        ns = next(n for t_, _, _, _, _, n in cross if t_ == task)
        W(f"| {task} | {QDIM[task]} | {ns} | "
          + " | ".join(f"**{row[c][0]:+.2f}** | {row[c][1]:.1f}"
                       for c in [("obj", "base"), ("aux", "base"), ("obj", "aux")]) + " |")
    W("\nσ = |mean| / SE of the per-seed aggregate; **episode-sampling variance only**.\n")

    W("### Cross-task absolute SR, by arm\n")
    W("Mean over all seeds and all 20 cells, with the seed-to-seed SD.\n")
    W("| task | base | obj | aux | combo |")
    W("|---|---|---|---|---|")
    for task in ARMS:
        d = {L: (m, s) for t_, L, m, s in cross_seed if t_ == task}
        W(f"| {task} | "
          + " | ".join(f"{d[L][0]:.1f} ± {d[L][1]:.1f}" if L in d else "—"
                       for L in ["base", "obj", "aux", "combo"]) + " |")
    W("")

    W("### Cross-task, by solver\n")
    W("The three core contrasts split by solver, mean over that solver's 5 tiers, "
      "σ in brackets.\n")
    W("| solver | " + " | ".join(
        f"{t} {a}−{b}" for t in ARMS
        for a, b in [("obj", "base"), ("aux", "base"), ("obj", "aux")]) + " |")
    W("|" + "---|" * (1 + 3 * len(ARMS)))
    for slv in SOLVERS:
        cells = []
        for task in ARMS:
            for a, b in [("obj", "base"), ("aux", "base"), ("obj", "aux")]:
                hit = [(m, s) for t_, sl, a_, b_, m, s in cross_solver
                       if t_ == task and sl == slv and a_ == a and b_ == b]
                cells.append(f"{hit[0][0]:+.2f} ({hit[0][1]:.1f})" if hit else "—")
        W(f"| {slv} | " + " | ".join(cells) + " |")
    W("")

    W("## Limitations\n")
    W("- **One training seed (3072) per arm.** Episode-sampling variance is measured; "
      "training variance is not. The LeWM paper reports ± of median 2.80 (max 7.5) "
      "across its 3 training seeds — larger than the sub-1 pp obj-vs-aux differences "
      "here. Resolving obj−aux to 2σ at the observed SD would need on the order of 100 "
      "training seeds.")
    W("- **Reacher has no combo arm** — it was never trained.")
    W("- **Cube's SR data predates the EGL render fix but is unaffected**: re-running "
      "the k1 baseline under the fixed renderer moved 10 cells by at most ±1.0 pp with "
      "295–300/300 episode-level agreement. Reacher moved 6–14 pp and was fully re-run. "
      "Push-T renders on the CPU via box2d and was never involved (200/200 "
      "episode-exact reproduction). The render-fidelity gate itself was also wrong for "
      "a while — it read `world.infos[\"pixels\"]`, a snapshot refreshed only on "
      "reset/step, so straight after `set_state` it returned the post-reset frame and "
      "reported cube at MAE 9.04; rendering explicitly gives 0.17, and reacher 0.0001.")
    W("- **q dimensionality does not order obj−aux across tasks.** Within cube, "
      "however, going from 9-d to 21-d q (adding 5 live arm joints) costs the aux arm "
      "−2.2 / −2.7 pp on two independent episode sets while leaving the L_obj arm "
      "unchanged (−0.2 / +0.5) — consistent in direction on both sets, but only "
      "reaching p=0.058, one training seed each.")
    W("- **mppi's budget ladder is not monotone** on Reacher and Cube (T5 sometimes "
      "beats T1), so it is not searching effectively there and its contrasts carry "
      "little information regardless of arm.")

    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"wrote {args.out} ({len(out)} lines)")


if __name__ == "__main__":
    main()
