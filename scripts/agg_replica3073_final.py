"""Final aggregation of the seed-3073 replication grid: 22 models x {cem,icem,mppi}
x 6 episode seeds, plus the seed-3072 originals for side-by-side comparison.

Reads the per-episode CSVs from GCS (success column, final_<task>_<cfg>_<sol>_s<seed>.csv;
the s3072 counterpart of cfg 'Xr73' is 'X'), computes per-seed SR, per-model means,
and paired Wilcoxon (n=6 episode seeds, arm - baseline within each seed grid).
gd is excluded (user); mppi ran at the stock T=0.5.

    usage: agg_replica3073_final.py <workdir>   (downloads CSVs into workdir)
    output: eval_results/replica3073_3solver_final.json
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

from scipy.stats import wilcoxon

BUCKET = "gs://prism-training-us/le-wm"
SEEDS = [101, 102, 103, 104, 105, 106]
SOLVERS = ["cem", "icem", "mppi"]

# task -> (gcs prefix per cfg, [(cfg3073, label)], baseline cfg3073)
ROSTER = {
    "Push-T": ("pusht", "final_eval", [
        ("c1r73", "LeWM 基线(像素,SIGReg 0.09)"),
        ("c3r73", "SCALE(sig+obj0.1)"),
        ("c5r73", "Aux(sig+aux0.3)"),
        ("dwr73", "DINO-WM"),
    ], "c1r73"),
    "Reacher": ("reacher", "final_eval", [
        ("r1r73", "LeWM 基线"),
        ("hqor73", "SCALE 半 q(最终选型)"),
        ("r2r73", "SCALE 全 q(obj0.15)"),
        ("r5r73", "Aux(aux0.4)"),
        ("dwr73", "DINO-WM"),
    ], "r1r73"),
    "Cube": ("cube", "final_eval", [
        ("k1r73", "LeWM 基线"),
        ("hqor73", "SCALE 半 q"),
        ("k2r73", "SCALE 全 q(obj_eff0.1)"),
        ("k4r73", "Aux(qhead_eff0.1)"),
        ("dwr73", "DINO-WM"),
    ], "k1r73"),
    "Two-Room": ("tworoom", "final_eval_tworoom", [
        ("t1r73", "LeWM 基线"),
        ("t2r73", "SCALE(obj0.1)"),
        ("t5r73", "Aux(qhead0.1)"),
        ("dwr73", "DINO-WM"),
    ], "t1r73"),
    "PointMaze": ("pointmaze", "final_eval_pointmaze", [
        ("p1r73", "LeWM 基线"),
        ("p2r73", "SCALE(obj0.1)"),
        ("p5r73", "Aux(qhead0.1)"),
        ("dwr73", "DINO-WM"),
    ], "p1r73"),
}
HALF_PREFIX = {"hqor73", "hq_obj"}  # half-data SCALE models live under final_eval_half

# s3072 counterparts whose cfg tag differs from just stripping 'r73'
S3072_NAME = {"c3r73": "c3_l01", "c5r73": "c5_l03",
              "r2r73": "r2_l015", "r5r73": "r5_l04", "hqor73": "hq_obj"}


def s3072_cfg(cfg73):
    return S3072_NAME.get(cfg73, cfg73[:-3])


def prefix_for(cfg, default):
    return "final_eval_half" if cfg in HALF_PREFIX else default


def fetch(workdir, task_env, prefix, cfg, sol):
    """Download the 6 seed CSVs for one cell; return {seed: sr_percent}."""
    out = {}
    for s in SEEDS:
        name = f"final_{task_env}_{cfg}_{sol}_s{s}.csv"
        local = workdir / name
        if not local.exists():
            r = subprocess.run(
                ["gcloud", "storage", "cp", f"{BUCKET}/{prefix}/{name}", str(local)],
                capture_output=True)
            if r.returncode != 0:
                continue
        with open(local) as f:
            succ = [int(r["success"]) for r in csv.DictReader(f)]
        if succ:
            out[s] = 100.0 * sum(succ) / len(succ)
    return out


def paired(a, b):
    """Wilcoxon on per-seed pairs (a-b); returns (mean_delta, p) or (None, None)."""
    ks = sorted(set(a) & set(b))
    if len(ks) < 6:
        return None, None
    d = [a[k] - b[k] for k in ks]
    if all(abs(x) < 1e-12 for x in d):
        return 0.0, 1.0
    stat = wilcoxon(d)
    return round(sum(d) / len(d), 2), round(float(stat.pvalue), 3)


def main():
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/replica_agg")
    workdir.mkdir(parents=True, exist_ok=True)
    report = {}
    for task, (env, prefix, arms, base73) in ROSTER.items():
        rows = []
        cells = {}  # (cfg, sol) -> {seed: sr}
        for cfg73, label in arms:
            cfg72 = s3072_cfg(cfg73)
            row = {"model": label, "cfg": cfg73}
            for sol in SOLVERS:
                for tag, cfg in (("s3073", cfg73), ("s3072", cfg72)):
                    pf = prefix_for(cfg, prefix)
                    srs = cells.setdefault((cfg, sol), fetch(workdir, env, pf, cfg, sol))
                    row[f"{sol}_{tag}"] = {
                        "mean": round(sum(srs.values()) / len(srs), 2) if srs else None,
                        "n_seeds": len(srs),
                        "per_seed": {str(k): round(v, 2) for k, v in sorted(srs.items())},
                    }
            rows.append(row)
        # paired deltas vs the task baseline, per solver, per seed grid
        for row in rows:
            if row["cfg"] == base73:
                continue
            for sol in SOLVERS:
                for tag, bcfg in (("s3073", base73), ("s3072", s3072_cfg(base73))):
                    acfg = row["cfg"] if tag == "s3073" else s3072_cfg(row["cfg"])
                    delta, p = paired(cells.get((acfg, sol), {}), cells.get((bcfg, sol), {}))
                    row[f"{sol}_{tag}"]["delta_vs_base"] = delta
                    row[f"{sol}_{tag}"]["p"] = p
        report[task] = rows
        done = sum(1 for r in rows for sol in SOLVERS if r[f"{sol}_s3073"]["n_seeds"] == 6)
        print(f"{task}: {done}/{len(rows)*3} s3073 cells complete", flush=True)
    out = Path(__file__).resolve().parent.parent / "eval_results/replica3073_3solver_final.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
