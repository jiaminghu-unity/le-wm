"""Compare every cell of the paper's Tables 13-17 (mean±SD, 4 methods x 2 solvers x 6 columns x 5 tasks)
with a recomputation from the eval CSVs in the repo, using the seed pairs 3072+3073 (PointMaze LeWM/SCALE/Aux:
3072+3074; DINO-WM 3072+3073)."""
import contextlib, io, sys
sys.path.insert(0, "scripts")
with contextlib.redirect_stdout(io.StringIO()):
    import rebuttal_table2_4seeds as T
P = {
"pusht": """cem LeWM 92.3±2.3 89.3±3.2 80.5±4.1 57.8±5.6 33.8±3.6 70.7±2.7
cem SCALE 94.8±3.2 92.1±2.6 84.2±3.6 61.6±6.9 32.8±5.5 73.1±2.7
cem Aux 94.5±2.0 92.6±2.4 84.4±4.1 65.1±6.9 34.9±6.4 74.3±3.4
cem DINO-WM 74.8±4.6 72.1±4.5 65.2±4.9 46.1±4.6 24.3±6.1 56.5±4.0
icem LeWM 86.0±2.4 79.8±2.1 73.4±5.9 55.2±4.8 25.1±5.1 63.9±3.0
icem SCALE 89.1±2.0 86.3±2.8 79.7±3.6 61.3±6.2 25.2±5.5 68.3±2.8
icem Aux 89.9±2.0 86.2±2.3 79.6±3.4 62.2±5.3 27.6±5.2 69.1±2.6
icem DINO-WM 70.0±4.5 65.3±4.5 58.8±5.1 41.5±8.8 23.1±3.7 51.7±4.3""",
"reacher": """cem LeWM 82.3±1.8 79.4±3.6 73.3±4.8 66.2±4.1 55.8±5.6 71.4±2.3
cem SCALE 82.2±3.8 79.1±5.1 74.5±4.4 65.2±6.0 57.4±5.5 71.7±3.1
cem Aux 81.6±4.6 78.8±4.2 74.9±3.2 65.8±5.6 55.9±6.8 71.4±3.3
cem DINO-WM 77.2±2.8 74.8±5.4 70.8±5.9 61.8±3.4 50.2±7.3 67.0±3.0
icem LeWM 84.9±2.6 84.2±3.4 83.8±3.4 79.0±6.1 62.2±3.6 78.8±1.7
icem SCALE 86.7±3.6 87.0±3.1 87.3±3.2 81.7±3.0 64.4±3.0 81.4±2.0
icem Aux 85.0±4.4 86.1±3.1 85.8±3.5 80.1±3.7 62.9±5.1 80.0±2.2
icem DINO-WM 78.7±4.9 78.2±4.8 80.8±3.5 76.0±4.0 53.0±5.5 73.3±3.1""",
"cube": """cem LeWM 67.6±4.6 65.8±3.8 60.8±4.6 56.8±3.5 53.8±4.0 60.9±3.4
cem SCALE 73.9±3.9 71.1±4.1 64.7±3.8 55.9±4.7 54.8±4.6 64.1±3.6
cem Aux 71.6±4.7 70.2±4.5 62.8±5.2 56.6±4.6 55.2±4.3 63.3±3.6
cem DINO-WM 73.4±3.1 66.9±2.3 58.4±4.4 51.6±4.5 53.9±3.8 60.9±2.9
icem LeWM 68.9±4.9 67.1±3.6 68.7±4.1 61.9±3.9 54.9±5.7 64.3±4.0
icem SCALE 76.8±3.7 76.6±4.0 75.2±3.0 65.8±3.3 59.8±3.9 70.8±3.1
icem Aux 73.8±3.6 73.0±4.7 72.5±3.8 64.8±3.4 57.8±3.8 68.4±3.5
icem DINO-WM 69.9±2.5 67.4±1.6 66.0±1.7 57.7±3.8 51.4±3.6 62.5±1.3""",
"tworoom": """cem LeWM 87.8±3.0 87.1±2.8 83.0±3.3 71.8±2.7 57.8±3.9 77.5±1.5
cem SCALE 99.7±0.7 99.3±0.8 94.0±2.0 83.5±2.6 69.7±4.2 89.2±0.8
cem Aux 88.3±2.7 87.7±2.9 82.4±3.1 70.6±3.1 59.7±2.5 77.7±1.3
cem DINO-WM 100.0±0.0 99.8±0.4 98.2±1.2 94.1±2.2 85.8±3.7 95.6±1.0
icem LeWM 91.1±1.6 92.9±1.7 88.5±2.0 77.3±2.2 62.4±2.8 82.5±1.2
icem SCALE 99.9±0.3 99.8±0.6 97.7±1.2 90.2±3.0 76.2±2.7 92.7±1.0
icem Aux 92.3±2.1 93.2±1.7 89.2±2.0 78.7±2.5 62.9±3.6 83.3±1.1
icem DINO-WM 100.0±0.0 100.0±0.0 99.8±0.5 98.6±0.8 92.5±2.1 98.2±0.5""",
"pointmaze": """cem LeWM 85.7±4.9 86.0±4.2 86.9±3.9 83.5±3.5 75.3±5.8 83.5±3.3
cem SCALE 87.6±3.8 87.4±3.9 86.8±3.8 84.8±3.9 78.1±3.4 84.9±2.7
cem Aux 85.0±4.6 83.9±4.7 82.7±4.8 79.8±4.7 78.1±5.6 81.9±3.9
cem DINO-WM 98.2±1.2 98.3±1.4 97.2±1.3 92.4±3.0 87.2±2.9 94.7±1.4
icem LeWM 84.0±4.1 83.6±4.3 82.4±3.1 81.2±4.6 73.2±4.8 80.9±3.5
icem SCALE 87.6±3.3 85.8±4.5 84.6±3.2 81.9±4.7 77.0±4.4 83.4±2.5
icem Aux 80.8±5.1 79.5±5.1 78.7±5.0 79.4±3.7 75.0±5.8 78.7±4.3
icem DINO-WM 93.4±2.6 92.2±2.6 90.8±3.4 90.3±2.6 85.2±2.6 90.4±2.1"""}
n = ok = 0; near = []; bad = []
for task, txt in P.items():
    for line in txt.splitlines():
        sol, arm, *cells = line.split()
        seeds = T.DINO_SEEDS if arm == "DINO-WM" else T.PAPER_SEEDS[task]
        for col, c in zip(T.COLS, cells):
            pm, psd = map(float, c.split("±"))
            m, sd1, sd0, k = T.cell(task, arm, seeds, sol, col)
            m = float(m); n += 1
            if f"{m:.1f}" == f"{pm:.1f}" and f"{sd1:.1f}" == f"{psd:.1f}":
                ok += 1
            elif abs(m - pm) <= 0.0501 and abs(sd1 - psd) <= 0.0501:
                near.append((task, sol, arm, col, c, f"{m:.3f}±{sd1:.3f}"))
            else:
                bad.append((task, sol, arm, col, c, f"{m:.3f}±{sd1:.3f}"))
print(f"cells compared: {n}; identical as printed: {ok}; differ only by rounding at a .x5 boundary: {len(near)}; real mismatches: {len(bad)}")
for x in near: print("  rounding:", x)
for x in bad: print("  MISMATCH:", x)
print("gains (SCALE - LeWM, Overall), exact vs the paper's Sec. 6.2 text:")
TXT = {"icem": [4.4, 2.6, 6.5, 10.2, 2.5], "cem": [2.4, 0.3, 3.2, 11.7, 1.4]}
for sol in ("icem", "cem"):
    for task, pg in zip(P, TXT[sol]):
        a = float(T.cell(task, "SCALE", T.PAPER_SEEDS[task], sol, "Overall")[0]); b = float(T.cell(task, "LeWM", T.PAPER_SEEDS[task], sol, "Overall")[0])
        print(f"  {sol:4s} {task:9s} LeWM {b:.3f}  SCALE {a:.3f}  exact gain {a-b:+.3f}  | paper text {pg:+.1f}  | from rounded table values {round(a,1)-round(b,1):+.1f}")
