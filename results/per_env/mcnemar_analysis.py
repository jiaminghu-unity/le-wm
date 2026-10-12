"""D0 — paired per-env analysis: determinism check, McNemar tests, paired CIs."""
import json
import numpy as np

ARMS = ["b1", "ours", "b3", "b3soft", "ourssoft"]
OFFS = [12, 24, 36, 48]

succ = {}
for v in ARMS:
    cols = []
    for off in OFFS:
        D = np.load(f"{v}_off{off}.npz")
        cols.append(D["success"][-1].astype(int))      # final-step official flags
    succ[v] = np.concatenate(cols)                      # (400,)

# determinism check vs earlier runs
ref = {"b1": [53, 26, 3, 1], "ours": [47, 14, 2, 0], "b3": [30, 7, 5, 0],
       "b3soft": [46, 17, 3, 0], "ourssoft": [47, 18, 3, 0]}
print("== determinism (pe2 vs earlier) ==")
for v in ARMS:
    got = [int(np.load(f"{v}_off{o}.npz")["success"][-1].sum()) for o in OFFS]
    print(f"  {v:9s} {got} vs {ref[v]} {'SAME' if got == ref[v] else 'DIFF'}")

def mcnemar(a, b):
    n01 = int(((a == 0) & (b == 1)).sum())   # b wins
    n10 = int(((a == 1) & (b == 0)).sum())   # a wins
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    from math import comb
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / 2 ** n
    return n10, n01, min(p, 1.0)

print("\n== paired McNemar (pooled 400 envs, identical init/goal) ==")
pairs = [("b1", "ours"), ("b1", "ourssoft"), ("b1", "b3soft"),
         ("ours", "b3"), ("ourssoft", "b3soft"), ("ours", "ourssoft"), ("b3soft", "b3")]
rng = np.random.RandomState(0)
for a, b in pairs:
    A, B = succ[a], succ[b]
    d = A.mean() - B.mean()
    n10, n01, p = mcnemar(A, B)
    boots = [(A[i].mean() - B[i].mean()) for i in
             (rng.randint(0, 400, (2000, 400)))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {a:9s} vs {b:9s}: {A.sum():3d} vs {B.sum():3d} | diff {d*100:+5.1f}pp "
          f"[{lo*100:+.1f},{hi*100:+.1f}] | only-{a} {n10:2d} only-{b} {n01:2d} | McNemar p={p:.4f}")
