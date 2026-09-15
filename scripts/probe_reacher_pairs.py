import sys
sys.path.insert(0, "scripts")
import numpy as np
from PIL import Image
from check_render_fidelity import measure

maes, missing, extra, pairs = measure("reacher", 2, want_pairs=True)
print("maes:", maes, flush=True)
for i, (st, r) in enumerate(pairs[:2]):
    Image.fromarray(st.astype(np.uint8)).save(f"/tmp/rpair_{i}_stored.png")
    Image.fromarray(r.astype(np.uint8)).save(f"/tmp/rpair_{i}_render.png")
    d = np.abs(st - r).astype(np.uint8)
    Image.fromarray((d * (255 // max(d.max(), 1))).astype(np.uint8)).save(f"/tmp/rpair_{i}_diff.png")
    print(f"pair{i}: diff max {np.abs(st-r).max()}, mean {np.abs(st-r).mean():.2f}", flush=True)
