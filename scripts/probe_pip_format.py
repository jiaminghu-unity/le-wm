"""PiP probe: obs formats of cube/scene lance + foreground-motion heatmap of cube
frames (to pick a never-occupied corner) + corner-candidate composite samples.
Reads lance REMOTELY from gs:// (a few fragments only, no staging)."""
import io, json
import numpy as np
import lance

B = "gs://prism-training-us/le-wm/datasets/ogbench"
out = {}
def peek(name):
    ds = lance.dataset(f"{B}/{name}")
    sch = {f.name: str(f.type) for f in ds.schema}
    t = ds.take(list(range(3)))
    row = {}
    for c in t.column_names:
        a = t.column(c).to_numpy(zero_copy_only=False)
        v = np.asarray(a[0]) if a.dtype == object else a
        row[c] = (str(v.dtype), list(np.asarray(v).shape))
    return sch, row, ds.count_rows()

for name in ["cube_single_expert.lance", "scene_play.lance"]:
    sch, row, n = peek(name)
    out[name] = {"rows": n, "cols": row}
    print(name, n, json.dumps(row), flush=True)

# motion heatmap over sampled cube frames (pixels = encoded image bytes)
from PIL import Image
def dec(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.int16)
ds = lance.dataset(f"{B}/cube_single_expert.lance")
n = ds.count_rows()
idx = np.linspace(0, n - 2, 400).astype(int)
heat = None
for i in idx:
    t = ds.take([int(i), int(i) + 1], columns=["pixels"])
    raw = t.column("pixels").to_pylist()
    A, Bm = dec(raw[0]), dec(raw[1])
    d = np.abs(A - Bm).mean(-1)
    heat = d if heat is None else np.maximum(heat, d)
print("cube img shape", A.shape, flush=True)
np.save("/tmp/pip_heat.npy", heat)
# corner occupancy (fraction of heatmap mass), 64px corners
s = heat.shape[0]; k = max(16, s * 64 // 224)
corners = {"TL": heat[:k, :k], "TR": heat[:k, -k:], "BL": heat[-k:, :k], "BR": heat[-k:, -k:]}
print("side", s, "patch", k, {c: round(float(v.max()), 1) for c, v in corners.items()}, flush=True)
try:
    Image.fromarray((heat / heat.max() * 255).astype(np.uint8)).save("/tmp/pip_heat.png")
    Image.fromarray(A.astype(np.uint8)).save("/tmp/pip_cube_sample.png")
    ds2 = lance.dataset(f"{B}/scene_play.lance")
    t2 = ds2.take([100], columns=["pixels"])
    f2 = dec(t2.column("pixels").to_pylist()[0])
    print("scene img shape", f2.shape, flush=True)
    Image.fromarray(f2.astype(np.uint8)).save("/tmp/pip_scene_sample.png")
except Exception as e:
    print("PNG save failed:", e, flush=True)
print("PROBE DONE", flush=True)
