import sys
sys.path.insert(0, "scripts")
import numpy as np
from pathlib import Path
import hdf5plugin  # noqa
import stable_worldmodel as swm

ds = swm.data.HDF5Dataset("reacher", keys_to_cache=["action"],
                          cache_dir=Path(swm.data.utils.get_cache_dir()))
ep = np.asarray(ds.get_col_data("episode_idx" if "episode_idx" in ds.column_names else "ep_idx")).reshape(-1)
ids = np.unique(ep)[:6]
for e in ids:
    row = int(np.nonzero(ep == e)[0][0])
    px = np.asarray(ds.get_row_data([row])["pixels"][0])
    # 四角各取 8x8 均值 = 地板/背景色采样
    c = [px[:8, :8].mean((0, 1)).round(0), px[:8, -8:].mean((0, 1)).round(0),
         px[-8:, :8].mean((0, 1)).round(0), px[-8:, -8:].mean((0, 1)).round(0)]
    print(f"ep{int(e)}: corners {[list(map(int,x)) for x in c]}", flush=True)
