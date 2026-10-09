"""Episode-aware clip dataset over the ContactWorld Zarr store — proposal §5.2/§6.

Clips: K+1 observations and exactly K actions (terminal action dropped), never
crossing episode boundaries; frameskip=1; both cameras sampled at identical rows;
256->224 resize, float [0,1], identical transform for both views.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr


class MultiViewClipDataset(Dataset):
    def __init__(self, data_root, splits_path, split="train", clip_len=3,
                 image_size=224, episodes_override=None):
        self.root = zarr.open(data_root, mode="r")
        ends = np.asarray(self.root["meta/episode_ends"][:]).astype(int)
        starts = np.concatenate([[0], ends[:-1]])
        if episodes_override is not None:
            eps = list(episodes_override)
        else:
            eps = json.load(open(splits_path))[split]
        self.clip_len = clip_len          # K transitions -> K+1 obs
        self.image_size = image_size
        self.index = []                   # (t0) start row of each clip
        for ep in eps:
            s, e = int(starts[ep]), int(ends[ep])
            for t0 in range(s, e - clip_len):   # need K+1 obs inside [s, e)
                self.index.append(t0)
        self.front = self.root["data/front"]
        self.wrist = self.root["data/wrist"]
        self.action = self.root["data/action"]

    def __len__(self):
        return len(self.index)

    def _img(self, arr, t0, k):
        x = torch.from_numpy(np.asarray(arr[t0:t0 + k]))          # (k,256,256,3) [0,1]
        x = x.permute(0, 3, 1, 2).float()
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False, antialias=True)
        return x

    def __getitem__(self, i):
        t0 = self.index[i]
        k = self.clip_len + 1
        front = self._img(self.front, t0, k)
        wrist = self._img(self.wrist, t0, k)
        acts = torch.from_numpy(np.asarray(self.action[t0:t0 + self.clip_len])).float()
        return {"front": front, "wrist": wrist, "action": acts, "t0": t0}
