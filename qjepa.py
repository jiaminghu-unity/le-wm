"""QJEPA: LeWM with the pixel encoder replaced by an MLP over the physical state q.

Everything else -- projector, ARPredictor, action encoder, SIGReg training flow,
planning interface (rollout / get_cost / criterion) -- is inherited from JEPA
unchanged, so this arm isolates exactly one variable: the INPUT modality.

Two encode paths, one metric convention:
  * training: batches carry `q` pre-normalized by get_q_normalizer (the same
    persisted per-variant stats every diagnostic uses);
  * planning: env infos carry raw `state`; q is built with build_q_raw and
    normalized with the SAME stats, stored as buffers (q_mean/q_std) so the
    checkpoint is self-contained. At load_pretrained time __init__ re-runs and the
    stats file may be absent -- the buffers then fall back to identity and are
    immediately overwritten by load_state_dict, which is why loading never needs
    the json. `goal_state` reaches encode as `state` via JEPA.get_cost's goal_*
    remapping; nothing on the planning side needs to change.

Eval-side caveat handled in scripts/budget_sweep_qinput.py: the stock eval
StandardScaler-processes `state`; that wrapper drops the state scalers so encode
receives RAW state, matching this file's normalization contract.
"""

import json
import os
from pathlib import Path

import torch
from einops import rearrange

import stable_worldmodel as swm

from jepa import JEPA
from utils import build_q_raw


class QJEPA(JEPA):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        q_dim=6,
        q_stats_file=None,
    ):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.register_buffer("q_mean", torch.zeros(q_dim))
        self.register_buffer("q_std", torch.ones(q_dim))
        if q_stats_file:
            cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
            path = Path(
                swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets"),
                q_stats_file,
            )
            if path.exists():
                stats = json.loads(path.read_text())
                self.q_mean.copy_(torch.tensor(stats["mean"], dtype=torch.float32))
                self.q_std.copy_(torch.tensor(stats["std"], dtype=torch.float32))
                print(f"[qjepa] q stats loaded from {path}", flush=True)
            else:
                # load_pretrained re-instantiates on nodes without the dataset;
                # load_state_dict overwrites these defaults right after.
                print(f"[qjepa] q stats file absent ({path}); buffers await state_dict", flush=True)

    def encode(self, info):
        if "q" in info:                       # training batches: pre-normalized q
            x = info["q"].float()
        else:                                 # planning: raw env state
            x = build_q_raw(info["state"].float())
            x = (x - self.q_mean) / self.q_std
        b = x.size(0)
        flat = rearrange(x, "b t ... -> (b t) ...")
        emb = self.projector(self.encoder(flat))
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info
