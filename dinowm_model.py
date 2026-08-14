"""DinoWM = PreJEPA with its extra_encoders registered as an nn.ModuleDict.

The package stores extra_encoders as a PLAIN dict (prejepa.py: `self.extra_encoders =
extra_encoders or {}`), which makes the action Embedder's parameters invisible to
PyTorch: they do not move to the GPU with the model (the observed crash: cuda input vs
cpu weight in the Embedder's Conv1d), they are excluded from the optimizer (the action
encoder would silently never train), and they are missing from state_dict (the saved
checkpoint would lose them). Wrapping in ModuleDict fixes all three at once; ModuleDict
supports the same keys()/items()/indexing the rest of PreJEPA uses.

The training config's _target_ points here, so the eval side's load_pretrained --
which re-instantiates from the saved config -- constructs the same class and the
state_dict keys match.
"""

import torch.nn as nn

from stable_worldmodel.wm.prejepa.prejepa import PreJEPA


class DinoWM(PreJEPA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.extra_encoders, nn.ModuleDict):
            self.extra_encoders = nn.ModuleDict(self.extra_encoders)
