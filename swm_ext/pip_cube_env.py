"""Picture-in-picture Cube env for the visible-distractor experiment.

Subclasses the registered swm/OGBCube-v0 env (base class resolved from the gym
registry so the import survives stable_worldmodel refactors) and composites a
prerecorded 48x48 scene clip into the bottom-left corner of EVERY rendered
frame — live observations and the goal render alike, matching how the training
frames in cube_pip.lance were built (scripts/gen_pip_dataset.py: corner
(y>=176, x<48), clip advances one frame per env step, wraps around).

Clips come from a npz (`clips`: uint8 (K, L, 48, 48, 3)) whose path is given by
the PIP_CLIPS env var. Clip choice increments per reset (deterministic given
the fixed episode order of budget_sweep); the goal render inside reset() uses
the same clip at frame 0 while live frames advance — the distractor state
mismatch between goal and observation is the point of the experiment.
"""

import importlib
import os

import gymnasium as gym
import numpy as np

import stable_worldmodel.envs  # noqa: F401  (registers swm/OGBCube-v0)

_spec = gym.spec("swm/OGBCube-v0")
_mod, _cls = str(_spec.entry_point).split(":")
_CubeEnv = getattr(importlib.import_module(_mod), _cls)

PATCH = 48
OY, OX = 224 - PATCH, 0


class PiPCubeEnv(_CubeEnv):
    def __init__(self, *args, **kwargs):
        clips_path = kwargs.pop("pip_clips", None) or os.environ.get("PIP_CLIPS")
        if not clips_path or not os.path.exists(clips_path):
            raise FileNotFoundError(f"PIP_CLIPS npz not found: {clips_path!r}")
        self._pip_clips = np.load(clips_path)["clips"]  # (K, L, 48, 48, 3) uint8
        assert self._pip_clips.dtype == np.uint8 and self._pip_clips.shape[2:] == (PATCH, PATCH, 3)
        self._pip_k = -1
        self._pip_t = 0
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._pip_k = (self._pip_k + 1) % len(self._pip_clips)
        self._pip_t = 0
        return super().reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self._pip_t += 1
        return super().step(*args, **kwargs)

    def render(self, *args, **kwargs):
        img = super().render(*args, **kwargs)
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[:2] == (224, 224):
            clip = self._pip_clips[max(self._pip_k, 0)]
            img = img.copy()
            img[OY:OY + PATCH, OX:OX + PATCH] = clip[self._pip_t % len(clip)]
        return img
