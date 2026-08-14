"""Gymnasium adapter for DINO-WM's PointMaze (UMaze), driveable by swm.World.

The underlying env (vendored from temporal-straightening: maze_model.py +
point_maze_wrapper.py, mujoco_py) speaks the old gym API: step returns a 4-tuple with
done always False, reset returns (obs_dict, state). swm.World needs a gymnasium env
plus the two conventions our eval protocol uses: `_set_state` / `_set_goal_state`
methods on the unwrapped env, and `render()` returning an HWC uint8 frame.

Decisions that are NOT cosmetic:

  * `prepare_for_render()` is called once in __init__. The camera (azimuth 90,
    elevation -90, the top-down view every dataset frame was rendered with) is
    configured ONLY there; rendering before it gives a different viewpoint and every
    pixel comparison would fail. The scene gate that validated this env went through
    the same call.
  * `_set_goal_state` stores the goal for termination and does NOT touch the env's
    visible target marker. The dataset frames were rendered with the default marker,
    and the gate matched them at MAE 0.039 -- moving the marker would put evaluation
    frames outside the training distribution.
  * Termination is computed here (`||pos - goal_xy|| < 0.5`), because the base env's
    `done` is always False. The criterion is DINO-WM's own `eval_state`, positions
    only, velocities ignored.

Import path: the vendored upstream files live outside the repo (a worker-local
directory), located via the PMENV_DIR environment variable.
"""

import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_PMENV = os.environ.get("PMENV_DIR")
if _PMENV and _PMENV not in sys.path:
    sys.path.insert(0, _PMENV)

from env.pointmaze.maze_model import U_MAZE  # noqa: E402
from env.pointmaze.point_maze_wrapper import PointMazeWrapper  # noqa: E402

SUCCESS_RADIUS = 0.5  # DINO-WM eval_state: ||goal[:2] - cur[:2]|| < 0.5


class DWMPointMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str = "rgb_array"):
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._env = PointMazeWrapper(maze_spec=U_MAZE)
        self._env.prepare_for_render()  # camera + return_value='obs'; see module docstring
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # (x, y, vx, vy) -- the same 4-d state the dataset's `state` column holds
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self._goal = None
        self._state = np.zeros(4, dtype=np.float32)
        self.env_name = "DWMPointMaze"

    # ---- gymnasium API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.seed(int(seed) if seed is not None else int(np.random.randint(2**31)))
        self._env.set_init_state(None)  # random; the callables pin the exact state next
        _, state = self._env.reset()
        self._state = np.asarray(state, dtype=np.float32).reshape(-1)[:4]
        self._goal = None
        return self._state.copy(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
        _, _, _, info = self._env.step(a)  # base done is always False
        self._state = np.asarray(info["state"], dtype=np.float32).reshape(-1)[:4]
        terminated = False
        if self._goal is not None:
            terminated = bool(
                np.linalg.norm(self._state[:2] - self._goal[:2]) < SUCCESS_RADIUS
            )
        return self._state.copy(), 0.0, terminated, False, {}

    def render(self):
        return np.asarray(self._env._render_frame())  # HWC uint8, 224x224

    # ---- eval-protocol conveniences (same contract as TwoRoomEnv's) ----
    def _set_state(self, state):
        """Reset the simulator to an exact dataset state (x, y, vx, vy)."""
        st = np.asarray(state, dtype=np.float64).reshape(-1)[:4]
        self._env.set_init_state(st)
        _, s = self._env.reset()
        self._state = np.asarray(s, dtype=np.float32).reshape(-1)[:4]

    def _set_goal_state(self, goal_state):
        """Store the goal for termination. Deliberately does not move the visible
        target marker -- see module docstring."""
        self._goal = np.asarray(goal_state, dtype=np.float64).reshape(-1)[:4]


def register():
    """Register with gymnasium under swm/DWMPointMaze-v0, idempotently."""
    if "swm/DWMPointMaze-v0" not in gym.registry:
        gym.register(id="swm/DWMPointMaze-v0",
                     entry_point="scripts.pointmaze_env:DWMPointMazeEnv")
        print("[pointmaze_env] registered swm/DWMPointMaze-v0", flush=True)
