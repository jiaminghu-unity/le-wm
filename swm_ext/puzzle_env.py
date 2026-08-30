"""stable_worldmodel-compatible fork of OGBench's Puzzle environment.

This is an out-of-tree (le-wm repo) fork of `ogbench.manipspace.envs.puzzle_env.PuzzleEnv`
with the same layer of changes that stable_worldmodel applies to OGBench's SceneEnv in
`stable_worldmodel/envs/ogbench/scene_env.py` (upstream stable_worldmodel does not ship a
Puzzle fork). The swm layer adds:
- `reset(options=...)` plumbing with `swm_spaces.reset_variation_space` and an optional
  `options['state']` (concatenated qpos/qvel) override.
- A `variation_space` (button colors, agent color, arm start position, floor colors,
  camera angle perturbation, light intensity) applied via `modify_mjcf_model` /
  `initialize_episode` / `initialize_arm`.
- Per-element target setters (`set_target_button_state`).
- `get_reset_info` / `get_step_info` emitting `env_name`, `target`, and `success` on top
  of the base `proprio/*` and `privileged/*` keys, plus `goal_privileged/button_{i}_state`.
- `render` defaulting to the 'front_pixels' camera and `render_multiview` support.
- kwargs-based `set_state(qpos, qvel, button_state_{i}=...)`.

Expert policy for data collection: use `swm_ext.expert_policy.PuzzleExpertPolicy`
(a small subclass of `stable_worldmodel.envs.ogbench.expert_policy.ExpertPolicy`).
Both `policy_type='markov_oracle'` and `policy_type='plan_oracle'` work: the puzzle
target task is always 'button', dispatching ButtonMarkovOracle / ButtonPlanOracle with
`gripper_always_closed=True`. NOTE: the upstream swm ExpertPolicy CANNOT be used
directly with this env, because its `set_env()` asserts
`spec.id in ['swm/OGBCube-v0', 'swm/OGBScene-v0']`, and its puzzle oracle branch
checks for lowercase `'puzzle' in spec.id`, which 'swm/OGBPuzzle-v0' does not match.
"""

import mujoco
import numpy as np
from dm_control import mjcf
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.utils import perturb_camera_angle
from ogbench.manipspace import lie


class PuzzleEnv(ManipSpaceEnv):
    """Puzzle environment.

    This environment implements the "Lights Out" puzzle game. The goal is to set all buttons to a specific state. It
    supports the following variants:
    - `env_type`: '3x3', '4x4', '4x5', '4x6'.

    In addition to `qpos` and `qvel`, it maintains the following state variables.
    - `button_states`: A binary array of size `num_buttons` representing the state of each button. Stored in
        `_cur_button_states`.
    """

    def __init__(
        self,
        env_type='3x3',
        ob_type='states',
        multiview=False,
        *args,
        **kwargs,
    ):
        """Initialize the Puzzle environment.

        Args:
            env_type: Environment type. One of '3x3', '4x4', '4x5', or '4x6'.
            ob_type: Observation type. Either 'states' or 'pixels'.
            multiview: Whether to render from multiple cameras in `render_multiview`.
            *args: Additional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self._env_type = env_type
        self._multiview = multiview
        self.env_name = 'Puzzle'

        # Set the puzzle size.
        self._num_button_states = 2

        if env_type == '3x3':
            self._num_rows = 3
            self._num_cols = 3
        elif env_type == '4x4':
            self._num_rows = 4
            self._num_cols = 4
        elif env_type == '4x5':
            self._num_rows = 4
            self._num_cols = 5
        elif env_type == '4x6':
            self._num_rows = 4
            self._num_cols = 6
        else:
            raise ValueError(f'Unknown env_type: {env_type}')

        self._num_buttons = self._num_rows * self._num_cols
        self._cur_button_states = np.array([0] * self._num_buttons)
        self._prev_button_states = np.array([0] * self._num_buttons)

        super().__init__(*args, **kwargs)

        self._ob_type = ob_type

        # Adjust arm sampling bounds to a smaller region.
        self._arm_sampling_bounds = np.asarray(
            [[0.25, -0.2, 0.20], [0.6, 0.2, 0.25]]
        )

        # Target info.
        self._target_task = 'button'
        self._target_button = 0
        self._target_button_states = np.array([0] * self._num_buttons)

        self.variation_space = swm_spaces.Dict(
            {
                'button': swm_spaces.Dict(
                    {
                        # Colors of the (state 0, state 1) buttons.
                        'color': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(2, 3),
                            dtype=np.float64,
                            init_value=np.array(
                                [
                                    self._colors['red'][:3],
                                    self._colors['blue'][:3],
                                ]
                            ),
                        ),
                    }
                ),
                'agent': swm_spaces.Dict(
                    {
                        'color': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(3,),
                            dtype=np.float64,
                            init_value=self._colors['purple'][:3],
                        ),
                        'ee_start_position': swm_spaces.Box(  # x, y, z positions
                            low=self._arm_sampling_bounds[0],
                            high=self._arm_sampling_bounds[1],
                            shape=(3,),
                            dtype=np.float64,
                            init_value=np.mean(
                                self._arm_sampling_bounds, axis=0
                            ),
                        ),
                    }
                ),
                'floor': swm_spaces.Dict(
                    {
                        'color': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(2, 3),
                            dtype=np.float64,
                            init_value=np.array(
                                [[0.08, 0.11, 0.16], [0.15, 0.18, 0.25]]
                            ),
                        ),
                    }
                ),
                'camera': swm_spaces.Dict(
                    {
                        'angle_delta': swm_spaces.Box(
                            low=-5.0,
                            high=5.0,
                            shape=(2, 2) if self._multiview else (1, 2),
                            dtype=np.float64,
                            init_value=np.zeros([2, 2])
                            if self._multiview
                            else np.zeros([1, 2]),
                        ),
                    }
                ),
                'light': swm_spaces.Dict(
                    {
                        'intensity': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(1,),
                            dtype=np.float64,
                            init_value=[0.6],
                        ),
                    }
                ),
            }
        )

    def set_state(self, qpos, qvel, **kwargs):
        button_data = {
            int(key.split('_')[-1]): value
            for key, value in kwargs.items()
            if key.startswith('button_state_')
        }
        if button_data:
            button_states = np.array(
                [button_data[i] for i in range(self._num_buttons)]
            )
            self._cur_button_states = button_states.copy()

        self._apply_button_states()
        super().set_state(qpos, qvel)

    def set_tasks(self):
        if self._num_rows == 3 and self._num_cols == 3:
            self.task_infos = [
                {
                    'task_name': 'task1',
                    'init_button_states': np.array(
                        [
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                        ]
                    ).flatten(),
                    'goal_button_states': np.array(
                        [
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task2',
                    'init_button_states': np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                    'goal_button_states': np.array(
                        [
                            [0, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task3',
                    'init_button_states': np.array(
                        [
                            [0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0],
                        ]
                    ).flatten(),
                    'goal_button_states': np.array(
                        [
                            [1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task4',
                    'init_button_states': np.array(
                        [
                            [0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0],
                        ]
                    ).flatten(),
                    'goal_button_states': np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task5',
                    'init_button_states': np.array(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                        ]
                    ).flatten(),
                    'goal_button_states': np.array(
                        [
                            [1, 0, 1],
                            [1, 0, 1],
                            [1, 0, 1],
                        ]
                    ).flatten(),
                },
            ]
        elif self._num_rows == 4 and self._num_cols == 4:
            self.task_infos = [
                {
                    'task_name': 'task1',
                    'init_button_states': np.zeros(16, dtype=np.int64),
                    'goal_button_states': np.ones(16, dtype=np.int64),
                },
                {
                    'task_name': 'task2',
                    'init_button_states': np.ones(16, dtype=np.int64),
                    'goal_button_states': np.array(
                        [
                            [1, 1, 1, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 1, 1, 1],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task3',
                    'init_button_states': np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                        ]
                    ).flatten(),
                    'goal_button_states': np.array(
                        [
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [1, 0, 1, 0],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task4',
                    'init_button_states': np.array(
                        [
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                        ]
                    ).flatten(),
                    'goal_button_states': np.zeros(16, dtype=np.int64),
                },
                {
                    'task_name': 'task5',
                    'init_button_states': np.array(
                        [
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0],
                        ]
                    ).flatten(),
                    'goal_button_states': np.zeros(16, dtype=np.int64),
                },
            ]
        elif self._num_rows == 4 and self._num_cols == 5:
            self.task_infos = [
                {
                    'task_name': 'task1',
                    'init_button_states': np.array(
                        [
                            [1, 1, 0, 1, 1],
                            [0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0],
                            [1, 1, 0, 1, 1],
                        ]
                    ).flatten(),
                    'goal_button_states': np.zeros(20, dtype=np.int64),
                },
                {
                    'task_name': 'task2',
                    'init_button_states': np.zeros(20, dtype=np.int64),
                    'goal_button_states': np.ones(20, dtype=np.int64),
                },
                {
                    'task_name': 'task3',
                    'init_button_states': np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                    'goal_button_states': np.array(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task4',
                    'init_button_states': np.zeros(20, dtype=np.int64),
                    'goal_button_states': np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task5',
                    'init_button_states': np.zeros(20, dtype=np.int64),
                    'goal_button_states': np.array(
                        [
                            [1, 0, 0, 0, 1],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [1, 0, 0, 0, 1],
                        ]
                    ).flatten(),
                },
            ]
        elif self._num_rows == 4 and self._num_cols == 6:
            self.task_infos = [
                {
                    'task_name': 'task1',
                    'init_button_states': np.array(
                        [
                            [1, 1, 0, 1, 1, 1],
                            [0, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 0],
                            [1, 1, 1, 0, 1, 1],
                        ]
                    ).flatten(),
                    'goal_button_states': np.zeros(24, dtype=np.int64),
                },
                {
                    'task_name': 'task2',
                    'init_button_states': np.ones(24, dtype=np.int64),
                    'goal_button_states': np.zeros(24, dtype=np.int64),
                },
                {
                    'task_name': 'task3',
                    'init_button_states': np.zeros(24, dtype=np.int64),
                    'goal_button_states': np.array(
                        [
                            [1, 1, 1, 1, 1, 0],
                            [1, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 1],
                            [0, 1, 1, 1, 1, 1],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task4',
                    'init_button_states': np.array(
                        [
                            [0, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1],
                            [1, 0, 1, 0, 1, 0],
                        ]
                    ).flatten(),
                    'goal_button_states': np.array(
                        [
                            [1, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 1],
                        ]
                    ).flatten(),
                },
                {
                    'task_name': 'task5',
                    'init_button_states': np.zeros(24, dtype=np.int64),
                    'goal_button_states': np.array(
                        [
                            [1, 0, 0, 0, 0, 1],
                            [0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 0],
                            [1, 0, 0, 0, 0, 1],
                        ]
                    ).flatten(),
                },
            ]

        if self._reward_task_id == 0:
            # Set default task.
            if self._num_rows == 3 and self._num_cols == 3:
                self._reward_task_id = 4
            elif self._num_rows == 4 and self._num_cols == 4:
                self._reward_task_id = 4
            elif self._num_rows == 4 and self._num_cols == 5:
                self._reward_task_id = 2
            elif self._num_rows == 4 and self._num_cols == 6:
                self._reward_task_id = 2

    def reset(self, options=None, *args, **kwargs):
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=None,
            options=options,
        )

        ob, info = super().reset(options, *args, **kwargs)

        if 'state' in options and options['state'] is not None:
            state = options['state']
            # state should be a np.ndarray representing the concatenation of qpos and qvel
            assert isinstance(state, np.ndarray), (
                'State option must be a numpy ndarray!'
            )
            assert state.ndim == 1, 'State option must be a 1D array!'
            assert state.shape[0] == self._model.nq + self._model.nv, (
                f'State option must have shape ({self._model.nq + self._model.nv},)!'
            )
            qpos = state[: self._model.nq]
            qvel = state[self._model.nq :]
            self.set_state(qpos, qvel)
            self.pre_step()
            self.post_step()
            ob = self.compute_observation()
            info = self.get_reset_info()

        return ob, info

    def add_objects(self, arena_mjcf):
        # Add button scene.
        button_outer_mjcf = mjcf.from_path(
            (self._desc_dir / 'button_outer.xml').as_posix()
        )
        arena_mjcf.include_copy(button_outer_mjcf)

        # Add buttons to the scene.
        distance = 0.05
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                button_mjcf = mjcf.from_path(
                    (self._desc_dir / 'button_inner.xml').as_posix()
                )
                pos_x = (
                    0.425 - distance * (self._num_rows - 1) + 2 * distance * i
                )
                pos_y = (
                    0.0 - distance * (self._num_cols - 1) + 2 * distance * j
                )
                button_mjcf.find('body', 'buttonbox_0').pos[:2] = np.array(
                    [pos_x, pos_y]
                )
                for tag in ['body', 'joint', 'geom', 'site']:
                    for item in button_mjcf.find_all(tag):
                        if (
                            hasattr(item, 'name')
                            and item.name is not None
                            and item.name.endswith('_0')
                        ):
                            item.name = (
                                item.name[:-2] + f'_{i * self._num_cols + j}'
                            )
                arena_mjcf.include_copy(button_mjcf)

        # Save button geoms.
        self._button_geoms_list = []
        for i in range(self._num_buttons):
            self._button_geoms_list.append(
                [arena_mjcf.find('geom', f'btngeom_{i}')]
            )

        # Add cameras.
        self.cameras = {
            'front': {
                'pos': (1.139, 0.000, 0.821),
                'xyaxes': (0.000, 1.000, 0.000, -0.627, 0.000, 0.779),
            },
            'front_pixels': {
                'pos': (0.905, 0.000, 0.762),
                'xyaxes': (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
            'side_pixels': {
                'pos': (0.400, -0.505, 0.762),
                'xyaxes': (1.000, 0.000, 0.000, 0.00, 0.771, 0.637),
            },
        }
        for camera_name, camera_kwargs in self.cameras.items():
            arena_mjcf.worldbody.add(
                'camera', name=camera_name, **camera_kwargs
            )

    def post_compilation_objects(self):
        # Button geom IDs.
        self._button_geom_ids_list = [
            [
                self._model.geom(geom.full_identifier).id
                for geom in button_geoms
            ]
            for button_geoms in self._button_geoms_list
        ]
        self._button_site_ids = [
            self._model.site(f'btntop_{i}').id
            for i in range(self._num_buttons)
        ]

    def _apply_button_states(self):
        # Adjust button colors based on the current state.
        for i in range(self._num_buttons):
            for gid in self._button_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = (
                    self.variation_space['button']['color'].value[0]
                    if self._cur_button_states[i] == 0
                    else self.variation_space['button']['color'].value[1]
                )
                self._model.geom(gid).rgba[3] = 1.0

        mujoco.mj_forward(self._model, self._data)

    def modify_mjcf_model(self, mjcf_model):
        # Modify floor color
        grid_texture = mjcf_model.find('texture', 'grid')
        texture_changed = grid_texture.rgb1 is None or not np.allclose(
            grid_texture.rgb1, self.variation_space['floor']['color'].value[0]
        )
        texture_changed = texture_changed or (
            grid_texture.rgb2 is None
            or not np.allclose(
                grid_texture.rgb2,
                self.variation_space['floor']['color'].value[1],
            )
        )
        grid_texture.rgb1 = self.variation_space['floor']['color'].value[0]
        grid_texture.rgb2 = self.variation_space['floor']['color'].value[1]

        # Modify arm color
        agent_color_changed = np.allclose(
            mjcf_model.find('material', 'ur5e/robotiq/black').rgba[:3],
            self.variation_space['agent']['color'].value,
        )
        agent_color_changed = agent_color_changed or np.allclose(
            mjcf_model.find('material', 'ur5e/robotiq/pad_gray').rgba[:3],
            self.variation_space['agent']['color'].value,
        )
        mjcf_model.find('material', 'ur5e/robotiq/black').rgba[:3] = (
            self.variation_space['agent']['color'].value
        )
        mjcf_model.find('material', 'ur5e/robotiq/pad_gray').rgba[:3] = (
            self.variation_space['agent']['color'].value
        )

        # Perturb camera angle
        camera_angle_changed = False
        cameras_to_vary = (
            ['front_pixels', 'side_pixels']
            if self._multiview
            else ['front_pixels']
        )
        for i, cam_name in enumerate(cameras_to_vary):
            cam = mjcf_model.find('camera', cam_name)
            cam.xyaxes = perturb_camera_angle(
                self.cameras[cam_name]['xyaxes'],
                self.variation_space['camera']['angle_delta'].value[i],
            )
            camera_angle_changed = camera_angle_changed or not np.allclose(
                cam.xyaxes, self.cameras[cam_name]['xyaxes']
            )

        # Modify light intensity
        light = mjcf_model.find('light', 'global')
        desired_diffuse = self.variation_space['light']['intensity'].value[
            0
        ] * np.ones((3), dtype=np.float32)
        light_changed = light.diffuse is None or not np.allclose(
            light.diffuse, desired_diffuse
        )
        light.diffuse = desired_diffuse

        if light_changed or texture_changed or camera_angle_changed or agent_color_changed:
            self.mark_dirty()

        return mjcf_model

    def initialize_episode(self):
        if not hasattr(self, '_prev_qpos'):
            self._prev_qpos = self._data.qpos.copy()
            self._prev_qvel = self._data.qvel.copy()

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == 'data_collection':
            # Randomize the scene.

            self.initialize_arm()

            # Randomize button states.
            for i in range(self._num_buttons):
                self._cur_button_states[i] = self.np_random.choice(
                    self._num_button_states
                )
            self._apply_button_states()

            # Set a new target.
            self.set_new_target(return_info=False)

            # NOTE: Goal observation is not used in data collection mode.
            self._cur_goal_ob = np.zeros_like(self.compute_observation())
        else:
            # Set button states based on the current task.

            # Get the current task info.
            init_button_states = self.cur_task_info[
                'init_button_states'
            ].copy()
            goal_button_states = self.cur_task_info[
                'goal_button_states'
            ].copy()

            # First, force set the current scene to the goal state to obtain the goal observation.
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            self._cur_button_states = goal_button_states.copy()
            self._apply_button_states()
            mujoco.mj_forward(self._model, self._data)

            # Do a few random steps to make the scene stable.
            for _ in range(5):
                action = self.action_space.sample()
                action[-1] = 1  # Close gripper.
                self.step(action)

            # Save the goal observation.
            self._cur_goal_ob = (
                self.compute_oracle_observation()
                if self._use_oracle_rep
                else self.compute_observation()
            )
            if self._render_goal:
                self._cur_goal_rendered = self.render()
            else:
                self._cur_goal_rendered = None

            # Now, do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            self._cur_button_states = init_button_states.copy()
            self._target_button_states = goal_button_states.copy()
            self._apply_button_states()

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Unused; defined for compatibility with the other environments.
        """
        assert self._mode == 'data_collection'

        # Set target button.
        self._target_button = self.np_random.choice(self._num_buttons)
        self._target_button_states[self._target_button] = (
            self._cur_button_states[self._target_button] + 1
        ) % self._num_button_states

        mujoco.mj_kinematics(self._model, self._data)

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def initialize_arm(self):
        # Sample initial effector position and orientation.
        eff_pos = self.variation_space['agent']['ee_start_position'].value
        cur_ori = self._effector_down_rotation
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori

        # Solve for initial joint positions using IK.
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._home_qpos,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def pre_step(self):
        self._prev_button_states = self._cur_button_states.copy()
        super().pre_step()

    def _compute_successes(self):
        """Compute object successes."""
        button_successes = [
            (self._cur_button_states[i] == self._target_button_states[i])
            for i in range(self._num_buttons)
        ]

        return button_successes

    def set_target_button_state(self, button_id, target_state):
        """Set the target state for a specific button."""
        if button_id < 0 or button_id >= self._num_buttons:
            raise ValueError(
                f'button_id out of range (maximum {self._num_buttons - 1})'
            )
        self._target_button_states[button_id] = target_state

    def post_step(self):
        # Update button states.
        for i in range(self._num_buttons):
            prev_joint_pos = self._prev_ob_info[f'privileged/button_{i}_pos'][
                0
            ]
            cur_joint_pos = self._data.joint(
                f'buttonbox_joint_{i}'
            ).qpos.copy()[0]
            if prev_joint_pos > -0.02 and cur_joint_pos <= -0.02:
                # Button pressed: change the state of the button and its neighbors.
                x, y = i // self._num_cols, i % self._num_cols
                for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self._num_rows and 0 <= ny < self._num_cols:
                        self._cur_button_states[nx * self._num_cols + ny] = (
                            self._cur_button_states[nx * self._num_cols + ny]
                            + 1
                        ) % self._num_button_states
        self._apply_button_states()

        # Evaluate successes.
        button_successes = self._compute_successes()
        if self._mode == 'data_collection':
            self._success = button_successes[self._target_button]
        else:
            self._success = all(button_successes)

    def get_reset_info(self):
        reset_info = self.compute_ob_info()
        reset_info['env_name'] = self.env_name
        reset_info['target'] = self._cur_goal_ob
        reset_info['success'] = self._success
        return reset_info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        ob_info['env_name'] = self.env_name
        ob_info['target'] = self._cur_goal_ob
        ob_info['success'] = self._success
        return ob_info

    def add_object_info(self, ob_info):
        # Button states.
        for i in range(self._num_buttons):
            ob_info[f'privileged/button_{i}_state'] = self._cur_button_states[
                i
            ]
            ob_info[f'privileged/button_{i}_pos'] = self._data.joint(
                f'buttonbox_joint_{i}'
            ).qpos.copy()
            ob_info[f'privileged/button_{i}_vel'] = self._data.joint(
                f'buttonbox_joint_{i}'
            ).qvel.copy()

        # Goal button states. In 'task' mode, these hold the task goal; in
        # 'data_collection' mode, they track the current per-button targets.
        for i in range(self._num_buttons):
            ob_info[f'goal_privileged/button_{i}_state'] = (
                self._target_button_states[i]
            )

        if self._mode == 'data_collection':
            # Target button info.
            ob_info['privileged/target_task'] = self._target_task

            ob_info['privileged/target_button'] = self._target_button
            ob_info['privileged/target_button_state'] = (
                self._target_button_states[self._target_button]
            )
            ob_info['privileged/target_button_top_pos'] = self._data.site_xpos[
                self._button_site_ids[self._target_button]
            ].copy()

        ob_info['prev_button_states'] = self._prev_button_states.copy()
        ob_info['button_states'] = self._cur_button_states.copy()

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0
            button_scaler = 120.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_buttons):
                button_state = np.eye(self._num_button_states)[
                    self._cur_button_states[i]
                ]
                ob.extend(
                    [
                        button_state,
                        ob_info[f'privileged/button_{i}_pos'] * button_scaler,
                        ob_info[f'privileged/button_{i}_vel'],
                    ]
                )

            return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        return self._cur_button_states.astype(np.float64)

    def compute_reward(self):
        if self._reward_task_id is None:
            return super().compute_reward()

        # Compute the reward based on the task.
        successes = self._compute_successes()
        reward = float(sum(successes) - len(successes))
        return reward

    def render(
        self,
        camera='front_pixels',
        *args,
        **kwargs,
    ):
        """Render the current scene from a specified camera view.

        Generates an RGB image of the current environment state from a single
        camera viewpoint. This method renders from one camera at a time.

        Args:
            camera (str, optional): Camera name to render from. Defaults to
                'front_pixels'. Supports any camera defined in self.cameras
                (e.g., 'front_pixels', 'side_pixels').
            *args: Additional positional arguments passed to parent render method.
            **kwargs: Additional keyword arguments passed to parent render method.

        Returns:
            ndarray: Rendered image with shape (H, W, C) where H is height,
                W is width, and C is the number of color channels (typically 3 for RGB).

        Note:
            For rendering from multiple cameras simultaneously, use the
            `render_multiview()` method instead.
        """
        return super().render(camera=camera, *args, **kwargs)

    def render_multiview(
        self,
        camera='front_pixels',
        *args,
        **kwargs,
    ):
        """Render the current scene from multiple camera views or a fallback single view.

        When multiview mode is enabled (`_multiview=True`), renders the scene from
        both 'front_pixels' and 'side_pixels' cameras and returns them as a
        dictionary. When multiview is disabled, falls back to rendering from a
        single camera.

        Args:
            camera (str, optional): Camera name to use for fallback rendering when
                multiview is disabled. Defaults to 'front_pixels'. Ignored when
                multiview is enabled.
            *args: Additional positional arguments passed to the render method.
            **kwargs: Additional keyword arguments passed to the render method.

        Returns:
            dict or ndarray: If multiview is enabled, returns a dictionary with camera
                names as keys ('front_pixels', 'side_pixels') and rendered images as
                values, where each image has shape (H, W, C). If multiview is disabled,
                returns a single rendered image array with shape (H, W, C).
        """

        if not self._multiview:
            return self.render(camera=camera, *args, **kwargs)

        cam_names = ['front_pixels', 'side_pixels']
        multi_view = {
            cam: self.render(camera=cam, *args, **kwargs) for cam in cam_names
        }
        return multi_view
