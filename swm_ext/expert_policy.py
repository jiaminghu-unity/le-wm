"""Expert (oracle) collection policy for the out-of-tree swm/OGBPuzzle-v0 environment.

The upstream `stable_worldmodel.envs.ogbench.expert_policy.ExpertPolicy` cannot be used
directly with `swm/OGBPuzzle-v0`: its `set_env()` asserts
`spec.id in ['swm/OGBCube-v0', 'swm/OGBScene-v0']`, and its puzzle oracle branch checks
for lowercase `'puzzle' in spec.id`, which never matches 'swm/OGBPuzzle-v0'. This
subclass only relaxes the env check and builds the button oracles; everything else
(`get_action`, noise handling, subtask chaining via `set_new_target`) is inherited.

Both `policy_type='markov_oracle'` and `policy_type='plan_oracle'` work: the puzzle
target task is always 'button', so ButtonMarkovOracle / ButtonPlanOracle are used with
`gripper_always_closed=True` (matching OGBench's own puzzle data-generation setup).
"""

import numpy as np
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle

from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy


class PuzzleExpertPolicy(ExpertPolicy):
    """Collection Policy for the swm/OGBPuzzle-v0 environment."""

    def set_env(self, env):
        self.env = env
        single_env = self.env.envs[0].unwrapped
        assert single_env.spec.id == 'swm/OGBPuzzle-v0', (
            'PuzzleExpertPolicy can only be used with the swm/OGBPuzzle-v0 environment.'
        )

        self._set_oracle_agents()

        # to be set at each env reset:
        self._p_stack = np.zeros(self.env.num_envs)
        self._xi = np.zeros(
            self.env.num_envs
        )  # action noise level for Markov oracle
        self._agents = [None] * self.env.num_envs

    def _set_oracle_agents(self):
        single_env = self.env.envs[0].unwrapped
        if self.type == 'markov_oracle':
            # create one independent oracle instance per environment
            self._oracle_agents = {
                'button': [
                    ButtonMarkovOracle(
                        env=single_env,
                        min_norm=self.min_norm,
                        gripper_always_closed=True,
                    )
                    for _ in range(self.env.num_envs)
                ],
            }
        else:
            self._oracle_agents = {
                'button': [
                    ButtonPlanOracle(
                        env=single_env,
                        noise=self.action_noise,
                        noise_smoothing=self.noise_smoothing,
                        gripper_always_closed=True,
                    )
                    for _ in range(self.env.num_envs)
                ],
            }
