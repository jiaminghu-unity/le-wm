"""Register the out-of-tree swm_ext environments with gymnasium / stable_worldmodel.

Mirrors how `stable_worldmodel/envs/__init__.py` registers `swm/OGBScene-v0` etc.:
it uses stable_worldmodel's own `register` helper so the id is also added to
`stable_worldmodel.envs.WORLDS`, making it usable through the swm World pipeline.

Usage: `import swm_ext.register` (idempotent) before creating the env, e.g.
`gymnasium.make('swm/OGBPuzzle-v0', env_type='3x3', ob_type='states', mode='data_collection')`.
"""

import gymnasium as gym

from stable_worldmodel.envs import register


def register_envs():
    if 'swm/OGBPuzzle-v0' in gym.registry:
        return

    register(
        id='swm/OGBPuzzle-v0',
        entry_point='swm_ext.puzzle_env:PuzzleEnv',
    )


register_envs()
