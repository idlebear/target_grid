from gymnasium.envs.registration import register

register(
    id="target_grid/TargetWorld-v0",
    entry_point="target_grid.envs:TargetWorldEnv",
)
