from gymnasium.envs.registration import register

register(
    id="target_grid/TargetWorld-v0",
    entry_point="target_grid.envs:TargetWorldEnv",
)

register(
    id="target_grid/SensorSchedulingLinear-v0",
    entry_point="target_grid.envs:SensorSchedulingLinearEnv",
)

register(
    id="target_grid/SensorSchedulingGrid-v0",
    entry_point="target_grid.envs:SensorSchedulingGridEnv",
)
