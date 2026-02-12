import numpy as np
import gymnasium as gym
import target_grid  # noqa: F401  # register envs
from target_grid.envs import build_section_iia_41_world_parameters


def run_linear():
    world_parameters = build_section_iia_41_world_parameters(
        lambda_energy=0.2,
        max_steps=50,
        screen_width=4000,
        screen_height=200,
    )
    env = gym.make(
        "target_grid/SensorSchedulingLinear-v0",
        render_mode="human",
        world_parameters=world_parameters,
    )
    obs, info = env.reset(seed=1)
    total_reward = 0.0
    for _ in range(25):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    print("Linear demo reward:", total_reward)


def run_grid():
    grid = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    env = gym.make(
        "target_grid/SensorSchedulingGrid-v0",
        render_mode="human",
        world_parameters={
            "grid_data": grid,
            "num_targets": 2,
            "absorbing_states": [(0, 0), (5, 5)],
            "sensor_specs": [
                {
                    "id": "s0",
                    "location": (0, 5),
                    "fov_deg": 360.0,
                    "range": 3.0,
                    "direction_deg": 0.0,
                    "energy_cost": 1.0,
                },
                {
                    "id": "s1",
                    "location": (5, 0),
                    "fov_deg": 120.0,
                    "range": 4.0,
                    "direction_deg": 90.0,
                    "energy_cost": 1.0,
                },
            ],
            "tracking_cost_mode": "unobserved",
            "lambda_energy": 0.15,
        },
    )
    obs, info = env.reset(seed=7)
    total_reward = 0.0
    for _ in range(25):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    print("Grid demo reward:", total_reward)


if __name__ == "__main__":
    run_linear()
    run_grid()
