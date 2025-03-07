import numpy as np
from target_grid.envs import TargetWorldEnv


def main():
    world_parameters = {
        "grid_data": np.array(
            [
                [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                [2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 2, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 1, 1, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        "agent_start_pos": (0, 0),
        "agent_start_dir": 0,
        "goal_pos": [(9, 9), (1, 8)],
        "hazard_cost": 1,
        "num_targets": 1,
        "target_move_prob": None,
        "agent": None,
    }

    env = TargetWorldEnv(
        render_mode="human",
        seed=13,
        size=world_parameters["grid_data"].shape[0],
        world_parameters=world_parameters,
    )
    env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())
    env.close()


if __name__ == "__main__":
    main()
