import numpy as np
from target_grid.envs import TargetWorldEnv
from enum import Enum


class Map_Feature(Enum):
    GRASS = 0
    SIDEWALK = 1
    N_ROAD = 2
    S_ROAD = 3
    E_ROAD = 4
    W_ROAD = 5


FEATURE_MAP = {
    0: Map_Feature.GRASS,
    1: Map_Feature.SIDEWALK,
    2: Map_Feature.N_ROAD,
    3: Map_Feature.S_ROAD,
    4: Map_Feature.E_ROAD,
    5: Map_Feature.W_ROAD,
}

ROAD_TYPES = [
    Map_Feature.N_ROAD,
    Map_Feature.S_ROAD,
    Map_Feature.E_ROAD,
    Map_Feature.W_ROAD,
]

# Colors for each type (RGB tuples)
COLOR_MAP = {
    Map_Feature.GRASS: (34, 139, 34, 255),  # Forest Green
    Map_Feature.SIDEWALK: (192, 192, 192, 255),  # Silver
    Map_Feature.N_ROAD: (50, 50, 50, 255),  # Dark Grey (Road surface)
    Map_Feature.S_ROAD: (50, 50, 50, 255),  # Dark Grey
    Map_Feature.E_ROAD: (50, 50, 50, 255),  # Dark Grey
    Map_Feature.W_ROAD: (50, 50, 50, 255),  # Dark Grey
}


def main():
    world_parameters = {
        "grid_data": np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        "hazard_data": np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        "feature_data": np.array(
            [
                [
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.E_ROAD,
                    Map_Feature.E_ROAD,
                    Map_Feature.E_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                ],
                [
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                ],
                [
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.GRASS,
                ],
                [
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.GRASS,
                ],
                [
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.S_ROAD,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.GRASS,
                    Map_Feature.GRASS,
                ],
                [
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.N_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.GRASS,
                    Map_Feature.GRASS,
                    Map_Feature.GRASS,
                    Map_Feature.GRASS,
                ],
                [
                    Map_Feature.W_ROAD,
                    Map_Feature.W_ROAD,
                    Map_Feature.S_ROAD,
                    Map_Feature.E_ROAD,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                ],
                [
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.E_ROAD,
                    Map_Feature.E_ROAD,
                    Map_Feature.E_ROAD,
                    Map_Feature.E_ROAD,
                ],
                [
                    Map_Feature.W_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.S_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.N_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                    Map_Feature.SIDEWALK,
                ],
                [
                    Map_Feature.SIDEWALK,
                    Map_Feature.GRASS,
                    Map_Feature.SIDEWALK,
                    Map_Feature.GRASS,
                    Map_Feature.GRASS,
                    Map_Feature.SIDEWALK,
                    Map_Feature.N_ROAD,
                    Map_Feature.SIDEWALK,
                    Map_Feature.GRASS,
                    Map_Feature.GRASS,
                ],
            ]
        ),
        "color_map": COLOR_MAP,
        "agent_start_pos": (0, 0),
        "agent_start_dir": 0,
        "goal_pos": [(9, 9), (1, 8)],
        "hazard_cost": 1,
        "num_targets": 1,
        "target_move_prob": None,
        "agent": None,
    }

    env = TargetWorldEnv(
        render_mode="file",
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
