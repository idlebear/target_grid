from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from polycheck import (
    # visibility_from_real_region,
    visibility_from_region,
)

from .graphs import GridGraph
from .objects import Wall, Goal, Target, Agent, Hazard
from .window import Colours, Window, SCREEN_HEIGHT, SCREEN_WIDTH


class Actions(Enum):
    north = 0
    north_east = 1
    east = 2
    south_east = 3
    south = 4
    south_west = 5
    west = 6
    north_west = 7
    action_space_size = 8


action_to_direction = {
    Actions.north.value: np.array([1, 0]),
    Actions.north_east.value: np.array([1, 1]),
    Actions.east.value: np.array([0, 1]),
    Actions.south_east.value: np.array([-1, 1]),
    Actions.south.value: np.array([-1, 0]),
    Actions.south_west.value: np.array([-1, -1]),
    Actions.west.value: np.array([0, -1]),
    Actions.north_west.value: np.array([1, -1]),
}


def action_to_node(node, action):
    return tuple(np.add(node, action_to_direction[action]))


class GridState(Enum):
    empty = 0
    target = 20
    agent = 30
    goal = 40
    occluded = 50
    collision = 70
    hazard = 80
    wall = 100
    max_value = 100


class TargetWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "file"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        size=10,
        seed=None,
        world_parameters=None,
    ):
        super().__init__()

        if world_parameters is not None:
            self.grid_data = world_parameters.get("grid_data")
            self.agent_start_pos = world_parameters.get("agent_start_pos")
            self.agent_start_dir = world_parameters.get("agent_start_dir")
            self.goal_pos = world_parameters.get("goal_pos")
            self.hazard_cost = world_parameters.get("hazard_cost")
            self.num_targets = world_parameters.get("num_targets")
            self.target_move_prob = world_parameters.get("target_move_prob")
            self.agent = world_parameters.get("agent")
        else:
            self.grid_data = None
            self.agent_start_pos = None
            self.agent_start_dir = None
            self.goal_pos = None
            self.hazard_cost = 0
            self.num_targets = 1
            self.target_move_prob = None
            self.agent = None

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.size = size

        # observations are a dictionary with keys for agent and target locations,
        # as well as a grid showing the visible portion of the world
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
                "grid": spaces.Box(
                    low=0, high=1, shape=(self.size, self.size), dtype=int
                ),
            }
        )
        # BUGBUG - add the distance to the goal as a wrapper

        # We have 8 actions, corresponding to "north", "northeast", "east" etc.
        self.action_space = spaces.Discrete(Actions.action_space_size.value)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "file":
            os.makedirs("render", exist_ok=True)

        # initialize the environment
        self.goal = None
        self.targets = []
        self.objects = []

        self.graph = None
        self.window = None
        self.clock = None

        self.visibility_cache = {}

    def _visibility_fn(self, x):
        """
        Calculate the visibility of all points of a grid from a single location.

        Parameters:
        x (tuple): The grid coordinates of the first point.
        grid (numpy.ndarray): A 2D array representing the grid.

        Returns:
        numpy.ndarray: A 2D array representing the visibility of each point
        from the first point.

        """
        visibility = self.visibility_cache.get(x)
        if visibility is not None:
            return visibility

        if self.grid_data is None:
            visibility = np.ones((self.size, self.size))
        else:
            occlusions = np.where(self.grid_data == 2, 1, 0)

            height, width = occlusions.shape
            ends = np.array(
                [[i, j] for j in range(height) for i in range(width)]
            )  # + 0.5
            start = (
                np.array(
                    [
                        x,
                    ]
                )
                # + 0.5
            )
            # visibility = visibility_from_real_region(
            #     data=occlusions,
            #     origin=(0, 0),
            #     resolution=1.0,
            #     starts=start,
            #     ends=ends,
            # ).reshape(height, width)
            # Bresenham's line algorithm (integer based)
            visibility = visibility_from_region(
                data=occlusions, starts=start, ends=ends
            ).reshape(height, width)
        self.visibility_cache[x] = visibility
        return visibility

    def place_agent(self):
        # Choose the agent's location uniformly at random
        while True:
            x = self.rng.integers(0, self.size, dtype=int)
            y = self.rng.integers(0, self.size, dtype=int)
            if self.grid_data[y, x] == 0 and self.goal_pos != (x, y):
                break
        return (x, y), self.rng.integers(0, Actions.action_space_size.value)

    def _get_obs(self):
        # base grid has the following values
        #   empty
        #   wall
        #   hazard
        #   goal
        #   occluded
        self.current_visibility = self._visibility_fn(self.agent.node)
        obs_data = np.zeros((3, self.size, self.size), dtype=float)
        obs_data[0, :, :] = self.current_visibility
        obs_data[0, :, :] *= GridState.occluded.value
        for obj in self.objects:
            if type(obj) is Hazard:
                obs_data[0, obj.node[1], obj.node[0]] = GridState.hazard.value
            elif type(obj) is Wall:
                obs_data[0, obj.node[1], obj.node[0]] = GridState.wall.value
        obs_data[0, self.goal.node[1], self.goal.node[0]] = GridState.goal.value
        obs_data[0, :, :] /= GridState.max_value.value

        # second grid has location of the agent and the target
        #   target facing east (along x-axis)
        #   agent facing east
        for target in self.targets:
            obs_data[1, target.node[1], target.node[0]] = (
                GridState.target.value + target.orientation
            )
        if obs_data[1, self.agent.node[1], self.agent.node[0]] != 0:
            obs_data[
                1, self.agent.node[1], self.agent.node[0]
            ] = GridState.collision.value
        else:
            obs_data[1, self.agent.node[1], self.agent.node[0]] = (
                GridState.agent.value + self.agent.orientation
            )
        obs_data[1, :, :] /= GridState.max_value.value

        # third grid has the distance to the goal
        obs_data[2, :, :] = self.distance_grid

        return {"agent": self.agent.node, "goal": self.goal.node, "grid": obs_data}

    def _get_info(self):
        return {"distance": self.graph.get_distance(self.agent.node, self.goal.node)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Create an empty grid
        if self.grid_data is None:
            self.grid_data = np.zeros((self.size, self.size), dtype=int)
            for i in range(self.size):
                while True:
                    x, y = self.rng.integers(0, self.size, size=2)
                    if self.grid_data[y, x] == 0:
                        break
                self.grid_data[y, x] = 1

        self.graph = GridGraph(
            edge_probability=1.0, grid_data=self.grid_data, seed=seed
        )

        # Place the obstacles in the grid by getting the indices of the obstacles in the
        # numpy array
        obs = list(zip(*self.grid_data.nonzero()))
        for y, x in obs:
            if self.grid_data[y, x] == 2:
                wall = Wall(node=(x, y), colour=Colours.grey)
                self.objects.append(wall)
            elif self.grid_data[y, x] == 1:
                hazard = Hazard(node=(x, y), colour=Colours.red)
                self.objects.append(hazard)

        # Place the agent
        if self.agent_start_pos is not None:
            agent_node = self.agent_start_pos
            agent_orientation = self.agent_start_dir
        else:
            agent_node, agent_orientation = self.place_agent()

        if self.agent is not None:
            self.agent.node = agent_node
            self.agent.orientation = agent_orientation
        else:
            self.agent = Agent(
                node=agent_node,
                orientation=agent_orientation,
                colour=Colours.blue,
            )

        if self.goal_pos is not None:
            self.goal = Goal(node=self.goal_pos, colour=Colours.green)
        else:
            while True:
                x, y = self.rng.integers(
                    0,
                    self.size,
                    [
                        2,
                    ],
                    dtype=int,
                )
                if self.grid_data[y, x] == 0 and self.agent.node != (x, y):
                    break
            self.goal = Goal(node=(x, y), colour=Colours.green)

        # Update the distances to the goal for all nodes
        distances = self.graph.get_distances(self.goal.node)
        self.distance_grid = np.zeros_like(self.grid_data, dtype=float)
        for node, distance in distances.items():
            self.distance_grid[node[1], node[0]] = distance
        self.distance_grid /= float(np.max(self.distance_grid))  # normalize

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location, or any of the walls, or the goal
        while True:
            x, y = self.rng.integers(0, self.size, size=2, dtype=int)
            if (
                self.grid_data[y, x] == 0
                and (x, y) != self.agent.node
                and (x, y) != self.goal.node
            ):
                break
        target = Target(
            node=(x, y),
            colour=Colours.red,
            action_space_size=Actions.action_space_size.value,
            rng=self.rng,
            action_to_node=action_to_node,
            graph=self.graph,
            move_prob=self.target_move_prob,
        )
        self.targets = [target]

        # reset the frame counter
        self.frame_count = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "file":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        next_node = action_to_node(self.agent.node, action)
        self.agent.node = self.graph.validate_node(self.agent.node, next_node)

        # step all of the targets
        for target in self.targets:
            target.step()

        # if there is a collision, the episode is over
        for target in self.targets:
            if np.array_equal(self.agent.node, target.node):
                terminated = True
                reward = -10
                observation = self._get_obs()
                info = self._get_info()
                return observation, reward, terminated, False, info

        # An episode is done iff the agent has reached the goal
        terminated = np.array_equal(self.agent.node, self.goal.node)
        if terminated:
            reward = 10
        else:
            reward = (
                -0.1 * self.distance_grid[self.agent.node[1], self.agent.node[0]]
            )  # incentivise movement towards the goal
        # reward = 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            Window.initialize_screen()

        if self.window is None:
            self.window = Window(
                screen_width=SCREEN_WIDTH,
                screen_height=SCREEN_HEIGHT,
                margin=10,
                display_size=self.size,
                frame_rate=self.metadata["render_fps"],
            )

        self.window.clear()
        self.graph.draw(self.window, self.current_visibility)

        for obj in self.objects:
            obj.draw(self.window, self.current_visibility[obj.node[1], obj.node[0]])

        for target in self.targets:
            target.draw(
                self.window, self.current_visibility[target.node[1], target.node[0]]
            )
        self.agent.draw(self.window)

        if self.render_mode == "human":
            self.window.display()
        elif self.render_mode == "file":
            path = f"target_world_frame_{self.frame_count:04d}.png"
            self.window.save_frame(path)

        else:  # rgb_array
            return self.window.render()

    def close(self):
        if self.window is not None:
            self.window.close()


if __name__ == "__main__":

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
        "goal_pos": None,
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
