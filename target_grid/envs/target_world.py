import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from polycheck import (
    visibility_from_region,
)

from .constants import (
    GridState,
    DEFAULT_HAZARD_COST,
    DEFAULT_TERMINAL_COST,
    DEFAULT_STEP_COST,
    DEFAULT_MAX_STEP,
    DEFAULT_MAX_RETRY,
    DEFAULT_SCREEN_HEIGHT,
    DEFAULT_SCREEN_WIDTH,
)
from .graphs import GridGraph
from .objects import Wall, Goal, Target, Agent, Hazard, Actions
from .window import Colors, Window


DEFAULT_WORLD_PARAMETERS = {
    "grid_data": None,
    "feature_data": None,
    "color_map": None,
    "edge_probability": 1.0,
    "hazard_data": None,
    "agent_start_pos": None,
    "agent_start_dir": None,
    "goal_pos": None,
    "hazard_cost": DEFAULT_HAZARD_COST,
    "terminal_cost": DEFAULT_TERMINAL_COST,
    "step_cost": DEFAULT_STEP_COST,
    "target_names": [],
    "target_properties": {},
    "target_seeds": None,
    "agent": None,
    "max_steps": DEFAULT_MAX_STEP,
    "view_range": None,
    "screen_width": DEFAULT_SCREEN_WIDTH,
    "screen_height": DEFAULT_SCREEN_HEIGHT,
}


class TargetWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "file"], "render_fps": 5}

    def __init__(
        self,
        render_mode=None,
        size=10,
        seed=None,
        world_parameters=None,
    ):
        super().__init__()

        # Start with a copy of default parameters
        params = DEFAULT_WORLD_PARAMETERS.copy()
        if world_parameters is not None:
            params.update(world_parameters)

        self.grid_data = params["grid_data"]
        self.feature_data = params["feature_data"]
        self.color_map = params["color_map"]
        self.edge_probability = params["edge_probability"]
        self.hazard_data = params["hazard_data"]
        self.agent_start_pos = params["agent_start_pos"]
        self.agent_start_dir = params["agent_start_dir"]
        self.goal_pos = params["goal_pos"]
        self.hazard_cost = params["hazard_cost"]
        self.terminal_cost = params["terminal_cost"]
        self.step_cost = params["step_cost"]
        self.target_names = params["target_names"]
        self.target_properties = params.get("target_properties", {})
        self.agent = params["agent"]
        self.max_steps = params["max_steps"]
        self.screen_width = params["screen_width"]
        self.screen_height = params["screen_height"]
        self.view_range = params["view_range"]

        assert self.grid_data is None or (
            np.max(self.grid_data) <= 1 and np.min(self.grid_data) >= 0
        )
        assert self.hazard_data is None or (
            np.max(self.hazard_data) <= 1 and np.min(self.hazard_data) >= 0
        )

        self.size = size
        self.max_retry = DEFAULT_MAX_RETRY
        self.reset_count = 0

        self.rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )
        self.seed = seed

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
                "targets": spaces.Tuple(
                    [
                        spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int)
                        for _ in self.target_names
                    ]
                ),
                "goal": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
                "grid": spaces.Box(
                    low=0, high=1, shape=(3, self.size, self.size), dtype=float
                ),
            }
        )
        self.action_space = spaces.Discrete(Actions.action_space_size.value)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "file":
            os.makedirs("render", exist_ok=True)

        self.goals = []
        self.targets = []
        self.objects = []

        if self.grid_data is None:
            self.grid_data = np.zeros((self.size, self.size), dtype=int)

        self.graph = GridGraph(
            edge_probability=self.edge_probability,
            grid_data=self.grid_data,
            seed=self.seed,
        )

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
        # Create a cache key that includes agent position and occluding target positions
        occluding_positions = tuple(
            target.node for target in self.targets if target.occluding
        )
        cache_key = (x, occluding_positions)

        visibility = self.visibility_cache.get(cache_key)
        if visibility is not None:
            return visibility

        if self.grid_data is None:
            visibility_grid = np.zeros((self.size, self.size))
        else:
            visibility_grid = self.grid_data.copy()

        # Add occluding targets to the temporary grid
        for pos in occluding_positions:
            visibility_grid[pos[1], pos[0]] = 1

        height, width = visibility_grid.shape
        ends = np.array([[i, j] for j in range(height) for i in range(width)])  # + 0.5
        start = (
            np.array(
                [
                    x,
                ]
            )
            # + 0.5
        )
        # Bresenham's line algorithm (integer based)
        visibility = visibility_from_region(
            data=visibility_grid, starts=start, ends=ends, max_range=self.view_range
        ).reshape(height, width)
        self.visibility_cache[cache_key] = visibility
        return visibility

    def get_graph(self):
        return self.graph

    def add_wall(self, pos):
        self.grid_data[pos(1), pos(0)] = 1
        self.objects.append(Wall(node=pos, color=Colors.grey))

    def add_hazard(self, pos):
        self.hazard_data[pos(1), pos(0)] = 1
        self.objects.append(Hazard(node=pos, color=Colors.red))

    def add_goal(self, pos):
        self.goals.append(Goal(node=pos, color=Colors.green))

    def set_target_properties(self, target_properties):
        self.target_properties = target_properties

    def place_agent(self):
        # Choose the agent's location uniformly at random
        attempts = 0
        while True:
            x = self.rng.integers(0, self.size, dtype=int)
            y = self.rng.integers(0, self.size, dtype=int)
            on_goal = False
            for goal in self.goals:
                if (x, y) == goal.node:
                    on_goal = True
                    break
            if self.grid_data[y, x] == 0 and not on_goal:
                break
            attempts += 1
            if attempts >= self.max_retry:
                raise ValueError("Could not place the agent after 10 attempts")

        return (x, y), self.rng.integers(0, Actions.action_space_size.value)

    def _get_random_target_start(self):
        attempts = 0
        while True:
            x, y = self.rng.integers(0, self.size, size=2, dtype=int)
            if not self.grid_data[y, x]:
                on_goal = False
                for goal in self.goals:
                    if (x, y) == goal.node:
                        on_goal = True
                        break
                if not on_goal and (x, y) != self.agent.node:
                    break
            attempts += 1
            if attempts >= self.max_retry:
                raise ValueError("Could not place the target after 10 attempts")
        return (x, y)

    def place_target(self, name=None, pos=None, seed=None):
        if name is not None:
            target_properties = self.target_properties[name]
        else:
            target_properties = self.target_properties[
                self.rng.choice(list(self.target_properties.keys()))
            ]
        move_prob = target_properties.get("transition_matrix", None)
        occluding = target_properties.get("occluding", False)
        motion_type = target_properties.get("motion_type", "random")
        if seed is None:
            seed = target_properties.get("seed", None)

        if pos is None:
            target_start = target_properties.get("start_positions", None)
            if target_start is not None:
                if isinstance(target_start, list):
                    # Randomly select one of the target starts
                    pos_idx = self.rng.integers(0, len(target_start))
                    pos = target_start[pos_idx]
                else:
                    pos = target_properties["target_start"]
            else:
                pos = self._get_random_target_start()

            target = Target(
                node=pos,
                color=target_properties.get("color", Colors.red),
                seed=seed,
                move_prob=move_prob,
                occluding=occluding,
                target_class=name,
                motion_type=motion_type,
            )
        else:
            target = Target(
                node=pos,
                color=target_properties.get("color", Colors.red),
                seed=seed,
                move_prob=move_prob,
                occluding=occluding,
                target_class=name,
                motion_type=motion_type,
            )
        return target

    def _get_obs(self):
        self.current_visibility = self._visibility_fn(self.agent.node)

        obs_data = np.zeros((3, self.size, self.size), dtype=float)
        obs_data[0, :, :] = self.current_visibility * GridState.OCCLUDED.value

        for obj in self.objects:
            if isinstance(obj, Hazard):
                obs_data[0, obj.node[1], obj.node[0]] = GridState.HAZARD.value
            elif isinstance(obj, Wall):
                obs_data[0, obj.node[1], obj.node[0]] = GridState.WALL.value
        for goal in self.goals:
            obs_data[0, goal.node[1], goal.node[0]] = GridState.GOAL.value
        obs_data[0, :, :] /= GridState.MAX_VALUE.value

        # Target and agent layer
        target_nodes = []
        for target in self.targets:
            if self.current_visibility[target.node[1], target.node[0]]:
                obs_data[1, target.node[1], target.node[0]] = (
                    GridState.TARGET.value + target.orientation
                )
                target_nodes.append(target.node)
            else:
                target_nodes.append(None)

        if obs_data[1, self.agent.node[1], self.agent.node[0]] != 0:
            obs_data[
                1, self.agent.node[1], self.agent.node[0]
            ] = GridState.COLLISION.value
        else:
            obs_data[1, self.agent.node[1], self.agent.node[0]] = (
                GridState.AGENT.value + self.agent.orientation
            )
        obs_data[1, :, :] /= GridState.MAX_VALUE.value

        obs_data[2, :, :] = self.distance_grid

        goal_nodes = np.array([goal.node for goal in self.goals], dtype=int).reshape(
            -1, 2
        )

        return {
            "agent": np.array(self.agent.node, dtype=int),
            "targets": tuple(
                (
                    np.array(t, dtype=int)
                    if t is not None
                    else np.array([-1, -1], dtype=int)
                )
                for t in target_nodes
            ),
            "goal": goal_nodes,
            "grid": obs_data,
        }

    def _get_info(self):
        distances = {}
        for goal in self.goals:
            distances[goal] = self.graph.get_distance(self.agent.node, goal.node)
        return {"distance": distances}

    def reset(self, options=None):
        self.steps = 0
        self.terminated = False

        if options is None:
            options = {}

        # clear the cache in case of updates
        self.visibility_cache = {}

        # Place the obstacles in the grid by getting the indices of the obstacles in the
        # numpy array
        self.objects = []
        obs = list(zip(*self.grid_data.nonzero()))
        for y, x in obs:
            wall = Wall(node=(x, y), color=Colors.grey)
            self.objects.append(wall)

        # and the hazards...
        if self.hazard_data is not None:
            obs = list(zip(*self.hazard_data.nonzero()))
            for y, x in obs:
                hazard = Hazard(node=(x, y), color=Colors.red)
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
                action_space_size=Actions.action_space_size.value,
                color=Colors.blue,
            )

        # Place the goal
        self.goals = []
        if self.goal_pos is not None:
            if isinstance(self.goal_pos, list):
                for goal_pos in self.goal_pos:
                    self.goals.append(Goal(node=tuple(goal_pos), color=Colors.green))
            else:
                self.goals.append(Goal(node=tuple(self.goal_pos), color=Colors.green))

        # Update the distances to the goal(s) for all nodes
        if len(self.goals):
            distances = []
            for goal in self.goals:
                d = self.graph.get_distances(goal.node)
                node_distances = np.zeros((self.size, self.size))
                for node, distance in d.items():
                    node_distances[node[1], node[0]] = distance
                distances.append(node_distances)
            distances = np.array(distances)
            self.distance_grid = np.min(distances, axis=0)
            self.distance_grid /= float(np.max(self.distance_grid))  # normalize
        else:
            self.distance_grid = np.zeros((self.size, self.size))

        target_names = options.get("target_names", None)
        if target_names is not None:
            self.target_names = target_names
            restore_initial_state = False
        else:
            self.target_names = self.target_properties.keys()
            restore_initial_state = options.get("restore_initial_state", False)
        if restore_initial_state and len(self.targets):
            target_seeds = options.get("target_seeds", None)
            for target_idx, target in enumerate(self.targets):
                if target_seeds is None:
                    # target will reuse its current seed
                    seed = None
                else:
                    if isinstance(target_seeds, list) or isinstance(
                        target_seeds, np.ndarray
                    ):
                        seed = target_seeds[target_idx]
                    else:
                        seed = int(target_seeds)
                target.reset(seed=seed)
        else:
            target_seeds = options.get("target_seeds", None)

            self.targets = []
            for target_idx, target_name in enumerate(self.target_names):
                if target_seeds is None:
                    seed = None
                else:
                    seed = (
                        target_seeds[target_idx]
                        if target_idx < len(target_seeds)
                        else None
                    )
                target = self.place_target(
                    name=target_name,
                    seed=seed,
                )
                self.targets.append(target)

        # reset the frame counter
        self.reset_count += 1
        self.frame_count = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "file":
            os.makedirs(f"./render/{self.reset_count:05d}", exist_ok=True)

        if self.render_mode == "human" or self.render_mode == "file":
            self._render_frame()

        return observation, info

    def step(self, action):

        if self.terminated:
            return self._get_obs(), 0, True, True, self._get_info()

        self.steps += 1
        if self.steps >= self.max_steps:
            return self._get_obs(), 0, True, True, self._get_info()

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.agent.step(self.graph, action)

        # step all of the targets
        for target in self.targets:
            target.step(self.graph)

        # if there is a collision, the episode is over
        if self.hazard_cost:
            for target in self.targets:
                if np.array_equal(self.agent.node, target.node):
                    self.terminated = True
                    reward = self.hazard_cost
                    observation = self._get_obs()
                    info = self._get_info()
                    return observation, reward, self.terminated, False, info

        # An episode is done iff the agent has reached the goal
        for goal in self.goals:
            self.terminated = np.array_equal(self.agent.node, goal.node)
            if self.terminated:
                break
        if self.terminated:
            reward = self.terminal_cost
        else:
            reward = (
                self.step_cost
                * self.distance_grid[self.agent.node[1], self.agent.node[0]]
            )  # incentivise movement towards the goal
        # reward = 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "file":
            self._render_frame()

        return observation, reward, self.terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "file":
            return self._render_frame()

    def _render_frame(self):
        self.frame_count += 1
        if self.window is None:
            Window.initialize(with_display=(self.render_mode == "human"))

        if self.window is None:
            self.window = Window(
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                margin=max(1, int(min(self.screen_height, self.screen_width) / 100)),
                display_size=self.size,
                frame_rate=self.metadata["render_fps"],
            )

        self.window.clear()
        self.graph.draw(
            self.window,
            self.current_visibility,
            feature_data=self.feature_data,
            color_map=self.color_map,
        )

        for obj in self.objects:
            obj.draw(self.window, self.current_visibility[obj.node[1], obj.node[0]])
        for goal in self.goals:
            goal.draw(self.window, self.current_visibility[goal.node[1], goal.node[0]])
        for target in self.targets:
            target.draw(
                self.window, self.current_visibility[target.node[1], target.node[0]]
            )
        self.agent.draw(self.window)

        if self.render_mode == "human":
            self.window.display()
        elif self.render_mode == "file":
            path = (
                f"./render/{self.reset_count:05d}/tw_frame_{self.frame_count:05d}.png"
            )
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
        "target_names": "One",
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
