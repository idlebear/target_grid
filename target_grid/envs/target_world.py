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
from .window import Colours, Window


DEFAULT_WORLD_PARAMETERS = {
    "grid_data": None,
    "agent_start_pos": None,
    "agent_start_dir": None,
    "goal_pos": None,
    "hazard_cost": DEFAULT_HAZARD_COST,
    "terminal_cost": DEFAULT_TERMINAL_COST,
    "step_cost": DEFAULT_STEP_COST,
    "num_targets": 0,
    "target_move_prob": None,
    "agent": None,
    "max_steps": DEFAULT_MAX_STEP,
    "screen_width": DEFAULT_SCREEN_WIDTH,
    "screen_height": DEFAULT_SCREEN_HEIGHT,
}


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
            self.grid_data = world_parameters.get(
                "grid_data", DEFAULT_WORLD_PARAMETERS["grid_data"]
            )
            self.agent_start_pos = world_parameters.get(
                "agent_start_pos", DEFAULT_WORLD_PARAMETERS["agent_start_pos"]
            )
            self.agent_start_dir = world_parameters.get(
                "agent_start_dir", DEFAULT_WORLD_PARAMETERS["agent_start_dir"]
            )
            self.goal_pos = world_parameters.get(
                "goal_pos", DEFAULT_WORLD_PARAMETERS["goal_pos"]
            )
            self.hazard_cost = world_parameters.get(
                "hazard_cost", DEFAULT_WORLD_PARAMETERS["hazard_cost"]
            )
            self.terminal_cost = world_parameters.get(
                "terminal_cost", DEFAULT_WORLD_PARAMETERS["terminal_cost"]
            )
            self.step_cost = world_parameters.get(
                "step_cost", DEFAULT_WORLD_PARAMETERS["step_cost"]
            )
            self.num_targets = world_parameters.get(
                "num_targets", DEFAULT_WORLD_PARAMETERS["num_targets"]
            )
            self.target_move_prob = world_parameters.get(
                "target_move_prob", DEFAULT_WORLD_PARAMETERS["target_move_prob"]
            )
            self.agent = world_parameters.get(
                "agent", DEFAULT_WORLD_PARAMETERS["agent"]
            )
            self.max_steps = world_parameters.get("max_steps", DEFAULT_MAX_STEP)
            self.screen_width = world_parameters.get(
                "screen_width", DEFAULT_SCREEN_WIDTH
            )
            self.screen_height = world_parameters.get(
                "screen_height", DEFAULT_SCREEN_HEIGHT
            )
        else:
            self.grid_data = DEFAULT_WORLD_PARAMETERS["grid_data"]
            self.agent_start_pos = DEFAULT_WORLD_PARAMETERS["agent_start_pos"]
            self.agent_start_dir = DEFAULT_WORLD_PARAMETERS["agent_start_dir"]
            self.goal_pos = DEFAULT_WORLD_PARAMETERS["goal_pos"]
            self.hazard_cost = DEFAULT_WORLD_PARAMETERS["hazard_cost"]
            self.terminal_cost = DEFAULT_WORLD_PARAMETERS["terminal_cost"]
            self.step_cost = DEFAULT_WORLD_PARAMETERS["step_cost"]
            self.num_targets = DEFAULT_WORLD_PARAMETERS["num_targets"]
            self.target_move_prob = DEFAULT_WORLD_PARAMETERS["target_move_prob"]
            self.agent = DEFAULT_WORLD_PARAMETERS["agent"]
            self.max_steps = DEFAULT_MAX_STEP
            self.screen_width = DEFAULT_SCREEN_WIDTH
            self.screen_height = DEFAULT_SCREEN_HEIGHT

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.size = size
        self.max_retry = DEFAULT_MAX_RETRY
        self.reset_count = 0

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
        self.goals = []
        self.targets = []
        self.objects = []

        # Create an empty grid, with only walls and empty (0,1) for the graph
        # connectivity
        if self.grid_data is None:
            self.grid_data = np.zeros((self.size, self.size), dtype=int)
            for i in range(self.size):
                attempts = 0
                while True:
                    x, y = self.rng.integers(0, self.size, size=2)
                    if self.grid_data[y, x] == 0:
                        break
                    attempts += 1
                    if attempts >= self.max_retry:
                        raise ValueError("Could not place the wall after 10 attempts")
                self.grid_data[y, x] = 1
            graph_data = self.grid_data
        else:
            graph_data = np.where(self.grid_data == 2, 1, 0)
        self.graph = GridGraph(edge_probability=1.0, grid_data=graph_data, seed=seed)

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

    def get_graph(self):
        return self.graph

    def add_wall(self, pos):
        self.grid_data[pos(1), pos(0)] = 2
        self.objects.append(Wall(node=pos, colour=Colours.grey))

    def add_hazard(self, pos):
        self.grid_data[pos(1), pos(0)] = 1
        self.objects.append(Hazard(node=pos, colour=Colours.red))

    def add_goal(self, pos):
        self.goals.append(Goal(node=pos, colour=Colours.green))

    def set_target_move_prob(self, move_prob):
        self.target_move_prob = move_prob

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

    def place_target(self, pos=None, move_prob=None):
        if move_prob is None:
            move_prob = self.target_move_prob
        if pos is None:
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
            target = Target(
                node=(x, y),
                colour=Colours.red,
                rng=self.rng,
                move_prob=move_prob,
            )
        else:
            target = Target(
                node=None,
                colour=Colours.red,
                rng=self.rng,
                move_prob=move_prob,
            )
        return target

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
        obs_data[0, :, :] *= GridState.OCCLUDED.value

        for obj in self.objects:
            if type(obj) is Hazard:
                obs_data[0, obj.node[1], obj.node[0]] = GridState.HAZARD.value
            elif type(obj) is Wall:
                obs_data[0, obj.node[1], obj.node[0]] = GridState.WALL.value
        for goal in self.goals:
            obs_data[0, goal.node[1], goal.node[0]] = GridState.GOAL.value
        obs_data[0, :, :] /= GridState.MAX_VALUE.value

        # second grid has location of the agent and the target
        #   target facing east (along x-axis)
        #   agent facing east
        target_nodes = []
        for target in self.targets:
            # if the target is visible, set the appropriate value in the grid
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

        # third grid has the distance to the goal
        obs_data[2, :, :] = self.distance_grid

        # make a list of the goal nodes
        goal_nodes = np.array([goal.node for goal in self.goals]).reshape(-1, 2)
        return {
            "agent": self.agent.node,
            "targets": target_nodes,
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

        # Place the obstacles in the grid by getting the indices of the obstacles in the
        # numpy array
        self.objects = []
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
                action_space_size=Actions.action_space_size.value,
                colour=Colours.blue,
            )

        # Place the goal
        self.goals = []
        if self.goal_pos is not None:
            if isinstance(self.goal_pos, list):
                for goal_pos in self.goal_pos:
                    self.goals.append(Goal(node=tuple(goal_pos), colour=Colours.green))
            else:
                self.goals.append(Goal(node=tuple(self.goal_pos), colour=Colours.green))
        # else:
        #     attempts = 0
        #     while True:
        #         x, y = self.rng.integers(
        #             0,
        #             self.size,
        #             [
        #                 2,
        #             ],
        #             dtype=int,
        #         )
        #         if self.grid_data[y, x] == 0 and self.agent.node != (x, y):
        #             break
        #         attempts += 1
        #         if attempts >= self.max_retry:
        #             raise ValueError("Could not place the goal after 10 attempts")
        #     self.goals.append(Goal(node=(x, y), colour=Colours.green))

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

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location, or any of the walls, or the goal
        self.targets = []
        for _ in range(self.num_targets):
            target = self.place_target()
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
        self.graph.draw(self.window, self.current_visibility)

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
