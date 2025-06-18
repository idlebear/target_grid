"""
This module defines objects that are capable of moving and drawing themselves.
"""

import numpy as np

from .actions import Actions, action_to_node, node_to_action
from .graphs import GridGraph
from .window import Window, Colors


class Object:
    def __init__(self, node, color, **kwargs):
        self.node = tuple(node)
        self.color = color
        # Ensure the color has an alpha channel
        if len(self.color) != 4:
            self.color = tuple(list(self.color)[:3] + [255])

    def reset(self):
        pass

    def step(self, graph: GridGraph, action: int):
        raise NotImplementedError

    def draw(self, window: Window):
        raise NotImplementedError

    def copy(self):
        return self.__class__(self.node, self.color)


class Wall(Object):
    def __init__(self, node, color=(0, 0, 0), **kwargs):
        super().__init__(node, color)

    def step(self, graph: GridGraph, action: int):
        pass

    def draw(self, window: Window, visibility: float = 1.0):
        color = list(self.color)
        # if visibility < 1.0:
        #     color[3] = int(visibility * 127)
        # else:
        #     color[3] = 255
        window.draw_rect(
            center=(self.node[0] + 0.5, self.node[1] + 0.5),
            height=1,
            width=1,
            color=color,
            use_transparency=True,
        )


class Hazard(Object):
    def __init__(self, node, color=(0, 0, 0), **kwargs):
        super().__init__(node, color)

    def step(self, graph: GridGraph, action: int):
        pass

    def draw(self, window: Window, visibility: float = 1.0):
        color = list(self.color)
        line_color = list(Colors.black)
        if visibility < 1.0:
            color[3] = int(visibility * 127)
        else:
            color[3] = 255
            line_color[3] = int(visibility * 255)
        window.draw_rect(
            center=(self.node[0] + 0.5, self.node[1] + 0.5),
            height=1,
            width=1,
            color=None,
            border_width=5,
            border_color=color,
            use_transparency=True,
        )
        # Draw an x through the hazard
        window.draw_line(
            start=(self.node[0] + 0.25, self.node[1] + 0.25),
            end=(self.node[0] + 0.75, self.node[1] + 0.75),
            color=color,
            width=5,
            use_transparency=True,
        )
        window.draw_line(
            start=(self.node[0] + 0.75, self.node[1] + 0.25),
            end=(self.node[0] + 0.25, self.node[1] + 0.75),
            color=color,
            width=5,
            use_transparency=True,
        )


class Goal(Object):
    def __init__(self, node, color=(0, 255, 0), **kwargs):
        super().__init__(node, color)

    def step(self, graph: GridGraph, action: int):
        pass

    def draw(self, window: Window, visibility: float = 1.0):
        color = list(self.color)
        line_color = list(Colors.black)
        if visibility < 1.0:
            color[3] = int(visibility * 127)
        else:
            color[3] = 255
            line_color[3] = int(visibility * 255)
        window.draw_rect(
            center=(self.node[0] + 0.5, self.node[1] + 0.5),
            height=1,
            width=1,
            color=color,
            use_transparency=True,
        )
        # Draw an box to show the goal when ocluded. Transparency
        # doesn't work with lines.
        window.draw_polyline(
            points=[
                (self.node[0] + 0.25, self.node[1] + 0.25),
                (self.node[0] + 0.75, self.node[1] + 0.25),
                (self.node[0] + 0.75, self.node[1] + 0.75),
                (self.node[0] + 0.25, self.node[1] + 0.75),
                (self.node[0] + 0.25, self.node[1] + 0.25),
            ],
            color=line_color,
            width=5,
            use_transparency=True,
        )


class Target(Object):
    def __init__(self, node, color=(255, 0, 0), **kwargs):
        super().__init__(node, color)
        self.orientation = kwargs.get("orientation", 0)
        self.action_space_size = Actions.action_space_size.value
        if self.action_space_size > 0:
            self.angle_increment = np.pi * 2.0 / self.action_space_size
        else:
            self.angle_increment = None

        self.seed = kwargs.get("seed", None)
        self.rng = np.random.default_rng(self.seed)

        self.initial_node = self.node
        self.initial_orientation = self.orientation
        self.move_prob = kwargs.get("move_prob", None)

    def reset(self, seed=None):
        self.node = self.initial_node
        self.orientation = self.initial_orientation
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def copy(self):
        new_target = self.__class__(self.node, self.color)
        new_target.orientation = self.orientation
        new_target.action_space_size = self.action_space_size
        new_target.angle_increment = self.angle_increment
        new_target.rng = self.rng
        new_target.move_prob = self.move_prob
        return new_target

    def step(self, graph: GridGraph):
        if self.node is None:
            return

        if self.move_prob is not None:
            node_index = graph.linear_index(self.node)
            next_node_index = self.rng.choice(
                len(self.move_prob[node_index]), p=self.move_prob[node_index]
            )
            next_node = graph.grid_index(next_node_index)
            action = node_to_action(src=self.node, dst=next_node)
        else:
            action = self.rng.integers(0, self.action_space_size)
            next_node = action_to_node(node=self.node, action=action)
        self.orientation = action
        self.node = graph.validate_node(self.node, next_node)

    def draw(self, window: Window, visibility: float = 1.0):
        color = list(self.color)
        color[3] = int(visibility * 255)
        border_color = list(Colors.black)
        border_color[3] = int(visibility * 255)
        if self.angle_increment is None:
            window.draw_circle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                radius=0.5,
                color=color,
                border_color=border_color,
                use_transparency=True,
            )
        else:
            window.draw_triangle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                size=0.75,
                orientation=self.orientation * self.angle_increment,
                color=color,
                border_width=1,
                border_color=border_color,
                use_transparency=True,
            )


class Agent(Object):
    def __init__(self, node, color=(0, 0, 255), **kwargs):
        super().__init__(node, color)
        self.orientation = kwargs.get("orientation", 0)
        self.action_space_size = kwargs.get("action_space_size", 0)
        if self.action_space_size > 0:
            self.angle_increment = np.pi * 2.0 / self.action_space_size
        else:
            self.angle_increment = 0
        self.rng = kwargs.get("rng", np.random.default_rng())
        self.step_function = kwargs.get("step_function", self._default_step)

    def copy(self):
        new_agent = self.__class__(self.node, self.color)
        new_agent.orientation = self.orientation
        new_agent.action_space_size = self.action_space_size
        new_agent.angle_increment = self.angle_increment
        new_agent.rng = self.rng
        new_agent.step_function = self.step_function
        return new_agent

    @staticmethod
    def _default_step(graph: GridGraph, node: tuple, action: int):
        return graph.validate_node(node, action_to_node(node, action))

    def step(self, graph: GridGraph, action: int):
        self.orientation = action
        # BUGBUG - expecting the step function to handle the validation, allowing
        #          jumps to nodes that aren't neighbours.
        self.node = self.step_function(graph, self.node, action)

    def draw(self, window: Window, visibility: float = 1.0):
        color = list(self.color)
        color[3] = int(visibility * 255)
        border_color = list(Colors.black)
        border_color[3] = int(visibility * 255)
        if self.angle_increment is None:
            window.draw_circle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                radius=0.5,
                color=color,
                border_width=1,
                border_color=border_color,
                use_transparency=True,
            )
        else:
            window.draw_triangle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                size=0.8,
                orientation=self.orientation * self.angle_increment,
                color=color,
                border_width=1,
                border_color=border_color,
                use_transparency=True,
            )
