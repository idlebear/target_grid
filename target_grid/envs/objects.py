"""
This module defines objects that are capable of moving and drawing themselves.
"""

import numpy as np

from .window import Window, Colours
from .graphs import GridGraph


class Object:
    def __init__(self, node, colour, **kwargs):
        self.node = node
        self.colour = colour

    def step(self, graph: GridGraph, action: int):
        raise NotImplementedError

    def draw(self, window: Window):
        raise NotImplementedError


class Wall(Object):
    def __init__(self, node, colour=(0, 0, 0), **kwargs):
        super().__init__(node, colour)

    def step(self, graph: GridGraph, action: int):
        pass

    def draw(self, window: Window, visibility: float = 1.0):
        colour = list(self.colour)
        colour[3] = int(visibility * 255)
        window.draw_rect(
            center=(self.node[0] + 0.5, self.node[1] + 0.5),
            height=1,
            width=1,
            colour=colour,
            use_transparency=True,
        )


class Hazard(Object):
    def __init__(self, node, colour=(0, 0, 0), **kwargs):
        super().__init__(node, colour)

    def step(self, graph: GridGraph, action: int):
        pass

    def draw(self, window: Window, visibility: float = 1.0):
        colour = list(self.colour)
        colour[3] = int(visibility * 255)
        line_colour = list(Colours.black)
        line_colour[3] = int(visibility * 255)
        window.draw_rect(
            center=(self.node[0] + 0.5, self.node[1] + 0.5),
            height=1,
            width=1,
            colour=colour,
            use_transparency=True,
        )
        # Draw an x through the hazard
        window.draw_line(
            start=(self.node[0] + 0.25, self.node[1] + 0.25),
            end=(self.node[0] + 0.75, self.node[1] + 0.75),
            colour=line_colour,
            width=5,
            use_transparency=True,
        )
        window.draw_line(
            start=(self.node[0] + 0.75, self.node[1] + 0.25),
            end=(self.node[0] + 0.25, self.node[1] + 0.75),
            colour=line_colour,
            width=5,
            use_transparency=True,
        )


class Goal(Object):
    def __init__(self, node, colour=(0, 255, 0), **kwargs):
        super().__init__(node, colour)

    def step(self, graph: GridGraph, action: int):
        pass

    def draw(self, window: Window, visibility: float = 1.0):
        colour = list(self.colour)
        colour[3] = int(visibility * 255)
        window.draw_rect(
            center=(self.node[0] + 0.5, self.node[1] + 0.5),
            height=1,
            width=1,
            colour=colour,
            use_transparency=True,
        )


class Target(Object):
    def __init__(self, node, colour=(255, 0, 0), **kwargs):
        super().__init__(node, colour)
        self.orientation = kwargs.get("orientation", 0)
        self.action_space_size = kwargs.get("action_space_size", 0)
        if self.action_space_size > 0:
            self.angle_increment = np.pi * 2.0 / kwargs["action_space_size"]
        else:
            self.angle_increment = None
        self.rng = kwargs.get("rng", np.random.default_rng())
        self.action_to_node = kwargs.get("action_to_node", None)
        self.graph = kwargs.get("graph", None)
        self.move_prob = kwargs.get("move_prob", None)

    def step(self):
        if self.action_to_node is not None:
            if self.move_prob is not None:
                # move according to the markov chain
                # action = ...
                pass
            else:
                action = self.rng.integers(0, self.action_space_size)
            self.orientation = action
            next_node = self.action_to_node(node=self.node, action=action)
            self.node = self.graph.validate_node(self.node, next_node)
        else:
            raise ValueError(
                "Target object must have an action_to_node function to step"
            )

    def draw(self, window: Window, visibility: float = 1.0):
        colour = list(self.colour)
        colour[3] = int(visibility * 255)
        border_colour = list(Colours.black)
        border_colour[3] = int(visibility * 255)
        if self.angle_increment is None:
            window.draw_circle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                radius=0.5,
                colour=colour,
                border_colour=border_colour,
                use_transparency=True,
            )
        else:
            window.draw_triangle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                size=0.75,
                orientation=self.orientation * self.angle_increment,
                colour=colour,
                border_width=1,
                border_colour=border_colour,
                use_transparency=True,
            )


class Agent(Object):
    def __init__(self, node, colour=(0, 0, 255), **kwargs):
        super().__init__(node, colour)
        self.orientation = kwargs.get("orientation", 0)
        self.action_space_size = kwargs.get("action_space_size", 0)
        if self.action_space_size > 0:
            self.angle_increment = np.pi * 2.0 / self.action_space_size
        else:
            self.angle_increment = 0
        self.rng = kwargs.get("rng", np.random.default_rng())
        self.step_function = kwargs.get("step_function", self._default_step)

    def _default_step(self, graph: GridGraph, action: int):
        self.orientation = action
        self.node = graph.next_node(self.node, action)

    def step(self, graph: GridGraph, action: int):
        self.step_function(graph, action)

    def draw(self, window: Window, visibility: float = 1.0):
        colour = list(self.colour)
        colour[3] = int(visibility * 255)
        if self.angle_increment is None:
            window.draw_circle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                radius=0.5,
                colour=colour,
                use_transparency=True,
            )
        else:
            window.draw_equilateral_triangle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                side=0.75,
                orientation=self.orientation * self.angle_increment,
                colour=colour,
                use_transparency=True,
            )
