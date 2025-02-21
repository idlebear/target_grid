"""
This module defines objects that are capable of moving and drawing themselves.
"""

import numpy as np

from window import Window
from graphs import GridGraph

from window import Colours

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
        window.draw_rect(center=(self.node[0] + 0.5, self.node[1] + 0.5), height=1, width=1, colour=colour, use_transparency=True)


class Goal(Object):
    def __init__(self, node, colour=(0, 255, 0), **kwargs):
        super().__init__(node, colour)

    def step(self, graph: GridGraph, action: int):
        pass

    def draw(self, window: Window, visibility: float = 1.0):
        colour = list(self.colour)
        colour[3] = int(visibility * 255)
        window.draw_rect(center=(self.node[0] + 0.5, self.node[1] + 0.5), height=1, width=1, colour=colour, use_transparency=True)


class Target(Object):
    def __init__(self, node, colour=(255, 0, 0), **kwargs):
        super().__init__(node, colour)
        self.orientation = 0
        if "action_space_size" in kwargs:
            self.action_space_size = kwargs["action_space_size"]
            self.angle_increment = np.pi * 2.0 / kwargs["action_space_size"]
        else:
            self.angle_increment = None
            self.action_space_size = 0

        if "rng" in kwargs:
            self.rng = kwargs["rng"]
        else:
            self.rng = np.random.default

        if "action_to_node" in kwargs:
            self.action_to_node = kwargs["action_to_node"]
        else:
            self.action_to_node = None

        if "graph" in kwargs:
            self.graph = kwargs["graph"]
        else:
            self.graph = None

    def step(self):
        if self.action_to_node is not None:
            action = self.rng.integers(0, self.action_space_size)
            self.orientation = action
            next_node = self.action_to_node(node=self.node, action=action)
            self.node = self.graph.validate_node(self.node, next_node)
        else:
            raise ValueError("Target object must have an action_to_node function to step")

    def draw(self, window: Window, visibility: float = 1.0):
        colour = list(self.colour)
        colour[3] = int(visibility * 255)
        border_colour = list(Colours.black)
        border_colour[3] = int(visibility * 255)
        if self.angle_increment == None:
            window.draw_circle(center=(self.node[0] + 0.5, self.node[1] + 0.5), radius=0.5, colour=colour, border_colour=border_colour, use_transparency=True)
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
        self.orientation = 0
        if "action_space_size" in kwargs:
            self.angle_increment = np.pi * 2.0 / kwargs["action_space_size"]
        else:
            self.angle_increment = None

    def step(self, graph: GridGraph, action: int):
        self.orientation = action
        self.node = graph.next_node(self.node, action)

    def draw(self, window: Window, visibility: float = 1.0):
        colour = list(self.colour)
        colour[3] = int(visibility * 255)
        if self.angle_increment == None:
            window.draw_circle(center=(self.node[0] + 0.5, self.node[1] + 0.5), radius=0.5, colour=colour, use_transparency=True)
        else:
            window.draw_equilateral_triangle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                side=0.75,
                orientation=self.orientation * self.angle_increment,
                colour=colour,
                use_transparency=True
            )
