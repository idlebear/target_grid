"""
This module defines objects that are capable of moving and drawing themselves.
"""

import numpy as np

from .actions import Actions, action_to_node, node_to_action
from .graphs import GridGraph
from .window import Window, Colours


class Object:
    def __init__(self, node, colour, **kwargs):
        self.node = node
        self.colour = colour
        # Ensure the colour has an alpha channel
        if len(self.colour) != 4:
            self.colour = tuple(list(self.colour)[:3] + [255])

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
        # if visibility < 1.0:
        #     colour[3] = int(visibility * 127)
        # else:
        #     colour[3] = 255
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
        line_colour = list(Colours.black)
        if visibility < 1.0:
            colour[3] = int(visibility * 127)
        else:
            colour[3] = 255
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
        line_colour = list(Colours.black)
        if visibility < 1.0:
            colour[3] = int(visibility * 127)
        else:
            colour[3] = 255
            line_colour[3] = int(visibility * 255)
        window.draw_rect(
            center=(self.node[0] + 0.5, self.node[1] + 0.5),
            height=1,
            width=1,
            colour=colour,
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
            colour=line_colour,
            width=5,
            use_transparency=True,
        )


class Target(Object):
    def __init__(self, node, colour=(255, 0, 0), **kwargs):
        super().__init__(node, colour)
        self.orientation = kwargs.get("orientation", 0)
        self.action_space_size = Actions.action_space_size.value
        if self.action_space_size > 0:
            self.angle_increment = np.pi * 2.0 / self.action_space_size
        else:
            self.angle_increment = None
        self.rng = kwargs.get("rng", np.random.default_rng())
        self.move_prob = kwargs.get("move_prob", None)

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

    @staticmethod
    def _default_step(graph: GridGraph, node: tuple, action: int):
        return graph.validate_node(node, action_to_node(node, action))

    def step(self, graph: GridGraph, action: int):
        self.orientation = action
        # BUGBUG - expecting the step function to handle the validation, allowing
        #          jumps to nodes that aren't neighbours.
        self.node = self.step_function(graph, self.node, action)

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
                border_width=1,
                border_colour=border_colour,
                use_transparency=True,
            )
        else:
            window.draw_triangle(
                center=(self.node[0] + 0.5, self.node[1] + 0.5),
                size=0.8,
                orientation=self.orientation * self.angle_increment,
                colour=colour,
                border_width=1,
                border_colour=border_colour,
                use_transparency=True,
            )
