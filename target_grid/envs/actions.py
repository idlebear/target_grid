"""
Definition of the Actions enum and utility functions to convert between actions
and directions.
"""

from enum import Enum
import numpy as np


class Actions(Enum):
    north = 0
    north_east = 1
    east = 2
    south_east = 3
    south = 4
    south_west = 5
    west = 6
    north_west = 7
    none = 8
    action_space_size = 9


action_to_direction = {
    Actions.north.value: np.array([1, 0]),
    Actions.north_east.value: np.array([1, 1]),
    Actions.east.value: np.array([0, 1]),
    Actions.south_east.value: np.array([-1, 1]),
    Actions.south.value: np.array([-1, 0]),
    Actions.south_west.value: np.array([-1, -1]),
    Actions.west.value: np.array([0, -1]),
    Actions.north_west.value: np.array([1, -1]),
    Actions.none.value: np.array([0, 0]),
}


def direction_to_action(direction):
    _direction_to_action = {
        (1, 0): Actions.north.value,
        (1, 1): Actions.north_east.value,
        (0, 1): Actions.east.value,
        (-1, 1): Actions.south_east.value,
        (-1, 0): Actions.south.value,
        (-1, -1): Actions.south_west.value,
        (0, -1): Actions.west.value,
        (1, -1): Actions.north_west.value,
        (0, 0): Actions.none.value,
    }
    return _direction_to_action[tuple(direction)]


def action_to_node(node, action):
    return tuple(np.add(node, action_to_direction[action]))


def node_to_action(src, dst):
    direction = np.subtract(dst, src)
    return direction_to_action(tuple(direction))


def random_action(rng):
    return rng.choice(Actions.action_space_size.value)
