"""
Definition of the Actions enum and utility functions to convert between actions
and directions.
"""

from enum import Enum
import numpy as np


class Actions(Enum):
    none = 0

    # Basic 8-directional actions
    north = 1
    north_east = 2
    east = 3
    south_east = 4
    south = 5
    south_west = 6
    west = 7
    north_west = 8

    # Extended actions for higher speed
    north_north = 28
    north_north_east = 13
    north_north_east_east = 14
    north_east_east = 15
    east_east = 16
    south_east_east = 17
    south_south_east_east = 18
    south_south_east = 19
    south_south = 20
    south_south_west = 21
    south_south_west_west = 22
    south_west_west = 23
    west_west = 24
    north_west_west = 25
    north_north_west_west = 26
    north_north_west = 27
    action_space_size = 9


action_to_direction = {
    Actions.none.value: np.array([0, 0]),
    Actions.north.value: np.array([1, 0]),
    Actions.north_east.value: np.array([1, 1]),
    Actions.east.value: np.array([0, 1]),
    Actions.south_east.value: np.array([-1, 1]),
    Actions.south.value: np.array([-1, 0]),
    Actions.south_west.value: np.array([-1, -1]),
    Actions.west.value: np.array([0, -1]),
    Actions.north_west.value: np.array([1, -1]),
    Actions.north_north.value: np.array([2, 0]),
    Actions.north_north_east.value: np.array([2, 1]),
    Actions.north_north_east_east.value: np.array([2, 2]),
    Actions.north_east_east.value: np.array([1, 2]),
    Actions.east_east.value: np.array([0, 2]),
    Actions.south_east_east.value: np.array([-1, 2]),
    Actions.south_south_east_east.value: np.array([-2, 2]),
    Actions.south_south_east.value: np.array([-2, 1]),
    Actions.south_south.value: np.array([-2, 0]),
    Actions.south_south_west.value: np.array([-2, -1]),
    Actions.south_south_west_west.value: np.array([-2, -2]),
    Actions.south_west_west.value: np.array([-1, -2]),
    Actions.west_west.value: np.array([0, -2]),
    Actions.north_west_west.value: np.array([1, -2]),
    Actions.north_north_west_west.value: np.array([2, -2]),
    Actions.north_north_west.value: np.array([2, -1]),
}


def direction_to_action(direction):
    _direction_to_action = {
        (0, 0): Actions.none.value,
        (1, 0): Actions.north.value,
        (1, 1): Actions.north_east.value,
        (0, 1): Actions.east.value,
        (-1, 1): Actions.south_east.value,
        (-1, 0): Actions.south.value,
        (-1, -1): Actions.south_west.value,
        (0, -1): Actions.west.value,
        (1, -1): Actions.north_west.value,
        (2, 0): Actions.north_north.value,
        (2, 1): Actions.north_north_east.value,
        (2, 2): Actions.north_north_east_east.value,
        (1, 2): Actions.north_east_east.value,
        (0, 2): Actions.east_east.value,
        (-1, 2): Actions.south_east_east.value,
        (-2, 2): Actions.south_south_east_east.value,
        (-2, 1): Actions.south_south_east.value,
        (-2, 0): Actions.south_south.value,
        (-2, -1): Actions.south_south_west.value,
        (-2, -2): Actions.south_south_west_west.value,
        (-1, -2): Actions.south_west_west.value,
        (0, -2): Actions.west_west.value,
        (1, -2): Actions.north_west_west.value,
        (2, -2): Actions.north_north_west_west.value,
        (2, -1): Actions.north_north_west.value,
    }
    return _direction_to_action[tuple(direction)]


def action_to_node(node, action):
    return tuple(np.add(node, action_to_direction[action]))


def node_to_action(src, dst):
    direction = np.subtract(dst, src)
    return direction_to_action(tuple(direction))


def random_action(rng):
    return rng.choice(Actions.action_space_size.value)
