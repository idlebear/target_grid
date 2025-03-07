"""
constants.py -- Constants for the gridworld environment
"""

from enum import Enum


class GridState(Enum):
    EMPTY = 0
    TARGET = 20
    AGENT = 30
    GOAL = 40
    OCCLUDED = 50
    COLLISION = 70
    HAZARD = 80
    WALL = 100
    MAX_VALUE = 100


# Other constants
DEFAULT_HAZARD_COST = 5
DEFAULT_TERMINAL_COST = -10
DEFAULT_STEP_COST = 0.1
DEFAULT_MAX_STEP = 200
DEFAULT_MAX_RETRY = 10

# Display size
DEFAULT_SCREEN_WIDTH = 1000
DEFAULT_SCREEN_HEIGHT = 1000
