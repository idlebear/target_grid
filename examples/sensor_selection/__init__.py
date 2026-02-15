from __future__ import annotations

from typing import Callable

import numpy as np

from .greedy import select_sensors as greedy_select_sensors
from .mcts_oce import select_sensors as mcts_oce_select_sensors
from .mcts_pomdp import select_sensors as mcts_pomdp_select_sensors
from .q_mdp import select_sensors as q_mdp_select_sensors
from .random_selector import select_sensors as random_select_sensors


# Selector implementations may accept additional keyword arguments
# (e.g., hmm, step_index, initial_true_state) for advanced policies.
SelectorFn = Callable[..., np.ndarray]


def get_selector(name: str) -> SelectorFn:
    key = str(name).strip().lower()
    if key == "greedy":
        return greedy_select_sensors
    if key in {"q_mdp", "qmdp"}:
        return q_mdp_select_sensors
    if key in {"mcts_oce", "mcts-oce", "mcts_oced", "oce_mcts", "oce-mcts"}:
        return mcts_oce_select_sensors
    if key in {"mcts_pomdp", "mcts-pomdp", "pomdp_mcts", "pomdp-mcts"}:
        return mcts_pomdp_select_sensors
    if key == "random":
        return random_select_sensors
    raise ValueError(
        "unknown selector "
        f"'{name}', expected 'greedy', 'q_mdp', 'mcts_oce', 'mcts_pomdp', or 'random'"
    )
