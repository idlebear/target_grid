from __future__ import annotations

import numpy as np


def select_sensors(
    state_distribution: np.ndarray,
    coverage_matrix: np.ndarray,
    max_active_sensors: int,
    rng: np.random.Generator,
    *,
    hmm=None,
    step_index: int | None = None,
    initial_true_state: int | None = None,
    planning_horizon: int | None = None,
    mcts_iterations: int | None = None,
    mcts_max_actions: int | None = None,
    lambda_energy: float | None = None,
    sensor_energy_costs: np.ndarray | None = None,
) -> np.ndarray:
    del step_index
    del initial_true_state
    del planning_horizon
    del mcts_iterations
    del mcts_max_actions
    num_sensors = int(coverage_matrix.shape[0])
    k = max(0, min(int(max_active_sensors), num_sensors))
    action = np.zeros((num_sensors,), dtype=np.int8)
    if k == 0:
        return action

    predicted_distribution = np.asarray(state_distribution, dtype=np.float64)
    if hmm is not None:
        predicted_distribution = np.asarray(
            hmm.transition(
                state_distribution=hmm.state_distribution,
                mode_distribution=hmm.mode_distribution,
                steps=1,
            ),
            dtype=np.float64,
        )
    if predicted_distribution.sum() > 0.0:
        predicted_distribution = predicted_distribution / predicted_distribution.sum()

    # Backward-compatible behavior when no lambda is supplied.
    if lambda_energy is None:
        selected = rng.choice(num_sensors, size=k, replace=False)
        action[selected] = 1
        return action

    lam = float(lambda_energy)
    if sensor_energy_costs is None:
        costs = np.ones((num_sensors,), dtype=np.float64)
    else:
        costs = np.asarray(sensor_energy_costs, dtype=np.float64).reshape(-1)
        if costs.shape[0] != num_sensors:
            raise ValueError("sensor_energy_costs length must match num_sensors")
        costs = np.clip(costs, 0.0, np.inf)

    # Thresholding determines how many sensors to activate; selection itself is random.
    # This preserves lambda-driven activity while remaining a random baseline.
    individual_gain = coverage_matrix.astype(np.float64) @ predicted_distribution
    eligible_count = int(np.sum(individual_gain > (lam * costs)))
    num_to_select = max(0, min(k, eligible_count))
    if num_to_select <= 0:
        return action

    selected = rng.choice(num_sensors, size=num_to_select, replace=False)
    action[selected] = 1
    return action
