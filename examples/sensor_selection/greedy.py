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
    lambda_energy: float | None = None,
    sensor_energy_costs: np.ndarray | None = None,
) -> np.ndarray:
    del rng  # deterministic selector
    del step_index
    del initial_true_state
    num_sensors = int(coverage_matrix.shape[0])
    k = max(0, min(int(max_active_sensors), num_sensors))
    action = np.zeros((num_sensors,), dtype=np.int8)
    if k == 0:
        return action

    predicted_distribution = np.asarray(state_distribution, dtype=np.float64)
    if hmm is not None:
        # One-step lookahead to align selected sensors with where the target
        # will be at the next transition.
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

    # Backward-compatible behavior: if no lambda is provided, pick top-k sensors
    # by expected covered mass.
    if lambda_energy is None:
        scores = coverage_matrix.astype(np.float64) @ predicted_distribution
        selected = np.argsort(-scores, kind="stable")[:k]
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

    covered = np.zeros((coverage_matrix.shape[1],), dtype=bool)
    selected: list[int] = []
    remaining = set(range(num_sensors))

    for _ in range(k):
        best_idx = None
        best_net_gain = -np.inf
        for sensor_idx in remaining:
            new_covered = coverage_matrix[sensor_idx, :] & (~covered)
            marginal_gain = float(np.sum(predicted_distribution[new_covered]))
            net_gain = marginal_gain - lam * float(costs[sensor_idx])
            if net_gain > best_net_gain:
                best_net_gain = net_gain
                best_idx = sensor_idx

        if best_idx is None or best_net_gain <= 0.0:
            break

        selected.append(int(best_idx))
        covered |= coverage_matrix[best_idx, :]
        remaining.remove(best_idx)

    action[selected] = 1
    return action
