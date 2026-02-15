from __future__ import annotations

import numpy as np


def _normalize_distribution(p: np.ndarray) -> np.ndarray:
    q = np.asarray(p, dtype=np.float64).reshape(-1)
    s = float(np.sum(q))
    if s <= 0.0:
        return np.zeros_like(q, dtype=np.float64)
    return q / s


def _infer_absorbing_states(transition_matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Mark states whose rows are effectively one-hot self-loops as absorbing.
    """
    p = np.asarray(transition_matrix, dtype=np.float64)
    n = int(p.shape[0])
    absorbing = np.zeros((n,), dtype=bool)
    for s in range(n):
        row = p[s]
        if abs(float(np.sum(row)) - 1.0) > tol:
            continue
        if (1.0 - float(row[s])) > tol:
            continue
        if float(np.sum(np.abs(np.delete(row, s)))) > tol:
            continue
        absorbing[s] = True
    return absorbing


def _build_tracking_split_matrix(
    coverage_matrix: np.ndarray,
    absorbing_mask: np.ndarray,
) -> np.ndarray:
    """
    Build T(i,l) in Eq. 18-style decomposition:
    - T(i,l)=1 for the unique covering sensor in the Section II-A model
    - for overlapping coverage, split equally across covering sensors
    - absorbing states contribute zero tracking cost
    """
    cover = np.asarray(coverage_matrix, dtype=np.float64).T  # [state, sensor]
    counts = np.sum(cover > 0.0, axis=1)
    t = np.zeros_like(cover, dtype=np.float64)
    valid = counts > 0
    t[valid, :] = cover[valid, :] / counts[valid, None]
    t[np.asarray(absorbing_mask, dtype=bool), :] = 0.0
    return t


def _solve_per_sensor_bellman_eq18(
    transition_matrix: np.ndarray,
    tracking_split: np.ndarray,
    wake_costs: np.ndarray,
    absorbing_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Eq. 18-style per-sensor Bellman recursion for all sensors at once.

    Returns:
    - V: value matrix [state, sensor]
    - Q_sleep: state-action value for u_l=0
    - Q_wake: state-action value for u_l=1
    """
    p = np.asarray(transition_matrix, dtype=np.float64)
    t = np.asarray(tracking_split, dtype=np.float64)
    c = np.asarray(wake_costs, dtype=np.float64).reshape(1, -1)
    absorbing = np.asarray(absorbing_mask, dtype=bool).reshape(-1)

    num_states = int(p.shape[0])
    num_sensors = int(t.shape[1])
    v = np.zeros((num_states, num_sensors), dtype=np.float64)

    # Eq. 18 reduces to: V = P V + min(P T, c), where c is wake cost.
    # Solve the transient block exactly for absorbing chains.
    immediate = np.minimum(p @ t, c)
    transient = np.flatnonzero(~absorbing)
    if transient.size > 0:
        q_tt = p[np.ix_(transient, transient)]
        lhs = np.eye(transient.size, dtype=np.float64) - q_tt
        rhs = immediate[transient, :]
        try:
            v_transient = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            v_transient = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        v[transient, :] = v_transient

    q_sleep = (p @ t) + (p @ v)
    q_wake = c + (p @ v)
    if np.any(absorbing):
        q_sleep[absorbing, :] = 0.0
        q_wake[absorbing, :] = 0.0
    return v, q_sleep, q_wake


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
    """
    Q_MDP baseline using Eq. 18 per-sensor Bellman recursion.

    For each sensor l we solve:
      J_l(p)=min_{u_l in {0,1}} sum_i [pP]_i (T(i,l) 1{u_l=0} + c_l 1{u_l=1})
             + sum_i [pP]_i J_l(e_i)

    where T(i,l) is the per-sensor tracking contribution and c_l is
    lambda*energy_cost_l. We then pick sensors with positive expected
    Bellman improvement (sleep minus wake), capped by max_active_sensors.
    """
    del rng
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

    if lambda_energy is None:
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
        predicted_distribution = _normalize_distribution(predicted_distribution)
        score = coverage_matrix.astype(np.float64) @ predicted_distribution
        selected = np.argsort(-score, kind="stable")[:k]
        action[selected] = 1
        return action

    lam = float(lambda_energy)
    if sensor_energy_costs is None:
        base_costs = np.ones((num_sensors,), dtype=np.float64)
    else:
        base_costs = np.asarray(sensor_energy_costs, dtype=np.float64).reshape(-1)
        if base_costs.shape[0] != num_sensors:
            raise ValueError("sensor_energy_costs length must match num_sensors")
        base_costs = np.clip(base_costs, 0.0, np.inf)
    wake_costs = lam * base_costs

    # Transition under the current mode belief.
    if hmm is not None:
        transition_matrix = np.asarray(
            hmm.get_mixed_transition_matrix(exp=1),
            dtype=np.float64,
        )
        belief = np.asarray(hmm.state_distribution, dtype=np.float64)
    else:
        n = int(coverage_matrix.shape[1])
        transition_matrix = np.eye(n, dtype=np.float64)
        belief = np.asarray(state_distribution, dtype=np.float64)

    belief = _normalize_distribution(belief)

    absorbing_mask = _infer_absorbing_states(transition_matrix)
    tracking_split = _build_tracking_split_matrix(
        coverage_matrix=coverage_matrix,
        absorbing_mask=absorbing_mask,
    )
    _, q_sleep, q_wake = _solve_per_sensor_bellman_eq18(
        transition_matrix=transition_matrix,
        tracking_split=tracking_split,
        wake_costs=wake_costs,
        absorbing_mask=absorbing_mask,
    )

    # Positive gain means waking sensor l reduces expected Eq. 18 cost.
    sleep_obj = belief @ q_sleep
    wake_obj = belief @ q_wake
    net_gain = sleep_obj - wake_obj

    positive = np.flatnonzero(net_gain > 0.0)
    if positive.size == 0:
        return action

    if positive.size > k:
        keep = positive[np.argsort(-net_gain[positive], kind="stable")[:k]]
    else:
        keep = positive
    action[keep] = 1
    return action
