from __future__ import annotations

from typing import Iterable
import os

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
import pandas as pd


def _solve_absorbing_linear_system(q: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Solve (I - Q)x = rhs for absorbing Markov chains.

    Falls back to least-squares when numerical singularities are encountered.
    """
    n = int(q.shape[0])
    lhs = np.eye(n, dtype=np.float64) - q
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _auto_lambda_grid(
    transition_matrix: np.ndarray,
    coverage_matrix: np.ndarray,
    absorbing_states: set[int],
) -> np.ndarray:
    """
    Build a compact lambda grid that captures action-threshold changes.
    """
    num_states = int(transition_matrix.shape[0])
    covered_states = np.any(coverage_matrix, axis=0)
    transient = [s for s in range(num_states) if s not in absorbing_states]

    thresholds: list[float] = []
    for s in transient:
        row = transition_matrix[s, covered_states]
        nz = row[row > 0.0]
        thresholds.extend(float(v) for v in nz)

    if not thresholds:
        return np.array([0.0, 1.0], dtype=np.float64)

    unique = np.array(sorted(set(round(v, 12) for v in thresholds)), dtype=np.float64)
    lambdas = [0.0]
    for idx in range(unique.size - 1):
        lambdas.append(float(0.5 * (unique[idx] + unique[idx + 1])))
    lambdas.append(float(unique[-1] + 1e-6))
    return np.array(sorted(set(lambdas)), dtype=np.float64)


def section_iia_observable_after_control_lower_bound(
    *,
    transition_matrix: np.ndarray,
    coverage_matrix: np.ndarray,
    absorbing_states: Iterable[int],
    start_state: int,
    lambda_values: Iterable[float] | None = None,
    start_distribution: Iterable[float] | None = None,
) -> pd.DataFrame:
    """
    Compute the Section II-A style lower bound via the observable-after-control
    relaxation.

    Assumptions:
    - tracking cost is "unobserved" (0 if any active sensor covers state, else 1)
    - target dynamics are action-independent
    - a state in `absorbing_states` ends the episode with zero future cost
    - one-step energy is lambda * (# active sensors), unit per-sensor energy
    """
    t = np.asarray(transition_matrix, dtype=np.float64)
    c = np.asarray(coverage_matrix, dtype=bool)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("transition_matrix must be square")
    if c.ndim != 2 or c.shape[1] != t.shape[0]:
        raise ValueError(
            "coverage_matrix must have shape (num_sensors, num_states)"
        )
    if np.any(np.sum(c.astype(np.int32), axis=0) > 1):
        raise ValueError(
            "observable-after-control lower bound currently assumes "
            "non-overlapping sensor coverage (Section II-A model)"
        )

    num_states = int(t.shape[0])
    if start_state < 0 or start_state >= num_states:
        raise ValueError("start_state out of bounds")
    start_dist = None
    if start_distribution is not None:
        start_dist = np.asarray(list(start_distribution), dtype=np.float64).reshape(-1)
        if start_dist.shape[0] != num_states:
            raise ValueError("start_distribution length must equal num_states")
        start_dist[start_dist < 0.0] = 0.0
        z = float(np.sum(start_dist))
        if z <= 0.0:
            raise ValueError("start_distribution must sum to > 0")
        start_dist /= z

    absorbing = {int(s) for s in absorbing_states}
    transient = [s for s in range(num_states) if s not in absorbing]
    if len(transient) == 0:
        return pd.DataFrame(
            [
                {
                    "policy": "lower_bound_oac",
                    "lower_bound_method": "observable_after_control",
                    "lambda_energy": 0.0,
                    "expected_steps_to_exit": 0.0,
                    "mean_active_sensors_per_step": 0.0,
                    "mean_tracking_error_per_step": 0.0,
                    "mean_energy_cost_per_step": 0.0,
                    "mean_total_cost_per_step": 0.0,
                }
            ]
        )

    covered_states = np.any(c, axis=0)
    transient_idx = np.array(transient, dtype=np.int32)
    q = t[np.ix_(transient_idx, transient_idx)]
    absorbing_idx = np.array(sorted(absorbing), dtype=np.int32)
    start_transient_idx = (
        None if start_state in absorbing else int(transient.index(int(start_state)))
    )
    if start_dist is not None:
        start_transient_dist = start_dist[transient_idx]

    if lambda_values is None:
        lambda_grid = _auto_lambda_grid(t, c, absorbing)
    else:
        lambda_grid = np.array([float(v) for v in lambda_values], dtype=np.float64)
        if lambda_grid.size == 0:
            raise ValueError("lambda_values must be non-empty when provided")
        if np.any(lambda_grid < 0.0):
            raise ValueError("lambda_values must be non-negative")
        lambda_grid = np.array(sorted(set(lambda_grid.tolist())), dtype=np.float64)

    rows: list[dict[str, float | str]] = []
    for lam in lambda_grid:
        immediate_tracking = np.zeros((len(transient),), dtype=np.float64)
        immediate_energy = np.zeros((len(transient),), dtype=np.float64)
        immediate_active = np.zeros((len(transient),), dtype=np.float64)

        for ridx, s in enumerate(transient):
            next_dist = t[s, :]
            # Activate a sensor for next states whose probability mass exceeds
            # the energy penalty threshold.
            active_states = covered_states & (next_dist > lam)
            observed_prob = float(np.sum(next_dist[active_states]))
            absorbing_prob = (
                float(np.sum(next_dist[absorbing_idx]))
                if absorbing_idx.size > 0
                else 0.0
            )
            num_active = float(np.sum(active_states))
            immediate_tracking[ridx] = max(0.0, 1.0 - absorbing_prob - observed_prob)
            immediate_energy[ridx] = float(lam) * num_active
            immediate_active[ridx] = num_active

        expected_tracking = _solve_absorbing_linear_system(q, immediate_tracking)
        expected_energy = _solve_absorbing_linear_system(q, immediate_energy)
        expected_active = _solve_absorbing_linear_system(q, immediate_active)
        expected_steps = _solve_absorbing_linear_system(
            q, np.ones((len(transient),), dtype=np.float64)
        )

        if start_dist is not None:
            steps = float(np.dot(start_transient_dist, expected_steps))
            tracking = float(np.dot(start_transient_dist, expected_tracking))
            energy = float(np.dot(start_transient_dist, expected_energy))
            active = float(np.dot(start_transient_dist, expected_active))
        else:
            if start_transient_idx is None:
                steps = 0.0
                tracking = 0.0
                energy = 0.0
                active = 0.0
            else:
                steps = float(expected_steps[start_transient_idx])
                tracking = float(expected_tracking[start_transient_idx])
                energy = float(expected_energy[start_transient_idx])
                active = float(expected_active[start_transient_idx])

        denom = steps if steps > 0.0 else 1.0
        rows.append(
            {
                "policy": "lower_bound_oac",
                "lower_bound_method": "observable_after_control",
                "lambda_energy": float(lam),
                "expected_steps_to_exit": float(steps),
                "mean_active_sensors_per_step": float(active / denom),
                "mean_tracking_error_per_step": float(tracking / denom),
                "mean_energy_cost_per_step": float(energy / denom),
                "mean_total_cost_per_step": float((tracking + energy) / denom),
            }
        )

    out = pd.DataFrame(rows)
    if lambda_values is None:
        # Keep only Pareto-relevant points for auto-generated lambda grids.
        out = out.sort_values(
            ["mean_active_sensors_per_step", "mean_tracking_error_per_step", "lambda_energy"]
        ).drop_duplicates(
            subset=["mean_active_sensors_per_step", "mean_tracking_error_per_step"],
            keep="first",
        )
    else:
        out = out.sort_values(["lambda_energy"])
    return out.reset_index(drop=True)


def _sum_top_k(values: np.ndarray, k: int) -> float:
    if k <= 0 or values.size == 0:
        return 0.0
    k_eff = min(int(k), int(values.size))
    if k_eff == int(values.size):
        return float(np.sum(values))
    # partial selection is faster than sorting full array
    idx = np.argpartition(values, -k_eff)[-k_eff:]
    return float(np.sum(values[idx]))


def section_iia_observable_after_control_lower_bound_for_k(
    *,
    transition_matrix: np.ndarray,
    coverage_matrix: np.ndarray,
    absorbing_states: Iterable[int],
    start_state: int,
    k_values: Iterable[int],
    lambda_energy: float,
    start_distribution: Iterable[float] | None = None,
) -> pd.DataFrame:
    """
    Section II-A observable-after-control lower bound parameterized by the same
    active-sensor budget k used in experiments.
    """
    t = np.asarray(transition_matrix, dtype=np.float64)
    c = np.asarray(coverage_matrix, dtype=bool)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("transition_matrix must be square")
    if c.ndim != 2 or c.shape[1] != t.shape[0]:
        raise ValueError(
            "coverage_matrix must have shape (num_sensors, num_states)"
        )
    if np.any(np.sum(c.astype(np.int32), axis=0) > 1):
        raise ValueError(
            "observable-after-control lower bound currently assumes "
            "non-overlapping sensor coverage (Section II-A model)"
        )

    num_states = int(t.shape[0])
    if start_state < 0 or start_state >= num_states:
        raise ValueError("start_state out of bounds")
    start_dist = None
    if start_distribution is not None:
        start_dist = np.asarray(list(start_distribution), dtype=np.float64).reshape(-1)
        if start_dist.shape[0] != num_states:
            raise ValueError("start_distribution length must equal num_states")
        start_dist[start_dist < 0.0] = 0.0
        z = float(np.sum(start_dist))
        if z <= 0.0:
            raise ValueError("start_distribution must sum to > 0")
        start_dist /= z

    k_list = sorted(set(int(k) for k in k_values))
    if len(k_list) == 0:
        raise ValueError("k_values must be non-empty")
    if any(k < 0 for k in k_list):
        raise ValueError("k_values must be non-negative")

    num_sensors = int(c.shape[0])
    covered_states = np.any(c, axis=0)
    covered_idx = np.flatnonzero(covered_states)
    absorbing_idx = np.array(sorted(absorbing), dtype=np.int32)

    absorbing = {int(s) for s in absorbing_states}
    transient = [s for s in range(num_states) if s not in absorbing]
    if len(transient) == 0:
        rows = []
        for k in k_list:
            k_eff = int(max(0, min(k, num_sensors)))
            rows.append(
                {
                    "policy": "lower_bound_oac",
                    "lower_bound_method": "observable_after_control",
                    "max_active_sensors": int(k_eff),
                    "lambda_energy": float(lambda_energy),
                    "expected_steps_to_exit": 0.0,
                    "mean_active_sensors_per_step": 0.0,
                    "mean_tracking_error_per_step": 0.0,
                    "mean_energy_cost_per_step": 0.0,
                    "mean_total_cost_per_step": 0.0,
                }
            )
        return pd.DataFrame(rows)

    transient_idx = np.array(transient, dtype=np.int32)
    q = t[np.ix_(transient_idx, transient_idx)]
    start_transient_idx = (
        None if start_state in absorbing else int(transient.index(int(start_state)))
    )
    if start_dist is not None:
        start_transient_dist = start_dist[transient_idx]

    rows: list[dict[str, float | int | str]] = []
    for k in k_list:
        k_eff = int(max(0, min(k, num_sensors)))
        immediate_tracking = np.zeros((len(transient),), dtype=np.float64)
        immediate_active = np.full((len(transient),), float(k_eff), dtype=np.float64)
        immediate_energy = np.full(
            (len(transient),), float(lambda_energy) * float(k_eff), dtype=np.float64
        )

        for ridx, s in enumerate(transient):
            p_coverable = t[s, covered_idx]
            observed_prob = _sum_top_k(p_coverable, k_eff)
            absorbing_prob = (
                float(np.sum(t[s, absorbing_idx]))
                if absorbing_idx.size > 0
                else 0.0
            )
            immediate_tracking[ridx] = max(0.0, 1.0 - absorbing_prob - observed_prob)

        expected_tracking = _solve_absorbing_linear_system(q, immediate_tracking)
        expected_energy = _solve_absorbing_linear_system(q, immediate_energy)
        expected_active = _solve_absorbing_linear_system(q, immediate_active)
        expected_steps = _solve_absorbing_linear_system(
            q, np.ones((len(transient),), dtype=np.float64)
        )

        if start_dist is not None:
            steps = float(np.dot(start_transient_dist, expected_steps))
            tracking = float(np.dot(start_transient_dist, expected_tracking))
            energy = float(np.dot(start_transient_dist, expected_energy))
            active = float(np.dot(start_transient_dist, expected_active))
        else:
            if start_transient_idx is None:
                steps = 0.0
                tracking = 0.0
                energy = 0.0
                active = 0.0
            else:
                steps = float(expected_steps[start_transient_idx])
                tracking = float(expected_tracking[start_transient_idx])
                energy = float(expected_energy[start_transient_idx])
                active = float(expected_active[start_transient_idx])

        denom = steps if steps > 0.0 else 1.0
        rows.append(
            {
                "policy": "lower_bound_oac",
                "lower_bound_method": "observable_after_control",
                "max_active_sensors": int(k_eff),
                "lambda_energy": float(lambda_energy),
                "expected_steps_to_exit": float(steps),
                "mean_active_sensors_per_step": float(active / denom),
                "mean_tracking_error_per_step": float(tracking / denom),
                "mean_energy_cost_per_step": float(energy / denom),
                "mean_total_cost_per_step": float((tracking + energy) / denom),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["max_active_sensors"]).reset_index(drop=True)
    return out
