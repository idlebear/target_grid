"""
Linear sensor scheduling environment.
"""

from __future__ import annotations

from math import isclose
from typing import Any

import numpy as np

from .constants import DEFAULT_MAX_STEP
from .constants import DEFAULT_SCREEN_HEIGHT, DEFAULT_SCREEN_WIDTH
from .sensor_scheduling_base import SensorSchedulingBaseEnv
from .sensors import (
    compute_linear_coverage_matrix,
    default_linear_sensors,
    parse_sensor_specs,
)


DEFAULT_LINEAR_PARAMETERS = {
    "num_states": 41,
    "num_targets": 1,
    "max_steps": DEFAULT_MAX_STEP,
    "lambda_energy": 1.0,
    "tracking_cost_mode": "unobserved",
    "tracking_reduce": "sum",
    "observation_mode": "discrete",
    "gaussian_sigma": 0.2,
    "transition_matrix": None,
    "transition_offsets": [-3, -2, -1, 0, 1, 2, 3],
    "transition_probabilities": [1 / 7.0] * 7,
    "boundary_behavior": "stay",  # stay | clip
    "absorbing_states": [],
    "sensor_specs": None,
    "sensor_visibility": None,
    "initial_target_states": None,
    "initial_belief": "uniform",
    "true_state_in_info": False,
    "screen_width": DEFAULT_SCREEN_WIDTH,
    "screen_height": DEFAULT_SCREEN_HEIGHT,
}


class SensorSchedulingLinearEnv(SensorSchedulingBaseEnv):
    def __init__(
        self,
        *,
        render_mode: str | None = None,
        seed: int | None = None,
        world_parameters: dict[str, Any] | None = None,
    ):
        params = dict(DEFAULT_LINEAR_PARAMETERS)
        if world_parameters is not None:
            params.update(world_parameters)

        num_states = int(params["num_states"])
        if num_states <= 0:
            raise ValueError("num_states must be >= 1")

        absorbing_states = set(int(v) for v in params.get("absorbing_states", []))
        for s in absorbing_states:
            if s < 0 or s >= num_states:
                raise ValueError(f"absorbing state {s} out of bounds")

        state_coords = np.array([[i, 0] for i in range(num_states)], dtype=np.int32)

        transition_matrix = params.get("transition_matrix", None)
        if transition_matrix is None:
            transition_matrix = self._build_transition_matrix(
                num_states=num_states,
                offsets=params["transition_offsets"],
                probabilities=params["transition_probabilities"],
                boundary_behavior=params.get("boundary_behavior", "stay"),
                absorbing_states=absorbing_states,
            )
        else:
            transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
            if transition_matrix.shape != (num_states, num_states):
                raise ValueError(
                    "transition_matrix must have shape (num_states, num_states)"
                )

        raw_sensor_specs = params.get("sensor_specs", None)
        if raw_sensor_specs is None:
            sensors = default_linear_sensors(num_states)
        else:
            sensors = parse_sensor_specs(raw_sensor_specs, grid_shape=(1, num_states))
        if len(sensors) == 0:
            raise ValueError("at least one sensor is required")

        sensor_visibility = params.get("sensor_visibility", None)
        if sensor_visibility is None:
            coverage = compute_linear_coverage_matrix(sensors, num_states)
        else:
            coverage = np.asarray(sensor_visibility, dtype=bool)
            if coverage.shape != (len(sensors), num_states):
                raise ValueError(
                    "sensor_visibility must have shape (num_sensors, num_states)"
                )

        initial_target_states = params.get("initial_target_states", None)
        if initial_target_states is None and params["num_targets"] == 1:
            # Default to center state for paper-aligned setup.
            initial_target_states = [num_states // 2]

        super().__init__(
            state_coords=state_coords,
            transition_matrix=transition_matrix,
            sensors=sensors,
            coverage_matrix=coverage,
            absorbing_states=absorbing_states,
            num_targets=int(params["num_targets"]),
            max_steps=int(params["max_steps"]),
            lambda_energy=float(params["lambda_energy"]),
            tracking_cost_mode=str(params["tracking_cost_mode"]),
            tracking_reduce=str(params["tracking_reduce"]),
            observation_mode=str(params["observation_mode"]),
            gaussian_sigma=float(params["gaussian_sigma"]),
            initial_target_states=initial_target_states,
            initial_belief=params["initial_belief"],
            true_state_in_info=bool(params["true_state_in_info"]),
            obstacle_grid=np.zeros((1, num_states), dtype=np.int8),
            screen_width=int(params["screen_width"]),
            screen_height=int(params["screen_height"]),
            render_mode=render_mode,
        )

        self._pending_seed = seed

    @staticmethod
    def _build_transition_matrix(
        *,
        num_states: int,
        offsets: list[int],
        probabilities: list[float],
        boundary_behavior: str,
        absorbing_states: set[int],
    ) -> np.ndarray:
        if len(offsets) != len(probabilities):
            raise ValueError("transition_offsets and transition_probabilities mismatch")
        p = np.array(probabilities, dtype=np.float64)
        if np.any(p < 0.0):
            raise ValueError("transition probabilities must be non-negative")
        total = float(np.sum(p))
        if not isclose(total, 1.0, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("transition probabilities must sum to 1")

        T = np.zeros((num_states, num_states), dtype=np.float64)
        for s in range(num_states):
            if s in absorbing_states:
                T[s, s] = 1.0
                continue

            for dx, prob in zip(offsets, p):
                nxt = s + int(dx)
                if 0 <= nxt < num_states:
                    T[s, nxt] += prob
                else:
                    if boundary_behavior == "clip":
                        nxt = min(max(nxt, 0), num_states - 1)
                        T[s, nxt] += prob
                    elif boundary_behavior == "stay":
                        T[s, s] += prob
                    else:
                        raise ValueError(
                            "boundary_behavior must be one of {'stay', 'clip'}"
                        )

        row_sum = T.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        T /= row_sum
        return T
