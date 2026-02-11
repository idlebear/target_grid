"""
2-D grid sensor scheduling environment with obstacle-aware transitions.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .constants import DEFAULT_MAX_STEP
from .constants import DEFAULT_SCREEN_HEIGHT, DEFAULT_SCREEN_WIDTH
from .sensor_scheduling_base import SensorSchedulingBaseEnv
from .sensors import (
    compute_grid_coverage_matrix,
    default_grid_sensors,
    parse_sensor_specs,
)


DEFAULT_GRID_PARAMETERS = {
    "grid_data": None,
    "size": 10,
    "num_targets": 1,
    "max_steps": DEFAULT_MAX_STEP,
    "lambda_energy": 1.0,
    "tracking_cost_mode": "unobserved",
    "tracking_reduce": "sum",
    "observation_mode": "discrete",
    "gaussian_sigma": 0.2,
    "transition_matrix": None,
    "move_diagonal": False,
    "allow_stay": True,
    "absorbing_states": [],
    "sensor_specs": None,
    "sensor_visibility": None,
    "initial_target_states": None,
    "initial_belief": "uniform",
    "true_state_in_info": False,
    "screen_width": DEFAULT_SCREEN_WIDTH,
    "screen_height": DEFAULT_SCREEN_HEIGHT,
}


class SensorSchedulingGridEnv(SensorSchedulingBaseEnv):
    def __init__(
        self,
        *,
        render_mode: str | None = None,
        seed: int | None = None,
        world_parameters: dict[str, Any] | None = None,
    ):
        params = dict(DEFAULT_GRID_PARAMETERS)
        if world_parameters is not None:
            params.update(world_parameters)

        grid_data = params.get("grid_data", None)
        if grid_data is None:
            size = int(params.get("size", 10))
            grid_data = np.zeros((size, size), dtype=np.int8)
        else:
            grid_data = np.asarray(grid_data, dtype=np.int8).copy()

        if grid_data.ndim != 2:
            raise ValueError("grid_data must be a 2-D array")
        if np.any((grid_data != 0) & (grid_data != 1)):
            raise ValueError("grid_data must contain only 0 (free) and 1 (obstacle)")

        free_cells = np.argwhere(grid_data == 0)
        if free_cells.shape[0] == 0:
            raise ValueError("grid contains no free traversable cells")
        # convert to (x, y)
        state_coords = np.array([[int(x), int(y)] for y, x in free_cells], dtype=np.int32)
        num_states = int(state_coords.shape[0])
        coord_to_state = {(int(x), int(y)): idx for idx, (x, y) in enumerate(state_coords)}

        absorbing_states = self._coerce_absorbing_states(
            raw=params.get("absorbing_states", []),
            coord_to_state=coord_to_state,
            num_states=num_states,
        )

        transition_matrix = params.get("transition_matrix", None)
        if transition_matrix is None:
            transition_matrix = self._build_neighbor_transition_matrix(
                state_coords=state_coords,
                coord_to_state=coord_to_state,
                move_diagonal=bool(params.get("move_diagonal", False)),
                allow_stay=bool(params.get("allow_stay", True)),
                absorbing_states=absorbing_states,
            )
        else:
            transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
            if transition_matrix.shape != (num_states, num_states):
                raise ValueError(
                    "transition_matrix must have shape (num_states, num_states)"
                )
            # Enforce absorbing rows.
            for s in absorbing_states:
                transition_matrix[s, :] = 0.0
                transition_matrix[s, s] = 1.0

        raw_sensor_specs = params.get("sensor_specs", None)
        if raw_sensor_specs is None:
            sensors = default_grid_sensors(grid_data)
        else:
            sensors = parse_sensor_specs(
                raw_sensor_specs,
                grid_shape=grid_data.shape,
                obstacle_grid=grid_data,
            )
        if len(sensors) == 0:
            raise ValueError("at least one sensor is required")

        sensor_visibility = params.get("sensor_visibility", None)
        if sensor_visibility is None:
            coverage = compute_grid_coverage_matrix(
                sensors=sensors,
                state_coords=state_coords,
                obstacle_grid=grid_data,
            )
        else:
            coverage = np.asarray(sensor_visibility, dtype=bool)
            if coverage.shape != (len(sensors), num_states):
                raise ValueError(
                    "sensor_visibility must have shape (num_sensors, num_states)"
                )

        initial_target_states = params.get("initial_target_states", None)
        if initial_target_states is not None:
            initial_target_states = [
                self._coerce_start_state(v, coord_to_state, num_states)
                for v in initial_target_states
            ]

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
            obstacle_grid=grid_data,
            screen_width=int(params["screen_width"]),
            screen_height=int(params["screen_height"]),
            render_mode=render_mode,
        )

        self._coord_to_state = coord_to_state
        self._pending_seed = seed

    @staticmethod
    def _coerce_start_state(
        value: Any,
        coord_to_state: dict[tuple[int, int], int],
        num_states: int,
    ) -> int:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
            if idx < 0 or idx >= num_states:
                raise ValueError(f"start state index {idx} out of bounds")
            return idx
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (tuple, list)) and len(value) == 2:
            key = (int(value[0]), int(value[1]))
            if key not in coord_to_state:
                raise ValueError(f"start state {key} is not a valid free cell")
            return int(coord_to_state[key])
        raise TypeError(f"unsupported start state type: {type(value).__name__}")

    @classmethod
    def _coerce_absorbing_states(
        cls,
        raw: Iterable[Any],
        coord_to_state: dict[tuple[int, int], int],
        num_states: int,
    ) -> set[int]:
        out: set[int] = set()
        for value in raw:
            out.add(cls._coerce_start_state(value, coord_to_state, num_states))
        return out

    @staticmethod
    def _build_neighbor_transition_matrix(
        *,
        state_coords: np.ndarray,
        coord_to_state: dict[tuple[int, int], int],
        move_diagonal: bool,
        allow_stay: bool,
        absorbing_states: set[int],
    ) -> np.ndarray:
        num_states = state_coords.shape[0]
        T = np.zeros((num_states, num_states), dtype=np.float64)

        if move_diagonal:
            deltas = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        else:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for s in range(num_states):
            if s in absorbing_states:
                T[s, s] = 1.0
                continue

            x, y = int(state_coords[s, 0]), int(state_coords[s, 1])
            next_states: list[int] = []
            for dx, dy in deltas:
                nxt = (x + dx, y + dy)
                if nxt in coord_to_state:
                    next_states.append(coord_to_state[nxt])
            if allow_stay or len(next_states) == 0:
                next_states.append(s)

            p = 1.0 / float(len(next_states))
            for nxt in next_states:
                T[s, nxt] += p

        row_sum = T.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        T /= row_sum
        return T
