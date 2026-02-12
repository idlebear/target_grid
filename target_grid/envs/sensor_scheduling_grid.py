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
    "discrete_sensor_model": "simple",  # simple | probabilistic
    "probabilistic_observation_correct_prob": 0.8,
    "gaussian_sigma": 0.2,
    "transition_matrix": None,
    "boundary_behavior": "stay",  # stay | clip | exit
    "transition_deltas": None,
    "transition_probabilities": None,
    "move_diagonal": False,
    "allow_stay": True,
    "absorbing_states": [],
    "sensor_specs": None,
    "sensor_visibility": None,
    "initial_target_states": None,
    "sample_initial_state_from_belief": False,
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
        network_state_coords = np.array(
            [[int(x), int(y)] for y, x in free_cells], dtype=np.int32
        )
        num_network_states = int(network_state_coords.shape[0])
        coord_to_state = {
            (int(x), int(y)): idx for idx, (x, y) in enumerate(network_state_coords)
        }

        rows, cols = grid_data.shape

        boundary_behavior = str(params.get("boundary_behavior", "stay"))
        if boundary_behavior not in {"stay", "clip", "exit"}:
            raise ValueError("boundary_behavior must be one of {'stay', 'clip', 'exit'}")
        add_exit_state = boundary_behavior == "exit"
        num_states = num_network_states + (1 if add_exit_state else 0)
        exit_state = num_states - 1 if add_exit_state else None

        absorbing_states = self._coerce_absorbing_states(
            raw=params.get("absorbing_states", []),
            coord_to_state=coord_to_state,
            num_network_states=num_network_states,
            add_exit_state=add_exit_state,
            exit_state=exit_state,
        )
        if add_exit_state:
            absorbing_states.add(int(exit_state))

        state_coords_list = network_state_coords.tolist()
        if add_exit_state:
            # External state intentionally outside the grid to represent "left network".
            state_coords_list.append([cols, rows])
        state_coords = np.array(state_coords_list, dtype=np.int32)

        transition_matrix = params.get("transition_matrix", None)
        if transition_matrix is None:
            deltas, probs = self._parse_transition_model(
                transition_deltas=params.get("transition_deltas", None),
                transition_probabilities=params.get("transition_probabilities", None),
                move_diagonal=bool(params.get("move_diagonal", False)),
                allow_stay=bool(params.get("allow_stay", True)),
            )
            transition_matrix = self._build_neighbor_transition_matrix(
                rows=rows,
                cols=cols,
                state_coords=network_state_coords,
                coord_to_state=coord_to_state,
                deltas=deltas,
                probabilities=probs,
                boundary_behavior=boundary_behavior,
                absorbing_states=absorbing_states,
                add_exit_state=add_exit_state,
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
                state_coords=network_state_coords,
                obstacle_grid=grid_data,
            )
        else:
            coverage = np.asarray(sensor_visibility, dtype=bool)
            if coverage.shape not in {
                (len(sensors), num_network_states),
                (len(sensors), num_states),
            }:
                raise ValueError(
                    "sensor_visibility must have shape "
                    "(num_sensors, num_states) or "
                    "(num_sensors, num_states+1 when boundary_behavior='exit')"
                )
        if add_exit_state and coverage.shape[1] == num_network_states:
            coverage = np.concatenate(
                [coverage, np.zeros((len(sensors), 1), dtype=bool)], axis=1
            )

        initial_target_states = params.get("initial_target_states", None)
        if initial_target_states is not None:
            initial_target_states = [
                self._coerce_start_state(
                    v,
                    coord_to_state,
                    num_network_states,
                    add_exit_state=add_exit_state,
                    exit_state=exit_state,
                )
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
            discrete_sensor_model=str(
                params.get("discrete_sensor_model", "simple")
            ),
            probabilistic_observation_correct_prob=float(
                params.get("probabilistic_observation_correct_prob", 0.8)
            ),
            gaussian_sigma=float(params["gaussian_sigma"]),
            initial_target_states=initial_target_states,
            initial_belief=params["initial_belief"],
            sample_initial_state_from_belief=bool(
                params.get("sample_initial_state_from_belief", False)
            ),
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
        num_network_states: int,
        *,
        add_exit_state: bool,
        exit_state: int | None,
    ) -> int:
        if isinstance(value, str) and value == "exit":
            if not add_exit_state:
                raise ValueError("state 'exit' requires boundary_behavior='exit'")
            return int(exit_state)
        if isinstance(value, (int, np.integer)):
            idx = int(value)
            max_states = num_network_states + (1 if add_exit_state else 0)
            if idx < 0 or idx >= max_states:
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
        num_network_states: int,
        add_exit_state: bool,
        exit_state: int | None,
    ) -> set[int]:
        out: set[int] = set()
        for value in raw:
            out.add(
                cls._coerce_start_state(
                    value,
                    coord_to_state,
                    num_network_states,
                    add_exit_state=add_exit_state,
                    exit_state=exit_state,
                )
            )
        return out

    @staticmethod
    def _parse_transition_model(
        *,
        transition_deltas: list[list[int]] | list[tuple[int, int]] | None,
        transition_probabilities: list[float] | None,
        move_diagonal: bool,
        allow_stay: bool,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        if transition_deltas is None:
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
            if allow_stay:
                deltas.append((0, 0))
        else:
            deltas = []
            for delta in transition_deltas:
                if len(delta) != 2:
                    raise ValueError("each transition delta must have length 2")
                deltas.append((int(delta[0]), int(delta[1])))
            if len(deltas) == 0:
                raise ValueError("transition_deltas must not be empty")

        if transition_probabilities is None:
            probs = np.full((len(deltas),), 1.0 / float(len(deltas)), dtype=np.float64)
        else:
            probs = np.asarray(transition_probabilities, dtype=np.float64)
            if probs.shape != (len(deltas),):
                raise ValueError(
                    "transition_probabilities length must match transition_deltas"
                )
            if np.any(probs < 0.0):
                raise ValueError("transition_probabilities must be non-negative")
            s = float(np.sum(probs))
            if not np.isclose(s, 1.0, atol=1e-8):
                raise ValueError("transition_probabilities must sum to 1")

        return deltas, probs

    @staticmethod
    def _build_neighbor_transition_matrix(
        *,
        rows: int,
        cols: int,
        state_coords: np.ndarray,
        coord_to_state: dict[tuple[int, int], int],
        deltas: list[tuple[int, int]],
        probabilities: np.ndarray,
        boundary_behavior: str,
        absorbing_states: set[int],
        add_exit_state: bool,
    ) -> np.ndarray:
        num_network_states = state_coords.shape[0]
        num_states = num_network_states + (1 if add_exit_state else 0)
        exit_state = num_states - 1 if add_exit_state else None
        T = np.zeros((num_states, num_states), dtype=np.float64)

        for s in range(num_states):
            if s in absorbing_states:
                T[s, s] = 1.0
                continue

            x, y = int(state_coords[s, 0]), int(state_coords[s, 1])
            for (dx, dy), p in zip(deltas, probabilities):
                tx, ty = x + dx, y + dy
                in_bounds = 0 <= tx < cols and 0 <= ty < rows

                if in_bounds:
                    nxt = (tx, ty)
                    if nxt in coord_to_state:
                        T[s, coord_to_state[nxt]] += p
                    else:
                        # Obstacle cells are non-traversable.
                        T[s, s] += p
                    continue

                # Out-of-bounds => crossed an edge.
                if boundary_behavior == "exit":
                    T[s, exit_state] += p
                elif boundary_behavior == "clip":
                    cx = min(max(tx, 0), cols - 1)
                    cy = min(max(ty, 0), rows - 1)
                    clipped = (cx, cy)
                    if clipped in coord_to_state:
                        T[s, coord_to_state[clipped]] += p
                    else:
                        # If clipped location is obstacle, remain in place.
                        T[s, s] += p
                elif boundary_behavior == "stay":
                    T[s, s] += p
                else:
                    raise ValueError(
                        "boundary_behavior must be one of {'stay', 'clip', 'exit'}"
                    )

        row_sum = T.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        T /= row_sum
        return T
