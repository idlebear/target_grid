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
    "discrete_sensor_model": "simple",  # simple | probabilistic
    "probabilistic_observation_correct_prob": 0.8,
    "gaussian_sigma": 0.2,
    "transition_matrix": None,
    "transition_offsets": [-3, -2, -1, 0, 1, 2, 3],
    "transition_probabilities": [1 / 7.0] * 7,
    "boundary_behavior": "stay",  # stay | clip | exit
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


SECTION_IIA_41_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
# Atia et al. (2011), Table I (simple model, linear network of 41 sensors)
SECTION_IIA_41_PROBABILITIES = [0.23, 0.10, 0.01, 0.33, 0.06, 0.05, 0.22]
# Atia et al. (2011), Table II (overlap network / continuous model)
SECTION_IIB_20_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
SECTION_IIB_20_PROBABILITIES = [
    1 / 64,
    6 / 64,
    15 / 64,
    20 / 64,
    15 / 64,
    6 / 64,
    1 / 64,
]

# 12 overlapping sensors over 20 locations, for Section II-B style experiments.
# Each tuple is an inclusive [start, end] interval in linear state indices.
SECTION_IIB_20_SENSOR_INTERVALS = [
    (0, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (7, 7),
    (8, 10),
    (9, 11),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 19),
    (18, 19),
]


def build_section_iia_41_world_parameters(
    *,
    lambda_energy: float = 0.2,
    max_steps: int = DEFAULT_MAX_STEP,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    screen_height: int = DEFAULT_SCREEN_HEIGHT,
) -> dict[str, Any]:
    """
    Build a paper-aligned Section II-A configuration:
    - 41 linear states
    - one non-overlapping sensor per state (41 sensors total)
    - Table I transition offsets/probabilities
    """
    num_states = 41
    sensor_specs = [
        {
            "id": f"s{i}",
            "location": (i, 0),
            "fov_deg": 360.0,
            "range": 0.0,  # exactly one covered cell per sensor
            "direction_deg": 0.0,
            "energy_cost": 1.0,
        }
        for i in range(num_states)
    ]
    # Explicit identity visibility matrix to enforce non-overlapping sensing.
    sensor_visibility = np.eye(num_states, dtype=bool)

    return {
        "num_states": num_states,
        "num_targets": 1,
        "max_steps": int(max_steps),
        "lambda_energy": float(lambda_energy),
        "tracking_cost_mode": "unobserved",
        "tracking_reduce": "sum",
        "observation_mode": "discrete",
        "transition_offsets": list(SECTION_IIA_41_OFFSETS),
        "transition_probabilities": list(SECTION_IIA_41_PROBABILITIES),
        "boundary_behavior": "exit",
        "sensor_specs": sensor_specs,
        "sensor_visibility": sensor_visibility,
        "initial_target_states": [num_states // 2],
        "screen_width": int(screen_width),
        "screen_height": int(screen_height),
    }


def build_section_iib_20_world_parameters(
    *,
    sensor_model: str = "simple",
    probabilistic_observation_correct_prob: float = 0.8,
    lambda_energy: float = 0.2,
    max_steps: int = DEFAULT_MAX_STEP,
    screen_width: int = DEFAULT_SCREEN_WIDTH,
    screen_height: int = DEFAULT_SCREEN_HEIGHT,
) -> dict[str, Any]:
    """
    Build a Section II-B style overlapping-sensor configuration:
    - 20 linear object locations
    - 12 overlapping sensors
    - Table II transition offsets/probabilities

    `sensor_model` controls discrete sensing:
    - "simple": perfect discrete observation when covered
    - "probabilistic": discrete noisy observation over covered states
    """
    sensor_model = str(sensor_model)
    if sensor_model not in {"simple", "probabilistic"}:
        raise ValueError("sensor_model must be one of {'simple', 'probabilistic'}")

    num_states = 20
    num_sensors = len(SECTION_IIB_20_SENSOR_INTERVALS)
    sensor_visibility = np.zeros((num_sensors, num_states), dtype=bool)
    sensor_specs = []
    for idx, (start, end) in enumerate(SECTION_IIB_20_SENSOR_INTERVALS):
        if start < 0 or end >= num_states or start > end:
            raise ValueError(f"invalid Section II-B interval: {(start, end)}")
        sensor_visibility[idx, start : end + 1] = True
        center = int((start + end) // 2)
        radius = float(max(center - start, end - center))
        sensor_specs.append(
            {
                "id": f"s{idx}",
                "location": (center, 0),
                "fov_deg": 360.0,
                "range": radius,
                "direction_deg": 0.0,
                "energy_cost": 1.0,
            }
        )

    return {
        "num_states": num_states,
        "num_targets": 1,
        "max_steps": int(max_steps),
        "lambda_energy": float(lambda_energy),
        "tracking_cost_mode": "hamming",
        "tracking_reduce": "sum",
        "observation_mode": "discrete",
        "discrete_sensor_model": sensor_model,
        "probabilistic_observation_correct_prob": float(
            probabilistic_observation_correct_prob
        ),
        "transition_offsets": list(SECTION_IIB_20_OFFSETS),
        "transition_probabilities": list(SECTION_IIB_20_PROBABILITIES),
        "boundary_behavior": "exit",
        "sensor_specs": sensor_specs,
        "sensor_visibility": sensor_visibility,
        "initial_target_states": [num_states // 2],
        "screen_width": int(screen_width),
        "screen_height": int(screen_height),
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

        num_network_states = int(params["num_states"])
        if num_network_states <= 0:
            raise ValueError("num_states must be >= 1")

        boundary_behavior = str(params.get("boundary_behavior", "stay"))
        if boundary_behavior not in {"stay", "clip", "exit"}:
            raise ValueError(
                "boundary_behavior must be one of {'stay', 'clip', 'exit'}"
            )
        add_exit_state = boundary_behavior == "exit"
        num_states = num_network_states + (1 if add_exit_state else 0)
        exit_state = num_states - 1 if add_exit_state else None

        absorbing_states = set()
        for raw in params.get("absorbing_states", []):
            if raw == "exit":
                if not add_exit_state:
                    raise ValueError(
                        "absorbing_states includes 'exit' but boundary_behavior!='exit'"
                    )
                absorbing_states.add(int(exit_state))
                continue
            s = int(raw)
            if s < 0 or s >= num_states:
                raise ValueError(f"absorbing state {s} out of bounds")
            absorbing_states.add(s)
        if add_exit_state and exit_state is not None:
            absorbing_states.add(int(exit_state))

        state_coords = [[i, 0] for i in range(num_network_states)]
        if add_exit_state:
            # Place the external absorbing exit immediately to the right.
            state_coords.append([num_network_states, 0])
        state_coords = np.array(state_coords, dtype=np.int32)

        transition_matrix = params.get("transition_matrix", None)
        if transition_matrix is None:
            transition_matrix = self._build_transition_matrix(
                num_network_states=num_network_states,
                offsets=params["transition_offsets"],
                probabilities=params["transition_probabilities"],
                boundary_behavior=boundary_behavior,
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
            sensors = default_linear_sensors(num_network_states)
        else:
            sensors = parse_sensor_specs(
                raw_sensor_specs, grid_shape=(1, num_network_states)
            )
        if len(sensors) == 0:
            raise ValueError("at least one sensor is required")

        sensor_visibility = params.get("sensor_visibility", None)
        if sensor_visibility is None:
            coverage = compute_linear_coverage_matrix(sensors, num_network_states)
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
            # Exit state is outside sensor network and unobservable.
            coverage = np.concatenate(
                [coverage, np.zeros((len(sensors), 1), dtype=bool)], axis=1
            )

        initial_target_states = params.get("initial_target_states", None)
        sample_initial_from_belief = bool(
            params.get("sample_initial_state_from_belief", False)
        )
        if (
            initial_target_states is None
            and params["num_targets"] == 1
            and not sample_initial_from_belief
        ):
            # Default to center state for paper-aligned setup.
            initial_target_states = [num_network_states // 2]

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
            discrete_sensor_model=str(params.get("discrete_sensor_model", "simple")),
            probabilistic_observation_correct_prob=float(
                params.get("probabilistic_observation_correct_prob", 0.8)
            ),
            gaussian_sigma=float(params["gaussian_sigma"]),
            initial_target_states=initial_target_states,
            initial_belief=params["initial_belief"],
            sample_initial_state_from_belief=sample_initial_from_belief,
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
        num_network_states: int,
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

        add_exit_state = boundary_behavior == "exit"
        num_states = num_network_states + (1 if add_exit_state else 0)
        exit_state = num_states - 1 if add_exit_state else None
        T = np.zeros((num_states, num_states), dtype=np.float64)
        for s in range(num_states):
            if s in absorbing_states:
                T[s, s] = 1.0
                continue

            for dx, prob in zip(offsets, p):
                nxt = s + int(dx)
                if 0 <= nxt < num_network_states:
                    T[s, nxt] += prob
                else:
                    if boundary_behavior == "exit":
                        T[s, exit_state] += prob
                    elif boundary_behavior == "clip":
                        nxt = min(max(nxt, 0), num_network_states - 1)
                        T[s, nxt] += prob
                    elif boundary_behavior == "stay":
                        T[s, s] += prob
                    else:
                        raise ValueError(
                            "boundary_behavior must be one of {'stay', 'clip', 'exit'}"
                        )

        row_sum = T.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        T /= row_sum
        return T
