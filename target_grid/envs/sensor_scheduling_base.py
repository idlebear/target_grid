"""
Shared base class for sensor scheduling environments.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .constants import DEFAULT_SCREEN_HEIGHT, DEFAULT_SCREEN_WIDTH
from .sensors import SensorSpec
from .window import Window, Colors


def _gaussian_pdf(x: float, mean: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    z = (mean - x) / sigma
    c = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return c * np.exp(-0.5 * z * z)


class SensorSchedulingBaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        *,
        state_coords: np.ndarray,
        transition_matrix: np.ndarray,
        sensors: Sequence[SensorSpec],
        coverage_matrix: np.ndarray,
        absorbing_states: Iterable[int] | None = None,
        num_targets: int = 1,
        max_steps: int = 200,
        lambda_energy: float = 1.0,
        tracking_cost_mode: str = "unobserved",
        tracking_reduce: str = "sum",
        observation_mode: str = "discrete",
        discrete_sensor_model: str = "simple",
        probabilistic_observation_correct_prob: float = 0.8,
        gaussian_sigma: float = 0.2,
        initial_target_states: Sequence[int] | None = None,
        initial_belief: str | np.ndarray = "uniform",
        sample_initial_state_from_belief: bool = False,
        true_state_in_info: bool = False,
        obstacle_grid: np.ndarray | None = None,
        occlude_sensors: bool = False,
        dynamic_coverage_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        screen_width: int = DEFAULT_SCREEN_WIDTH,
        screen_height: int = DEFAULT_SCREEN_HEIGHT,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.state_coords = np.asarray(state_coords, dtype=np.int32)
        if self.state_coords.ndim != 2 or self.state_coords.shape[1] != 2:
            raise ValueError("state_coords must have shape (num_states, 2)")
        self.num_states = int(self.state_coords.shape[0])
        if self.num_states <= 0:
            raise ValueError("at least one state is required")

        self.transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        if self.transition_matrix.shape != (self.num_states, self.num_states):
            raise ValueError(
                "transition_matrix must have shape (num_states, num_states)"
            )
        if np.any(self.transition_matrix < 0.0):
            raise ValueError("transition_matrix contains negative values")
        row_sum = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sum, 1.0, atol=1e-8):
            raise ValueError("every transition matrix row must sum to 1")

        self.sensors = list(sensors)
        self.num_sensors = len(self.sensors)
        if self.num_sensors <= 0:
            raise ValueError("at least one sensor is required")
        self.base_coverage_matrix = np.asarray(coverage_matrix, dtype=np.float32)
        if self.base_coverage_matrix.shape != (self.num_sensors, self.num_states):
            raise ValueError(
                "coverage_matrix must have shape (num_sensors, num_states)"
            )
        self.base_coverage_matrix = np.clip(self.base_coverage_matrix, 0.0, 1.0)
        self.coverage_matrix = self.base_coverage_matrix.copy()

        self.num_targets = int(num_targets)
        if self.num_targets <= 0:
            raise ValueError("num_targets must be >= 1")

        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be >= 1")

        self.lambda_energy = float(lambda_energy)
        self.tracking_cost_mode = str(tracking_cost_mode)
        self.tracking_reduce = str(tracking_reduce)
        self.observation_mode = str(observation_mode)
        if self.observation_mode not in {"discrete", "continuous_gaussian"}:
            raise ValueError("observation_mode must be discrete or continuous_gaussian")
        self.discrete_sensor_model = str(discrete_sensor_model)
        if self.discrete_sensor_model not in {"simple", "probabilistic"}:
            raise ValueError(
                "discrete_sensor_model must be one of {'simple', 'probabilistic'}"
            )
        self.probabilistic_observation_correct_prob = float(
            probabilistic_observation_correct_prob
        )
        if not (0.0 <= self.probabilistic_observation_correct_prob <= 1.0):
            raise ValueError(
                "probabilistic_observation_correct_prob must be in [0, 1]"
            )
        self.gaussian_sigma = float(gaussian_sigma)
        self.true_state_in_info = bool(true_state_in_info)
        self.occlude_sensors = bool(occlude_sensors)
        self._dynamic_coverage_fn = dynamic_coverage_fn

        self._state_lookup = {
            (int(self.state_coords[idx, 0]), int(self.state_coords[idx, 1])): idx
            for idx in range(self.num_states)
        }

        self.absorbing_states = self._parse_state_set(absorbing_states)
        self._enforce_absorbing_rows()

        self.initial_target_states = (
            None
            if initial_target_states is None
            else [self._coerce_state_value(v) for v in initial_target_states]
        )
        self.initial_belief = initial_belief
        self.sample_initial_state_from_belief = bool(sample_initial_state_from_belief)

        self.obstacle_grid = (
            None
            if obstacle_grid is None
            else np.asarray(obstacle_grid, dtype=np.int8).copy()
        )
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)

        self.action_space = spaces.MultiBinary(self.num_sensors)
        self.observation_space = spaces.Dict(
            {
                "belief": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_targets, self.num_states),
                    dtype=np.float32,
                ),
                "sensor_mask": spaces.MultiBinary(self.num_sensors),
                "measurement_valid": spaces.MultiBinary(
                    (self.num_targets, self.num_sensors)
                ),
                "measurements": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_targets, self.num_sensors),
                    dtype=np.float32,
                ),
                "absorbed_mask": spaces.MultiBinary(self.num_targets),
                "time_step": spaces.Discrete(self.max_steps + 1),
            }
        )

        self.render_mode = render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode={render_mode}")

        self.sensor_energy_costs = np.array(
            [float(s.energy_cost) for s in self.sensors], dtype=np.float64
        )

        self.steps = 0
        self.target_states = np.zeros((self.num_targets,), dtype=np.int32)
        self.absorbed_mask = np.zeros((self.num_targets,), dtype=np.int8)
        self.belief = np.zeros((self.num_targets, self.num_states), dtype=np.float64)
        self.last_action = np.zeros((self.num_sensors,), dtype=np.int8)
        self.last_measurements = np.zeros(
            (self.num_targets, self.num_sensors), dtype=np.float32
        )
        self.last_measurement_valid = np.zeros(
            (self.num_targets, self.num_sensors), dtype=np.int8
        )
        self._terminated = False
        self._truncated = False
        self.window: Window | None = None
        self._warned_human_unavailable = False

    def _occluder_state_indices(self) -> np.ndarray:
        if not self.occlude_sensors:
            return np.empty((0,), dtype=np.int32)
        out: list[int] = []
        rows = cols = 0
        if self.obstacle_grid is not None:
            rows, cols = self.obstacle_grid.shape
        for s in self.target_states:
            idx = int(s)
            if idx < 0 or idx >= self.num_states:
                continue
            x = int(self.state_coords[idx, 0])
            y = int(self.state_coords[idx, 1])
            # Skip non-physical states (e.g. synthetic exit nodes).
            if self.obstacle_grid is not None and not (0 <= x < cols and 0 <= y < rows):
                continue
            out.append(idx)
        if len(out) == 0:
            return np.empty((0,), dtype=np.int32)
        return np.asarray(sorted(set(out)), dtype=np.int32)

    def _build_measurement_target_occupancy(self) -> np.ndarray | None:
        if self.obstacle_grid is None:
            return None
        blocker = np.zeros_like(self.obstacle_grid, dtype=np.float32)
        occ_states = self._occluder_state_indices()
        if occ_states.size == 0:
            return blocker
        rows, cols = blocker.shape
        for idx in occ_states:
            x = int(self.state_coords[idx, 0])
            y = int(self.state_coords[idx, 1])
            if 0 <= x < cols and 0 <= y < rows:
                blocker[y, x] = 1
        return blocker

    def _build_prediction_target_occupancy(
        self,
        predicted_belief: np.ndarray,
    ) -> np.ndarray | None:
        if self.obstacle_grid is None:
            return None
        if predicted_belief.shape != (self.num_targets, self.num_states):
            raise ValueError(
                "predicted_belief must have shape (num_targets, num_states)"
            )
        rows, cols = self.obstacle_grid.shape
        valid_state_mask = (
            (self.state_coords[:, 0] >= 0)
            & (self.state_coords[:, 0] < cols)
            & (self.state_coords[:, 1] >= 0)
            & (self.state_coords[:, 1] < rows)
        )
        if not np.any(valid_state_mask):
            return np.zeros_like(self.obstacle_grid, dtype=np.float32)

        xs = self.state_coords[valid_state_mask, 0].astype(np.int32)
        ys = self.state_coords[valid_state_mask, 1].astype(np.int32)
        no_target_occupancy = np.ones((rows, cols), dtype=np.float32)
        for target_idx in range(self.num_targets):
            target_grid = np.zeros((rows, cols), dtype=np.float32)
            target_probs = np.asarray(
                predicted_belief[target_idx, valid_state_mask], dtype=np.float32
            )
            np.add.at(target_grid, (ys, xs), target_probs)
            np.clip(target_grid, 0.0, 1.0, out=target_grid)
            no_target_occupancy *= 1.0 - target_grid

        return np.clip(1.0 - no_target_occupancy, 0.0, 1.0)

    def _dynamic_coverage_from_target_occupancy(
        self, target_occupancy: np.ndarray | None
    ) -> np.ndarray:
        if (
            not self.occlude_sensors
            or self._dynamic_coverage_fn is None
            or target_occupancy is None
            or not np.any(target_occupancy)
        ):
            return self.base_coverage_matrix.copy()

        dynamic = np.asarray(
            self._dynamic_coverage_fn(np.asarray(target_occupancy, dtype=np.float32)),
            dtype=np.float32,
        )
        if dynamic.shape != (self.num_sensors, self.num_states):
            raise ValueError(
                "dynamic coverage matrix must have shape (num_sensors, num_states)"
            )
        return np.clip(dynamic, 0.0, 1.0)

    def _prediction_coverage_matrix(self, predicted_belief: np.ndarray) -> np.ndarray:
        target_occupancy = self._build_prediction_target_occupancy(predicted_belief)
        return self._dynamic_coverage_from_target_occupancy(target_occupancy)

    def _refresh_coverage_matrix(self) -> None:
        target_occupancy = self._build_measurement_target_occupancy()
        self.coverage_matrix[:, :] = self._dynamic_coverage_from_target_occupancy(
            target_occupancy
        )

    def _coerce_state_value(self, value: Any) -> int:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
            if idx < 0 or idx >= self.num_states:
                raise ValueError(f"state index {idx} out of bounds")
            return idx

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, (tuple, list)) and len(value) == 2:
            key = (int(value[0]), int(value[1]))
            if key not in self._state_lookup:
                raise ValueError(f"unknown state coordinate {key}")
            return int(self._state_lookup[key])

        raise TypeError(f"unsupported state value type: {type(value).__name__}")

    def _parse_state_set(self, values: Iterable[int] | None) -> set[int]:
        if values is None:
            return set()
        return {self._coerce_state_value(v) for v in values}

    def _enforce_absorbing_rows(self) -> None:
        for idx in self.absorbing_states:
            self.transition_matrix[idx, :] = 0.0
            self.transition_matrix[idx, idx] = 1.0

    def _random_start_state(self) -> int:
        candidates = [
            s for s in range(self.num_states) if s not in self.absorbing_states
        ]
        if not candidates:
            candidates = list(range(self.num_states))
        return int(self.np_random.choice(candidates))

    def _resolve_initial_belief_array(
        self, init: str | np.ndarray
    ) -> np.ndarray:
        """
        Resolve initial belief specification into a normalized
        (num_targets, num_states) array.
        """
        if isinstance(init, str):
            if init == "uniform":
                arr = np.full(
                    (self.num_targets, self.num_states),
                    1.0 / float(self.num_states),
                    dtype=np.float64,
                )
                return arr
            if init == "uniform_non_absorbing":
                mask = np.ones((self.num_states,), dtype=np.float64)
                for idx in self.absorbing_states:
                    mask[idx] = 0.0
                if np.all(mask == 0.0):
                    mask[:] = 1.0
                mask /= np.sum(mask)
                arr = np.broadcast_to(
                    mask.reshape(1, -1), (self.num_targets, self.num_states)
                ).copy()
                return arr
            raise ValueError(f"unknown initial_belief mode '{init}'")

        arr = np.asarray(init, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != self.num_states:
                raise ValueError("initial belief vector length mismatch")
            arr = np.broadcast_to(
                arr.reshape(1, -1), (self.num_targets, self.num_states)
            )
        elif arr.shape != (self.num_targets, self.num_states):
            raise ValueError("initial belief array shape mismatch")
        arr = arr.copy()
        arr[arr < 0.0] = 0.0
        row_sum = arr.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0.0] = 1.0
        return arr / row_sum

    def _reset_belief(self, options: dict[str, Any] | None = None) -> np.ndarray:
        if options is None:
            options = {}
        init = options.get("initial_belief", self.initial_belief)
        arr = self._resolve_initial_belief_array(init)
        self.belief[:, :] = arr
        return arr

    def _signal_mean(self, sensor_idx: int, state_idx: int) -> float:
        sx, sy = self.sensors[sensor_idx].location
        x, y = int(self.state_coords[state_idx, 0]), int(
            self.state_coords[state_idx, 1]
        )
        d2 = float((x - sx) * (x - sx) + (y - sy) * (y - sy))
        return 1.0 / (1e-6 + d2)

    def _generate_measurements(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        measurements = np.zeros((self.num_targets, self.num_sensors), dtype=np.float32)
        valid = np.zeros((self.num_targets, self.num_sensors), dtype=np.int8)

        for k in range(self.num_targets):
            true_state = int(self.target_states[k])
            for sensor_idx in range(self.num_sensors):
                if action[sensor_idx] == 0:
                    continue
                vis_prob = float(self.coverage_matrix[sensor_idx, true_state])
                if vis_prob <= 0.0:
                    continue
                if self.np_random.random() > vis_prob:
                    continue
                valid[k, sensor_idx] = 1
                if self.observation_mode == "discrete":
                    if self.discrete_sensor_model == "simple":
                        measurements[k, sensor_idx] = float(true_state)
                    else:
                        covered_states = np.flatnonzero(
                            self.coverage_matrix[sensor_idx, :] > 1e-8
                        )
                        if covered_states.size <= 1:
                            measured_state = true_state
                        else:
                            p = self.probabilistic_observation_correct_prob
                            if self.np_random.random() < p:
                                measured_state = true_state
                            else:
                                candidates = covered_states[covered_states != true_state]
                                measured_state = int(
                                    self.np_random.choice(candidates)
                                )
                        measurements[k, sensor_idx] = float(measured_state)
                else:
                    mu = self._signal_mean(sensor_idx, true_state)
                    measurements[k, sensor_idx] = float(
                        self.np_random.normal(loc=mu, scale=self.gaussian_sigma)
                    )

        return measurements, valid

    def _update_belief(
        self,
        action: np.ndarray,
        measurements: np.ndarray,
        measurement_valid: np.ndarray,
    ) -> None:
        eps = 1e-12
        predicted_belief = np.zeros_like(self.belief)
        for k in range(self.num_targets):
            if self.absorbed_mask[k]:
                predicted_belief[k, :] = 0.0
                predicted_belief[k, int(self.target_states[k])] = 1.0
            else:
                predicted_belief[k, :] = self.belief[k, :] @ self.transition_matrix

        prediction_coverage = self._prediction_coverage_matrix(predicted_belief)
        for k in range(self.num_targets):
            if self.absorbed_mask[k]:
                # absorbed targets are fully known and no longer evolve
                self.belief[k, :] = predicted_belief[k, :]
                continue

            pred = predicted_belief[k, :]
            like = np.ones((self.num_states,), dtype=np.float64)

            for sensor_idx in range(self.num_sensors):
                if action[sensor_idx] == 0:
                    continue

                covered_prob = np.asarray(
                    prediction_coverage[sensor_idx, :], dtype=np.float64
                )
                covered_prob = np.clip(covered_prob, 0.0, 1.0)
                covered = covered_prob > 1e-8
                is_valid = bool(measurement_valid[k, sensor_idx])
                meas = float(measurements[k, sensor_idx])

                if self.observation_mode == "discrete":
                    if not is_valid:
                        sensor_like = 1.0 - covered_prob
                    else:
                        state_idx = int(round(meas))
                        sensor_like = np.zeros((self.num_states,), dtype=np.float64)
                        if self.discrete_sensor_model == "simple":
                            if 0 <= state_idx < self.num_states:
                                sensor_like[state_idx] = covered_prob[state_idx]
                        else:
                            covered_states = np.flatnonzero(covered)
                            if covered_states.size == 1:
                                only = int(covered_states[0])
                                if state_idx == only:
                                    sensor_like[only] = covered_prob[only]
                            elif covered_states.size > 1:
                                p = self.probabilistic_observation_correct_prob
                                miss = (1.0 - p) / float(covered_states.size - 1)
                                sensor_like[covered_states] = (
                                    covered_prob[covered_states] * miss
                                )
                                if (
                                    0 <= state_idx < self.num_states
                                    and covered[state_idx]
                                ):
                                    sensor_like[state_idx] = covered_prob[state_idx] * p
                    like *= sensor_like
                else:
                    if not is_valid:
                        sensor_like = 1.0 - covered_prob
                    else:
                        means = np.array(
                            [
                                self._signal_mean(sensor_idx, s)
                                for s in range(self.num_states)
                            ],
                            dtype=np.float64,
                        )
                        sensor_like = _gaussian_pdf(meas, means, self.gaussian_sigma)
                        sensor_like = sensor_like * covered_prob
                    like *= sensor_like

            post = pred * like
            total = float(np.sum(post))
            if total <= eps:
                # fallback to prediction if observations fully eliminate support
                pred_sum = float(np.sum(pred))
                if pred_sum <= eps:
                    post = np.full((self.num_states,), 1.0 / float(self.num_states))
                else:
                    post = pred / pred_sum
            else:
                post /= total
            self.belief[k, :] = post

    def _compute_tracking_cost(self) -> float:
        per_target = np.zeros((self.num_targets,), dtype=np.float64)
        for k in range(self.num_targets):
            true_state = int(self.target_states[k])
            if true_state in self.absorbing_states:
                # Terminal absorbing states are cost-free.
                per_target[k] = 0.0
                continue
            if self.tracking_cost_mode == "unobserved":
                seen = bool(
                    np.any(
                        (self.last_action > 0)
                        & (self.last_measurement_valid[k, :] > 0)
                    )
                )
                per_target[k] = 0.0 if seen else 1.0
            elif self.tracking_cost_mode == "hamming":
                estimate = int(np.argmax(self.belief[k, :]))
                per_target[k] = 0.0 if estimate == true_state else 1.0
            elif self.tracking_cost_mode == "distance":
                estimate = int(np.argmax(self.belief[k, :]))
                tx, ty = self.state_coords[true_state]
                ex, ey = self.state_coords[estimate]
                per_target[k] = float(np.hypot(tx - ex, ty - ey))
            else:
                raise ValueError(
                    f"unknown tracking_cost_mode '{self.tracking_cost_mode}'"
                )

        if self.tracking_reduce == "sum":
            return float(np.sum(per_target))
        if self.tracking_reduce == "mean":
            return float(np.mean(per_target))
        raise ValueError(f"unknown tracking_reduce '{self.tracking_reduce}'")

    def _get_obs(self) -> dict[str, Any]:
        return {
            "belief": self.belief.astype(np.float32),
            "sensor_mask": self.last_action.astype(np.int8),
            "measurement_valid": self.last_measurement_valid.astype(np.int8),
            "measurements": self.last_measurements.astype(np.float32),
            "absorbed_mask": self.absorbed_mask.astype(np.int8),
            "time_step": int(self.steps),
        }

    def _get_info(
        self, energy_cost: float = 0.0, tracking_cost: float = 0.0
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "energy_cost": float(energy_cost),
            "tracking_cost": float(tracking_cost),
            "total_cost": float(energy_cost + tracking_cost),
            "num_active_sensors": int(np.sum(self.last_action)),
            "absorbed_mask": self.absorbed_mask.astype(np.int8).copy(),
            "occlude_sensors": bool(self.occlude_sensors),
        }
        if self.true_state_in_info:
            info["true_state"] = self.target_states.astype(np.int32).copy()
        return info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = getattr(self, "_pending_seed", None)
            if hasattr(self, "_pending_seed"):
                self._pending_seed = None
        super().reset(seed=seed)
        self.steps = 0
        self._terminated = False
        self._truncated = False

        if options is None:
            options = {}

        starts = options.get("initial_target_states", self.initial_target_states)
        if starts is None:
            if self.sample_initial_state_from_belief:
                init_dist = self._resolve_initial_belief_array(
                    options.get("initial_belief", self.initial_belief)
                )
                sampled = []
                for k in range(self.num_targets):
                    p = np.asarray(init_dist[k, :], dtype=np.float64)
                    sampled.append(int(self.np_random.choice(self.num_states, p=p)))
                self.target_states[:] = sampled
            else:
                self.target_states[:] = [
                    self._random_start_state() for _ in range(self.num_targets)
                ]
        else:
            if len(starts) != self.num_targets:
                raise ValueError(
                    "initial_target_states length mismatch with num_targets"
                )
            self.target_states[:] = [self._coerce_state_value(v) for v in starts]

        self.absorbed_mask[:] = np.array(
            [1 if int(s) in self.absorbing_states else 0 for s in self.target_states],
            dtype=np.int8,
        )
        self._terminated = bool(np.all(self.absorbed_mask > 0))

        self._reset_belief(options=options)
        for k in range(self.num_targets):
            if self.absorbed_mask[k]:
                self.belief[k, :] = 0.0
                self.belief[k, int(self.target_states[k])] = 1.0

        self._refresh_coverage_matrix()

        self.last_action[:] = 0
        self.last_measurements[:, :] = 0.0
        self.last_measurement_valid[:, :] = 0

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def _normalize_action(self, action: np.ndarray | Sequence[int]) -> np.ndarray:
        a = np.asarray(action, dtype=np.int8).reshape(-1)
        if a.shape != (self.num_sensors,):
            raise ValueError(
                f"action has wrong shape {a.shape}, expected ({self.num_sensors},)"
            )
        a = (a > 0).astype(np.int8)
        return a

    def step(self, action):
        if self._terminated:
            obs = self._get_obs()
            if self.render_mode == "human":
                self.render()
            return obs, 0.0, True, False, self._get_info()

        if self._truncated:
            obs = self._get_obs()
            if self.render_mode == "human":
                self.render()
            return obs, 0.0, False, True, self._get_info()

        self.steps += 1
        a = self._normalize_action(action)
        self.last_action[:] = a

        for k in range(self.num_targets):
            state = int(self.target_states[k])
            if state in self.absorbing_states:
                self.absorbed_mask[k] = 1
                continue
            nxt = int(
                self.np_random.choice(
                    self.num_states, p=self.transition_matrix[state, :]
                )
            )
            self.target_states[k] = nxt
            if nxt in self.absorbing_states:
                self.absorbed_mask[k] = 1

        self._refresh_coverage_matrix()
        measurements, valid = self._generate_measurements(a)
        self.last_measurements[:, :] = measurements
        self.last_measurement_valid[:, :] = valid
        self._update_belief(a, measurements, valid)

        energy_cost = self.lambda_energy * float(np.dot(a, self.sensor_energy_costs))
        tracking_cost = self._compute_tracking_cost()
        reward = -(energy_cost + tracking_cost)

        self._terminated = bool(np.all(self.absorbed_mask > 0))
        self._truncated = bool(self.steps >= self.max_steps and not self._terminated)

        obs = self._get_obs()
        info = self._get_info(energy_cost=energy_cost, tracking_cost=tracking_cost)
        if self.render_mode == "human":
            self.render()
        return obs, reward, self._terminated, self._truncated, info

    def _grid_shape(self) -> tuple[int, int]:
        if self.obstacle_grid is not None:
            return int(self.obstacle_grid.shape[0]), int(self.obstacle_grid.shape[1])
        max_x = int(np.max(self.state_coords[:, 0]))
        max_y = int(np.max(self.state_coords[:, 1]))
        return max_y + 1, max_x + 1

    def _ensure_window(self, with_display: bool) -> Window:
        if self.window is None:
            rows, cols = self._grid_shape()
            Window.initialize(with_display=with_display)
            self.window = Window(
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                margin=max(1, int(min(self.screen_height, self.screen_width) / 100)),
                display_origin=(0, 0),
                display_size=(cols, rows),
                frame_rate=self.metadata["render_fps"],
            )
        return self.window

    def _draw_window_frame(self, with_display: bool) -> Window:
        window = self._ensure_window(with_display=with_display)
        window.clear()

        rows, cols = self._grid_shape()
        valid_state_cells = {
            (int(self.state_coords[s, 0]), int(self.state_coords[s, 1]))
            for s in range(self.num_states)
        }

        for y in range(rows):
            for x in range(cols):
                if self.obstacle_grid is not None and self.obstacle_grid[y, x] != 0:
                    color = Colors.dark_grey
                elif (x, y) in valid_state_cells:
                    color = Colors.light_blue
                else:
                    color = Colors.white
                window.draw_rect(
                    center=(x + 0.5, y + 0.5),
                    height=1.0,
                    width=1.0,
                    color=color,
                    border_width=1,
                    border_color=Colors.black,
                )

        # Highlight cells covered by at least one currently active sensor.
        active_sensor_idx = np.where(self.last_action > 0)[0]
        if len(active_sensor_idx) > 0:
            covered = np.any(self.coverage_matrix[active_sensor_idx, :] > 1e-8, axis=0)
            for state_idx, is_covered in enumerate(covered):
                if not is_covered:
                    continue
                x, y = int(self.state_coords[state_idx, 0]), int(
                    self.state_coords[state_idx, 1]
                )
                window.draw_rect(
                    center=(x + 0.5, y + 0.5),
                    height=0.9,
                    width=0.9,
                    color=(255, 255, 0, 90),
                    border_width=0,
                    use_transparency=True,
                )

        for idx in self.absorbing_states:
            x, y = int(self.state_coords[idx, 0]), int(self.state_coords[idx, 1])
            window.draw_rect(
                center=(x + 0.5, y + 0.5),
                height=0.6,
                width=0.6,
                color=Colors.orange,
                border_width=1,
                border_color=Colors.black,
            )

        for sensor_idx, sensor in enumerate(self.sensors):
            x, y = sensor.location
            color = Colors.blue if self.last_action[sensor_idx] == 0 else Colors.cyan
            window.draw_circle(center=(x + 0.5, y + 0.5), color=color, radius=0.18)

        for k in range(self.num_targets):
            s = int(self.target_states[k])
            x, y = int(self.state_coords[s, 0]), int(self.state_coords[s, 1])
            color = Colors.red if self.absorbed_mask[k] == 0 else Colors.magenta
            window.draw_triangle(
                center=(x + 0.5, y + 0.5),
                size=0.6,
                orientation=0.0,
                color=color,
                border_width=1,
                border_color=Colors.black,
            )

        return window

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            window = self._draw_window_frame(with_display=False)
            return window.render()
        if self.render_mode == "human":
            try:
                window = self._draw_window_frame(with_display=True)
                window.display()
            except Exception as exc:  # pragma: no cover - runtime/display dependent
                if not self._warned_human_unavailable:
                    print(f"Human render unavailable: {exc}")
                    self._warned_human_unavailable = True
            return None
        return None

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None
