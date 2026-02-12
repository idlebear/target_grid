"""
Shared base class for sensor scheduling environments.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

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
        true_state_in_info: bool = False,
        obstacle_grid: np.ndarray | None = None,
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
        self.coverage_matrix = np.asarray(coverage_matrix, dtype=bool)
        if self.coverage_matrix.shape != (self.num_sensors, self.num_states):
            raise ValueError(
                "coverage_matrix must have shape (num_sensors, num_states)"
            )

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

    def _reset_belief(self, options: dict[str, Any] | None = None) -> None:
        if options is None:
            options = {}
        init = options.get("initial_belief", self.initial_belief)
        if isinstance(init, str):
            if init == "uniform":
                self.belief[:, :] = 1.0 / float(self.num_states)
                return
            if init == "uniform_non_absorbing":
                mask = np.ones((self.num_states,), dtype=np.float64)
                for idx in self.absorbing_states:
                    mask[idx] = 0.0
                if np.all(mask == 0.0):
                    mask[:] = 1.0
                mask /= np.sum(mask)
                for k in range(self.num_targets):
                    self.belief[k, :] = mask
                return
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
        self.belief[:, :] = arr / row_sum

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
                if not self.coverage_matrix[sensor_idx, true_state]:
                    continue
                valid[k, sensor_idx] = 1
                if self.observation_mode == "discrete":
                    if self.discrete_sensor_model == "simple":
                        measurements[k, sensor_idx] = float(true_state)
                    else:
                        covered_states = np.flatnonzero(
                            self.coverage_matrix[sensor_idx, :]
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
        for k in range(self.num_targets):
            if self.absorbed_mask[k]:
                # absorbed targets are fully known and no longer evolve
                self.belief[k, :] = 0.0
                self.belief[k, int(self.target_states[k])] = 1.0
                continue

            pred = self.belief[k, :] @ self.transition_matrix
            like = np.ones((self.num_states,), dtype=np.float64)

            for sensor_idx in range(self.num_sensors):
                if action[sensor_idx] == 0:
                    continue

                covered = self.coverage_matrix[sensor_idx, :]
                is_valid = bool(measurement_valid[k, sensor_idx])
                meas = float(measurements[k, sensor_idx])

                if self.observation_mode == "discrete":
                    if not is_valid:
                        sensor_like = np.where(covered, 0.0, 1.0)
                    else:
                        state_idx = int(round(meas))
                        sensor_like = np.zeros((self.num_states,), dtype=np.float64)
                        if self.discrete_sensor_model == "simple":
                            if (
                                0 <= state_idx < self.num_states
                                and covered[state_idx]
                            ):
                                sensor_like[state_idx] = 1.0
                        else:
                            covered_states = np.flatnonzero(covered)
                            if covered_states.size == 1:
                                only = int(covered_states[0])
                                if state_idx == only:
                                    sensor_like[only] = 1.0
                            elif covered_states.size > 1:
                                p = self.probabilistic_observation_correct_prob
                                miss = (1.0 - p) / float(covered_states.size - 1)
                                sensor_like[covered_states] = miss
                                if (
                                    0 <= state_idx < self.num_states
                                    and covered[state_idx]
                                ):
                                    sensor_like[state_idx] = p
                    like *= sensor_like
                else:
                    if not is_valid:
                        sensor_like = np.where(covered, 0.0, 1.0)
                    else:
                        means = np.array(
                            [
                                self._signal_mean(sensor_idx, s)
                                for s in range(self.num_states)
                            ],
                            dtype=np.float64,
                        )
                        sensor_like = _gaussian_pdf(meas, means, self.gaussian_sigma)
                        sensor_like = np.where(covered, sensor_like, 0.0)
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
            if self.tracking_cost_mode == "unobserved":
                seen = bool(
                    np.any(
                        (self.last_action > 0)
                        & self.coverage_matrix[:, true_state].astype(np.int8)
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
            covered = np.any(self.coverage_matrix[active_sensor_idx, :], axis=0)
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
