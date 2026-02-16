"""
Sensor specifications and helper utilities for sensor scheduling environments.
"""

from dataclasses import dataclass, replace
from math import atan2, inf, pi
from typing import Iterable, List, Sequence

import numpy as np


try:
    from polycheck import sensor_visibility_from_region
except Exception:  # pragma: no cover - optional dependency may fail at import/runtime
    sensor_visibility_from_region = None


def _normalize_angle_deg(angle: float) -> float:
    return float(angle) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    """Return the signed shortest angular difference a-b in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


@dataclass(frozen=True)
class SensorSpec:
    id: str
    location: tuple[int, int]
    fov_deg: float = 360.0
    range: float = inf
    direction_deg: float = 0.0
    energy_cost: float = 1.0
    noise_model: str = "perfect"


def validate_sensor_spec(
    sensor: SensorSpec,
    grid_shape: tuple[int, int] | None = None,
    obstacle_grid: np.ndarray | None = None,
) -> SensorSpec:
    if not (0.0 <= sensor.fov_deg <= 360.0):
        raise ValueError(f"sensor '{sensor.id}' has invalid fov_deg={sensor.fov_deg}")
    if sensor.range < 0.0:
        raise ValueError(f"sensor '{sensor.id}' has invalid range={sensor.range}")
    if sensor.energy_cost < 0.0:
        raise ValueError(
            f"sensor '{sensor.id}' has invalid energy_cost={sensor.energy_cost}"
        )

    x, y = sensor.location
    if grid_shape is not None:
        rows, cols = grid_shape
        if x < 0 or x >= cols or y < 0 or y >= rows:
            raise ValueError(
                f"sensor '{sensor.id}' location {sensor.location} out of bounds"
            )
    if obstacle_grid is not None and obstacle_grid[y, x] != 0:
        raise ValueError(
            f"sensor '{sensor.id}' location {sensor.location} is on an obstacle"
        )

    return replace(sensor, direction_deg=_normalize_angle_deg(sensor.direction_deg))


def parse_sensor_specs(
    specs: Iterable[SensorSpec | dict],
    grid_shape: tuple[int, int] | None = None,
    obstacle_grid: np.ndarray | None = None,
) -> List[SensorSpec]:
    parsed: List[SensorSpec] = []
    ids = set()
    for idx, raw in enumerate(specs):
        if isinstance(raw, SensorSpec):
            sensor = raw
        elif isinstance(raw, dict):
            sensor = SensorSpec(**raw)
        else:
            raise TypeError(
                f"invalid sensor spec type at index {idx}: {type(raw).__name__}"
            )
        sensor = validate_sensor_spec(
            sensor=sensor, grid_shape=grid_shape, obstacle_grid=obstacle_grid
        )
        if sensor.id in ids:
            raise ValueError(f"duplicate sensor id: '{sensor.id}'")
        ids.add(sensor.id)
        parsed.append(sensor)
    return parsed


def angle_gate(
    src: tuple[int, int],
    dst: tuple[int, int],
    direction_deg: float,
    fov_deg: float,
) -> bool:
    if fov_deg >= 360.0:
        return True
    dx = float(dst[0] - src[0])
    dy = float(dst[1] - src[1])
    if dx == 0.0 and dy == 0.0:
        return True
    bearing = atan2(dy, dx) * 180.0 / pi
    return abs(_angle_diff_deg(bearing, direction_deg)) <= (fov_deg * 0.5)


def _fallback_visibility(
    obstacle_grid: np.ndarray,
    start: tuple[int, int],
    ends: np.ndarray,
) -> np.ndarray:
    """
    Fallback line-of-sight visibility (Bresenham-style integer ray marching).
    """
    rows, cols = obstacle_grid.shape
    out = np.zeros((len(ends),), dtype=float)

    print(
        "WARNING: Using fallback visibility implementation. Install polycheck for better performance."
    )

    for idx, (x1, y1) in enumerate(ends):
        x0, y0 = start
        if x1 < 0 or x1 >= cols or y1 < 0 or y1 >= rows:
            continue

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        visible = True
        while True:
            if (x, y) == (x1, y1):
                break
            if (x, y) != (x0, y0) and obstacle_grid[y, x] != 0:
                visible = False
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        out[idx] = 1.0 if visible else 0.0

    return out


def _sensor_specs_to_polycheck_array(sensors: Sequence[SensorSpec]) -> np.ndarray:
    sensor_array = np.zeros((len(sensors), 5), dtype=np.float32)
    for sensor_idx, sensor in enumerate(sensors):
        sensor_range = float(sensor.range)
        if sensor_range < 0.0:
            sensor_range = 0.0
        sensor_array[sensor_idx, 0] = float(sensor.location[0])
        sensor_array[sensor_idx, 1] = float(sensor.location[1])
        sensor_array[sensor_idx, 2] = np.float32(sensor_range)
        sensor_array[sensor_idx, 3] = np.float32(np.deg2rad(sensor.direction_deg))
        sensor_array[sensor_idx, 4] = np.float32(np.deg2rad(sensor.fov_deg))
    return sensor_array


def compute_grid_coverage_matrix(
    sensors: Sequence[SensorSpec],
    state_coords: np.ndarray,
    obstacle_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute visibility probabilities M where M[i, j] is the probability that
    sensor i can observe state j given range, FOV, direction, and occupancy.
    """
    num_sensors = len(sensors)
    num_states = state_coords.shape[0]
    coverage = np.zeros((num_sensors, num_states), dtype=np.float32)
    if num_sensors == 0 or num_states == 0:
        return coverage

    occupancy_grid = np.asarray(obstacle_grid, dtype=np.float32)
    occupancy_grid = np.clip(occupancy_grid, 0.0, 1.0)
    rows, cols = occupancy_grid.shape
    ends = np.array([[x, y] for y in range(rows) for x in range(cols)], dtype=int)
    xs = state_coords[:, 0].astype(np.int32)
    ys = state_coords[:, 1].astype(np.int32)
    valid_state_mask = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)

    if sensor_visibility_from_region is not None:
        sensor_data = _sensor_specs_to_polycheck_array(sensors)
        per_sensor_visibility, _ = sensor_visibility_from_region(
            occupancy_grid, sensor_data
        )
        valid_x = xs[valid_state_mask]
        valid_y = ys[valid_state_mask]
        coverage[:, valid_state_mask] = per_sensor_visibility[:, valid_y, valid_x]
        np.clip(coverage, 0.0, 1.0, out=coverage)
        return coverage

    state_index = {
        (int(xs[j]), int(ys[j])): j for j in np.flatnonzero(valid_state_mask)
    }
    for sensor_idx, sensor in enumerate(sensors):  # pragma: no cover
        sx, sy = sensor.location
        vis = _fallback_visibility(occupancy_grid, (sx, sy), ends).reshape(rows, cols)
        for x, y in state_index:
            if vis[y, x] <= 0.0:
                continue
            dx = float(x - sx)
            dy = float(y - sy)
            dist = float(np.hypot(dx, dy))
            if dist > sensor.range:
                continue
            if not angle_gate(
                src=sensor.location,
                dst=(x, y),
                direction_deg=sensor.direction_deg,
                fov_deg=sensor.fov_deg,
            ):
                continue
            coverage[sensor_idx, state_index[(x, y)]] = 1.0

    return coverage


def compute_linear_coverage_matrix(
    sensors: Sequence[SensorSpec],
    num_states: int,
) -> np.ndarray:
    """
    1-D coverage helper. Uses the x-coordinate of SensorSpec.location.
    """
    coverage = np.zeros((len(sensors), num_states), dtype=bool)
    for sensor_idx, sensor in enumerate(sensors):
        sx = int(sensor.location[0])
        for state in range(num_states):
            dx = abs(state - sx)
            if dx > sensor.range:
                continue
            if sensor.fov_deg < 180.0:
                # In 1-D, treat direction in [0,180) as +x and [180,360) as -x.
                facing_right = sensor.direction_deg < 180.0
                if facing_right and state < sx:
                    continue
                if not facing_right and state > sx:
                    continue
            coverage[sensor_idx, state] = True
    return coverage


def default_grid_sensors(grid_data: np.ndarray) -> List[SensorSpec]:
    sensors: List[SensorSpec] = []
    rows, cols = grid_data.shape
    idx = 0
    for y in range(rows):
        for x in range(cols):
            if grid_data[y, x] == 0:
                sensors.append(
                    SensorSpec(
                        id=f"s{idx}",
                        location=(x, y),
                        fov_deg=360.0,
                        range=0.0,
                        direction_deg=0.0,
                        energy_cost=1.0,
                    )
                )
                idx += 1
    return sensors


def default_linear_sensors(num_states: int) -> List[SensorSpec]:
    return [
        SensorSpec(
            id=f"s{i}",
            location=(i, 0),
            fov_deg=360.0,
            range=0.0,
            direction_deg=0.0,
            energy_cost=1.0,
        )
        for i in range(num_states)
    ]
