from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import gymnasium as gym
import numpy as np
import pandas as pd

import target_grid  # noqa: F401  # registers env IDs in the installed environment
from target_grid.envs import (
    SensorSchedulingLinearEnv,
    build_section_iia_41_world_parameters,
)

from latex import write_table
from lower_bounds import section_iia_observable_after_control_lower_bound
from sensor_selection import get_selector


THIS_DIR = Path(__file__).resolve().parent

SECTION_IIA_OFFSETS = [-3, -2, -1, 0, 1, 2, 3]
DEFAULT_CLASS_TRANSITION_PROBABILITIES = (
    (0.23, 0.10, 0.01, 0.33, 0.06, 0.05, 0.22),  # baseline
    (0.35, 0.18, 0.08, 0.20, 0.08, 0.06, 0.05),  # left-biased
    (0.05, 0.06, 0.08, 0.20, 0.08, 0.18, 0.35),  # right-biased
)
DEFAULT_CLASS_PRIOR = (1.0, 0.0, 0.0)  # all mass on baseline class by default


def _load_hmm_class():
    try:
        from hmm import HMM  # type: ignore

        return HMM
    except ModuleNotFoundError:
        python_root = THIS_DIR.parent.parent
        if str(python_root) not in sys.path:
            sys.path.insert(0, str(python_root))
        hmm_path = python_root / "hmm.py"
        spec = importlib.util.spec_from_file_location("_local_hmm_module", hmm_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load HMM module from {hmm_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.HMM


HMM = _load_hmm_class()


@dataclass(frozen=True)
class ExperimentConfig:
    selector_name: str
    max_active_sensors_cap: int
    lambda_values: tuple[float, ...]
    episodes_per_setting: int
    class_transition_probabilities: tuple[tuple[float, ...], ...]
    class_prior: tuple[float, ...]
    initial_belief: tuple[float, ...]
    max_steps: int
    seed: int
    output_dir: Path
    write_plot: bool
    write_latex: bool
    write_lower_bound: bool


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        values.append(float(token))
    return tuple(values)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        values.append(int(token))
    return tuple(values)


def _parse_lambda_values(raw: str) -> tuple[float, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        value = float(token)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"lambda must be in (0,1], got {value}")
        values.append(value)
    if len(values) == 0:
        raise ValueError("at least one lambda value is required")
    return tuple(sorted(set(values)))


def _parse_class_transition_probabilities(raw: str) -> tuple[tuple[float, ...], ...]:
    classes: list[tuple[float, ...]] = []
    for cls in raw.split(";"):
        cls = cls.strip()
        if cls == "":
            continue
        probs = _parse_float_tuple(cls)
        classes.append(probs)
    return tuple(classes)


def _parse_initial_belief(raw: str, num_states: int) -> tuple[float, ...]:
    text = str(raw).strip()
    key = text.lower()
    if key in {"center_delta", "delta_center"}:
        arr = np.zeros((num_states,), dtype=np.float64)
        arr[(num_states - 1) // 2] = 1.0
        return tuple(arr.tolist())
    if key in {"uniform", "uniform_all"}:
        arr = np.full((num_states,), 1.0 / float(num_states), dtype=np.float64)
        return tuple(arr.tolist())
    if key in {"uniform_non_exit", "uniform_non_absorbing"}:
        arr = np.zeros((num_states,), dtype=np.float64)
        if num_states <= 1:
            arr[0] = 1.0
        else:
            arr[: num_states - 1] = 1.0 / float(num_states - 1)
        return tuple(arr.tolist())
    if key.startswith("delta:"):
        idx = int(key.split(":", 1)[1])
        if idx < 0 or idx >= num_states:
            raise ValueError(f"delta index out of bounds: {idx}")
        arr = np.zeros((num_states,), dtype=np.float64)
        arr[idx] = 1.0
        return tuple(arr.tolist())

    probs = _parse_float_tuple(text)
    if len(probs) == num_states - 1:
        probs = tuple(list(probs) + [0.0])
    if len(probs) != num_states:
        raise ValueError(
            "initial belief must have length num_states (or num_states-1 to exclude exit)"
        )
    arr = np.asarray(probs, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("initial belief probabilities must be non-negative")
    s = float(np.sum(arr))
    if s <= 0.0:
        raise ValueError("initial belief must sum to > 0")
    arr /= s
    return tuple(arr.tolist())


def _normalize_probabilities(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1D list")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    s = float(arr.sum())
    if s <= 0.0:
        raise ValueError(f"{name} sum must be > 0")
    arr /= s
    return tuple(arr.tolist())


def _build_candidate_transition_matrices(
    class_transition_probabilities: tuple[tuple[float, ...], ...],
) -> np.ndarray:
    matrices = []
    for probs in class_transition_probabilities:
        probs = tuple(float(x) for x in probs)
        if len(probs) != len(SECTION_IIA_OFFSETS):
            raise ValueError(
                "each target class transition vector must match "
                f"offset count={len(SECTION_IIA_OFFSETS)}"
            )
        if not np.isclose(np.sum(np.asarray(probs, dtype=np.float64)), 1.0, atol=1e-8):
            raise ValueError("each target class transition vector must sum to 1")
        T = SensorSchedulingLinearEnv._build_transition_matrix(
            num_network_states=41,
            offsets=list(SECTION_IIA_OFFSETS),
            probabilities=list(probs),
            boundary_behavior="exit",
            absorbing_states={41},
        )
        matrices.append(T)
    return np.asarray(matrices, dtype=np.float64)


def _build_emission_matrix_for_action(
    action: np.ndarray, coverage_matrix: np.ndarray
) -> np.ndarray:
    num_states = int(coverage_matrix.shape[1])
    num_observations = num_states + 1
    erasure_obs = num_states

    emission = np.zeros((num_states, num_observations), dtype=np.float64)
    if np.any(action > 0):
        covered = np.any(coverage_matrix[action > 0, :], axis=0)
    else:
        covered = np.zeros((num_states,), dtype=bool)

    for s in range(num_states):
        if covered[s]:
            emission[s, s] = 1.0
        else:
            emission[s, erasure_obs] = 1.0
    return emission


def _extract_observation_index(obs: dict, num_states: int) -> int:
    valid = np.asarray(obs["measurement_valid"], dtype=np.int8).reshape(1, -1)[0]
    measurements = np.asarray(obs["measurements"], dtype=np.float64).reshape(1, -1)[0]
    valid_indices = np.flatnonzero(valid > 0)
    if valid_indices.size == 0:
        return num_states  # erasure
    state_idx = int(round(float(measurements[int(valid_indices[0])])))
    if state_idx < 0 or state_idx >= num_states:
        return num_states
    return state_idx


def _entropy(p: np.ndarray) -> float:
    q = np.asarray(p, dtype=np.float64)
    q = np.clip(q, 1e-12, 1.0)
    return float(-np.sum(q * np.log(q)))


def _run_single_episode(
    *,
    selector_name: str,
    max_active_sensors_cap: int,
    class_idx: int,
    episode_seed: int,
    class_prior: tuple[float, ...],
    candidate_transition_matrices: np.ndarray,
    class_transition_probabilities: tuple[tuple[float, ...], ...],
    initial_belief: tuple[float, ...],
    lambda_energy: float,
    max_steps: int,
) -> dict[str, float | int | bool | str]:
    selector_fn = get_selector(selector_name)
    world_parameters = build_section_iia_41_world_parameters(
        lambda_energy=lambda_energy,
        max_steps=max_steps,
        screen_width=1400,
        screen_height=180,
    )
    world_parameters["transition_probabilities"] = list(
        class_transition_probabilities[class_idx]
    )
    world_parameters["sample_initial_state_from_belief"] = True
    world_parameters["initial_target_states"] = None
    world_parameters["initial_belief"] = list(initial_belief)

    env = gym.make(
        "target_grid/SensorSchedulingLinear-v0",
        render_mode=None,
        world_parameters=world_parameters,
    )

    total_reward = 0.0
    total_tracking_cost = 0.0
    total_energy_cost = 0.0
    total_total_cost = 0.0
    total_active_sensors = 0.0
    total_action_selection_time_s = 0.0
    steps = 0

    try:
        obs, info = env.reset(seed=episode_seed)
        del info
        unwrapped = env.unwrapped
        initial_true_state = int(unwrapped.target_states[0])
        coverage_matrix = np.asarray(unwrapped.coverage_matrix, dtype=bool)
        num_states = int(unwrapped.num_states)
        num_sensors = int(unwrapped.num_sensors)
        k = max(0, min(int(max_active_sensors_cap), num_sensors))
        sensor_energy_costs = np.asarray(unwrapped.sensor_energy_costs, dtype=np.float64)

        state_prior = np.asarray(initial_belief, dtype=np.float64).reshape(-1)
        if state_prior.shape[0] != num_states:
            raise ValueError(
                "initial_belief length does not match environment state count"
            )
        state_prior[state_prior < 0.0] = 0.0
        if state_prior.sum() <= 0.0:
            raise ValueError("initial_belief must contain positive probability mass")
        state_prior /= state_prior.sum()

        hmm = HMM(
            num_states=num_states,
            num_observations=num_states + 1,  # state index or erasure
            num_modes=int(candidate_transition_matrices.shape[0]),
            transitions=candidate_transition_matrices,
            emission_probabilities=np.full(
                (num_states, num_states + 1),
                1.0 / float(num_states + 1),
                dtype=np.float64,
            ),
            distributions={
                "state": state_prior,
                "mode": np.asarray(class_prior, dtype=np.float64),
            },
        )

        selector_rng = np.random.default_rng(episode_seed + 9973)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            t0 = perf_counter()
            action = selector_fn(
                np.asarray(hmm.state_distribution, dtype=np.float64).copy(),
                coverage_matrix=coverage_matrix,
                max_active_sensors=k,
                rng=selector_rng,
                hmm=hmm,
                step_index=int(steps),
                initial_true_state=int(initial_true_state),
                lambda_energy=float(lambda_energy),
                sensor_energy_costs=sensor_energy_costs,
            )
            total_action_selection_time_s += perf_counter() - t0
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            total_reward += float(reward)
            total_tracking_cost += float(info["tracking_cost"])
            total_energy_cost += float(info["energy_cost"])
            total_total_cost += float(info["total_cost"])
            total_active_sensors += float(info["num_active_sensors"])

            observation_idx = _extract_observation_index(obs, num_states)
            emission = _build_emission_matrix_for_action(action, coverage_matrix)
            hmm.forward_step(
                observation=int(observation_idx),
                emission_probabilities=emission,
            )

        if truncated:
            raise RuntimeError(
                "episode truncated before exit; increase --max-steps for full runs"
            )

        true_class_idx = int(class_idx)
        inferred_class_idx = int(np.argmax(hmm.mode_distribution))
        class_posterior_true = float(hmm.mode_distribution[true_class_idx])
        steps_f = float(steps)

        return {
            "selector": selector_name,
            "max_active_sensors_cap": int(k),
            "lambda_energy": float(lambda_energy),
            "episode_seed": int(episode_seed),
            "target_class_idx": true_class_idx,
            "initial_true_state": initial_true_state,
            "inferred_class_idx": inferred_class_idx,
            "class_inference_correct": bool(inferred_class_idx == true_class_idx),
            "class_posterior_true": class_posterior_true,
            "final_state_entropy": _entropy(hmm.state_distribution),
            "final_mode_entropy": _entropy(hmm.mode_distribution),
            "steps_to_exit": int(steps),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "total_reward": float(total_reward),
            "total_tracking_cost": float(total_tracking_cost),
            "total_energy_cost": float(total_energy_cost),
            "total_cost": float(total_total_cost),
            "active_sensors_per_step": float(total_active_sensors / steps_f),
            "tracking_error_per_step": float(total_tracking_cost / steps_f),
            "energy_cost_per_step": float(total_energy_cost / steps_f),
            "total_cost_per_step": float(total_total_cost / steps_f),
            "total_action_selection_time_s": float(total_action_selection_time_s),
            "action_selection_time_per_step_ms": float(
                1000.0 * total_action_selection_time_s / steps_f
            ),
        }
    finally:
        env.close()


def _ci95(values: pd.Series) -> float:
    n = int(values.shape[0])
    if n <= 1:
        return 0.0
    return 1.96 * float(values.std(ddof=1)) / float(np.sqrt(n))


def _run_experiment(
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_transition_matrices = _build_candidate_transition_matrices(
        config.class_transition_probabilities
    )
    rng = np.random.default_rng(config.seed)
    episode_plan = [
        (
            int(rng.choice(len(config.class_prior), p=np.asarray(config.class_prior))),
            int(rng.integers(0, 2**31 - 1)),
        )
        for _ in range(config.episodes_per_setting)
    ]

    rows: list[dict[str, float | int | bool | str]] = []
    for lambda_energy in config.lambda_values:
        for episode_idx, (class_idx, episode_seed) in enumerate(episode_plan):
            row = _run_single_episode(
                selector_name=config.selector_name,
                max_active_sensors_cap=int(config.max_active_sensors_cap),
                class_idx=class_idx,
                episode_seed=episode_seed,
                class_prior=config.class_prior,
                candidate_transition_matrices=candidate_transition_matrices,
                class_transition_probabilities=config.class_transition_probabilities,
                initial_belief=config.initial_belief,
                lambda_energy=float(lambda_energy),
                max_steps=config.max_steps,
            )
            row["episode_idx"] = int(episode_idx)
            rows.append(row)

    episodes_df = pd.DataFrame(rows)
    if episodes_df.empty:
        raise RuntimeError("no episode data generated")

    summary_df = (
        episodes_df.groupby(["selector", "lambda_energy"], sort=True)
        .agg(
            num_episodes=("episode_idx", "count"),
            max_active_sensors_cap=("max_active_sensors_cap", "first"),
            mean_steps_to_exit=("steps_to_exit", "mean"),
            std_steps_to_exit=("steps_to_exit", "std"),
            mean_active_sensors_per_step=("active_sensors_per_step", "mean"),
            std_active_sensors_per_step=("active_sensors_per_step", "std"),
            mean_tracking_error_per_step=("tracking_error_per_step", "mean"),
            std_tracking_error_per_step=("tracking_error_per_step", "std"),
            mean_energy_cost_per_step=("energy_cost_per_step", "mean"),
            std_energy_cost_per_step=("energy_cost_per_step", "std"),
            mean_total_cost_per_step=("total_cost_per_step", "mean"),
            std_total_cost_per_step=("total_cost_per_step", "std"),
            mean_action_selection_time_per_step_ms=(
                "action_selection_time_per_step_ms",
                "mean",
            ),
            std_action_selection_time_per_step_ms=(
                "action_selection_time_per_step_ms",
                "std",
            ),
            mean_class_posterior_true=("class_posterior_true", "mean"),
            class_inference_accuracy=("class_inference_correct", "mean"),
        )
        .reset_index()
    )
    summary_df["ci95_tracking_error_per_step"] = (
        episodes_df.groupby(["selector", "lambda_energy"], sort=True)[
            "tracking_error_per_step"
        ]
        .apply(_ci95)
        .values
    )
    summary_df["ci95_active_sensors_per_step"] = (
        episodes_df.groupby(["selector", "lambda_energy"], sort=True)[
            "active_sensors_per_step"
        ]
        .apply(_ci95)
        .values
    )
    summary_df["ci95_action_selection_time_per_step_ms"] = (
        episodes_df.groupby(["selector", "lambda_energy"], sort=True)[
            "action_selection_time_per_step_ms"
        ]
        .apply(_ci95)
        .values
    )

    class_summary_df = (
        episodes_df.groupby(
            ["selector", "lambda_energy", "target_class_idx"], sort=True
        )
        .agg(
            num_episodes=("episode_idx", "count"),
            max_active_sensors_cap=("max_active_sensors_cap", "first"),
            mean_tracking_error_per_step=("tracking_error_per_step", "mean"),
            mean_active_sensors_per_step=("active_sensors_per_step", "mean"),
            class_inference_accuracy=("class_inference_correct", "mean"),
            mean_class_posterior_true=("class_posterior_true", "mean"),
        )
        .reset_index()
    )

    return episodes_df, summary_df, class_summary_df


def _compute_lower_bound_summary(
    config: ExperimentConfig,
) -> pd.DataFrame:
    candidate_transition_matrices = _build_candidate_transition_matrices(
        config.class_transition_probabilities
    )
    world_parameters = build_section_iia_41_world_parameters(
        lambda_energy=float(config.lambda_values[0]),
        max_steps=config.max_steps,
        screen_width=1400,
        screen_height=180,
    )
    env = SensorSchedulingLinearEnv(render_mode=None, world_parameters=world_parameters)
    try:
        start_state = int(np.argmax(np.asarray(config.initial_belief, dtype=np.float64)))
        class_rows: list[pd.DataFrame] = []
        for class_idx in range(candidate_transition_matrices.shape[0]):
            class_df = section_iia_observable_after_control_lower_bound(
                transition_matrix=np.asarray(
                    candidate_transition_matrices[class_idx], dtype=np.float64
                ),
                coverage_matrix=np.asarray(env.coverage_matrix, dtype=bool),
                absorbing_states=set(int(s) for s in env.absorbing_states),
                start_state=start_state,
                lambda_values=config.lambda_values,
                start_distribution=config.initial_belief,
            )
            class_df["target_class_idx"] = int(class_idx)
            class_df["class_prior"] = float(config.class_prior[class_idx])
            class_rows.append(class_df)
    finally:
        env.close()

    per_class_df = pd.concat(class_rows, ignore_index=True)
    rows: list[dict[str, float | int | str]] = []
    for lam in config.lambda_values:
        block = per_class_df[
            np.isclose(
                per_class_df["lambda_energy"].to_numpy(dtype=np.float64),
                float(lam),
            )
        ].copy()
        if block.empty:
            continue
        w = block["class_prior"].to_numpy(dtype=np.float64)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w = np.ones_like(w, dtype=np.float64) / float(len(w))
        else:
            w = w / w_sum

        def wavg(col: str) -> float:
            v = block[col].to_numpy(dtype=np.float64)
            return float(np.dot(w, v))

        rows.append(
            {
                "selector": "lower_bound_oac",
                "lambda_energy": float(lam),
                "num_episodes": 0,
                "max_active_sensors_cap": int(config.max_active_sensors_cap),
                "mean_steps_to_exit": wavg("expected_steps_to_exit"),
                "std_steps_to_exit": 0.0,
                "mean_active_sensors_per_step": wavg("mean_active_sensors_per_step"),
                "std_active_sensors_per_step": 0.0,
                "mean_tracking_error_per_step": wavg("mean_tracking_error_per_step"),
                "std_tracking_error_per_step": 0.0,
                "mean_energy_cost_per_step": wavg("mean_energy_cost_per_step"),
                "std_energy_cost_per_step": 0.0,
                "mean_total_cost_per_step": wavg("mean_total_cost_per_step"),
                "std_total_cost_per_step": 0.0,
                "mean_class_posterior_true": np.nan,
                "class_inference_accuracy": np.nan,
                "ci95_tracking_error_per_step": 0.0,
                "ci95_active_sensors_per_step": 0.0,
                "lower_bound_method": "observable_after_control",
            }
        )

    return pd.DataFrame(rows).sort_values("lambda_energy").reset_index(drop=True)


def _write_outputs(
    *,
    config: ExperimentConfig,
    episodes_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    lower_bound_summary_df: pd.DataFrame | None,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"multiclass_hmm_{config.selector_name}"
    episodes_path = config.output_dir / f"{stem}_episode_stats.csv"
    summary_path = config.output_dir / f"{stem}_summary_stats.csv"
    class_summary_path = config.output_dir / f"{stem}_class_summary_stats.csv"
    episodes_df.to_csv(episodes_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    class_summary_df.to_csv(class_summary_path, index=False)
    if lower_bound_summary_df is not None and not lower_bound_summary_df.empty:
        lb_stem = "multiclass_hmm_lower_bound"
        lb_summary_path = config.output_dir / f"{lb_stem}_summary_stats.csv"
        lower_bound_summary_df.to_csv(lb_summary_path, index=False)

    if config.write_plot:
        from plots import plot

        plot_frames = [
            summary_df[
                [
                    "selector",
                    "mean_active_sensors_per_step",
                    "mean_tracking_error_per_step",
                ]
            ].copy()
        ]
        selector_order = [config.selector_name]
        if lower_bound_summary_df is not None and not lower_bound_summary_df.empty:
            plot_frames.append(
                lower_bound_summary_df[
                    [
                        "selector",
                        "mean_active_sensors_per_step",
                        "mean_tracking_error_per_step",
                    ]
                ].copy()
            )
            selector_order.append("lower_bound_oac")

        plot_df = pd.concat(plot_frames, ignore_index=True).sort_values(
            "mean_active_sensors_per_step"
        )
        plot_path = config.output_dir / f"{stem}_tracking_error_vs_sensors_awake.pdf"
        plot(
            x="mean_active_sensors_per_step",
            y="mean_tracking_error_per_step",
            hue="selector",
            data=plot_df,
            order=selector_order,
            hue_order=selector_order,
            x_label="Sensors awake per unit time",
            y_label="Tracking error per unit time",
            legend_location="best",
            plot_name=str(plot_path),
            plot_type="line",
            options={
                "width": 8,
                "height": 6,
                "legend_title": "Selector",
                "lineplot_kwargs": {"marker": "o", "sort": True, "errorbar": None},
            },
        )

    if config.write_latex:
        lambda_labels = [f"{float(v):.3f}" for v in summary_df["lambda_energy"]]
        categories = {
            "L1": {
                "name": "Selector",
                "column": "selector",
                "labels": [config.selector_name],
                "proper_name": {config.selector_name: config.selector_name},
            },
            "L2": {
                "name": "$\\lambda$ (energy weight)",
                "column": "lambda_energy",
                "labels": list(summary_df["lambda_energy"]),
                "proper_name": dict(zip(summary_df["lambda_energy"], lambda_labels)),
            },
        }
        columns = {
            "tracking_error_per_step": "Tracking Error / Step",
            "active_sensors_per_step": "Sensors Awake / Step",
            "energy_cost_per_step": "Energy Cost / Step",
            "total_cost_per_step": "Total Cost / Step",
            "class_inference_correct": "Class Correct",
        }
        column_properties = [
            {"type": "ci", "highlight": "min", "decimals": 3},
            {"type": "ci", "highlight": "none", "decimals": 2},
            {"type": "ci", "highlight": "none", "decimals": 3},
            {"type": "ci", "highlight": "min", "decimals": 3},
            {"type": "ci", "highlight": "max", "decimals": 3},
        ]
        latex_path = config.output_dir / f"{stem}_results_table.tex"
        write_table(
            df=episodes_df,
            categories=categories,
            columns=columns,
            column_properties=column_properties,
            title="Multiclass HMM Sensor Scheduling Results",
            caption=(
                "Multiclass target experiment with class-conditioned transition models, "
                "HMM belief tracking, and pluggable sensor selection."
            ),
            label=f"tab:{stem}",
            output_file_path=str(latex_path),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multiclass target experiment with class priors, HMM state/mode tracking, "
            "and pluggable sensor selectors (greedy or random)."
        )
    )
    parser.add_argument(
        "--selector",
        type=str,
        choices=["greedy", "q_mdp", "random"],
        default="greedy",
        help="Sensor selection module to use.",
    )
    parser.add_argument(
        "--max-active-sensors",
        type=str,
        default="41",
        help=(
            "Comma-separated k values; the maximum value is used as a hard cap "
            "for active sensors in each step."
        ),
    )
    parser.add_argument(
        "--episodes-per-setting",
        type=int,
        default=100,
        help="Number of episodes for each lambda setting.",
    )
    parser.add_argument(
        "--class-transition-probabilities",
        type=str,
        default=";".join(
            ",".join(f"{x:.8f}" for x in cls)
            for cls in DEFAULT_CLASS_TRANSITION_PROBABILITIES
        ),
        help=(
            "Semicolon-separated class transition probability vectors. "
            "Each class vector is comma-separated over offsets [-3,-2,-1,0,1,2,3]."
        ),
    )
    parser.add_argument(
        "--class-prior",
        type=str,
        default=",".join(f"{x:.8f}" for x in DEFAULT_CLASS_PRIOR),
        help="Comma-separated class prior probabilities.",
    )
    parser.add_argument(
        "--initial-belief",
        type=str,
        default="center_delta",
        help=(
            "Initial belief over states. Presets: center_delta, uniform_non_exit, "
            "uniform, delta:<idx>, or comma-separated probabilities."
        ),
    )
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated lambda sweep values in (0,1].",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Episode cap (must be large enough to allow exit).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base RNG seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "results" / "multiclass_hmm",
        help="Directory for CSV/plot/LaTeX outputs.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip chart generation.",
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help="Skip LaTeX table generation.",
    )
    parser.add_argument(
        "--skip-lower-bound",
        action="store_true",
        help="Skip lower-bound generation and overlay.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    class_transition_probabilities = _parse_class_transition_probabilities(
        args.class_transition_probabilities
    )
    if len(class_transition_probabilities) == 0:
        raise ValueError("at least one target class transition model is required")
    for probs in class_transition_probabilities:
        _normalize_probabilities(probs, "class transition probabilities")
    class_transition_probabilities = tuple(
        _normalize_probabilities(probs, "class transition probabilities")
        for probs in class_transition_probabilities
    )

    class_prior = _normalize_probabilities(
        _parse_float_tuple(args.class_prior), "class prior"
    )
    if len(class_prior) != len(class_transition_probabilities):
        raise ValueError(
            "class prior length must match number of class transition models"
        )

    k_values = _parse_int_tuple(args.max_active_sensors)
    if len(k_values) == 0:
        raise ValueError("at least one k value is required for the cap")
    if any(k < 0 for k in k_values):
        raise ValueError("k values must be non-negative")
    max_active_sensors_cap = int(max(k_values))

    lambda_values = _parse_lambda_values(args.lambda_values)

    num_states = 42  # 41 network states + 1 exit state for Section II-A setup
    initial_belief = _parse_initial_belief(args.initial_belief, num_states=num_states)

    config = ExperimentConfig(
        selector_name=str(args.selector),
        max_active_sensors_cap=max_active_sensors_cap,
        lambda_values=lambda_values,
        episodes_per_setting=int(args.episodes_per_setting),
        class_transition_probabilities=class_transition_probabilities,
        class_prior=class_prior,
        initial_belief=initial_belief,
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        output_dir=Path(args.output_dir),
        write_plot=not bool(args.skip_plot),
        write_latex=not bool(args.skip_latex),
        write_lower_bound=not bool(args.skip_lower_bound),
    )

    episodes_df, summary_df, class_summary_df = _run_experiment(config)
    lower_bound_summary_df = None
    if config.write_lower_bound:
        lower_bound_summary_df = _compute_lower_bound_summary(config)
    _write_outputs(
        config=config,
        episodes_df=episodes_df,
        summary_df=summary_df,
        class_summary_df=class_summary_df,
        lower_bound_summary_df=lower_bound_summary_df,
    )

    stem = f"multiclass_hmm_{config.selector_name}"
    print(f"Wrote episode CSV: {config.output_dir / f'{stem}_episode_stats.csv'}")
    print(f"Wrote summary CSV: {config.output_dir / f'{stem}_summary_stats.csv'}")
    print(
        f"Wrote class summary CSV: {config.output_dir / f'{stem}_class_summary_stats.csv'}"
    )
    if lower_bound_summary_df is not None and not lower_bound_summary_df.empty:
        print(
            "Wrote lower-bound summary CSV: "
            f"{config.output_dir / 'multiclass_hmm_lower_bound_summary_stats.csv'}"
        )
    if config.write_plot:
        print(
            "Wrote plot: "
            f"{config.output_dir / f'{stem}_tracking_error_vs_sensors_awake.pdf'}"
        )
    if config.write_latex:
        print(f"Wrote LaTeX table: {config.output_dir / f'{stem}_results_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
