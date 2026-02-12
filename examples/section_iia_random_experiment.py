from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import gymnasium as gym
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
SENSOR_WORLD_ROOT = THIS_DIR.parent
if str(SENSOR_WORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SENSOR_WORLD_ROOT))

import target_grid  # noqa: F401  # registers env IDs
from target_grid.envs import build_section_iia_41_world_parameters

from latex import write_table


@dataclass(frozen=True)
class ExperimentConfig:
    episodes_per_setting: int
    awake_probabilities: tuple[float, ...]
    lambda_energy: float
    max_steps: int
    seed: int
    output_dir: Path


def _parse_awake_probabilities(raw: str) -> tuple[float, ...]:
    probs: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        p = float(token)
        if p < 0.0 or p > 1.0:
            raise ValueError(f"awake probability must be in [0,1], got {p}")
        probs.append(p)
    if len(probs) == 0:
        raise ValueError("at least one awake probability must be provided")
    # deterministic order and dedupe
    return tuple(sorted(set(probs)))


def _ci95(values: pd.Series) -> float:
    n = int(values.shape[0])
    if n <= 1:
        return 0.0
    return 1.96 * float(values.std(ddof=1)) / float(np.sqrt(n))


def _sample_random_action(
    num_sensors: int,
    awake_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return (rng.random(num_sensors) < awake_probability).astype(np.int8)


def _run_single_episode(
    env: gym.Env,
    *,
    awake_probability: float,
    episode_seed: int,
) -> dict[str, float | int | bool]:
    obs, info = env.reset(seed=episode_seed)
    del obs, info

    total_reward = 0.0
    total_tracking_cost = 0.0
    total_energy_cost = 0.0
    total_active_sensors = 0.0
    total_total_cost = 0.0
    steps = 0

    unwrapped = env.unwrapped
    rng = np.random.default_rng(episode_seed + 7919)
    num_sensors = int(unwrapped.num_sensors)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = _sample_random_action(num_sensors, awake_probability, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        del obs
        steps += 1
        total_reward += float(reward)
        total_tracking_cost += float(info["tracking_cost"])
        total_energy_cost += float(info["energy_cost"])
        total_active_sensors += float(info["num_active_sensors"])
        total_total_cost += float(info["total_cost"])

    if truncated:
        raise RuntimeError(
            "episode truncated before target exit; increase max_steps for full runs"
        )

    final_state = int(unwrapped.target_states[0])
    exit_state = int(unwrapped.num_states - 1)
    reached_exit = bool(final_state == exit_state)
    if not reached_exit:
        raise RuntimeError(
            f"episode terminated in state {final_state}, expected exit state {exit_state}"
        )

    steps_f = float(steps)
    return {
        "steps_to_exit": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reached_exit": reached_exit,
        "total_reward": total_reward,
        "total_tracking_cost": total_tracking_cost,
        "total_energy_cost": total_energy_cost,
        "total_cost": total_total_cost,
        "active_sensors_per_step": total_active_sensors / steps_f,
        "tracking_error_per_step": total_tracking_cost / steps_f,
        "energy_cost_per_step": total_energy_cost / steps_f,
        "total_cost_per_step": total_total_cost / steps_f,
    }


def _run_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, float | int | bool | str]] = []
    base_rng = np.random.default_rng(config.seed)

    for awake_probability in config.awake_probabilities:
        world_parameters = build_section_iia_41_world_parameters(
            lambda_energy=config.lambda_energy,
            max_steps=config.max_steps,
            screen_width=1600,
            screen_height=180,
        )

        env = gym.make(
            "target_grid/SensorSchedulingLinear-v0",
            render_mode=None,
            world_parameters=world_parameters,
        )
        try:
            for episode_idx in range(config.episodes_per_setting):
                episode_seed = int(base_rng.integers(0, 2**31 - 1))
                stats = _run_single_episode(
                    env,
                    awake_probability=awake_probability,
                    episode_seed=episode_seed,
                )
                stats["experiment"] = "section_iia_random"
                stats["policy"] = "random_bernoulli"
                stats["awake_probability"] = float(awake_probability)
                stats["awake_probability_label"] = f"{awake_probability:.2f}"
                stats["episode_idx"] = int(episode_idx)
                stats["episode_seed"] = episode_seed
                stats["lambda_energy"] = float(config.lambda_energy)
                episode_rows.append(stats)
        finally:
            env.close()

    episodes_df = pd.DataFrame(episode_rows)
    if episodes_df.empty:
        raise RuntimeError("no episode results were generated")

    grouped = episodes_df.groupby("awake_probability", sort=True)
    summary_df = grouped.agg(
        num_episodes=("episode_idx", "count"),
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
    ).reset_index()
    summary_df["ci95_tracking_error_per_step"] = grouped["tracking_error_per_step"].apply(
        _ci95
    ).values
    summary_df["ci95_active_sensors_per_step"] = grouped["active_sensors_per_step"].apply(
        _ci95
    ).values
    summary_df["ci95_total_cost_per_step"] = grouped["total_cost_per_step"].apply(_ci95).values
    summary_df["awake_probability_label"] = summary_df["awake_probability"].map(
        lambda x: f"{float(x):.2f}"
    )
    summary_df["policy"] = "random_bernoulli"

    return episodes_df, summary_df


def _write_outputs(
    episodes_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    output_dir: Path,
    write_plot: bool,
    write_latex: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_csv = output_dir / "section_iia_random_episode_stats.csv"
    summary_csv = output_dir / "section_iia_random_summary_stats.csv"
    episodes_df.to_csv(episodes_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    if write_plot:
        from plots import plot

        plot_df = summary_df.sort_values("mean_active_sensors_per_step").copy()
        plot_path = output_dir / "section_iia_random_tracking_error_vs_sensors_awake.pdf"
        plot(
            x="mean_active_sensors_per_step",
            y="mean_tracking_error_per_step",
            hue="policy",
            data=plot_df,
            order=["random_bernoulli"],
            hue_order=["random_bernoulli"],
            x_label="Sensors awake per unit time",
            y_label="Tracking error per unit time",
            legend_location="best",
            plot_name=str(plot_path),
            plot_type="line",
            options={
                "width": 8,
                "height": 6,
                "legend_title": "Policy",
                "lineplot_kwargs": {"marker": "o", "sort": True, "errorbar": None},
            },
        )

    categories = {
        "L1": {
            "name": "Random $p_{awake}$",
            "column": "awake_probability_label",
            "labels": list(summary_df["awake_probability_label"]),
            "proper_name": {
                label: label for label in summary_df["awake_probability_label"]
            },
        },
    }
    columns = {
        "active_sensors_per_step": "Sensors Awake / Step",
        "tracking_error_per_step": "Tracking Error / Step",
        "energy_cost_per_step": "Energy Cost / Step",
        "total_cost_per_step": "Total Cost / Step",
        "steps_to_exit": "Steps to Exit",
    }
    column_properties = [
        {"type": "ci", "highlight": "none", "decimals": 2},
        {"type": "ci", "highlight": "min", "decimals": 3},
        {"type": "ci", "highlight": "none", "decimals": 3},
        {"type": "ci", "highlight": "min", "decimals": 3},
        {"type": "ci", "highlight": "min", "decimals": 1},
    ]

    if write_latex:
        latex_path = output_dir / "section_iia_random_results_table.tex"
        write_table(
            df=episodes_df,
            categories=categories,
            columns=columns,
            column_properties=column_properties,
            title="Section II-A Random Policy Results",
            caption=(
                "Random Bernoulli sensor scheduling on the Section II-A 41-state model. "
                "Each row aggregates full episodes that run until target exit."
            ),
            label="tab:section_iia_random_results",
            output_file_path=str(latex_path),
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Section II-A random-policy experiments (full episodes to exit), "
            "export CSV statistics, plot tradeoff curve, and produce LaTeX tables."
        )
    )
    parser.add_argument(
        "--episodes-per-setting",
        type=int,
        default=200,
        help="Number of episodes for each awake-probability setting.",
    )
    parser.add_argument(
        "--awake-probabilities",
        type=str,
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated Bernoulli awake probabilities for random actions.",
    )
    parser.add_argument(
        "--lambda-energy",
        type=float,
        default=0.2,
        help="Energy-weight parameter used by the environment.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Step cap per episode. Must be high enough to allow exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base RNG seed for reproducible experiment runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "results" / "section_iia_random",
        help="Directory for CSV, plot, and LaTeX outputs.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generating the tracking-error vs sensors-awake chart.",
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help="Skip generating the LaTeX table.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = ExperimentConfig(
        episodes_per_setting=int(args.episodes_per_setting),
        awake_probabilities=_parse_awake_probabilities(args.awake_probabilities),
        lambda_energy=float(args.lambda_energy),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        output_dir=Path(args.output_dir),
    )

    episodes_df, summary_df = _run_experiment(config)
    _write_outputs(
        episodes_df,
        summary_df,
        output_dir=config.output_dir,
        write_plot=not bool(args.skip_plot),
        write_latex=not bool(args.skip_latex),
    )

    print(f"Wrote episode CSV: {config.output_dir / 'section_iia_random_episode_stats.csv'}")
    print(f"Wrote summary CSV: {config.output_dir / 'section_iia_random_summary_stats.csv'}")
    if not bool(args.skip_plot):
        print(
            "Wrote plot: "
            f"{config.output_dir / 'section_iia_random_tracking_error_vs_sensors_awake.pdf'}"
        )
    if not bool(args.skip_latex):
        print(
            f"Wrote LaTeX table: {config.output_dir / 'section_iia_random_results_table.tex'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
