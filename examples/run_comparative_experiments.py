from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent


def _parse_csv_list(raw: str) -> list[str]:
    out: list[str] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(token)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiclass experiments for multiple selectors, then generate "
            "comparative plots/tables."
        )
    )
    parser.add_argument(
        "--selectors",
        type=str,
        default="greedy,q_mdp,random,mcts_oced",
        help=(
            "Comma-separated selector names. "
            "Defaults to greedy,q_mdp,random,mcts_oced."
        ),
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        default=THIS_DIR / "results" / "multiclass_hmm",
        help="Directory for per-selector experiment CSV outputs.",
    )
    parser.add_argument(
        "--comparative-results-dir",
        type=Path,
        default=THIS_DIR / "results",
        help="Root directory scanned by compare_experiment_results.py.",
    )
    parser.add_argument(
        "--comparative-output-dir",
        type=Path,
        default=THIS_DIR / "results" / "comparative",
        help="Directory for comparative plots/tables.",
    )
    parser.add_argument("--episodes-per-setting", type=int, default=100)
    parser.add_argument(
        "--lambda-values",
        type=str,
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--max-active-sensors", type=str, default="41")
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--mcts-iterations", type=int, default=128)
    parser.add_argument("--mcts-max-actions", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting in both per-selector and comparative scripts.",
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help="Skip LaTeX output in both per-selector and comparative scripts.",
    )
    parser.add_argument(
        "--skip-lower-bound",
        action="store_true",
        help="Skip lower-bound generation in per-selector runs.",
    )
    return parser


def _run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    selectors = _parse_csv_list(args.selectors)
    if not selectors:
        raise ValueError("at least one selector is required")

    exp_script = THIS_DIR / "multiclass_hmm_sensor_experiment.py"
    cmp_script = THIS_DIR / "compare_experiment_results.py"
    exp_out = Path(args.experiment_output_dir)
    exp_out.mkdir(parents=True, exist_ok=True)

    for selector in selectors:
        cmd = [
            sys.executable,
            str(exp_script),
            "--selector",
            selector,
            "--episodes-per-setting",
            str(int(args.episodes_per_setting)),
            "--lambda-values",
            str(args.lambda_values),
            "--max-active-sensors",
            str(args.max_active_sensors),
            "--planning-horizon",
            str(int(args.planning_horizon)),
            "--mcts-iterations",
            str(int(args.mcts_iterations)),
            "--mcts-max-actions",
            str(int(args.mcts_max_actions)),
            "--max-steps",
            str(int(args.max_steps)),
            "--seed",
            str(int(args.seed)),
            "--output-dir",
            str(exp_out),
        ]
        if bool(args.skip_plot):
            cmd.append("--skip-plot")
        if bool(args.skip_latex):
            cmd.append("--skip-latex")
        if bool(args.skip_lower_bound):
            cmd.append("--skip-lower-bound")
        _run_cmd(cmd)

    cmp_cmd = [
        sys.executable,
        str(cmp_script),
        "--results-dir",
        str(Path(args.comparative_results_dir)),
        "--output-dir",
        str(Path(args.comparative_output_dir)),
    ]
    if bool(args.skip_plot):
        cmp_cmd.append("--skip-plot")
    if bool(args.skip_latex):
        cmp_cmd.append("--skip-latex")
    _run_cmd(cmp_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
