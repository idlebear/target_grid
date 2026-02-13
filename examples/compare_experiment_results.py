from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import pandas as pd

from latex import write_table


THIS_DIR = Path(__file__).resolve().parent


def _parse_dataset_and_default_method(stem: str) -> tuple[str, str]:
    if stem.startswith("multiclass_hmm_"):
        return "multiclass_hmm", stem.replace("multiclass_hmm_", "", 1)
    if stem.startswith("section_iia_random"):
        return "section_iia_random", "random_bernoulli"
    if stem.startswith("section_iia_lower_bound"):
        return "section_iia_random", "lower_bound_oac"
    return stem, stem


def _control_label_from_row(row: pd.Series) -> str:
    if "max_active_sensors" in row and pd.notna(row["max_active_sensors"]):
        return f"k={int(row['max_active_sensors'])}"
    if "lambda_energy" in row and pd.notna(row["lambda_energy"]):
        return f"lambda={float(row['lambda_energy']):.3f}"
    if "awake_probability_label" in row and pd.notna(row["awake_probability_label"]):
        return f"p={str(row['awake_probability_label'])}"
    if "awake_probability" in row and pd.notna(row["awake_probability"]):
        return f"p={float(row['awake_probability']):.2f}"
    return "all"


def _control_sort_key(label: str):
    s = str(label)
    if s.startswith("k="):
        try:
            return (0, float(s[2:]))
        except ValueError:
            return (0, float("inf"))
    if s.startswith("lambda="):
        try:
            return (1, float(s[len("lambda=") :]))
        except ValueError:
            return (1, float("inf"))
    if s.startswith("p="):
        try:
            return (2, float(s[2:]))
        except ValueError:
            return (2, float("inf"))
    return (3, s)


def _discover_files(results_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    summary_files = sorted(
        p
        for p in results_dir.rglob("*_summary_stats.csv")
        if not p.name.endswith("_class_summary_stats.csv")
    )
    episode_files = sorted(results_dir.rglob("*_episode_stats.csv"))
    class_summary_files = sorted(results_dir.rglob("*_class_summary_stats.csv"))
    return summary_files, episode_files, class_summary_files


def _load_summary_frames(summary_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in summary_files:
        df = pd.read_csv(path)
        stem = path.stem.replace("_summary_stats", "")
        dataset, default_method = _parse_dataset_and_default_method(stem)
        df["dataset"] = dataset
        if "selector" in df.columns:
            df["method"] = df["selector"].astype(str)
        elif "policy" in df.columns:
            df["method"] = df["policy"].astype(str)
        elif "method" not in df.columns:
            df["method"] = default_method
        df["method_label"] = df["dataset"].astype(str) + ":" + df["method"].astype(str)
        if "mean_active_sensors_per_step" not in df.columns and "active_sensors_per_step" in df.columns:
            df["mean_active_sensors_per_step"] = df["active_sensors_per_step"]
        if "mean_tracking_error_per_step" not in df.columns and "tracking_error_per_step" in df.columns:
            df["mean_tracking_error_per_step"] = df["tracking_error_per_step"]
        if "mean_total_cost_per_step" not in df.columns and "total_cost_per_step" in df.columns:
            df["mean_total_cost_per_step"] = df["total_cost_per_step"]
        df["control_label"] = df.apply(_control_label_from_row, axis=1)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def _load_episode_frames(episode_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in episode_files:
        df = pd.read_csv(path)
        stem = path.stem.replace("_episode_stats", "")
        dataset, default_method = _parse_dataset_and_default_method(stem)
        df["dataset"] = dataset
        if "selector" in df.columns:
            df["method"] = df["selector"].astype(str)
        elif "policy" in df.columns:
            df["method"] = df["policy"].astype(str)
        elif "method" not in df.columns:
            df["method"] = default_method
        df["method_label"] = df["dataset"].astype(str) + ":" + df["method"].astype(str)
        df["control_label"] = df.apply(_control_label_from_row, axis=1)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_class_summary_frames(class_summary_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in class_summary_files:
        df = pd.read_csv(path)
        stem = path.stem.replace("_class_summary_stats", "")
        dataset, default_method = _parse_dataset_and_default_method(stem)
        df["dataset"] = dataset
        if "selector" in df.columns:
            df["method"] = df["selector"].astype(str)
        elif "policy" in df.columns:
            df["method"] = df["policy"].astype(str)
        elif "method" not in df.columns:
            df["method"] = default_method
        df["method_label"] = df["dataset"].astype(str) + ":" + df["method"].astype(str)
        df["control_label"] = df.apply(_control_label_from_row, axis=1)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_comparative_plots(
    episodes_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    from plots import plot

    if summary_df.empty:
        return

    summary_df = summary_df.copy()
    summary_df = summary_df.sort_values("mean_active_sensors_per_step")
    method_order = sorted(summary_df["method_label"].unique())

    if {
        "mean_active_sensors_per_step",
        "mean_tracking_error_per_step",
    }.issubset(summary_df.columns):
        plot(
            x="mean_active_sensors_per_step",
            y="mean_tracking_error_per_step",
            hue="method_label",
            data=summary_df,
            order=method_order,
            hue_order=method_order,
            x_label="Sensors awake per unit time",
            y_label="Tracking error per unit time",
            legend_location="best",
            plot_name=str(
                output_dir / "comparative_tracking_error_vs_sensors_awake.pdf"
            ),
            plot_type="line",
            options={
                "width": 9,
                "height": 6,
                "legend_title": "Method",
                "lineplot_kwargs": {"marker": "o", "sort": True, "errorbar": None},
            },
        )

    if {
        "mean_active_sensors_per_step",
        "mean_total_cost_per_step",
    }.issubset(summary_df.columns):
        plot(
            x="mean_active_sensors_per_step",
            y="mean_total_cost_per_step",
            hue="method_label",
            data=summary_df,
            order=method_order,
            hue_order=method_order,
            x_label="Sensors awake per unit time",
            y_label="Total cost per unit time",
            legend_location="best",
            plot_name=str(output_dir / "comparative_total_cost_vs_sensors_awake.pdf"),
            plot_type="line",
            options={
                "width": 9,
                "height": 6,
                "legend_title": "Method",
                "lineplot_kwargs": {"marker": "o", "sort": True, "errorbar": None},
            },
        )

    if (
        not class_summary_df.empty
        and {
            "mean_active_sensors_per_step",
            "class_inference_accuracy",
            "target_class_idx",
            "method_label",
        }.issubset(class_summary_df.columns)
    ):
        cls_df = class_summary_df.copy()
        cls_df["method_class"] = (
            cls_df["method_label"].astype(str)
            + ":class"
            + cls_df["target_class_idx"].astype(str)
        )
        order = sorted(cls_df["method_class"].unique())
        plot(
            x="mean_active_sensors_per_step",
            y="class_inference_accuracy",
            hue="method_class",
            data=cls_df,
            order=order,
            hue_order=order,
            x_label="Sensors awake per unit time",
            y_label="Class inference accuracy",
            legend_location="best",
            plot_name=str(
                output_dir / "comparative_class_accuracy_vs_sensors_awake.pdf"
            ),
            plot_type="line",
            options={
                "width": 10,
                "height": 6,
                "legend_title": "Method:Class",
                "lineplot_kwargs": {"marker": "o", "sort": True, "errorbar": None},
            },
        )

    if (
        not episodes_df.empty
        and "action_selection_time_per_step_ms" in episodes_df.columns
    ):
        ep_df = episodes_df[
            ["method_label", "action_selection_time_per_step_ms"]
        ].copy()
        ep_df = ep_df.dropna(subset=["action_selection_time_per_step_ms"])
        if not ep_df.empty:
            method_order_ep = sorted(ep_df["method_label"].astype(str).unique())
            plot(
                x="method_label",
                y="action_selection_time_per_step_ms",
                hue="method_label",
                data=ep_df,
                order=method_order_ep,
                hue_order=method_order_ep,
                x_label="Method",
                y_label="Action Selection Time (ms/step)",
                legend_location="best",
                plot_name=str(
                    output_dir / "comparative_action_selection_time_violin.pdf"
                ),
                plot_type="violin",
                options={
                    "width": 10,
                    "height": 6,
                    "legend_title": "Method",
                    "label_fontsize": 13,
                },
            )


def _write_latex_tables(
    episodes_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if episodes_df.empty and summary_df.empty:
        return

    # Prefer episode-level data for CI estimates.
    source_df = episodes_df if not episodes_df.empty else summary_df

    for dataset, dataset_df in source_df.groupby("dataset", sort=True):
        methods = sorted(dataset_df["method"].astype(str).unique())
        controls = sorted(
            dataset_df["control_label"].astype(str).unique(),
            key=_control_sort_key,
        )

        categories = {
            "L1": {
                "name": "Method",
                "column": "method",
                "labels": methods,
                "proper_name": {m: m for m in methods},
            },
            "L2": {
                "name": "Setting",
                "column": "control_label",
                "labels": controls,
                "proper_name": {c: c for c in controls},
            },
        }

        columns = {}
        column_properties = []
        if "tracking_error_per_step" in dataset_df.columns:
            columns["tracking_error_per_step"] = "Tracking Error / Step"
            column_properties.append({"type": "ci", "highlight": "min", "decimals": 3})
        elif "mean_tracking_error_per_step" in dataset_df.columns:
            columns["mean_tracking_error_per_step"] = "Tracking Error / Step"
            column_properties.append({"type": "ci", "highlight": "min", "decimals": 3})

        if "active_sensors_per_step" in dataset_df.columns:
            columns["active_sensors_per_step"] = "Sensors Awake / Step"
            column_properties.append(
                {"type": "ci", "highlight": "none", "decimals": 2}
            )
        elif "mean_active_sensors_per_step" in dataset_df.columns:
            columns["mean_active_sensors_per_step"] = "Sensors Awake / Step"
            column_properties.append(
                {"type": "ci", "highlight": "none", "decimals": 2}
            )

        if "total_cost_per_step" in dataset_df.columns:
            columns["total_cost_per_step"] = "Total Cost / Step"
            column_properties.append({"type": "ci", "highlight": "min", "decimals": 3})
        elif "mean_total_cost_per_step" in dataset_df.columns:
            columns["mean_total_cost_per_step"] = "Total Cost / Step"
            column_properties.append({"type": "ci", "highlight": "min", "decimals": 3})

        if "class_inference_correct" in dataset_df.columns:
            columns["class_inference_correct"] = "Class Correct"
            column_properties.append({"type": "ci", "highlight": "max", "decimals": 3})
        elif "class_inference_accuracy" in dataset_df.columns:
            columns["class_inference_accuracy"] = "Class Accuracy"
            column_properties.append({"type": "ci", "highlight": "max", "decimals": 3})

        if not columns:
            continue

        out_path = output_dir / f"comparative_{dataset}_results_table.tex"
        write_table(
            df=dataset_df,
            categories=categories,
            columns=columns,
            column_properties=column_properties,
            title=f"Comparative Results ({dataset})",
            caption=(
                "Comparative metrics aggregated by method and experiment setting. "
                "Values are reported as mean and confidence interval."
            ),
            label=f"tab:comparative_{dataset}",
            output_file_path=str(out_path),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process experiment CSV outputs and generate comparative plots + LaTeX tables."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=THIS_DIR / "results",
        help="Root directory containing experiment CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "results" / "comparative",
        help="Directory where comparative outputs are written.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip comparative plot generation.",
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help="Skip LaTeX table generation.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    summary_files, episode_files, class_summary_files = _discover_files(args.results_dir)
    if not summary_files and not episode_files:
        raise RuntimeError(
            f"no experiment CSV files found under '{args.results_dir}'"
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _load_summary_frames(summary_files)
    episodes_df = _load_episode_frames(episode_files)
    class_summary_df = _load_class_summary_frames(class_summary_files)

    # Persist combined CSVs for reproducibility.
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "comparative_combined_summary.csv", index=False)
    if not episodes_df.empty:
        episodes_df.to_csv(output_dir / "comparative_combined_episodes.csv", index=False)
    if not class_summary_df.empty:
        class_summary_df.to_csv(
            output_dir / "comparative_combined_class_summary.csv", index=False
        )

    if not bool(args.skip_plot):
        _write_comparative_plots(episodes_df, summary_df, class_summary_df, output_dir)
    if not bool(args.skip_latex):
        _write_latex_tables(episodes_df, summary_df, output_dir)

    print(f"Wrote outputs to: {output_dir}")
    if not summary_df.empty:
        print(f"Combined summary rows: {len(summary_df)}")
    if not episodes_df.empty:
        print(f"Combined episode rows: {len(episodes_df)}")
    if not class_summary_df.empty:
        print(f"Combined class summary rows: {len(class_summary_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
