from enum import IntEnum
import os

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import matplotlib as mpl

mpl.use("pdf")

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import distinctipy
# import scienceplots
# plt.style.use(['science', 'ieee'])

# # width as measured in inkscape
# width = 10  # 3.487
# height = width  # / 1.5


# TABLE formatting
TITLE = "Stochastic Trials"
CAPTION = "none"


class Tags(IntEnum):
    METHOD = 0
    COST_FN = 1
    STEPS = 2
    HORIZON = 3
    SAMPLES = 4
    SEED = 5


def plot(
    x,
    y,
    hue,
    data,
    order,
    hue_order=None,
    colors=None,
    x_label=None,
    y_label=None,
    x_lim=None,
    y_lim=None,
    legend_location="best",
    plot_name="default.pdf",
    plot_type="line",
    annotations=None,
    overlays=None,
    add_emphasis=False,
    options=None,
):

    if hue_order is None:
        hue_order = order

    y_scale = "linear"
    x_scale = "linear"
    flierprops = None
    label_fontsize = 18  # defaults; can be overridden via options
    legend_fontsize_override = None
    legend_title = None
    legend_title_fontsize = None
    width = 10
    height = 10
    equal = False
    if options is not None:
        if "flierprops" in options:
            flierprops = options["flierprops"]
            if not isinstance(flierprops, dict):
                flierprops = dict(
                    marker="o",
                    markerfacecolor="grey",
                    markersize=2,
                    alpha=0.5,
                    linestyle="none",
                )
        y_scale = options.get("y_scale", "linear")
        x_scale = options.get("x_scale", "linear")
        if y_scale not in ["linear", "log"]:
            raise ValueError(f"Unknown y_scale: {y_scale}")
        if x_scale not in ["linear", "log"]:
            raise ValueError(f"Unknown x_scale: {x_scale}")

        width = options.get("width", width)
        height = options.get("height", height)
        equal = options.get("equal", equal)
    lineplot_kwargs = options.get("lineplot_kwargs", {}) if options is not None else {}
    style = options.get("style", hue) if options is not None else hue
    style_order = (
        options.get("style_order") if options is not None else None
    )
    if options is not None:
        label_fontsize = options.get("label_fontsize", label_fontsize)
        legend_fontsize_override = options.get("legend_fontsize")
        legend_title = options.get("legend_title")
        legend_title_fontsize = options.get("legend_title_fontsize")

    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    if colors is None:
        # get a list of distinct colors
        colors = distinctipy.get_colors(len(order), pastel_factor=0.5)
        if add_emphasis:
            # mute all colors except the first one
            colors = [
                colors[0],
            ] + [
                [
                    max(255, int(c[0] + 0.1 * c[0])),
                    max(255, int(c[1] + 0.1 * c[1])),
                    max(255, int(c[2] + 0.1 * c[2])),
                ]
                for c in colors[1:]
            ]
        colors = [tuple(c) for c in colors]

    if annotations is not None:
        for annotation in annotations:
            if annotation["type"] == "vline":
                ax.axvline(
                    x=annotation["x"],
                    color=annotation["color"],
                    linestyle=annotation["linestyle"],
                    linewidth=annotation["linewidth"],
                )
            elif annotation["type"] == "hline":
                ax.axhline(
                    y=annotation["y"],
                    color=annotation["color"],
                    linestyle=annotation["linestyle"],
                    linewidth=annotation["linewidth"],
                )
            elif annotation["type"] == "slope":
                ax.axline(
                    xy1=(annotation["x"], annotation["y"]),
                    slope=annotation["slope"],
                    color=annotation["color"],
                    linestyle=annotation["linestyle"],
                    linewidth=annotation["linewidth"],
                )
            elif annotation["type"] == "text":
                ax.text(
                    annotation["x"],
                    annotation["y"],
                    annotation["label"],
                    color=annotation["color"],
                    fontsize=annotation["fontsize"],
                    ha="left",
                    va="bottom",
                )
            elif annotation["type"] == "vspan":
                ax.axvspan(
                    annotation["xmin"],
                    annotation["xmax"],
                    color=annotation.get("color", "grey"),
                    alpha=annotation.get("alpha", 0.1),
                    linewidth=0,
                )

    if plot_type == "line":
        lineplot_args = dict(
            x=x,
            y=y,
            hue=hue,
            data=data,
            # order=order,
            hue_order=hue_order,
            palette=colors,
            linewidth=2.5,
        )
        if style is not None:
            lineplot_args["style"] = style
            if style_order is not None:
                lineplot_args["style_order"] = style_order
        sb.lineplot(**lineplot_args, **lineplot_kwargs)
    elif plot_type == "violin":
        sb.violinplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            width=0.8,
            order=order,
            hue_order=hue_order,
            palette=colors,
            density_norm="area",
            cut=0,
            inner=None,
        )

    elif plot_type == "swarm":
        sb.swarmplot(
            x=x,
            y=y,
            data=data,
            hue=hue,
            hue_order=hue_order,
            palette=colors,
            dodge=True,
            # style=hue,
            # style_order=order,
        )
    elif plot_type == "box":
        sb.boxplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            order=order,
            hue_order=hue_order,
            palette=colors,
            showfliers=False,
            whis=[5, 75],
            showmeans=True,
            flierprops=flierprops,
        )
    elif plot_type == "scatter":
        sb.scatterplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            hue_order=hue_order,
            palette=colors,
            style=hue,
            style_order=order,
            ax=ax,
        )
    elif plot_type == "strip":
        sb.stripplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            hue_order=hue_order,
            palette=colors,
            jitter=0.2,
            dodge=True,
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    if overlays:
        for overlay in overlays:
            if overlay is None:
                continue
            overlay_type = overlay.get("type", "line")
            if overlay_type == "line":
                ax.plot(
                    overlay["x"],
                    overlay["y"],
                    label=overlay.get("label"),
                    color=overlay.get("color", "black"),
                    linestyle=overlay.get("linestyle", "-"),
                    linewidth=overlay.get("linewidth", 2.5),
                    alpha=overlay.get("alpha", 1.0),
                    zorder=overlay.get("zorder"),
                )

    ax.tick_params(axis="both", which="major", labelsize=16)
    handles, labels = ax.get_legend_handles_labels()

    if x_scale == "log":
        ax.set_xscale("log")
    if y_scale == "log":
        ax.set_yscale("log")

    if order is None or len(order) < 3:
        ncol = 1
        base_legend_size = 14
    elif len(order) < 5:
        ncol = 2
        base_legend_size = 14
    else:
        ncol = 3
        base_legend_size = 12

    fontsize = (
        legend_fontsize_override
        if legend_fontsize_override is not None
        else base_legend_size
    )
    if legend_title_fontsize is None:
        legend_title_fontsize = fontsize

    ax.legend(
        handles=handles,
        labels=labels,
        loc=legend_location,
        title=legend_title,
        ncol=ncol,
        title_fontsize=legend_title_fontsize,
        fontsize=fontsize,
    )
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)

    if options is not None:
        x_tick_rotation = options.get("x_tick_rotation")
        x_tick_labelsize = options.get("x_tick_labelsize")
        x_tick_ha = options.get("x_tick_ha")
        x_tick_va = options.get("x_tick_va")
        if x_tick_rotation is not None:
            ax.tick_params(axis="x", labelrotation=x_tick_rotation)
        if x_tick_labelsize is not None:
            ax.tick_params(axis="x", labelsize=x_tick_labelsize)
        if x_tick_ha or x_tick_va:
            for label in ax.get_xticklabels():
                if x_tick_ha:
                    label.set_ha(x_tick_ha)
                if x_tick_va:
                    label.set_va(x_tick_va)

    # make axes equal
    if equal:
        ax.set_aspect("equal", adjustable="box")

    # # make legend transparent
    # legend = ax.get_legend()
    # legend.get_frame().set_alpha(0.0)
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_facecolor("none")
    # legend.get_frame().set_edgecolor("none")

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    fig.savefig(plot_name)
    plt.close(fig)


def plot_environment(ax, grid_data, limits):
    # get the locations of the grid data that are non-zero
    obstacles = np.argwhere(grid_data)
    # plot the grid data
    for obstacle in obstacles:
        # plot the occlusion with a rectangle
        ax.add_patch(
            plt.Rectangle(
                (obstacle[1], obstacle[0]),
                1,
                1,
                color="black",
                alpha=0.5,
            )
        )


def plot_segmented_lines(
    ax,
    df,
    columns,
):
    """
    Plots line segments with color changes based on a visibility column.
    """

    # Ensure the DataFrame is sorted properly (if it's not in order)
    df = df.sort_index()

    # Iterate through the change points to plot line segments
    for x_col, y_col, visible_col in columns:
        # Find change points in the 'visible' column
        change_points = df[df[visible_col] != df[visible_col].shift()].index.tolist()
        # Add the index of end of the dataframe as a change point
        change_points.append(df.index[-1])

        for start, end in zip(change_points[:-1], change_points[1:]):
            segment = df.loc[start:end]

            # use the visibility value (0-10) to select a hue from
            # green (0) to red (10) in a smooth curve
            if segment[visible_col].iloc[0] == 0:
                color = "green"
            elif segment[visible_col].iloc[0] < 3:
                color = "yellow"
            elif segment[visible_col].iloc[0] < 6:
                color = "orange"
            else:
                color = "red"

            if start == change_points[0]:
                # draw a circle patch at the start of the trajectory
                ax.add_patch(
                    plt.Circle(
                        (segment[x_col].iloc[0], segment[y_col].iloc[0]),
                        0.2,
                        color=color,
                        alpha=0.5,
                    )
                )
            ax.plot(
                segment[x_col],
                segment[y_col],
                color=color,
                linewidth=2,
                linestyle="-",
            )

    # draw a block at the end of the trajectory
    ax.add_patch(
        plt.Rectangle(
            (segment[x_col].iloc[-1] - 0.25, segment[y_col].iloc[-1] - 0.25),
            0.5,
            0.5,
            color=color,
            alpha=0.5,
        )
    )


def plot_results(results, limits, grid_data, path):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    # plot the environment
    plot_environment(ax[0], grid_data, limits)
    plot_environment(ax[1], grid_data, limits)

    # results is a dataframe with columns:
    # trial, depth, node_x, node_y, target_x, target_y, safe, visible, visibility_ratio, done
    # plot the agent positions and whether the target is visible for each segment of the trial
    for trial in results["trial"].unique():
        trial_results = results[results["trial"] == trial]

        # plot the agent positions
        plot_segmented_lines(
            ax[0],
            trial_results,
            columns=[("node_x", "node_y", "visible")],
        )

        # find the number of target_xx_x columns
        num_targets = len(
            [
                col
                for col in trial_results.columns
                if col.startswith("target_") and col.endswith("_x")
            ]
        )

        target_columns = []
        for target_idx in range(num_targets):
            target_col_x = f"target_{target_idx}_x"
            target_col_y = f"target_{target_idx}_y"
            vis_col = f"target_{target_idx}_vis"

            target_columns.append((target_col_x, target_col_y, vis_col))

        # plot the target positions
        plot_segmented_lines(
            ax[1],
            trial_results,
            columns=target_columns,
        )

    ax[0].set_xlabel("X position")
    ax[0].set_ylabel("Y position")
    if limits is not None:
        ax[0].set_xlim(limits[0])
        ax[0].set_ylim(limits[1])
    ax[1].set_xlabel("X position")
    ax[1].set_ylabel("Y position")
    if limits is not None:
        ax[1].set_xlim(limits[0])
        ax[1].set_ylim(limits[1])

    ax[0].set_title("Agent Trajectory")
    ax[0].grid(True)
    ax[0].set_aspect("equal")
    ax[1].set_title("Target Trajectory")
    ax[1].grid(True)
    ax[1].set_aspect("equal")

    # save the figure
    fig.savefig(path + "trajectories.pdf")
    plt.close(fig)


def plot_scenarios(scenarios, results, dir, prefix):

    for scenario in scenarios:
        # get the grid data for the scenario
        grid_data = scenario["grid_data"]

        # get the limits for the scenario
        scenario_limits = [(0, scenario["Mx"]), (0, scenario["My"])]

        # get the results for the scenario
        scenario_results = results[results["scenario"] == scenario["scenario"]]

        # plot the results for the scenario
        path = f"{dir}/{scenario['scenario']}/"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        plot_results(scenario_results, scenario_limits, grid_data, path)
