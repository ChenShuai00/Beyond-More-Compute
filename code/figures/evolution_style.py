"""Drawing style for annual hardware and external-access distributions."""
from __future__ import annotations
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch,Patch,Rectangle
from matplotlib.ticker import PercentFormatter
FIGURE_WIDTH_MM=183
FIGURE_HEIGHT_MM=155
RASTER_DPI=600

def configure_matplotlib(analysis_id: str, text_color: str) -> None:
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans","Liberation Sans"],"font.size":7,"axes.labelsize":7.5,"axes.titlesize":8,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"axes.linewidth":.7,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"text.color":text_color,"axes.labelcolor":text_color,"axes.titlecolor":text_color,"axes.edgecolor":text_color,"xtick.color":text_color,"ytick.color":text_color,"pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":analysis_id})

def draw_evolution(
    access_annual: pd.DataFrame,
    gpu_annual: pd.DataFrame,
    config: dict[str, Any],
    component_configs: dict[str, Any],
) -> tuple[Any, list[Any], Any]:
    plot = config["plot"]
    access_config = component_configs["hardware_api"]
    gpu_config = component_configs["gpu_hardware"]
    configure_matplotlib(config["analysis_id"], plot["axis_text_color"])
    if int(plot["width_mm"]) != FIGURE_WIDTH_MM or int(plot["height_mm"]) != FIGURE_HEIGHT_MM:
        raise ValueError("Configured figure dimensions do not match the locked dimensions")
    if int(plot["raster_dpi"]) != RASTER_DPI:
        raise ValueError("Configured raster DPI does not match the locked resolution")

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        3,
        left=0.065,
        right=0.95,
        bottom=0.095,
        top=0.86,
        hspace=0.70,
        wspace=0.31,
        height_ratios=[1.0, 0.78],
    )
    gpu_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    inset_config = plot["access_inset"]
    access_grid = grid[1, :].subgridspec(
        1,
        2,
        width_ratios=[float(value) for value in inset_config["width_ratios"]],
        wspace=float(inset_config["wspace"]),
    )
    access_axis = fig.add_subplot(access_grid[0, 0])
    inset_axis = fig.add_subplot(access_grid[0, 1])
    axes = [*gpu_axes, access_axis]
    fig.add_artist(
        Rectangle(
            (0.002, 0.002),
            0.996,
            0.996,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=plot["frame_color"],
            linewidth=float(plot["frame_linewidth"]),
            zorder=20,
        )
    )

    years = list(range(int(access_config["year_min"]), int(access_config["year_max"]) + 1))
    for state in access_config["configurations"]:
        rows = access_annual.loc[
            access_annual["configuration"].eq(state["id"])
        ].sort_values("year")
        access_axis.plot(
            rows["year"],
            rows["paper_share"],
            color=state["color"],
            linestyle=state["line_style"],
            marker=state["marker"],
            linewidth=float(access_config["plot"]["line_width"]),
            markersize=float(access_config["plot"]["marker_size"]),
            markerfacecolor="white",
            markeredgecolor=state["color"],
            markeredgewidth=0.9,
            label=state["display_name"],
            zorder=3,
        )
    access_axis.set(
        title="Hardware–API resource configurations",
        xlabel="Publication year",
        ylabel="Share of papers",
        xlim=(years[0] - 0.18, years[-1] + 0.18),
        ylim=(0, float(access_config["plot"]["y_max"])),
        xticks=years,
        yticks=[value / 10 for value in range(8)],
    )
    access_axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    access_axis.grid(axis="y", color=plot["grid_color"], linewidth=0.45, alpha=0.75, zorder=0)
    access_axis.text(
        -0.08,
        1.10,
        "d",
        transform=access_axis.transAxes,
        fontweight="bold",
        fontsize=8,
        ha="left",
        va="top",
    )
    access_handles, access_labels = access_axis.get_legend_handles_labels()
    fig.legend(
        handles=access_handles,
        labels=access_labels,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.445),
        ncol=4,
        fontsize=6.7,
        handlelength=2.3,
        columnspacing=1.55,
        borderaxespad=0,
    )

    inset_axis.set_facecolor("white")
    inset_ids = set(inset_config["configurations"])
    for state in access_config["configurations"]:
        if state["id"] not in inset_ids:
            continue
        rows = access_annual.loc[
            access_annual["configuration"].eq(state["id"])
        ].sort_values("year")
        inset_axis.plot(
            rows["year"],
            rows["paper_share"],
            color=state["color"],
            linestyle=state["line_style"],
            marker=state["marker"],
            linewidth=float(access_config["plot"]["line_width"]),
            markersize=float(access_config["plot"]["marker_size"]),
            markerfacecolor="white",
            markeredgecolor=state["color"],
            markeredgewidth=0.9,
            zorder=3,
        )
    inset_axis.set(
        title=inset_config["title"],
        xlim=(years[0] - 0.18, years[-1] + 0.18),
        ylim=(-0.02, float(inset_config["ylim"][1])),
        xticks=[int(value) for value in inset_config["xticks"]],
        yticks=[float(value) for value in inset_config["yticks"]],
    )
    inset_axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    inset_axis.yaxis.tick_right()
    inset_axis.tick_params(axis="both", labelsize=5, length=2.2, width=0.55, pad=1.2)
    inset_axis.title.set_fontsize(5.5)
    inset_axis.grid(axis="y", color=plot["grid_color"], linewidth=0.35, alpha=0.7, zorder=0)
    for spine in inset_axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#8A8A8A")
        spine.set_linewidth(0.55)

    source_ymin, source_ymax = [float(value) for value in inset_config["source_ylim"]]
    access_xmin, access_xmax = access_axis.get_xlim()
    inset_xmin, _ = inset_axis.get_xlim()
    source_box = Rectangle(
        (access_xmin, source_ymin),
        access_xmax - access_xmin,
        source_ymax - source_ymin,
        transform=access_axis.transData,
        facecolor="none",
        edgecolor=inset_config["connector_color"],
        linewidth=float(inset_config["connector_linewidth"]),
        linestyle=inset_config["connector_linestyle"],
        zorder=4,
    )
    access_axis.add_patch(source_box)
    for source_y, inset_y in ((source_ymin, 0.0), (source_ymax, float(inset_config["ylim"][1]))):
        connector = ConnectionPatch(
            xyA=(access_xmax, source_y),
            coordsA=access_axis.transData,
            xyB=(inset_xmin, inset_y),
            coordsB=inset_axis.transData,
            color=inset_config["connector_color"],
            linewidth=float(inset_config["connector_linewidth"]),
            linestyle=inset_config["connector_linestyle"],
            arrowstyle="-",
            clip_on=False,
            zorder=4,
        )
        fig.add_artist(connector)

    for index, (axis, variable) in enumerate(zip(gpu_axes, gpu_config["variables"], strict=True)):
        rows = gpu_annual.loc[gpu_annual["variable"].eq(variable["id"])].sort_values("year")
        colors = variable["colors"]
        x = rows["year"].to_numpy(dtype=float)
        axis.fill_between(
            x,
            rows["q25"].to_numpy(dtype=float),
            rows["q75"].to_numpy(dtype=float),
            color=colors["iqr"],
            alpha=float(gpu_config["plot"]["iqr_alpha"]),
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            rows["median"].to_numpy(dtype=float),
            color=colors["median"],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=3.8,
            markeredgewidth=0.5,
            zorder=3,
        )
        axis.plot(
            x,
            rows["p90"].to_numpy(dtype=float),
            color=colors["p90"],
            marker="^",
            linestyle="--",
            linewidth=1.2,
            markersize=3.2,
            markeredgewidth=0.5,
            zorder=2,
        )
        axis.set(
            title=variable["display_name"],
            ylabel=variable["unit"],
            xlim=(years[0] - 0.15, years[-1] + 0.15),
            ylim=tuple(float(value) for value in variable["ylim"]),
            xticks=years,
            yticks=[float(value) for value in variable["yticks"]],
        )
        axis.grid(axis="y", color=plot["grid_color"], linewidth=0.45, alpha=0.75, zorder=0)
        axis.text(
            -0.20,
            1.10,
            chr(97 + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
            ha="left",
            va="top",
        )

    statistic_handles = [
        Line2D(
            [],
            [],
            color=plot["legend_median_color"],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=3.8,
            label="Median",
        ),
        Patch(
            facecolor=plot["legend_iqr_color"],
            edgecolor="none",
            label="Interquartile range",
        ),
        Line2D(
            [],
            [],
            color=plot["legend_p90_color"],
            marker="^",
            linestyle="--",
            linewidth=1.2,
            markersize=3.2,
            label="90th percentile",
        ),
    ]
    fig.legend(
        handles=statistic_handles,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.985),
        ncol=3,
        fontsize=6.7,
        handlelength=2.3,
        columnspacing=1.7,
        borderaxespad=0,
    )
    fig.text(0.525, 0.485, "Publication year", ha="center", va="center", fontsize=7.5)
    return fig, axes, inset_axis
