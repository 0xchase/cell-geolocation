#!/usr/bin/env python3
"""Render detailed replay cases and long-range endpoint diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from spoofing_category_overview import GRID, INK, MUTED, load_world, setup_map


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"
WORLD = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
FIGURES = ROOT / "paper" / "figs"

REPLAY = "#2f6f9f"
MIXED = "#9a762d"
RECIPROCAL_A = "#6a51a3"
RECIPROCAL_B = "#d95f8d"

REPLAY_ORDER = ["Changsha", "Beijing", "Kermanshah", "Wuhan"]
DIAGNOSTIC_ORDER = ["Lijiang", "Crimea", "Harbin"]


def bounds(group: pd.DataFrame) -> tuple[float, float, float, float]:
    xs = np.r_[group["home_lon"].to_numpy(), group["destination_lon"].iloc[0]]
    ys = np.r_[group["home_lat"].to_numpy(), group["destination_lat"].iloc[0]]
    xspan = max(float(xs.max() - xs.min()), 1.0)
    yspan = max(float(ys.max() - ys.min()), 1.0)
    xmargin = max(0.8, 0.10 * xspan)
    ymargin = max(0.55, 0.14 * yspan)
    return (
        max(-180.0, float(xs.min() - xmargin)),
        min(180.0, float(xs.max() + xmargin)),
        max(-80.0, float(ys.min() - ymargin)),
        min(84.0, float(ys.max() + ymargin)),
    )


def map_panel(
    ax: plt.Axes,
    group: pd.DataFrame,
    summary: pd.Series,
    rings,
    *,
    color: str,
    title_prefix: str,
) -> None:
    bbox = bounds(group)
    setup_map(ax, rings, bbox)
    destination = (
        float(summary["destination_lon"]), float(summary["destination_lat"])
    )
    segments = [
        [(float(row.home_lon), float(row.home_lat)), destination]
        for row in group.itertuples(index=False)
    ]
    ax.add_collection(LineCollection(
        segments, colors=color, linewidths=0.30,
        alpha=max(0.035, min(0.22, 16 / max(len(group), 1))),
        zorder=1, rasterized=True,
    ))
    sizes = 4 + 2.0 * np.sqrt(group["observations"].clip(upper=200))
    ax.scatter(
        group["home_lon"], group["home_lat"], s=sizes, color=color,
        alpha=0.55, edgecolor="white", linewidth=0.18, zorder=2, rasterized=True,
    )
    ax.scatter(
        [destination[0]], [destination[1]], s=34, marker="*", color=color,
        edgecolor="white", linewidth=0.5, zorder=4,
    )
    source_lon = float(summary["source_lon"])
    source_lat = float(summary["source_lat"])
    ax.scatter(
        [source_lon], [source_lat], s=18, facecolor="white", edgecolor=color,
        linewidth=0.8, zorder=4,
    )
    source_left = source_lon < (bbox[0] + bbox[1]) / 2
    source_offset = (3, 3) if source_left else (-3, 3)
    ax.annotate(
        str(summary["source_region"]), (source_lon, source_lat),
        xytext=source_offset, textcoords="offset points", fontsize=4.6,
        ha="left" if source_left else "right", color=INK, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.7},
        zorder=5,
    )
    destination_left = destination[0] < (bbox[0] + bbox[1]) / 2
    destination_offset = (3, -8) if destination_left else (-3, -8)
    ax.annotate(
        str(summary["destination_name"]), destination,
        xytext=destination_offset, textcoords="offset points",
        ha="left" if destination_left else "right", fontsize=4.6,
        color=color, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.7},
        zorder=5,
    )
    ax.set_xticks(np.linspace(bbox[0], bbox[1], 4))
    ax.set_yticks(np.linspace(bbox[2], bbox[3], 3))
    ax.tick_params(labelsize=4.2, length=1.5, pad=0.8)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.set_title(
        f"{title_prefix}  {summary['source_region']} → {summary['destination_name']}\n"
        f"{int(summary['identities']):,} identities · "
        f"{int(summary['observations']):,} observations · "
        f"{summary['median_displacement_km']:,.0f} km",
        loc="left", fontsize=6.15, fontweight="bold", color=color, pad=2.5,
    )


def onset_panel(
    ax: plt.Axes,
    group: pd.DataFrame,
    summary: pd.Series,
    *,
    color: str,
) -> None:
    first = pd.to_datetime(group["first_seen"])
    months = first.dt.to_period("M").dt.to_timestamp()
    counts = months.value_counts().sort_index()
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="MS")
    counts = counts.reindex(full, fill_value=0)
    ax.bar(counts.index, counts.values, width=24, color=color, alpha=0.82)
    ax.set_ylabel("First-seen identities")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(2, math.ceil(len(full) / 6))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelsize=4.4)
    ax.set_title(
        "First destination appearances",
        loc="left", fontsize=6.15, fontweight="bold", pad=2.5,
    )
    peak_month = counts.idxmax()
    peak = int(counts.max())
    ax.annotate(
        f"{peak:,}",
        (peak_month, peak), xytext=(0, -2), textcoords="offset points",
        ha="center", va="top", fontsize=4.5, color="white", fontweight="bold",
    )
    ax.text(
        0.99, 0.96,
        f"source p90 radius: {summary['source_p90_radius_km']:,.0f} km\n"
        f"span: {pd.to_datetime(summary['first_seen']):%b %Y}–"
        f"{pd.to_datetime(summary['last_seen']):%b %Y}",
        transform=ax.transAxes, ha="right", va="top", fontsize=4.8, color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "#c8c4bd", "linewidth": 0.35,
              "alpha": 0.80, "pad": 1.3},
    )


def reciprocal_row(
    ax_map: plt.Axes,
    ax_time: plt.Axes,
    members: pd.DataFrame,
    summary: pd.DataFrame,
    rings,
    number: int,
) -> None:
    pair = members[members["evidence_class"].eq("reciprocal swap")]
    bbox = bounds(pair)
    setup_map(ax_map, rings, bbox)
    colors = {"3111_11254": RECIPROCAL_A, "4367_11811": RECIPROCAL_B}
    labels = {"3111_11254": "Chifeng → Jingmen", "4367_11811": "Jingmen → Chifeng"}
    for site_id, group in pair.groupby("site_id"):
        row = summary[summary["site_id"].eq(site_id)].iloc[0]
        destination = (row.destination_lon, row.destination_lat)
        segments = [[(r.home_lon, r.home_lat), destination] for r in group.itertuples()]
        ax_map.add_collection(LineCollection(
            segments, colors=colors[site_id], linewidths=0.55, alpha=0.18,
            zorder=1, rasterized=True,
        ))
        ax_map.scatter(group["home_lon"], group["home_lat"], s=9,
                       facecolor="white", edgecolor=colors[site_id], linewidth=0.55,
                       zorder=3, rasterized=True)
        ax_map.scatter([destination[0]], [destination[1]], s=26, marker="D",
                       color=colors[site_id], edgecolor="white", linewidth=0.4,
                       label=f"{labels[site_id]} ({len(group)} IDs)", zorder=4)
    ax_map.annotate("Jingmen", (112.54, 31.11), xytext=(3, -9),
                    textcoords="offset points", fontsize=5, fontweight="bold")
    ax_map.annotate("Chifeng", (118.11, 43.67), xytext=(3, 3),
                    textcoords="offset points", fontsize=5, fontweight="bold")
    ax_map.legend(frameon=False, loc="upper left", fontsize=4.6)
    ax_map.set_xticks(np.linspace(bbox[0], bbox[1], 4))
    ax_map.set_yticks(np.linspace(bbox[2], bbox[3], 3))
    ax_map.tick_params(labelsize=4.2, length=1.5, pad=0.8)
    ax_map.set_title(
        f"{number}  Reciprocal coordinate swap — exact endpoint reversal",
        loc="left", fontsize=6.15, fontweight="bold", color=RECIPROCAL_A, pad=2.5,
    )

    for site_id, group in pair.groupby("site_id"):
        months = pd.to_datetime(group["first_seen"]).dt.to_period("M").dt.to_timestamp()
        counts = months.value_counts().sort_index()
        ax_time.plot(counts.index, counts.values, marker="o", markersize=2.7,
                     linewidth=1.05, color=colors[site_id], label=labels[site_id])
    ax_time.set_ylabel("First-seen identities")
    ax_time.grid(color=GRID, linewidth=0.4)
    ax_time.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_time.legend(frameon=False, fontsize=4.7, loc="upper right")
    ax_time.set_title(
        "Both directions persist; geometry matches to <1 km",
        loc="left", fontsize=6.15, fontweight="bold", pad=2.5,
    )


def render_atlas(
    members: pd.DataFrame,
    summary: pd.DataFrame,
    destinations: list[str],
    output: Path,
    preview: Path,
    *,
    diagnostic: bool,
) -> None:
    rings = load_world(WORLD)
    rows = len(destinations) + (1 if diagnostic else 0)
    fig = plt.figure(figsize=(7.15, 2.17 * rows + 0.45))
    grid = fig.add_gridspec(
        rows, 2, width_ratios=[1.16, 0.84], left=0.075, right=0.985,
        bottom=0.055, top=0.975, hspace=0.63, wspace=0.27,
    )
    for index, destination_name in enumerate(destinations):
        ax_map = fig.add_subplot(grid[index, 0])
        ax_time = fig.add_subplot(grid[index, 1])
        group = members[members["destination_name"].eq(destination_name)].copy()
        site_id = group["site_id"].iloc[0]
        row = summary[summary["site_id"].eq(site_id)].iloc[0]
        color = MIXED if diagnostic else REPLAY
        map_panel(
            ax_map, group, row, rings, color=color,
            title_prefix=str(index + 1),
        )
        onset_panel(ax_time, group, row, color=color)

    if diagnostic:
        reciprocal_row(
            fig.add_subplot(grid[-1, 0]), fig.add_subplot(grid[-1, 1]),
            members, summary, rings, rows,
        )

    for index in range(1, rows):
        y = grid[index, 0].get_position(fig).y1 + 0.024
        fig.add_artist(plt.Line2D(
            [0.06, 0.99], [y, y], transform=fig.transFigure,
            color="#c8c4bd", linewidth=0.45,
        ))
    fig.text(
        0.5, 0.012,
        "○ source centroid   ★/◆ destination   ·   administrative boundaries: Natural Earth",
        ha="center", fontsize=4.3, color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(preview, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-output", type=Path,
        default=FIGURES / "identity_replay_case_studies.pdf",
    )
    parser.add_argument(
        "--diagnostic-output", type=Path,
        default=FIGURES / "destination_cluster_diagnostics.pdf",
    )
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 5.5,
        "axes.labelsize": 5.1, "xtick.labelsize": 4.7, "ytick.labelsize": 4.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    members = pd.read_csv(
        DATA / "destination_cluster_members.csv",
        parse_dates=["first_seen", "last_seen"],
    )
    summary = pd.read_csv(
        DATA / "destination_cluster_case_summary.csv",
        parse_dates=["first_seen", "last_seen"],
    )
    render_atlas(
        members, summary, REPLAY_ORDER,
        args.replay_output, args.replay_output.with_suffix(".png"), diagnostic=False,
    )
    render_atlas(
        members, summary, DIAGNOSTIC_ORDER,
        args.diagnostic_output, args.diagnostic_output.with_suffix(".png"), diagnostic=True,
    )
    print(args.replay_output)
    print(args.diagnostic_output)


if __name__ == "__main__":
    main()
