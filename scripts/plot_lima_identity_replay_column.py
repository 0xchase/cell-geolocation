#!/usr/bin/env python3
"""Render the Lima identity-replay evidence at publication column width."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_helpers import TILE_ATTRIBUTION, add_osm_basemap, draw_geojson_layer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"
FIGS = ROOT / "paper" / "figs"
OUTPUT = FIGS / "lima_identity_replay_column.pdf"
COUNTRIES = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
UA_BBOX = (22.0, 41.0, 42.0, 55.0)
LIMA_BBOX = (-77.059, -77.041, -12.049, -12.031)

PALETTE = {
    "Vodafone UA": "#b23a48",
    "lifecell": "#2f6f9f",
    "3Mob/other": "#c9743a",
    "Kyivstar/other": "#4f7f52",
    "MNC 702": "#8e6aa7",
    "MNC 707": "#7d7d7d",
}
LOCATION_COLORS = {
    "Ukraine home": "#2f6f9f",
    "Lima replay": "#b23a48",
    "Elsewhere": "#8b8b8b",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 5.8,
        "axes.titlesize": 6.7,
        "axes.titleweight": "bold",
        "axes.labelsize": 5.5,
        "xtick.labelsize": 5.0,
        "ytick.labelsize": 5.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "lima": pd.read_csv(DATA / "lima_replay_observations.csv", parse_dates=["timestamp"]),
        "counts": pd.read_csv(DATA / "lima_replay_location_counts.csv"),
        "timeline": pd.read_csv(
            DATA / "lima_replay_monthly_locations.csv", parse_dates=["month"]
        ),
        "homes": pd.read_csv(DATA / "lima_replay_home_identities.csv"),
        "ukraine": pd.read_csv(DATA / "lima_ukraine_reference_density.csv"),
    }


def setup_country_map(ax: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(
        ax, COUNTRIES, bbox, facecolor="#f5f1e8", edgecolor="#8a8176",
        linewidth=0.35, zorder=0,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    center_lat = (bbox[2] + bbox[3]) / 2
    ax.set_aspect(1 / math.cos(math.radians(center_lat)), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#8a8176")
        spine.set_linewidth(0.4)


def draw_lima(ax: plt.Axes, lima: pd.DataFrame) -> None:
    used_tiles = add_osm_basemap(
        ax, LIMA_BBOX, zoom=15, alpha=1.0, grayscale=False,
        source="carto_voyager", zorder=0,
    )
    if not used_tiles:
        setup_country_map(ax, LIMA_BBOX)
    operators = [name for name in PALETTE if name in set(lima.operator)]
    for operator in operators:
        group = lima[lima.operator.eq(operator)]
        ax.scatter(
            group.lon, group.lat, s=7, color=PALETTE[operator], alpha=0.72,
            edgecolor="white", linewidth=0.18, rasterized=True, zorder=3,
        )
    center_lon, center_lat = lima.lon.mean(), lima.lat.mean()
    ax.scatter(center_lon, center_lat, marker="x", s=20, color="#191919", linewidth=0.8, zorder=4)
    ax.annotate(
        "Jirón Máncora cluster", (center_lon, center_lat), xytext=(0, 7),
        textcoords="offset points", ha="center", fontsize=4.9, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.7},
        zorder=5,
    )
    ax.set_xlim(LIMA_BBOX[0], LIMA_BBOX[1])
    ax.set_ylim(LIMA_BBOX[2], LIMA_BBOX[3])
    center_lat = (LIMA_BBOX[2] + LIMA_BBOX[3]) / 2
    ax.set_aspect(1 / math.cos(math.radians(center_lat)), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#8a8176")
        spine.set_linewidth(0.4)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=2.8,
               markerfacecolor=PALETTE[name], markeredgecolor="white",
               markeredgewidth=0.2, label=name)
        for name in operators
    ]
    ax.legend(
        handles=handles, loc="lower left", ncol=2, frameon=True,
        facecolor="white", edgecolor="none", framealpha=0.78,
        fontsize=4.1, handletextpad=0.2, columnspacing=0.5, borderpad=0.25,
    )
    ax.set_title("A. Lima neighborhood cluster", loc="left", pad=2)


def draw_ukraine(ax: plt.Axes, homes: pd.DataFrame, ukraine: pd.DataFrame) -> None:
    setup_country_map(ax, UA_BBOX)
    weights = np.log10(ukraine.obs.clip(lower=1) + 1)
    ax.scatter(
        ukraine.lon_bin, ukraine.lat_bin, s=0.4 + weights * 0.65,
        color="#686868", alpha=0.22, edgecolor="none", rasterized=True, zorder=2,
    )
    for operator, group in homes.groupby("operator"):
        ax.scatter(
            group.lon, group.lat, s=5.5, color=PALETTE.get(operator, "#777777"),
            alpha=0.82, edgecolor="white", linewidth=0.18, rasterized=True, zorder=3,
        )
    ax.text(
        0.985, 0.025, "grey: all MCC 255 reports\ncolored: Lima identities at home",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.2,
        bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.3,
              "alpha": 0.80, "pad": 1.2},
        zorder=4,
    )
    ax.set_title("B. Ukrainian home footprint", loc="left", pad=2)


def draw_counts(ax: plt.Axes, counts: pd.DataFrame) -> None:
    order = [name for name in ["Lima replay", "Ukraine home"] if name in set(counts.location)]
    counts = counts.set_index("location").loc[order].reset_index()
    labels = ["Lima", "Ukraine\nhome"]
    bars = ax.bar(
        labels, counts.obs,
        color=[LOCATION_COLORS[name] for name in counts.location], width=0.62,
    )
    ax.set_yscale("log")
    ax.set_ylim(1_000, counts.obs.max() * 3.2)
    for bar, row in zip(bars, counts.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, row.obs * 1.13,
            f"{row.obs:,}\n{row.identities:,} IDs",
            ha="center", va="bottom", fontsize=4.5,
        )
    ax.grid(axis="y", color="#d7d9dc", linewidth=0.4, zorder=-1)
    ax.set_ylabel("Reports (log scale)")
    ax.set_title("C. Home/Lima report ratio", loc="left", pad=2)
    ax.set_box_aspect(1)


def draw_timeline(ax: plt.Axes, timeline: pd.DataFrame) -> None:
    for location in ["Ukraine home", "Lima replay"]:
        group = timeline[timeline.location.eq(location)].sort_values("month")
        if group.empty:
            continue
        ax.plot(
            group.month, group.identities, marker="o", markersize=2.2,
            linewidth=1.0, color=LOCATION_COLORS[location], label=location,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Distinct identities (log)")
    ax.grid(color="#d7d9dc", linewidth=0.4, zorder=-1)
    locator = mdates.MonthLocator(interval=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.legend(
        loc="lower left", ncol=2, frameon=True, facecolor="white",
        edgecolor="none", framealpha=0.78, fontsize=4.4,
        handlelength=1.3, handletextpad=0.3, columnspacing=0.7,
    )
    onset = timeline[
        timeline.location.eq("Lima replay") & timeline.identities.ge(50)
    ]
    if not onset.empty:
        first = onset.month.min()
        ax.axvline(first, color=LOCATION_COLORS["Lima replay"], linestyle="--",
                   linewidth=0.7, alpha=0.75)
        ax.annotate(
            f"ramp begins\n{first:%b %Y}", xy=(first, 55), xytext=(-4, 5),
            textcoords="offset points", ha="right", va="bottom",
            fontsize=4.4, color=LOCATION_COLORS["Lima replay"],
        )
    ax.set_title("D. Monthly onset", loc="left", pad=2)
    ax.set_box_aspect(1)


def render() -> None:
    configure_style()
    data = load_data()
    fig = plt.figure(figsize=(3.35, 3.75))
    grid = fig.add_gridspec(
        2, 2, left=0.10, right=0.99, bottom=0.085, top=0.975,
        hspace=0.34, wspace=0.28,
    )
    draw_lima(fig.add_subplot(grid[0, 0]), data["lima"])
    draw_ukraine(fig.add_subplot(grid[0, 1]), data["homes"], data["ukraine"])
    draw_counts(fig.add_subplot(grid[1, 0]), data["counts"])
    draw_timeline(fig.add_subplot(grid[1, 1]), data["timeline"])
    attribution = TILE_ATTRIBUTION["carto_voyager"].replace(r"\copyright{}", "©")
    fig.text(0.985, 0.012, attribution, ha="right", fontsize=3.8, color="#666666")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    render()
