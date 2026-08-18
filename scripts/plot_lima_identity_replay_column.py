#!/usr/bin/env python3
"""Render the Lima identity-replay evidence at publication column width."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
        "timeline": pd.read_csv(
            DATA / "lima_replay_monthly_locations.csv", parse_dates=["month"]
        ),
        "homes": pd.read_csv(DATA / "lima_replay_home_identities.csv"),
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
    attribution = TILE_ATTRIBUTION["carto_voyager"].replace(r"\copyright{}", "©")
    ax.text(
        0.99, 0.01, attribution, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=3.2, color="#666666",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.3},
        zorder=6,
    )
    ax.set_box_aspect(1)
    ax.set_title("Lima", loc="left", pad=2)


def draw_ukraine(ax: plt.Axes, homes: pd.DataFrame) -> None:
    setup_country_map(ax, UA_BBOX)
    draw_geojson_layer(
        ax, COUNTRIES, UA_BBOX, countries={"Ukraine", "UKR"},
        facecolor="#f3e4c4", edgecolor="#8f6a35", linewidth=0.65, zorder=1,
    )
    for operator, group in homes.groupby("operator"):
        ax.scatter(
            group.lon, group.lat, s=5.5, color=PALETTE.get(operator, "#777777"),
            alpha=0.82, edgecolor="white", linewidth=0.18, rasterized=True, zorder=3,
        )
    ax.set_box_aspect(1)
    ax.set_title("Ukraine", loc="left", pad=2)


def draw_timeline(ax: plt.Axes, timeline: pd.DataFrame) -> None:
    for location in ["Ukraine home", "Lima replay"]:
        group = timeline[timeline.location.eq(location)].sort_values("month")
        if group.empty:
            continue
        ax.plot(
            group.month, group.identities, marker="o", markersize=2.2,
            linewidth=1.0, color=LOCATION_COLORS[location],
            label={"Ukraine home": "Ukraine", "Lima replay": "Lima"}[location],
        )
    ax.set_yscale("log")
    ax.set_ylabel("Cohort identities (log)")
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
    ax.set_title("Same-identity observations", loc="left", pad=2)


def render() -> None:
    configure_style()
    data = load_data()
    fig = plt.figure(figsize=(3.35, 3.05))
    grid = fig.add_gridspec(
        2, 2, left=0.025, right=0.998, bottom=0.075, top=0.985,
        height_ratios=[1.08, 0.55], hspace=0.10, wspace=0.018,
    )
    draw_lima(fig.add_subplot(grid[0, 0]), data["lima"])
    draw_ukraine(fig.add_subplot(grid[0, 1]), data["homes"])
    draw_timeline(fig.add_subplot(grid[1, :]), data["timeline"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    render()
