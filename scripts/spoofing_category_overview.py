#!/usr/bin/env python3
"""Render a full-page overview of strongly supported spoofing activity.

The figure reads only the auditable CSVs in ``data/spoofing``.  It does not
query ClickHouse, and each row pairs a population-level view with the feature
that distinguishes that category from the others.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrowPatch, Polygon

from extract_spoofing_categories import KEY, MULTI_WINDOW_DAYS
from plot_helpers import add_osm_basemap
from summarize_fixed_gnss_like_examples import corridor_points


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"
WORLD = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
DEFAULT_OUTPUT = ROOT / "paper" / "figs" / "spoofing_category_overview.pdf"

FIXED = "#b23a48"
FIXED_SUGGESTIVE = "#9a762d"
MULTIPLE_WEST = "#6a51a3"
MULTIPLE_EAST = "#d95f8d"
REPLAY = "#2f6f9f"
REBROADCAST = "#3f7d5a"
INK = "#292724"
MUTED = "#77736d"
GRID = "#d8d5cf"
LAND = "#f1eee7"
WATER = "#e8f0f4"


@dataclass(frozen=True)
class Ring:
    points: np.ndarray
    bounds: tuple[float, float, float, float]


def load_world(path: Path) -> list[Ring]:
    """Load lightly decimated exterior rings suitable for small paper panels."""
    obj = json.loads(path.read_text())
    rings: list[Ring] = []
    for feature in obj["features"]:
        geom = feature.get("geometry")
        if not geom:
            continue
        if geom["type"] == "Polygon":
            polygons = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polygons = geom["coordinates"]
        else:
            continue
        for polygon in polygons:
            if not polygon or len(polygon[0]) < 3:
                continue
            points = np.asarray(polygon[0], dtype=float)
            # Cap detail because each map is only a few inches wide.
            step = max(1, math.ceil(len(points) / 220))
            points = points[::step]
            if not np.array_equal(points[0], points[-1]):
                points = np.vstack([points, points[0]])
            bounds = (
                float(points[:, 0].min()), float(points[:, 0].max()),
                float(points[:, 1].min()), float(points[:, 1].max()),
            )
            rings.append(Ring(points, bounds))
    return rings


def setup_map(
    ax: plt.Axes,
    rings: list[Ring],
    bbox: tuple[float, float, float, float],
    *,
    equal: bool = False,
) -> None:
    west, east, south, north = bbox
    patches = []
    for ring in rings:
        xmin, xmax, ymin, ymax = ring.bounds
        if xmax < west or xmin > east or ymax < south or ymin > north:
            continue
        patches.append(Polygon(ring.points, closed=True))
    ax.add_collection(PatchCollection(
        patches, facecolor=LAND, edgecolor="#aaa49a", linewidth=0.16, zorder=0
    ))
    ax.set_facecolor(WATER)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", linewidth=0.35, alpha=0.85, zorder=-1)
    ax.tick_params(length=2, pad=1)


def draw_arc(
    ax: plt.Axes,
    source_lon: float,
    source_lat: float,
    destination_lon: float,
    destination_lat: float,
    *,
    color: str,
    linewidth: float,
    alpha: float = 0.65,
    curvature: float = 0.10,
    zorder: int = 2,
) -> None:
    patch = FancyArrowPatch(
        (source_lon, source_lat),
        (destination_lon, destination_lat),
        arrowstyle="-",
        connectionstyle=f"arc3,rad={curvature}",
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        transform=ax.transData,
        zorder=zorder,
    )
    ax.add_patch(patch)


def panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=6.45, fontweight="bold", pad=2.5)


def load_data() -> dict[str, pd.DataFrame]:
    names = [
        "all_synchronized_events", "all_event_members", "all_event_away_points",
        "fixed_gnss_like_examples",
        "alternating_gnss_decoy_pairs", "identity_replay_events",
        "bulk_identity_rebroadcast_sites", "bulk_identity_rebroadcast_members",
        "coherent_identity_replay_sites", "lima_replay_locations",
    ]
    date_columns = {
        "all_synchronized_events": ["onset_day"],
        "all_event_members": ["onset_day", "onset_ts", "t_first_away", "t_last_away"],
        "all_event_away_points": ["t_first", "t_last"],
        "fixed_gnss_like_examples": [
            "first_onset", "last_onset", "related_first_onset", "related_last_onset",
        ],
        "alternating_gnss_decoy_pairs": [
            "onset_day", "destination_a_first_seen", "destination_a_last_seen",
            "destination_b_first_seen", "destination_b_last_seen",
        ],
        "identity_replay_events": ["onset_day"],
        "bulk_identity_rebroadcast_sites": ["t_start", "t_end"],
        "bulk_identity_rebroadcast_members": ["first_seen", "last_seen"],
        "coherent_identity_replay_sites": ["first_seen", "last_seen"],
        "lima_replay_locations": ["first_seen", "last_seen"],
    }
    return {
        name: pd.read_csv(DATA / f"{name}.csv", parse_dates=date_columns.get(name))
        for name in names
    }


def fixed_example_label(row: pd.Series) -> str:
    if row["destination_lat"] < 35:
        return "Queen Alia area"
    if row["source_lat"] < 50:
        return "Azov / Rostov"
    if row["source_lat"] > 57:
        return "Pskov"
    if row["source_lon"] > 39:
        return "Petushki / Shatura"
    return "Moscow / Sheremetyevo"


def example_axis_histogram(
    example: pd.Series, away: pd.DataFrame, bins: np.ndarray
) -> np.ndarray:
    event_ids = set(str(example["qualifying_event_ids"]).split(";"))
    points = (
        away[away["event_id"].isin(event_ids)]
        .drop(columns="event_id")
        .drop_duplicates()
    )
    corridor = corridor_points(
        points, float(example["destination_lat"]), float(example["destination_lon"])
    )
    histogram, _ = np.histogram(
        corridor["axis_fraction"], bins=bins, weights=corridor["observations"]
    )
    return histogram / histogram.sum() if histogram.sum() else histogram.astype(float)


def fixed_examples_row(
    ax_map: plt.Axes,
    ax_shape: plt.Axes,
    d: dict,
    rings: list[Ring],
    *,
    tier: str,
    number: int,
) -> None:
    examples = d["fixed_gnss_like_examples"]
    examples = examples[examples["evidence_tier"].eq(tier)].copy()
    if tier == "strong":
        colors = [FIXED, "#df7580"]
        title = f"{number}  Strong fixed GNSS-like displacement — two examples"
        shape_title = "Member-only mixture: ≥25% intermediate mass"
    else:
        colors = ["#7f6427", FIXED_SUGGESTIVE, "#c49a44"]
        title = f"{number}  Suggestive fixed displacement — three examples"
        shape_title = "Member-only mixture: 15–25% intermediate mass"
    examples["label"] = examples.apply(fixed_example_label, axis=1)
    examples = examples.sort_values("label").reset_index(drop=True)
    ax_map.set_axis_off()
    panel_title(ax_map, title)
    gap = 0.035
    width = (1 - gap * (len(examples) - 1)) / len(examples)
    for index, (color, (_, row)) in enumerate(
        zip(colors, examples.iterrows(), strict=True)
    ):
        inset = ax_map.inset_axes([index * (width + gap), 0.02, width, 0.84])
        lon_span = abs(row.destination_lon - row.source_lon)
        lat_span = abs(row.destination_lat - row.source_lat)
        lon_margin = max(0.06, lon_span * 0.28)
        lat_margin = max(0.045, lat_span * 0.28)
        bbox = (
            min(row.source_lon, row.destination_lon) - lon_margin,
            max(row.source_lon, row.destination_lon) + lon_margin,
            min(row.source_lat, row.destination_lat) - lat_margin,
            max(row.source_lat, row.destination_lat) + lat_margin,
        )
        used_tiles = add_osm_basemap(
            inset,
            bbox,
            zoom=10 if lon_span < 0.32 else 9,
            alpha=1.0,
            grayscale=False,
            source="carto_voyager",
        )
        if not used_tiles:
            setup_map(inset, rings, bbox)
        draw_arc(
            inset,
            row.source_lon,
            row.source_lat,
            row.destination_lon,
            row.destination_lat,
            color=color,
            linewidth=0.8 + 0.08 * math.sqrt(row.qualifying_unique_cells),
            alpha=0.75,
            curvature=0.06,
        )
        inset.scatter(
            row.source_lon, row.source_lat, s=10, facecolor="white",
            edgecolor=color, linewidth=0.8, zorder=3,
        )
        inset.scatter(
            row.destination_lon, row.destination_lat, s=22, marker="*",
            color=color, edgecolor="white", linewidth=0.35, zorder=4,
        )
        for text, x, y, offset in [
            ("source", row.source_lon, row.source_lat, (2, 2)),
            ("destination", row.destination_lon, row.destination_lat, (-2, -7)),
        ]:
            inset.annotate(
                text, (x, y), xytext=offset, textcoords="offset points",
                fontsize=3.7, color=INK,
                ha="left" if offset[0] > 0 else "right", va="bottom",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.78},
                zorder=5,
            )
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(False)
        inset.set_title(
            f"{row.label}\n{row.median_displacement_km:.0f} km · "
            f"{int(row.qualifying_unique_cells)} cells",
            fontsize=4.9, fontweight="bold", color=color, pad=1.5,
        )
    ax_map.text(
        0.5, -0.04,
        "○ source   ★ common destination   ·   "
        "Basemap © OpenStreetMap contributors, © CARTO",
        transform=ax_map.transAxes, ha="center", va="top", fontsize=3.8, color=MUTED,
    )

    bins = np.linspace(0.15, 1.15, 26)
    centers = (bins[:-1] + bins[1:]) / 2
    for color, (_, row) in zip(colors, examples.iterrows(), strict=True):
        histogram = example_axis_histogram(row, d["all_event_away_points"], bins)
        share = 100 * row.member_intermediate_share
        ax_shape.plot(
            centers, histogram, color=color, linewidth=1.35,
            label=f"{row.label}  ({share:.1f}%)",
        )
    ax_shape.axvspan(0.2, 0.8, color="#eadfd0", alpha=0.55, zorder=-1)
    ax_shape.axvline(1.0, color=MUTED, linestyle=":", linewidth=0.7)
    ax_shape.set_xlim(0.15, 1.15)
    ax_shape.set_ylim(bottom=0)
    ax_shape.set_xlabel("Along home→destination axis  (1 = destination)")
    ax_shape.set_ylabel("Observation share")
    ax_shape.legend(frameon=False, loc="upper left", fontsize=4.8)
    ax_shape.grid(axis="y", color=GRID, linewidth=0.4)
    panel_title(ax_shape, shape_title)


def multiple_row(ax_space: plt.Axes, ax_time: plt.Axes, d: dict) -> None:
    pairs = d["alternating_gnss_decoy_pairs"]
    event_ids = set(pairs["event_id"])
    events = d["all_synchronized_events"].set_index("event_id")
    away = d["all_event_away_points"]
    points = away[away["event_id"].isin(event_ids)].copy()
    points["onset"] = points["event_id"].map(events["onset_day"])
    points = points[
        (points["t_first"] >= points["onset"]-pd.Timedelta(days=2))
        & (points["t_first"] <= points["onset"]+pd.Timedelta(days=MULTI_WINDOW_DAYS))
    ]
    endpoints = points[(points["observed_lon"] <= 40.25) | (points["observed_lon"] >= 40.45)].copy()
    endpoints["side"] = np.where(endpoints["observed_lon"] < 40.3, "west", "east")
    colors = endpoints["side"].map({"west": MULTIPLE_WEST, "east": MULTIPLE_EAST})
    refs = points[KEY+["reference_lat", "reference_lon"]].drop_duplicates()
    ax_space.scatter(refs["reference_lon"], refs["reference_lat"], s=7, facecolor="white",
                     edgecolor=MUTED, linewidth=0.55, label="stable homes")
    ax_space.scatter(endpoints["observed_lon"], endpoints["observed_lat"],
                     s=5+2*np.sqrt(endpoints["observations"]), c=colors,
                     alpha=0.7, edgecolor="none")
    ax_space.set_xlim(40.0, 40.62)
    ax_space.set_ylim(56.02, 56.24)
    ax_space.set_xticks([40.0, 40.2, 40.4, 40.6])
    ax_space.set_xlabel("Longitude")
    ax_space.set_ylabel("Latitude")
    ax_space.grid(color=GRID, linewidth=0.4)
    ax_space.legend(frameon=False, loc="upper center", fontsize=4.9)
    panel_title(ax_space, "3  Multiple destinations — shared cells, opposite bearings")

    ax_time.scatter(endpoints["t_first"], endpoints["observed_lon"],
                    s=5+2*np.sqrt(endpoints["observations"]), c=colors,
                    alpha=0.72, edgecolor="none")
    ax_time.axhline(40.3, color=MUTED, linestyle=":", linewidth=0.7)
    ax_time.set_ylabel("Destination longitude")
    ax_time.set_xlabel("First observed at position")
    ax_time.xaxis.set_major_locator(mdates.MonthLocator())
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_time.grid(color=GRID, linewidth=0.4)
    panel_title(ax_time, "West ends in March; east begins two days later")


def identity_row(ax_map: plt.Axes, ax_bar: plt.Axes, d: dict, rings: list[Ring]) -> None:
    sites = d["coherent_identity_replay_sites"].copy()
    lima = d["lima_replay_locations"]
    lima_count = int(lima.loc[lima["location"].eq("Lima replay"), "identities"].iloc[0])
    sites.loc[sites["destination_name"].eq("Lima"), "identities"] = lima_count
    setup_map(ax_map, rings, (-90, 125, -25, 62))
    for row in sites.sort_values("identities").itertuples():
        draw_arc(ax_map, row.source_lon, row.source_lat, row.destination_lon, row.destination_lat,
                 color=REPLAY, linewidth=0.35+0.055*math.sqrt(row.identities), alpha=0.55)
    ax_map.scatter(sites["source_lon"], sites["source_lat"], s=7, facecolor="white",
                   edgecolor=REPLAY, linewidth=0.7, zorder=3)
    ax_map.scatter(sites["destination_lon"], sites["destination_lat"], s=15,
                   marker="D", color=REPLAY, edgecolor="white", linewidth=0.35, zorder=4)
    # Retain the two smaller synchronized-onset examples that fall below the
    # global >=40-identity endpoint scan; omit Crimea because the deeper site
    # audit shows that endpoint is dominated by a broad-source cluster.
    events = d["identity_replay_events"].copy()
    small = events[
        events["destination_lon"].round(2).isin([55.56, 117.21])
    ].copy()
    for row in small.itertuples():
        draw_arc(ax_map, row.source_lon, row.source_lat, row.destination_lon, row.destination_lat,
                 color=REPLAY, linewidth=0.65, alpha=0.55)
    ax_map.scatter(small["source_lon"], small["source_lat"], s=5, facecolor="white",
                   edgecolor=REPLAY, linewidth=0.55, zorder=3)
    ax_map.scatter(small["destination_lon"], small["destination_lat"], s=11,
                   marker="D", facecolor="white", edgecolor=REPLAY, linewidth=0.7, zorder=4)
    ax_map.set_xticks([-60, 0, 60, 120])
    ax_map.set_yticks([-20, 20, 60])
    panel_title(ax_map, "4  Identity replay — stable homes plus distant clones")

    routes = sites.assign(
        route=sites["source_region"] + " → " + sites["destination_name"],
        cells=sites["identities"],
        distance=sites["median_displacement_km"],
    )[["route", "cells", "distance"]]
    small_routes = pd.DataFrame([
        {"route": "Samara → Oman", "cells": int(small.loc[
            small["destination_lon"].round(2).eq(55.56), "n_cells"].sum()),
         "distance": float(small.loc[
             small["destination_lon"].round(2).eq(55.56), "median_baseline_km"].median())},
        {"route": "Tatarstan → Shandong", "cells": int(small.loc[
            small["destination_lon"].round(2).eq(117.21), "n_cells"].sum()),
         "distance": float(small.loc[
             small["destination_lon"].round(2).eq(117.21), "median_baseline_km"].median())},
    ])
    routes = pd.concat([routes, small_routes], ignore_index=True).sort_values("cells")
    bars = ax_bar.barh(routes["route"], routes["cells"], color=REPLAY, height=0.62)
    for bar, row in zip(bars, routes.itertuples(index=False), strict=True):
        ax_bar.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    f"{int(row.cells)} cells · {row.distance:,.0f} km", va="center", fontsize=5.1)
    ax_bar.set_xlim(0, routes["cells"].max()*1.47)
    ax_bar.set_xlabel("Distinct identities at distant endpoint")
    ax_bar.grid(axis="x", color=GRID, linewidth=0.4)
    panel_title(ax_bar, "Five large clusters + two smaller onset sets")


MCC_LABEL = {
    250: "Russia 250", 425: "Israel/PS 425", 724: "Brazil 724", 740: "Ecuador 740",
    257: "Belarus 257", 525: "Singapore 525", 415: "Lebanon 415", 520: "Thailand 520",
    234: "UK 234", 240: "Sweden 240", 452: "Vietnam 452",
}


def bulk_row(ax_map: plt.Axes, ax_bar: plt.Axes, d: dict, rings: list[Ring]) -> None:
    members = d["bulk_identity_rebroadcast_members"]
    site = d["bulk_identity_rebroadcast_sites"].iloc[0]
    setup_map(ax_map, rings, (-180, 180, -55, 78))
    for row in members.itertuples():
        draw_arc(ax_map, row.home_lon, row.home_lat, site.destination_lon, site.destination_lat,
                 color=REBROADCAST, linewidth=0.25, alpha=0.10, curvature=0.08, zorder=1)
    ax_map.scatter(members["home_lon"], members["home_lat"], s=4, color=REBROADCAST,
                   alpha=0.55, edgecolor="none", zorder=2)
    ax_map.scatter([site.destination_lon], [site.destination_lat], s=30, marker="*",
                   color=REBROADCAST, edgecolor="white", linewidth=0.45, zorder=4)
    ax_map.text(site.destination_lon+5, site.destination_lat+1, "Weihai", fontsize=5.2,
                color=REBROADCAST, fontweight="bold")
    ax_map.set_xticks([-120, -60, 0, 60, 120])
    ax_map.set_yticks([-30, 0, 30, 60])
    panel_title(ax_map, "5  Bulk rebroadcast — 157 identities from 11 MCCs")

    counts = members.groupby("mcc").size().sort_values(ascending=False)
    top = counts.head(7)
    if len(counts) > 7:
        top.loc[-1] = counts.iloc[7:].sum()
    labels = [MCC_LABEL.get(int(mcc), f"MCC {int(mcc)}") if mcc != -1 else "Other MCCs"
              for mcc in top.index]
    plot = pd.Series(top.to_numpy(), index=labels).sort_values()
    ax_bar.barh(plot.index, plot.values, color=REBROADCAST, height=0.62)
    for y, value in enumerate(plot.values):
        ax_bar.text(value+0.8, y, str(int(value)), va="center", fontsize=5.1)
    ax_bar.set_xlim(0, plot.max()*1.22)
    ax_bar.set_xlabel("Distinct identities at the exact Weihai point")
    ax_bar.grid(axis="x", color=GRID, linewidth=0.4)
    panel_title(ax_bar, "MCC composition at the exact Weihai point")


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 5.8,
        "axes.labelsize": 5.5,
        "xtick.labelsize": 5.0,
        "ytick.labelsize": 5.0,
        "legend.fontsize": 5.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    rings = load_world(WORLD)
    fig = plt.figure(figsize=(7.15, 9.25))
    grid = fig.add_gridspec(
        5, 2, width_ratios=[1.08, 0.92],
        left=0.085, right=0.992, bottom=0.042, top=0.978,
        hspace=0.64, wspace=0.28,
    )
    axes = [(fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1])) for row in range(5)]

    fixed_examples_row(*axes[0], data, rings, tier="strong", number=1)
    fixed_examples_row(*axes[1], data, rings, tier="suggestive", number=2)
    multiple_row(*axes[2], data)
    identity_row(*axes[3], data, rings)
    bulk_row(*axes[4], data, rings)

    for row in range(1, 5):
        y = axes[row][0].get_position().y1 + 0.018
        fig.add_artist(plt.Line2D([0.065, 0.995], [y, y], transform=fig.transFigure,
                                  color="#c8c4bd", linewidth=0.45))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if preview is not None:
        fig.savefig(preview, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_OUTPUT.with_suffix(".png"))
    args = parser.parse_args()
    make_figure(load_data(), args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
