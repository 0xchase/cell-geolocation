#!/usr/bin/env python3
"""Plot compact route maps beside time-versus-displacement trajectories.

The default path reads reproducible CSV outputs under
``data/spoofing/high_quality``.  ``--refresh-trajectories`` refreshes hourly
position aggregates through the repository's read-only ClickHouse helper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from spoofing_category_overview import load_world, setup_map


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing" / "high_quality"
FIGS = ROOT / "paper" / "figs"
WORLD = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
TRAJECTORY_DATA = DATA / "controlled_campaign_trajectories.csv"

MUTED = "#686d72"
GRID = "#d9dcdf"
SOURCE = "#276c99"
DESTINATION = "#b33b45"
CONTROLLED = "#2878a8"
CAMPAIGN_COLORS = ["#2878a8", "#b24d86", "#d27b25", "#32866e"]


@dataclass(frozen=True)
class Campaign:
    key: str
    source: str
    destination: str
    source_lat: float
    source_lon: float
    destination_lat: float
    destination_lon: float
    first_day: str
    last_day: str
    source_tolerance: float = 0.18
    destination_tolerance: float = 0.07


CAMPAIGNS = {
    "pakistan": Campaign(
        "pakistan", "Islamabad region, Pakistan", "Changsha, China",
        33.6014, 73.1611, 28.19, 113.22, "2025-01-11", "2025-02-12",
    ),
    "mashhad": Campaign(
        "mashhad", "Mashhad, Iran", "Tehran, Iran",
        36.2830, 59.61, 35.67, 51.50, "2025-08-27", "2025-08-28",
    ),
    "kazerun": Campaign(
        "kazerun", "Kazerun / Fars, Iran", "Tehran, Iran",
        29.63, 51.63, 35.67, 51.50, "2025-06-20", "2025-06-20",
    ),
    "urmia": Campaign(
        "urmia", "Urmia, Iran", "Tehran, Iran",
        37.55, 45.01, 35.68, 51.50, "2024-02-29", "2024-02-29",
    ),
    "moscow": Campaign(
        "moscow", "Moscow, Russia", "Crimea",
        55.75, 37.66, 45.05, 33.98, "2025-07-28", "2025-07-28",
    ),
    "tatarstan": Campaign(
        "tatarstan", "Tatarstan, Russia", "Jinan, China",
        55.82, 52.06, 36.85, 117.21, "2024-09-02", "2024-09-02",
    ),
    "myanmar_oct": Campaign(
        "myanmar_oct", "Oct. 8 source, Shan State", "Yunnan, China",
        20.79, 97.03, 22.37, 99.86, "2024-10-08", "2024-10-08",
    ),
    "myanmar_feb": Campaign(
        "myanmar_feb", "Feb. 12 source, Shan State", "Yunnan, China",
        20.80, 97.17, 22.36, 99.86, "2025-02-12", "2025-02-12",
    ),
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.2,
        "axes.titlesize": 7.3,
        "axes.labelsize": 5.8,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def load_inputs() -> pd.DataFrame:
    members = pd.read_csv(
        DATA / "method2_members.csv",
        parse_dates=["onset_ts", "destination_first", "destination_last"],
    )
    return members


def select_members(members: pd.DataFrame, spec: Campaign) -> pd.DataFrame:
    start = pd.Timestamp(spec.first_day)
    end = pd.Timestamp(spec.last_day) + pd.Timedelta(days=1)
    selected = members[
        ((members["src_lat10"] / 10 - spec.source_lat).abs() < spec.source_tolerance)
        & ((members["src_lon10"] / 10 - spec.source_lon).abs() < spec.source_tolerance)
        & ((members["top_plat"] / 100 - spec.destination_lat).abs() < spec.destination_tolerance)
        & ((members["top_plon"] / 100 - spec.destination_lon).abs() < spec.destination_tolerance)
        & members["onset_ts"].ge(start)
        & members["onset_ts"].lt(end)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No controlled members found for {spec.key}")
    return selected


def _reference_tuples(group: pd.DataFrame) -> str:
    rows = group[["mcc", "mnc", "lac", "cid", "cell_type", "rlat", "rlon"]].drop_duplicates()
    return ",".join(
        "tuple(toUInt16({}),toUInt16({}),toUInt32({}),toInt64({}),'{}',"
        "toFloat64({}),toFloat64({}))".format(
            int(row.mcc), int(row.mnc), int(row.lac), int(row.cid), row.cell_type,
            float(row.rlat) / 100, float(row.rlon) / 100,
        )
        for row in rows.itertuples(index=False)
    )


def _identity_tuples(group: pd.DataFrame) -> str:
    rows = group[["mcc", "mnc", "lac", "cid", "cell_type"]].drop_duplicates()
    return ",".join(
        f"({int(row.mcc)},{int(row.mnc)},{int(row.lac)},{int(row.cid)},'{row.cell_type}')"
        for row in rows.itertuples(index=False)
    )


def refresh_trajectories(members: pd.DataFrame) -> pd.DataFrame:
    """Read hourly position samples for the plotted identities from ClickHouse."""
    from ch_remote import ch_df

    frames = []
    for key, spec in CAMPAIGNS.items():
        group = select_members(members, spec)
        refs = _reference_tuples(group)
        identities = _identity_tuples(group)
        lookback_days = 100 if key == "pakistan" else 7
        start = group["onset_ts"].min().floor("D") - pd.Timedelta(days=lookback_days)
        end = group["destination_last"].max().ceil("D") + pd.Timedelta(days=7)
        frame = ch_df(f"""
          WITH refs AS (
            SELECT tupleElement(x,1) AS mcc,tupleElement(x,2) AS mnc,
                   tupleElement(x,3) AS lac,tupleElement(x,4) AS cid,
                   tupleElement(x,5) AS cell_type,tupleElement(x,6) AS ref_lat,
                   tupleElement(x,7) AS ref_lon
            FROM (SELECT arrayJoin([{refs}]) AS x)
          )
          SELECT g.mcc,g.mnc,g.lac,g.cid,toString(g.cell_type) AS cell_type,
                 toStartOfHour(g.timestamp) AS timestamp,g.lat,g.lon,
                 count() AS samples,r.ref_lat AS reference_lat,r.ref_lon AS reference_lon,
                 greatCircleDistance(g.lon,g.lat,r.ref_lon,r.ref_lat)/1000 AS displacement_km
          FROM (
            SELECT * FROM cell.geos
            PREWHERE (mcc,mnc,lac,cid,cell_type) IN ({identities})
            WHERE timestamp >= toDateTime('{start:%Y-%m-%d %H:%M:%S}')
              AND timestamp < toDateTime('{end:%Y-%m-%d %H:%M:%S}')
              AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
              AND NOT (abs(lat)<=0.01 AND abs(lon)<=0.01)
          ) AS g
          INNER JOIN refs AS r ON g.mcc=r.mcc AND g.mnc=r.mnc AND g.lac=r.lac
            AND g.cid=r.cid AND toString(g.cell_type)=r.cell_type
          GROUP BY g.mcc,g.mnc,g.lac,g.cid,g.cell_type,timestamp,g.lat,g.lon,
                   r.ref_lat,r.ref_lon
          ORDER BY timestamp,g.mcc,g.mnc,g.lac,g.cid
        """, settings={"max_threads": 6})
        frame.insert(0, "campaign", key)
        frames.append(frame)
        print(f"{key}: {len(group)} identities, {len(frame):,} hourly position points")
    trajectories = pd.concat(frames, ignore_index=True)
    trajectories["timestamp"] = pd.to_datetime(trajectories["timestamp"])
    trajectories.to_csv(TRAJECTORY_DATA, index=False)
    return trajectories


def load_trajectories() -> pd.DataFrame:
    if not TRAJECTORY_DATA.exists():
        raise FileNotFoundError(
            f"{TRAJECTORY_DATA} is missing; rerun with --refresh-trajectories"
        )
    return pd.read_csv(TRAJECTORY_DATA, parse_dates=["timestamp"])


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6371.0088
    lat1, lat2 = np.radians([a_lat, b_lat])
    dlat = lat2 - lat1
    dlon = np.radians(b_lon - a_lon)
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(h)))


def draw_route_map(
    ax: plt.Axes,
    rings,
    specs: list[Campaign],
    bbox: tuple[float, float, float, float],
    labels: list[tuple[str, float, float]],
    colors: list[str] | None = None,
    source_offsets: list[tuple[float, float]] | None = None,
    distance_texts: list[str | None] | None = None,
    destination_offset: tuple[float, float] = (4, -9),
    destination_ha: str = "left",
    title: str = "A. Reported displacement geography",
) -> None:
    setup_map(ax, rings, bbox)
    colors = colors or [SOURCE] * len(specs)
    source_offsets = source_offsets or [(4, 5)] * len(specs)
    for index, (spec, color) in enumerate(zip(specs, colors)):
        arrow = FancyArrowPatch(
            (spec.source_lon, spec.source_lat),
            (spec.destination_lon, spec.destination_lat),
            arrowstyle="-|>", connectionstyle="arc3,rad=-0.10",
            mutation_scale=8, linewidth=1.25, color=color, alpha=0.88, zorder=4,
        )
        ax.add_patch(arrow)
        ax.scatter(spec.source_lon, spec.source_lat, s=27, marker="o", c=color,
                   edgecolors="white", linewidths=0.65, zorder=6)
        ax.scatter(spec.destination_lon, spec.destination_lat, s=54, marker="*", c=DESTINATION,
                   edgecolors="white", linewidths=0.65, zorder=7)
        midpoint_lon = (spec.source_lon + spec.destination_lon) / 2
        midpoint_lat = (spec.source_lat + spec.destination_lat) / 2
        distance_text = (
            distance_texts[index] if distance_texts is not None
            else f"{haversine_km(spec.source_lat, spec.source_lon, spec.destination_lat, spec.destination_lon):,.0f} km"
        )
        if distance_text:
            ax.text(midpoint_lon, midpoint_lat + 0.35, distance_text,
                    fontsize=4.8, color=color, ha="center", va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6}, zorder=8)
    for text, lon, lat in labels:
        ax.text(lon, lat, text, ha="center", va="center", fontsize=5.1,
                color="#73777a", fontweight="bold", alpha=0.86, zorder=2)
    for i, spec in enumerate(specs):
        color = colors[i]
        ax.annotate(spec.source, (spec.source_lon, spec.source_lat), xytext=source_offsets[i],
                    textcoords="offset points", fontsize=5.4, color=color,
                    fontweight="bold", zorder=9)
    destinations: dict[tuple[float, float], str] = {}
    for spec in specs:
        destinations[(spec.destination_lon, spec.destination_lat)] = spec.destination
    for (lon, lat), destination in destinations.items():
        ax.annotate(destination, (lon, lat), xytext=destination_offset, textcoords="offset points",
                    ha=destination_ha, fontsize=5.4, color=DESTINATION, fontweight="bold", zorder=9)
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def draw_displacement_scatter(
    ax: plt.Axes,
    trajectories: list[pd.DataFrame],
    specs: list[Campaign],
    labels: list[str],
    colors: list[str],
    title: str = "B. Displacement from stable home over time",
) -> None:
    for trajectory, spec, label, color in zip(
        trajectories, specs, labels, colors, strict=True
    ):
        sizes = 3.0 + 1.5 * np.log1p(trajectory["samples"].clip(lower=1))
        ax.scatter(
            trajectory["timestamp"], trajectory["displacement_km"],
            s=sizes, color=color, alpha=0.30, linewidths=0, rasterized=True,
            label=label, zorder=3,
        )
        route_km = haversine_km(
            spec.source_lat, spec.source_lon, spec.destination_lat, spec.destination_lon
        )
        ax.hlines(
            route_km, trajectory["timestamp"].min(), trajectory["timestamp"].max(),
            color=color, linewidth=0.8, linestyle=(0, (3, 2)), alpha=0.85, zorder=2,
        )
    combined = pd.concat(trajectories, ignore_index=True)
    expected = max(
        haversine_km(s.source_lat, s.source_lon, s.destination_lat, s.destination_lon)
        for s in specs
    )
    upper = max(expected * 1.10, float(combined["displacement_km"].quantile(0.995)) * 1.04)
    ax.set_ylim(-upper * 0.015, upper)
    ax.set_ylabel("Displacement from stable home (km)")
    ax.set_xlabel("Observation time")
    ax.grid(color=GRID, linewidth=0.45, zorder=-2)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title(title, loc="left", fontweight="bold", pad=2)
    ax.legend(
        loc="upper left", frameon=True, facecolor="white", edgecolor="none",
        framealpha=0.82, fontsize=4.8, markerscale=1.6, ncol=min(len(labels), 3),
        handletextpad=0.35, columnspacing=0.8,
    )
    if len(specs) == 1:
        route_km = haversine_km(
            specs[0].source_lat, specs[0].source_lon,
            specs[0].destination_lat, specs[0].destination_lon,
        )
        ax.text(
            0.99, min(route_km / upper + 0.025, 0.94), f"destination ≈ {route_km:,.0f} km",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=4.7,
            color=colors[0],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.5},
        )


def render_single(
    members: pd.DataFrame,
    trajectory_data: pd.DataFrame,
    rings,
    spec: Campaign,
    output: Path,
    bbox: tuple[float, float, float, float],
    labels: list[tuple[str, float, float]],
    destination_offset: tuple[float, float] = (4, -9),
    destination_ha: str = "left",
) -> None:
    group = select_members(members, spec)
    trajectory = trajectory_data[trajectory_data["campaign"].eq(spec.key)].copy()
    if trajectory.empty:
        raise RuntimeError(f"No trajectory points found for {spec.key}")
    fig = plt.figure(figsize=(7.15, 1.95))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[0.88, 1.38],
        left=0.06, right=0.99, bottom=0.20, top=0.91, wspace=0.23,
    )
    draw_route_map(
        fig.add_subplot(grid[0, 0]), rings, [spec], bbox, labels,
        destination_offset=destination_offset, destination_ha=destination_ha,
    )
    draw_displacement_scatter(
        fig.add_subplot(grid[0, 1]), [trajectory], [spec],
        [f"{len(group)} controlled identities"], [CONTROLLED],
    )
    fig.text(0.99, 0.02, "Natural Earth · coordinate-bin centers shown", ha="right", va="bottom",
             fontsize=4.3, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=500)
    plt.close(fig)


def draw_pakistan_map(
    ax: plt.Axes, rings, group: pd.DataFrame, spec: Campaign,
) -> None:
    """Render the actual Islamabad source footprint and common destination."""
    bbox = (65.0, 119.0, 21.0, 40.5)
    setup_map(ax, rings, bbox)
    source_lat = group["rlat"] / 100
    source_lon = group["rlon"] / 100
    destination_lat = group["top_plat"] / 100
    destination_lon = group["top_plon"] / 100
    source_center = (float(source_lon.median()), float(source_lat.median()))
    destination_center = (float(destination_lon.median()), float(destination_lat.median()))

    arrow = FancyArrowPatch(
        source_center, destination_center, arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.10", mutation_scale=8,
        linewidth=1.2, color=MUTED, alpha=0.72, zorder=3,
    )
    ax.add_patch(arrow)
    ax.scatter(
        source_lon, source_lat, s=8, color=SOURCE, alpha=0.55,
        edgecolors="white", linewidths=0.25, zorder=4,
    )
    ax.scatter(
        *source_center, s=28, color=SOURCE, edgecolors="white",
        linewidths=0.6, zorder=5,
    )
    ax.scatter(
        destination_lon, destination_lat, s=8, color=DESTINATION, alpha=0.55,
        edgecolors="white", linewidths=0.25, zorder=4,
    )
    ax.scatter(
        *destination_center, s=38, marker="D", color=DESTINATION,
        edgecolors="white", linewidths=0.6, zorder=5,
    )
    ax.annotate(
        "Islamabad sources", source_center, xytext=(5, 7),
        textcoords="offset points", fontsize=5.4, color=SOURCE,
        fontweight="bold", zorder=6,
    )
    ax.annotate(
        "Common destination\nChangsha", destination_center, xytext=(-5, -8),
        textcoords="offset points", ha="right", va="top", fontsize=5.4,
        color=DESTINATION, fontweight="bold", zorder=6,
    )
    midpoint = (
        (source_center[0] + destination_center[0]) / 2,
        (source_center[1] + destination_center[1]) / 2 + 0.45,
    )
    ax.text(
        *midpoint,
        f"{haversine_km(spec.source_lat, spec.source_lon, spec.destination_lat, spec.destination_lon):,.0f} km",
        ha="center", va="bottom", fontsize=5.0, color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
        zorder=6,
    )
    for text, lon, lat in [
        ("PAKISTAN", 69.5, 30.5), ("INDIA", 79.0, 24.5), ("CHINA", 102.0, 35.3)
    ]:
        ax.text(
            lon, lat, text, ha="center", va="center", fontsize=5.0,
            color="#7c8083", fontweight="bold", alpha=0.75, zorder=2,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Islamabad to Changsha", loc="left", pad=3)


def draw_pakistan_activity(
    ax: plt.Axes, trajectory: pd.DataFrame, spec: Campaign,
) -> None:
    """Show cohort participation at each endpoint without an empty 4,000 km axis."""
    key = ["mcc", "mnc", "lac", "cid", "cell_type"]
    route_km = haversine_km(
        spec.source_lat, spec.source_lon, spec.destination_lat, spec.destination_lon
    )
    frame = trajectory.copy()
    frame["location"] = np.where(
        frame["displacement_km"].gt(route_km / 2), "Changsha", "Islamabad"
    )
    frame["week"] = frame["timestamp"].dt.to_period("W-MON").dt.start_time
    weekly = (
        frame[["week", "location", *key]].drop_duplicates()
        .groupby(["week", "location"]).size().unstack(fill_value=0)
        .reindex(columns=["Islamabad", "Changsha"], fill_value=0)
    )
    styles = {"Islamabad": SOURCE, "Changsha": DESTINATION}
    for location in ["Islamabad", "Changsha"]:
        ax.plot(
            weekly.index, weekly[location], color=styles[location],
            linewidth=1.25, marker="o", markersize=2.2,
            markeredgecolor="white", markeredgewidth=0.25,
            label=location, zorder=3,
        )
    context_start = trajectory["timestamp"].min().normalize().replace(day=1)
    ax.set_xlim(context_start, pd.Timestamp("2025-10-05"))
    ax.set_ylim(-2, 75)
    ax.set_ylabel("Distinct identities observed per week")
    ax.set_xlabel("Observation week")
    ax.grid(color=GRID, linewidth=0.45, zorder=-2)
    locator = mdates.MonthLocator(interval=2)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title("Weekly activity at each location", loc="left", pad=3)
    ax.legend(
        loc="upper right", ncol=2, frameon=True, facecolor="white",
        edgecolor="none", framealpha=0.82, fontsize=4.8,
        handletextpad=0.35, columnspacing=0.8,
    )


def render_pakistan(
    members: pd.DataFrame, trajectory_data: pd.DataFrame, rings, output: Path,
) -> None:
    spec = CAMPAIGNS["pakistan"]
    group = select_members(members, spec)
    trajectory = trajectory_data[trajectory_data["campaign"].eq(spec.key)].copy()
    if trajectory.empty:
        raise RuntimeError("No trajectory points found for pakistan")
    fig = plt.figure(figsize=(7.15, 2.15))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[0.90, 1.50],
        left=0.045, right=0.99, bottom=0.20, top=0.92, wspace=0.17,
    )
    draw_pakistan_map(fig.add_subplot(grid[0, 0]), rings, group, spec)
    draw_pakistan_activity(fig.add_subplot(grid[0, 1]), trajectory, spec)
    fig.text(
        0.045, 0.02, "Boundaries: Natural Earth", ha="left", va="bottom",
        fontsize=4.3, color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=500)
    plt.close(fig)


def render_iran(
    members: pd.DataFrame, trajectory_data: pd.DataFrame, rings, output: Path,
) -> None:
    specs = [CAMPAIGNS["urmia"], CAMPAIGNS["kazerun"], CAMPAIGNS["mashhad"]]
    groups = [select_members(members, spec) for spec in specs]
    row_labels = ["Urmia → Tehran", "Kazerun → Tehran", "Mashhad → Tehran"]
    colors = [CAMPAIGN_COLORS[1], CAMPAIGN_COLORS[2], CAMPAIGN_COLORS[0]]
    trajectories = [trajectory_data[trajectory_data["campaign"].eq(spec.key)].copy() for spec in specs]
    fig = plt.figure(figsize=(7.15, 2.15))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[0.88, 1.38],
        left=0.06, right=0.99, bottom=0.19, top=0.91, wspace=0.23,
    )
    draw_route_map(
        fig.add_subplot(grid[0, 0]), rings, specs, (42.0, 62.0, 26.0, 39.5),
        [("IRAN", 53.2, 32.5), ("TURKEY", 44.5, 38.2), ("IRAQ", 44.8, 33.0),
         ("AFGHANISTAN", 61.0, 34.0), ("Persian Gulf", 51.5, 27.7)],
        colors=colors,
    )
    draw_displacement_scatter(
        fig.add_subplot(grid[0, 1]), trajectories, specs,
        [f"{label} ({len(group)})" for label, group in zip(row_labels, groups)], colors,
    )
    fig.text(0.99, 0.02, "Natural Earth · coordinate-bin centers shown", ha="right", va="bottom",
             fontsize=4.3, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=500)
    plt.close(fig)


def render_myanmar(
    members: pd.DataFrame, trajectory_data: pd.DataFrame, rings, output: Path,
) -> None:
    specs = [CAMPAIGNS["myanmar_oct"], CAMPAIGNS["myanmar_feb"]]
    groups = [select_members(members, spec) for spec in specs]
    row_labels = ["Oct. 8, 2024", "Feb. 12, 2025"]
    colors = [CAMPAIGN_COLORS[1], CAMPAIGN_COLORS[0]]
    trajectories = [trajectory_data[trajectory_data["campaign"].eq(spec.key)].copy() for spec in specs]
    fig = plt.figure(figsize=(7.15, 2.05))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[0.88, 1.38],
        left=0.06, right=0.99, bottom=0.20, top=0.91, wspace=0.23,
    )
    draw_route_map(
        fig.add_subplot(grid[0, 0]), rings, specs, (94.0, 102.2, 18.2, 24.7),
        [("MYANMAR", 96.1, 22.6), ("CHINA / YUNNAN", 100.2, 23.5),
         ("THAILAND", 98.0, 19.3), ("LAOS", 101.2, 20.8)], colors=colors,
        source_offsets=[(4, 9), (4, -13)], distance_texts=["328–341 km", None],
    )
    draw_displacement_scatter(
        fig.add_subplot(grid[0, 1]), trajectories, specs,
        [f"{label} ({len(group)})" for label, group in zip(row_labels, groups)], colors,
    )
    fig.text(0.99, 0.02, "Natural Earth · coordinate-bin centers shown", ha="right", va="bottom",
             fontsize=4.3, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=500)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-trajectories", action="store_true",
        help="Refresh the hourly observation CSV through the read-only database connection.",
    )
    args = parser.parse_args()
    configure_style()
    members = load_inputs()
    if args.refresh_trajectories:
        trajectory_data = refresh_trajectories(members)
    else:
        trajectory_data = load_trajectories()
    rings = load_world(WORLD)
    render_pakistan(members, trajectory_data, rings, FIGS / "pakistan_changsha_campaign.pdf")
    render_iran(members, trajectory_data, rings, FIGS / "iran_controlled_campaigns.pdf")
    render_single(
        members, trajectory_data, rings, CAMPAIGNS["moscow"], FIGS / "moscow_crimea_campaign.pdf",
        (27.0, 43.5, 42.0, 58.5),
        [("RUSSIA", 39.5, 54.0), ("UKRAINE", 31.4, 49.4), ("Black Sea", 34.0, 43.3)],
    )
    render_single(
        members, trajectory_data, rings, CAMPAIGNS["tatarstan"], FIGS / "tatarstan_jinan_campaign.pdf",
        (44.0, 123.0, 30.0, 60.5),
        [("RUSSIA", 67.0, 56.0), ("KAZAKHSTAN", 67.0, 45.0), ("MONGOLIA", 102.0, 47.5),
         ("CHINA", 105.0, 34.5)],
        destination_offset=(-4, -9), destination_ha="right",
    )
    render_myanmar(members, trajectory_data, rings, FIGS / "myanmar_yunnan_campaigns.pdf")
    for name in [
        "pakistan_changsha_campaign.pdf", "iran_controlled_campaigns.pdf",
        "moscow_crimea_campaign.pdf", "tatarstan_jinan_campaign.pdf",
        "myanmar_yunnan_campaigns.pdf",
    ]:
        print(FIGS / name)


if __name__ == "__main__":
    main()
