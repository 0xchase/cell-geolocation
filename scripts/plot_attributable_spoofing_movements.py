#!/usr/bin/env python3
"""Plot five attribution-screen positives as geographic movements.

The figure is intentionally geography-first: each panel draws the returned
cell-position trajectories over a quiet labelled basemap.  Figure generation
reads the auditable CSVs under ``data/spoofing``; ``--refresh-timelines``
refreshes the two cohort timelines through the read-only database helper.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_helpers import add_osm_basemap


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"
ATTRIBUTION = DATA / "attribution_search"
OUTPUT = ROOT / "paper" / "figs" / "attributable_spoofing_movements"
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]

BLUE = "#286f9b"
RED = "#b43c49"
PURPLE = "#7654a6"
INK = "#272b2e"
MUTED = "#6f767b"
PALE = "#e8ecee"

QUEEN_ALIA = (35.993, 31.723)
SHEREMETYEVO = (37.415, 55.972)
QUEEN_TIMELINE = ATTRIBUTION / "queen_alia_weekly_distance.csv"
MOSCOW_TIMELINE = ATTRIBUTION / "moscow_weekly_distance.csv"
CONTROLLED_TRAJECTORIES = DATA / "high_quality" / "controlled_campaign_trajectories.csv"
LYON_VORONEZH_TRAJECTORY = ATTRIBUTION / "lyon_voronezh_trajectory.csv"


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.2,
        "axes.titlesize": 6.3,
        "axes.labelsize": 5.8,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))


def square_bbox(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    padding: float = 1.16,
    minimum_km: float = 28,
) -> tuple[float, float, float, float]:
    """Return a square-in-distance lon/lat extent for a square plot panel."""
    lon_mid = float((np.min(lons) + np.max(lons)) / 2)
    lat_mid = float((np.min(lats) + np.max(lats)) / 2)
    cos_lat = max(math.cos(math.radians(lat_mid)), 0.25)
    width_km = float(np.max(lons) - np.min(lons)) * 111.32 * cos_lat
    height_km = float(np.max(lats) - np.min(lats)) * 111.32
    span_km = max(width_km, height_km, minimum_km) * padding
    lon_span = span_km / (111.32 * cos_lat)
    lat_span = span_km / 111.32
    return (
        lon_mid - lon_span / 2,
        lon_mid + lon_span / 2,
        lat_mid - lat_span / 2,
        lat_mid + lat_span / 2,
    )


def zoom_for(bbox: tuple[float, float, float, float]) -> int:
    lon_mid = (bbox[0] + bbox[1]) / 2
    lat_mid = (bbox[2] + bbox[3]) / 2
    width = float(haversine_km(bbox[0], lat_mid, bbox[1], lat_mid))
    height = float(haversine_km(lon_mid, bbox[2], lon_mid, bbox[3]))
    span = max(width, height)
    if span > 2500:
        return 4
    if span > 1200:
        return 5
    if span > 600:
        return 6
    if span > 250:
        return 7
    if span > 120:
        return 8
    if span <= 35:
        return 11
    if span <= 75:
        return 10
    return 9


def setup_panel(ax: plt.Axes, bbox: tuple[float, float, float, float], title: str) -> None:
    ax.set_facecolor(PALE)
    add_osm_basemap(
        ax,
        bbox,
        zoom=zoom_for(bbox),
        source="carto_voyager",
        alpha=0.91,
        grayscale=True,
        grayscale_brightness=1.02,
        grayscale_contrast=1.05,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_box_aspect(1)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#73797d")


def label_point(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    color: str,
    *,
    offset: tuple[float, float] = (4, 4),
    align: str = "left",
) -> None:
    ax.annotate(
        text,
        xy,
        xytext=offset,
        textcoords="offset points",
        ha=align,
        va="bottom",
        fontsize=5.4,
        fontweight="bold",
        color=color,
        bbox={
            "facecolor": "white",
            "edgecolor": "#7e8488",
            "linewidth": 0.72,
            "alpha": 0.94,
            "pad": 1.05,
        },
        zorder=12,
    )


def add_scale_bar(ax: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    west, east, south, north = bbox
    width_km = float(haversine_km(west, (south + north) / 2, east, (south + north) / 2))
    options = np.array([2, 5, 10, 20, 25, 50, 100], dtype=float)
    choices = options[options <= width_km * 0.28]
    scale = float(choices[-1] if len(choices) else options[0])
    lat = south + 0.055 * (north - south)
    lon0 = west + 0.055 * (east - west)
    lon1 = lon0 + scale / (111.32 * math.cos(math.radians(lat)))
    ax.plot([lon0, lon1], [lat, lat], color=INK, linewidth=1.45, zorder=15)
    ax.plot([lon0, lon0], [lat - 0.004 * (north - south), lat + 0.004 * (north - south)],
            color=INK, linewidth=0.8, zorder=15)
    ax.plot([lon1, lon1], [lat - 0.004 * (north - south), lat + 0.004 * (north - south)],
            color=INK, linewidth=0.8, zorder=15)
    ax.text(
        (lon0 + lon1) / 2,
        lat + 0.014 * (north - south),
        f"{int(scale)} km",
        ha="center",
        va="bottom",
        fontsize=4.6,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.3},
        zorder=15,
    )


def plot_distance_timeline(
    ax: plt.Axes,
    observations: pd.DataFrame,
    *,
    time_column: str,
    distance_column: str,
    color: str = RED,
    xlim: tuple[str, str] | None = None,
    colored_phases: tuple[tuple[str, str, str], ...] = (),
) -> None:
    """Plot weekly cohort displacement with an interquartile band."""
    work = observations[[*KEY, time_column, distance_column]].dropna().copy()
    work["week"] = pd.to_datetime(work[time_column]).dt.to_period("W-MON").dt.start_time
    per_identity = work.groupby(KEY + ["week"], as_index=False)[distance_column].median()
    summary = (
        per_identity.groupby("week")[distance_column]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_index()
    )
    if xlim is not None:
        full_index = pd.date_range(
            pd.Timestamp(xlim[0]).to_period("W-MON").start_time,
            pd.Timestamp(xlim[1]).to_period("W-MON").start_time,
            freq="W-TUE",
        )
        # Period(W-MON).start_time is Tuesday. Reindexing prevents long gaps
        # from being silently connected by a line.
        summary = summary.reindex(full_index)
    ax.fill_between(
        summary.index,
        summary.q25.to_numpy(dtype=float),
        summary.q75.to_numpy(dtype=float),
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )
    ax.plot(summary.index, summary["median"], color=color, linewidth=1.15, zorder=3)
    for start, end, phase_color in colored_phases:
        phase = summary.loc[pd.Timestamp(start):pd.Timestamp(end)]
        ax.plot(phase.index, phase["median"], color=phase_color, linewidth=1.45, zorder=4)
    ax.axhline(0, color=INK, linewidth=0.55, zorder=2)
    ax.grid(axis="y", color="#d7dbde", linewidth=0.42, zorder=0)
    ax.set_title("Weekly displacement", loc="left", fontsize=5.9, fontweight="bold", pad=2)
    ax.set_ylabel("km from source", labelpad=1.0)
    ax.set_xlabel("")
    if xlim is not None and (pd.Timestamp(xlim[1]) - pd.Timestamp(xlim[0])).days > 550:
        locator = mdates.MonthLocator(interval=6)
    else:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if xlim is not None:
        ax.set_xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))
    upper = float(np.nanmax(np.r_[summary.q75.to_numpy(), summary["median"].to_numpy()]))
    ax.set_ylim(-max(upper * 0.025, 0.12), max(upper * 1.13, 1.0))
    ax.tick_params(length=2.0, width=0.5, pad=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)


def load_queen_alia() -> tuple[pd.DataFrame, pd.DataFrame]:
    members = pd.read_csv(ATTRIBUTION / "slow_attractor_members.csv")
    members = members[
        np.isclose(members.target_lat, 31.75)
        & np.isclose(members.target_lon, 35.97)
    ].copy()
    positions = pd.read_csv(
        DATA / "remaining_search" / "moving_position_screen.csv",
        parse_dates=["t_first", "t_last"],
    ).merge(members[KEY].drop_duplicates(), on=KEY, how="inner")
    positions["lat"] = positions.plat / 100
    positions["lon"] = positions.plon / 100

    # Reapply the strict corridor used by the attribution screen so the map
    # cannot acquire unrelated historic positions from the same identities.
    target_lat, target_lon = 31.75, 35.97
    kept = []
    for _, group in positions.groupby(KEY, sort=False):
        source_lat = float(group.rlat.iloc[0]) / 100
        source_lon = float(group.rlon.iloc[0]) / 100
        y_scale = 111.32
        x_scale = y_scale * math.cos(math.radians((source_lat + target_lat) / 2))
        dx = (target_lon - source_lon) * x_scale
        dy = (target_lat - source_lat) * y_scale
        baseline = math.hypot(dx, dy)
        px = (group.lon - source_lon) * x_scale
        py = (group.lat - source_lat) * y_scale
        fraction = (px * dx + py * dy) / baseline**2
        cross = np.abs(px * dy - py * dx) / baseline
        mask = fraction.between(0.03, 1.25) & (cross <= max(1.5, 0.06 * baseline))
        kept.append(group[mask])
    return members, pd.concat(kept, ignore_index=True)


def collect_weekly_distance(
    members: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch per-identity weekly median distance using read-only ClickHouse."""
    from ch_remote import ch_df

    members = members[KEY + ["source_lat", "source_lon"]].drop_duplicates(KEY)
    keys = ",".join(
        f"({int(row.mcc)},{int(row.mnc)},{int(row.lac)},{int(row.cid)},'{row.cell_type}')"
        for row in members.itertuples(index=False)
    )
    branches: list[str] = []
    for row in members.itertuples(index=False):
        condition = (
            f"(mcc={int(row.mcc)} AND mnc={int(row.mnc)} "
            f"AND lac={int(row.lac)} AND cid={int(row.cid)} "
            f"AND toString(cell_type)='{row.cell_type}')"
        )
        distance = (
            "greatCircleDistance(lon,lat,"
            f"{float(row.source_lon):.8f},{float(row.source_lat):.8f})/1000"
        )
        branches.extend((condition, distance))
    distance_expression = f"multiIf({','.join(branches)},NULL)"
    query = f"""
    SELECT
      toStartOfWeek(timestamp, 1) AS week,
      mcc, mnc, lac, cid, toString(cell_type) AS cell_type,
      quantileExact(0.5)({distance_expression}) AS distance_km,
      count() AS observations
    FROM cell.geos
    PREWHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({keys})
    WHERE timestamp >= toDateTime('{start}') AND timestamp < toDateTime('{end}')
      AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
      AND NOT (abs(lat) <= 0.01 AND abs(lon) <= 0.01)
    GROUP BY week, mcc, mnc, lac, cid, cell_type
    ORDER BY week, mcc, mnc, lac, cid, cell_type
    """
    return ch_df(
        query,
        settings={"max_threads": 6, "optimize_aggregation_in_order": 0},
    )


def load_queen_timeline(members: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    if refresh:
        frame = collect_weekly_distance(members, start="2024-05-01", end="2026-07-01")
        frame.to_csv(QUEEN_TIMELINE, index=False)
    if not QUEEN_TIMELINE.exists():
        raise FileNotFoundError(f"Missing {QUEEN_TIMELINE}; run with --refresh-timelines")
    return pd.read_csv(QUEEN_TIMELINE, parse_dates=["week"])


def load_moscow_timeline(data: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    members = data[KEY + ["source_lat", "source_lon"]].drop_duplicates(KEY)
    if refresh:
        frame = collect_weekly_distance(members, start="2024-05-01", end="2026-07-01")
        frame.to_csv(MOSCOW_TIMELINE, index=False)
    if not MOSCOW_TIMELINE.exists():
        raise FileNotFoundError(f"Missing {MOSCOW_TIMELINE}; run with --refresh-timelines")
    return pd.read_csv(MOSCOW_TIMELINE, parse_dates=["week"])


def draw_queen_alia(ax: plt.Axes, time_ax: plt.Axes, refresh: bool) -> None:
    members, positions = load_queen_alia()
    lons = np.r_[members.source_lon, positions.lon, QUEEN_ALIA[0]]
    lats = np.r_[members.source_lat, positions.lat, QUEEN_ALIA[1]]
    bbox = square_bbox(lons, lats, padding=1.18)
    setup_panel(ax, bbox, "Amman → Queen Alia")

    for _, group in positions.sort_values("t_first").groupby(KEY, sort=False):
        source = members.merge(group[KEY].iloc[[0]], on=KEY).iloc[0]
        xs = np.r_[source.source_lon, group.lon.to_numpy()]
        ys = np.r_[source.source_lat, group.lat.to_numpy()]
        ax.plot(xs, ys, color=MUTED, linewidth=0.42, alpha=0.22, zorder=3)
    ax.scatter(
        positions.lon, positions.lat, s=2.2, color=RED, alpha=0.12,
        edgecolor="none", zorder=4,
    )
    ax.scatter(
        members.source_lon, members.source_lat, s=7, color=BLUE, alpha=0.46,
        edgecolor="white", linewidth=0.25, zorder=6,
    )
    ax.scatter(*QUEEN_ALIA, s=48, marker="*", color=RED, edgecolor="white", linewidth=0.65, zorder=11)
    label_point(ax, QUEEN_ALIA, "Queen Alia", RED, offset=(-4, 5), align="right")
    ax.text(
        0.98, 0.025, "134 identities · 3 PLMNs",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8, color=INK,
        bbox={"facecolor": "white", "edgecolor": "#858b8e", "linewidth": 0.7,
              "alpha": 0.93, "pad": 0.9}, zorder=14,
    )
    add_scale_bar(ax, bbox)
    distances = load_queen_timeline(members, refresh)
    plot_distance_timeline(
        time_ax,
        distances,
        time_column="week",
        distance_column="distance_km",
        xlim=("2024-05-01", "2026-07-01"),
    )


def draw_moscow(ax: plt.Axes, time_ax: plt.Axes, refresh: bool) -> None:
    data = pd.read_csv(DATA / "news_validation" / "moscow_cps_trajectory.csv", parse_dates=["day"])
    data = data[(data.along_fraction >= -0.05) & (data.along_fraction <= 1.08)
                & (data.cross_track_km.abs() <= 0.65)].copy()
    sources = data[KEY + ["source_lat", "source_lon"]].drop_duplicates(KEY)
    lons = np.r_[data.returned_lon, sources.source_lon, SHEREMETYEVO[0]]
    lats = np.r_[data.returned_lat, sources.source_lat, SHEREMETYEVO[1]]
    bbox = square_bbox(lons, lats, padding=1.15)
    setup_panel(ax, bbox, "Moscow → Sheremetyevo")

    data["week"] = data.day.dt.to_period("W-MON").dt.start_time
    weekly = data.sort_values("day").groupby(KEY + ["week"], as_index=False).tail(1)
    for _, group in weekly.sort_values("week").groupby(KEY, sort=False):
        ax.plot(
            group.returned_lon, group.returned_lat,
            color=MUTED, linewidth=0.38, alpha=0.20, zorder=3,
        )
    latest = data.sort_values("day").groupby(KEY).tail(1)
    ax.scatter(
        sources.source_lon, sources.source_lat, s=6.5, color=BLUE, alpha=0.45,
        edgecolor="white", linewidth=0.23, zorder=6,
    )
    ax.scatter(
        latest.returned_lon, latest.returned_lat, s=5.5, color=RED, alpha=0.36,
        edgecolor="white", linewidth=0.20, zorder=7,
    )
    ax.scatter(*SHEREMETYEVO, s=48, marker="*", color=RED, edgecolor="white", linewidth=0.65, zorder=11)
    label_point(ax, SHEREMETYEVO, "Sheremetyevo", RED, offset=(-4, -12), align="right")
    ax.text(
        0.98, 0.025, "274 identities · 5 PLMNs",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.8, color=INK,
        bbox={"facecolor": "white", "edgecolor": "#858b8e", "linewidth": 0.7,
              "alpha": 0.93, "pad": 0.9}, zorder=14,
    )
    add_scale_bar(ax, bbox)
    distances = load_moscow_timeline(data, refresh)
    plot_distance_timeline(
        time_ax,
        distances,
        time_column="week",
        distance_column="distance_km",
        xlim=("2024-05-01", "2026-07-01"),
    )


def draw_vladimir(ax: plt.Axes, time_ax: plt.Axes) -> None:
    raw = pd.read_csv(ATTRIBUTION / "alternating_attractor_raw.csv.gz", parse_dates=["timestamp"])
    targets = pd.read_csv(ATTRIBUTION / "alternating_attractor_campaign.csv")
    west_target = tuple(targets.loc[targets.phase.eq("west"), ["target_lon", "target_lat"]].iloc[0])
    east_target = tuple(targets.loc[targets.phase.eq("east"), ["target_lon", "target_lat"]].iloc[0])
    homes = (
        raw[raw.timestamp < pd.Timestamp("2025-02-12")]
        .groupby(KEY, as_index=False)[["lat", "lon"]]
        .median()
    )
    west = raw[raw.timestamp.between("2025-02-12", "2025-02-25", inclusive="left")].copy()
    east = raw[raw.timestamp.between("2025-03-14", "2025-06-21", inclusive="left")].copy()
    lons = np.r_[homes.lon, west.lon, east.lon, west_target[0], east_target[0]]
    lats = np.r_[homes.lat, west.lat, east.lat, west_target[1], east_target[1]]
    bbox = square_bbox(lons, lats, padding=1.12, minimum_km=70)
    setup_panel(ax, bbox, "Vladimir · two targets")

    for phase, color in ((west, PURPLE), (east, RED)):
        daily = phase.assign(day=phase.timestamp.dt.floor("D"))
        daily = daily.groupby(KEY + ["day"], as_index=False)[["lon", "lat"]].median()
        for identity, group in daily.sort_values("day").groupby(KEY, sort=False):
            home = homes.set_index(KEY).loc[identity]
            xs = np.r_[home.lon, group.lon.to_numpy()]
            ys = np.r_[home.lat, group.lat.to_numpy()]
            ax.plot(xs, ys, color=MUTED, linewidth=0.48, alpha=0.25, zorder=4)
    ax.scatter(
        homes.lon, homes.lat, s=8, color=BLUE, alpha=0.62,
        edgecolor="white", linewidth=0.3, zorder=7,
    )
    ax.scatter(*west_target, s=45, marker="*", color=PURPLE, edgecolor="white", linewidth=0.6, zorder=11)
    ax.scatter(*east_target, s=45, marker="*", color=RED, edgecolor="white", linewidth=0.6, zorder=11)
    label_point(ax, west_target, "February target", PURPLE, offset=(4, 3))
    label_point(ax, east_target, "March target", RED, offset=(-4, 3), align="right")
    ax.text(
        0.98, 0.025, "27 identities · 5 PLMNs",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.7, color=INK,
        bbox={"facecolor": "white", "edgecolor": "#858b8e", "linewidth": 0.7,
              "alpha": 0.93, "pad": 0.9}, zorder=14,
    )
    add_scale_bar(ax, bbox)
    distance = raw.merge(
        homes.rename(columns={"lat": "source_lat", "lon": "source_lon"}),
        on=KEY,
        how="left",
    )
    distance["distance_km"] = haversine_km(
        distance.lon, distance.lat, distance.source_lon, distance.source_lat
    )
    plot_distance_timeline(
        time_ax,
        distance,
        time_column="timestamp",
        distance_column="distance_km",
        color=INK,
        xlim=("2023-11-01", "2026-07-01"),
        colored_phases=(
            ("2025-02-12", "2025-02-25", PURPLE),
            ("2025-03-14", "2025-08-31", RED),
        ),
    )


def load_islamabad() -> pd.DataFrame:
    data = pd.read_csv(CONTROLLED_TRAJECTORIES, parse_dates=["timestamp"])
    data = data[data.campaign.eq("pakistan")].copy()
    if data.empty:
        raise RuntimeError("No Islamabad–Changsha trajectory data found")
    return data


def draw_islamabad(ax: plt.Axes, time_ax: plt.Axes) -> None:
    data = load_islamabad()
    identities = data[KEY].drop_duplicates()
    sources = data[KEY + ["reference_lat", "reference_lon"]].drop_duplicates(KEY)
    source = (float(sources.reference_lon.median()), float(sources.reference_lat.median()))
    changsha = (113.22, 28.19)
    destinations = (
        data[data.displacement_km.gt(2000)]
        .groupby(KEY, as_index=False)[["lat", "lon"]]
        .median()
    )
    routes = sources.merge(destinations, on=KEY, how="inner")
    bbox = square_bbox(
        np.r_[sources.reference_lon.to_numpy(), changsha[0]],
        np.r_[sources.reference_lat.to_numpy(), changsha[1]],
        padding=1.08,
        minimum_km=4000,
    )
    setup_panel(ax, bbox, "Islamabad → Changsha")
    for row in routes.itertuples(index=False):
        ax.plot(
            [row.reference_lon, row.lon], [row.reference_lat, row.lat],
            color=MUTED, linewidth=0.16, alpha=0.13, zorder=4,
        )
    ax.scatter(
        sources.reference_lon, sources.reference_lat, s=5.5, color=BLUE,
        alpha=0.48, edgecolor="white", linewidth=0.2, zorder=6,
    )
    ax.scatter(
        destinations.lon, destinations.lat, s=3.8, color=RED,
        alpha=0.32, edgecolor="white", linewidth=0.16, zorder=7,
    )
    ax.scatter(*changsha, s=48, marker="*", color=RED, edgecolor="white", linewidth=0.65, zorder=11)
    label_point(ax, source, "Islamabad", BLUE, offset=(4, 4))
    label_point(ax, changsha, "Changsha", RED, offset=(-4, 4), align="right")
    midpoint = ((source[0] + changsha[0]) / 2, (source[1] + changsha[1]) / 2)
    ax.text(
        midpoint[0], midpoint[1] + 2.2, "3,845 km",
        ha="center", va="bottom", fontsize=4.7, color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
        zorder=12,
    )
    ax.text(
        0.98, 0.025, f"{len(identities)} identities · 4 PLMNs",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.7, color=INK,
        bbox={"facecolor": "white", "edgecolor": "#858b8e", "linewidth": 0.7,
              "alpha": 0.93, "pad": 0.9}, zorder=14,
    )
    add_scale_bar(ax, bbox)

    route_km = 2000
    activity = data.assign(
        endpoint=np.where(data.displacement_km.gt(route_km), "Changsha", "Islamabad"),
        week=data.timestamp.dt.to_period("W-MON").dt.start_time,
    )
    weekly = (
        activity[["week", "endpoint", *KEY]].drop_duplicates()
        .groupby(["week", "endpoint"]).size().unstack(fill_value=0)
        .reindex(columns=["Islamabad", "Changsha"], fill_value=0)
    )
    full_index = pd.date_range("2024-04-30", "2026-06-30", freq="W-TUE")
    weekly = weekly.reindex(full_index)
    for endpoint, color in (("Islamabad", BLUE), ("Changsha", RED)):
        time_ax.plot(
            weekly.index, weekly[endpoint], color=color, linewidth=1.05,
            label=endpoint, zorder=3,
        )
    time_ax.grid(axis="y", color="#d7dbde", linewidth=0.42, zorder=0)
    time_ax.set_title("Weekly endpoint activity", loc="left", fontsize=5.9, fontweight="bold", pad=2)
    time_ax.set_ylabel("identities", labelpad=1.0)
    time_ax.set_xlim(pd.Timestamp("2024-05-01"), pd.Timestamp("2026-07-01"))
    time_ax.set_ylim(-2, 75)
    time_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    time_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(time_ax.xaxis.get_major_locator()))
    time_ax.tick_params(length=2.0, width=0.5, pad=1.2)
    time_ax.spines["top"].set_visible(False)
    time_ax.spines["right"].set_visible(False)
    time_ax.spines["left"].set_linewidth(0.55)
    time_ax.spines["bottom"].set_linewidth(0.55)
    time_ax.legend(
        loc="upper right", ncol=1, frameon=False, fontsize=4.4,
        handlelength=1.2, handletextpad=0.35,
    )


def collect_lyon_voronezh() -> pd.DataFrame:
    """Fetch the fixed cohort observed at both airport attractors."""
    from ch_remote import ch_df

    cohort = ch_df(
        """
        WITH
          lyon AS (
            SELECT DISTINCT mcc, mnc, lac, cid, toString(cell_type) AS cell_type
            FROM cell.cellpos
            WHERE mcc = 250 AND mnc IN (54, 96, 98) AND cell_type = 'gsm'
              AND plat BETWEEN 4569 AND 4576 AND plon BETWEEN 504 AND 512
          ),
          voronezh AS (
            SELECT DISTINCT mcc, mnc, lac, cid, toString(cell_type) AS cell_type
            FROM cell.cellpos
            WHERE mcc = 250 AND mnc IN (54, 96, 98) AND cell_type = 'gsm'
              AND plat BETWEEN 5179 AND 5183 AND plon BETWEEN 3920 AND 3925
          )
        SELECT mcc, mnc, lac, cid, cell_type
        FROM lyon INNER JOIN voronezh USING (mcc, mnc, lac, cid, cell_type)
        ORDER BY mcc, mnc, lac, cid, cell_type
        """
    )
    if len(cohort) != 135:
        raise RuntimeError(f"Lyon--Voronezh cohort changed: expected 135 identities, got {len(cohort)}")
    keys = ",".join(
        f"({int(row.mcc)},{int(row.mnc)},{int(row.lac)},{int(row.cid)},'{row.cell_type}')"
        for row in cohort.itertuples(index=False)
    )
    return ch_df(
        f"""
        SELECT
          timestamp, mcc, mnc, lac, cid, toString(cell_type) AS cell_type,
          lat, lon,
          multiIf(
            lat BETWEEN 45.69 AND 45.76 AND lon BETWEEN 5.04 AND 5.12, 'Lyon',
            lat BETWEEN 51.79 AND 51.83 AND lon BETWEEN 39.20 AND 39.25, 'Voronezh',
            'Other'
          ) AS endpoint
        FROM cell.geos
        PREWHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({keys})
        WHERE
          (lat BETWEEN 45.69 AND 45.76 AND lon BETWEEN 5.04 AND 5.12)
          OR (lat BETWEEN 51.79 AND 51.83 AND lon BETWEEN 39.20 AND 39.25)
        ORDER BY timestamp, mcc, mnc, lac, cid, cell_type
        """,
        settings={"max_threads": 6, "optimize_aggregation_in_order": 0},
    )


def load_lyon_voronezh(refresh: bool) -> pd.DataFrame:
    if refresh:
        frame = collect_lyon_voronezh()
        frame.to_csv(LYON_VORONEZH_TRAJECTORY, index=False)
    if not LYON_VORONEZH_TRAJECTORY.exists():
        raise FileNotFoundError(
            f"Missing {LYON_VORONEZH_TRAJECTORY}; run with --refresh-timelines"
        )
    frame = pd.read_csv(LYON_VORONEZH_TRAJECTORY, parse_dates=["timestamp"])
    if frame[KEY].drop_duplicates().shape[0] != 135:
        raise RuntimeError("Stored Lyon--Voronezh trajectory does not contain 135 identities")
    return frame


def draw_lyon_voronezh(ax: plt.Axes, time_ax: plt.Axes, refresh: bool) -> None:
    data = load_lyon_voronezh(refresh)
    endpoints = (
        data.groupby(KEY + ["endpoint"], as_index=False)[["lat", "lon"]]
        .median()
    )
    lyon = endpoints[endpoints.endpoint.eq("Lyon")].copy()
    voronezh = endpoints[endpoints.endpoint.eq("Voronezh")].copy()
    routes = lyon.merge(voronezh, on=KEY, suffixes=("_lyon", "_voronezh"))
    lyon_airport = (5.07955, 45.72035)
    voronezh_airport = (39.22502, 51.81275)
    bbox = square_bbox(
        np.r_[routes.lon_lyon, routes.lon_voronezh],
        np.r_[routes.lat_lyon, routes.lat_voronezh],
        padding=1.12,
        minimum_km=2800,
    )
    setup_panel(ax, bbox, "Lyon → Voronezh")
    for row in routes.itertuples(index=False):
        ax.plot(
            [row.lon_lyon, row.lon_voronezh],
            [row.lat_lyon, row.lat_voronezh],
            color=MUTED, linewidth=0.58, alpha=0.42, zorder=4,
        )
    ax.scatter(
        routes.lon_lyon, routes.lat_lyon, s=6.0, color=BLUE,
        alpha=0.50, edgecolor="white", linewidth=0.22, zorder=6,
    )
    ax.scatter(
        routes.lon_voronezh, routes.lat_voronezh, s=5.0, color=RED,
        alpha=0.40, edgecolor="white", linewidth=0.20, zorder=7,
    )
    ax.scatter(*lyon_airport, s=42, marker="*", color=BLUE,
               edgecolor="white", linewidth=0.6, zorder=11)
    ax.scatter(*voronezh_airport, s=46, marker="*", color=RED,
               edgecolor="white", linewidth=0.65, zorder=11)
    label_point(ax, lyon_airport, "Lyon airport", BLUE, offset=(4, 4))
    label_point(ax, voronezh_airport, "Voronezh airport", RED,
                offset=(-4, 4), align="right")
    midpoint = (
        (lyon_airport[0] + voronezh_airport[0]) / 2,
        (lyon_airport[1] + voronezh_airport[1]) / 2,
    )
    ax.text(
        midpoint[0], midpoint[1] + 1.1, "2,567 km",
        ha="center", va="bottom", fontsize=4.7, color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
        zorder=12,
    )
    ax.text(
        0.98, 0.025, "135 identities · 3 PLMNs",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.7, color=INK,
        bbox={"facecolor": "white", "edgecolor": "#858b8e", "linewidth": 0.7,
              "alpha": 0.93, "pad": 0.9}, zorder=14,
    )
    add_scale_bar(ax, bbox)

    activity = data.assign(week=data.timestamp.dt.to_period("W-MON").dt.start_time)
    weekly = (
        activity[["week", "endpoint", *KEY]].drop_duplicates()
        .groupby(["week", "endpoint"]).size().unstack(fill_value=0)
        .reindex(columns=["Lyon", "Voronezh"], fill_value=0)
    )
    full_index = pd.date_range("2025-09-30", "2026-02-24", freq="W-TUE")
    weekly = weekly.reindex(full_index)
    for endpoint, color in (("Lyon", BLUE), ("Voronezh", RED)):
        time_ax.plot(
            weekly.index, weekly[endpoint], color=color, linewidth=1.05,
            label=endpoint, zorder=3,
        )
    time_ax.grid(axis="y", color="#d7dbde", linewidth=0.42, zorder=0)
    time_ax.set_title("Weekly endpoint activity", loc="left", fontsize=5.9,
                      fontweight="bold", pad=2)
    time_ax.set_ylabel("identities", labelpad=1.0)
    time_ax.set_xlim(pd.Timestamp("2025-10-01"), pd.Timestamp("2026-02-28"))
    time_ax.set_ylim(-2, 105)
    locator = mdates.MonthLocator(interval=1)
    time_ax.xaxis.set_major_locator(locator)
    time_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    time_ax.tick_params(length=2.0, width=0.5, pad=1.2)
    time_ax.spines["top"].set_visible(False)
    time_ax.spines["right"].set_visible(False)
    time_ax.spines["left"].set_linewidth(0.55)
    time_ax.spines["bottom"].set_linewidth(0.55)
    time_ax.legend(
        loc="upper right", ncol=1, frameon=False, fontsize=4.4,
        handlelength=1.2, handletextpad=0.35,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-timelines",
        action="store_true",
        help="refresh cohort timelines through the read-only database helper",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure()
    fig = plt.figure(figsize=(7.15, 2.88))
    grid = fig.add_gridspec(
        2, 5,
        left=0.046, right=0.994, top=0.84, bottom=0.13,
        height_ratios=[1.0, 0.40], hspace=0.25, wspace=0.20,
    )
    map_axes = [fig.add_subplot(grid[0, index]) for index in range(5)]
    time_axes = [fig.add_subplot(grid[1, index]) for index in range(5)]
    draw_queen_alia(map_axes[0], time_axes[0], args.refresh_timelines)
    draw_moscow(map_axes[1], time_axes[1], args.refresh_timelines)
    draw_vladimir(map_axes[2], time_axes[2])
    draw_islamabad(map_axes[3], time_axes[3])
    draw_lyon_voronezh(map_axes[4], time_axes[4], args.refresh_timelines)
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=BLUE, markeredgecolor="white", label="Earlier/original cell estimate"),
        Line2D([0], [0], color=MUTED, linewidth=1.1, label="Returned-position path"),
        Line2D([0], [0], marker="*", linestyle="none", markersize=7,
               markerfacecolor=RED, markeredgecolor="white", label="False-location target"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.988), ncol=3,
        frameon=False, fontsize=5.4, handletextpad=0.45, columnspacing=1.3,
    )
    fig.text(
        0.018, 0.025, "Basemap © OpenStreetMap contributors, © CARTO",
        ha="left", va="bottom", fontsize=4.25, color=MUTED,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=260)
    plt.close(fig)


if __name__ == "__main__":
    main()
