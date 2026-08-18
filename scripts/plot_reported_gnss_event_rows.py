#!/usr/bin/env python3
"""Plot reported-event rows in the same map/inset/activity format as Iran."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from ch_remote import ch_df
from mine_reported_gnss_events import fetch_destination_identities
from plot_helpers import add_osm_basemap
from spoofing_category_overview import load_world, setup_map


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/spoofing/news_validation"
FIGS = ROOT / "paper/figs"
WORLD = ROOT / "data/reference/ne_10m_admin_0_map_units.geojson"
WEEKLY_DATA = DATA / "reported_event_weekly_activity.csv"
POINT_DATA = DATA / "reported_event_endpoint_points.csv"
CONTEXT_WEEKS_BEFORE = 12
CONTEXT_WEEKS_AFTER = 52

BLUE = "#286f9b"
RED = "#b43c49"
INK = "#26292c"
MUTED = "#71777b"
GRID = "#d9dde0"


@dataclass(frozen=True)
class EventPlot:
    event_id: str
    source_name: str
    destination_name: str
    period: str
    markers: tuple[tuple[str, str], ...] = ()
    map_title: str | None = None
    activity_title: str = "Weekly activity of the same identities"
    source_legend: str | None = None
    destination_legend: str | None = None
    display_end: str | None = None
    summary_note: str | None = None
    continuation_note: str | None = None


STRONG = (
    EventPlot(
        "queen_alia_sep2024", "Levant", "Queen Alia", "September 2024",
        (("2024-08-24", "Reported switch\ntoward Amman"),),
    ),
    EventPlot(
        "moscow_nov2024", "Moscow", "Sheremetyevo", "November 2024",
        (("2024-11-20", "Moscow-region\ndrone alert"),),
        map_title="November 2024\ntransient update",
        activity_title="Apple-returned coordinates: short-lived November episode",
        source_legend="Original Moscow estimate",
        destination_legend="Sheremetyevo-directed estimate",
        display_end="2025-03-31",
        summary_note="Most affected identities return within days",
    ),
    EventPlot(
        "moscow_may2025", "Moscow", "Sheremetyevo", "May 2025",
        (("2025-05-07", "GNSS escalation\nreported"),
         ("2025-05-09", "Victory Day")),
        map_title="May 2025 CPS\nlandmark migration",
        activity_title="Apple-returned landmark state (not RF-transmitter duration)",
        source_legend="Original Moscow estimate",
        destination_legend="Sheremetyevo-directed estimate",
        summary_note="Identities migrate one-way during May",
        continuation_note="Updated CPS state persists\nthrough the plotted window",
    ),
)

ENDPOINT = (
    EventPlot(
        "islamabad_changsha_2025", "Islamabad", "Changsha", "January–February 2025",
        (("2024-12-30", "Pakistan aviation\nadvisory"),
         ("2025-01-15", "Changsha jammer\nidentified")),
    ),
    EventPlot(
        "iran_war_jun2025", "Southern Iran", "Tehran", "June 2025",
        (("2025-06-13", "War begins"),
         ("2025-06-20", "Navigation reports\npeak")),
    ),
)


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.2,
        "axes.titlesize": 7.0,
        "axes.labelsize": 5.9,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(DATA / "known_gnss_events.csv", parse_dates=["screen_start", "screen_end"])
    candidates = pd.read_csv(DATA / "event_candidate_bins.csv", parse_dates=["onset_day"])
    weekly = pd.read_csv(
        WEEKLY_DATA, parse_dates=["week", "window_start", "window_end"],
    )
    points = pd.read_csv(POINT_DATA)
    return events, candidates, weekly, points


def event_row(events: pd.DataFrame, event_id: str) -> pd.Series:
    return events.loc[events.event_id.eq(event_id)].iloc[0]


def significant_sources(candidates: pd.DataFrame, event_id: str) -> pd.DataFrame:
    return candidates[candidates.event_id.eq(event_id) & candidates.significant.eq(True)].copy()


def weighted_center(sources: pd.DataFrame) -> tuple[float, float]:
    weights = sources.n_onsets.to_numpy(dtype=float)
    return (
        float(np.average(sources.source_lon, weights=weights)),
        float(np.average(sources.source_lat, weights=weights)),
    )


def weighted_point_center(points: pd.DataFrame) -> tuple[float, float]:
    weights = points.observations.to_numpy(dtype=float)
    return (
        float(np.average(points.lon, weights=weights)),
        float(np.average(points.lat, weights=weights)),
    )


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = np.radians([lat1, lat2])
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def refresh_weekly(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read-only refresh of weekly source/destination participation."""
    frames: list[pd.DataFrame] = []
    point_frames: list[pd.DataFrame] = []
    for spec in (*STRONG, *ENDPOINT):
        event = event_row(events, spec.event_id)
        sources = significant_sources(candidates, spec.event_id)
        source_lon, source_lat = weighted_center(sources)
        destination_lon = float(event.known_dest_lon)
        destination_lat = float(event.known_dest_lat)
        source_radius = max(
            25.0,
            max(
                haversine_km(source_lon, source_lat, float(row.source_lon), float(row.source_lat))
                for row in sources.itertuples(index=False)
            ) + 12.0,
        )
        destination_radius = float(event.known_dest_radius_km)
        identity_event = event.copy()
        identity_event["screen_start"] = pd.Timestamp(event.screen_start).strftime("%Y-%m-%d")
        identity_event["screen_end"] = pd.Timestamp(event.screen_end).strftime("%Y-%m-%d")
        identities = fetch_destination_identities(identity_event)
        keys = ",".join(
            f"({int(row.mcc)},{int(row.mnc)},{int(row.lac)},{int(row.cid)},'{row.cell_type}')"
            for row in identities.itertuples(index=False)
        )
        # Preserve a long post-event window even when it contains no rows.  A
        # raw query naturally stops at the cohort's last observation, which
        # otherwise makes a completed episode look as though the plot merely
        # ended while it was still active.
        start = (
            pd.Timestamp(event.screen_start) - pd.Timedelta(weeks=CONTEXT_WEEKS_BEFORE)
        ).strftime("%Y-%m-%d")
        end = (
            pd.Timestamp(event.screen_end) + pd.Timedelta(weeks=CONTEXT_WEEKS_AFTER, days=1)
        ).strftime("%Y-%m-%d")
        event_start = pd.Timestamp(event.screen_start).strftime("%Y-%m-%d")
        map_destination_end = (
            pd.Timestamp(event.screen_end) + pd.Timedelta(days=15)
        ).strftime("%Y-%m-%d")
        query = f"""
        WITH
          greatCircleDistance(lon,lat,{source_lon:.8f},{source_lat:.8f})/1000 AS source_km,
          greatCircleDistance(lon,lat,{destination_lon:.8f},{destination_lat:.8f})/1000 AS destination_km,
          if(destination_km < source_km, 'destination', 'source') AS endpoint
        SELECT
          toStartOfWeek(timestamp, 1) AS week,
          endpoint,
          uniqExact((mcc,mnc,lac,cid,cell_type)) AS identities,
          count() AS observations
        FROM cell.geos
        PREWHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({keys})
        WHERE timestamp >= toDateTime('{start}')
          AND timestamp < toDateTime('{end}')
          AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
          AND NOT (abs(lat) <= 0.01 AND abs(lon) <= 0.01)
          AND ((endpoint = 'source' AND source_km <= {source_radius:.3f})
            OR (endpoint = 'destination' AND destination_km <= {destination_radius:.3f}))
        GROUP BY week, endpoint
        ORDER BY week, endpoint
        """
        frame = ch_df(query, settings={"max_threads": 6, "optimize_aggregation_in_order": 0})
        if frame.empty:
            raise RuntimeError(f"No weekly activity found for {spec.event_id}")
        frame.insert(0, "event_id", spec.event_id)
        frame["cohort_size"] = len(identities)
        frame["source_lon"] = source_lon
        frame["source_lat"] = source_lat
        frame["destination_lon"] = destination_lon
        frame["destination_lat"] = destination_lat
        frame["window_start"] = start
        frame["window_end"] = end
        frames.append(frame)
        point_query = f"""
        WITH
          greatCircleDistance(lon,lat,{source_lon:.8f},{source_lat:.8f})/1000 AS source_km,
          greatCircleDistance(lon,lat,{destination_lon:.8f},{destination_lat:.8f})/1000 AS destination_km,
          if(destination_km < source_km, 'destination', 'source') AS endpoint
        SELECT
          endpoint, mcc, mnc, lac, cid, toString(cell_type) AS cell_type,
          quantileExact(0.5)(lon) AS point_lon,
          quantileExact(0.5)(lat) AS point_lat,
          count() AS observations
        FROM cell.geos
        PREWHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({keys})
        WHERE timestamp >= toDateTime('{start}')
          AND timestamp < toDateTime('{end}')
          AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
          AND NOT (abs(lat) <= 0.01 AND abs(lon) <= 0.01)
          -- Map the original estimate before the event and the newly returned
          -- estimate during/just after it.  Aggregating the entire context
          -- window would make the November map silently depict the distinct
          -- May 2025 migration instead.
          AND ((endpoint = 'source'
                AND source_km <= {source_radius:.3f}
                AND timestamp < toDateTime('{event_start}'))
            OR (endpoint = 'destination'
                AND destination_km <= {destination_radius:.3f}
                AND timestamp >= toDateTime('{event_start}')
                AND timestamp < toDateTime('{map_destination_end}')))
        GROUP BY endpoint, mcc, mnc, lac, cid, cell_type
        ORDER BY endpoint, mcc, mnc, lac, cid, cell_type
        """
        point_frame = ch_df(
            point_query,
            settings={"max_threads": 6, "optimize_aggregation_in_order": 0},
        )
        if point_frame.empty:
            raise RuntimeError(f"No endpoint points found for {spec.event_id}")
        point_frame = point_frame.rename(
            columns={"point_lon": "lon", "point_lat": "lat"},
        )
        point_frame.insert(0, "event_id", spec.event_id)
        point_frames.append(point_frame)
        print(f"{spec.event_id}: {len(identities)} identities, {len(frame)} weekly rows", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(WEEKLY_DATA, index=False)
    point_result = pd.concat(point_frames, ignore_index=True)
    point_result.to_csv(POINT_DATA, index=False)
    return result, point_result


def square_map_bbox(
    lons: np.ndarray,
    lats: np.ndarray,
    minimum_lon_span: float = 0.55,
) -> tuple[float, float, float, float]:
    """Return a geographically scaled extent that fills a square map panel."""
    lon_mid = float((lons.min() + lons.max()) / 2)
    lat_mid = float((lats.min() + lats.max()) / 2)
    cos_lat = max(float(np.cos(np.deg2rad(lat_mid))), 0.25)
    raw_lon_span = max(float(lons.max() - lons.min()) * 1.28, minimum_lon_span)
    raw_lat_span = max(float(lats.max() - lats.min()) * 1.34, minimum_lon_span * cos_lat)
    lon_span = max(raw_lon_span, raw_lat_span / cos_lat)
    lat_span = lon_span * cos_lat
    return (
        lon_mid - lon_span / 2,
        lon_mid + lon_span / 2,
        lat_mid - lat_span / 2,
        lat_mid + lat_span / 2,
    )


def draw_main_map(
    ax: plt.Axes,
    rings,
    spec: EventPlot,
    sources: pd.DataFrame,
    points: pd.DataFrame,
) -> None:
    source_points = points[points.endpoint.eq("source")]
    destination_points = points[points.endpoint.eq("destination")]
    source = weighted_point_center(source_points)
    destination = weighted_point_center(destination_points)
    lons = np.r_[source_points.lon.to_numpy(), destination_points.lon.to_numpy()]
    lats = np.r_[source_points.lat.to_numpy(), destination_points.lat.to_numpy()]
    bbox = square_map_bbox(lons, lats)
    setup_map(ax, rings, bbox, equal=False)
    ax.add_patch(FancyArrowPatch(
        source, destination, arrowstyle="-|>", mutation_scale=8.0,
        connectionstyle="arc3,rad=-0.08", color=RED,
        linewidth=1.15, alpha=0.88, zorder=4,
    ))
    ax.scatter(
        source_points.lon, source_points.lat,
        s=6.5, color=BLUE,
        edgecolor="white", linewidth=0.30, alpha=0.60, zorder=5,
    )
    ax.scatter(
        [source[0]], [source[1]], s=28, color=BLUE,
        edgecolor="white", linewidth=0.55, zorder=6,
    )
    ax.scatter(
        destination_points.lon, destination_points.lat,
        s=6.5, color=RED, edgecolor="white", linewidth=0.30,
        alpha=0.55, zorder=5,
    )
    ax.scatter(
        [destination[0]], [destination[1]], s=34, marker="D", color=RED,
        edgecolor="white", linewidth=0.55, zorder=6,
    )
    ax.annotate(
        spec.source_name, source, xytext=(3, 4), textcoords="offset points",
        fontsize=5.3, fontweight="bold", color=BLUE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
        zorder=8,
    )
    ax.annotate(
        spec.destination_name, destination, xytext=(-3, -8), textcoords="offset points",
        ha="right", fontsize=5.3, fontweight="bold", color=RED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
        zorder=8,
    )
    distance = haversine_km(source[0], source[1], destination[0], destination[1])
    ax.set_title(
        spec.map_title or f"{spec.source_name} to {spec.destination_name}\n{spec.period}",
        loc="left", fontweight="bold", pad=2,
    )
    ax.text(
        0.02, 0.02,
        f"{int(sources.n_onsets.sum())} detector onsets · {distance:,.0f} km",
        transform=ax.transAxes, fontsize=4.8, color=INK, va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#b6babd", "linewidth": 0.45,
              "alpha": 0.92, "pad": 1.0}, zorder=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_box_aspect(1)
    ax.set_aspect("auto")
    ax.set_anchor("W")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color(INK)


def inset_bbox(lons: np.ndarray, lats: np.ndarray) -> tuple[float, float, float, float]:
    lon_mid = float(np.median(lons))
    lat_mid = float(np.median(lats))
    lon_span = max(float(lons.max() - lons.min()) * 1.55, 0.075)
    lat_span = max(float(lats.max() - lats.min()) * 1.55, 0.060)
    return (
        lon_mid - lon_span / 2,
        lon_mid + lon_span / 2,
        lat_mid - lat_span / 2,
        lat_mid + lat_span / 2,
    )


def detail_zoom(bbox: tuple[float, float, float, float]) -> int:
    span = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    if span <= 0.09:
        return 13
    if span <= 0.20:
        return 12
    if span <= 0.45:
        return 11
    return 10


def draw_detail(
    ax: plt.Axes,
    lons: np.ndarray,
    lats: np.ndarray,
    title: str,
    color: str,
    sizes: np.ndarray | float,
) -> None:
    bbox = inset_bbox(lons, lats)
    ax.set_facecolor("#e8edf0")
    add_osm_basemap(
        ax, bbox, zoom=detail_zoom(bbox), alpha=0.88,
        grayscale=True, grayscale_brightness=0.98,
        grayscale_contrast=1.25, source="carto_voyager",
    )
    ax.scatter(lons, lats, s=sizes, color=color, edgecolor="white", linewidth=0.4, zorder=6)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_box_aspect(1)
    ax.set_aspect("auto")
    ax.set_title(title, loc="left", fontsize=5.8, fontweight="bold", color=color, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(0.55)


def draw_activity(ax: plt.Axes, spec: EventPlot, activity: pd.DataFrame) -> None:
    sub = activity[activity.event_id.eq(spec.event_id)].copy()
    weekly = sub.pivot(index="week", columns="endpoint", values="identities").fillna(0)
    window_start = pd.Timestamp(sub.window_start.iloc[0])
    window_end = pd.Timestamp(sub.window_end.iloc[0])
    if spec.display_end is not None:
        window_end = min(window_end, pd.Timestamp(spec.display_end))
    monday_start = window_start - pd.Timedelta(days=window_start.weekday())
    monday_end = window_end - pd.Timedelta(days=window_end.weekday())
    all_weeks = pd.date_range(monday_start, monday_end, freq="W-MON")
    weekly = weekly.reindex(all_weeks, fill_value=0)
    source = weekly.get("source", pd.Series(0, index=weekly.index))
    destination = weekly.get("destination", pd.Series(0, index=weekly.index))
    ax.bar(weekly.index, source, width=5.6, color=BLUE, edgecolor="white", linewidth=0.35,
           label=spec.source_legend or spec.source_name, zorder=3)
    ax.bar(weekly.index, -destination, width=5.6, color=RED, edgecolor="white", linewidth=0.35,
           label=spec.destination_legend or spec.destination_name, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.75, zorder=4)
    peak = max(float(source.max()), float(destination.max()), 1.0)
    step = max(5, int(math.ceil(peak / 4 / 5) * 5))
    limit = step * 4
    ticks = np.arange(-limit, limit + step, step)
    ax.set_ylim(-limit * 1.06, limit * 1.06)
    ax.set_yticks(ticks, [str(abs(int(value))) for value in ticks])
    ax.set_xlim(all_weeks.min() - pd.Timedelta(days=5), all_weeks.max() + pd.Timedelta(days=5))
    ax.set_ylabel("Cohort identities per week", labelpad=0.5)
    ax.set_xlabel("Observation week")
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=-2)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title(spec.activity_title, loc="left", pad=3)
    ax.legend(
        loc="upper right", ncol=2, frameon=True, facecolor="white",
        edgecolor="none", framealpha=0.84, fontsize=4.8,
        handletextpad=0.35, columnspacing=0.8,
    )
    for index, (date, label) in enumerate(spec.markers):
        when = pd.Timestamp(date)
        ax.axvline(when, color=MUTED, linewidth=0.65, linestyle=(0, (2, 1.5)), zorder=2)
        ax.scatter([when], [0], marker="D", s=13, facecolor=INK,
                   edgecolor="white", linewidth=0.35, zorder=6)
        ax.text(
            when, 0.91 - 0.19 * index, label,
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=4.4, color=INK,
            bbox={"facecolor": "white", "edgecolor": "#8f9599",
                  "linewidth": 0.75, "alpha": 0.95, "pad": 0.75}, zorder=7,
        )
    if spec.summary_note is not None:
        ax.text(
            0.015, 0.035, spec.summary_note,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=4.5, color=INK,
            bbox={"facecolor": "white", "edgecolor": "#9a9fa2",
                  "linewidth": 0.55, "alpha": 0.94, "pad": 0.8}, zorder=8,
        )
    cohort_size = int(sub.cohort_size.iloc[0])
    continuation_note = spec.continuation_note
    if continuation_note is None and float(destination.tail(4).mean()) >= max(3.0, 0.10 * cohort_size):
        continuation_note = "Destination activity continues\nthrough plotted window"
    if continuation_note is not None:
        ax.text(
            0.985, 0.035,
            continuation_note,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=4.4, color=RED,
            bbox={"facecolor": "white", "edgecolor": "#9a9fa2",
                  "linewidth": 0.55, "alpha": 0.94, "pad": 0.8}, zorder=8,
        )


def make_figure(specs: tuple[EventPlot, ...], stem: str) -> None:
    events, candidates, weekly, points = load_inputs()
    rings = load_world(WORLD)
    fig = plt.figure(figsize=(7.15, 2.08 * len(specs) + 0.12))
    outer = fig.add_gridspec(
        len(specs), 1,
        left=0.045, right=0.992, bottom=0.055, top=0.982,
        hspace=0.32,
    )
    for row_index, spec in enumerate(specs):
        event = event_row(events, spec.event_id)
        sources = significant_sources(candidates, spec.event_id)
        event_points = points[points.event_id.eq(spec.event_id)].copy()
        row = outer[row_index, 0].subgridspec(1, 2, width_ratios=[1.08, 1.54], wspace=0.25)
        map_grid = row[0, 0].subgridspec(1, 2, width_ratios=[0.76, 0.32], wspace=0.045)
        ax_map = fig.add_subplot(map_grid[0, 0])
        detail_grid = map_grid[0, 1].subgridspec(2, 1, hspace=0.34)
        ax_source = fig.add_subplot(detail_grid[0, 0])
        ax_destination = fig.add_subplot(detail_grid[1, 0])
        ax_activity = fig.add_subplot(row[0, 1])
        draw_main_map(ax_map, rings, spec, sources, event_points)
        source_points = event_points[event_points.endpoint.eq("source")]
        destination_points = event_points[event_points.endpoint.eq("destination")]
        draw_detail(
            ax_source,
            source_points.lon.to_numpy(), source_points.lat.to_numpy(),
            "Source", BLUE, 7.5,
        )
        draw_detail(
            ax_destination,
            destination_points.lon.to_numpy(), destination_points.lat.to_numpy(),
            "Destination", RED, 7.5,
        )
        draw_activity(ax_activity, spec, weekly)
    fig.text(
        0.045, 0.012,
        "Boundaries: Natural Earth · detail maps: CARTO / OpenStreetMap",
        ha="left", va="bottom", fontsize=4.4, color=MUTED,
    )
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}.pdf", dpi=300)
    fig.savefig(FIGS / f"{stem}.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refresh weekly data from read-only ClickHouse")
    args = parser.parse_args()
    configure()
    if args.refresh or not WEEKLY_DATA.exists() or not POINT_DATA.exists():
        events = pd.read_csv(DATA / "known_gnss_events.csv", parse_dates=["screen_start", "screen_end"])
        candidates = pd.read_csv(DATA / "event_candidate_bins.csv", parse_dates=["onset_day"])
        refresh_weekly(events, candidates)
    make_figure(STRONG, "reported_gnss_event_rows_strong")
    make_figure(ENDPOINT, "reported_gnss_event_rows_endpoint")


if __name__ == "__main__":
    main()
