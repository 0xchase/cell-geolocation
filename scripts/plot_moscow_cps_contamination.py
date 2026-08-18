#!/usr/bin/env python3
"""Show why the persistent Moscow displacement is a CPS database state.

The underlying observations are read-only queries of Apple's returned cell
landmark coordinates.  They are not live RF or GNSS receiver measurements.
This figure separates the May 2025 one-way backend migration from the gradual
source-to-Sheremetyevo geometry that follows it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/spoofing/news_validation"
FIGS = ROOT / "paper/figs"
POINT_DATA = DATA / "reported_event_endpoint_points.csv"
WEEKLY_DATA = DATA / "reported_event_weekly_activity.csv"
TRAJECTORY_DATA = DATA / "moscow_cps_trajectory.csv"

EVENT_ID = "moscow_may2025"
QUERY_START = "2025-02-01"
QUERY_END = "2026-07-01"
SVO_LAT = 55.972
SVO_LON = 37.415

BLUE = "#286f9b"
RED = "#b43c49"
PALE_RED = "#e2a4aa"
INK = "#26292c"
MUTED = "#71777b"
GRID = "#d9dde0"
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.3,
        "axes.titlesize": 7.1,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.3,
        "ytick.labelsize": 5.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def refresh_trajectory() -> pd.DataFrame:
    points = pd.read_csv(POINT_DATA)
    event_points = points[points.event_id.eq(EVENT_ID)].copy()
    identities = event_points[KEY].drop_duplicates()
    weekly = pd.read_csv(WEEKLY_DATA)
    meta = weekly[weekly.event_id.eq(EVENT_ID)].iloc[0]
    keys = ",".join(
        f"({int(row.mcc)},{int(row.mnc)},{int(row.lac)},{int(row.cid)},'{row.cell_type}')"
        for row in identities.itertuples(index=False)
    )
    query = f"""
    WITH
      greatCircleDistance(lon,lat,{float(meta.source_lon):.8f},{float(meta.source_lat):.8f})/1000 AS source_km,
      greatCircleDistance(lon,lat,{SVO_LON:.8f},{SVO_LAT:.8f})/1000 AS destination_km,
      if(destination_km < source_km, 'destination', 'source') AS endpoint
    SELECT
      mcc, mnc, lac, cid, toString(cell_type) AS cell_type,
      toDate(timestamp) AS day, endpoint,
      quantileExact(0.5)(lat) AS returned_lat,
      quantileExact(0.5)(lon) AS returned_lon,
      count() AS observations
    FROM cell.geos
    PREWHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({keys})
    WHERE timestamp >= toDateTime('{QUERY_START}')
      AND timestamp < toDateTime('{QUERY_END}')
      AND ((endpoint = 'source' AND source_km <= 25)
        OR (endpoint = 'destination' AND destination_km <= 20))
    GROUP BY mcc,mnc,lac,cid,cell_type,day,endpoint
    ORDER BY mcc,mnc,lac,cid,cell_type,day,endpoint
    """
    frame = ch_df(
        query,
        settings={"max_threads": 6, "optimize_aggregation_in_order": 0},
    )
    if frame.empty:
        raise RuntimeError("No Moscow trajectory observations returned")

    source = (
        event_points[event_points.endpoint.eq("source")][KEY + ["lat", "lon"]]
        .rename(columns={"lat": "source_lat", "lon": "source_lon"})
        .drop_duplicates(KEY)
    )
    frame = frame.merge(source, on=KEY, how="inner", validate="many_to_one")
    lat_mid = (frame.source_lat + SVO_LAT) / 2
    km_per_lon = 111.32 * np.cos(np.radians(lat_mid))
    vx = (SVO_LON - frame.source_lon) * km_per_lon
    vy = (SVO_LAT - frame.source_lat) * 110.57
    px = (frame.returned_lon - frame.source_lon) * km_per_lon
    py = (frame.returned_lat - frame.source_lat) * 110.57
    length_sq = vx * vx + vy * vy
    frame["along_fraction"] = (px * vx + py * vy) / length_sq
    frame["cross_track_km"] = (px * vy - py * vx) / np.sqrt(length_sq)
    frame.to_csv(TRAJECTORY_DATA, index=False)
    print(f"Wrote {len(frame):,} daily CPS-return rows to {TRAJECTORY_DATA}")
    return frame


def load_trajectory(refresh: bool) -> pd.DataFrame:
    if refresh or not TRAJECTORY_DATA.exists():
        frame = refresh_trajectory()
    else:
        frame = pd.read_csv(TRAJECTORY_DATA)
    frame["day"] = pd.to_datetime(frame.day)
    return frame


def bordered_note(ax: plt.Axes, x, y, text: str, **kwargs) -> None:
    options = {
        "ha": "center",
        "va": "top",
        "fontsize": 4.7,
        "color": INK,
        "bbox": {
            "facecolor": "white",
            "edgecolor": "#8f9599",
            "linewidth": 0.75,
            "alpha": 0.96,
            "pad": 0.8,
        },
        "zorder": 8,
    }
    options.update(kwargs)
    ax.text(x, y, text, **options)


def draw_transition(ax: plt.Axes, frame: pd.DataFrame) -> None:
    view = frame[frame.day.between("2025-04-19", "2025-06-21")]
    counts = (
        view.groupby(["day", "endpoint"])[KEY]
        .size()
        .unstack(fill_value=0)
        .reindex(pd.date_range("2025-04-19", "2025-06-21"), fill_value=0)
    )
    source = counts.get("source", pd.Series(0, index=counts.index))
    destination = counts.get("destination", pd.Series(0, index=counts.index))
    ax.bar(counts.index, source, width=0.86, color=BLUE, linewidth=0,
           label="Original Moscow estimate", zorder=3)
    ax.bar(counts.index, -destination, width=0.86, color=RED, linewidth=0,
           label="Sheremetyevo-directed estimate", zorder=3)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=4)
    ax.set_ylim(-292, 292)
    ticks = np.arange(-280, 281, 70)
    ax.set_yticks(ticks, [str(abs(int(value))) for value in ticks])
    ax.set_ylabel("Fixed identities per day")
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=-2)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("A. The May episode is a one-way CPS landmark migration", loc="left", pad=3)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=5.0,
              handletextpad=0.35, columnspacing=0.9)
    for when, label, ypos in [
        (pd.Timestamp("2025-05-07"), "Reported GNSS\nescalation", 0.87),
        (pd.Timestamp("2025-05-09"), "Victory Day", 0.63),
    ]:
        ax.axvline(when, color=MUTED, linewidth=0.65, linestyle=(0, (2, 1.5)), zorder=2)
        bordered_note(
            ax, when, ypos, label,
            transform=ax.get_xaxis_transform(),
        )
    onset = (
        frame[frame.endpoint.eq("destination")]
        .groupby(KEY).day.min().median()
    )
    ax.annotate(
        f"Median identity update: {onset:%b %d}",
        xy=(onset, -145), xytext=(pd.Timestamp("2025-05-26"), -225),
        fontsize=5.0, color=INK,
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.65},
        bbox={"facecolor": "white", "edgecolor": "#8f9599",
              "linewidth": 0.75, "alpha": 0.96, "pad": 0.8},
        zorder=8,
    )


def first_last_destination(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = frame[frame.endpoint.eq("destination")].sort_values(KEY + ["day"])
    first = destination.groupby(KEY, as_index=False).first()
    last = destination.groupby(KEY, as_index=False).last()
    return first, last


def draw_geometry(ax: plt.Axes, frame: pd.DataFrame) -> None:
    first, last = first_last_destination(frame)
    paired = first.merge(last, on=KEY, suffixes=("_first", "_last"), validate="one_to_one")
    for row in paired.itertuples(index=False):
        ax.plot(
            [row.along_fraction_first, row.along_fraction_last],
            [row.cross_track_km_first, row.cross_track_km_last],
            color="#b9bdc0", linewidth=0.38, alpha=0.45, zorder=1,
        )
    ax.scatter(first.along_fraction, first.cross_track_km, s=7.0,
               facecolor=PALE_RED, edgecolor="white", linewidth=0.25,
               alpha=0.78, label="First updated estimate", zorder=3)
    ax.scatter(last.along_fraction, last.cross_track_km, s=8.0, marker="D",
               facecolor=RED, edgecolor="white", linewidth=0.25,
               alpha=0.78, label="Latest estimate", zorder=4)
    ax.axhline(0, color=INK, linewidth=0.7, linestyle=(0, (2, 1.5)))
    ax.axvline(0, color=BLUE, linewidth=0.65)
    ax.axvline(1, color=RED, linewidth=0.65)
    ax.set_xlim(0.18, 1.04)
    ax.set_ylim(-0.42, 0.42)
    ax.set_xlabel("Position along each identity's source→Sheremetyevo axis\n(0 = original; 1 = airport coordinate)")
    ax.set_ylabel("Perpendicular offset (km)")
    ax.grid(color=GRID, linewidth=0.4, zorder=-2)
    ax.set_title("B. Updates follow each identity's airport axis", loc="left", pad=3)
    ax.legend(loc="upper left", frameon=False, fontsize=4.8,
              handletextpad=0.35, borderaxespad=0.3)
    median_cross = float(last.cross_track_km.abs().median())
    inside = float((last.cross_track_km.abs() <= 0.10).mean())
    bordered_note(
        ax, 0.985, 0.04,
        f"Latest median offset: {median_cross * 1000:.0f} m\n{inside:.0%} within 100 m of axis",
        transform=ax.transAxes, ha="right", va="bottom",
    )


def draw_progression(ax: plt.Axes, frame: pd.DataFrame) -> None:
    ordered = frame.sort_values(KEY + ["day"])
    ordered["week"] = ordered.day - pd.to_timedelta(ordered.day.dt.weekday, unit="D")
    per_identity = ordered.groupby(KEY + ["week"], as_index=False).last()
    cohort_summary = per_identity.groupby("week").along_fraction.agg(median="median")
    updated = ordered[ordered.endpoint.eq("destination")]
    per_updated_identity = updated.groupby(KEY + ["week"], as_index=False).last()
    summary = per_updated_identity.groupby("week").along_fraction.agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
        identities="size",
    )
    ax.fill_between(summary.index, summary.q25, summary.q75,
                    color=PALE_RED, alpha=0.42, linewidth=0, label="Middle 50%")
    ax.plot(cohort_summary.index, cohort_summary["median"], color=MUTED,
            linewidth=0.8, linestyle=(0, (2, 1.5)),
            label="Whole-cohort median", zorder=3)
    ax.plot(summary.index, summary["median"], color=RED, linewidth=1.35,
            label="Median updated estimate", zorder=4)
    ax.axhline(0, color=BLUE, linewidth=0.65)
    ax.axhline(1, color=RED, linewidth=0.65)
    ax.axvline(pd.Timestamp("2025-05-07"), color=MUTED, linewidth=0.65,
               linestyle=(0, (2, 1.5)))
    ax.set_xlim(pd.Timestamp("2025-02-01"), pd.Timestamp("2026-07-01"))
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0],
                  ["0\noriginal", ".25", ".50", ".75", "1\nSheremetyevo"])
    ax.set_ylabel("Fraction along source→airport axis")
    ax.set_xlabel("CPS observation week")
    ax.grid(axis="y", color=GRID, linewidth=0.45, zorder=-2)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title("C. Updated estimates keep moving toward Sheremetyevo", loc="left", pad=3)
    ax.legend(loc="lower right", frameon=False, fontsize=4.8,
              handletextpad=0.35, borderaxespad=0.4)
    first, last = first_last_destination(frame)
    improvement = float(
        (last.set_index(KEY).along_fraction - first.set_index(KEY).along_fraction > 0).mean()
    )
    bordered_note(
        ax, 0.02, 0.96,
        f"{improvement:.0%} finish closer to the airport coordinate",
        transform=ax.transAxes, ha="left", va="top",
    )
    bordered_note(
        ax, pd.Timestamp("2025-05-07"), 0.70,
        "Reported GNSS\nescalation",
        transform=ax.get_xaxis_transform(), ha="center", va="top",
    )


def make_figure(frame: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.15, 4.88))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[0.88, 1.12],
        left=0.072, right=0.992, bottom=0.165, top=0.965,
        hspace=0.48, wspace=0.27,
    )
    draw_transition(fig.add_subplot(grid[0, :]), frame)
    draw_geometry(fig.add_subplot(grid[1, 0]), frame)
    draw_progression(fig.add_subplot(grid[1, 1]), frame)
    fig.text(
        0.072, 0.030,
        "These timestamps are repeated Apple CPS query results. Persistence measures the returned landmark state, not continuous RF-transmitter uptime.",
        ha="left", va="bottom", fontsize=5.1, color=INK, fontweight="bold",
    )
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "moscow_cps_contamination.pdf", dpi=350)
    fig.savefig(FIGS / "moscow_cps_contamination.png", dpi=240)
    plt.close(fig)
    print(FIGS / "moscow_cps_contamination.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refresh daily data with read-only ClickHouse queries")
    args = parser.parse_args()
    configure()
    make_figure(load_trajectory(args.refresh))


if __name__ == "__main__":
    main()
