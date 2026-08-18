#!/usr/bin/env python3
"""Build compact publication figures for the positive Iran spoofing cases.

The Asaluyeh extract is read from ``cell.summary_full`` through the repository's
read-only ClickHouse helper.  The domestic panels use the detector's auditable
CSV outputs under ``data/spoofing``; consecutive event bins that describe the
same source/destination campaign are combined before plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

from ch_remote import ch_df
from plot_helpers import add_osm_basemap
from spoofing_category_overview import load_world, setup_map


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"
FIGS = ROOT / "paper" / "figs"
WORLD = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"

ASALUYEH_DATA = DATA / "iran_asaluyeh_foreign_identities.csv"
CAMPAIGN_DATA = DATA / "iran_coordinate_displacement_members.csv"
DOMESTIC_ACTIVITY_DATA = DATA / "iran_domestic_campaign_activity.csv"
CONTEXT_DATA = DATA / "iran_event_context.csv"
ASALUYEH_FIG = FIGS / "iran_asaluyeh_spoofing.pdf"
DOMESTIC_FIG = FIGS / "iran_domestic_spoofing.pdf"

INK = "#222426"
MUTED = "#686d70"
GRID = "#d7d9da"
SOURCE = "#276c99"
DESTINATION = "#b4232f"
COUNTRY_COLORS = {
    "QA": "#6f4aa8",
    "BH": "#168a7a",
    "SA": "#d06b20",
    "AE": "#2678b8",
}
COUNTRY_NAMES = {
    "QA": "Qatar",
    "BH": "Bahrain",
    "SA": "Saudi Arabia",
    "AE": "United Arab Emirates",
}
OPERATOR_NAMES = {11: "MCI", 35: "Irancell", 20: "Rightel"}

CAMPAIGNS = {
    "mashhad_tehran_apr2024": {
        "events": ["362_595_2024-04-12", "363_595_2024-04-12"],
        "source": "Mashhad",
        "destination": "Tehran",
        "title": "Apr. 12–13, 2024",
    },
    "kazerun_tehran_jun2025": {
        "events": ["296_516_2025-06-20"],
        "source": "Kazerun",
        "destination": "Tehran",
        "title": "Jun. 20–22, 2025",
    },
    "mashhad_tehran_aug2025": {
        "events": ["362_596_2025-08-27", "362_596_2025-08-28"],
        "source": "Mashhad",
        "destination": "Tehran",
        "title": "Aug. 27–Sep. 7, 2025",
    },
}

ACTIVITY_WINDOWS = {
    "mashhad_tehran_apr2024": ("2024-01-01", "2024-12-01"),
    "kazerun_tehran_jun2025": ("2025-03-01", "2025-10-01"),
    "mashhad_tehran_aug2025": ("2025-06-01", "2025-12-01"),
    "mashhad_kermanshah": ("2025-03-01", "2026-07-15"),
}

EVENT_MARKERS = {
    "kazerun_tehran_jun2025": [
        ("2025-06-20", "Snapp GPS/map-error\nreports peak · Jun. 20"),
    ],
    "mashhad_tehran_aug2025": [
        ("2025-08-20", "Security-related GPS\ndisruption confirmed · Aug. 20"),
    ],
    "mashhad_kermanshah": [
        ("2025-06-23", "Kermanshah missile\nsites struck · Jun. 23"),
    ],
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.2,
        "axes.titlesize": 7.2,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.3,
        "ytick.labelsize": 5.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def extract_asaluyeh() -> pd.DataFrame:
    """Refresh the positive Gulf-MCC cluster from the read-only summary table."""
    frame = ch_df(
        """
        SELECT
            multiIf(mcc=427,'QA',mcc=426,'BH',mcc=420,'SA',mcc=424,'AE','') AS home_iso,
            mcc,mnc,lac,cid,cell_type,first_seen,last_seen,obs,glat,glon,n_pos
        FROM cell.summary_full
        WHERE cid > 0
          AND mcc IN (420,424,426,427)
          AND glat BETWEEN 27.3 AND 27.8
          AND glon BETWEEN 52.2 AND 52.9
          AND NOT (glat=0 AND glon=0)
        ORDER BY home_iso,mnc,lac,cid,cell_type
        """
    )
    if len(frame) < 100 or set(frame["home_iso"]) != {"QA", "BH", "SA", "AE"}:
        raise RuntimeError(
            "Asaluyeh extract failed its positive-cluster sanity check: "
            f"{len(frame)} rows from {sorted(set(frame['home_iso']))}"
        )
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ASALUYEH_DATA, index=False)
    return frame


def export_domestic_campaigns() -> pd.DataFrame:
    """Combine detector bins into three distinct domestic campaigns."""
    source = pd.read_csv(DATA / "coordinate_reassignment_members.csv")
    key = ["mcc", "mnc", "lac", "cid", "cell_type"]
    pieces: list[pd.DataFrame] = []
    for campaign_id, spec in CAMPAIGNS.items():
        subset = source[
            source["mcc"].eq(432) & source["event_id"].isin(spec["events"])
        ].copy()
        subset.insert(0, "campaign_id", campaign_id)
        subset.insert(1, "source_name", spec["source"])
        subset.insert(2, "destination_name", spec["destination"])
        subset.insert(3, "campaign_dates", spec["title"])
        subset = subset.sort_values("away_obs", ascending=False).drop_duplicates(key)
        # One member of the first April detector bin moves only within the
        # Mashhad source area rather than into the common Tehran endpoint.  It
        # is part of the event-level audit but not part of the positive cluster
        # visualized here.
        subset = subset[
            subset["top_destination_lat"].between(35.5, 35.85)
            & subset["top_destination_lon"].between(50.9, 51.7)
        ]
        pieces.append(subset)
    result = pd.concat(pieces, ignore_index=True)
    expected = {
        "mashhad_tehran_apr2024": 19,
        "kazerun_tehran_jun2025": 19,
        "mashhad_tehran_aug2025": 26,
    }
    actual = result.groupby("campaign_id").size().to_dict()
    if actual != expected:
        raise RuntimeError(f"Domestic campaign membership changed: {actual} != {expected}")
    result.to_csv(CAMPAIGN_DATA, index=False)
    return result


def _identity_tuples(group: pd.DataFrame) -> str:
    key = ["mcc", "mnc", "lac", "cid", "cell_type"]
    rows = group[key].drop_duplicates()
    return ",".join(
        f"({int(row.mcc)},{int(row.mnc)},{int(row.lac)},{int(row.cid)},'{row.cell_type}')"
        for row in rows.itertuples(index=False)
    )


def refresh_domestic_activity(
    campaigns: pd.DataFrame,
    replay: pd.DataFrame,
) -> pd.DataFrame:
    """Read weekly endpoint participation for the four plotted cohorts."""
    frames: list[pd.DataFrame] = []
    groups = {
        campaign_id: campaigns[campaigns["campaign_id"].eq(campaign_id)].copy()
        for campaign_id in CAMPAIGNS
    }
    groups["mashhad_kermanshah"] = replay.copy()
    for campaign_id, group in groups.items():
        if not group["mcc"].eq(432).all():
            raise RuntimeError(f"{campaign_id} contains non-Iranian MCC identities")
        if group[["mcc", "mnc", "lac", "cid", "cell_type"]].duplicated().any():
            raise RuntimeError(f"{campaign_id} contains duplicate identity keys")
        replay_case = campaign_id == "mashhad_kermanshah"
        source_lat_col = "home_lat" if replay_case else "reference_lat"
        source_lon_col = "home_lon" if replay_case else "reference_lon"
        destination_lat_col = "destination_lat" if replay_case else "top_destination_lat"
        destination_lon_col = "destination_lon" if replay_case else "top_destination_lon"
        source_lat = float(group[source_lat_col].median())
        source_lon = float(group[source_lon_col].median())
        destination_lat = float(group[destination_lat_col].median())
        destination_lon = float(group[destination_lon_col].median())
        # The Kermanshah cohort's stable locations span the Mashhad metro area;
        # the other source and destination clusters are much tighter.
        radius_km = 80 if replay_case else 35
        start, end = ACTIVITY_WINDOWS[campaign_id]
        identities = _identity_tuples(group)
        frame = ch_df(f"""
          WITH
            greatCircleDistance(lon,lat,{source_lon},{source_lat})/1000 AS source_km,
            greatCircleDistance(lon,lat,{destination_lon},{destination_lat})/1000 AS destination_km,
            if(source_km <= destination_km, 'source', 'destination') AS endpoint
          SELECT
            toStartOfWeek(timestamp, 1) AS week,
            endpoint,
            uniqExact((mcc,mnc,lac,cid,cell_type)) AS identities,
            count() AS observations
          FROM cell.geos
          PREWHERE (mcc,mnc,lac,cid,cell_type) IN ({identities})
          WHERE timestamp >= toDateTime('{start}')
            AND timestamp < toDateTime('{end}')
            AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
            AND NOT (abs(lat) <= 0.01 AND abs(lon) <= 0.01)
            AND least(source_km, destination_km) <= {radius_km}
          GROUP BY week, endpoint
          ORDER BY week, endpoint
        """, settings={"max_threads": 6})
        if frame.empty:
            raise RuntimeError(f"No endpoint activity found for {campaign_id}")
        frame.insert(0, "campaign_id", campaign_id)
        frame.insert(3, "cohort_size", len(group))
        frame.insert(4, "source_name", "Mashhad" if replay_case else CAMPAIGNS[campaign_id]["source"])
        frame.insert(5, "destination_name", "Kermanshah" if replay_case else CAMPAIGNS[campaign_id]["destination"])
        frames.append(frame)
        print(f"{campaign_id}: {len(group)} identities, {len(frame)} weekly endpoint rows")
    result = pd.concat(frames, ignore_index=True)
    result["week"] = pd.to_datetime(result["week"])
    result.to_csv(DOMESTIC_ACTIVITY_DATA, index=False)
    return result


def load_domestic_activity() -> pd.DataFrame:
    if not DOMESTIC_ACTIVITY_DATA.exists():
        raise FileNotFoundError(
            f"{DOMESTIC_ACTIVITY_DATA} is missing; rerun with --refresh-domestic-activity"
        )
    return pd.read_csv(DOMESTIC_ACTIVITY_DATA, parse_dates=["week"])


def load_inputs(
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if refresh or not ASALUYEH_DATA.exists():
        asaluyeh = extract_asaluyeh()
    else:
        asaluyeh = pd.read_csv(ASALUYEH_DATA)
    campaigns = export_domestic_campaigns()
    replay = pd.read_csv(DATA / "destination_cluster_members.csv")
    replay = replay[
        replay["site_id"].eq("3426_4726") & replay["mcc"].eq(432)
    ].copy()
    if len(replay) != 42:
        raise RuntimeError(f"Expected 42 Kermanshah replay identities; got {len(replay)}")
    context = pd.read_csv(CONTEXT_DATA, parse_dates=["start_date", "end_date"])
    required_context = {
        "true_promise", "twelve_day_war", "iran_blackout", "gnss_peak",
        "midnight_hammer", "kermanshah_strikes", "gps_acknowledgment",
        "drone_security_statement", "regional_gnss_2026",
    }
    missing = required_context - set(context["event_id"])
    if missing:
        raise RuntimeError(f"Iran context CSV is missing: {sorted(missing)}")
    return asaluyeh, campaigns, replay, context


def plot_asaluyeh_points(
    ax: plt.Axes,
    frame: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    title: str,
    *,
    zoom: int,
    point_size: float,
) -> None:
    ax.set_facecolor("#e8edf0")
    used_tiles = add_osm_basemap(
        ax, bbox, zoom=zoom, alpha=0.94, grayscale=False,
        source="opentopomap",
    )
    if not used_tiles:
        ax.grid(color="white", linewidth=0.45)
    for iso in ["QA", "BH", "SA", "AE"]:
        group = frame[frame["home_iso"].eq(iso)]
        ax.scatter(
            group["glon"], group["glat"], s=point_size,
            facecolor=COUNTRY_COLORS[iso], edgecolor="white", linewidth=0.38,
            alpha=0.92, label=f"{COUNTRY_NAMES[iso]} ({len(group)})",
            zorder=3, rasterized=False,
        )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(bbox[2:]))), adjustable="box")
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=3)
    if bbox[1] - bbox[0] > 0.2:
        labels = [("Bandar Siraf", 52.34, 27.662), ("Asaluyeh", 52.605, 27.469)]
    else:
        labels = [("Asaluyeh", 52.606, 27.469)]
    for label, lon, lat in labels:
        ax.text(
            lon, lat, label, fontsize=4.8, color="#484b4d", fontweight="bold",
            ha="center", va="center", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(length=2, pad=1)


def render_asaluyeh(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.copy()
    frame["first_seen"] = pd.to_datetime(frame["first_seen"])
    fig = plt.figure(figsize=(7.15, 2.46))
    grid = fig.add_gridspec(
        1, 3, width_ratios=[1.02, 1.02, 1.16],
        left=0.055, right=0.992, bottom=0.19, top=0.89, wspace=0.32,
    )
    ax_all = fig.add_subplot(grid[0, 0])
    ax_core = fig.add_subplot(grid[0, 1])
    ax_time = fig.add_subplot(grid[0, 2])

    all_bbox = (52.18, 52.73, 27.43, 27.74)
    core_bbox = (52.59, 52.68, 27.46, 27.53)
    plot_asaluyeh_points(
        ax_all, frame, all_bbox, f"A. All {len(frame)} foreign identities", zoom=11, point_size=14,
    )
    core = frame[
        frame["glon"].between(core_bbox[0], core_bbox[1])
        & frame["glat"].between(core_bbox[2], core_bbox[3])
    ]
    plot_asaluyeh_points(
        ax_core, core, core_bbox, f"B. {len(core)} converge at Asaluyeh", zoom=14,
        point_size=17,
    )
    ax_all.add_patch(Rectangle(
        (core_bbox[0], core_bbox[2]), core_bbox[1] - core_bbox[0],
        core_bbox[3] - core_bbox[2], fill=False, edgecolor=DESTINATION,
        linewidth=0.85, linestyle=(0, (2, 1.4)), zorder=5,
    ))

    months = frame["first_seen"].dt.to_period("M").dt.to_timestamp()
    start = months.min()
    end = months.max()
    full_months = pd.date_range(start, end, freq="MS")
    bottom = np.zeros(len(full_months))
    for iso in ["QA", "BH", "SA", "AE"]:
        counts = (
            months[frame["home_iso"].eq(iso)].value_counts()
            .reindex(full_months, fill_value=0).sort_index()
        )
        ax_time.bar(
            full_months, counts.to_numpy(), width=25, bottom=bottom,
            color=COUNTRY_COLORS[iso], edgecolor="white", linewidth=0.25,
            label=COUNTRY_NAMES[iso],
        )
        bottom += counts.to_numpy()
    war_start = pd.Timestamp("2025-06-13")
    war_end = pd.Timestamp("2025-07-15")
    ax_time.axvspan(war_start, war_end, color="#f1c7c5", alpha=0.50, zorder=-1)
    during = int(frame["first_seen"].between(war_start, war_end).sum())
    ymax = max(bottom) * 1.13
    ax_time.annotate(
        f"{during}/{len(frame)} first appear\nJun. 13–Jul. 15, 2025",
        xy=(pd.Timestamp("2025-06-27"), ymax * 0.73),
        xytext=(pd.Timestamp("2024-10-01"), ymax * 0.96),
        arrowprops={"arrowstyle": "->", "color": DESTINATION, "lw": 0.8},
        color=DESTINATION, fontsize=5.7, fontweight="bold", ha="center", va="top",
    )
    ax_time.set_ylim(0, ymax)
    ax_time.set_title("C. Conflict-linked first appearances", loc="left", fontweight="bold", pad=3)
    ax_time.set_ylabel("First-seen identities per month")
    ax_time.grid(axis="y", color=GRID, linewidth=0.45, zorder=-2)
    ax_time.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_time.legend(
        loc="upper left", bbox_to_anchor=(0.0, -0.27), ncol=2,
        frameon=False, fontsize=5.2, handlelength=1.2, columnspacing=0.9,
    )

    fig.text(
        0.37, 0.045,
        "Points are distinct cellular identities; color is the network's MCC country.  "
        "Terrain © OpenStreetMap contributors, OpenTopoMap (CC-BY-SA), SRTM.",
        ha="center", va="center", fontsize=4.7, color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)


def curved_arrow(
    ax: plt.Axes,
    source: tuple[float, float],
    destination: tuple[float, float],
) -> None:
    arrow = FancyArrowPatch(
        source, destination, connectionstyle="arc3,rad=-0.10",
        arrowstyle="-|>", mutation_scale=8.5, linewidth=1.15,
        edgecolor=DESTINATION, facecolor=DESTINATION, alpha=0.88, zorder=4,
    )
    ax.add_patch(arrow)


def operator_counts(group: pd.DataFrame) -> str:
    counts = group.groupby("mnc").size().to_dict()
    return " · ".join(
        f"{OPERATOR_NAMES[mnc]} {counts.get(mnc, 0)}" for mnc in [11, 35, 20]
        if counts.get(mnc, 0)
    )


def cluster_bbox(
    group: pd.DataFrame,
    lon_col: str,
    lat_col: str,
) -> tuple[float, float, float, float]:
    """Return a small but non-degenerate view around binned member positions."""
    lon_mid = float(group[lon_col].median())
    lat_mid = float(group[lat_col].median())
    lon_span = max(float(group[lon_col].max() - group[lon_col].min()) * 1.55, 0.075)
    lat_span = max(float(group[lat_col].max() - group[lat_col].min()) * 1.55, 0.060)
    return (
        lon_mid - lon_span / 2, lon_mid + lon_span / 2,
        lat_mid - lat_span / 2, lat_mid + lat_span / 2,
    )


def cluster_zoom(bbox: tuple[float, float, float, float]) -> int:
    span = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    if span <= 0.09:
        return 13
    if span <= 0.20:
        return 12
    if span <= 0.45:
        return 11
    return 10


def draw_reported_points(
    ax: plt.Axes,
    group: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    color: str,
) -> int:
    """Plot every identity at its reported coordinate without display offsets."""
    points = group[[lon_col, lat_col]].dropna()
    ax.scatter(
        points[lon_col], points[lat_col], s=10,
        facecolor=color, edgecolor="white", linewidth=0.28,
        alpha=0.68, zorder=6, rasterized=False,
    )
    return int(len(points.drop_duplicates()))


def cluster_inset(
    ax: plt.Axes,
    group: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    title: str,
    color: str,
) -> tuple[float, float, float, float]:
    bbox = cluster_bbox(group, lon_col, lat_col)
    ax.set_facecolor("#e8edf0")
    used_tiles = add_osm_basemap(
        ax, bbox, zoom=cluster_zoom(bbox), alpha=0.88,
        grayscale=True, grayscale_brightness=1.08, grayscale_contrast=0.92,
        source="opentopomap",
    )
    if not used_tiles:
        ax.grid(color="white", linewidth=0.35)
    draw_reported_points(ax, group, lon_col, lat_col, color)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    # Every magnification occupies the same physical box; the small local
    # extents make the negligible lon/lat distortion preferable to four pairs
    # of visibly different-sized insets.
    ax.set_aspect("auto")
    ax.set_box_aspect(1)
    ax.set_title(title, loc="left", fontsize=5.8, fontweight="bold", color=color, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor(color)
    return bbox


def domestic_case(
    fig: plt.Figure,
    outer,
    group: pd.DataFrame,
    activity: pd.DataFrame,
    rings,
    dates: str,
    source_name: str,
    destination_name: str,
    *,
    replay: bool = False,
) -> None:
    inner = outer.subgridspec(
        1, 2, width_ratios=[1.08, 1.54], wspace=0.25,
    )
    map_grid = inner[0, 0].subgridspec(
        1, 2, width_ratios=[0.76, 0.32], wspace=0.045,
    )
    ax_map = fig.add_subplot(map_grid[0, 0])
    inset_grid = map_grid[0, 1].subgridspec(2, 1, hspace=0.34)
    ax_source = fig.add_subplot(inset_grid[0, 0])
    ax_destination = fig.add_subplot(inset_grid[1, 0])
    ax_activity = fig.add_subplot(inner[0, 1])
    if replay:
        source_lon_col, source_lat_col = "home_lon", "home_lat"
        destination_lon_col, destination_lat_col = "destination_lon", "destination_lat"
        observations = int(group["observations"].sum())
        distance = float(group["median_displacement_km"].median())
    else:
        source_lon_col, source_lat_col = "reference_lon", "reference_lat"
        destination_lon_col, destination_lat_col = "top_destination_lon", "top_destination_lat"
        observations = int(group["away_obs"].sum())
        distance = float(group["med_km"].median())

    source_lon = float(group[source_lon_col].median())
    source_lat = float(group[source_lat_col].median())
    destination_lon = float(group[destination_lon_col].median())
    destination_lat = float(group[destination_lat_col].median())

    national_bbox = (43.2, 63.7, 24.2, 40.5)
    setup_map(ax_map, rings, national_bbox)
    ax_map.set_aspect(1 / np.cos(np.deg2rad(32.5)), adjustable="box")
    curved_arrow(ax_map, (source_lon, source_lat), (destination_lon, destination_lat))
    ax_map.scatter(
        group[source_lon_col], group[source_lat_col], s=7, facecolor=SOURCE,
        edgecolor="white", linewidth=0.25, alpha=0.48, zorder=5,
    )
    ax_map.scatter(
        [source_lon], [source_lat], s=30, facecolor=SOURCE, edgecolor="white",
        linewidth=0.65, zorder=6,
    )
    ax_map.scatter(
        group[destination_lon_col], group[destination_lat_col], s=7,
        facecolor=DESTINATION, edgecolor="white", linewidth=0.25,
        alpha=0.48, zorder=5,
    )
    ax_map.scatter(
        [destination_lon], [destination_lat], s=38, marker="D",
        facecolor=DESTINATION, edgecolor="white", linewidth=0.6, zorder=7,
    )
    ax_map.annotate(
        source_name, (source_lon, source_lat), xytext=(3, 4),
        textcoords="offset points", fontsize=5.3, fontweight="bold",
        color=SOURCE, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
        zorder=8,
    )
    ax_map.annotate(
        destination_name, (destination_lon, destination_lat), xytext=(3, -8),
        textcoords="offset points", fontsize=5.3, fontweight="bold",
        color=DESTINATION,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
        zorder=8,
    )
    ax_map.set_title(
        f"{source_name} to {destination_name}\n{dates}",
        loc="left", fontsize=7.0, fontweight="bold", color=INK, pad=2.5,
    )
    ax_map.text(
        0.02, 0.02,
        f"{len(group)} identities · {distance:,.0f} km",
        transform=ax_map.transAxes, fontsize=4.8, color=INK, va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#babdc0", "linewidth": 0.35,
              "alpha": 0.92, "pad": 1.2}, zorder=9,
    )
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_xlabel("")
    ax_map.set_ylabel("")

    cluster_inset(
        ax_source, group, source_lon_col, source_lat_col, "Source", SOURCE,
    )
    cluster_inset(
        ax_destination, group, destination_lon_col, destination_lat_col,
        "Destination", DESTINATION,
    )

    campaign_id = str(activity["campaign_id"].iloc[0])
    start, end = map(pd.Timestamp, ACTIVITY_WINDOWS[campaign_id])
    weekly = activity.pivot(index="week", columns="endpoint", values="identities")
    monday_start = start - pd.Timedelta(days=start.weekday())
    all_weeks = pd.date_range(
        monday_start, end - pd.Timedelta(days=1), freq="W-MON",
    )
    weekly = weekly.reindex(all_weeks, fill_value=0).fillna(0)
    source_values = weekly.get("source", pd.Series(0, index=weekly.index))
    destination_values = weekly.get("destination", pd.Series(0, index=weekly.index))
    ax_activity.bar(
        weekly.index, source_values, width=5.6, color=SOURCE,
        edgecolor="white", linewidth=0.35, label=source_name, zorder=3,
    )
    ax_activity.bar(
        weekly.index, -destination_values, width=5.6, color=DESTINATION,
        edgecolor="white", linewidth=0.35, label=destination_name, zorder=3,
    )
    ax_activity.axhline(0, color=INK, linewidth=0.75, zorder=4)
    tick_step = max(5, int(np.ceil(len(group) / 10) * 5))
    limit = tick_step * 2
    ticks = np.arange(-limit, limit + tick_step, tick_step)
    ax_activity.set_ylim(-limit * 1.06, limit * 1.06)
    ax_activity.set_yticks(ticks, [str(abs(int(value))) for value in ticks])
    ax_activity.set_xlim(all_weeks.min() - pd.Timedelta(days=5),
                         all_weeks.max() + pd.Timedelta(days=5))
    ax_activity.set_ylabel("Cohort identities per week", labelpad=0.5)
    ax_activity.set_xlabel("Observation week")
    ax_activity.grid(axis="y", color=GRID, linewidth=0.45, zorder=-2)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax_activity.xaxis.set_major_locator(locator)
    ax_activity.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_activity.set_title("Weekly activity of the same identities", loc="left", pad=3)
    ax_activity.legend(
        loc="upper right", ncol=2, frameon=True, facecolor="white",
        edgecolor="none", framealpha=0.82, fontsize=4.8,
        handletextpad=0.35, columnspacing=0.8,
    )
    for event_date, label in EVENT_MARKERS.get(campaign_id, []):
        when = pd.Timestamp(event_date)
        ax_activity.axvline(
            when, color=MUTED, linewidth=0.65, linestyle=(0, (2, 1.5)),
            alpha=0.88, zorder=2,
        )
        ax_activity.scatter(
            [when], [0], marker="D", s=13, facecolor=INK,
            edgecolor="white", linewidth=0.35, zorder=6,
        )
        ax_activity.text(
            when, 0.91, label,
            transform=ax_activity.get_xaxis_transform(),
            ha="center", va="top", fontsize=4.45, color=INK,
            bbox={
                "facecolor": "white", "edgecolor": "#aeb3b7",
                "linewidth": 0.45, "alpha": 0.92, "pad": 0.7,
            },
            zorder=7,
        )


def render_event_timeline(ax: plt.Axes, context: pd.DataFrame) -> None:
    context = context.set_index("event_id")
    events = [
        ("A  Mashhad → Tehran", "2024-04-12", "2024-04-13", 19, "#6f4aa8"),
        ("B  Kazerun → Tehran", "2025-06-20", "2025-06-22", 19, "#b4232f"),
        ("D  Mashhad → Kermanshah", "2025-06-26", "2025-08-26", 42, "#2878b5"),
        ("C  Mashhad → Tehran", "2025-08-27", "2025-09-07", 26, "#d06b20"),
    ]
    y_positions = np.arange(len(events))[::-1]
    for y, (label, start, end, identities, color) in zip(y_positions, events):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        # One-day events need a visible minimum width at the full collection scale.
        ax.plot(
            [start_ts, end_ts], [y, y], color=color, linewidth=5.0,
            solid_capstyle="round", zorder=3,
        )
        ax.scatter([start_ts, end_ts], [y, y], s=12, color=color,
                   edgecolor="white", linewidth=0.35, zorder=4)
        ax.text(
            end_ts + pd.Timedelta(days=10), y + 0.12,
            f"{identities} IDs", fontsize=5.1, color=color,
            fontweight="bold", va="bottom",
        )
    # The Kermanshah replay returns twice in June 2026 after the main 2025 run.
    recurrence_start = pd.Timestamp("2026-06-21")
    recurrence_end = pd.Timestamp("2026-06-30")
    replay_y = y_positions[2]
    ax.plot(
        [pd.Timestamp("2025-08-26"), recurrence_start], [replay_y, replay_y],
        color="#2878b5", linewidth=0.7, linestyle=(0, (1.5, 2.0)), alpha=0.55,
    )
    ax.plot(
        [recurrence_start, recurrence_end], [replay_y, replay_y],
        color="#2878b5", linewidth=5.0, solid_capstyle="round", zorder=3,
    )
    ax.scatter(
        [recurrence_start, recurrence_end], [replay_y, replay_y], s=12,
        color="#2878b5", edgecolor="white", linewidth=0.35, zorder=4,
    )
    ax.text(
        recurrence_start - pd.Timedelta(days=12), replay_y + 0.12,
        "2 IDs recur", fontsize=5.1, color="#2878b5", fontweight="bold",
        ha="right", va="bottom",
    )
    ax.set_yticks(y_positions, [row[0] for row in events])
    ax.set_xlim(pd.Timestamp("2024-03-15"), pd.Timestamp("2026-07-15"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", which="major", length=3, pad=1.5, labelsize=5.0)
    ax.tick_params(axis="x", which="minor", length=1.5, color="#a9adb0")
    ax.tick_params(axis="y", length=0, pad=4, labelsize=5.4)
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.5, zorder=0)
    ax.set_ylim(-0.55, len(events) - 0.18)
    # The April episode overlaps Iran's first direct attack on Israel.  This is
    # a geopolitical alignment, not an actor attribution.
    attack = context.at["true_promise", "start_date"]
    ax.axvline(attack, color="#3f4244", linewidth=0.65,
               linestyle=(0, (2, 1.5)), alpha=0.8, zorder=1)
    ax.text(
        attack, 3.58, "Apr. 13: Iran attacks Israel", fontsize=4.45,
        color="#3f4244", va="bottom", ha="center",
    )
    # Documented regional interference supplies context for the small 2026
    # recurrence.  The shading communicates context without attributing the
    # two-identity recurrence to a particular 2026 action.
    ax.axvspan(context.at["regional_gnss_2026", "start_date"],
               context.at["regional_gnss_2026", "end_date"],
               color="#cfe2ef", alpha=0.42, zorder=-1)
    ax.text(
        pd.Timestamp("2026-05-10"), 3.56,
        "documented regional GNSS interference", fontsize=4.35,
        color="#315d78", va="bottom", ha="center",
    )
    ax.set_title("E. Displacement episodes across the collection period", loc="left",
                 fontsize=7.0, fontweight="bold", pad=2.5)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)


def render_context_timeline(ax: plt.Axes, context: pd.DataFrame) -> None:
    """Align the 2025 dataset episodes with independently reported context."""
    context = context.set_index("event_id")
    war_start = context.at["twelve_day_war", "start_date"]
    war_end = context.at["twelve_day_war", "end_date"]
    blackout_start = context.at["iran_blackout", "start_date"]
    blackout_end = context.at["iran_blackout", "end_date"]
    peak_start = context.at["gnss_peak", "start_date"]
    peak_end = context.at["gnss_peak", "end_date"]
    us_strike = context.at["midnight_hammer", "start_date"]
    kermanshah_strike = context.at["kermanshah_strikes", "start_date"]
    gps_ack = context.at["gps_acknowledgment", "start_date"]
    drone_statement = context.at["drone_security_statement", "start_date"]
    ax.set_xlim(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-09-15"))
    ax.set_ylim(-0.48, 3.50)
    for y in [0, 2]:
        ax.axhspan(y - 0.42, y + 0.42, color="#f4f5f5", zorder=-3)
    for y in [0.5, 1.5, 2.5]:
        ax.axhline(y, color="#e5e6e7", linewidth=0.45, zorder=-2)

    # Conflict lane.
    ax.plot(
        [war_start, war_end], [3, 3],
        color="#9d3439", linewidth=5.0, solid_capstyle="round", zorder=3,
    )
    ax.text(
        war_start + (war_end - war_start) / 2, 3.18,
        "12-day war · Jun. 13–24", fontsize=4.8, color="#9d3439",
        fontweight="bold", ha="center", va="bottom",
    )
    for when in [us_strike, kermanshah_strike]:
        date = pd.Timestamp(when)
        ax.scatter([date], [3], marker="D", s=12, color="#5b2629",
                   edgecolor="white", linewidth=0.35, zorder=5)
    ax.text(
        pd.Timestamp("2025-07-01"), 3.24,
        "Jun. 22 · U.S. nuclear-site strikes\n"
        "Jun. 23 · Kermanshah missile sites struck",
        fontsize=4.45, color="#5b2629", va="top", ha="left",
    )

    # Information-access lane.
    ax.plot(
        [blackout_start, blackout_end], [2, 2],
        color="#3f4244", linewidth=5.0, solid_capstyle="round", zorder=3,
    )
    ax.text(
        blackout_start + (blackout_end - blackout_start) / 2, 2.18,
        "national Internet blackout · Jun. 18–25", fontsize=4.8,
        color="#3f4244", fontweight="bold", ha="center", va="bottom",
    )

    # Navigation lane.
    ax.plot(
        [peak_start, peak_end], [1, 1],
        color="#7b4b8f", linewidth=5.2, solid_capstyle="round", zorder=4,
    )
    ax.text(
        pd.Timestamp("2025-06-05"), 0.78,
        "measured spoofing peak · Jun. 21–22", fontsize=4.4,
        color="#6f4a80", ha="left", va="top",
    )
    for date, label, align in [
        (gps_ack, "domestic GPS disruption\nacknowledged · Jul. 14", "center"),
        (drone_statement, "minister cites drone threat\nAug. 20", "center"),
    ]:
        ax.scatter([date], [1], marker="D", s=13, color="#7b4b8f",
                   edgecolor="white", linewidth=0.35, zorder=5)
        ax.text(date, 1.18, label, fontsize=4.35, color="#6f4a80",
                ha=align, va="bottom")

    # The three 2025 dataset episodes share the expanded scale below the
    # external context, making the temporal alignments visible without implying
    # that any individual military action caused a recorded reassignment.
    dataset_events = [
        ("B · 19 IDs", "2025-06-20", "2025-06-22", "#b4232f", 0.22),
        ("D · 42 IDs", "2025-06-26", "2025-08-26", "#2878b5", 0.00),
        ("C · 26 IDs", "2025-08-27", "2025-09-07", "#d06b20", -0.22),
    ]
    for label, start, end, color, y in dataset_events:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        ax.plot([start_ts, end_ts], [y, y], color=color, linewidth=4.2,
                solid_capstyle="round", zorder=3)
        if label.startswith("B"):
            ax.text(start_ts - pd.Timedelta(days=2), y, label, fontsize=4.6,
                    color=color, fontweight="bold", ha="right", va="center")
        else:
            ax.text(
                start_ts + (end_ts - start_ts) / 2, y, label,
                fontsize=4.45, color="white", fontweight="bold",
                ha="center", va="center", zorder=5,
            )

    ax.set_yticks([3, 2, 1, 0], ["conflict", "Internet", "GNSS", "dataset"])
    ax.tick_params(axis="y", length=0, pad=4, labelsize=4.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.tick_params(axis="x", which="major", length=3, pad=1.3,
                   labelsize=4.8, labelbottom=True)
    ax.tick_params(axis="x", which="minor", length=1.3, color="#a9adb0")
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.5, zorder=-3)
    ax.set_title("F. June–September 2025 events", loc="left",
                 fontsize=7.0, fontweight="bold", pad=2.0)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)


def render_domestic(
    campaigns: pd.DataFrame,
    replay: pd.DataFrame,
    activity: pd.DataFrame,
    output: Path,
) -> None:
    rings = load_world(WORLD)
    fig = plt.figure(figsize=(7.15, 8.15))
    grid = fig.add_gridspec(
        4, 1,
        left=0.045, right=0.992, bottom=0.045, top=0.982,
        hspace=0.32,
    )
    order = [
        "mashhad_tehran_apr2024",
        "kazerun_tehran_jun2025",
        "mashhad_tehran_aug2025",
    ]
    for outer, campaign_id in zip([grid[0, 0], grid[1, 0], grid[2, 0]], order):
        spec = CAMPAIGNS[campaign_id]
        group = campaigns[campaigns["campaign_id"].eq(campaign_id)]
        domestic_case(
            fig, outer, group,
            activity[activity["campaign_id"].eq(campaign_id)],
            rings, spec["title"], spec["source"],
            spec["destination"],
        )
    domestic_case(
        fig, grid[3, 0], replay,
        activity[activity["campaign_id"].eq("mashhad_kermanshah")],
        rings, "Jun.–Aug. 2025 and Jun. 2026",
        "Mashhad", "Kermanshah", replay=True,
    )
    fig.text(
        0.045, 0.010,
        "Boundaries: Natural Earth · detail maps: OpenTopoMap / OpenStreetMap / SRTM",
        ha="left", va="bottom",
        fontsize=4.4, color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="Refresh the Asaluyeh CSV through the read-only database connection.",
    )
    parser.add_argument(
        "--refresh-domestic-activity", action="store_true",
        help="Refresh weekly activity for the four domestic cohorts through the read-only database connection.",
    )
    parser.add_argument("--asaluyeh-output", type=Path, default=ASALUYEH_FIG)
    parser.add_argument("--domestic-output", type=Path, default=DOMESTIC_FIG)
    args = parser.parse_args()
    configure_style()
    asaluyeh, campaigns, replay, context = load_inputs(args.refresh)
    if args.refresh_domestic_activity:
        domestic_activity = refresh_domestic_activity(campaigns, replay)
    else:
        domestic_activity = load_domestic_activity()
    render_asaluyeh(asaluyeh, args.asaluyeh_output)
    render_domestic(campaigns, replay, domestic_activity, args.domestic_output)
    print(args.asaluyeh_output)
    print(args.domestic_output)


if __name__ == "__main__":
    main()
