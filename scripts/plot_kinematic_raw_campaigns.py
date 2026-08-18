#!/usr/bin/env python3
"""Plot every strong raw-time kinematic campaign as a map and pair timeline."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_helpers import TILE_ATTRIBUTION, add_osm_basemap
from spoofing_category_overview import load_world, setup_map


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing" / "remaining_search"
FIGS = ROOT / "paper" / "figs"
WORLD = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
OUTPUT = FIGS / "kinematic_raw_campaigns.pdf"
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]
EARTH_KM = 6371.0088

COLORS = ["#2774a6", "#d07824", "#3b8b68", "#a34c89", "#6b63a8"]
DESTINATION = "#b43b48"
GRID = "#d7d9dc"
MUTED = "#676b70"
BASEMAP = "carto_voyager"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.0,
        "axes.titlesize": 7.0,
        "axes.labelsize": 5.6,
        "xtick.labelsize": 5.0,
        "ytick.labelsize": 5.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))


def components(frame: pd.DataFrame) -> list[int]:
    """Reproduce the source/destination/time linkage used by the search summary."""
    values = frame.reset_index(drop=True)
    parent = list(range(len(values)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if abs((values.onset_day.iloc[left] - values.onset_day.iloc[right]).days) > 30:
                continue
            source_distance = haversine(
                values.source_lat.iloc[left], values.source_lon.iloc[left],
                values.source_lat.iloc[right], values.source_lon.iloc[right],
            )
            destination_distance = haversine(
                values.destination_lat.iloc[left], values.destination_lon.iloc[left],
                values.destination_lat.iloc[right], values.destination_lon.iloc[right],
            )
            if source_distance <= 100 and destination_distance <= 25:
                union(left, right)
    roots = [root(index) for index in range(len(values))]
    labels = {value: index + 1 for index, value in enumerate(dict.fromkeys(roots))}
    return [labels[value] for value in roots]


def load_campaign_pairs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    campaigns = pd.read_csv(
        DATA / "kinematic_raw_campaigns.csv", parse_dates=["first_event", "last_event"]
    )
    batches = pd.read_csv(DATA / "kinematic_raw_event_batches.csv", parse_dates=["onset_day"])
    strong = batches[batches.strong_coordinated_raw_batch].copy().reset_index(drop=True)
    strong["component"] = components(strong)

    pairs = pd.read_csv(
        DATA / "kinematic_impossible_pairs.csv",
        parse_dates=["home_timestamp", "away_timestamp"],
    )
    pairs["onset_day"] = pairs.away_timestamp.dt.floor("D")
    pairs["dest_lat5"] = (np.trunc(pairs.destination_lat * 20) * 5).astype(int)
    pairs["dest_lon5"] = (np.trunc(pairs.destination_lon * 20) * 5).astype(int)
    pairs = pairs.sort_values("gap_seconds").drop_duplicates(
        ["onset_day", "dest_lat5", "dest_lon5", *KEY]
    )

    by_campaign: dict[str, pd.DataFrame] = {}
    for campaign in campaigns.itertuples(index=False):
        event_rows = strong[strong.component.eq(campaign.component)]
        event_keys = {
            (row.onset_day, int(round(row.destination_lat * 100)),
             int(round(row.destination_lon * 100)))
            for row in event_rows.itertuples(index=False)
        }
        mask = [
            (row.onset_day, row.dest_lat5, row.dest_lon5) in event_keys
            for row in pairs.itertuples(index=False)
        ]
        group = pairs.loc[mask].sort_values(["away_timestamp", "mcc", "mnc"]).copy()
        if len(group) != int(event_rows.identities.sum()):
            raise RuntimeError(
                f"{campaign.campaign_id}: selected {len(group)} pairs but event batches contain "
                f"{int(event_rows.identities.sum())} identity-days"
            )
        by_campaign[campaign.campaign_id] = group
    return campaigns, by_campaign


def map_bbox(group: pd.DataFrame) -> tuple[float, float, float, float]:
    lons = pd.concat([group.home_lon, group.away_lon])
    lats = pd.concat([group.home_lat, group.away_lat])
    lon_span = max(float(lons.max() - lons.min()), 0.45)
    lat_span = max(float(lats.max() - lats.min()), 0.38)
    lon_margin = max(0.16, lon_span * 0.12)
    lat_margin = max(0.13, lat_span * 0.12)
    return (
        float(lons.min() - lon_margin), float(lons.max() + lon_margin),
        float(lats.min() - lat_margin), float(lats.max() + lat_margin),
    )


def basemap_zoom(bbox: tuple[float, float, float, float]) -> int:
    span = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    if span > 3.2:
        return 6
    if span > 1.7:
        return 7
    return 8


def operator_colors(group: pd.DataFrame) -> dict[tuple[int, int], str]:
    operators = sorted({(int(row.mcc), int(row.mnc)) for row in group.itertuples(index=False)})
    return {operator: COLORS[index % len(COLORS)] for index, operator in enumerate(operators)}


def draw_map(ax: plt.Axes, group: pd.DataFrame, colors, rings) -> None:
    bbox = map_bbox(group)
    used_tiles = add_osm_basemap(
        ax, bbox, zoom=basemap_zoom(bbox), alpha=0.94,
        grayscale=False, zorder=0, source=BASEMAP,
    )
    if not used_tiles:
        setup_map(ax, rings, bbox)
    for row in group.itertuples(index=False):
        color = colors[(int(row.mcc), int(row.mnc))]
        ax.plot(
            [row.home_lon, row.away_lon], [row.home_lat, row.away_lat],
            color=color, linewidth=0.45, alpha=0.22, zorder=2,
        )
    for operator, color in colors.items():
        subset = group[(group.mcc == operator[0]) & (group.mnc == operator[1])]
        ax.scatter(
            subset.home_lon, subset.home_lat, s=8.5, color=color,
            edgecolor="white", linewidth=0.3, alpha=0.88, zorder=3,
        )
    ax.scatter(
        group.away_lon, group.away_lat, s=17, marker="*", color=DESTINATION,
        edgecolor="white", linewidth=0.35, alpha=0.92, zorder=4,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    center_lat = (bbox[2] + bbox[3]) / 2
    ax.set_aspect(1 / math.cos(math.radians(center_lat)), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#8c8984")
        spine.set_linewidth(0.4)
    ax.set_title("Reported home → common destination", loc="left", pad=2)


def draw_timeline(ax: plt.Axes, group: pd.DataFrame, colors) -> None:
    for row in group.itertuples(index=False):
        color = colors[(int(row.mcc), int(row.mnc))]
        ax.plot(
            [row.home_timestamp, row.away_timestamp], [0, row.distance_km],
            color=color, linewidth=0.55, alpha=0.33, zorder=2,
        )
        ax.scatter(
            row.home_timestamp, 0, s=4.5, facecolor="white", edgecolor=color,
            linewidth=0.45, alpha=0.9, zorder=3,
        )
        ax.scatter(
            row.away_timestamp, row.distance_km, s=8.5, color=color,
            edgecolor="white", linewidth=0.25, alpha=0.9, zorder=4,
        )
    start = min(group.home_timestamp.min(), group.away_timestamp.min())
    end = max(group.home_timestamp.max(), group.away_timestamp.max())
    span = end - start
    margin = max(pd.Timedelta(minutes=7), span * 0.035)
    ax.set_xlim(start - margin, end + margin)
    ax.set_ylim(-max(4, group.distance_km.max() * 0.035), group.distance_km.max() * 1.10)
    ax.grid(color=GRID, linewidth=0.42, zorder=-1)
    ax.set_ylabel("Home-to-destination\ndisplacement (km)")
    ax.set_xlabel("Raw observation time (UTC)")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title("Nearest raw home/destination pairs", loc="left", pad=2)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=3.2,
               markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.25,
               label=f"{mcc}–{mnc}")
        for (mcc, mnc), color in colors.items()
    ]
    ax.legend(
        handles=handles, title="MCC–MNC", loc="upper left", ncol=len(handles),
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.78,
        fontsize=4.5, title_fontsize=4.5, handletextpad=0.2, columnspacing=0.6,
        borderpad=0.25,
    )


def render() -> None:
    configure_style()
    campaigns, groups = load_campaign_pairs()
    rings = load_world(WORLD)
    fig = plt.figure(figsize=(7.15, 9.15))
    grid = fig.add_gridspec(
        len(campaigns), 2, width_ratios=[0.93, 1.32],
        left=0.055, right=0.99, bottom=0.045, top=0.975,
        hspace=0.42, wspace=0.17,
    )
    for row_index, campaign in enumerate(campaigns.itertuples(index=False)):
        group = groups[campaign.campaign_id]
        colors = operator_colors(group)
        map_ax = fig.add_subplot(grid[row_index, 0])
        time_ax = fig.add_subplot(grid[row_index, 1])
        draw_map(map_ax, group, colors, rings)
        draw_timeline(time_ax, group, colors)
        identities = group[KEY].drop_duplicates().shape[0]
        operators = group[["mcc", "mnc"]].drop_duplicates().shape[0]
        row_title = (
            f"{campaign.campaign_id}  ·  {identities} identities; "
            f"max {int(campaign.maximum_daily_identities)}/day; "
            f"{operators} PLMNs; min |Δt| {int(group.gap_seconds.min())} s"
        )
        fig.text(
            0.055, map_ax.get_position().y1 + 0.019, row_title,
            ha="left", va="bottom", fontsize=7.2, fontweight="bold", color="#2f3134",
        )
    attribution = TILE_ATTRIBUTION[BASEMAP].replace(r"\copyright{}", "©")
    fig.text(0.99, 0.012, attribution, ha="right", fontsize=4.4, color=MUTED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    render()
