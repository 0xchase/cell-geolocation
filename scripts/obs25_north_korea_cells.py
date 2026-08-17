#!/usr/bin/env python3
"""Analyze cell identities geolocated in North Korea."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helpers import ADMIN1_GEOJSON, COUNTRIES_GEOJSON, add_osm_basemap, draw_geojson_layer
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"

GROUP_ORDER = ["China MCC 460", "South Korea MCC 450", "North Korea MCC 467", "Other foreign MCC"]
GROUP_COLORS = {
    "China MCC 460": "#2f6f9f",
    "South Korea MCC 450": "#c9743a",
    "North Korea MCC 467": "#b23a48",
    "Other foreign MCC": "#6f5b7b",
}

COUNTRY_LABELS = {
    "中国": "China OSM polygon",
    "조선민주주의인민공화국": "North Korea OSM polygon",
    "대한민국": "South Korea OSM polygon",
    "": "Unresolved OSM polygon",
}

KP_GROUP_SQL = """
multiIf(
    mcc = 460, 'China MCC 460',
    mcc = 450, 'South Korea MCC 450',
    mcc = 467, 'North Korea MCC 467',
    'Other foreign MCC'
)
"""


def _geojson_rings_for_country(path: Path, country: str) -> list[list[list[float]]]:
    with open(path) as f:
        features = json.load(f)["features"]
    rings: list[list[list[float]]] = []
    for feature in features:
        props = feature.get("properties", {})
        values = {
            str(props.get("ISO_A2", "")),
            str(props.get("ADM0_A3", "")),
            str(props.get("ADM0_ISO", "")),
            str(props.get("iso_a2", "")),
            str(props.get("adm0_a3", "")),
            str(props.get("ADMIN", "")),
            str(props.get("admin", "")),
            str(props.get("NAME", "")),
            str(props.get("name", "")),
        }
        if country not in values:
            continue
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            rings.extend(geom["coordinates"])
        elif geom["type"] == "MultiPolygon":
            for polygon in geom["coordinates"]:
                rings.extend(polygon)
    return rings


def _point_segment_distance_km(lon: float, lat: float, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    xscale = 111.32 * math.cos(math.radians(lat))
    yscale = 110.57
    px, py = lon * xscale, lat * yscale
    ax, ay = lon1 * xscale, lat1 * yscale
    bx, by = lon2 * xscale, lat2 * yscale
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    cx, cy = ax + t * vx, ay + t * vy
    return math.hypot(px - cx, py - cy)


def _distance_to_boundary_km(lon: float, lat: float, rings: list[list[list[float]]]) -> float:
    best = float("inf")
    for ring in rings:
        for (lon1, lat1), (lon2, lat2) in zip(ring, ring[1:]):
            best = min(best, _point_segment_distance_km(lon, lat, lon1, lat1, lon2, lat2))
    return best


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_data() -> dict[str, pd.DataFrame]:
    summary = ch_df(
        """
        SELECT
            count() AS cells,
            sum(obs) AS obs,
            uniqExact(mcc) AS mccs,
            uniqExact((mcc,mnc)) AS operators,
            min(first_seen) AS first_seen,
            max(last_seen) AS last_seen,
            countIf(mcc = 460) AS china_cells,
            countIf(mcc = 450) AS south_korea_cells,
            countIf(mcc = 467) AS north_korea_cells,
            countIf(mcc NOT IN (460,450,467)) AS other_cells
        FROM cell.summary_full
        WHERE country_iso = 'KP'
        """
    )

    points = ch_df(
        f"""
        SELECT
            {KP_GROUP_SQL} AS group_name,
            mcc, mnc, lac, cid, cell_type,
            obs,
            first_seen,
            last_seen,
            glat AS lat,
            glon AS lon,
            country_osm,
            region,
            city
        FROM cell.summary_full
        WHERE country_iso = 'KP'
        ORDER BY group_name, lat, lon
        """
    )
    kp_rings = _geojson_rings_for_country(COUNTRIES_GEOJSON, "KP")
    points["boundary_km"] = [
        _distance_to_boundary_km(float(lon), float(lat), kp_rings)
        for lon, lat in zip(points["lon"], points["lat"], strict=False)
    ]

    composition = ch_df(
        f"""
        SELECT
            concat(toString(mcc), '/', toString(mnc), ' ', toString(cell_type)) AS operator,
            {KP_GROUP_SQL} AS group_name,
            count() AS cells,
            sum(obs) AS obs,
            min(first_seen) AS first_seen,
            max(last_seen) AS last_seen
        FROM cell.summary_full
        WHERE country_iso = 'KP'
        GROUP BY operator, group_name
        ORDER BY cells DESC
        """
    )

    grid = ch_df(
        f"""
        SELECT
            round(glat, 2) AS lat,
            round(glon, 2) AS lon,
            {KP_GROUP_SQL} AS group_name,
            count() AS cells,
            sum(obs) AS obs
        FROM cell.summary_full
        WHERE country_iso = 'KP'
        GROUP BY lat, lon, group_name
        HAVING cells >= 2
        ORDER BY cells DESC
        """
    )

    osm_split = ch_df(
        f"""
        SELECT
            country_osm,
            {KP_GROUP_SQL} AS group_name,
            count() AS cells,
            sum(obs) AS obs
        FROM cell.summary_full
        WHERE country_iso = 'KP'
        GROUP BY country_osm, group_name
        ORDER BY cells DESC
        """
    )

    monthly = ch_df(
        f"""
        WITH nk AS (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE country_iso = 'KP'
        )
        SELECT
            toStartOfMonth(g.timestamp) AS month,
            {KP_GROUP_SQL.replace('mcc', 'g.mcc')} AS group_name,
            count() AS obs,
            uniqExact((g.mcc,g.mnc,g.lac,g.cid,g.cell_type)) AS cells
        FROM cell.geos AS g
        INNER JOIN nk USING (mcc,mnc,lac,cid,cell_type)
        GROUP BY month, group_name
        ORDER BY month, group_name
        """
    )

    tracks = ch_df(
        f"""
        WITH nk AS (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE country_iso = 'KP'
        )
        SELECT
            {KP_GROUP_SQL.replace('mcc', 'g.mcc')} AS group_name,
            concat(toString(g.mcc), '/', toString(g.mnc), '/', toString(g.lac), '/', toString(g.cid), ' ', toString(g.cell_type)) AS tower_id,
            g.mcc, g.mnc, g.lac, g.cid, g.cell_type,
            count() AS raw_obs,
            min(g.timestamp) AS first_seen,
            max(g.timestamp) AS last_seen,
            min(g.lat) AS min_lat,
            max(g.lat) AS max_lat,
            min(g.lon) AS min_lon,
            max(g.lon) AS max_lon,
            greatCircleDistance(min(g.lon), min(g.lat), max(g.lon), max(g.lat)) / 1000 AS bbox_km,
            countIf(g.lat BETWEEN 37.6 AND 43.2 AND g.lon BETWEEN 124.0 AND 131.2) AS obs_korea_bbox,
            countIf(NOT (g.lat BETWEEN 37.6 AND 43.2 AND g.lon BETWEEN 124.0 AND 131.2)) AS obs_outside_korea_bbox
        FROM cell.geos AS g
        INNER JOIN nk USING (mcc,mnc,lac,cid,cell_type)
        WHERE NOT (g.lat = 0 AND g.lon = 0)
        GROUP BY group_name, tower_id, g.mcc, g.mnc, g.lac, g.cid, g.cell_type
        ORDER BY bbox_km DESC
        """
    )

    for df, col in [(points, "group_name"), (composition, "group_name"), (grid, "group_name"), (osm_split, "group_name"), (monthly, "group_name"), (tracks, "group_name")]:
        df[col] = pd.Categorical(df[col], GROUP_ORDER, ordered=True)
    points["first_seen"] = pd.to_datetime(points["first_seen"])
    points["last_seen"] = pd.to_datetime(points["last_seen"])
    composition["first_seen"] = pd.to_datetime(composition["first_seen"])
    composition["last_seen"] = pd.to_datetime(composition["last_seen"])
    monthly["month"] = pd.to_datetime(monthly["month"])
    tracks["first_seen"] = pd.to_datetime(tracks["first_seen"])
    tracks["last_seen"] = pd.to_datetime(tracks["last_seen"])
    osm_split["osm_label"] = osm_split["country_osm"].fillna("").map(lambda v: COUNTRY_LABELS.get(v, str(v) if str(v) else "Unresolved OSM polygon"))

    return {
        "summary": summary,
        "points": points,
        "composition": composition,
        "grid": grid,
        "osm_split": osm_split,
        "monthly": monthly,
        "tracks": tracks,
    }


def setup_map(ax: plt.Axes, bbox: tuple[float, float, float, float], *, zoom: int | None = None) -> None:
    ax.set_facecolor("#dceaf2")
    used_osm = False
    if zoom is not None:
        used_osm = add_osm_basemap(ax, bbox, zoom=zoom, alpha=0.58, grayscale=True)
    if not used_osm:
        draw_geojson_layer(
            ax,
            COUNTRIES_GEOJSON,
            bbox,
            facecolor="#f5f1e8",
            edgecolor="#756d63",
            linewidth=0.45,
            zorder=0,
        )
    draw_geojson_layer(
        ax,
        COUNTRIES_GEOJSON,
        bbox,
        countries={"KP"},
        facecolor="#f4d6b6",
        edgecolor="#3f3831",
        linewidth=0.85,
        alpha=0.46 if used_osm else 0.74,
        zorder=1.2,
    )
    draw_geojson_layer(
        ax,
        COUNTRIES_GEOJSON,
        bbox,
        countries={"CN", "KR", "RU", "JP"},
        facecolor="none" if used_osm else "#f5f1e8",
        edgecolor="#5f574f",
        linewidth=0.55,
        alpha=0.95,
        zorder=1.1,
    )
    draw_geojson_layer(
        ax,
        ADMIN1_GEOJSON,
        bbox,
        countries={"KP", "CN", "KR", "RU"},
        facecolor="none",
        edgecolor="#92887b",
        linewidth=0.26,
        alpha=0.72,
        zorder=1.3,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, color="#ffffff", alpha=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if used_osm:
        ax.text(0.01, 0.01, "Map data (c) OpenStreetMap contributors", transform=ax.transAxes, fontsize=5.4, color="#555", zorder=20)


def add_city_labels(ax: plt.Axes, labels: list[tuple[str, float, float]], *, fontsize: float = 6.8) -> None:
    for text, lon, lat in labels:
        ax.text(
            lon,
            lat,
            text,
            fontsize=fontsize,
            ha="center",
            va="center",
            color="#2f2a25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            zorder=8,
        )


def draw_group_scatter(ax: plt.Axes, data: pd.DataFrame, *, size_scale: float = 5.5, alpha: float = 0.86, edge: float = 0.25) -> None:
    for group in GROUP_ORDER:
        part = data[data["group_name"] == group]
        if part.empty:
            continue
        size = 10 + size_scale * np.sqrt(part["cells"].astype(float))
        ax.scatter(
            part["lon"],
            part["lat"],
            s=size,
            color=GROUP_COLORS[group],
            alpha=alpha,
            edgecolor="white",
            linewidth=edge,
            zorder=4 if group != "North Korea MCC 467" else 5,
        )


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    summary = data["summary"].iloc[0]
    points = data["points"].copy()
    grid = data["grid"].copy()
    composition = data["composition"].copy()
    monthly = data["monthly"].copy()
    osm_split = data["osm_split"].copy()
    tracks = data["tracks"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.03)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
        }
    )

    fig = plt.figure(figsize=(14.8, 11.8), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.22, 1.0], height_ratios=[1.15, 0.95, 1.05])
    fig.suptitle(
        "North Korea-tagged cell identities are mostly border spillover, with sparse native DPRK observations",
        fontsize=14.2,
        fontweight="bold",
    )

    ax_map = fig.add_subplot(gs[0, 0])
    setup_map(ax_map, (123.7, 131.35, 37.5, 43.25), zoom=6)
    draw_group_scatter(ax_map, grid, size_scale=6.0, alpha=0.88)
    add_city_labels(
        ax_map,
        [
            ("Sinuiju", 124.42, 40.09),
            ("Pyongyang", 125.74, 39.03),
            ("Hyesan / Baishan", 128.20, 41.38),
            ("Samjiyon", 128.17, 41.97),
            ("Rason", 130.30, 42.25),
            ("DMZ / Kaesong", 126.62, 37.97),
        ],
    )
    ax_map.set_title(
        f"A. {int(summary['cells']):,} KP-tagged identities cluster on China and DMZ borders"
    )
    legend_items = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GROUP_COLORS[g], markeredgecolor="white", markersize=6.5, label=g)
        for g in GROUP_ORDER
    ]
    ax_map.legend(handles=legend_items, loc="lower right", frameon=True, fontsize=7.2, title="", borderpad=0.45)

    ax_comp = fig.add_subplot(gs[0, 1])
    top_comp = composition.sort_values("cells", ascending=False).head(18).copy()
    top_comp = top_comp.sort_values("cells", ascending=True)
    ax_comp.barh(top_comp["operator"], top_comp["cells"], color=[GROUP_COLORS[str(g)] for g in top_comp["group_name"]])
    ax_comp.set_xscale("log")
    ax_comp.set_xlabel("Distinct cell identities, log scale")
    ax_comp.set_ylabel("")
    ax_comp.set_title(
        f"B. Composition: {int(summary['china_cells']):,} Chinese, {int(summary['south_korea_cells']):,} South Korean, {int(summary['north_korea_cells']):,} native DPRK"
    )
    xmax = max(top_comp["cells"]) * 5.5
    ax_comp.set_xlim(0.75, xmax)
    for patch, cells, obs in zip(ax_comp.patches, top_comp["cells"], top_comp["obs"], strict=False):
        ax_comp.text(cells * 1.13, patch.get_y() + patch.get_height() / 2, f"{int(cells):,} IDs; {int(obs):,} obs", va="center", fontsize=6.8)
    patches = [Patch(facecolor=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    ax_comp.legend(handles=patches, loc="lower right", frameon=True, fontsize=7.0)

    ax_time = fig.add_subplot(gs[1, 0])
    for group in GROUP_ORDER:
        part = monthly[monthly["group_name"] == group].sort_values("month")
        if part.empty:
            continue
        ax_time.plot(
            part["month"],
            part["obs"],
            marker="o",
            markersize=3.6,
            linewidth=1.6,
            color=GROUP_COLORS[group],
            label=group,
        )
    ax_time.set_yscale("log")
    ax_time.set_ylim(0.8, max(5000, monthly["obs"].max() * 1.35))
    ax_time.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_time.tick_params(axis="x", rotation=35)
    ax_time.set_xlabel("")
    ax_time.set_ylabel("Raw observations, log scale")
    ax_time.set_title("C. Observation time series: Chinese border IDs are steady, no surge")
    ax_time.legend(loc="upper left", frameon=True, fontsize=7.0)

    ax_osm = fig.add_subplot(gs[1, 1])
    osm_totals = osm_split.groupby("osm_label", observed=True)["cells"].sum().sort_values(ascending=True)
    y_labels = list(osm_totals.index)
    left = np.zeros(len(y_labels))
    y_pos = np.arange(len(y_labels))
    for group in GROUP_ORDER:
        vals = (
            osm_split[osm_split["group_name"] == group]
            .set_index("osm_label")["cells"]
            .reindex(y_labels, fill_value=0)
            .astype(float)
        )
        ax_osm.barh(y_pos, vals, left=left, color=GROUP_COLORS[group], label=group)
        left += vals.to_numpy()
    ax_osm.set_yticks(y_pos)
    ax_osm.set_yticklabels(y_labels)
    ax_osm.set_xlabel("Distinct cell identities")
    ax_osm.set_title("D. `country_iso=KP` disagrees with OSM labels at border edges")
    ax_osm.set_xlim(0, max(left) * 1.22)
    for y, total in zip(y_pos, left, strict=False):
        ax_osm.text(total + max(left) * 0.015, y, f"{int(total):,}", va="center", fontsize=7.0)
    ax_osm.legend(loc="lower right", frameon=True, fontsize=6.8)

    ax_native = fig.add_subplot(gs[2, 0])
    setup_map(ax_native, (123.95, 130.95, 39.55, 42.55), zoom=7)
    native_points = points[points["group_name"].astype(str) == "North Korea MCC 467"].copy()
    native_kp_osm = native_points[native_points["country_osm"] == "조선민주주의인민공화국"]
    native_edge_osm = native_points[native_points["country_osm"] != "조선민주주의인민공화국"]
    china_grid = grid[grid["group_name"] == "China MCC 460"]
    if not china_grid.empty:
        ax_native.scatter(
            china_grid["lon"],
            china_grid["lat"],
            s=8 + 1.8 * np.sqrt(china_grid["cells"].astype(float)),
            color=GROUP_COLORS["China MCC 460"],
            alpha=0.20,
            edgecolor="none",
            zorder=3,
        )
    ax_native.scatter(
        native_kp_osm["lon"],
        native_kp_osm["lat"],
        s=33,
        color=GROUP_COLORS["North Korea MCC 467"],
        alpha=0.94,
        edgecolor="white",
        linewidth=0.45,
        zorder=6,
        label="MCC 467 in DPRK OSM polygon",
    )
    ax_native.scatter(
        native_edge_osm["lon"],
        native_edge_osm["lat"],
        s=33,
        color=GROUP_COLORS["North Korea MCC 467"],
        alpha=0.68,
        marker="^",
        edgecolor="white",
        linewidth=0.45,
        zorder=6,
        label="MCC 467 on polygon edge",
    )
    add_city_labels(
        ax_native,
        [
            ("Sinuiju\n467 GSM", 124.42, 40.16),
            ("Samjiyon\n467 GSM/LTE", 128.25, 42.08),
            ("Rason\n467 LTE", 130.48, 42.32),
        ],
        fontsize=6.7,
    )
    native_count = int(summary["north_korea_cells"])
    native_obs = int(data["points"].loc[data["points"]["group_name"] == "North Korea MCC 467", "obs"].sum())
    native_max_boundary = float(native_points["boundary_km"].max()) if not native_points.empty else 0.0
    ax_native.text(
        0.02,
        0.98,
        f"{len(native_kp_osm):,} MCC 467 IDs fall in the DPRK OSM polygon\n"
        f"{len(native_edge_osm):,} more sit on polygon-edge labels\n"
        f"max distance from DPRK boundary: {native_max_boundary:.1f} km",
        transform=ax_native.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "#d0cbc4", "alpha": 0.86, "pad": 2.6},
        zorder=9,
    )
    ax_native.legend(loc="lower right", frameon=True, fontsize=6.7)
    ax_native.set_title(f"E. Non-spillover evidence: native MCC 467 has {native_count:,} IDs / {native_obs:,} observations")

    ax_boundary = fig.add_subplot(gs[2, 1])
    rng = np.random.default_rng(42)
    for y, group in enumerate(GROUP_ORDER):
        part = points[points["group_name"].astype(str) == group]
        if part.empty:
            continue
        jitter = rng.uniform(-0.20, 0.20, len(part))
        ax_boundary.scatter(
            part["boundary_km"],
            y + jitter,
            s=5 if len(part) > 1000 else 18,
            color=GROUP_COLORS[group],
            alpha=0.16 if len(part) > 1000 else 0.62,
            edgecolor="none",
            rasterized=True,
            zorder=3,
        )
        ax_boundary.text(
            10.25,
            y,
            f"n={len(part):,}; max {part['boundary_km'].max():.1f} km",
            va="center",
            fontsize=6.8,
        )
    foreign_inland = points[(points["mcc"] != 467) & (points["boundary_km"] >= 10.0)]
    all_inland = points[points["boundary_km"] >= 10.0]
    stable_lt10 = int((tracks["bbox_km"] <= 10).sum())
    moved_gt100 = int((tracks["bbox_km"] > 100).sum())
    ax_boundary.axvline(10.0, color="#333333", linestyle="--", linewidth=1.0)
    ax_boundary.text(
        0.02,
        0.98,
        f"Foreign-MCC IDs >=10 km inland: {len(foreign_inland):,}\n"
        f"All KP-tagged IDs >=10 km inland: {len(all_inland):,}\n"
        f"Track sanity: {stable_lt10:,}/{len(tracks):,} stay within 10 km; {moved_gt100} jump >100 km",
        transform=ax_boundary.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#d0cbc4", "alpha": 0.88, "pad": 3.0},
        zorder=8,
    )
    ax_boundary.set_yticks(range(len(GROUP_ORDER)))
    ax_boundary.set_yticklabels(GROUP_ORDER)
    ax_boundary.set_xlim(-0.15, 13.4)
    ax_boundary.set_ylim(-0.65, len(GROUP_ORDER) - 0.35)
    ax_boundary.grid(True, axis="x", linewidth=0.35, alpha=0.7)
    ax_boundary.grid(False, axis="y")
    ax_boundary.set_xlabel("Distance to DPRK boundary (km)")
    ax_boundary.set_ylabel("")
    ax_boundary.set_title("F. Boundary-distance check: no foreign-MCC inland candidates")
    ax_boundary.text(
        10.05,
        -0.52,
        "10 km inland threshold",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=6.7,
        color="#333333",
    )

    fig.text(
        0.5,
        0.005,
        "Interpretation: foreign MCC rows are border-adjacent and should be treated as spillover/artifact candidates; "
        "the only positive non-spillover signal is sparse native DPRK MCC 467, which is still observed near border cities rather than inland.",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.965), w_pad=2.4, h_pad=2.4)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs25_north_korea_cells.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    data = load_data()
    make_figure(data, args.output, args.preview)
    summary = data["summary"].iloc[0]
    tracks = data["tracks"]
    print(
        f"{args.output}\n"
        f"KP-tagged identities: {int(summary['cells']):,}; observations: {int(summary['obs']):,}; "
        f"native MCC 467: {int(summary['north_korea_cells']):,}; "
        f"moved >100 km: {int((tracks['bbox_km'] > 100).sum()):,}"
    )


if __name__ == "__main__":
    main()
