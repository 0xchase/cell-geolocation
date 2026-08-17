#!/usr/bin/env python3
"""Plot global counts of cell identities whose observed coordinates move."""

from __future__ import annotations

import argparse
import math
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helpers import COUNTRIES_GEOJSON, draw_geojson_layer
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"
CACHE_DIR = ROOT / ".cache" / "moving_tower_thresholds"

THRESHOLDS = [
    (">10 miles", 10.0, 16.0934),
    (">100 miles", 100.0, 160.934),
    (">1,000 miles", 1000.0, 1609.34),
    (">10,000 miles", 10000.0, 16093.4),
]

MAP_COLORS = {
    ">100 miles": "#d08c2f",
    ">1,000 miles": "#b23a48",
    ">10,000 miles": "#2f6f9f",
}

CONTINENTS = [
    ("North America", (-170, -50, 5, 75)),
    ("South America", (-90, -30, -58, 15)),
    ("Europe", (-25, 45, 34, 72)),
    ("Africa", (-20, 55, -36, 38)),
    ("Asia", (45, 180, -10, 78)),
    ("Oceania", (105, 180, -50, 5)),
]


def ch_csv(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def movement_query(lo: int, hi: int, mnc_lo: int | None = None, mnc_hi: int | None = None) -> str:
    mnc_filter = ""
    if mnc_lo is not None and mnc_hi is not None:
        mnc_filter = f"AND mnc BETWEEN {mnc_lo} AND {mnc_hi}"
    return f"""
    SELECT
        countIf(km > 16.0934) AS moved_gt_10_mi,
        countIf(km > 160.934) AS moved_gt_100_mi,
        countIf(km > 1609.34) AS moved_gt_1000_mi,
        countIf(km > 16093.4) AS moved_gt_10000_mi
    FROM
    (
        SELECT
            mcc,
            mnc,
            lac,
            cid,
            cell_type,
            greatCircleDistance(min(lon), min(lat), max(lon), max(lat)) / 1000 AS km
        FROM cell.geos
        WHERE cid > 0
          AND NOT (lat = 0 AND lon = 0)
          AND mcc BETWEEN {lo} AND {hi}
          {mnc_filter}
        GROUP BY mcc, mnc, lac, cid, cell_type
        HAVING count() > 1
    )
    """


def endpoint_query(lo: int, hi: int, mnc_lo: int | None = None, mnc_hi: int | None = None) -> str:
    mnc_filter = ""
    if mnc_lo is not None and mnc_hi is not None:
        mnc_filter = f"AND mnc BETWEEN {mnc_lo} AND {mnc_hi}"
    return f"""
    SELECT
        mcc,
        mnc,
        lac,
        cid,
        cell_type,
        argMin(lat, lon) AS lat_west,
        min(lon) AS lon_west,
        argMax(lat, lon) AS lat_east,
        max(lon) AS lon_east,
        greatCircleDistance(min(lon), min(lat), max(lon), max(lat)) / 1000 AS km
    FROM cell.geos
    WHERE cid > 0
      AND NOT (lat = 0 AND lon = 0)
      AND mcc BETWEEN {lo} AND {hi}
      {mnc_filter}
    GROUP BY mcc, mnc, lac, cid, cell_type
    HAVING count() > 1 AND km > 160.934
    """


def path_query(lo: int, hi: int, mnc_lo: int | None = None, mnc_hi: int | None = None) -> str:
    mnc_filter = ""
    if mnc_lo is not None and mnc_hi is not None:
        mnc_filter = f"AND mnc BETWEEN {mnc_lo} AND {mnc_hi}"
    return f"""
    WITH moving AS
    (
        SELECT
            mcc,
            mnc,
            lac,
            cid,
            cell_type,
            greatCircleDistance(min(lon), min(lat), max(lon), max(lat)) / 1000 AS km
        FROM cell.geos
        WHERE cid > 0
          AND NOT (lat = 0 AND lon = 0)
          AND mcc BETWEEN {lo} AND {hi}
          {mnc_filter}
        GROUP BY mcc, mnc, lac, cid, cell_type
        HAVING count() > 1 AND km > 1609.34
    )
    SELECT
        concat(toString(g.mcc), '/', toString(g.mnc), '/', toString(g.lac), '/', toString(g.cid), '/', toString(g.cell_type)) AS tower_id,
        g.mcc,
        g.mnc,
        g.lac,
        g.cid,
        g.cell_type,
        m.km,
        g.lat,
        g.lon,
        g.timestamp
    FROM cell.geos AS g
    INNER JOIN moving AS m USING (mcc, mnc, lac, cid, cell_type)
    WHERE g.cid > 0
      AND NOT (g.lat = 0 AND g.lon = 0)
      AND g.mcc BETWEEN {lo} AND {hi}
      {mnc_filter}
    ORDER BY tower_id, timestamp
    """


def query_range_adaptive(lo: int, hi: int, *, verbose: bool = False) -> pd.Series:
    """Query a disjoint MCC range, splitting if ClickHouse hits temp-file limits."""

    try:
        row = ch_csv(movement_query(lo, hi)).iloc[0]
        if verbose:
            print(f"queried MCC {lo}-{hi}")
        return row
    except subprocess.CalledProcessError:
        if lo == hi:
            return query_mnc_range_adaptive(lo, 0, 999, verbose=verbose)
        mid = (lo + hi) // 2
        left = query_range_adaptive(lo, mid, verbose=verbose)
        right = query_range_adaptive(mid + 1, hi, verbose=verbose)
        return left.add(right, fill_value=0)


def query_mnc_range_adaptive(mcc: int, lo: int, hi: int, *, verbose: bool = False) -> pd.Series:
    """Fallback for very dense MCCs that still exceed temp limits by themselves."""

    try:
        row = ch_csv(movement_query(mcc, mcc, lo, hi)).iloc[0]
        if verbose:
            print(f"queried MCC {mcc}, MNC {lo}-{hi}")
        return row
    except subprocess.CalledProcessError:
        if lo == hi:
            raise
        mid = (lo + hi) // 2
        left = query_mnc_range_adaptive(mcc, lo, mid, verbose=verbose)
        right = query_mnc_range_adaptive(mcc, mid + 1, hi, verbose=verbose)
        return left.add(right, fill_value=0)


def query_endpoints_range_adaptive(lo: int, hi: int, *, verbose: bool = False) -> pd.DataFrame:
    try:
        df = ch_csv(endpoint_query(lo, hi))
        if verbose:
            print(f"queried endpoint MCC {lo}-{hi}")
        return df
    except subprocess.CalledProcessError:
        if lo == hi:
            return query_endpoints_mnc_range_adaptive(lo, 0, 999, verbose=verbose)
        mid = (lo + hi) // 2
        left = query_endpoints_range_adaptive(lo, mid, verbose=verbose)
        right = query_endpoints_range_adaptive(mid + 1, hi, verbose=verbose)
        return pd.concat([left, right], ignore_index=True)


def query_endpoints_mnc_range_adaptive(mcc: int, lo: int, hi: int, *, verbose: bool = False) -> pd.DataFrame:
    try:
        df = ch_csv(endpoint_query(mcc, mcc, lo, hi))
        if verbose:
            print(f"queried endpoint MCC {mcc}, MNC {lo}-{hi}")
        return df
    except subprocess.CalledProcessError:
        if lo == hi:
            raise
        mid = (lo + hi) // 2
        left = query_endpoints_mnc_range_adaptive(mcc, lo, mid, verbose=verbose)
        right = query_endpoints_mnc_range_adaptive(mcc, mid + 1, hi, verbose=verbose)
        return pd.concat([left, right], ignore_index=True)


def query_paths_range_adaptive(lo: int, hi: int, *, verbose: bool = False) -> pd.DataFrame:
    try:
        df = ch_csv(path_query(lo, hi))
        if verbose:
            print(f"queried path MCC {lo}-{hi}")
        return df
    except subprocess.CalledProcessError:
        if lo == hi:
            return query_paths_mnc_range_adaptive(lo, 0, 999, verbose=verbose)
        mid = (lo + hi) // 2
        left = query_paths_range_adaptive(lo, mid, verbose=verbose)
        right = query_paths_range_adaptive(mid + 1, hi, verbose=verbose)
        return pd.concat([left, right], ignore_index=True)


def query_paths_mnc_range_adaptive(mcc: int, lo: int, hi: int, *, verbose: bool = False) -> pd.DataFrame:
    try:
        df = ch_csv(path_query(mcc, mcc, lo, hi))
        if verbose:
            print(f"queried path MCC {mcc}, MNC {lo}-{hi}")
        return df
    except subprocess.CalledProcessError:
        if lo == hi:
            raise
        mid = (lo + hi) // 2
        left = query_paths_mnc_range_adaptive(mcc, lo, mid, verbose=verbose)
        right = query_paths_mnc_range_adaptive(mcc, mid + 1, hi, verbose=verbose)
        return pd.concat([left, right], ignore_index=True)


def load_counts(verbose: bool = False) -> pd.DataFrame:
    rows = []
    for lo in range(1, 1000, 100):
        hi = min(lo + 99, 999)
        rows.append(query_range_adaptive(lo, hi, verbose=verbose))
    total = pd.concat(rows, axis=1).sum(axis=1)
    counts = {
        ">10 miles": int(total["moved_gt_10_mi"]),
        ">100 miles": int(total["moved_gt_100_mi"]),
        ">1,000 miles": int(total["moved_gt_1000_mi"]),
        ">10,000 miles": int(total["moved_gt_10000_mi"]),
    }
    return pd.DataFrame(
        {
            "threshold": [label for label, _, _ in THRESHOLDS],
            "miles": [miles for _, miles, _ in THRESHOLDS],
            "cell_identities": [counts[label] for label, _, _ in THRESHOLDS],
        }
    )


def load_endpoints(verbose: bool = False) -> pd.DataFrame:
    frames = []
    for lo in range(1, 1000, 100):
        hi = min(lo + 99, 999)
        frames.append(query_endpoints_range_adaptive(lo, hi, verbose=verbose))
    endpoints = pd.concat(frames, ignore_index=True)
    endpoints["miles"] = endpoints["km"] * 0.621371
    endpoints["threshold_class"] = pd.cut(
        endpoints["miles"],
        bins=[100, 1000, 10000, float("inf")],
        labels=[">100 miles", ">1,000 miles", ">10,000 miles"],
        right=True,
        include_lowest=False,
    ).astype(str)
    west = endpoints[["mcc", "mnc", "lac", "cid", "cell_type", "miles", "threshold_class", "lat_west", "lon_west"]].rename(
        columns={"lat_west": "lat", "lon_west": "lon"}
    )
    west["endpoint"] = "west"
    east = endpoints[["mcc", "mnc", "lac", "cid", "cell_type", "miles", "threshold_class", "lat_east", "lon_east"]].rename(
        columns={"lat_east": "lat", "lon_east": "lon"}
    )
    east["endpoint"] = "east"
    return pd.concat([west, east], ignore_index=True)


def load_paths(verbose: bool = False) -> pd.DataFrame:
    frames = []
    for lo in range(1, 1000, 100):
        hi = min(lo + 99, 999)
        frames.append(query_paths_range_adaptive(lo, hi, verbose=verbose))
    paths = pd.concat(frames, ignore_index=True)
    if paths.empty:
        return paths
    paths["timestamp"] = pd.to_datetime(paths["timestamp"])
    paths["miles"] = paths["km"] * 0.621371
    paths["threshold_class"] = pd.cut(
        paths["miles"],
        bins=[1000, 10000, float("inf")],
        labels=[">1,000 miles", ">10,000 miles"],
        right=True,
        include_lowest=False,
    ).astype(str)
    return paths


def cached_frame(
    name: str,
    builder,
    *,
    refresh: bool = False,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    path = CACHE_DIR / f"{name}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=parse_dates)
    df = builder()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def geographic_aspect(bbox: tuple[float, float, float, float]) -> float:
    """Approximate local lon/lat map proportions at the bbox midpoint."""

    mid_lat = (bbox[2] + bbox[3]) / 2
    return 1 / max(math.cos(math.radians(mid_lat)), 0.30)


def setup_world_axis(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    title: str,
    *,
    axis_labels: bool = True,
) -> None:
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(
        ax,
        COUNTRIES_GEOJSON,
        bbox,
        facecolor="#f5f1e8",
        edgecolor="#8a8176",
        linewidth=0.28,
        alpha=1.0,
        zorder=0,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect(geographic_aspect(bbox), adjustable="box", anchor="C")
    ax.set_title(title, fontsize=9.4 if not axis_labels else 10.0)
    ax.set_xlabel("Longitude" if axis_labels else "")
    ax.set_ylabel("Latitude" if axis_labels else "")
    ax.tick_params(labelsize=8)
    ax.locator_params(axis="x", nbins=6)
    ax.locator_params(axis="y", nbins=5)
    ax.grid(True, linewidth=0.30, color="white", alpha=0.70)


def map_display_ratio(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[1] - bbox[0]) / ((bbox[3] - bbox[2]) * geographic_aspect(bbox))


def add_map_axis(
    fig: plt.Figure,
    bbox: tuple[float, float, float, float],
    title: str,
    *,
    center_x: float,
    bottom: float,
    height: float,
    axis_labels: bool = True,
) -> plt.Axes:
    fig_w, fig_h = fig.get_size_inches()
    width = map_display_ratio(bbox) * height * fig_h / fig_w
    ax = fig.add_axes([center_x - width / 2, bottom, width, height])
    setup_world_axis(ax, bbox, title, axis_labels=axis_labels)
    return ax


def in_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.Series:
    xmin, xmax, ymin, ymax = bbox
    return df["lon"].between(xmin, xmax) & df["lat"].between(ymin, ymax)


def draw_path_traces(
    ax: plt.Axes,
    paths: pd.DataFrame,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    line_alpha: float = 0.22,
    point_alpha: float = 0.55,
    linewidth: float = 0.45,
    point_size: float = 3.0,
) -> None:
    for _, group in paths.groupby("tower_id", sort=False):
        group = group.sort_values("timestamp")
        if bbox is None:
            visible = np.ones(len(group), dtype=bool)
        else:
            visible = in_bbox(group, bbox).to_numpy()
        if not visible.any():
            continue
        color = MAP_COLORS[">10,000 miles"] if group["miles"].iloc[0] > 10000 else MAP_COLORS[">1,000 miles"]
        lons = group["lon"].to_numpy(dtype=float)
        lats = group["lat"].to_numpy(dtype=float)
        plot_lons: list[float] = []
        plot_lats: list[float] = []
        for i, is_visible in enumerate(visible):
            if not is_visible:
                if plot_lons and not np.isnan(plot_lons[-1]):
                    plot_lons.append(np.nan)
                    plot_lats.append(np.nan)
                continue
            if i > 0 and (not visible[i - 1] or abs(lons[i] - lons[i - 1]) > 180):
                plot_lons.append(np.nan)
                plot_lats.append(np.nan)
            plot_lons.append(lons[i])
            plot_lats.append(lats[i])
        if np.count_nonzero(visible) > 1:
            ax.plot(plot_lons, plot_lats, color=color, alpha=line_alpha, linewidth=linewidth, zorder=2)
        ax.scatter(
            lons[visible],
            lats[visible],
            color=color,
            s=point_size,
            alpha=point_alpha,
            linewidth=0,
            zorder=3,
        )


def paths_touching_bbox(paths: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    xmin, xmax, ymin, ymax = bbox
    ids = paths[
        paths["lon"].between(xmin, xmax)
        & paths["lat"].between(ymin, ymax)
    ]["tower_id"].unique()
    return paths[paths["tower_id"].isin(ids)]


def make_plot(counts: pd.DataFrame, endpoints: pd.DataFrame, paths: pd.DataFrame, output: Path, preview: Path | None = None) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(15.8, 16.0), constrained_layout=False)
    fig.suptitle("Cell identities with long-range coordinate movement", fontsize=15, fontweight="bold", y=0.992)

    ax = fig.add_axes([0.055, 0.825, 0.225, 0.115])
    sns.barplot(
        data=counts,
        x="threshold",
        y="cell_identities",
        hue="threshold",
        palette={
            ">10 miles": "#b23a48",
            ">100 miles": "#a94855",
            ">1,000 miles": "#8f4c60",
            ">10,000 miles": "#6f5065",
        },
        legend=False,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Movement threshold counts")
    ax.set_xlabel("Bounding-box movement")
    ax.set_ylabel("Cell identities (log scale)")
    ax.set_ylim(1, counts["cell_identities"].max() * 2.2)
    ax.set_xticks(range(len(counts)), ["10 mi", "100 mi", "1,000 mi", "10,000 mi"])

    for patch, value in zip(ax.patches, counts["cell_identities"], strict=False):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value * 1.08,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bbox = (-180, 180, -58, 78)
    ax = add_map_axis(
        fig,
        bbox,
        "Endpoint locations for identities moving >100 miles",
        center_x=0.515,
        bottom=0.825,
        height=0.115,
    )
    for label in [">100 miles", ">1,000 miles", ">10,000 miles"]:
        rows = endpoints[endpoints["threshold_class"] == label]
        if rows.empty:
            continue
        ax.scatter(
            rows["lon"],
            rows["lat"],
            s=14 if label == ">100 miles" else 24 if label == ">1,000 miles" else 46,
            color=MAP_COLORS[label],
            alpha=0.38 if label == ">100 miles" else 0.60 if label == ">1,000 miles" else 0.95,
            edgecolor="white",
            linewidth=0.20,
            label=f"{label} ({len(rows) // 2:,} IDs)",
            zorder=2 if label == ">100 miles" else 3,
        )
    ax.legend(title="Largest threshold crossed", loc="lower left", frameon=True, fontsize=8)

    world_10k = paths[paths["miles"] > 10000]
    ax = add_map_axis(
        fig,
        (-180, 180, -58, 78),
        f"Paths moving >10,000 miles\n({world_10k['tower_id'].nunique():,} IDs)",
        center_x=0.86,
        bottom=0.840,
        height=0.085,
    )
    draw_path_traces(ax, world_10k, line_alpha=0.68, point_alpha=0.95, linewidth=0.80, point_size=10)

    map_slots = [
        (0.265, 0.555, 0.200),
        (0.745, 0.555, 0.200),
        (0.265, 0.330, 0.200),
        (0.745, 0.330, 0.200),
        (0.265, 0.105, 0.200),
        (0.745, 0.105, 0.200),
    ]
    for i, ((continent, bbox), (center_x, bottom, height)) in enumerate(zip(CONTINENTS, map_slots, strict=True)):
        continent_paths = paths_touching_bbox(paths, bbox)
        ax = add_map_axis(
            fig,
            bbox,
            f"{continent}: paths moving >1,000 miles ({continent_paths['tower_id'].nunique():,} IDs)",
            center_x=center_x,
            bottom=bottom,
            height=height,
            axis_labels=False,
        )
        if i < 4:
            ax.tick_params(labelbottom=False)
        draw_path_traces(
            ax,
            continent_paths,
            bbox=bbox,
            line_alpha=0.24,
            point_alpha=0.58,
            linewidth=0.46,
            point_size=4,
        )

    handles = [
        plt.Line2D([0], [0], color=MAP_COLORS[">1,000 miles"], marker="o", linestyle="-", linewidth=1.0, markersize=4, label=">1,000 miles"),
        plt.Line2D([0], [0], color=MAP_COLORS[">10,000 miles"], marker="o", linestyle="-", linewidth=1.2, markersize=5, label=">10,000 miles"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncols=2, frameon=True, fontsize=8)
    fig.text(
        0.5,
        0.025,
        "Bars are cumulative. Path panels draw raw observations for identities whose min/max observed coordinate span exceeds the threshold; continent panels show only observations inside the map window.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "all_moving_tower_thresholds.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    counts = cached_frame("counts", lambda: load_counts(verbose=args.verbose), refresh=args.refresh_cache)
    print(counts.to_string(index=False))
    endpoints = cached_frame("endpoints", lambda: load_endpoints(verbose=args.verbose), refresh=args.refresh_cache)
    paths = cached_frame(
        "paths",
        lambda: load_paths(verbose=args.verbose),
        refresh=args.refresh_cache,
        parse_dates=["timestamp"],
    )
    print(f">1,000-mile path identities: {paths['tower_id'].nunique():,}")
    print(f">10,000-mile path identities: {paths[paths['miles'] > 10000]['tower_id'].nunique():,}")
    make_plot(counts, endpoints, paths, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
