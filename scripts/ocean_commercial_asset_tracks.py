#!/usr/bin/env python3
"""Plot timestamped offshore tracks for selected candidate commercial assets."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from shapely.geometry import box, shape


ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
DEFAULT_OUTPUT = ROOT / "paper" / "figs" / "ocean_commercial_asset_tracks.pdf"
HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
CH_PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")
EARTH_KM = 6371.0088


@dataclass(frozen=True)
class TrackCase:
    label: str
    title: str
    subtitle: str
    mcc: int
    mnc: int
    lac: int
    cids: tuple[int, ...]
    start: str
    end: str


CASES = (
    TrackCase(
        "a",
        "Mirs Bay",
        "460–0 · CID 221388104 · Jun–Jul 2024",
        460,
        0,
        9559,
        (221388104,),
        "2024-06-01",
        "2024-08-01",
    ),
    TrackCase(
        "b",
        "South China Sea",
        "460–0 · paired CIDs · Jul–Sep 2025",
        460,
        0,
        9945,
        (113443649, 113443650),
        "2025-07-01",
        "2025-10-01",
    ),
    TrackCase(
        "c",
        "Taiwan Strait",
        "466–92 · CID 132597013 · Nov–Dec 2024",
        466,
        92,
        25300,
        (132597013,),
        "2024-11-01",
        "2025-01-01",
    ),
    TrackCase(
        "d",
        "Bohai Sea near Dalian",
        "460–0 · CID 3822338 · Oct–Dec 2025",
        460,
        0,
        16666,
        (3822338,),
        "2025-10-01",
        "2026-01-01",
    ),
)

REGIONAL_EXTENT = (104.0, 128.0, 17.5, 39.5)


def query_daily_positions() -> pd.DataFrame:
    predicates = []
    for case in CASES:
        cid_list = ",".join(str(cid) for cid in case.cids)
        predicates.append(
            f"(mcc={case.mcc} AND mnc={case.mnc} AND lac={case.lac} "
            f"AND cid IN ({cid_list}) AND timestamp >= '{case.start}' "
            f"AND timestamp < '{case.end}')"
        )
    sql = f"""
SELECT
    toDate(timestamp) AS day,
    mcc, mnc, lac, cid,
    median(lat) AS lat,
    median(lon) AS lon,
    count() AS observations
FROM cell.geos
WHERE {' OR '.join(predicates)}
GROUP BY day, mcc, mnc, lac, cid
ORDER BY mcc, mnc, lac, cid, day
FORMAT CSVWithNames
""".strip()
    command = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        HOST,
        f"clickhouse-client --password {CH_PASSWORD} --readonly 1 --max_threads 4",
    ]
    proc = subprocess.run(command, input=sql, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"Read-only ClickHouse query failed: {proc.stderr.strip()}")
    data = pd.read_csv(io.StringIO(proc.stdout))
    data["day"] = pd.to_datetime(data["day"])
    return data


def load_land() -> list:
    collection = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    return [shape(feature["geometry"]) for feature in collection["features"]]


def case_rows(data: pd.DataFrame, case: TrackCase) -> pd.DataFrame:
    return data[
        (data.mcc == case.mcc)
        & (data.mnc == case.mnc)
        & (data.lac == case.lac)
        & data.cid.isin(case.cids)
    ].copy()


def haversine_path_km(rows: pd.DataFrame) -> float:
    if len(rows) < 2:
        return 0.0
    radians = np.radians(rows[["lat", "lon"]].to_numpy())
    delta = np.diff(radians, axis=0)
    a = np.sin(delta[:, 0] / 2) ** 2
    a += np.cos(radians[:-1, 0]) * np.cos(radians[1:, 0]) * np.sin(delta[:, 1] / 2) ** 2
    return float(np.sum(2 * EARTH_KM * np.arcsin(np.sqrt(a))))


def padded_extent(rows: pd.DataFrame) -> tuple[float, float, float, float]:
    lon_min, lon_max = rows.lon.min(), rows.lon.max()
    lat_min, lat_max = rows.lat.min(), rows.lat.max()
    lon_pad = max((lon_max - lon_min) * 0.12, 0.008)
    lat_pad = max((lat_max - lat_min) * 0.12, 0.008)
    lon_min, lon_max = lon_min - lon_pad, lon_max + lon_pad
    lat_min, lat_max = lat_min - lat_pad, lat_max + lat_pad
    center_lat = (lat_min + lat_max) / 2
    lon_km = (lon_max - lon_min) * math.cos(math.radians(center_lat))
    lat_km = lat_max - lat_min
    if lon_km < lat_km:
        target_lon_span = lat_km / math.cos(math.radians(center_lat))
        center_lon = (lon_min + lon_max) / 2
        lon_min, lon_max = center_lon - target_lon_span / 2, center_lon + target_lon_span / 2
    else:
        target_lat_span = lon_km
        center_lat = (lat_min + lat_max) / 2
        lat_min, lat_max = center_lat - target_lat_span / 2, center_lat + target_lat_span / 2
    return lon_min, lon_max, lat_min, lat_max


def add_scale_bar(ax, extent: tuple[float, float, float, float]) -> None:
    lon_min, lon_max, lat_min, lat_max = extent
    width_km = (lon_max - lon_min) * 111.0 * math.cos(math.radians((lat_min + lat_max) / 2))
    candidates = (1, 2, 5, 10, 20, 50)
    scale_km = max(value for value in candidates if value <= width_km * 0.28)
    scale_lon = scale_km / (111.0 * math.cos(math.radians((lat_min + lat_max) / 2)))
    x0 = lon_min + 0.055 * (lon_max - lon_min)
    y0 = lat_min + 0.065 * (lat_max - lat_min)
    ax.plot([x0, x0 + scale_lon], [y0, y0], color="#303030", linewidth=1.3, zorder=8)
    ax.text(
        x0 + scale_lon / 2,
        y0 + 0.017 * (lat_max - lat_min),
        f"{scale_km} km",
        fontsize=6.4,
        ha="center",
        va="bottom",
        color="#303030",
        zorder=8,
    )


def render(data: pd.DataFrame, output: Path) -> None:
    plate = ccrs.PlateCarree()
    land = load_land()
    fig = plt.figure(figsize=(7.15, 4.25))
    outer_grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.08, 2),
        left=0.045,
        right=0.970,
        top=0.91,
        bottom=0.13,
        wspace=0.20,
    )
    quad_grid = outer_grid[0, 1].subgridspec(2, 2, hspace=0.24, wspace=0.17)
    regional_ax = fig.add_subplot(outer_grid[0, 0], projection=plate)
    axes = [
        fig.add_subplot(quad_grid[0, 0], projection=plate),
        fig.add_subplot(quad_grid[0, 1], projection=plate),
        fig.add_subplot(quad_grid[1, 0], projection=plate),
        fig.add_subplot(quad_grid[1, 1], projection=plate),
    ]
    cmap = mpl.colormaps["viridis"]
    norm = Normalize(0, 1)

    regional_ax.set_extent(REGIONAL_EXTENT, crs=plate)
    regional_ax.set_facecolor("#dceaf2")
    regional_bounds = box(REGIONAL_EXTENT[0], REGIONAL_EXTENT[2], REGIONAL_EXTENT[1], REGIONAL_EXTENT[3])
    regional_land = [geometry for geometry in land if geometry.intersects(regional_bounds)]
    regional_ax.add_geometries(
        regional_land,
        crs=plate,
        facecolor="#f3f0e7",
        edgecolor="#68625b",
        linewidth=0.45,
        zorder=0,
    )
    regional_ax.gridlines(
        crs=plate,
        draw_labels=False,
        linewidth=0.28,
        color="white",
        alpha=0.9,
    )
    regional_ax.text(
        119.0,
        23.7,
        "Taiwan",
        transform=plate,
        fontsize=5.7,
        fontweight="bold",
        color="#4d4943",
        zorder=3,
    )
    regional_ax.text(
        111.0,
        32.0,
        "China",
        transform=plate,
        fontsize=5.7,
        fontweight="bold",
        color="#4d4943",
        zorder=3,
    )
    for case in CASES:
        rows = case_rows(data, case)
        marker_lon = float(rows.lon.mean())
        marker_lat = float(rows.lat.mean())
        regional_ax.scatter(
            marker_lon,
            marker_lat,
            s=28,
            marker="o",
            facecolor="#b23a48",
            edgecolor="white",
            linewidth=0.7,
            transform=plate,
            zorder=4,
        )
        regional_ax.text(
            marker_lon + 0.34,
            marker_lat + 0.30,
            f"({case.label})",
            transform=plate,
            fontsize=6.5,
            fontweight="bold",
            color="#8f2736",
            ha="left",
            va="bottom",
            zorder=5,
        )
    regional_ax.spines["geo"].set_edgecolor("#68625b")
    regional_ax.spines["geo"].set_linewidth(0.55)

    for ax, case in zip(axes, CASES, strict=True):
        rows = case_rows(data, case)
        extent = padded_extent(rows)
        ax.set_extent(extent, crs=plate)
        ax.set_facecolor("#dceaf2")
        bounds = box(extent[0], extent[2], extent[1], extent[3])
        local_land = [geometry for geometry in land if geometry.intersects(bounds)]
        if local_land:
            ax.add_geometries(
                local_land,
                crs=plate,
                facecolor="#f3f0e7",
                edgecolor="#777067",
                linewidth=0.4,
                zorder=0,
            )

        gridlines = ax.gridlines(
            crs=plate,
            draw_labels=False,
            linewidth=0.35,
            color="white",
            alpha=0.95,
        )
        ax.set_xticks([extent[0], extent[1]], crs=plate)
        ax.set_yticks([extent[2], extent[3]], crs=plate)
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".3f"))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".3f"))
        ax.tick_params(axis="both", labelsize=6.2, length=2.2, pad=1.5)

        start = pd.Timestamp(case.start)
        end = pd.Timestamp(case.end)
        for _, track in rows.groupby("cid"):
            track = track.sort_values("day")
            xy = track[["lon", "lat"]].to_numpy()
            time_fraction = ((track.day - start) / (end - start)).to_numpy(dtype=float)
            if len(xy) > 1:
                segments = np.stack([xy[:-1], xy[1:]], axis=1)
                collection = LineCollection(
                    segments,
                    colors="#60666a",
                    linewidths=0.75,
                    alpha=0.66,
                    transform=plate,
                    zorder=3,
                )
                ax.add_collection(collection)
            ax.scatter(
                track.lon,
                track.lat,
                c=time_fraction,
                cmap=cmap,
                norm=norm,
                s=12,
                edgecolor="white",
                linewidth=0.25,
                transform=plate,
                zorder=4,
            )

        total_path = sum(haversine_path_km(track.sort_values("day")) for _, track in rows.groupby("cid"))
        ax.text(
            0.025,
            0.975,
            f"{len(rows)} daily positions · {total_path:.0f} km traced",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
            zorder=7,
        )
        add_scale_bar(ax, extent)
        ax.spines["geo"].set_edgecolor("#68625b")
        ax.spines["geo"].set_linewidth(0.55)

    regional_position = regional_ax.get_position()
    fig.text(
        regional_position.x0 + regional_position.width / 2,
        regional_position.y1 + 0.060,
        "Regional context",
        ha="center",
        va="top",
        fontsize=7.1,
        fontweight="bold",
    )
    fig.text(
        regional_position.x0 + regional_position.width / 2,
        regional_position.y1 + 0.026,
        "Locations of detailed tracks",
        ha="center",
        va="top",
        fontsize=6.2,
    )
    for ax, case in zip(axes, CASES, strict=True):
        position = ax.get_position()
        fig.text(
            position.x0 + position.width / 2,
            position.y1 + 0.060,
            f"({case.label}) {case.title}",
            ha="center",
            va="top",
            fontsize=7.1,
            fontweight="bold",
        )
        fig.text(
            position.x0 + position.width / 2,
            position.y1 + 0.026,
            case.subtitle,
            ha="center",
            va="top",
            fontsize=5.8,
        )
    quad_left = min(ax.get_position().x0 for ax in axes)
    quad_right = max(ax.get_position().x1 for ax in axes)
    color_ax = fig.add_axes([quad_left + 0.16 * (quad_right - quad_left), 0.072, 0.68 * (quad_right - quad_left), 0.012])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal")
    colorbar.set_ticks([0, 1], labels=["Earlier", "Later"])
    colorbar.ax.tick_params(labelsize=6.4, length=2)
    colorbar.set_label("Observation date within each panel", fontsize=6.5, labelpad=-1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=360)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)
    print(f"[figure] {output}")
    print(f"[preview] {output.with_suffix('.png')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    render(query_daily_positions(), args.output)


if __name__ == "__main__":
    main()
