#!/usr/bin/env python3
"""Plot defensible offshore identity relocations as standalone panels."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import box, shape


ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "figs"
EARTH_KM = 6371.0088


@dataclass(frozen=True)
class MovementCase:
    slug: str
    title: str
    start_lat: float
    start_lon: float
    start_last_seen: str
    start_label: str
    start_observations: int
    end_lat: float
    end_lon: float
    end_first_seen: str
    end_label: str
    end_observations: int
    bbox: tuple[float, float, float, float]
    vessel_speed: bool = True


CASES = [
    MovementCase(
        slug="banglalink",
        title="Bangladesh 470–4 · CID 61252",
        start_lat=22.37,
        start_lon=90.12,
        start_last_seen="2025-02-06 17:37:48",
        start_label="3–6 Feb",
        start_observations=4,
        end_lat=19.54,
        end_lon=87.19,
        end_first_seen="2025-02-07 17:19:40",
        end_label="7 Feb",
        end_observations=1,
        bbox=(86.55, 90.55, 19.05, 22.85),
    ),
    MovementCase(
        slug="algeria_18413",
        title="Algeria 603–3 · CID 18413",
        start_lat=36.77,
        start_lon=5.66,
        start_last_seen="2025-09-28 16:28:56",
        start_label="28 Sep",
        start_observations=1,
        end_lat=40.80,
        end_lon=3.54,
        end_first_seen="2025-09-29 17:03:19",
        end_label="29–30 Sep",
        end_observations=2,
        bbox=(2.85, 6.20, 36.20, 41.25),
    ),
    MovementCase(
        slug="algeria_18611",
        title="Algeria 603–3 · CID 18611",
        start_lat=36.66,
        start_lon=5.46,
        start_last_seen="2025-09-28 16:03:23",
        start_label="25–28 Sep",
        start_observations=4,
        end_lat=40.80,
        end_lon=3.53,
        end_first_seen="2025-09-29 17:29:03",
        end_label="29–30 Sep",
        end_observations=2,
        bbox=(2.85, 6.20, 36.20, 41.25),
    ),
    MovementCase(
        slug="china_nr_relocation",
        title="China Mobile NR · CID 43168731137",
        start_lat=31.09,
        start_lon=122.97,
        start_last_seen="2025-11-13 19:47:07",
        start_label="Aug–Nov 2025",
        start_observations=142,
        end_lat=32.79,
        end_lon=121.37,
        end_first_seen="2026-03-23 04:59:27",
        end_label="Mar–Jun 2026",
        end_observations=151,
        bbox=(120.75, 123.55, 30.45, 33.25),
        vessel_speed=False,
    ),
]


def world_geometries() -> list:
    collection = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    return [shape(feature["geometry"]) for feature in collection["features"]]


def haversine_km(case: MovementCase) -> float:
    lat1, lat2 = math.radians(case.start_lat), math.radians(case.end_lat)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(case.end_lon - case.start_lon)
    a = math.sin(delta_lat / 2) ** 2
    a += math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(a))


def elapsed_hours(case: MovementCase) -> float:
    start = datetime.fromisoformat(case.start_last_seen)
    end = datetime.fromisoformat(case.end_first_seen)
    hours = (end - start).total_seconds() / 3600
    if hours <= 0:
        raise ValueError(f"Non-positive movement interval for {case.slug}")
    return hours


def render_case(case: MovementCase, geometries: list, output_dir: Path) -> None:
    plate = ccrs.PlateCarree()
    fig = plt.figure(figsize=(2.25, 2.62))
    ax = fig.add_axes([0.16, 0.11, 0.81, 0.78], projection=plate)
    ax.set_extent(case.bbox, crs=plate)
    ax.set_facecolor("#dceaf2")
    ax.add_geometries(
        geometries,
        crs=plate,
        facecolor="#f4f1e8",
        edgecolor="#777067",
        linewidth=0.35,
        zorder=0,
    )

    gridlines = ax.gridlines(
        crs=plate,
        draw_labels=True,
        linewidth=0.32,
        color="#ffffff",
        alpha=0.9,
        linestyle="-",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 6.3}
    gridlines.ylabel_style = {"size": 6.3}

    path_color = "#b23a48"
    ax.plot(
        [case.start_lon, case.end_lon],
        [case.start_lat, case.end_lat],
        transform=plate,
        color=path_color,
        linewidth=1.35,
        linestyle=(0, (3.2, 2.2)),
        zorder=3,
    )
    ax.scatter(
        [case.start_lon, case.end_lon],
        [case.start_lat, case.end_lat],
        transform=plate,
        marker="o",
        s=27,
        facecolor="#147d85",
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )

    dx = case.bbox[1] - case.bbox[0]
    dy = case.bbox[3] - case.bbox[2]
    start_on_right = case.start_lon > (case.bbox[0] + case.bbox[1]) / 2
    end_on_right = case.end_lon > (case.bbox[0] + case.bbox[1]) / 2
    start_on_top = case.start_lat > (case.bbox[2] + case.bbox[3]) / 2
    end_on_top = case.end_lat > (case.bbox[2] + case.bbox[3]) / 2
    ax.text(
        case.start_lon + (-0.035 if start_on_right else 0.035) * dx,
        case.start_lat + (-0.045 if start_on_top else 0.055) * dy,
        f"{case.start_label}  (n={case.start_observations})",
        transform=plate,
        fontsize=6.4,
        ha="right" if start_on_right else "left",
        va="top" if start_on_top else "bottom",
        color="#14535a",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
        zorder=5,
    )
    ax.text(
        case.end_lon + (-0.035 if end_on_right else 0.035) * dx,
        case.end_lat + (-0.045 if end_on_top else 0.055) * dy,
        f"{case.end_label}  (n={case.end_observations})",
        transform=plate,
        fontsize=6.4,
        ha="right" if end_on_right else "left",
        va="top" if end_on_top else "bottom",
        color="#14535a",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
        zorder=5,
    )

    distance = haversine_km(case)
    speed_knots = distance / elapsed_hours(case) / 1.852
    if case.vessel_speed:
        movement_summary = f"{distance:.0f} km  ·  ≥{speed_knots:.1f} kn"
    else:
        gap_days = elapsed_hours(case) / 24
        movement_summary = f"{distance:.0f} km  ·  {gap_days:.0f}-day gap"
    ax.text(
        0.025,
        0.025,
        movement_summary,
        transform=ax.transAxes,
        fontsize=6.5,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "#aaa49a",
            "linewidth": 0.35,
            "alpha": 0.94,
            "pad": 1.8,
        },
        zorder=6,
    )
    fig.text(0.565, 0.965, case.title, ha="center", va="top", fontsize=7.2, fontweight="bold")
    ax.spines["geo"].set_edgecolor("#68625b")
    ax.spines["geo"].set_linewidth(0.5)

    output = output_dir / f"ocean_vessel_{case.slug}.pdf"
    preview = output.with_suffix(".png")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=360)
    fig.savefig(preview, dpi=240)
    plt.close(fig)
    print(
        f"[case] {case.slug}: {distance:.1f} km, "
        f"{elapsed_hours(case):.2f} h, >= {speed_knots:.1f} kn"
    )
    print(f"[figure] {output}")
    print(f"[preview] {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    geometries = world_geometries()
    for case in CASES:
        extent = box(case.bbox[0], case.bbox[2], case.bbox[1], case.bbox[3])
        local_geometries = [geometry for geometry in geometries if geometry.intersects(extent)]
        render_case(case, local_geometries, args.output_dir)


if __name__ == "__main__":
    main()
