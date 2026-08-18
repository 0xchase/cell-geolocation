#!/usr/bin/env python3
"""Plot a global overview of the buffered offshore cell-position snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "oceans" / "ocean_cell_positions.csv"
BOUNDARIES = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
DEFAULT_OUTPUT = ROOT / "paper" / "figs" / "ocean_activity_global.pdf"
GRID_DEGREES = 0.5


def world_geometries() -> list:
    collection = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    return [shape(feature["geometry"]) for feature in collection["features"]]


def load_positions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    columns = ["mcc", "mnc", "lac", "cid", "cell_type", "lat", "lon", "observations"]
    raw = pd.read_csv(path, usecols=columns)
    valid = (
        raw["lat"].between(-90, 90)
        & raw["lon"].between(-180, 180)
        & raw["observations"].gt(0)
    )
    if not valid.all():
        raise ValueError(f"Input contains {(~valid).sum():,} invalid rows")

    total_rows = len(raw)
    total_observations = int(raw["observations"].sum())
    positions = (
        raw.groupby(["lat", "lon"], as_index=False)
        .agg(observations=("observations", "sum"))
    )

    # Aggregate to a fixed geographic grid so dense regions remain legible on
    # a page-width world map. Faint exact-coordinate points are plotted too.
    lon_for_bin = positions["lon"].clip(-180, np.nextafter(180.0, -np.inf))
    lat_for_bin = positions["lat"].clip(-90, np.nextafter(90.0, -np.inf))
    positions["grid_lon"] = (
        np.floor((lon_for_bin + 180) / GRID_DEGREES) * GRID_DEGREES
        - 180
        + GRID_DEGREES / 2
    )
    positions["grid_lat"] = (
        np.floor((lat_for_bin + 90) / GRID_DEGREES) * GRID_DEGREES
        - 90
        + GRID_DEGREES / 2
    )
    grid = (
        positions.groupby(["grid_lat", "grid_lon"], as_index=False)
        .agg(observations=("observations", "sum"), exact_positions=("lat", "size"))
    )
    if int(grid["observations"].sum()) != total_observations:
        raise RuntimeError("Gridding changed the total observation count")
    return positions, grid, total_rows, total_observations


def make_plot(input_path: Path, output: Path) -> None:
    positions, grid, total_rows, total_observations = load_positions(input_path)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    projection = ccrs.Robinson(central_longitude=0)
    plate = ccrs.PlateCarree()
    fig = plt.figure(figsize=(7.0, 3.62))
    ax = fig.add_axes([0.012, 0.15, 0.976, 0.835], projection=projection)
    ax.set_global()
    ax.set_facecolor("#e9f1f5")
    ax.add_geometries(
        world_geometries(),
        crs=plate,
        facecolor="#f4f1e8",
        edgecolor="#918b82",
        linewidth=0.18,
        zorder=0,
    )

    # Preserve the footprint of every exact 0.01-degree coordinate underneath
    # the observation-weighted grid cells.
    ax.scatter(
        positions["lon"],
        positions["lat"],
        transform=plate,
        s=0.45,
        c="#293241",
        alpha=0.22,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    norm = LogNorm(vmin=1, vmax=float(grid["observations"].max()))
    activity = ax.scatter(
        grid["grid_lon"],
        grid["grid_lat"],
        transform=plate,
        s=8.5,
        marker="s",
        c=grid["observations"],
        cmap="magma_r",
        norm=norm,
        alpha=0.90,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    # Flag only the two manually identified classes of implausible coordinates;
    # they remain in the totals and activity layer above.
    suspected = positions[
        ((positions["lat"].abs() <= 1) & (positions["lon"].abs() <= 1))
        | (positions["lat"] >= 83)
    ]
    ax.scatter(
        suspected["lon"],
        suspected["lat"],
        transform=plate,
        s=13,
        marker="x",
        c="#00a6a6",
        linewidths=0.65,
        alpha=0.9,
        rasterized=True,
        zorder=3,
    )

    ax.spines["geo"].set_edgecolor("#777777")
    ax.spines["geo"].set_linewidth(0.45)
    cbar = fig.colorbar(
        activity,
        ax=ax,
        orientation="horizontal",
        pad=0.035,
        fraction=0.055,
        aspect=46,
    )
    cbar.set_label(f"Observations per {GRID_DEGREES:g}° grid cell (log scale)", labelpad=2)
    cbar.ax.tick_params(labelsize=6.6, length=2.5, pad=1.5)
    cbar.outline.set_linewidth(0.35)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="x",
                markeredgecolor="#00a6a6",
                markeredgewidth=0.9,
                markersize=5,
                label="Suspected sentinel / polar outlier",
            )
        ],
        loc="lower left",
        bbox_to_anchor=(0.018, 0.012),
        frameon=True,
        facecolor="white",
        edgecolor="#b8b8b8",
        framealpha=0.94,
        fontsize=6.5,
        borderpad=0.35,
        handletextpad=0.3,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=360, bbox_inches="tight", pad_inches=0.015)
    preview = output.with_suffix(".png")
    fig.savefig(preview, dpi=240, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    print(f"[figure] {output}")
    print(f"[preview] {preview}")
    print(
        "[audit] "
        f"rows={total_rows:,}, exact_positions={len(positions):,}, "
        f"grid_cells={len(grid):,}, observations={total_observations:,}, "
        f"flagged_positions={len(suspected):,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_plot(args.input, args.output)


if __name__ == "__main__":
    main()
