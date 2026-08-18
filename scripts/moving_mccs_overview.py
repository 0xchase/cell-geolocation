#!/usr/bin/env python3
"""Build the paper overview of exact cell-identity position spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "moving-mccs" / "identities.csv.zst"
COUNTRIES = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
DEFAULT_OUTPUT = ROOT / "paper" / "figs" / "moving_mccs_overview.pdf"

BANDS = [
    "10-25 km",
    "25-100 km",
    "100-500 km",
    "500-1,000 km",
    "1,000-5,000 km",
    "5,000-10,000 km",
    "10,000+ km",
]
SHORT_BANDS = ["10–25", "25–100", "100–500", "500–1k", "1k–5k", "5k–10k", "10k+"]
BAND_COLORS = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#8c6bb1", "#88419d", "#6e016b"]
SUPPORT_LABELS = ["2–9", "10–99", "100–999", "1,000+"]
SUPPORT_COLORS = ["#d9d9d9", "#9ecae1", "#4292c6", "#08519c"]


def load_world_polygons(path: Path) -> list[Polygon]:
    obj = json.loads(path.read_text())
    patches: list[Polygon] = []
    for feature in obj["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for polygon in polygons:
            if polygon and len(polygon[0]) >= 3:
                patches.append(Polygon(polygon[0], closed=True))
    return patches


def draw_world(ax: plt.Axes, patches: list[Polygon]) -> None:
    collection = PatchCollection(
        patches,
        facecolor="#f4f1e9",
        edgecolor="#aaa49a",
        linewidth=0.18,
        zorder=0,
    )
    ax.add_collection(collection)
    ax.set_facecolor("#e7f0f5")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 83)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-30, 0, 30, 60])
    ax.grid(color="white", linewidth=0.35, alpha=0.8, zorder=-1)


def make_figure(df: pd.DataFrame, output: Path, preview: Path | None) -> None:
    df["distance_band"] = pd.Categorical(df["distance_band"], categories=BANDS, ordered=True)
    df = df.sort_values("distance_band")
    band_counts = df.groupby("distance_band", observed=False).size().reindex(BANDS)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.0, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1])

    ax = fig.add_subplot(grid[0, 0])
    bars = ax.bar(np.arange(len(BANDS)), band_counts.values, color=BAND_COLORS, width=0.78)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(BANDS)), SHORT_BANDS, rotation=34, ha="right")
    ax.set_ylabel("Distinct cell identities (log scale)")
    ax.set_xlabel("Exact maximum observed span (km)")
    ax.set_title("A. Most candidates span less than 25 km", loc="left")
    ax.grid(axis="y", linewidth=0.4, alpha=0.45)
    for bar, value in zip(bars, band_counts.values, strict=True):
        label = f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else f"{value / 1_000:.1f}k" if value >= 10_000 else f"{value:,}"
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.18, label,
                ha="center", va="bottom", fontsize=6.4)
    ax.set_ylim(500, band_counts.max() * 8)

    ax = fig.add_subplot(grid[0, 1])
    spans = np.sort(df["max_span_km"].to_numpy(dtype=float))
    thresholds = np.logspace(1, np.log10(spans.max()), 260)
    remaining = len(spans) - np.searchsorted(spans, thresholds, side="left")
    ax.plot(thresholds, remaining, color="#7b2e52", linewidth=1.8)
    for threshold in (25, 100, 1000, 10000):
        count = int((spans >= threshold).sum())
        ax.scatter([threshold], [count], color="#7b2e52", s=16, zorder=3)
        ax.annotate(f"{count:,}", (threshold, count), xytext=(3, 4),
                    textcoords="offset points", fontsize=6.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Minimum exact span threshold (km, log scale)")
    ax.set_ylabel("Identities at or above threshold (log scale)")
    ax.set_title("B. Counts fall rapidly with distance", loc="left")
    ax.grid(which="both", linewidth=0.4, alpha=0.4)

    ax = fig.add_subplot(grid[1, 0])
    support = pd.cut(
        df["total_observations"],
        bins=[1, 9, 99, 999, np.inf],
        labels=SUPPORT_LABELS,
        include_lowest=True,
    )
    support_table = (
        pd.crosstab(df["distance_band"], support, normalize="index")
        .reindex(index=BANDS, columns=SUPPORT_LABELS, fill_value=0)
    )
    bottom = np.zeros(len(BANDS))
    for label, color in zip(SUPPORT_LABELS, SUPPORT_COLORS, strict=True):
        values = support_table[label].to_numpy() * 100
        ax.bar(np.arange(len(BANDS)), values, bottom=bottom, color=color, width=0.78,
               label=f"{label} observations")
        bottom += values
    ax.set_xticks(np.arange(len(BANDS)), SHORT_BANDS, rotation=34, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of identities (%)")
    ax.set_xlabel("Exact maximum observed span (km)")
    ax.set_title("C. Support differs across distance bands", loc="left")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    ax = fig.add_subplot(grid[1, 1])
    polygons = load_world_polygons(COUNTRIES)
    draw_world(ax, polygons)
    long = df[df["max_span_km"] >= 100]
    lon = np.concatenate([long["endpoint_a_lon"].to_numpy(), long["endpoint_b_lon"].to_numpy()])
    lat = np.concatenate([long["endpoint_a_lat"].to_numpy(), long["endpoint_b_lat"].to_numpy()])
    valid = np.isfinite(lon) & np.isfinite(lat) & (lat >= -58) & (lat <= 83)
    hb = ax.hexbin(
        lon[valid], lat[valid], gridsize=(90, 36), mincnt=1,
        norm=LogNorm(), cmap="magma_r", linewidths=0, alpha=0.88, zorder=2,
    )
    cbar = fig.colorbar(hb, ax=ax, orientation="horizontal", pad=0.02, fraction=0.07)
    cbar.set_label("Maximum-span endpoints per hexagon (log scale)", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.8)
    ax.set_title(f"D. Endpoints for spans ≥100 km ({len(long):,} identities)", loc="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_OUTPUT.with_suffix(".png"))
    args = parser.parse_args()
    columns = [
        "distance_band", "max_span_km", "endpoint_a_lat", "endpoint_a_lon",
        "endpoint_b_lat", "endpoint_b_lon", "total_observations",
    ]
    df = pd.read_csv(args.input, compression="zstd", usecols=columns)
    print(f"loaded {len(df):,} exact-span identities")
    make_figure(df, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
