#!/usr/bin/env python3
"""Render compact USENIX figures from data/out-of-country CSV exports only."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_helpers import (  # noqa: E402
    ADMIN1_GEOJSON,
    COUNTRIES_GEOJSON,
    TILE_ATTRIBUTION,
    add_osm_basemap,
    draw_geojson_layer,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "out-of-country"
FIGS = ROOT / "paper" / "figs"
BASEMAP = "carto_voyager_nolabels_retina"
MAP_UNITS = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"

SYRIA_BBOX = (35.50, 42.50, 35.00, 37.40)
WEST_BANK_BBOX = (34.82, 35.62, 31.30, 32.62)
GAZA_BBOX = (34.20, 34.60, 31.20, 31.60)
GOLAN_BBOX = (35.55, 36.15, 32.70, 33.45)

SYRIA_GROUPS = [
    ("syrian_cells", "Syrian (417)", "#0072B2"),
    ("turkish_cells", "Turkish (286)", "#D55E00"),
    ("iraqi_cells", "Iraqi (418)", "#E69F00"),
    ("china_cells", "China Mobile (460/00)", "#7A3E9D"),
]
WB_GROUPS = [
    ("palestinian_cells", "Palestinian (425/05–06)", "#0072B2"),
    ("israeli_cells", "Israeli (425, other MNC)", "#D55E00"),
    ("jordanian_cells", "Jordanian (416)", "#009E73"),
    ("egyptian_cells", "Egyptian (602)", "#CC79A7"),
    ("other_foreign_cells", "Other regional MCC", "#7A3E9D"),
]
GAZA_GROUPS = [
    ("palestinian_cells", "Palestinian (425/05–06)", "#0072B2"),
    ("israeli_cells", "Israeli (425, other MNC)", "#D55E00"),
    ("egyptian_cells", "Egyptian (602)", "#CC79A7"),
    ("libyan_cells", "Libyan (606)", "#7A3E9D"),
    ("saudi_cells", "Saudi (420)", "#E69F00"),
    ("turkish_cells", "Turkish (286)", "#56B4E9"),
    ("jordanian_cells", "Jordanian (416)", "#009E73"),
    ("cypriot_cells", "Cypriot (280)", "#A6761D"),
    ("emirati_cells", "Emirati (424)", "#F0E442"),
    ("other_country_cells", "Other country MCC", "#666666"),
]
GOLAN_GROUPS = [
    ("syrian_cells", "Syrian MCC 417", "#0072B2"),
    ("israeli_cells", "Israeli MCC 425", "#D55E00"),
]
SYRIA_CITIES = [
    ("Gaziantep", 37.38, 37.07),
    ("Şanlıurfa", 38.79, 37.17),
    ("Aleppo", 37.16, 36.20),
    ("Idlib", 36.63, 35.93),
    ("Raqqa", 39.01, 35.95),
    ("Hasakah", 40.75, 36.50),
    ("Qamishli", 41.22, 37.05),
]
WB_CITIES = [
    ("Jenin", 35.30, 32.46),
    ("Nablus", 35.26, 32.22),
    ("Ramallah", 35.20, 31.90),
    ("Jerusalem", 35.21, 31.77),
    ("Jericho", 35.44, 31.86),
    ("Hebron", 35.10, 31.53),
]
GAZA_CITIES = [
    ("Gaza City", 34.466, 31.507),
    ("Khan Yunis", 34.303, 31.342),
    ("Rafah", 34.256, 31.288),
]
GOLAN_CITIES = [
    ("Majdal Shams", 35.7697, 33.2697),
    ("Buq'ata", 35.7790, 33.2010),
    ("Quneitra", 35.8246, 33.1257),
    ("Katzrin", 35.6914, 32.9923),
]
MCC_NAMES = {
    204: "Netherlands", 206: "Belgium", 208: "France", 216: "Hungary",
    222: "Italy", 230: "Czechia", 231: "Slovakia", 232: "Austria",
    238: "Denmark", 260: "Poland", 262: "Germany", 270: "Luxembourg",
    293: "Slovenia",
}

def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing figure input: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def setup_map(ax: plt.Axes, bbox: tuple[float, float, float, float], countries: set[str], zoom: int) -> None:
    ax.set_facecolor("#dceaf2")
    drawn = add_osm_basemap(
        ax, bbox, zoom=zoom, source=BASEMAP, alpha=0.96,
        grayscale=False, zorder=0,
    )
    if not drawn:
        draw_geojson_layer(
            ax, COUNTRIES_GEOJSON, bbox, countries=countries,
            facecolor="#f5f1e8", edgecolor="#69635c", linewidth=0.45, zorder=0,
        )
    if ADMIN1_GEOJSON.exists():
        draw_geojson_layer(
            ax, ADMIN1_GEOJSON, bbox, countries=countries,
            facecolor="none", edgecolor="#7c746b", linewidth=0.25, alpha=0.75, zorder=2,
        )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    # Correct plate-carree's longitude stretch at the map midpoint.
    ax.set_aspect(1 / math.cos(math.radians((bbox[2] + bbox[3]) / 2)), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#817a72")


def add_map_reference_overlay(
    ax: plt.Axes, bbox: tuple[float, float, float, float], countries: set[str],
    city_labels: list[tuple[str, float, float]],
) -> None:
    """Keep political boundaries and selected cities legible above dense cells."""
    if MAP_UNITS.exists():
        draw_geojson_layer(
            ax, MAP_UNITS, bbox, countries=countries,
            facecolor="none", edgecolor="#443f3a", linewidth=0.58,
            alpha=0.95, zorder=4.5,
        )
    for label, lon, lat in city_labels:
        ax.plot(lon, lat, marker="o", markersize=1.55, color="#27231f", zorder=5.1)
        text = ax.annotate(
            label, (lon, lat), xytext=(2.0, 1.6), textcoords="offset points",
            fontsize=4.9, color="#27231f", weight="semibold",
            ha="left", va="bottom", zorder=5.2,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=1.35, foreground="white")])


def grid_arrays(
    grid: list[dict[str, str]], groups: list[tuple[str, str, str]],
    bbox: tuple[float, float, float, float], grid_step: tuple[float, float],
) -> tuple[dict, list[int], float]:
    years = sorted({int(r["year"]) for r in grid})
    west, east, south, north = bbox
    dlat, dlon = grid_step
    nx = math.ceil((east - west) / dlon)
    ny = math.ceil((north - south) / dlat)
    arrays = {(key, year): np.zeros((ny, nx), dtype=float) for key, _, _ in groups for year in years}
    for r in grid:
        iy, ix, year = int(r["iy"]), int(r["ix"]), int(r["year"])
        for key, _, _ in groups:
            arrays[(key, year)][iy, ix] = float(r[key])
    positive = np.concatenate([a[a > 0] for a in arrays.values() if np.any(a > 0)])
    vmax = float(max(10, math.ceil(np.percentile(positive, 99.5) / 10) * 10))
    return arrays, years, vmax


def dominant_rgba(arrays: dict, groups: list[tuple[str, str, str]], year: int, vmax: float) -> np.ndarray:
    stack = np.stack([arrays[(key, year)] for key, _, _ in groups])
    winner = stack.argmax(axis=0)
    value = stack.max(axis=0)
    seen = stack.sum(axis=0) > 0
    scale = np.log10(np.clip(value, 1, vmax)) / math.log10(vmax)
    rgba = np.zeros(value.shape + (4,), dtype=float)
    for i, (_key, _label, color) in enumerate(groups):
        mask = winner == i
        rgba[mask, :3] = mpl.colors.to_rgb(color)
    rgba[..., 3] = seen.astype(float) * (0.30 + 0.55 * scale)
    return rgba


def collection_array(
    grid: list[dict[str, str]], groups: list[tuple[str, str, str]],
    bbox: tuple[float, float, float, float], grid_step: tuple[float, float],
    bin_factor: int = 1,
) -> tuple[dict[str, np.ndarray], float]:
    west, east, south, north = bbox
    dlat, dlon = grid_step
    nx = math.ceil((east - west) / dlon)
    ny = math.ceil((north - south) / dlat)
    arrays = {key: np.zeros((ny, nx), dtype=float) for key, _, _ in groups}
    for r in grid:
        iy, ix = int(r["iy"]), int(r["ix"])
        for key, _, _ in groups:
            arrays[key][iy, ix] = float(r[key])
    if bin_factor > 1:
        pad_y = (-ny) % bin_factor
        pad_x = (-nx) % bin_factor
        arrays = {
            key: np.pad(a, ((0, pad_y), (0, pad_x)))
            .reshape(
                (ny + pad_y) // bin_factor, bin_factor,
                (nx + pad_x) // bin_factor, bin_factor,
            )
            .sum(axis=(1, 3))
            for key, a in arrays.items()
        }
    positive = np.concatenate([a[a > 0] for a in arrays.values() if np.any(a > 0)])
    vmax = float(max(10, math.ceil(np.percentile(positive, 99.5) / 10) * 10))
    return arrays, vmax


def collection_rgba(
    arrays: dict[str, np.ndarray], groups: list[tuple[str, str, str]], vmax: float,
) -> np.ndarray:
    stack = np.stack([arrays[key] for key, _, _ in groups])
    winner = stack.argmax(axis=0)
    value = stack.max(axis=0)
    seen = stack.sum(axis=0) > 0
    scale = np.log10(np.clip(value, 1, vmax)) / math.log10(vmax)
    rgba = np.zeros(value.shape + (4,), dtype=float)
    for i, (_key, _label, color) in enumerate(groups):
        mask = winner == i
        base = np.asarray(mpl.colors.to_rgb(color))
        strength = 0.58 + 0.42 * scale[mask]
        rgba[mask, :3] = 1.0 - (1.0 - base) * strength[:, None]
    rgba[..., 3] = seen.astype(float)
    return rgba


def make_collection_maps(data: Path, output: Path) -> None:
    cases = [
        {
            "title": "(a) Northern Syria",
            "grid": data / "northern-syria-grid.csv",
            "groups": SYRIA_GROUPS,
            "raster_groups": SYRIA_GROUPS,
            "cell_overlay_groups": [],
            "overlay_groups": [],
            "legend_ncol": 2,
            "legend_font": 5.3,
            "bbox": SYRIA_BBOX,
            "step": (0.004505, 0.00558),
            "bin_factor": 4,
            "countries": {"SY", "TR", "Iraq"},
            "cities": SYRIA_CITIES,
            "zoom": 7,
        },
        {
            "title": "(b) West Bank",
            "grid": data / "west-bank-grid.csv",
            "groups": WB_GROUPS,
            "raster_groups": WB_GROUPS[:2],
            "cell_overlay_groups": [],
            "overlay_groups": WB_GROUPS[2:],
            "overlay_scale": 1.0,
            "legend_ncol": 1,
            "legend_font": 3.8,
            "bbox": WEST_BANK_BBOX,
            "step": (0.0022525, 0.00265),
            "bin_factor": 1,
            "countries": {"Palestine", "IL", "JO"},
            "cities": WB_CITIES,
            "zoom": 9,
        },
        {
            "title": "(c) Gaza Strip",
            "grid": data / "gaza-grid.csv",
            "groups": GAZA_GROUPS,
            "raster_groups": GAZA_GROUPS[:2],
            "cell_overlay_groups": [],
            "overlay_groups": GAZA_GROUPS[2:],
            "overlay_scale": 0.42,
            "legend_ncol": 2,
            "legend_font": 3.15,
            "bbox": GAZA_BBOX,
            "step": (0.00112625, 0.001325),
            "bin_factor": 1,
            "countries": {"PS", "IL", "EG"},
            "cities": GAZA_CITIES,
            "zoom": 11,
        },
    ]
    fig = plt.figure(figsize=(7.0, 3.10))
    gs = fig.add_gridspec(
        1, 3, width_ratios=(1.75, 0.66, 0.66),
        left=0.025, right=0.985, bottom=0.11, top=0.93, wspace=0.08,
    )
    for spec, case in zip(gs, cases, strict=True):
        ax = fig.add_subplot(spec)
        setup_map(ax, case["bbox"], case["countries"], case["zoom"])
        ax.set_anchor("N")
        arrays, vmax = collection_array(
            rows(case["grid"]), case["groups"], case["bbox"], case["step"],
            case["bin_factor"],
        )
        ax.imshow(
            collection_rgba(arrays, case["raster_groups"], vmax),
            origin="lower", extent=case["bbox"], interpolation="nearest", zorder=3,
        )
        for group in case["cell_overlay_groups"]:
            key = group[0]
            positive = arrays[key][arrays[key] > 0]
            if positive.size:
                group_vmax = float(max(2, np.percentile(positive, 99.5)))
                overlay_rgba = collection_rgba(arrays, [group], group_vmax)
                overlay_rgba[overlay_rgba[..., 3] > 0, :3] = mpl.colors.to_rgb(group[2])
                ax.imshow(
                    overlay_rgba,
                    origin="lower", extent=case["bbox"],
                    interpolation="nearest", zorder=4.0,
                )
        west, _east, south, _north = case["bbox"]
        dlat, dlon = (
            step * case["bin_factor"] for step in case["step"]
        )
        for key, _label, color in case["overlay_groups"]:
            iy, ix = np.nonzero(arrays[key] > 0)
            counts = arrays[key][iy, ix]
            marker_sizes = 2.1 + 0.8 * np.log10(counts + 1)
            # Singleton categories need a callout-sized symbol to survive the
            # reduction from the source figure to a two-column paper page.
            if counts.size <= 5:
                marker_sizes = np.maximum(marker_sizes, 8.0)
            marker_sizes *= case.get("overlay_scale", 1.0)
            marker_x = west + (ix + 0.5) * dlon
            marker_y = south + (iy + 0.5) * dlat
            ax.scatter(
                marker_x, marker_y, s=marker_sizes, marker="o",
                color=color, alpha=0.96, edgecolor="#333333",
                linewidth=0.28 * math.sqrt(case.get("overlay_scale", 1.0)),
                zorder=4.2,
            )
        add_map_reference_overlay(ax, case["bbox"], case["countries"], case["cities"])
        ax.set_title(case["title"], fontsize=8.0, weight="semibold", pad=3)
        raster_handles = [
            Line2D([0], [0], color=color, linewidth=3.0, label=label)
            for _, label, color in case["raster_groups"] + case["cell_overlay_groups"]
        ]
        overlay_handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=4.0,
                   markerfacecolor=color, markeredgecolor="#333333",
                   markeredgewidth=0.3, label=label)
            for _, label, color in case["overlay_groups"]
        ]
        ax.legend(
            handles=raster_handles + overlay_handles,
            loc="upper center", bbox_to_anchor=(0.5, -0.055),
            ncol=case["legend_ncol"], frameon=False,
            fontsize=case["legend_font"], handlelength=1.2,
            handletextpad=0.45, columnspacing=0.65, labelspacing=0.25,
        )
    attribution = TILE_ATTRIBUTION[BASEMAP].replace(r"\copyright{}", "©")
    fig.text(0.985, 0.012, attribution, ha="right", fontsize=5.0, color="#555")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The PDF backend otherwise rasterizes image artists near its low default
    # DPI; 600 dpi keeps the 1 km analytical grid and 2x basemap tiles crisp.
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output.relative_to(ROOT)}")


def make_golan_map(data: Path, output: Path) -> None:
    grid = rows(data / "golan-heights-grid.csv")
    arrays, vmax = collection_array(
        grid, GOLAN_GROUPS, GOLAN_BBOX, (0.001802, 0.00215), bin_factor=1,
    )
    totals = {
        key: int(sum(float(row[key]) for row in grid))
        for key, _label, _color in GOLAN_GROUPS
    }

    line_rows = rows(data / "golan-undof-lines.csv")
    lines: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in sorted(line_rows, key=lambda item: (item["line"], int(item["seq"]))):
        lines[row["line"]].append((float(row["lon"]), float(row["lat"])))

    fig, ax = plt.subplots(figsize=(3.35, 4.35))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.15, top=0.985)
    setup_map(ax, GOLAN_BBOX, {"IL", "SY", "JO", "LB"}, zoom=10)

    # The official UNDOF map identifies the western Alpha ceasefire line and
    # eastern Bravo line. Shade only their shared latitude range so the figure
    # does not imply a separation area beyond the supplied line geometries.
    alpha = np.asarray(lines["Alpha"])
    bravo = np.asarray(lines["Bravo"])
    lo = max(float(alpha[:, 1].min()), float(bravo[:, 1].min()))
    hi = min(float(alpha[:, 1].max()), float(bravo[:, 1].max()))
    common_lat = np.linspace(lo, hi, 500)

    def interpolate_lon(points: np.ndarray) -> np.ndarray:
        order = np.argsort(points[:, 1])
        return np.interp(common_lat, points[order, 1], points[order, 0])

    alpha_lon = interpolate_lon(alpha)
    bravo_lon = interpolate_lon(bravo)
    ax.fill_betweenx(
        common_lat, alpha_lon, bravo_lon, color="#E9C46A", alpha=0.20,
        linewidth=0, zorder=2.6,
    )

    ax.imshow(
        collection_rgba(arrays, GOLAN_GROUPS, vmax),
        origin="lower", extent=GOLAN_BBOX, interpolation="nearest", zorder=3,
    )
    ax.plot(alpha[:, 0], alpha[:, 1], color="#24282a", linewidth=0.9,
            linestyle="-", zorder=4.7)
    ax.plot(bravo[:, 0], bravo[:, 1], color="#24282a", linewidth=0.9,
            linestyle=(0, (3, 2)), zorder=4.7)

    for label, lon, lat in GOLAN_CITIES:
        ax.plot(lon, lat, marker="o", markersize=1.7, color="#27231f", zorder=5.1)
        text = ax.annotate(
            label, (lon, lat), xytext=(2.0, 1.7), textcoords="offset points",
            fontsize=5.5, color="#27231f", weight="semibold",
            ha="left", va="bottom", zorder=5.2,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=1.4, foreground="white")])

    zone = ax.text(
        35.862, 33.055, "UNDOF area of\nseparation", ha="center", va="center",
        fontsize=5.2, color="#5b4a20", rotation=78, zorder=5.3,
    )
    zone.set_path_effects([path_effects.withStroke(linewidth=1.6, foreground="white")])

    handles = [
        Line2D([0], [0], color=color, linewidth=3.2,
               label=f"{label} · {totals[key]:,}")
        for key, label, color in GOLAN_GROUPS
    ] + [
        Line2D([0], [0], color="#24282a", linewidth=0.9,
               linestyle="-", label="Alpha ceasefire line"),
        Line2D([0], [0], color="#24282a", linewidth=0.9,
               linestyle=(0, (3, 2)), label="Bravo line"),
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.025),
        ncol=2, frameon=False, fontsize=5.35, handlelength=1.55,
        handletextpad=0.45, columnspacing=0.75, labelspacing=0.35,
    )

    attribution = TILE_ATTRIBUTION[BASEMAP].replace(r"\copyright{}", "©")
    fig.text(
        0.985, 0.012,
        f"{attribution}; Alpha/Bravo lines © OpenStreetMap contributors",
        ha="right", fontsize=3.9, color="#555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output.relative_to(ROOT)} (vmax={vmax:g} cells/bin)")


def annual_lookup(path: Path) -> dict[int, dict[str, int]]:
    return {
        int(r["year"]): {k: int(float(v)) for k, v in r.items() if k != "year"}
        for r in rows(path)
    }


def plot_timeline(ax: plt.Axes, monthly: list[dict[str, str]], groups: list[tuple[str, str, str]]) -> None:
    dates = [datetime.strptime(r["month"][:10], "%Y-%m-%d") for r in monthly]
    for key, label, color in groups:
        vals = np.array([int(float(r[key])) for r in monthly])
        vals = np.where(vals > 0, vals, np.nan)
        ax.plot(dates, vals, color=color, linewidth=1.35, marker="o", markersize=2.1, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("Distinct cells (log)")
    ax.set_xlabel("Month")
    ax.grid(axis="y", which="both", color="#dedede", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=25)


def make_case_figure(
    *,
    grid_path: Path,
    annual_path: Path,
    monthly_path: Path,
    groups: list[tuple[str, str, str]],
    bbox: tuple[float, float, float, float],
    grid_step: tuple[float, float],
    countries: set[str],
    city_labels: list[tuple[str, float, float]],
    zoom: int,
    figsize: tuple[float, float],
    output: Path,
) -> None:
    grid = rows(grid_path)
    monthly = rows(monthly_path)
    annual_lookup(annual_path)  # Validate that the annual plotting input exists and parses.
    arrays, years, vmax = grid_arrays(grid, groups, bbox, grid_step)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, len(years), height_ratios=(2.65, 1.05),
        left=0.07, right=0.985, bottom=0.16, top=0.89, hspace=0.27, wspace=0.05,
    )
    for i, year in enumerate(years):
        ax = fig.add_subplot(gs[0, i])
        setup_map(ax, bbox, countries, zoom)
        ax.imshow(
            dominant_rgba(arrays, groups, year, vmax), origin="lower", extent=bbox,
            interpolation="nearest", zorder=3,
        )
        add_map_reference_overlay(ax, bbox, countries, city_labels)
        ax.set_title(str(year), fontsize=7.4, pad=2.0)

    tax = fig.add_subplot(gs[1, :])
    plot_timeline(tax, monthly, groups)
    handles = [Line2D([0], [0], color=color, linewidth=2.4, label=label) for _, label, color in groups]
    fig.legend(handles=handles, loc="upper center", ncol=len(groups), frameon=False,
               bbox_to_anchor=(0.51, 0.985), handlelength=1.5, columnspacing=1.2)
    attribution = TILE_ATTRIBUTION[BASEMAP].replace(r"\copyright{}", "©")
    fig.text(0.985, 0.025, attribution, ha="right", fontsize=5.2, color="#555")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output.relative_to(ROOT)} (shared map vmax={vmax:g} cells/bin)")


def make_aliasing_figure(data: Path, output: Path) -> None:
    raw = rows(data)
    agg = defaultdict(lambda: [0, 0])
    for r in raw:
        host = r["located_iso"]
        agg[host][0] += int(r["foreign_cells"])
        agg[host][1] += int(r["identities_with_local_suffix"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw={"width_ratios": (0.8, 2.2)})
    ax = axes[0]
    hosts = ["DE", "AT"]
    pct = [100 * agg[h][1] / agg[h][0] for h in hosts]
    bars = ax.bar(["Germany", "Austria"], pct, color=["#2f6f9f", "#b23a48"], width=0.62)
    for bar, p, h in zip(bars, pct, hosts, strict=True):
        ax.text(bar.get_x() + bar.get_width()/2, p + 1.4,
                f"{p:.1f}%\n{agg[h][1]:,}/{agg[h][0]:,}", ha="center", fontsize=6.0)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Foreign identities sharing a local suffix (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.45)

    ax = axes[1]
    selected = []
    for host in ("DE", "AT"):
        subset = sorted(
            (r for r in raw if r["located_iso"] == host),
            key=lambda r: int(r["foreign_cells"]), reverse=True,
        )[:8]
        selected.extend(subset)
    selected.reverse()
    yy = np.arange(len(selected))
    xx = [float(r["median_nearest_km"]) for r in selected]
    colors = ["#2f6f9f" if r["located_iso"] == "DE" else "#b23a48" for r in selected]
    sizes = [10 + 0.35 * math.sqrt(int(r["foreign_cells"])) for r in selected]
    labels = [
        f"{r['located_iso']}: {MCC_NAMES.get(int(r['mcc']), r['mcc'])} "
        f"{int(r['mcc']):03d}/{int(r['mnc']):02d}"
        for r in selected
    ]
    ax.scatter(xx, yy, s=sizes, c=colors, alpha=0.82, edgecolor="white", linewidth=0.45)
    ax.set_yticks(yy, labels)
    ax.set_xscale("log")
    ax.axvline(5, color="#777", linestyle="--", linewidth=0.7)
    ax.text(5.3, len(selected) - 0.3, "5 km", fontsize=5.3, color="#666", va="top")
    ax.set_xlabel("Median separation of matched identity pair (km, log scale)")
    ax.set_ylabel("Host: foreign PLMN")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e0e0e0", linewidth=0.45, which="both")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#2f6f9f", label="Host Germany"),
        Line2D([0], [0], marker="o", linestyle="", color="#b23a48", label="Host Austria"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.7, w_pad=1.1)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--figs", type=Path, default=FIGS)
    args = parser.parse_args()
    paper_style()

    make_collection_maps(args.data, args.figs / "cross_border_case_maps.pdf")
    make_golan_map(args.data, args.figs / "golan_heights_mcc_map.pdf")
    make_aliasing_figure(
        args.data / "aliasing" / "germany-austria-local-suffix-matches.csv",
        args.figs / "germany_austria_aliasing_diagnostic.pdf",
    )


if __name__ == "__main__":
    main()
