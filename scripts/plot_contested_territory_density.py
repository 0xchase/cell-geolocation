#!/usr/bin/env python3
"""Render bare Transnistria and Karabakh panels for ``results.tex``."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage

from extract_contested_territory_density import CASES, ROOT
from plot_helpers import add_osm_basemap, draw_geojson_layer


DATA = ROOT / "data" / "out-of-country" / "contested-territories"
FIGS = ROOT / "paper" / "figs"
ADMIN0 = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
BASEMAP = "carto_voyager_nolabels"
PRIMARY_COLOR = "#c02a3c"
SECONDARY_COLOR = "#2f6f9f"
VMAX = 200.0
PIXELS = 699
DPI = 300

LABELS = {
    "transnistria": [
        ("Chișinău", 28.8353, 47.0105),
        ("Tiraspol", 29.6333, 46.8403),
        ("Rîbnița", 29.0011, 47.7664),
    ],
    "karabakh": [
        ("Yerevan", 44.5152, 40.1872),
        ("Ganja", 46.3606, 40.6828),
        ("Khankendi", 46.7528, 39.8153),
    ],
}


def ramp(name: str, light: str, dark: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [light, dark], N=256)


def load_grid(key: str) -> tuple[np.ndarray, np.ndarray]:
    case = CASES[key]
    primary = np.zeros((case.nbins, case.nbins), dtype=float)
    secondary = np.zeros((case.nbins, case.nbins), dtype=float)
    path = DATA / f"{key}-density-grid.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            iy, ix = int(row["iy"]), int(row["ix"])
            primary[iy, ix] = int(row["primary_cells"])
            secondary[iy, ix] = int(row["secondary_cells"])
    return primary, secondary


def density_rgba(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    stack = np.stack([primary, secondary])
    winner = stack.argmax(axis=0)
    value = stack.max(axis=0)
    seen = stack.sum(axis=0) > 0
    strength = np.log10(np.clip(value, 1.0, VMAX)) / np.log10(VMAX)
    cmaps = [
        ramp("primary", "#edb6bd", PRIMARY_COLOR),
        ramp("secondary", "#b7cee0", SECONDARY_COLOR),
    ]
    rgba = np.zeros(value.shape + (4,), dtype=float)
    for index, cmap in enumerate(cmaps):
        mask = winner == index
        rgba[mask] = cmap(strength[mask])
    rgba[..., 3] = seen.astype(float)
    return rgba


def add_labels(ax: plt.Axes, key: str) -> None:
    for label, lon, lat in LABELS[key]:
        ax.plot(lon, lat, marker="o", markersize=1.8, color="#292724", zorder=5)
        text = ax.annotate(
            label, (lon, lat), xytext=(2.0, 1.5), textcoords="offset points",
            fontsize=4.6, color="#292724", weight="semibold", zorder=5.1,
        )
        text.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])


def render(key: str) -> None:
    case = CASES[key]
    primary, secondary = load_grid(key)
    side = PIXELS / DPI
    fig = plt.figure(figsize=(side, side), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor("#dceaf2")
    add_osm_basemap(
        ax, case.bbox, zoom=8, alpha=1.0, grayscale=False,
        zorder=0, source=BASEMAP,
    )
    countries = {"MD", "RO", "UA"} if key == "transnistria" else {"AM", "AZ", "GE", "IR"}
    draw_geojson_layer(
        ax, ADMIN0, case.bbox, countries=countries, facecolor="none",
        edgecolor="#292724", linewidth=0.55, alpha=0.90, zorder=2.2,
    )
    ax.imshow(
        density_rgba(primary, secondary), origin="lower", extent=case.bbox,
        interpolation="nearest", aspect="auto", zorder=3,
    )
    draw_geojson_layer(
        ax, ADMIN0, case.bbox, countries=countries, facecolor="none",
        edgecolor="#292724", linewidth=0.42, alpha=0.88, zorder=4,
    )
    add_labels(ax, key)
    ax.set_xlim(case.bbox[0], case.bbox[1])
    ax.set_ylim(case.bbox[2], case.bbox[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    output = FIGS / f"{key}_cellular_density.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, pad_inches=0)
    plt.close(fig)
    with PILImage.open(output) as image:
        image.convert("RGB").save(output, optimize=True)
    print(f"[figure] {output.relative_to(ROOT)} ({PIXELS}x{PIXELS})")


def main() -> None:
    for key in CASES:
        render(key)


if __name__ == "__main__":
    main()
