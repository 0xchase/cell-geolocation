#!/usr/bin/env python3
"""Render the bare Georgia panel placed beside the Ukraine progression maps."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage

from extract_georgia_russia_density import BBOX, NBINS
from plot_helpers import add_osm_basemap, draw_geojson_layer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "out-of-country" / "additional-cases" / "georgia-russia-density-grid.csv"
OUTPUT = ROOT / "paper" / "figs" / "georgia_russia_density.png"
ADMIN0 = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
ADMIN1 = ROOT / "data" / "reference" / "ne_10m_admin_1_georgia.geojson"

BASEMAP = "carto_voyager_nolabels"
RU_COLOR = "#c02a3c"
DOMESTIC_COLOR = "#2f6f9f"
VMAX = 200.0  # Ukraine progression's shared 1 km-bin scale.
PIXELS = 699
DPI = 300


def ramp(name: str, light: str, dark: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, [light, dark], N=256)


def load_grid() -> tuple[np.ndarray, np.ndarray]:
    russian = np.zeros((NBINS, NBINS), dtype=float)
    georgian = np.zeros((NBINS, NBINS), dtype=float)
    with DATA.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iy, ix = int(row["iy"]), int(row["ix"])
            russian[iy, ix] = int(row["russian_cells"])
            georgian[iy, ix] = int(row["georgian_cells"])
    return russian, georgian


def density_rgba(russian: np.ndarray, georgian: np.ndarray) -> np.ndarray:
    stack = np.stack([russian, georgian])
    winner = stack.argmax(axis=0)
    value = stack.max(axis=0)
    seen = stack.sum(axis=0) > 0
    strength = np.log10(np.clip(value, 1.0, VMAX)) / np.log10(VMAX)
    cmaps = [ramp("ru", "#edb6bd", RU_COLOR), ramp("ge", "#b7cee0", DOMESTIC_COLOR)]
    rgba = np.zeros(value.shape + (4,), dtype=float)
    for i, cmap in enumerate(cmaps):
        mask = winner == i
        rgba[mask] = cmap(strength[mask])
    rgba[..., 3] = seen.astype(float)
    return rgba


def render(output: Path = OUTPUT) -> None:
    russian, georgian = load_grid()
    side = PIXELS / DPI
    fig = plt.figure(figsize=(side, side), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor("#dceaf2")
    add_osm_basemap(ax, BBOX, zoom=8, alpha=1.0, grayscale=False, zorder=0, source=BASEMAP)
    draw_geojson_layer(ax, ADMIN1, BBOX, facecolor="none", edgecolor="#8a8279",
                       linewidth=0.25, alpha=0.80, zorder=2)
    draw_geojson_layer(ax, ADMIN0, BBOX, countries={"GE"}, facecolor="none",
                       edgecolor="#292724", linewidth=0.58, alpha=0.92, zorder=2.2)
    ax.imshow(density_rgba(russian, georgian), origin="lower", extent=BBOX,
              interpolation="nearest", aspect="auto", zorder=3)
    ax.set_xlim(BBOX[0], BBOX[1]); ax.set_ylim(BBOX[2], BBOX[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values(): spine.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, pad_inches=0)
    plt.close(fig)
    with PILImage.open(output) as image:
        image.convert("RGB").save(output, optimize=True)
    print(f"[figure] {output.relative_to(ROOT)} ({PIXELS}x{PIXELS})")


if __name__ == "__main__":
    render()
