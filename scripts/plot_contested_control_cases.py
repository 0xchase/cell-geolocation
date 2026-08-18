#!/usr/bin/env python3
"""Render the focused South Ossetia map and Karabakh activity timeline."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from extract_georgia_russia_density import BBOX as GEORGIA_GRID_BBOX
from plot_georgia_russia_density import density_rgba, load_grid
from plot_helpers import add_osm_basemap, draw_geojson_layer


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
DATA = ROOT / "data" / "out-of-country" / "contested-territories"
ADMIN0 = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
ADMIN1 = ROOT / "data" / "reference" / "ne_10m_admin_1_georgia.geojson"
SOUTH_OSSETIA_BBOX = (43.35, 44.75, 41.75, 42.65)
RED = "#c02a3c"
BLUE = "#2f6f9f"
PIXELS = 900
DPI = 300


def outlined_label(ax: plt.Axes, label: str, lon: float, lat: float, **kwargs) -> None:
    text = ax.annotate(
        label,
        (lon, lat),
        fontsize=7.4,
        color="#292724",
        weight="semibold",
        zorder=6,
        **kwargs,
    )
    text.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])


def render_south_ossetia() -> None:
    russian, georgian = load_grid()
    side = PIXELS / DPI
    fig = plt.figure(figsize=(side, side), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor("#dceaf2")
    add_osm_basemap(
        ax,
        SOUTH_OSSETIA_BBOX,
        zoom=9,
        alpha=1.0,
        grayscale=False,
        zorder=0,
        source="carto_voyager_nolabels",
    )
    draw_geojson_layer(
        ax,
        ADMIN1,
        SOUTH_OSSETIA_BBOX,
        facecolor="none",
        edgecolor="#817b74",
        linewidth=0.48,
        alpha=0.88,
        zorder=2,
    )
    draw_geojson_layer(
        ax,
        ADMIN0,
        SOUTH_OSSETIA_BBOX,
        countries={"GE", "RU"},
        facecolor="none",
        edgecolor="#292724",
        linewidth=0.65,
        alpha=0.92,
        zorder=2.2,
    )
    ax.imshow(
        density_rgba(russian, georgian),
        origin="lower",
        extent=GEORGIA_GRID_BBOX,
        interpolation="nearest",
        aspect="auto",
        zorder=3,
    )
    ax.plot(43.9700, 42.2250, marker="o", markersize=3.0, color="#292724", zorder=6)
    outlined_label(ax, "Tskhinvali", 43.9700, 42.2250, xytext=(4, 3), textcoords="offset points")
    ax.plot(44.1120, 41.9840, marker="o", markersize=2.5, color="#292724", zorder=6)
    outlined_label(ax, "Gori", 44.1120, 41.9840, xytext=(4, -10), textcoords="offset points")
    outlined_label(ax, "South Ossetia", 44.03, 42.46, ha="center")
    ax.set_xlim(SOUTH_OSSETIA_BBOX[0], SOUTH_OSSETIA_BBOX[1])
    ax.set_ylim(SOUTH_OSSETIA_BBOX[2], SOUTH_OSSETIA_BBOX[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    output = FIGS / "south_ossetia_cellular_density.png"
    fig.savefig(output, dpi=DPI, pad_inches=0)
    plt.close(fig)
    print(f"[figure] {output.relative_to(ROOT)}")


def load_timeline() -> dict[str, list[tuple[date, int]]]:
    series: dict[str, list[tuple[date, int]]] = {"Azerbaijani": [], "Armenian": []}
    path = DATA / "karabakh-quarterly-active.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            series[row["network"]].append(
                (date.fromisoformat(row["quarter"]), int(row["active_cells"]))
            )
    return series


def render_karabakh_timeline() -> None:
    series = load_timeline()
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=DPI)
    for network, color in (("Azerbaijani", RED), ("Armenian", BLUE)):
        points = sorted(series[network])
        dates = [point[0] for point in points]
        values = [point[1] for point in points]
        ax.plot(
            dates,
            values,
            color=color,
            marker="o",
            markersize=3.8,
            linewidth=1.8,
            label=network,
            zorder=3,
        )
        ax.annotate(
            f"{values[0]:,}",
            (dates[0], values[0]),
            xytext=(1, 7 if network == "Azerbaijani" else -13),
            textcoords="offset points",
            fontsize=7,
            color=color,
            ha="left",
        )
        ax.annotate(
            f"{values[-1]:,}",
            (dates[-1], values[-1]),
            xytext=(-1, 7 if network == "Azerbaijani" else -13),
            textcoords="offset points",
            fontsize=7,
            color=color,
            ha="right",
        )
    ax.set_yscale("log")
    ax.set_ylim(1, 2000)
    ax.set_ylabel("Active cell identities per quarter", fontsize=8)
    ax.set_xlabel("", fontsize=8)
    ax.grid(axis="y", which="major", color="#d9d4ce", linewidth=0.55)
    ax.grid(axis="y", which="minor", color="#ece8e3", linewidth=0.35)
    # Supply explicit half-year labels to keep the compact panel readable.
    ticks = [date(2023, 10, 1), date(2024, 4, 1), date(2024, 10, 1),
             date(2025, 4, 1), date(2025, 10, 1), date(2026, 4, 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["2023 Q4", "2024 Q2", "2024 Q4", "2025 Q2", "2025 Q4", "2026 Q2"], rotation=35, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="center right", frameon=False, fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.35)
    pdf = FIGS / "karabakh_operator_timeline.pdf"
    png = FIGS / "karabakh_operator_timeline.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    print(f"[figure] {pdf.relative_to(ROOT)}")


def main() -> None:
    render_south_ossetia()
    render_karabakh_timeline()


if __name__ == "__main__":
    main()
