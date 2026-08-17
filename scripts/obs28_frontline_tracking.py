#!/usr/bin/env python3
"""The Russian cell-network frontier advancing west through the Donbas, year by year.

Each panel plots that year's raw observations — red = Russian-operator cells
(MCC 250), blue = Ukrainian-operator cells (MCC 255) — with no smoothing. Over
them we draw the "Russian frontier": the western reach of Russian-network
observations per latitude band. The current year's frontier is bold; earlier
years' frontiers are faint, so each panel shows how far the frontier has pushed
west. The documented front (Project Owl OSINT / owlmaps) is kept as a thin
reference.

Caveat: observation volume is crawl-biased (sparse 2023, dense 2026), so read the
frontier's position, not the raw count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helpers import setup_context_map
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
FRONTLINE_GEOJSON = DATA_ROOT / "enrichment" / "frontline_by_year.geojson"
PLOTS = ROOT / "plots"
OUTPUT_DPI = 800
PREVIEW_DPI = 260

DONBAS = (35.4, 39.9, 47.2, 50.1)  # xmin, xmax, ymin, ymax
BAND_STEP = 0.1                    # latitude band for the frontier
LON_BIN = 0.05                     # longitude bin when profiling Russian density
LON_SIGMA = 2.0                    # smooth the longitude profile (bins) before edge-finding
EDGE_FRAC = 0.10                   # western edge = 10% of the band's peak density
FRONTIER_MIN_RU = 60               # min Russian obs in a band to define a frontier
RU_COLOR = "#c02a3c"
UA_COLOR = "#2f6f9f"
CELL_FRONTIER_COLOR = "#111111"
FRONT_COLORS = {2023: "#e0a51e", 2024: "#e8791f", 2025: "#d94f2a", 2026: "#7a1f12"}

CITIES = [
    (38.000, 48.595, "Bakhmut", "5/2023"),
    (37.750, 48.139, "Avdiivka", "2/2024"),
    (37.259, 47.780, "Vuhledar", "10/2024"),
    (37.177, 48.281, "Pokrovsk", "contested ’25"),
]


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_grid_by_year() -> pd.DataFrame:
    # Only cells whose geocoded location is inside Ukraine, so Russia-proper cells
    # (east of the international border) don't confound the occupation frontier.
    return ch_df(
        f"""
        SELECT toYear(g.timestamp) AS yr, round(g.lat, 3) AS lat, round(g.lon, 3) AS lon,
               countIf(g.mcc = 250) AS ru, countIf(g.mcc = 255) AS ua
        FROM cell.geos AS g
        INNER JOIN (SELECT mcc, mnc, lac, cid, cell_type FROM cell.summary_full
                    WHERE country_iso = 'UA') AS s
            USING (mcc, mnc, lac, cid, cell_type)
        WHERE g.mcc IN (250, 255) AND g.cid > 0 AND NOT (g.lat = 0 AND g.lon = 0)
          AND g.lat BETWEEN {DONBAS[2]} AND {DONBAS[3]}
          AND g.lon BETWEEN {DONBAS[0]} AND {DONBAS[1]}
        GROUP BY yr, lat, lon
        HAVING ru + ua > 0
        """
    )


def load_fronts() -> dict[int, list[list[list[float]]]]:
    fc = json.loads(FRONTLINE_GEOJSON.read_text())
    out: dict[int, list[list[list[float]]]] = {}
    for feat in fc["features"]:
        g = feat["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        out[int(feat["properties"]["year"])] = [[[p[0], p[1]] for p in line] for line in lines]
    return out


def _gauss1d(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return np.convolve(a, k / k.sum(), mode="same")


def russian_frontier(gy: pd.DataFrame, band_centers: np.ndarray) -> np.ndarray:
    """Western edge of the dense Russian cluster per latitude band. Build a
    smoothed longitude density profile of Russian obs in the band, then take the
    westernmost longitude where that density rises to EDGE_FRAC of the band's
    peak. The longitude smoothing bridges small gaps and the peak-relative
    threshold ignores the thin roaming-SIM speckle, giving a stable line that
    hugs the occupied mass. Finally smooth over latitude."""
    lon_edges = np.arange(DONBAS[0], DONBAS[1] + LON_BIN, LON_BIN)
    lon_bc = (lon_edges[:-1] + lon_edges[1:]) / 2
    out = np.full(band_centers.shape, np.nan)
    for i, c in enumerate(band_centers):
        sub = gy[(gy["lat"] >= c - BAND_STEP / 2) & (gy["lat"] < c + BAND_STEP / 2)]
        if sub["ru"].sum() < FRONTIER_MIN_RU:
            continue
        hist, _ = np.histogram(sub["lon"], bins=lon_edges, weights=sub["ru"])
        prof = _gauss1d(hist, LON_SIGMA)
        thr = max(4.0, EDGE_FRAC * prof.max())
        dense = np.where(prof >= thr)[0]
        if dense.size:
            out[i] = lon_bc[dense.min()]
    return pd.Series(out).rolling(5, center=True, min_periods=2).median().to_numpy()


def plot_line(ax: plt.Axes, lines: list[list[list[float]]], **kw) -> None:
    for line in lines:
        ax.plot([p[0] for p in line], [p[1] for p in line], solid_capstyle="round", **kw)


def draw_cities(ax: plt.Axes) -> None:
    for lon, lat, name, when in CITIES:
        ax.scatter([lon], [lat], marker="s", s=15, facecolor="white",
                   edgecolor="#111", linewidth=0.8, zorder=8)
        ax.annotate(f"{name}\n{when}", (lon, lat), textcoords="offset points",
                    xytext=(4, 3), fontsize=5.6, color="#1a1a1a", zorder=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 0.6})


def make_figure(grid: pd.DataFrame, fronts: dict[int, list], output: Path, preview: Path | None) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.02)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    years = sorted(fronts)
    band_c = np.arange(DONBAS[2] + BAND_STEP / 2, DONBAS[3], BAND_STEP)
    frontiers = {y: russian_frontier(grid[grid["yr"] == y], band_c) for y in years}

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 12.4), constrained_layout=True)
    fig.suptitle("The Russian cell-network frontier advancing west through the Donbas",
                 fontsize=14.5, fontweight="bold")

    for ax, year in zip(axes.flat, years, strict=False):
        setup_context_map(
            ax, DONBAS, countries={"UA", "RU"},
            admin_names={"Donets'k", "Luhans'k", "Kharkiv", "Dnipropetrovs'k", "Zaporizhzhya"},
        )
        gy = grid[grid["yr"] == year]
        # Raw observations that year: blue = Ukrainian, red = Russian.
        for sub, cnt, color, z in [(gy[gy["ua"] > 0], "ua", UA_COLOR, 2.3),
                                   (gy[gy["ru"] > 0], "ru", RU_COLOR, 2.6)]:
            c = sub[cnt].to_numpy()
            ax.scatter(sub["lon"], sub["lat"], s=1.4 + np.log10(c + 1) * 2.4, c=color,
                       alpha=0.40, linewidth=0, rasterized=True, zorder=z)

        # Documented front: prior years faint, this year bold — the advance.
        for other in years:
            if other >= year:
                continue
            plot_line(ax, fronts[other], color=FRONT_COLORS[other], linewidth=1.2, alpha=0.5, zorder=4)
        plot_line(ax, fronts[year], color=FRONT_COLORS[year], linewidth=3.2, alpha=0.98, zorder=6)
        # Cell-tower-derived Russian frontier, dashed: previous year faint, this year bold.
        prev = [y for y in years if y < year]
        if prev:
            ax.plot(frontiers[prev[-1]], band_c, color=CELL_FRONTIER_COLOR, linewidth=1.3,
                    alpha=0.4, linestyle=(0, (4, 3)), zorder=6.2)
        ax.plot(frontiers[year], band_c, color=CELL_FRONTIER_COLOR, linewidth=2.0, alpha=0.95,
                linestyle=(0, (5, 2)), zorder=6.5, solid_capstyle="round")
        draw_cities(ax)

        obs = int(gy[["ru", "ua"]].to_numpy().sum())
        ax.set_title(f"{year}", fontsize=11.5)
        ax.set_xlim(DONBAS[0], DONBAS[1])
        ax.set_ylim(DONBAS[2], DONBAS[3])
        ax.set_aspect("auto")
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.015, 0.02, f"{obs:,} obs", transform=ax.transAxes, fontsize=7.5,
                color="#3f3a35", ha="left", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5})

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RU_COLOR, markersize=8, label="Russian-network obs (MCC 250)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=UA_COLOR, markersize=8, label="Ukrainian-network obs (MCC 255)"),
        plt.Line2D([0], [0], color=FRONT_COLORS[2026], linewidth=3.2, label="Documented front (this year bold; prior years faint)"),
        plt.Line2D([0], [0], color=CELL_FRONTIER_COLOR, linewidth=2.0, linestyle=(0, (5, 2)), label="Cell-tower frontier (this year; previous year faint)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor="#111", markersize=8, label="City (with fall date)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, frameon=True, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))

    note = (
        "Raw observations, no smoothing, cells geocoded inside Ukraine only (Russia-proper excluded): red = "
        "Russian-operator cells, blue = Ukrainian, that calendar year. Bold line = documented front (Project Owl OSINT / "
        "owlmaps), this year bold and prior years faint, so the real front's westward advance is visible in each panel. "
        "Dashed line = cell-tower frontier: the western edge of the dense Russian cluster inside Ukraine per 0.1° latitude "
        "band (westernmost longitude reaching 10% of the band's peak density, so roaming-SIM speckle is skipped). "
        "The dashed cell frontier tracks the bold front. Counts are crawl-biased — read positions, not volume."
    )
    fig.text(0.5, -0.05, note, ha="center", va="top", fontsize=8, wrap=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=OUTPUT_DPI, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs28_frontline_tracking.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    grid = load_grid_by_year()
    fronts = load_fronts()
    make_figure(grid, fronts, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
