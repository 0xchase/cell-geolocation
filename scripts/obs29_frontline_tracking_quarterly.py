#!/usr/bin/env python3
"""Quarterly version of obs28: the Russian cell-network frontier advancing west
through the Donbas, one panel per 3-month period (2023 Q4 – 2026 Q2).

Same method as obs28_frontline_tracking.py — raw observations (cells geocoded
inside Ukraine only), the documented front (Project Owl OSINT / owlmaps) bold,
and the cell-tower frontier (western edge of the dense Russian cluster per
latitude band) dashed — but grouped by calendar quarter and paired with
quarterly front-line snapshots. Each panel also carries the previous quarter's
front and frontier faint, so quarter-to-quarter movement is visible.

Caveat: observation volume is crawl-biased and thinner per quarter, so early
panels are sparse; read the frontier's position, not the counts.
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
FRONTLINE_GEOJSON = DATA_ROOT / "enrichment" / "frontline_by_quarter.geojson"
PLOTS = ROOT / "plots"
OUTPUT_DPI = 700
PREVIEW_DPI = 200

DONBAS = (35.4, 39.9, 47.2, 50.1)  # xmin, xmax, ymin, ymax
BAND_STEP = 0.1
LON_BIN = 0.05
LON_SIGMA = 2.0
EDGE_FRAC = 0.10
FRONTIER_MIN_RU = 40               # lower than obs28: quarterly data is thinner
NCOLS = 4
RU_COLOR = "#c02a3c"
UA_COLOR = "#2f6f9f"
CELL_FRONTIER_COLOR = "#111111"
FRONT_COLOR = "#7a1f12"
FRONT_PREV_COLOR = "#d98a5a"


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_grid_by_quarter() -> pd.DataFrame:
    return ch_df(
        f"""
        SELECT toStartOfQuarter(g.timestamp) AS q,
               round(g.lat, 3) AS lat, round(g.lon, 3) AS lon,
               countIf(g.mcc = 250) AS ru, countIf(g.mcc = 255) AS ua
        FROM cell.geos AS g
        INNER JOIN (SELECT mcc, mnc, lac, cid, cell_type FROM cell.summary_full
                    WHERE country_iso = 'UA') AS s
            USING (mcc, mnc, lac, cid, cell_type)
        WHERE g.mcc IN (250, 255) AND g.cid > 0 AND NOT (g.lat = 0 AND g.lon = 0)
          AND g.lat BETWEEN {DONBAS[2]} AND {DONBAS[3]}
          AND g.lon BETWEEN {DONBAS[0]} AND {DONBAS[1]}
        GROUP BY q, lat, lon
        HAVING ru + ua > 0
        """
    )


def load_fronts() -> dict[str, list[list[list[float]]]]:
    fc = json.loads(FRONTLINE_GEOJSON.read_text())
    out: dict[str, list[list[list[float]]]] = {}
    for feat in fc["features"]:
        g = feat["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        out[feat["properties"]["quarter"]] = [[[p[0], p[1]] for p in line] for line in lines]
    return out


def quarter_label(q: str) -> str:
    y, m, _ = q.split("-")
    return f"{y} Q{(int(m) - 1) // 3 + 1}"


def _gauss1d(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return np.convolve(a, k / k.sum(), mode="same")


def russian_frontier(gy: pd.DataFrame, band_centers: np.ndarray) -> np.ndarray:
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


def make_figure(grid: pd.DataFrame, fronts: dict[str, list], output: Path, preview: Path | None) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    quarters = sorted(grid["q"].unique())
    band_c = np.arange(DONBAS[2] + BAND_STEP / 2, DONBAS[3], BAND_STEP)
    frontiers = {q: russian_frontier(grid[grid["q"] == q], band_c) for q in quarters}

    nrows = (len(quarters) + NCOLS - 1) // NCOLS
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(3.5 * NCOLS, 3.55 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle("The Russian cell-network frontier advancing west through the Donbas, by quarter",
                 fontsize=14, fontweight="bold")

    for k, q in enumerate(quarters):
        ax = axes[k]
        setup_context_map(ax, DONBAS, countries={"UA", "RU"},
                          admin_names={"Donets'k", "Luhans'k", "Kharkiv", "Dnipropetrovs'k", "Zaporizhzhya"})
        gy = grid[grid["q"] == q]
        for sub, cnt, color, z in [(gy[gy["ua"] > 0], "ua", UA_COLOR, 2.3),
                                   (gy[gy["ru"] > 0], "ru", RU_COLOR, 2.6)]:
            c = sub[cnt].to_numpy()
            ax.scatter(sub["lon"], sub["lat"], s=1.2 + np.log10(c + 1) * 2.2, c=color,
                       alpha=0.42, linewidth=0, rasterized=True, zorder=z)

        prev = quarters[k - 1] if k > 0 else None
        # Documented front: previous quarter faint, this quarter bold.
        if prev and prev in fronts:
            plot_line(ax, fronts[prev], color=FRONT_PREV_COLOR, linewidth=1.1, alpha=0.7, zorder=4)
        if q in fronts:
            plot_line(ax, fronts[q], color=FRONT_COLOR, linewidth=2.6, alpha=0.98, zorder=6)
        # Cell-tower frontier: previous quarter faint dashed, this quarter bold dashed.
        if prev:
            ax.plot(frontiers[prev], band_c, color=CELL_FRONTIER_COLOR, linewidth=1.1,
                    alpha=0.4, linestyle=(0, (4, 3)), zorder=6.2)
        ax.plot(frontiers[q], band_c, color=CELL_FRONTIER_COLOR, linewidth=1.9, alpha=0.95,
                linestyle=(0, (5, 2)), zorder=6.5, solid_capstyle="round")

        obs = int(gy[["ru", "ua"]].to_numpy().sum())
        ax.set_title(quarter_label(q), fontsize=11)
        ax.set_xlim(DONBAS[0], DONBAS[1])
        ax.set_ylim(DONBAS[2], DONBAS[3])
        ax.set_aspect("auto")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
        ax.text(0.02, 0.02, f"{obs:,} obs", transform=ax.transAxes, fontsize=6.5,
                color="#3f3a35", ha="left", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.2})

    for k in range(len(quarters), len(axes)):
        axes[k].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RU_COLOR, markersize=8, label="Russian-network obs (MCC 250)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=UA_COLOR, markersize=8, label="Ukrainian-network obs (MCC 255)"),
        plt.Line2D([0], [0], color=FRONT_COLOR, linewidth=2.6, label="Documented front (this quarter; previous faint)"),
        plt.Line2D([0], [0], color=CELL_FRONTIER_COLOR, linewidth=1.9, linestyle=(0, (5, 2)), label="Cell-tower frontier (this quarter; previous faint)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=True, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.015))
    note = (
        "One 3-month period per panel. Cells geocoded inside Ukraine only: red = Russian-operator, blue = Ukrainian. "
        "Bold line = that quarter's documented front (Project Owl OSINT / owlmaps), previous quarter faint. Dashed = "
        "cell-tower frontier (western edge of the dense Russian cluster per 0.1° latitude band), previous quarter faint. "
        "Volume is crawl-biased and thin early — read positions, not counts."
    )
    fig.text(0.5, -0.035, note, ha="center", va="top", fontsize=8, wrap=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=OUTPUT_DPI, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs29_frontline_tracking_quarterly.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    grid = load_grid_by_quarter()
    fronts = load_fronts()
    make_figure(grid, fronts, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
