#!/usr/bin/env python3
"""Generate a four-panel figure for MCC 553/Tuvalu cells at Fort Bragg."""

from __future__ import annotations

import argparse
import json
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import seaborn as sns

from plot_helpers import add_osm_basemap, setup_context_map
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"
MIL_POLY = DATA_ROOT / "enrichment" / "military_poly.jsonl"

COLORS = {
    "lte": "#b23a48",
    "gsm": "#2f6f9f",
    96: "#b23a48",
    97: "#2f6f9f",
    987: "#4f7f52",
}

FORT_BBOX = (-79.40, -78.92, 35.02, 35.27)
TUVALU_BBOX = (176.0, 180.1, -11.3, -5.4)


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def fort_bragg_polygons() -> list[list[tuple[float, float]]]:
    def walk(obj):
        if (
            isinstance(obj, list)
            and obj
            and isinstance(obj[0], list)
            and len(obj[0]) == 2
            and all(isinstance(v, (int, float)) for v in obj[0])
        ):
            yield [(float(lon), float(lat)) for lon, lat in obj]
        elif isinstance(obj, list):
            for item in obj:
                yield from walk(item)

    rings: list[list[tuple[float, float]]] = []
    with open(MIL_POLY) as f:
        for line in f:
            if '"name": "Fort Bragg"' not in line:
                continue
            row = json.loads(line)
            rings.extend(walk(row["poly"]))
    return rings


def load_data() -> dict[str, pd.DataFrame]:
    countries = ch_df(
        """
        SELECT country_iso, country, count() AS cells
        FROM cell.summary_full
        WHERE cid > 0 AND mcc = 553
        GROUP BY country_iso, country
        ORDER BY cells DESC
        """
    )

    structure = ch_df(
        """
        SELECT mnc, cell_type, count() AS cells,
               min(cid) AS min_cid, max(cid) AS max_cid,
               uniqExact(lac) AS lacs,
               uniqExact(cid) AS cids,
               min(first_seen) AS first_seen, max(last_seen) AS last_seen
        FROM cell.summary_full
        WHERE cid > 0 AND mcc = 553
        GROUP BY mnc, cell_type
        ORDER BY cell_type, mnc
        """
    )

    cells = ch_df(
        """
        SELECT mnc, cell_type, lac, cid, obs, first_seen, last_seen, glat AS lat, glon AS lon
        FROM cell.summary_full
        WHERE cid > 0 AND mcc = 553
        ORDER BY first_seen
        """
    )
    cells["first_seen"] = pd.to_datetime(cells["first_seen"])
    cells["last_seen"] = pd.to_datetime(cells["last_seen"])
    cells["span_days"] = (cells["last_seen"] - cells["first_seen"]).dt.total_seconds() / 86400

    quarters = ch_df(
        """
        SELECT toDate(toStartOfQuarter(first_seen)) AS quarter, mnc, count() AS new_cells
        FROM cell.summary_full
        WHERE cid > 0 AND mcc = 553
        GROUP BY quarter, mnc
        ORDER BY quarter, mnc
        """
    )
    quarters["quarter"] = pd.to_datetime(quarters["quarter"])
    quarters["quarter_label"] = quarters["quarter"].map(lambda d: f"{d.year} Q{((d.month - 1) // 3) + 1}")

    overview = ch_df(
        """
        SELECT
            count() AS cell_count,
            sum(obs) AS observation_count,
            uniqExact(mnc) AS mncs,
            uniqExact(lac) AS lacs,
            uniqExact(cid) AS cids,
            min(first_seen) AS min_first_seen,
            max(last_seen) AS max_last_seen,
            quantile(0.5)(dateDiff('day', first_seen, last_seen)) AS median_span_days
        FROM cell.summary_full
        WHERE cid > 0 AND mcc = 553
        """
    )
    overview["min_first_seen"] = pd.to_datetime(overview["min_first_seen"])
    overview["max_last_seen"] = pd.to_datetime(overview["max_last_seen"])

    return {"countries": countries, "structure": structure, "cells": cells, "quarters": quarters, "overview": overview}


def draw_world_context(ax: plt.Axes, cells: pd.DataFrame, overview: pd.Series) -> None:
    ax.axis("off")
    ax.set_title("A. MCC 553 belongs to Tuvalu, but every observed cell is at Fort Bragg")
    left = ax.inset_axes([0.04, 0.28, 0.32, 0.50])
    right = ax.inset_axes([0.44, 0.24, 0.54, 0.56])

    setup_context_map(
        left,
        TUVALU_BBOX,
        countries={"TV", "TUV", "Tuvalu", "Fiji", "FJ"},
        label_points=[("Tuvalu", 179.2, -8.5), ("Fiji", 178.1, -17.8)],
    )
    left.scatter([], [], color="#b23a48", label="MCC 553 cells")
    left.text(
        0.50,
        0.05,
        "0 cells in Tuvalu",
        transform=left.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.35, "alpha": 0.82, "pad": 1.5},
        zorder=5,
    )

    setup_context_map(
        right,
        (-84.5, -75.0, 32.7, 37.4),
        countries={"US", "USA", "United States of America"},
        label_points=[("Fort Bragg", -79.36, 35.12), ("NC", -79.8, 35.7)],
    )
    right.scatter(cells["lon"], cells["lat"], s=20, color="#b23a48", alpha=0.72, edgecolor="white", linewidth=0.4, zorder=4)
    right.text(
        0.02,
        0.04,
        f"{int(overview['cell_count'])} cells, {int(overview['observation_count'])} observations\n{overview['min_first_seen'].date()} to {overview['max_last_seen'].date()}",
        transform=right.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.35, "alpha": 0.82, "pad": 1.2},
        zorder=5,
    )
    for panel in [left, right]:
        panel.set_xlabel("")
        panel.set_ylabel("")
        panel.tick_params(labelsize=6)


def draw_fort_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    ax.set_title("C. Most cells cluster in western Fort Bragg training areas")
    ax.set_facecolor("#dceaf2")
    add_osm_basemap(ax, FORT_BBOX, zoom=12, alpha=0.58, grayscale=True, zorder=0)
    for ring in fort_bragg_polygons():
        patch = Polygon(ring, closed=True, facecolor="#f2ead8", edgecolor="#7a7065", linewidth=0.7, alpha=0.72, zorder=1)
        ax.add_patch(patch)
    for tech, df in cells.groupby("cell_type"):
        ax.scatter(
            df["lon"],
            df["lat"],
            s=42 if tech == "lte" else 52,
            color=COLORS[tech],
            label=tech.upper(),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
    ax.text(
        -79.355,
        35.116,
        "main cluster",
        fontsize=7.5,
        ha="center",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        zorder=5,
    )
    ax.set_xlim(FORT_BBOX[0], FORT_BBOX[1])
    ax.set_ylim(FORT_BBOX[2], FORT_BBOX[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, color="white", alpha=0.7)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="", loc="lower left", frameon=True, fontsize=8)


def draw_lifetime_panel(ax: plt.Axes, cells: pd.DataFrame) -> None:
    plot = cells.sort_values(["first_seen", "mnc", "cid"]).copy().reset_index(drop=True)
    plot["label"] = plot.apply(lambda r: f"{int(r.mnc)}/{r.cell_type.upper()} {int(r.cid)}", axis=1)
    ax.set_title("D. Identities persist for months, not exercise-length bursts")
    for i, row in plot.iterrows():
        start = mdates.date2num(row["first_seen"])
        end = mdates.date2num(row["last_seen"])
        width = max(end - start, 0.55)
        ax.barh(i, width, left=start, height=0.68, color=COLORS.get(int(row["mnc"]), "#8a8f98"), alpha=0.82)
        ax.text(end + 3, i, f"{int(row.obs)} obs", va="center", fontsize=6.5)
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels(plot["label"], fontsize=5.6)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.set_xlabel("First to last observation")
    ax.set_ylabel("")
    ax.grid(True, axis="x", linewidth=0.35, alpha=0.7)


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    countries = data["countries"].copy()
    structure = data["structure"].copy()
    cells = data["cells"].copy()
    quarters = data["quarters"].copy()
    overview = data["overview"].iloc[0]

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.7), constrained_layout=True)
    fig.suptitle(
        "Tuvalu MCC 553 appears only at Fort Bragg, not in Tuvalu",
        fontsize=14,
        fontweight="bold",
    )

    # A. Tuvalu-vs-Fort Bragg context.
    ax = axes[0, 0]
    draw_world_context(ax, cells, overview)

    # B. Private-network-like PLMN/LAC structure.
    ax = axes[0, 1]
    structure["row"] = structure.apply(lambda r: f"MNC {int(r.mnc)} / {r.cell_type.upper()}", axis=1)
    structure = structure.sort_values("cells", ascending=True)
    ax.barh(structure["row"], structure["cells"], color=[COLORS.get(int(m), "#8a8f98") for m in structure["mnc"]])
    for patch, cells_n, lacs, cids, min_cid, max_cid in zip(ax.patches, structure["cells"], structure["lacs"], structure["cids"], structure["min_cid"], structure["max_cid"], strict=False):
        ax.text(cells_n + 0.25, patch.get_y() + patch.get_height() / 2, f"{cells_n} IDs; {int(lacs)} LACs; CID {min_cid}-{max_cid}", va="center", fontsize=7)
    ax.set_title("B. Three private-looking MNCs, tiny LAC/CID ranges")
    ax.set_xlabel("Cells")
    ax.set_ylabel("")
    ax.set_xlim(0, max(structure["cells"]) * 1.75)

    # C. Fort Bragg local map with OSM military polygon.
    ax = axes[1, 0]
    draw_fort_map(ax, cells)

    # D. Per-cell recurrence and ephemerality.
    ax = axes[1, 1]
    draw_lifetime_panel(ax, cells)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS / "obs12_fort_bragg_mcc553.pdf",
        help="PDF output path.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
