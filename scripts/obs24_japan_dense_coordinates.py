#!/usr/bin/env python3
"""Generate a four-panel figure for Japanese dense single-coordinate clusters."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

from plot_helpers import add_osm_basemap
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"

CITY_LABELS = {
    (34.64056, 135.42111): "Osaka / Nanko-kita",
    (26.22083, 127.72556): "Naha / Shuri",
    (34.96028, 135.74695): "Kyoto / Minami",
    (33.61639, 130.43805): "Fukuoka",
    (38.31916, 140.96722): "Sendai",
    (32.78111, 130.70833): "Kumamoto",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_data() -> dict[str, pd.DataFrame]:
    top = ch_df(
        """
        SELECT round(glat,5) AS lat, round(glon,5) AS lon, count() AS cells,
               uniqExact(mcc) AS mccs, arraySort(groupUniqArray(10)(mcc)) AS mcc_list
        FROM cell.summary
        WHERE cid > 0 AND NOT (glat=0 AND glon=0)
        GROUP BY lat, lon
        ORDER BY cells DESC
        LIMIT 20
        """
    )
    mcc_compare = ch_df(
        """
        SELECT mcc, countIf(cells >= 100) AS points_ge100,
               max(cells) AS max_cells,
               quantile(0.99)(cells) AS p99_cells
        FROM
        (
            SELECT mcc, round(glat,5) AS lat, round(glon,5) AS lon, count() AS cells
            FROM cell.summary
            WHERE cid > 0 AND NOT (glat=0 AND glon=0)
            GROUP BY mcc, lat, lon
        )
        GROUP BY mcc
        HAVING max_cells >= 100
        ORDER BY max_cells DESC
        LIMIT 12
        """
    )
    osaka_detail = ch_df(
        """
        SELECT mnc, cell_type, count() AS cells, min(first_seen) AS first_seen, max(last_seen) AS last_seen
        FROM cell.summary
        WHERE cid > 0 AND mcc = 440
          AND abs(glat - 34.64056) < 0.00002 AND abs(glon - 135.42111) < 0.00002
        GROUP BY mnc, cell_type
        ORDER BY cells DESC
        """
    )
    quarters = ch_df(
        """
        SELECT toStartOfQuarter(last_seen) AS quarter, count() AS cells
        FROM cell.summary
        WHERE cid > 0 AND mcc = 440
          AND abs(glat - 34.64056) < 0.00002 AND abs(glon - 135.42111) < 0.00002
        GROUP BY quarter
        ORDER BY quarter
        """
    )
    top["label"] = top.apply(lambda r: CITY_LABELS.get((round(r["lat"], 5), round(r["lon"], 5)), f"{r['lat']:.2f}, {r['lon']:.2f}"), axis=1)
    osaka_detail["first_seen"] = pd.to_datetime(osaka_detail["first_seen"])
    osaka_detail["last_seen"] = pd.to_datetime(osaka_detail["last_seen"])
    quarters["quarter"] = pd.to_datetime(quarters["quarter"])
    return {"top": top, "mcc_compare": mcc_compare, "osaka_detail": osaka_detail, "quarters": quarters}


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    top = data["top"].copy()
    mcc_compare = data["mcc_compare"].copy()
    osaka_detail = data["osaka_detail"].copy()
    quarters = data["quarters"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.04)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.7), constrained_layout=True)
    fig.suptitle("Japanese dense single-coordinate clusters are benign infrastructure, not rogue sites", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    plot_top = top.head(12).sort_values("cells", ascending=True)
    ax.barh(plot_top["label"], plot_top["cells"], color=["#b23a48" if m == 1 else "#8b8b8b" for m in plot_top["mccs"]])
    ax.set_xscale("log")
    ax.set_title("A. The densest coordinates are single-country Japanese points")
    ax.set_xlabel("Distinct cells at exact rounded coordinate (log)")
    ax.set_ylabel("")
    for patch, cells, mccs in zip(ax.patches, plot_top["cells"], plot_top["mccs"], strict=False):
        ax.text(cells * 1.12, patch.get_y() + patch.get_height() / 2, f"{cells:,}; {int(mccs)} MCC", va="center", fontsize=7.2)

    ax = axes[0, 1]
    bbox = (135.405, 135.437, 34.630, 34.652)
    add_osm_basemap(ax, bbox, zoom=15, alpha=0.88, grayscale=True)
    ax.scatter([135.42111], [34.64056], s=190, color="#b23a48", edgecolor="white", linewidth=0.9, zorder=4)
    ax.text(135.42111, 34.646, "Osaka Nanko-kita\n16,205 distinct cells", ha="center", fontsize=7.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0}, zorder=5)
    ax.set_title("B. Top point is a real urban/port coordinate, not blank space")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(0.02, 0.02, "Map data (c) OpenStreetMap contributors", transform=ax.transAxes, fontsize=5.8, color="#555", zorder=7)

    ax = axes[1, 0]
    comp = mcc_compare.sort_values("max_cells", ascending=True)
    ax.barh(comp["mcc"].astype(str), comp["max_cells"], color=["#b23a48" if m == 440 else "#8b8b8b" for m in comp["mcc"]])
    ax.set_xscale("log")
    ax.set_title("C. Japan is the global outlier for point-density maxima")
    ax.set_xlabel("Maximum cells at one coordinate by MCC (log)")
    ax.set_ylabel("MCC")
    for patch, max_cells, points in zip(ax.patches, comp["max_cells"], comp["points_ge100"], strict=False):
        ax.text(max_cells * 1.12, patch.get_y() + patch.get_height() / 2, f"{max_cells:,}; {points} points >=100", va="center", fontsize=7.2)

    ax = axes[1, 1]
    ax.bar(quarters["quarter"], quarters["cells"], width=70, color="#2f6f9f")
    ax.set_title("D. Osaka hotspot is one operator/technology, not multi-country")
    ax.set_xlabel("Last-seen quarter")
    ax.set_ylabel("Cells last seen in quarter")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    if not osaka_detail.empty:
        row = osaka_detail.iloc[0]
        ax.text(
            0.98,
            0.95,
            f"MCC 440 / MNC {int(row['mnc'])}, {row['cell_type']}\n{int(row['cells']):,} distinct cells\n{row['first_seen'].date()} to {row['last_seen'].date()}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.4, "alpha": 0.82, "pad": 2},
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs24_japan_dense_coordinates.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
