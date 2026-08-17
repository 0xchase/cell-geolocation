#!/usr/bin/env python3
"""Generate a four-panel figure for a cloned T-Mobile NR identity carried through China.

The corrected dataset shows 1158 observations (vs 6 in the deduplicated snapshot):
the genuine cell transmits continuously in Louisville while a copy of its identity
moves Shenzhen/HK -> Shanghai across Feb-May 2026.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_helpers import COUNTRIES_GEOJSON, add_osm_basemap, draw_geojson_layer
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"

MCC, MNC, LAC, CID = 310, 260, 11011072, 4343169326
COLORS = {
    "Louisville, KY": "#b23a48",
    "Shenzhen/Hong Kong": "#2f6f9f",
    "Shanghai": "#c9743a",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_data() -> pd.DataFrame:
    df = ch_df(
        f"""
        SELECT lat, lon, timestamp
        FROM cell.geos
        WHERE mcc = {MCC} AND mnc = {MNC} AND lac = {LAC} AND cid = {CID}
        ORDER BY timestamp
        """
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    def classify(row):
        if row["lon"] < 0:
            return "Louisville, KY"
        return "Shanghai" if row["lon"] > 118 else "Shenzhen/Hong Kong"

    df["endpoint"] = df.apply(classify, axis=1)
    louisville = df[df["endpoint"] == "Louisville, KY"][["lat", "lon"]].mean()
    df["distance_from_louisville_km"] = df.apply(lambda r: haversine_km(louisville["lat"], louisville["lon"], r["lat"], r["lon"]), axis=1)
    df["sequence"] = range(1, len(df) + 1)
    return df


def make_figure(df: pd.DataFrame, output: Path, preview: Path | None) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.04)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.7), constrained_layout=True)
    fig.suptitle("A cloned T-Mobile 5G identity is carried across China while the real cell keeps transmitting", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    bbox = (-125, 125, 10, 55)
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(ax, COUNTRIES_GEOJSON, bbox, facecolor="#f5f1e8", edgecolor="#8a8176", linewidth=0.35, zorder=0)
    endpoints = df.groupby("endpoint", as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"), obs=("lat", "size"))
    ax.plot(endpoints["lon"], endpoints["lat"], color="#555", linewidth=1.2, linestyle="--", zorder=2)
    ax.scatter(endpoints["lon"], endpoints["lat"], s=140, color=[COLORS[e] for e in endpoints["endpoint"]], edgecolor="white", linewidth=0.8, zorder=3)
    for _, row in endpoints.iterrows():
        ax.text(row["lon"], row["lat"] + 3.2, f"{row['endpoint']}\n{int(row['obs'])} reports", ha="center", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0})
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("auto")
    ax.set_title("A. Same NR identity reported 12,700 km apart")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax = axes[0, 1]
    louisville_bbox = (-85.724, -85.708, 38.160, 38.175)
    add_osm_basemap(ax, louisville_bbox, zoom=15, alpha=0.86, grayscale=True)
    ky = df[df["endpoint"] == "Louisville, KY"]
    ax.scatter(ky["lon"], ky["lat"], s=65, color=COLORS["Louisville, KY"], edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_title("B. Kentucky reports are tightly clustered (n=1,041)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.text(0.02, 0.02, "Map data (c) OpenStreetMap contributors", transform=ax.transAxes, fontsize=5.8, color="#555", zorder=7)

    ax = axes[1, 0]
    cn_bbox = (110.0, 126.0, 19.0, 34.0)
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(ax, COUNTRIES_GEOJSON, cn_bbox, facecolor="#f5f1e8", edgecolor="#8a8176", linewidth=0.4, zorder=0)
    cn = df[df["endpoint"] != "Louisville, KY"]
    for name, grp in cn.groupby("endpoint"):
        ax.scatter(grp["lon"], grp["lat"], s=90, color=COLORS[str(name)], edgecolor="white",
                   linewidth=0.6, zorder=4, label=f"{name} (n={len(grp)})")
        ax.annotate(f"{grp['timestamp'].min():%b %d} - {grp['timestamp'].max():%b %d %Y}",
                    xy=(grp["lon"].mean(), grp["lat"].mean() + 0.9), ha="center", fontsize=7.2,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0}, zorder=5)
    ax.set_xlim(cn_bbox[0], cn_bbox[1])
    ax.set_ylim(cn_bbox[2], cn_bbox[3])
    ax.legend(loc="lower left", frameon=True, fontsize=7.5)
    ax.set_title("C. The China-side reports move: Shenzhen/HK, then Shanghai")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax = axes[1, 1]
    sns.scatterplot(data=df, x="timestamp", y="distance_from_louisville_km", hue="endpoint",
                    palette=COLORS, s=18, alpha=0.65, edgecolor="none", ax=ax)
    ax.set_title("D. Home and clone are observed concurrently, Feb-May 2026")
    ax.set_xlabel("")
    ax.set_ylabel("Distance from Kentucky cluster (km)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="", loc="upper left", frameon=True, fontsize=8)
    ax.text(
        0.98,
        0.05,
        f"PLMN {MCC}/{MNC}; TAC {LAC}; NR CID {CID}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.4, "alpha": 0.78, "pad": 2},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs17_tmobile_nr_pingpong.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    df = load_data()
    make_figure(df, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
