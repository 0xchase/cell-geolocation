#!/usr/bin/env python3
"""Generate a four-panel figure for long-range displaced cell identities."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_helpers import COUNTRIES_GEOJSON, draw_geojson_layer
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"

# The previous version hardcoded the ten top rows from the deduplicated
# snapshot. Those are void: that table preserved ~0.45% of observations, so its
# displacement ranking was an artifact. The list is now derived by query.
TOP_N = 10

CAUSE_COLORS = {
    "sparse (few reports)": "#8b8b8b",
    "systematic clone/replay": "#b23a48",
    "test/equipment": "#2f6f9f",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def region_of(lat: float, lon: float) -> str:
    """Coarse label for an endpoint, enough to describe an excursion."""
    if 105 <= lon <= 125 and 18 <= lat <= 42:
        return "China/HK"
    if -85 <= lon <= -30 and -35 <= lat <= 13:
        return "S. America"
    if -170 <= lon <= -50:
        return "N. America"
    if 20 <= lon <= 45 and 44 <= lat <= 56:
        return "Ukraine/RU"
    if -12 <= lon <= 40 and 35 <= lat <= 72:
        return "Europe"
    if 125 <= lon <= 150 and 30 <= lat <= 46:
        return "Japan/Korea"
    return f"lat {lat:.0f} / lon {lon:.0f}"


def cause(row) -> str:
    """Classify by evidence, not by a hand-written label.

    A handful of observations at the far endpoint is consistent with a stray
    positioning error; a sustained far-endpoint presence is not.
    """
    if row["mcc"] in (1, 999) or 100 <= row["mcc"] <= 199 or 800 <= row["mcc"] <= 899:
        return "test/equipment"
    # In the corrected data every top hit has hundreds of observations, so the
    # old "one-off outlier" class (2 reports) no longer exists at the top of the
    # ranking. Separate by how sustained the identity is instead.
    if row["obs"] >= 50:
        return "systematic clone/replay"
    return "sparse (few reports)"


def load_data() -> dict[str, pd.DataFrame]:
    top = ch_df(
        f"""
        SELECT mcc, mnc, lac, cid, cell_type, obs, n_pos,
               round(greatCircleDistance(lon_min, lat_min, lon_max, lat_max)/1000, 1) AS bbox_km,
               round(glat, 3) AS home_lat, round(glon, 3) AS home_lon,
               round(lat_min, 3) AS lat_min, round(lon_min, 3) AS lon_min,
               round(lat_max, 3) AS lat_max, round(lon_max, 3) AS lon_max
        FROM cell.summary
        WHERE cid > 0 AND NOT (glat = 0 AND glon = 0) AND n_pos > 1
        ORDER BY bbox_km DESC
        LIMIT {TOP_N}
        """
    )
    top["cause"] = top.apply(cause, axis=1)
    # Describe each row by where it sits and how far the excursion reaches.
    def away_corner(r):
        best, best_d = None, -1.0
        for la in (r["lat_min"], r["lat_max"]):
            for lo in (r["lon_min"], r["lon_max"]):
                d = (la - r["home_lat"]) ** 2 + (lo - r["home_lon"]) ** 2
                if d > best_d:
                    best, best_d = (la, lo), d
        return best

    top["label"] = top.apply(
        lambda r: f"{region_of(r['home_lat'], r['home_lon'])} <-> {region_of(*away_corner(r))}",
        axis=1,
    )
    ukraine = ch_df(
        """
        SELECT count() AS moved_ids, min(bbox_km) AS min_km, max(bbox_km) AS max_km
        FROM
        (
            SELECT mnc,lac,cid,cell_type,count() AS obs,
                   greatCircleDistance(min(lon),min(lat),max(lon),max(lat))/1000 AS bbox_km
            FROM cell.geos
            WHERE cid > 0 AND mcc = 255 AND NOT (lat = 0 AND lon = 0)
            GROUP BY mnc,lac,cid,cell_type
            HAVING obs > 1 AND bbox_km > 500
        )
        """
    )
    picks = top.head(4)
    ors = " OR ".join(
        f"(mcc={r.mcc} AND mnc={r.mnc} AND lac={r.lac} AND cid={r.cid})"
        for r in picks.itertuples()
    )
    examples = ch_df(
        f"""
        SELECT mcc, mnc, lac, cid, cell_type,
               round(lat, 3) AS lat, round(lon, 3) AS lon,
               min(timestamp) AS timestamp, count() AS obs
        FROM cell.geos
        WHERE cid > 0 AND ({ors})
        GROUP BY mcc, mnc, lac, cid, cell_type, lat, lon
        ORDER BY mcc, mnc, lac, cid, timestamp
        """
    )
    examples["timestamp"] = pd.to_datetime(examples["timestamp"])
    examples["track"] = examples.apply(lambda r: f"{int(r['mcc'])}/{int(r['mnc'])}/{int(r['lac'])}/{int(r['cid'])}", axis=1)
    track_labels = {
        f"{r.mcc}/{r.mnc}/{r.lac}/{r.cid}": f"{r.mcc}/{r.mnc} {r.label}"
        for r in picks.itertuples()
    }
    examples["track_label"] = examples["track"].map(track_labels)
    examples = examples[examples["track_label"].notna()]
    return {"top": top, "ukraine": ukraine, "examples": examples}


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    top = data["top"].copy()
    ukraine = data["ukraine"].iloc[0]
    examples = data["examples"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.04)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.7), constrained_layout=True)
    fig.suptitle("Long-range displaced cell identities expose clones, test leakage, and one-off errors", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    bars = top.sort_values("bbox_km", ascending=True)
    labels = bars.apply(
        lambda r: f"{r['mcc']}/{r['mnc']} {r['cell_type']} cid {int(r['cid'])}\n{r['label']}", axis=1
    )
    ax.barh(labels, bars["bbox_km"], color=[CAUSE_COLORS[c] for c in bars["cause"]])
    ax.axvline(500, color="#333", linestyle="--", linewidth=0.9)
    ax.set_title("A. Top detector hits are physically impossible for fixed cells")
    ax.set_xlabel("Bounding-box span (km)")
    ax.set_ylabel("")
    for patch, km in zip(ax.patches, bars["bbox_km"], strict=False):
        ax.text(km + 250, patch.get_y() + patch.get_height() / 2, f"{km:,.0f} km", va="center", fontsize=7)

    ax = axes[0, 1]
    bbox = (-125, 125, -40, 55)
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(ax, COUNTRIES_GEOJSON, bbox, facecolor="#f5f1e8", edgecolor="#8a8176", linewidth=0.35, zorder=0)
    for label, group in examples.groupby("track_label"):
        color = "#b23a48" if "clone" in label or "T-Mobile" in label else "#2f6f9f" if "Shenzhen" in label else "#8b8b8b"
        ax.plot(group["lon"], group["lat"], linestyle="--", linewidth=1.0, color=color, alpha=0.75, zorder=2)
        ax.scatter(group["lon"], group["lat"], s=45, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        r = group.iloc[-1]
        ax.text(r["lon"], r["lat"] + 2.2, label, ha="center", fontsize=6.8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.7}, zorder=4)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("auto")
    ax.set_title("B. Example tracks jump across continents")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax = axes[1, 0]
    sns.scatterplot(data=top, x="obs", y="bbox_km", hue="cause", style="cell_type", palette=CAUSE_COLORS, s=90, edgecolor="white", linewidth=0.7, ax=ax)
    ax.axhline(500, color="#333", linestyle="--", linewidth=0.9)
    ax.set_title("C. Every top hit is sustained, not a stray report")
    ax.set_xlabel("Observation count for identity")
    ax.set_ylabel("Bounding-box span (km)")
    ax.legend(title="", frameon=True, fontsize=8, loc="lower right")

    ax = axes[1, 1]
    caveat = pd.DataFrame(
        {
            "metric": ["Ukrainian moved IDs", "Ukraine min span", "Ukraine max span", "Detector threshold"],
            "value": [ukraine["moved_ids"], ukraine["min_km"], ukraine["max_km"], 500],
        }
    )
    ax.barh(caveat["metric"], caveat["value"], color=["#b23a48", "#b23a48", "#b23a48", "#8b8b8b"])
    ax.set_xscale("log")
    ax.set_title("D. Ukrainian concentration validates the Lima follow-up")
    ax.set_xlabel("Count or km (log scale)")
    ax.set_ylabel("")
    for patch, value in zip(ax.patches, caveat["value"], strict=False):
        label = f"{int(value):,}" if value >= 100 else f"{value:g}"
        ax.text(value * 1.12, patch.get_y() + patch.get_height() / 2, label, va="center", fontsize=8)
    ax.text(
        0.98,
        0.05,
        "Exact: corrected source retains all observations (no dedup)",
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
    parser.add_argument("--output", type=Path, default=PLOTS / "obs21_long_range_displaced_cells.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
