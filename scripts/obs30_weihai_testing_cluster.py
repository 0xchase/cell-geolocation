#!/usr/bin/env python3
"""A second Shenzhen-class equipment-test cluster at Weihai, Shandong.

New finding, not a revision: this cluster is invisible in the deduplicated
snapshot. It surfaced by chasing the top long-range displaced identities in the
corrected data, which are dominated by Brazilian GSM cells whose far endpoint is
a single point in Weihai. Individual cells show sustained presence there (e.g.
724/4/50847/31011109: 832 observations at home in Itajai, 99 at Weihai across
Dec 2024 - Mar 2025), which is far too persistent to be positioning error.

The foreign-identity activity is episodic rather than open-ended: it ramps from
mid-2024, peaks Dec 2024 - Mar 2025, and decays to nothing by roughly Sep 2025,
while the local Chinese cells at the same point continue throughout. That shape
- a bounded campaign against a persistent local background - is the substantive
result, and it is only visible because the corrected source retains history.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_helpers import COUNTRIES_GEOJSON, draw_geojson_layer
from ch_remote import ch_df

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"

# Tight box around the cluster point (37.33 N, 122.03 E).
BOX = (37.30, 37.36, 122.00, 122.06)

MCC_NAME = {
    234: "UK", 240: "Sweden", 250: "Russia", 257: "Belarus", 415: "Lebanon",
    425: "Israel", 452: "Vietnam", 460: "China", 520: "Thailand",
    525: "Singapore", 724: "Brazil", 740: "Ecuador",
}


def load_data() -> dict[str, pd.DataFrame]:
    by_mcc = ch_df(
        f"""
        SELECT mcc, uniqExact((mnc, lac, cid, cell_type)) AS cells, count() AS obs,
               uniqExact(mnc) AS operators, min(timestamp) AS first, max(timestamp) AS last
        FROM cell.geos
        WHERE lat BETWEEN {BOX[0]} AND {BOX[1]}
          AND lon BETWEEN {BOX[2]} AND {BOX[3]} AND cid > 0
        GROUP BY mcc ORDER BY cells DESC
        """
    )
    timeline = ch_df(
        f"""
        SELECT toStartOfMonth(timestamp) AS month,
               multiIf(mcc = 460, 'China (local)', 'Foreign') AS origin,
               count() AS obs, uniqExact((mcc, mnc, lac, cid)) AS cells
        FROM cell.geos
        WHERE lat BETWEEN {BOX[0]} AND {BOX[1]}
          AND lon BETWEEN {BOX[2]} AND {BOX[3]} AND cid > 0
        GROUP BY month, origin ORDER BY month
        """
    )
    by_mcc["country"] = by_mcc["mcc"].map(MCC_NAME).fillna(by_mcc["mcc"].astype(str))
    by_mcc["first"] = pd.to_datetime(by_mcc["first"])
    by_mcc["last"] = pd.to_datetime(by_mcc["last"])
    timeline["month"] = pd.to_datetime(timeline["month"])
    return {"by_mcc": by_mcc, "timeline": timeline}


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    by_mcc = data["by_mcc"].copy()
    timeline = data["timeline"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.04)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=True)
    fig.suptitle(
        "A second equipment-test cluster: one point in Weihai, Shandong radiates 12 countries' cell identities",
        fontsize=13.5, fontweight="bold",
    )

    # A. Where it is.
    ax = axes[0, 0]
    bbox = (105, 135, 18, 47)
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(ax, COUNTRIES_GEOJSON, bbox, facecolor="#f5f1e8",
                       edgecolor="#8a8176", linewidth=0.4, zorder=0)
    ax.scatter([122.03], [37.33], s=190, color="#b23a48", edgecolor="white",
               linewidth=1.0, zorder=4)
    ax.text(122.03, 38.6, "Weihai\n(this finding)", ha="center", fontsize=8.5, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.4}, zorder=5)
    # Shenzhen, for scale against the cluster the paper already documents.
    ax.scatter([114.0], [22.6], s=120, color="#2f6f9f", edgecolor="white",
               linewidth=0.9, zorder=4)
    ax.text(114.0, 20.6, "Shenzhen\n(obs16)", ha="center", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2}, zorder=5)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A. A single coastal point in Shandong")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # B. Who shows up there.
    ax = axes[0, 1]
    bars = by_mcc.sort_values("cells")
    colors = ["#8b8b8b" if m == 460 else "#b23a48" for m in bars["mcc"]]
    ax.barh(bars["country"] + " (" + bars["mcc"].astype(str) + ")", bars["cells"], color=colors)
    ax.set_xscale("log")
    for patch, n in zip(ax.patches, bars["cells"], strict=False):
        ax.text(n * 1.15, patch.get_y() + patch.get_height() / 2, f"{n:,}", va="center", fontsize=7.5)
    ax.set_title("B. 12 country codes at one coordinate (grey = local)")
    ax.set_xlabel("Distinct cell identities (log)")
    ax.set_xlim(0.7, bars["cells"].max() * 6)

    # C. How long each has been present.
    ax = axes[1, 0]
    span = by_mcc.sort_values("first")
    for i, row in enumerate(span.itertuples()):
        color = "#8b8b8b" if row.mcc == 460 else "#b23a48"
        ax.plot([row.first, row.last], [i, i], linewidth=4.0, color=color,
                solid_capstyle="butt", alpha=0.85)
    ax.set_yticks(range(len(span)))
    ax.set_yticklabels(span["country"] + " (" + span["mcc"].astype(str) + ")", fontsize=8)
    ax.set_title("C. Each foreign code appears for months, then goes quiet")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)

    # D. Local vs foreign over time.
    ax = axes[1, 1]
    for origin, grp in timeline.groupby("origin"):
        grp = grp.sort_values("month")
        ax.plot(grp["month"], grp["cells"], marker="o", markersize=3.2, linewidth=1.6,
                color="#8b8b8b" if "China" in str(origin) else "#b23a48", label=str(origin))
    ax.set_yscale("log")
    ax.set_title("D. The foreign burst runs mid-2024 to late-2025, then stops")
    ax.set_ylabel("Distinct identities per month (log)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="", frameon=True, fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs30_weihai_testing_cluster.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    make_figure(load_data(), args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
