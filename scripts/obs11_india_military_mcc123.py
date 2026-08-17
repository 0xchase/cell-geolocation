#!/usr/bin/env python3
"""Generate a four-panel figure for India's apparent MCC 123/45 defense LTE."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_helpers import setup_context_map
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"

COLORS = {
    "India": "#b23a48",
    "Other": "#8a8f98",
    "LTE": "#2f6f9f",
}

BASE_SHORT = {
    "Sheikh ul-Alam International Airport": "Srinagar airport",
    "Leh Kushok Bakula Rimpochee Airport": "Leh airport",
    "Karwar Naval Base - INS Kadamba": "Karwar / INS Kadamba",
    "Indian Naval Academy Ezhimala": "Naval Academy Ezhimala",
    "INS Varsha": "INS Varsha",
    "Mahajan Field Firing Ranges": "Mahajan ranges",
    "Ramgarh Test Range": "Ramgarh test range",
    "Naliya Air Force Station": "Naliya AFS",
    "Sukna Cantonment": "Sukna Cantonment",
    "Yol Camp": "Yol Camp",
    "Tattoo Ground Army Garrison": "Tattoo Ground",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_data() -> dict[str, pd.DataFrame]:
    composition = ch_df(
        """
        SELECT
            concat('123/', toString(mnc), ' ', toString(cell_type)) AS plmn,
            mnc,
            cell_type,
            count() AS cells,
            countIf(country_iso = 'IN') / count() AS india_frac
        FROM cell.summary_full
        WHERE cid > 0 AND mcc = 123
        GROUP BY plmn, mnc, cell_type
        ORDER BY cells DESC
        LIMIT 8
        """
    )
    composition["india_pct"] = composition["india_frac"] * 100

    countries = ch_df(
        """
        SELECT
            multiIf(country_iso = 'IN', 'India', country = '', 'Ungeocoded', country) AS country_label,
            count() AS cells
        FROM cell.summary_full
        WHERE cid > 0 AND mcc = 123 AND mnc = 45 AND cell_type = 'lte'
        GROUP BY country_label
        ORDER BY cells DESC
        LIMIT 10
        """
    )

    bases = ch_df(
        """
        SELECT
            base,
            any(country) AS country,
            count() AS cells,
            min(first_seen) AS first_seen,
            max(last_seen) AS last_seen,
            avg(glat) AS lat,
            avg(glon) AS lon
        FROM cell.mil_cells
        WHERE cid > 0 AND mcc = 123 AND mnc = 45 AND cell_type = 'lte'
        GROUP BY base
        ORDER BY cells DESC, base
        """
    )
    bases["base_label"] = bases["base"].map(BASE_SHORT).fillna(bases["base"])
    bases["country_group"] = bases["country"].map(lambda c: "India" if c == "India" or c == "" else "Other")
    bases["first_seen"] = pd.to_datetime(bases["first_seen"])
    bases["last_seen"] = pd.to_datetime(bases["last_seen"])
    bases["span_days"] = (bases["last_seen"] - bases["first_seen"]).dt.total_seconds() / 86400

    return {"composition": composition, "countries": countries, "bases": bases}


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    composition = data["composition"].copy()
    countries = data["countries"].copy()
    bases = data["bases"].copy()

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
        "Unassigned MCC 123/45 LTE clusters around Indian military geography",
        fontsize=14,
        fontweight="bold",
    )

    # A. MCC 123 is dominated by one LTE PLMN.
    ax = axes[0, 0]
    composition = composition.sort_values("cells", ascending=True)
    colors = ["#b23a48" if row.mnc == 45 and row.cell_type == "lte" else "#8a8f98" for row in composition.itertuples()]
    ax.barh(composition["plmn"], composition["cells"], color=colors)
    ax.set_xscale("log")
    for patch, cells, india_pct in zip(ax.patches, composition["cells"], composition["india_pct"], strict=False):
        ax.text(cells * 1.08, patch.get_y() + patch.get_height() / 2, f"{cells:,}; {india_pct:.0f}% IN", va="center", fontsize=8)
    ax.set_title("A. MCC 123 is not random: 123/45 LTE dominates")
    ax.set_xlabel("Cells (log scale)")
    ax.set_ylabel("")

    # B. Country distribution for the dominant PLMN.
    ax = axes[0, 1]
    countries = countries.sort_values("cells", ascending=True)
    ax.barh(countries["country_label"], countries["cells"], color=["#b23a48" if c == "India" else "#8a8f98" for c in countries["country_label"]])
    ax.set_xscale("log")
    for patch, cells in zip(ax.patches, countries["cells"], strict=False):
        ax.text(cells * 1.08, patch.get_y() + patch.get_height() / 2, f"{cells:,}", va="center", fontsize=8)
    ax.set_title("B. 123/45 LTE is overwhelmingly geocoded to India")
    ax.set_xlabel("Cells (log scale)")
    ax.set_ylabel("")

    # C. Military-site map.
    ax = axes[1, 0]
    setup_context_map(
        ax,
        (67.0, 90.0, 6.0, 36.5),
        countries={"IN", "PK", "CN", "NP", "BD", "BT", "LK", "MM"},
        label_points=[
            ("India", 78.5, 22.0),
            ("Pakistan", 70.2, 29.5),
            ("China", 84.0, 33.5),
            ("Bay of Bengal", 87.0, 13.5),
        ],
    )
    india_bases = bases[bases["country_group"] == "India"]
    other_bases = bases[bases["country_group"] != "India"]
    ax.scatter(india_bases["lon"], india_bases["lat"], s=35 + india_bases["cells"] * 25, color="#b23a48", alpha=0.84, edgecolor="white", linewidth=0.5, label="Indian military polygon", zorder=3)
    if not other_bases.empty:
        ax.scatter(other_bases["lon"], other_bases["lat"], s=60, color="#8a8f98", marker="x", label="outlier polygon", zorder=4)
    for name in ["Leh airport", "Srinagar airport", "Karwar / INS Kadamba", "INS Varsha"]:
        rows = bases[bases["base_label"] == name]
        if not rows.empty:
            r = rows.iloc[0]
            ax.text(r["lon"], r["lat"], name, fontsize=7, ha="center", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.8}, zorder=5)
    ax.set_title("C. Military intersections span sensitive Indian sites")
    ax.legend(title="", loc="lower left", frameon=True, fontsize=8)

    # D. Persistence intervals at named bases.
    ax = axes[1, 1]
    timeline = bases[bases["country_group"] == "India"].sort_values(["cells", "span_days"], ascending=False).head(12).sort_values("first_seen")
    y = range(len(timeline))
    ax.hlines(y, timeline["first_seen"], timeline["last_seen"], color="#b23a48", linewidth=2.2)
    ax.scatter(timeline["first_seen"], y, color="#b23a48", s=25, zorder=3)
    ax.scatter(timeline["last_seen"], y, color="#2f6f9f", s=25, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(timeline["base_label"])
    ax.set_title("D. Repeated sightings persist for days to months")
    ax.set_xlabel("Observation window")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(True, axis="x", linewidth=0.35)
    ax.grid(False, axis="y")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS / "obs11_india_military_mcc123.pdf",
        help="PDF output path.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
