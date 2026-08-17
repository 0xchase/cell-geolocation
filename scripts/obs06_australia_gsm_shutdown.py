#!/usr/bin/env python3
"""Generate a four-panel figure for Australia's GSM/3G shutdown signal."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

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
STALE_CUTOFF = "2025-06-01"
AUS_BBOX = (112.0, 154.0, -45.0, -9.0)

AUSTRALIA_CITY_LABELS = [
    ("Perth", 115.86, -31.95),
    ("Adelaide", 138.60, -34.93),
    ("Melbourne", 144.96, -37.81),
    ("Sydney", 151.21, -33.87),
    ("Brisbane", 153.03, -27.47),
    ("Darwin", 130.84, -12.46),
]

COUNTRIES = {
    208: "France",
    234: "United Kingdom",
    262: "Germany",
    505: "Australia",
}

TECH_LABELS = {
    "gsm": "GSM/UMTS",
    "lte": "LTE",
    "nr": "5G NR",
}

MNC_LABELS = {
    1: "Telstra (505/1)",
    2: "Optus (505/2)",
    3: "Vodafone/TPG (505/3)",
    13: "Rail/Private (505/13)",
    62: "MNC 62",
    16: "MNC 16",
}

TECH_COLORS = {
    "GSM/UMTS": "#b23a48",
    "LTE": "#2f6f9f",
    "5G NR": "#4f7f52",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_data() -> dict[str, pd.DataFrame]:
    by_country_tech = ch_df(
        f"""
        SELECT
            mcc,
            cell_type,
            count() AS cells,
            countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count() AS stale_frac
        FROM cell.summary
        WHERE cid > 0 AND mcc IN (208, 234, 262, 505) AND cell_type IN ('gsm', 'lte', 'nr')
        GROUP BY mcc, cell_type
        ORDER BY mcc, cell_type
        """
    )
    by_country_tech["country"] = by_country_tech["mcc"].map(COUNTRIES)
    by_country_tech["technology"] = by_country_tech["cell_type"].map(TECH_LABELS)
    by_country_tech["stale_pct"] = by_country_tech["stale_frac"] * 100

    timeline = ch_df(
        """
        SELECT
            toDate(toStartOfQuarter(last_seen)) AS quarter,
            count() AS cells
        FROM cell.summary
        WHERE cid > 0 AND mcc = 505 AND cell_type = 'gsm'
        GROUP BY quarter
        ORDER BY quarter
        """
    )
    timeline["quarter"] = pd.to_datetime(timeline["quarter"])
    timeline["quarter_label"] = timeline["quarter"].map(lambda d: f"{d.year} Q{((d.month - 1) // 3) + 1}")

    mncs = ch_df(
        f"""
        SELECT
            mnc,
            count() AS cells,
            countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count() AS stale_frac
        FROM cell.summary
        WHERE cid > 0 AND mcc = 505 AND cell_type = 'gsm'
        GROUP BY mnc
        HAVING cells >= 100
        ORDER BY cells DESC
        """
    )
    mncs["operator"] = mncs["mnc"].map(MNC_LABELS).fillna("MNC " + mncs["mnc"].astype(str))
    mncs["stale_pct"] = mncs["stale_frac"] * 100

    spatial = ch_df(
        f"""
        SELECT
            multiIf(
                timestamp < toDateTime('{STALE_CUTOFF}'),
                'Before shutdown cutoff',
                'After shutdown cutoff'
            ) AS period,
            mcc,
            mnc,
            lac,
            cid,
            cell_type,
            avg(lat) AS avg_lat,
            avg(lon) AS avg_lon
        FROM cell.geos
        WHERE
            cid > 0
            AND mcc = 505
            AND cell_type = 'gsm'
            AND lat BETWEEN {AUS_BBOX[2]} AND {AUS_BBOX[3]}
            AND lon BETWEEN {AUS_BBOX[0]} AND {AUS_BBOX[1]}
            AND NOT (lat = 0 AND lon = 0)
        GROUP BY period, mcc, mnc, lac, cid, cell_type
        ORDER BY period, mnc
        """
    )
    spatial = spatial.rename(columns={"avg_lat": "lat", "avg_lon": "lon"})

    return {"by_country_tech": by_country_tech, "timeline": timeline, "mncs": mncs, "spatial": spatial}


def annotate_barh(ax: plt.Axes, values: pd.Series, labels: pd.Series, pad: float) -> None:
    for patch, value, label in zip(ax.patches, values, labels, strict=False):
        ax.text(
            value + pad,
            patch.get_y() + patch.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )


def draw_australia_seen_map(ax: plt.Axes, points: pd.DataFrame, title: str, color: str, marker_size: float) -> None:
    setup_context_map(ax, AUS_BBOX, countries={"AU", "AUS", "Australia"}, label_points=AUSTRALIA_CITY_LABELS)
    ax.scatter(
        points["lon"],
        points["lat"],
        s=marker_size,
        c=color,
        alpha=0.34 if len(points) > 20000 else 0.58,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    ax.set_title(f"{title}\nn={len(points):,} distinct GSM/UMTS IDs", fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    by_country_tech = data["by_country_tech"].copy()
    timeline = data["timeline"].copy()
    mncs = data["mncs"].copy()
    spatial = data["spatial"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 13.4), constrained_layout=True)
    fig.suptitle(
        "Australia's 3G shutdown leaves a nationwide stale GSM/UMTS layer",
        fontsize=14,
        fontweight="bold",
    )

    # A. Country-tech controls.
    ax = axes[0, 0]
    heat = by_country_tech.pivot(index="country", columns="technology", values="stale_pct").loc[
        ["Australia", "United Kingdom", "France", "Germany"],
        ["GSM/UMTS", "LTE", "5G NR"],
    ]
    sns.heatmap(
        heat,
        annot=True,
        fmt=".1f",
        cmap="rocket_r",
        vmin=0,
        vmax=100,
        cbar_kws={"label": f"Stale before {STALE_CUTOFF} (%)"},
        ax=ax,
    )
    ax.set_title("A. Technology staleness vs control countries")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # B. Australia by technology: GSM is the outlier, not all radio layers.
    ax = axes[0, 1]
    au = by_country_tech[by_country_tech["mcc"] == 505].copy()
    au["technology"] = pd.Categorical(au["technology"], ["GSM/UMTS", "LTE", "5G NR"], ordered=True)
    au = au.sort_values("technology")
    sns.barplot(data=au, x="technology", y="stale_pct", hue="technology", palette=TECH_COLORS, legend=False, ax=ax)
    for patch, stale, cells in zip(ax.patches, au["stale_pct"], au["cells"], strict=False):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            stale + 2.0,
            f"{stale:.1f}%\nn={cells:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("B. Australia: GSM stale, LTE/5G remain healthy")
    ax.set_xlabel("")
    ax.set_ylabel(f"Cells last seen before {STALE_CUTOFF} (%)")
    ax.set_ylim(0, 108)

    # C. Last-active cliff for Australian GSM.
    ax = axes[1, 0]
    ax.bar(timeline["quarter_label"], timeline["cells"], color=TECH_COLORS["GSM/UMTS"])
    ax.set_title("C. Australian GSM cells drop after 2025 Q1")
    ax.set_xlabel("Quarter of last observation")
    ax.set_ylabel("GSM/UMTS cells")
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x):,}")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.axvspan(4.5, 6.5, color="#eeeeee", alpha=0.75, zorder=0)
    ax.text(4.58, timeline["cells"].max() * 0.88, "shutdown cliff", fontsize=8, color="#444444")

    # D. MNC-level check.
    ax = axes[1, 1]
    mncs = mncs.sort_values("stale_pct", ascending=True)
    colors = ["#b23a48" if row.cells >= 1000 else "#8a8f98" for row in mncs.itertuples()]
    ax.barh(mncs["operator"], mncs["stale_pct"], color=colors)
    annotate_barh(ax, mncs["stale_pct"], mncs["cells"].map(lambda x: f"n={x:,}"), 1.0)
    ax.set_title("D. Major Australian GSM MNCs are all stale")
    ax.set_xlabel(f"Cells last seen before {STALE_CUTOFF} (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 108)

    # E/F. Direct observation maps before and after the shutdown cutoff.
    before = spatial[spatial["period"] == "Before shutdown cutoff"].copy()
    after = spatial[spatial["period"] == "After shutdown cutoff"].copy()
    draw_australia_seen_map(
        axes[2, 0],
        before,
        f"E. Australian GSM/UMTS cells seen before {STALE_CUTOFF}",
        "#7f8790",
        1.0,
    )
    draw_australia_seen_map(
        axes[2, 1],
        after,
        f"F. Australian GSM/UMTS cells seen after {STALE_CUTOFF}",
        TECH_COLORS["GSM/UMTS"],
        3.0,
    )

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
        default=PLOTS / "obs06_australia_gsm_shutdown.pdf",
        help="PDF output path.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
