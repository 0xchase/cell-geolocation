#!/usr/bin/env python3
"""Generate a four-panel figure for the Gaza network-collapse observation."""

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
OUTPUT_DPI = 700
PREVIEW_DPI = 360

PALETTE = {
    "Palestinian": "#b23a48",
    "Israeli": "#2f6f9f",
    "Gaza": "#b23a48",
    "West Bank": "#4f7f52",
}

CITY_LABELS = {
    "\u062e\u0627\u0646 \u064a\u0648\u0646\u0633": "Khan Yunis",
    "\u063a\u0632\u0629": "Gaza City",
    "\u0631\u0641\u062d": "Rafah",
    "\u062f\u064a\u0631 \u0627\u0644\u0628\u0644\u062d": "Deir al-Balah",
    "\u0646\u0627\u0628\u0644\u0633": "Nablus",
    "\u0627\u0644\u062e\u0644\u064a\u0644": "Hebron",
    "\u0627\u0644\u0628\u064a\u0631\u0629": "Al-Bireh",
    "\u0631\u0627\u0645 \u0627\u0644\u0644\u0647": "Ramallah",
    "\u0628\u064a\u062a \u0644\u062d\u0645": "Bethlehem",
}

CITY_ZONE = {
    "Khan Yunis": "Gaza",
    "Gaza City": "Gaza",
    "Rafah": "Gaza",
    "Deir al-Balah": "Gaza",
    "Nablus": "West Bank",
    "Hebron": "West Bank",
    "Al-Bireh": "West Bank",
    "Ramallah": "West Bank",
    "Bethlehem": "West Bank",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def load_data() -> dict[str, pd.DataFrame | float]:
    global_baseline = ch_df(
        f"""
        SELECT countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count() AS stale_frac
        FROM cell.summary
        WHERE cid > 0
        """
    )["stale_frac"].iloc[0]

    operators = ch_df(
        f"""
        SELECT
            mnc,
            multiIf(
                mnc = 5, 'Jawwal (425/5)',
                mnc = 6, 'Ooredoo (425/6)',
                mnc = 2, 'Cellcom (425/2)',
                mnc = 3, 'Pelephone (425/3)',
                mnc = 1, 'Partner (425/1)',
                mnc = 7, 'Hot Mobile (425/7)',
                concat('MNC ', toString(mnc))
            ) AS operator,
            multiIf(mnc IN (5, 6), 'Palestinian', 'Israeli') AS operator_group,
            count() AS cells,
            countIf(last_seen < toDateTime('{STALE_CUTOFF}')) AS stale_cells,
            stale_cells / cells AS stale_frac
        FROM cell.summary
        WHERE
            mcc = 425
            AND mnc IN (1, 2, 3, 5, 6, 7)
            AND glat BETWEEN 31.2 AND 31.6
            AND glon BETWEEN 34.2 AND 34.6
        GROUP BY mnc, operator, operator_group
        ORDER BY operator_group DESC, stale_frac DESC
        """
    )

    city_literals = ", ".join(sql_string(c) for c in CITY_LABELS)
    cities = ch_df(
        f"""
        SELECT
            city,
            count() AS cells,
            countIf(last_seen < toDateTime('{STALE_CUTOFF}')) AS stale_cells,
            stale_cells / cells AS stale_frac
        FROM cell.summary_full
        WHERE
            mcc = 425
            AND mnc IN (5, 6)
            AND country_osm LIKE '%alestin%'
            AND city IN ({city_literals})
        GROUP BY city
        ORDER BY cells DESC
        """
    )
    cities["city_en"] = cities["city"].map(CITY_LABELS)
    cities["zone"] = cities["city_en"].map(CITY_ZONE)
    city_order = [
        "Khan Yunis",
        "Rafah",
        "Deir al-Balah",
        "Gaza City",
        "Al-Bireh",
        "Ramallah",
        "Nablus",
        "Hebron",
        "Bethlehem",
    ]
    cities["city_en"] = pd.Categorical(cities["city_en"], categories=city_order, ordered=True)
    cities = cities.sort_values("city_en")

    timeline = ch_df(
        f"""
        SELECT
            toDate(toStartOfQuarter(last_seen)) AS quarter,
            multiIf(mnc IN (5, 6), 'Palestinian', 'Israeli') AS operator_group,
            count() AS cells
        FROM cell.summary
        WHERE
            mcc = 425
            AND mnc IN (1, 2, 3, 5, 6, 7)
            AND glat BETWEEN 31.2 AND 31.6
            AND glon BETWEEN 34.2 AND 34.6
        GROUP BY quarter, operator_group
        ORDER BY quarter, operator_group
        """
    )
    timeline["quarter"] = pd.to_datetime(timeline["quarter"])
    timeline["quarter_label"] = timeline["quarter"].map(
        lambda d: f"{d.year} Q{((d.month - 1) // 3) + 1}"
    )

    yearly_spatial = ch_df(
        """
        SELECT
            yr AS year,
            avg_lat AS lat,
            avg_lon AS lon,
            operator_group
        FROM
        (
            SELECT
                toYear(timestamp) AS yr,
                mcc,
                mnc,
                lac,
                cid,
                cell_type,
                multiIf(mnc IN (5, 6), 'Palestinian', 'Israeli') AS operator_group,
                avg(lat) AS avg_lat,
                avg(lon) AS avg_lon
            FROM cell.geos
            WHERE
                mcc = 425
                AND mnc IN (1, 2, 3, 5, 6, 7)
                AND lat BETWEEN 31.2 AND 31.6
                AND lon BETWEEN 34.2 AND 34.6
                AND lat != 0
                AND lon != 0
            GROUP BY yr, mcc, mnc, lac, cid, cell_type, operator_group
        )
        ORDER BY year, operator_group
        """
    )

    return {
        "global_baseline": float(global_baseline),
        "operators": operators,
        "cities": cities,
        "timeline": timeline,
        "yearly_spatial": yearly_spatial,
    }


def annotate_barh(ax: plt.Axes, values: pd.Series, labels: pd.Series) -> None:
    for patch, value, label in zip(ax.patches, values, labels, strict=False):
        ax.text(
            value + 1.3,
            patch.get_y() + patch.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )


def make_figure(data: dict[str, pd.DataFrame | float], output: Path, preview: Path | None) -> None:
    baseline = data["global_baseline"]
    operators = data["operators"].copy()
    cities = data["cities"].copy()
    timeline = data["timeline"].copy()
    yearly_spatial = data["yearly_spatial"].copy()

    assert isinstance(baseline, float)
    assert isinstance(operators, pd.DataFrame)
    assert isinstance(cities, pd.DataFrame)
    assert isinstance(timeline, pd.DataFrame)
    assert isinstance(yearly_spatial, pd.DataFrame)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(15.8, 8.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.08, 1.0])
    fig.suptitle(
        "Gaza: Palestinian cellular networks went stale while Israeli networks persisted",
        fontsize=14,
        fontweight="bold",
    )

    # A. Operator-level staleness in Gaza.
    ax = fig.add_subplot(gs[0, 0])
    operators["stale_pct"] = operators["stale_frac"] * 100
    operators = operators.sort_values("stale_pct", ascending=True)
    sns.barplot(
        data=operators,
        x="stale_pct",
        y="operator",
        hue="operator_group",
        dodge=False,
        palette=PALETTE,
        ax=ax,
    )
    ax.axvline(baseline * 100, color="#444444", linestyle="--", linewidth=1)
    ax.text(baseline * 100 + 1.2, -0.45, "global baseline", fontsize=8, color="#444444")
    annotate_barh(ax, operators["stale_pct"], operators["cells"].map(lambda x: f"n={x:,}"))
    ax.set_title("A. Stale fraction by operator in Gaza bbox")
    ax.set_xlabel(f"Cells last seen before {STALE_CUTOFF} (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 105)
    ax.get_legend().remove()

    # B. City-level control: same Palestinian operators in Gaza vs West Bank.
    ax = fig.add_subplot(gs[0, 1])
    cities["stale_pct"] = cities["stale_frac"] * 100
    sns.barplot(
        data=cities,
        x="stale_pct",
        y="city_en",
        hue="zone",
        dodge=False,
        palette=PALETTE,
        ax=ax,
    )
    ax.axvline(baseline * 100, color="#444444", linestyle="--", linewidth=1)
    annotate_barh(ax, cities["stale_pct"], cities["cells"].map(lambda x: f"n={x:,}"))
    ax.set_title("B. Same Palestinian PLMNs: Gaza vs West Bank")
    ax.set_xlabel(f"Cells last seen before {STALE_CUTOFF} (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 105)
    ax.legend(title="", loc="lower right", frameon=True)

    # C. Timing of last sightings.
    ax = fig.add_subplot(gs[0, 2:])
    sns.barplot(
        data=timeline,
        x="quarter_label",
        y="cells",
        hue="operator_group",
        palette=PALETTE,
        ax=ax,
    )
    ax.set_title("C. Last-active quarter for cells in Gaza bbox")
    ax.set_xlabel("Quarter of last observation")
    ax.set_ylabel("Cells")
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x):,}")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="", loc="upper left", frameon=True)

    # D-G. Spatial distribution by year of observation.
    years = sorted(yearly_spatial["year"].unique())
    for i, year in enumerate(years):
        ax = fig.add_subplot(gs[1, i])
        setup_context_map(
            ax,
            (34.2, 34.6, 31.2, 31.6),
            countries={"PS", "IL", "EG"},
            admin_names={"Gaza Strip"},
            label_points=[
                ("Gaza City", 34.47, 31.51),
                ("Khan Yunis", 34.31, 31.35),
                ("Israel", 34.55, 31.32),
                ("Egypt", 34.23, 31.23),
            ],
        )
        year_points = yearly_spatial[yearly_spatial["year"] == year]
        for group, alpha, size in [("Israeli", 0.72, 6.2), ("Palestinian", 0.58, 4.4)]:
            rows = year_points[year_points["operator_group"] == group]
            if rows.empty:
                continue
            ax.scatter(
                rows["lon"],
                rows["lat"],
                s=size,
                color=PALETTE[group],
                alpha=alpha,
                linewidth=0,
                rasterized=True,
                zorder=3,
            )
        counts = year_points.groupby("operator_group").size()
        p_count = int(counts.get("Palestinian", 0))
        i_count = int(counts.get("Israeli", 0))
        panel = chr(ord("D") + i)
        ax.set_title(f"{panel}. Cells seen in {year}\nPalestinian {p_count:,}; Israeli {i_count:,}", fontsize=9)
        if i > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Palestinian operators", markerfacecolor=PALETTE["Palestinian"], markersize=7, alpha=0.75),
        plt.Line2D([0], [0], marker="o", color="w", label="Israeli operators", markerfacecolor=PALETTE["Israeli"], markersize=7, alpha=0.80),
    ]
    fig.legend(
        handles=handles,
        title="",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncols=2,
        frameon=True,
        fontsize=8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=OUTPUT_DPI, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS / "obs01_gaza_network_collapse.pdf",
        help="PDF output path.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Optional PNG preview path for visual inspection.",
    )
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
