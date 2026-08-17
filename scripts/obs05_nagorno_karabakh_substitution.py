#!/usr/bin/env python3
"""Generate a cautious Nagorno-Karabakh operator-asymmetry figure."""

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
STALE_CUTOFF = "2025-06-01"

PALETTE = {
    "Armenia MCC 283": "#b23a48",
    "Azerbaijan MCC 400": "#2f6f9f",
}

TECH_PALETTE = {
    "Armenia LTE": "#b23a48",
    "Armenia GSM": "#d7838d",
    "Azerbaijan LTE": "#2f6f9f",
    "Azerbaijan GSM": "#79a8c7",
}

SCOPE_ORDER = [
    "AZ/OSM-Azerbaijan core",
    "Armenia-side Syunik",
    "AZ rows, OSM Armenia",
    "Other/ambiguous",
]


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def scope_sql() -> str:
    return """
    multiIf(
        country_iso = 'AZ' AND country_osm = 'Azərbaycan', 'AZ/OSM-Azerbaijan core',
        country_iso = 'AM', 'Armenia-side Syunik',
        country_iso = 'AZ' AND country_osm = 'Հայաստան', 'AZ rows, OSM Armenia',
        'Other/ambiguous'
    )
    """


def op_sql(prefix: str = "") -> str:
    p = f"{prefix}." if prefix else ""
    return f"multiIf({p}mcc = 283, 'Armenia MCC 283', 'Azerbaijan MCC 400')"


def tech_sql(prefix: str = "") -> str:
    p = f"{prefix}." if prefix else ""
    return f"""
    multiIf(
        {p}mcc = 283 AND {p}cell_type = 'lte', 'Armenia LTE',
        {p}mcc = 283 AND {p}cell_type = 'gsm', 'Armenia GSM',
        {p}mcc = 400 AND {p}cell_type = 'lte', 'Azerbaijan LTE',
        'Azerbaijan GSM'
    )
    """


def load_data() -> dict[str, pd.DataFrame | float]:
    baseline = float(
        ch_df(
            f"""
            SELECT countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count() AS stale_frac
            FROM cell.summary
            WHERE cid > 0
            """
        )["stale_frac"].iloc[0]
    )

    scope_split = ch_df(
        f"""
        SELECT
            {scope_sql()} AS scope,
            {op_sql()} AS operator,
            count() AS cells,
            sum(obs) AS total_obs,
            round(countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count(), 3) AS stale_frac
        FROM cell.summary_full
        WHERE
            cid > 0
            AND glat BETWEEN 39.5 AND 40.2
            AND glon BETWEEN 46.3 AND 47.0
            AND mcc IN (283, 400)
        GROUP BY scope, operator
        ORDER BY scope, operator
        """
    )
    scope_split["scope"] = pd.Categorical(scope_split["scope"], SCOPE_ORDER, ordered=True)

    core_tech = ch_df(
        f"""
        SELECT
            {op_sql()} AS operator,
            {tech_sql()} AS tech_group,
            concat(toString(mcc), '/', toString(mnc), ' ', toString(cell_type)) AS plmn,
            cell_type,
            mnc,
            count() AS cells,
            sum(obs) AS total_obs,
            round(countIf(obs = 1) / count(), 3) AS one_obs_frac,
            round(countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count(), 3) AS stale_frac,
            min(first_seen) AS first_seen_min,
            max(last_seen) AS last_seen_max
        FROM cell.summary_full
        WHERE
            cid > 0
            AND glat BETWEEN 39.5 AND 40.2
            AND glon BETWEEN 46.3 AND 47.0
            AND country_iso = 'AZ'
            AND country_osm = 'Azərbaycan'
            AND mcc IN (283, 400)
        GROUP BY operator, tech_group, plmn, cell_type, mnc
        ORDER BY cells DESC
        """
    )

    core_timeline = ch_df(
        f"""
        WITH core AS (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE
                cid > 0
                AND glat BETWEEN 39.5 AND 40.2
                AND glon BETWEEN 46.3 AND 47.0
                AND country_iso = 'AZ'
                AND country_osm = 'Azərbaycan'
                AND mcc IN (283, 400)
        )
        SELECT
            toDate(toStartOfQuarter(g.timestamp)) AS quarter,
            {tech_sql("g")} AS tech_group,
            count() AS raw_obs,
            uniqExact((g.mcc,g.mnc,g.lac,g.cid,g.cell_type)) AS cells
        FROM cell.geos AS g
        INNER JOIN core USING (mcc,mnc,lac,cid,cell_type)
        WHERE NOT (g.lat = 0 AND g.lon = 0)
        GROUP BY quarter, tech_group
        ORDER BY quarter, tech_group
        """
    )
    core_timeline["quarter"] = pd.to_datetime(core_timeline["quarter"])

    spatial = ch_df(
        f"""
        SELECT
            glat AS lat,
            glon AS lon,
            {scope_sql()} AS scope,
            {op_sql()} AS operator,
            cell_type,
            mnc,
            last_seen < toDateTime('{STALE_CUTOFF}') AS stale
        FROM cell.summary_full
        WHERE
            cid > 0
            AND glat BETWEEN 39.5 AND 40.2
            AND glon BETWEEN 46.3 AND 47.0
            AND mcc IN (283, 400)
            AND glat != 0
            AND glon != 0
        """
    )
    spatial["scope"] = pd.Categorical(spatial["scope"], SCOPE_ORDER, ordered=True)

    national_controls = ch_df(
        f"""
        SELECT
            multiIf(country_iso = 'AM', 'Armenia national', 'Azerbaijan national') AS scope,
            {op_sql()} AS operator,
            cell_type,
            mnc,
            count() AS cells,
            round(countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count(), 3) AS stale_frac
        FROM cell.summary_full
        WHERE
            cid > 0
            AND (
                (country_iso = 'AM' AND mcc = 283)
                OR (country_iso = 'AZ' AND mcc = 400)
            )
        GROUP BY scope, operator, cell_type, mnc
        ORDER BY scope, operator, cell_type, mnc
        """
    )

    return {
        "baseline": baseline,
        "scope_split": scope_split,
        "core_tech": core_tech,
        "core_timeline": core_timeline,
        "spatial": spatial,
        "national_controls": national_controls,
    }


def make_figure(data: dict[str, pd.DataFrame | float], output: Path, preview: Path | None) -> None:
    baseline = float(data["baseline"])
    scope_split = data["scope_split"].copy()
    core_tech = data["core_tech"].copy()
    core_timeline = data["core_timeline"].copy()
    spatial = data["spatial"].copy()
    national_controls = data["national_controls"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.03)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), constrained_layout=False)
    fig.suptitle(
        "Nagorno-Karabakh: post-2023 operator asymmetry, with border and crawl confounds separated",
        fontsize=13.6,
        fontweight="bold",
        y=0.982,
    )

    # A. Show the bbox confound explicitly.
    ax = axes[0, 0]
    pivot = (
        scope_split.pivot_table(index="scope", columns="operator", values="cells", fill_value=0, aggfunc="sum", observed=False)
        .reindex(SCOPE_ORDER)
        .fillna(0)
    )
    y = range(len(pivot))
    left = pd.Series([0] * len(pivot), index=pivot.index, dtype=float)
    for operator in ["Armenia MCC 283", "Azerbaijan MCC 400"]:
        vals = pivot.get(operator, pd.Series([0] * len(pivot), index=pivot.index)).astype(float)
        ax.barh(list(y), vals, left=left, color=PALETTE[operator], label=operator)
        left += vals
    ax.set_yticks(list(y))
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_title("A. Original geographic box mixes core rows with Armenia-side Syunik")
    ax.set_xlabel("Distinct cell identities")
    ax.set_ylabel("")
    ax.legend(loc="lower right", frameon=True, fontsize=7.5)
    for i, scope in enumerate(pivot.index):
        rows = scope_split[scope_split["scope"] == scope]
        labels = []
        for _, row in rows.iterrows():
            labels.append(f"{row['operator'].split()[0]} {int(row['cells']):,}; stale {row['stale_frac'] * 100:.0f}%")
        if labels:
            ax.text(left.iloc[i] + max(left.max() * 0.015, 10), i, " | ".join(labels), va="center", fontsize=6.8)
    ax.set_xlim(0, max(left.max() * 1.42, 1))

    # B. Core area only, split by technology/MNC.
    ax = axes[0, 1]
    tech = core_tech.copy()
    tech["label"] = tech.apply(lambda r: f"{r['tech_group']}\n{r['plmn']}", axis=1)
    tech = tech.sort_values("cells", ascending=True)
    colors = [PALETTE[o] for o in tech["operator"]]
    ax.barh(tech["label"], tech["cells"], color=colors)
    ax.set_title("B. Core AZ/OSM-Azerbaijan rows: stale signal is mostly Armenian LTE")
    ax.set_xlabel("Distinct cell identities")
    ax.set_ylabel("")
    ax.set_xlim(0, tech["cells"].max() * 1.72)
    for patch, cells, stale, one_obs in zip(ax.patches, tech["cells"], tech["stale_frac"], tech["one_obs_frac"], strict=False):
        ax.text(
            cells + tech["cells"].max() * 0.025,
            patch.get_y() + patch.get_height() / 2,
            f"{int(cells):,}; stale {stale * 100:.0f}%; one-obs {one_obs * 100:.0f}%",
            va="center",
            fontsize=6.8,
        )
    ax.axvline(0, color="#333", linewidth=0.6)

    # C. Raw observation timeline, not last_seen histogram.
    ax = axes[1, 0]
    for tech_group in ["Armenia LTE", "Armenia GSM", "Azerbaijan LTE", "Azerbaijan GSM"]:
        part = core_timeline[core_timeline["tech_group"] == tech_group].sort_values("quarter")
        if part.empty:
            continue
        ax.plot(
            part["quarter"],
            part["raw_obs"],
            marker="o",
            markersize=3.7,
            linewidth=1.55,
            color=TECH_PALETTE[tech_group],
            label=tech_group,
        )
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(5000, core_timeline["raw_obs"].max() * 1.35))
    quarters = pd.to_datetime(sorted(core_timeline["quarter"].unique()))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_xticks(quarters)
    ax.set_xticklabels([f"{d.year} Q{((d.month - 1) // 3) + 1}" for d in quarters], rotation=35, ha="right")
    ax.set_title("C. Raw observations: Armenian LTE is early and stops; Azerbaijani IDs continue")
    ax.set_xlabel("Observation quarter")
    ax.set_ylabel("Raw observations in core, log scale")
    ax.legend(loc="upper left", frameon=True, fontsize=7)
    ax.text(
        0.02,
        0.04,
        "Caveat: 2026 Azerbaijani surge is partly a collection/crawl burst, not direct evidence of construction timing.",
        transform=ax.transAxes,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#d0cbc4", "alpha": 0.86, "pad": 2.0},
    )

    # D. Spatial map.
    ax = axes[1, 1]
    setup_context_map(
        ax,
        (46.3, 47.0, 39.5, 40.2),
        countries={"AM", "AZ"},
        admin_names={"Kəlbəcər", "Tərtər", "Xocalı", "Xocavənd", "Şuşa", "Laçın", "Syunik"},
        label_points=[
            ("Stepanakert/\nKhankendi", 46.75, 39.82),
            ("Syunik\n(confound)", 46.39, 39.56),
            ("Azerbaijan", 46.91, 40.08),
        ],
    )
    armenia_side = spatial[(spatial["scope"] == "Armenia-side Syunik") & (spatial["operator"] == "Armenia MCC 283")]
    core_azeri = spatial[(spatial["scope"] == "AZ/OSM-Azerbaijan core") & (spatial["operator"] == "Azerbaijan MCC 400")]
    core_arm_stale = spatial[
        (spatial["scope"] == "AZ/OSM-Azerbaijan core")
        & (spatial["operator"] == "Armenia MCC 283")
        & (spatial["stale"].astype(int) == 1)
    ]
    core_arm_fresh = spatial[
        (spatial["scope"] == "AZ/OSM-Azerbaijan core")
        & (spatial["operator"] == "Armenia MCC 283")
        & (spatial["stale"].astype(int) == 0)
    ]
    ax.scatter(armenia_side["lon"], armenia_side["lat"], s=11, alpha=0.20, linewidth=0, color=PALETTE["Armenia MCC 283"], rasterized=True, label="Armenian IDs on Armenia side")
    ax.scatter(core_azeri["lon"], core_azeri["lat"], s=10, alpha=0.32, linewidth=0, color=PALETTE["Azerbaijan MCC 400"], rasterized=True, label="Azerbaijani IDs in core")
    ax.scatter(core_arm_fresh["lon"], core_arm_fresh["lat"], s=18, alpha=0.72, linewidth=0, color=PALETTE["Armenia MCC 283"], rasterized=True, label="Armenian IDs in core, fresh")
    ax.scatter(core_arm_stale["lon"], core_arm_stale["lat"], s=22, alpha=0.78, linewidth=0.8, marker="x", color=PALETTE["Armenia MCC 283"], rasterized=True, label="Armenian IDs in core, stale")
    ax.set_title("D. Map: plausible signal is the stale Armenian cluster inside the core")
    ax.legend(loc="upper left", frameon=True, fontsize=6.5)

    def weighted_stale(frame: pd.DataFrame) -> float:
        if frame.empty or frame["cells"].sum() == 0:
            return float("nan")
        return float((frame["cells"] * frame["stale_frac"]).sum() / frame["cells"].sum())

    arm_lte_nat = weighted_stale(national_controls[
        (national_controls["scope"] == "Armenia national")
        & (national_controls["operator"] == "Armenia MCC 283")
        & (national_controls["cell_type"] == "lte")
    ])
    core_arm_lte = weighted_stale(core_tech[(core_tech["operator"] == "Armenia MCC 283") & (core_tech["cell_type"] == "lte")])
    fig.text(
        0.5,
        0.012,
        f"Interpretation: conflict/exodus is plausible only after filtering to AZ/OSM-Azerbaijan core rows. "
        f"Armenian LTE stale fraction in that core is {core_arm_lte * 100:.0f}% vs {arm_lte_nat * 100:.1f}% for Armenian LTE nationally; "
        "but the dataset begins after the Sep-2023 offensive and cannot prove pre/post substitution.",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.95), w_pad=3.0, h_pad=2.8)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS / "obs05_nagorno_karabakh_substitution.pdf",
        help="PDF output path.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
