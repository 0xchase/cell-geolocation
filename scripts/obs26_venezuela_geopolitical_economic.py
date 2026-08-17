#!/usr/bin/env python3
"""Analyze Venezuela cell identities for geopolitical and economic signals."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helpers import ADMIN1_GEOJSON, COUNTRIES_GEOJSON, add_osm_basemap, draw_geojson_layer
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"

GROUP_ORDER = ["Venezuela MCC 734", "Colombia MCC 732", "Other foreign MCC"]
GROUP_COLORS = {
    "Venezuela MCC 734": "#2f6f9f",
    "Colombia MCC 732": "#c9743a",
    "Other foreign MCC": "#b23a48",
}

TIMELINE_ORDER = ["Venezuela GSM", "Venezuela LTE", "Colombia MCC 732", "Other foreign MCC"]
TIMELINE_COLORS = {
    "Venezuela GSM": "#2f6f9f",
    "Venezuela LTE": "#4f7f52",
    "Colombia MCC 732": "#c9743a",
    "Other foreign MCC": "#b23a48",
}

MCC_NAMES = {
    338: "Jamaica",
    363: "Aruba",
    374: "Trinidad/Tobago",
    460: "China",
    634: "Sudan",
    732: "Colombia",
    734: "Venezuela",
}

VE_OPERATORS = {
    2: "Digitel",
    4: "Movistar",
    6: "Movilnet",
    9: "DirecTV",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def group_sql(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"""
    multiIf(
        {prefix}mcc = 734, 'Venezuela MCC 734',
        {prefix}mcc = 732, 'Colombia MCC 732',
        'Other foreign MCC'
    )
    """


def timeline_sql(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"""
    multiIf(
        {prefix}mcc = 734 AND {prefix}cell_type = 'gsm', 'Venezuela GSM',
        {prefix}mcc = 734 AND {prefix}cell_type = 'lte', 'Venezuela LTE',
        {prefix}mcc = 734, 'Venezuela other',
        {prefix}mcc = 732, 'Colombia MCC 732',
        'Other foreign MCC'
    )
    """


def operator_label(mcc: int, mnc: int, cell_type: str) -> str:
    if int(mcc) == 734:
        brand = VE_OPERATORS.get(int(mnc), f"MNC {int(mnc)}")
    elif int(mcc) == 732:
        brand = "Colombia border"
    else:
        brand = MCC_NAMES.get(int(mcc), f"MCC {int(mcc)}")
    return f"{brand} {str(cell_type).upper()}"


def load_data() -> dict[str, pd.DataFrame]:
    summary = ch_df(
        """
        SELECT
            count() AS cells,
            sum(obs) AS obs,
            uniqExact(mcc) AS mccs,
            uniqExact((mcc,mnc)) AS operators,
            min(first_seen) AS first_seen,
            max(last_seen) AS last_seen,
            countIf(mcc = 734) AS venezuela_cells,
            countIf(mcc = 732) AS colombia_cells,
            countIf(mcc NOT IN (734,732)) AS other_foreign_cells
        FROM cell.summary_full
        WHERE country_iso = 'VE'
        """
    )

    grid = ch_df(
        f"""
        SELECT
            round(glat, 3) AS lat,
            round(glon, 3) AS lon,
            {group_sql()} AS group_name,
            count() AS cells,
            sum(obs) AS obs
        FROM cell.summary_full
        WHERE country_iso = 'VE'
        GROUP BY lat, lon, group_name
        HAVING cells >= 2 OR group_name != 'Venezuela MCC 734'
        ORDER BY cells DESC
        """
    )

    composition = ch_df(
        f"""
        SELECT
            mcc, mnc, cell_type,
            {group_sql()} AS group_name,
            count() AS cells,
            sum(obs) AS obs,
            min(first_seen) AS first_seen,
            max(last_seen) AS last_seen
        FROM cell.summary_full
        WHERE country_iso = 'VE'
        GROUP BY mcc, mnc, cell_type, group_name
        ORDER BY cells DESC
        """
    )

    monthly = ch_df(
        f"""
        WITH ve AS (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE country_iso = 'VE'
        )
        SELECT
            toStartOfMonth(g.timestamp) AS month,
            {timeline_sql("g")} AS group_name,
            count() AS obs,
            uniqExact((g.mcc,g.mnc,g.lac,g.cid,g.cell_type)) AS cells
        FROM cell.geos AS g
        INNER JOIN ve USING (mcc,mnc,lac,cid,cell_type)
        GROUP BY month, group_name
        ORDER BY month, group_name
        """
    )

    daily_event = ch_df(
        f"""
        WITH ve AS (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE country_iso = 'VE'
        )
        SELECT
            toDate(g.timestamp) AS day,
            {timeline_sql("g")} AS group_name,
            count() AS obs,
            uniqExact((g.mcc,g.mnc,g.lac,g.cid,g.cell_type)) AS cells
        FROM cell.geos AS g
        INNER JOIN ve USING (mcc,mnc,lac,cid,cell_type)
        WHERE g.timestamp BETWEEN '2024-07-01' AND '2024-09-30 23:59:59'
        GROUP BY day, group_name
        ORDER BY day, group_name
        """
    )

    west_border = ch_df(
        f"""
        SELECT
            round(glat, 3) AS lat,
            round(glon, 3) AS lon,
            {group_sql()} AS group_name,
            mcc,
            count() AS cells,
            sum(obs) AS obs
        FROM cell.summary_full
        WHERE glat BETWEEN 5.3 AND 11.8
          AND glon BETWEEN -73.0 AND -66.7
          AND mcc IN (732,734)
          AND (country_iso IN ('VE','CO') OR country_osm IN ('Venezuela','Colombia'))
        GROUP BY lat, lon, group_name, mcc
        HAVING cells >= 1
        ORDER BY cells DESC
        """
    )

    other_foreign = ch_df(
        """
        SELECT
            mcc, mnc, lac, cid, cell_type, obs,
            first_seen, last_seen,
            glat AS lat,
            glon AS lon,
            region,
            city,
            suburb
        FROM cell.summary_full
        WHERE country_iso = 'VE'
          AND mcc NOT IN (734,732)
        ORDER BY mcc, mnc, first_seen
        """
    )

    audit = ch_df(
        """
        SELECT 'Venezuelan MCC in Guyana' AS metric, count() AS cells, sum(obs) AS obs
        FROM cell.summary_full
        WHERE mcc = 734 AND country_iso = 'GY'
        UNION ALL
        SELECT 'Venezuelan MCC near Guyana/Essequibo bbox' AS metric, count() AS cells, sum(obs) AS obs
        FROM cell.summary_full
        WHERE mcc = 734 AND glat BETWEEN 1 AND 9 AND glon BETWEEN -61.5 AND -56
        UNION ALL
        SELECT 'Colombian MCC inside VE join' AS metric, count() AS cells, sum(obs) AS obs
        FROM cell.summary_full
        WHERE country_iso = 'VE' AND mcc = 732
        UNION ALL
        SELECT 'Non-Colombia foreign inside VE' AS metric, count() AS cells, sum(obs) AS obs
        FROM cell.summary_full
        WHERE country_iso = 'VE' AND mcc NOT IN (734,732)
        """
    )

    movement = ch_df(
        """
        WITH
        ve AS (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE country_iso = 'VE'
        ),
        tracks AS (
            SELECT
                g.mcc, g.mnc, g.lac, g.cid, g.cell_type,
                count() AS raw_obs,
                greatCircleDistance(min(g.lon), min(g.lat), max(g.lon), max(g.lat)) / 1000 AS bbox_km
            FROM cell.geos AS g
            INNER JOIN ve USING (mcc,mnc,lac,cid,cell_type)
            WHERE NOT (g.lat = 0 AND g.lon = 0)
            GROUP BY g.mcc, g.mnc, g.lac, g.cid, g.cell_type
        )
        SELECT
            count() AS identities,
            sum(raw_obs) AS raw_obs,
            countIf(bbox_km > 10) AS moved_gt10km,
            countIf(bbox_km > 100) AS moved_gt100km,
            round(quantile(0.50)(bbox_km), 3) AS median_bbox_km,
            round(quantile(0.95)(bbox_km), 3) AS p95_bbox_km,
            round(max(bbox_km), 2) AS max_bbox_km
        FROM tracks
        """
    )

    for df in [grid, composition, west_border]:
        df["group_name"] = pd.Categorical(df["group_name"], GROUP_ORDER, ordered=True)
    monthly["group_name"] = pd.Categorical(monthly["group_name"], TIMELINE_ORDER + ["Venezuela other"], ordered=True)
    daily_event["group_name"] = pd.Categorical(daily_event["group_name"], TIMELINE_ORDER + ["Venezuela other"], ordered=True)
    composition["operator"] = composition.apply(lambda r: operator_label(r["mcc"], r["mnc"], r["cell_type"]), axis=1)
    composition = (
        composition.groupby(["operator", "group_name"], as_index=False, observed=True)
        .agg(cells=("cells", "sum"), obs=("obs", "sum"), first_seen=("first_seen", "min"), last_seen=("last_seen", "max"))
        .sort_values("cells", ascending=False)
    )
    monthly["month"] = pd.to_datetime(monthly["month"])
    daily_event["day"] = pd.to_datetime(daily_event["day"])
    if not other_foreign.empty:
        other_foreign["first_seen"] = pd.to_datetime(other_foreign["first_seen"])
        other_foreign["last_seen"] = pd.to_datetime(other_foreign["last_seen"])
        other_foreign["country"] = other_foreign["mcc"].map(lambda x: MCC_NAMES.get(int(x), f"MCC {int(x)}"))

    return {
        "summary": summary,
        "grid": grid,
        "composition": composition,
        "monthly": monthly,
        "daily_event": daily_event,
        "west_border": west_border,
        "other_foreign": other_foreign,
        "audit": audit,
        "movement": movement,
    }


def setup_map(ax: plt.Axes, bbox: tuple[float, float, float, float], *, zoom: int | None = None, osm_alpha: float = 0.58) -> None:
    ax.set_facecolor("#dceaf2")
    used_osm = False
    if zoom is not None:
        used_osm = add_osm_basemap(ax, bbox, zoom=zoom, alpha=osm_alpha, grayscale=True)
    if not used_osm:
        draw_geojson_layer(
            ax,
            COUNTRIES_GEOJSON,
            bbox,
            facecolor="#f5f1e8",
            edgecolor="#756d63",
            linewidth=0.45,
            zorder=0,
        )
    for countries, face, edge, alpha, lw in [
        ({"VE"}, "#f4d6b6", "#3f3831", 0.50 if used_osm else 0.82, 0.88),
        ({"CO", "BR", "GY", "TT", "AW", "CW"}, "none" if used_osm else "#f5f1e8", "#5f574f", 0.95, 0.55),
    ]:
        draw_geojson_layer(
            ax,
            COUNTRIES_GEOJSON,
            bbox,
            countries=countries,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            alpha=alpha,
            zorder=1.1,
        )
    draw_geojson_layer(
        ax,
        ADMIN1_GEOJSON,
        bbox,
        countries={"VE", "CO", "BR", "GY", "TT"},
        facecolor="none",
        edgecolor="#92887b",
        linewidth=0.25,
        alpha=0.72,
        zorder=1.3,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, color="#ffffff", alpha=0.55)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if used_osm:
        ax.text(0.01, 0.01, "Map data (c) OpenStreetMap contributors", transform=ax.transAxes, fontsize=5.2, color="#555", zorder=20)


def draw_points(ax: plt.Axes, df: pd.DataFrame, *, max_size: float = 90, alpha: float = 0.72, native_alpha: float = 0.38) -> None:
    for group in GROUP_ORDER:
        part = df[df["group_name"] == group]
        if part.empty:
            continue
        size = np.clip(7 + 5.0 * np.sqrt(part["cells"].astype(float)), 8, max_size)
        ax.scatter(
            part["lon"],
            part["lat"],
            s=size,
            color=GROUP_COLORS[group],
            alpha=native_alpha if group == "Venezuela MCC 734" else alpha,
            edgecolor="white" if group != "Venezuela MCC 734" else "none",
            linewidth=0.35,
            zorder=4 if group != "Venezuela MCC 734" else 3,
        )


def add_labels(ax: plt.Axes, labels: list[tuple[str, float, float]], *, fontsize: float = 6.7) -> None:
    for text, lon, lat in labels:
        ax.text(
            lon,
            lat,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#2f2a25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            zorder=8,
        )


def event_window_stats(daily_event: pd.DataFrame) -> tuple[int, int, int]:
    daily = daily_event.groupby("day", observed=True)["obs"].sum()
    election_day = int(daily.get(pd.Timestamp("2024-07-28"), 0))
    pre_median = int(daily[(daily.index >= "2024-07-01") & (daily.index <= "2024-07-27")].median())
    post_median = int(daily[(daily.index >= "2024-07-29") & (daily.index <= "2024-08-11")].median())
    return election_day, pre_median, post_median


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    summary = data["summary"].iloc[0]
    grid = data["grid"].copy()
    composition = data["composition"].copy()
    monthly = data["monthly"].copy()
    daily_event = data["daily_event"].copy()
    west_border = data["west_border"].copy()
    other_foreign = data["other_foreign"].copy()
    audit = data["audit"].copy()
    movement = data["movement"].iloc[0]

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.02)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
        }
    )

    fig = plt.figure(figsize=(15.4, 11.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.18, 1.0], height_ratios=[1.12, 0.95, 1.03])
    fig.suptitle(
        "Venezuela cell identities show stable national coverage, border spillover, and sparse economic-geography leads",
        fontsize=14.0,
        fontweight="bold",
    )

    ax_map = fig.add_subplot(gs[0, 0])
    setup_map(ax_map, (-73.4, -60.5, 4.0, 12.45), zoom=6, osm_alpha=0.50)
    draw_points(ax_map, grid, max_size=80, alpha=0.86, native_alpha=0.32)
    add_labels(
        ax_map,
        [
            ("Caracas", -66.90, 10.50),
            ("Maracaibo\n/ Lake oil belt", -71.45, 10.65),
            ("Colombia border", -72.35, 7.95),
            ("Orinoco / Bolivar", -63.6, 7.3),
            ("Trinidad", -61.3, 10.6),
            ("Guyana / Essequibo\nnegative check", -60.82, 6.2),
        ],
        fontsize=6.6,
    )
    ax_map.set_title(
        f"A. {int(summary['cells']):,} Venezuela-tagged IDs; foreign IDs concentrate at borders/coasts"
    )
    ax_map.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GROUP_COLORS[g], markeredgecolor="white", markersize=6.5, label=g)
            for g in GROUP_ORDER
        ],
        loc="lower right",
        frameon=True,
        fontsize=7.0,
    )

    ax_comp = fig.add_subplot(gs[0, 1])
    top_comp = composition.sort_values("cells", ascending=False).head(15).copy()
    top_comp = top_comp.sort_values("cells", ascending=True)
    ax_comp.barh(top_comp["operator"], top_comp["cells"], color=[GROUP_COLORS[str(g)] for g in top_comp["group_name"]])
    ax_comp.set_xscale("log")
    ax_comp.set_xlabel("Distinct cell identities, log scale")
    ax_comp.set_ylabel("")
    ax_comp.set_title(
        f"B. Native operators dominate: {int(summary['venezuela_cells']):,} native vs {int(summary['colombia_cells']):,} Colombian-border IDs"
    )
    ax_comp.set_xlim(0.75, top_comp["cells"].max() * 4.4)
    for patch, cells, obs in zip(ax_comp.patches, top_comp["cells"], top_comp["obs"], strict=False):
        ax_comp.text(cells * 1.12, patch.get_y() + patch.get_height() / 2, f"{int(cells):,} IDs; {int(obs):,} obs", va="center", fontsize=6.7)
    ax_comp.legend(handles=[Patch(facecolor=GROUP_COLORS[g], label=g) for g in GROUP_ORDER], loc="lower right", frameon=True, fontsize=6.8)

    ax_time = fig.add_subplot(gs[1, 0])
    for group in TIMELINE_ORDER:
        part = monthly[monthly["group_name"] == group].sort_values("month")
        if part.empty:
            continue
        ax_time.plot(part["month"], part["obs"], color=TIMELINE_COLORS[group], marker="o", markersize=3.2, linewidth=1.55, label=group)
    ax_time.set_yscale("log")
    ax_time.set_ylim(0.8, max(90000, monthly["obs"].max() * 1.35))
    ax_time.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_time.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_time.tick_params(axis="x", rotation=35)
    ax_time.set_xlabel("")
    ax_time.set_ylabel("Raw observations, log scale")
    ax_time.set_title("C. Event audit: election/sanctions dates are not visible as clean network shocks")
    event_marks = [
        (pd.Timestamp("2024-04-17"), "OFAC oil\nwind-down"),
        (pd.Timestamp("2024-07-28"), "Election"),
        (pd.Timestamp("2025-09-01"), "Lagunillas\noil context"),
        (pd.Timestamp("2026-01-15"), "Collection\nburst"),
    ]
    for x, label in event_marks:
        ax_time.axvline(x, color="#4b4741", linestyle="--", linewidth=0.75, alpha=0.82)
        ax_time.text(x, ax_time.get_ylim()[1] / 1.45, label, rotation=90, ha="right", va="top", fontsize=6.3, color="#3d3934")
    election_day, pre_med, post_med = event_window_stats(daily_event)
    ax_time.text(
        0.02,
        0.05,
        f"July 28 election day: {election_day:,} obs; July pre-median {pre_med:,}; two-week post-median {post_med:,}.",
        transform=ax_time.transAxes,
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "#d0cbc4", "alpha": 0.84, "pad": 2.0},
    )
    ax_time.legend(loc="upper left", frameon=True, fontsize=6.8)

    ax_border = fig.add_subplot(gs[1, 1])
    setup_map(ax_border, (-73.0, -66.6, 5.45, 11.65), zoom=7, osm_alpha=0.56)
    draw_points(ax_border, west_border, max_size=72, alpha=0.88, native_alpha=0.22)
    add_labels(
        ax_border,
        [
            ("Cucuta /\nSan Antonio", -72.47, 7.86),
            ("Arauca /\nEl Amparo", -70.76, 7.10),
            ("Puerto Carreno", -67.49, 6.20),
            ("La Guajira", -72.05, 11.25),
        ],
        fontsize=6.3,
    )
    ax_border.set_title("D. Colombia-Venezuela border spillover is strong but geographically ordinary")

    ax_foreign = fig.add_subplot(gs[2, 0])
    ax_foreign.axis("off")
    ax_foreign.set_title("E. Non-Colombia foreign MCCs are seven sparse leads, not a broad phenomenon")
    insets = [
        ("Caracas: Aruba MCC 363", (-66.94, -66.84, 10.38, 10.47), other_foreign[other_foreign["mcc"] == 363], "upper left"),
        ("Zulia oil belt: China 460 + Sudan 634", (-71.82, -71.08, 9.94, 10.78), other_foreign[other_foreign["mcc"].isin([460, 634])], "upper right"),
        ("Sucre coast: Jamaica 338 + Trinidad 374", (-63.12, -62.88, 10.62, 10.79), other_foreign[other_foreign["mcc"].isin([338, 374])], "lower left"),
    ]
    positions = {
        "upper left": [0.02, 0.19, 0.30, 0.70],
        "upper right": [0.35, 0.19, 0.30, 0.70],
        "lower left": [0.68, 0.19, 0.30, 0.70],
    }
    foreign_palette = {338: "#8f4c60", 363: "#6f5b7b", 374: "#c9743a", 460: "#2f6f9f", 634: "#b23a48"}
    for title, bbox, subset, pos_key in insets:
        iax = ax_foreign.inset_axes(positions[pos_key])
        setup_map(iax, bbox, zoom=12 if "Caracas" in title or "Sucre" in title else 10, osm_alpha=0.68)
        for _, row in subset.iterrows():
            iax.scatter(row["lon"], row["lat"], s=48, color=foreign_palette.get(int(row["mcc"]), "#333"), edgecolor="white", linewidth=0.55, zorder=5)
            dx = 0.012 if "Zulia" not in title else 0.018
            iax.text(row["lon"] + dx, row["lat"], f"{MCC_NAMES.get(int(row['mcc']), int(row['mcc']))}", fontsize=5.8, va="center", zorder=6)
        iax.set_title(title, fontsize=7.2)
        iax.set_xlabel("")
        iax.set_ylabel("")
        iax.tick_params(labelsize=5.8)
    ax_foreign.text(
        0.02,
        0.03,
        "Interpretation: China MCC 460 lies at Lagunillas/Fabricio Ojeda in the Lake Maracaibo oil belt, but it is only 2 LTE IDs / 4 obs.\n"
        "The coastal Caribbean and Caracas one-offs are better treated as leads or contamination until independently verified.",
        transform=ax_foreign.transAxes,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#d0cbc4", "alpha": 0.86, "pad": 2.2},
    )

    ax_audit = fig.add_subplot(gs[2, 1])
    ax_audit.axis("off")
    ax_audit.set_title("F. Skeptical checks: what the data supports")
    audit_map = {str(r["metric"]): (int(r["cells"]), int(r["obs"])) for _, r in audit.iterrows()}
    lines = [
        ("Native network", f"{int(summary['venezuela_cells']):,} MCC 734 IDs across ordinary population/oil/coastal centers."),
        ("Colombia border", f"{audit_map.get('Colombian MCC inside VE join', (0, 0))[0]:,} MCC 732 IDs; concentrated at known border towns."),
        ("Essequibo/Guyana", f"{audit_map.get('Venezuelan MCC in Guyana', (0, 0))[0]:,} MCC 734 IDs in country_iso=GY; no dataset support for a Guyana deployment."),
        ("Foreign one-offs", f"{audit_map.get('Non-Colombia foreign inside VE', (0, 0))[0]:,} non-Colombia foreign IDs; mostly coastal/Caracas/Zulia leads."),
        ("Movement sanity", f"{int(movement['moved_gt100km']):,} IDs move >100 km; p95 span {float(movement['p95_bbox_km']):.2f} km."),
        ("Election window", "July 28, 2024 is not a clean cellular anomaly in this crowdsourced dataset."),
    ]
    y = 0.92
    for label, text in lines:
        ax_audit.text(0.02, y, label, fontsize=8.0, fontweight="bold", transform=ax_audit.transAxes, va="top")
        ax_audit.text(0.34, y, text, fontsize=7.6, transform=ax_audit.transAxes, va="top")
        y -= 0.135
    ax_audit.text(
        0.02,
        0.03,
        "Event labels are context for hypothesis generation, not causal claims. The strongest signal is stable geography, not disruption.",
        transform=ax_audit.transAxes,
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#d0cbc4", "alpha": 0.86, "pad": 2.2},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs26_venezuela_geopolitical_economic.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    data = load_data()
    make_figure(data, args.output, args.preview)
    summary = data["summary"].iloc[0]
    movement = data["movement"].iloc[0]
    print(
        f"{args.output}\n"
        f"VE-tagged identities: {int(summary['cells']):,}; observations: {int(summary['obs']):,}; "
        f"Colombia MCC 732: {int(summary['colombia_cells']):,}; "
        f"other foreign: {int(summary['other_foreign_cells']):,}; "
        f"moved >100 km: {int(movement['moved_gt100km']):,}"
    )


if __name__ == "__main__":
    main()
