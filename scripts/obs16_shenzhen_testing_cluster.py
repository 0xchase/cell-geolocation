#!/usr/bin/env python3
"""Generate a four-panel figure for the Shenzhen multi-country testing cluster."""

from __future__ import annotations

import argparse
import ast
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

MCC_NAME = {
    86: "China test/legacy",
    441: "Japan",
    250: "Russia",
    262: "Germany",
    412: "Afghanistan",
    424: "UAE",
    452: "Vietnam",
    454: "Hong Kong",
    460: "China",
    462: "Iran",
    466: "Taiwan",
    515: "Philippines",
    520: "Thailand",
    540: "Solomon Is.",
    734: "Venezuela",
}

GROUP_COLORS = {
    "China": "#b23a48",
    "Hong Kong": "#2f6f9f",
    "Other foreign/test": "#c9743a",
}

CONTEXT_AREAS = [
    ("Bao'an RF labs", 113.78, 113.91, 22.54, 22.72, "#c9743a"),
    ("ZTE/Xili R&D", 113.91, 114.01, 22.54, 22.62, "#4f7f52"),
    ("Longhua/Dalang labs", 113.90, 114.06, 22.61, 22.74, "#8f4c60"),
]

CONTEXT_NOTES = [
    "Online search found many recognized wireless-device testing labs in Shenzhen, especially Bao'an and Longhua/Dalang.",
    "ISED's laboratory list includes Shenzhen labs authorized for RSS-130/132/133/139/210/216/247/248 testing in Bao'an, Fuhai, Xixiang, Shajing, Longhua, and Dalang.",
    "ZTE documents a China Unicom 5G field test in Xili, Shenzhen; public profiles place ZTE R&D/industrial facilities on Liuxian Road in Xili/Nanshan.",
    "Data check: the bbox has 98k identities, but only 26 non-China/Hong Kong identities. This is sparse test leakage, not a large foreign deployment.",
]


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def mcc_group(mcc: int) -> str:
    if mcc == 460:
        return "China"
    if mcc == 454:
        return "Hong Kong"
    return "Other foreign/test"


def mcc_label(mcc: int) -> str:
    return MCC_NAME.get(int(mcc), f"MCC {int(mcc)}")


def parse_mcc_list(value) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return [int(v) for v in parsed]


def load_data() -> dict[str, pd.DataFrame]:
    condition = "cid > 0 AND lat BETWEEN 22.4 AND 22.8 AND lon BETWEEN 113.8 AND 114.2"
    summary = ch_df(
        f"""
        SELECT
               uniqExact(mcc) AS countries,
               uniqExactIf(mcc, mcc NOT IN (454, 460)) AS other_countries,
               count() AS obs,
               countIf(mcc NOT IN (454, 460)) AS other_obs,
               uniqExact((mcc,mnc,lac,cid,cell_type)) AS cells,
               uniqExactIf((mcc,mnc,lac,cid,cell_type), mcc IN (454, 460)) AS cn_hk_cells,
               uniqExactIf((mcc,mnc,lac,cid,cell_type), mcc NOT IN (454, 460)) AS other_cells,
               uniqExact((round(lat,2), round(lon,2))) AS grid_points
        FROM cell.geos
        WHERE {condition}
        """
    )
    mccs = ch_df(
        f"""
        SELECT mcc, count() AS obs, uniqExact((mcc,mnc,lac,cid)) AS cells, uniqExact(mnc) AS mncs
        FROM cell.geos
        WHERE {condition}
        GROUP BY mcc
        ORDER BY cells DESC
        """
    )
    grid = ch_df(
        f"""
        SELECT round(lat, 2) AS lat, round(lon, 2) AS lon, count() AS obs,
               uniqExact((mcc,mnc,lac,cid)) AS cells,
               uniqExact(mcc) AS mccs,
               countIf(mcc NOT IN (454,460)) AS foreign_obs
        FROM cell.geos
        WHERE {condition}
        GROUP BY lat, lon
        HAVING cells >= 20
        ORDER BY cells DESC
        LIMIT 220
        """
    )
    multicountry = ch_df(
        f"""
        SELECT round(lat, 2) AS lat, round(lon, 2) AS lon, count() AS obs,
               uniqExact((mcc,mnc,lac,cid)) AS cells,
               uniqExact(mcc) AS mccs,
               arraySort(groupUniqArray(20)(mcc)) AS mcc_list
        FROM cell.geos
        WHERE {condition}
        GROUP BY lat, lon
        HAVING mccs >= 3
        ORDER BY cells DESC
        LIMIT 12
        """
    )
    foreign_bins = ch_df(
        f"""
        SELECT round(lat, 2) AS lat, round(lon, 2) AS lon,
               count() AS obs,
               uniqExact((mcc,mnc,lac,cid,cell_type)) AS cells,
               uniqExact(mcc) AS mccs,
               arraySort(groupUniqArray(20)(mcc)) AS mcc_list
        FROM cell.geos
        WHERE {condition} AND mcc NOT IN (454, 460)
        GROUP BY lat, lon
        ORDER BY cells DESC, obs DESC
        LIMIT 20
        """
    )
    monthly = ch_df(
        f"""
        SELECT toStartOfMonth(timestamp) AS month,
               multiIf(mcc = 460, 'China',
                       mcc = 454, 'Hong Kong',
                       'Other foreign/test') AS group_name,
               count() AS obs,
               uniqExact((mcc,mnc,lac,cid,cell_type)) AS cells
        FROM cell.geos
        WHERE {condition}
        GROUP BY month, group_name
        ORDER BY month, group_name
        """
    )
    mccs["country"] = mccs["mcc"].map(mcc_label)
    mccs["group_name"] = mccs["mcc"].map(lambda x: mcc_group(int(x)))
    multicountry["mcc_list"] = multicountry["mcc_list"].map(parse_mcc_list)
    foreign_bins["mcc_list"] = foreign_bins["mcc_list"].map(parse_mcc_list)
    monthly["month"] = pd.to_datetime(monthly["month"])
    return {
        "summary": summary,
        "mccs": mccs,
        "grid": grid,
        "multicountry": multicountry,
        "foreign_bins": foreign_bins,
        "monthly": monthly,
    }


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    summary = data["summary"].iloc[0]
    mccs = data["mccs"].copy()
    grid = data["grid"].copy()
    multicountry = data["multicountry"].copy()
    foreign_bins = data["foreign_bins"].copy()
    monthly = data["monthly"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.04)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.2), constrained_layout=True)
    fig.suptitle("Shenzhen RF-test districts explain sparse foreign-MCC leakage", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    plot_mccs = mccs.sort_values("cells", ascending=True)
    colors = [GROUP_COLORS[g] for g in plot_mccs["group_name"]]
    ax.barh(plot_mccs["country"], plot_mccs["cells"], color=colors)
    ax.set_xscale("log")
    ax.set_title(
        f"A. {int(summary['cells']):,} IDs, but only {int(summary['other_cells'])} non-China/HK"
    )
    ax.set_xlabel("Distinct cell identities in Shenzhen bbox (log)")
    ax.set_ylabel("")
    for patch, cells, obs in zip(ax.patches, plot_mccs["cells"], plot_mccs["obs"], strict=False):
        ax.text(max(cells * 1.15, 1.2), patch.get_y() + patch.get_height() / 2, f"{cells:,} IDs; {obs:,} obs", va="center", fontsize=6.8)

    ax = axes[0, 1]
    bbox = (113.78, 114.22, 22.38, 22.72)
    add_osm_basemap(ax, bbox, zoom=11, alpha=0.80, grayscale=True)
    for label, xmin, xmax, ymin, ymax, color in CONTEXT_AREAS:
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                alpha=0.12,
                zorder=2,
            )
        )
        ax.text(
            (xmin + xmax) / 2,
            (ymin + ymax) / 2,
            label,
            ha="center",
            va="center",
            fontsize=6.0,
            color="#2b2b2b",
            bbox={"facecolor": "white", "edgecolor": color, "linewidth": 0.35, "alpha": 0.78, "pad": 0.8},
            zorder=6,
        )
    sc = ax.scatter(
        grid["lon"],
        grid["lat"],
        s=18 + grid["cells"].clip(upper=800) / 8,
        color="#8a8f98",
        alpha=0.36,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.scatter(
        foreign_bins["lon"],
        foreign_bins["lat"],
        s=42 + foreign_bins["cells"].clip(upper=12) * 12,
        color="#c9743a",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.55,
        zorder=5,
        label="Non-China/HK MCC bin",
    )
    for i, row in foreign_bins.head(2).iterrows():
        labels = ", ".join(mcc_label(m) for m in row["mcc_list"][:3])
        dx = 0.018 if i == foreign_bins.index[0] else -0.018
        dy = 0.018 if i == foreign_bins.index[0] else -0.016
        ax.text(
            row["lon"] + dx,
            row["lat"] + dy,
            f"{int(row['cells'])} IDs\n{labels}",
            ha="center",
            fontsize=6.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
            zorder=6,
        )
    ax.set_title("B. Sparse foreign/test pockets near RF-test districts")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(title="", loc="lower left", frameon=True, fontsize=7)
    ax.text(0.02, 0.02, "Map data (c) OpenStreetMap contributors", transform=ax.transAxes, fontsize=5.8, color="#555", zorder=7)

    ax = axes[1, 0]
    mc = foreign_bins.head(8).copy()
    mc["label"] = mc.apply(lambda r: f"{r['lat']:.2f}, {r['lon']:.2f}\n{[int(m) for m in r['mcc_list']]}", axis=1)
    mc = mc.sort_values("cells", ascending=True)
    ax.barh(mc["label"], mc["cells"], color="#8f4c60")
    for patch, mcc_count, obs in zip(ax.patches, mc["mccs"], mc["obs"], strict=False):
        ax.text(patch.get_width() + 0.4, patch.get_y() + patch.get_height() / 2, f"{int(mcc_count)} MCCs; {int(obs)} obs", va="center", fontsize=7)
    ax.set_title("D. Non-China/HK evidence is small and localized")
    ax.set_xlabel("Foreign/test identities per 0.01-degree bin")
    ax.set_ylabel("")
    ax.set_xlim(0, max(15, mc["cells"].max() * 1.45))

    ax = axes[0, 2]
    ax.axis("off")
    ax.set_title("C. Online context: Shenzhen is an RF/wireless test hub")
    y = 0.96
    for note in CONTEXT_NOTES:
        ax.text(
            0.02,
            y,
            "- " + note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            wrap=True,
        )
        y -= 0.215
    ax.text(
        0.02,
        0.03,
        "Sources checked: ISED recognized wireless-device test labs; ZTE/China Unicom Xili 5G field-test article; ZTE public profiles.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#5c5752",
        wrap=True,
    )

    ax = axes[1, 1]
    type_counts = (
        mccs.assign(group=mccs["group_name"])
        .groupby(["group", "mcc"], as_index=False)
        .agg(cells=("cells", "sum"))
    )
    group_totals = type_counts.groupby("group", as_index=False).agg(cells=("cells", "sum"))
    sns.barplot(data=group_totals, x="group", y="cells", hue="group", palette=GROUP_COLORS, legend=False, ax=ax)
    for patch, value in zip(ax.patches, group_totals["cells"], strict=False):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value * 1.12,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_yscale("log")
    ax.set_title("E. Foreign/test leakage is tiny")
    ax.set_xlabel("")
    ax.set_ylabel("Distinct identities (log)")
    ax.tick_params(axis="x", rotation=12)

    ax = axes[1, 2]
    sns.lineplot(data=monthly, x="month", y="cells", hue="group_name", marker="o", linewidth=1.8, palette=GROUP_COLORS, ax=ax)
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(20000, monthly["cells"].max() * 1.8))
    ax.set_title("F. Recurrence over the crawl window")
    ax.set_xlabel("")
    ax.set_ylabel("Distinct identities per month (log)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="", loc="upper left", frameon=True, fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs16_shenzhen_testing_cluster.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
