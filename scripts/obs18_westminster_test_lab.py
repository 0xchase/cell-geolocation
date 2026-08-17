#!/usr/bin/env python3
"""Generate a four-panel figure for the Westminster, Colorado multi-country test lab."""

from __future__ import annotations

import argparse
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
    1: "001 test",
    111: "111 unassigned",
    310: "US 310",
    311: "US 311",
    333: "333 unassigned",
    432: "Iran",
    694: "694 unassigned",
    706: "El Salvador",
    777: "777 unassigned",
    913: "913 unassigned",
    964: "964 unassigned",
    988: "988 unassigned",
}

COLORS = {"Unassigned/test": "#b23a48", "Real foreign": "#2f6f9f", "US": "#4f7f52"}
WESTMOOR_CONTEXT = [
    "Westmoor Place is a technology-office campus at 11000/11300/11400 Westmoor Circle.",
    "LGS/CACI is documented at 11300; its public profile includes wireless communications equipment, spectrum management, SIGINT/EW, and cyber.",
    "Campus tenant reporting also names General Dynamics and Coalfire; Inovonics Wireless is documented at 11000.",
    "Interpretation: likely benign RF/cellular test activity at campus scale, not a public carrier site. Specific-tenant attribution remains unresolved.",
]


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def classify(mcc: int) -> str:
    if mcc in {1, 111, 333, 694, 777, 913, 964, 988}:
        return "Unassigned/test"
    if mcc in {310, 311}:
        return "US"
    return "Real foreign"


def load_data() -> dict[str, pd.DataFrame]:
    bbox = "lat BETWEEN 39.8955 AND 39.8965 AND lon BETWEEN -105.1255 AND -105.1245"
    detail = ch_df(
        f"""
        SELECT mcc, mnc, cell_type, count() AS obs,
               uniqExact((mcc,mnc,lac,cid)) AS cells,
               min(timestamp) AS first_seen,
               max(timestamp) AS last_seen,
               avg(lat) AS avg_lat,
               avg(lon) AS avg_lon
        FROM cell.geos
        WHERE cid > 0 AND {bbox}
        GROUP BY mcc, mnc, cell_type
        ORDER BY obs DESC
        """
    )
    coords = ch_df(
        f"""
        SELECT round(lat, 5) AS lat, round(lon, 5) AS lon,
               count() AS obs, uniqExact((mcc,mnc,lac,cid)) AS cells, uniqExact(mcc) AS mccs
        FROM cell.geos
        WHERE cid > 0 AND {bbox}
        GROUP BY lat, lon
        ORDER BY obs DESC
        """
    )
    monthly = ch_df(
        f"""
        SELECT toStartOfMonth(timestamp) AS month,
               multiIf(mcc IN (1,111,333,694,777,913,964,988), 'Unassigned/test',
                       mcc IN (310,311), 'US',
                       'Real foreign') AS class,
               count() AS obs,
               uniqExact((mcc,mnc,lac,cid)) AS cells
        FROM cell.geos
        WHERE cid > 0 AND {bbox}
        GROUP BY month, class
        ORDER BY month, class
        """
    )
    summary = ch_df(
        f"""
        SELECT uniqExact(mcc) AS mccs, count() AS obs, uniqExact((mcc,mnc,lac,cid)) AS cells,
               min(timestamp) AS first_seen, max(timestamp) AS last_seen
        FROM cell.geos
        WHERE cid > 0 AND {bbox}
        """
    )
    for df in [detail]:
        df["country"] = df["mcc"].map(MCC_NAME).fillna(df["mcc"].map(lambda x: f"MCC {x}"))
        df["class"] = df["mcc"].map(lambda x: classify(int(x)))
        df["first_seen"] = pd.to_datetime(df["first_seen"])
        df["last_seen"] = pd.to_datetime(df["last_seen"])
    monthly["month"] = pd.to_datetime(monthly["month"])
    summary["first_seen"] = pd.to_datetime(summary["first_seen"])
    summary["last_seen"] = pd.to_datetime(summary["last_seen"])
    return {"detail": detail, "coords": coords, "monthly": monthly, "summary": summary}


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    detail = data["detail"].copy()
    coords = data["coords"].copy()
    monthly = data["monthly"].copy()
    summary = data["summary"].iloc[0]

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.04)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(15.4, 8.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.0, 0.92], height_ratios=[1.0, 1.0])
    ax_mcc = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_context = fig.add_subplot(gs[0, 2])
    ax_tech = fig.add_subplot(gs[1, 0])
    ax_monthly = fig.add_subplot(gs[1, 1:])
    fig.suptitle("A Westminster, Colorado technology campus has the fingerprint of a cellular test lab", fontsize=14, fontweight="bold")

    ax = ax_mcc
    mccs = detail.groupby(["mcc", "country", "class"], as_index=False).agg(obs=("obs", "sum"), cells=("cells", "sum")).sort_values("obs", ascending=True)
    ax.barh(mccs["country"], mccs["obs"], color=[COLORS[c] for c in mccs["class"]])
    for patch, obs, cells in zip(ax.patches, mccs["obs"], mccs["cells"], strict=False):
        ax.text(obs + 0.5, patch.get_y() + patch.get_height() / 2, f"{obs}; {cells} IDs", va="center", fontsize=7)
    ax.set_title(f"A. {int(summary['mccs'])} MCCs at one address, mostly test/unassigned")
    ax.set_xlabel("Observations")
    ax.set_ylabel("")
    ax.set_xlim(0, max(35, mccs["obs"].max() * 1.35))

    ax = ax_map
    bbox = (-105.129, -105.121, 39.892, 39.900)
    add_osm_basemap(ax, bbox, zoom=16, alpha=0.88, grayscale=True)
    campus_box = Rectangle(
        (-105.12545, 39.89525),
        0.0041,
        0.00155,
        facecolor="#f6ead0",
        edgecolor="#5c554c",
        linewidth=0.85,
        alpha=0.36,
        zorder=2,
    )
    ax.add_patch(campus_box)
    ax.scatter(coords["lon"], coords["lat"], s=55 + coords["cells"] * 18, color="#b23a48", edgecolor="white", linewidth=0.6, alpha=0.88, zorder=4)
    ax.scatter([-105.121702], [39.896593], s=55, color="#2f6f9f", marker="s", edgecolor="white", linewidth=0.6, alpha=0.92, zorder=4)
    ax.text(
        coords["lon"].mean() - 0.0004,
        coords["lat"].mean() + 0.0015,
        "Observed cell cluster\nnear 11300/11400 Westmoor",
        ha="center",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.35, "alpha": 0.82, "pad": 1.0},
        zorder=5,
    )
    ax.text(
        -105.12178,
        39.8958,
        "11000 Westmoor\npublic GPS reference",
        ha="right",
        va="top",
        fontsize=6.7,
        bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.35, "alpha": 0.82, "pad": 0.9},
        zorder=5,
    )
    ax.text(
        -105.1246,
        39.89505,
        "Westmoor Technology Park",
        ha="center",
        va="top",
        fontsize=7.0,
        color="#3c3834",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.8},
        zorder=5,
    )
    ax.set_title("B. Reports collapse onto\nWestmoor Technology Park", fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(0.02, 0.02, "Map data (c) OpenStreetMap contributors", transform=ax.transAxes, fontsize=5.8, color="#555", zorder=7)

    ax = ax_context
    ax.axis("off")
    ax.set_title("C. Online context supports\nlab-scale explanation", fontsize=10)
    y = 0.96
    for item in WESTMOOR_CONTEXT:
        ax.text(
            0.03,
            y,
            "\u2022 " + item,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#252525",
            wrap=True,
        )
        y -= 0.205
    ax.text(
        0.03,
        0.04,
        "Sources checked: CACI, GovCB/SAM profile, CREJ, Inovonics, Denver office-market reports.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#5c5752",
        wrap=True,
    )

    ax = ax_tech
    type_counts = detail.groupby(["cell_type", "class"], as_index=False).agg(obs=("obs", "sum"))
    sns.barplot(data=type_counts, x="cell_type", y="obs", hue="class", palette=COLORS, ax=ax)
    ax.set_title("D. GSM dominates, with fresh NR test-code activity")
    ax.set_xlabel("Radio technology")
    ax.set_ylabel("Observations")
    ax.legend(title="", frameon=True, fontsize=8)

    ax = ax_monthly
    sns.lineplot(data=monthly, x="month", y="obs", hue="class", marker="o", linewidth=1.8, palette=COLORS, ax=ax)
    ax.set_title("E. Recurrence over two years separates lab activity from a one-off glitch")
    ax.set_xlabel("")
    ax.set_ylabel("Observations per month")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="", frameon=True, fontsize=8)
    ax.text(
        0.98,
        0.95,
        f"{summary['first_seen'].date()} to {summary['last_seen'].date()}",
        transform=ax.transAxes,
        ha="right",
        va="top",
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
    parser.add_argument("--output", type=Path, default=PLOTS / "obs18_westminster_test_lab.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
