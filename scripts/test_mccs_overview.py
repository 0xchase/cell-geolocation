#!/usr/bin/env python3
"""Plot broad statistics for testing, unassigned, and private MCCs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "test-mccs"
WORLD = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
FIGS = ROOT / "paper" / "figs"

CATEGORIES = ("Testing", "Unassigned", "Private")
FILES = {
    "Testing": DATA / "testing.csv",
    "Unassigned": DATA / "unassigned.csv",
    "Private": DATA / "private.csv",
}
CATEGORY_COLORS = {
    "Testing": "#b23a48",
    "Unassigned": "#c9743a",
    "Private": "#2f6f9f",
}
TECHNOLOGIES = ("gsm", "lte", "nr")
TECH_LABELS = {"gsm": "GSM/UMTS", "lte": "LTE", "nr": "NR"}
TECH_COLORS = {"gsm": "#8a8f98", "lte": "#4b78a8", "nr": "#4f7f52"}
PORTON_BOUNDARY = [
    [-1.708669, 51.1353327], [-1.7031176, 51.13344], [-1.7024261, 51.1337172],
    [-1.7026833, 51.1337839], [-1.7026298, 51.1338769], [-1.7023465, 51.1337917],
    [-1.7022836, 51.1338741], [-1.7025859, 51.1339649], [-1.7022074, 51.1343589],
    [-1.7009512, 51.1339583], [-1.7011512, 51.1336944], [-1.7000206, 51.1333452],
    [-1.7003601, 51.1329419], [-1.6983404, 51.1323016], [-1.6955596, 51.1351431],
    [-1.6956388, 51.1352712], [-1.7024624, 51.1375074], [-1.7033709, 51.1364123],
    [-1.7052305, 51.1369574], [-1.708669, 51.1353327],
]
TESLA_BOUNDARY = [
    [13.7867413, 52.4003316], [13.7869545, 52.400241], [13.786879, 52.399925],
    [13.7867982, 52.3979876], [13.7871134, 52.3901593], [13.7879113, 52.3894237],
    [13.7967227, 52.39091], [13.8004229, 52.3927934], [13.8017892, 52.3934618],
    [13.8022817, 52.396216], [13.8039217, 52.3980215], [13.8132805, 52.4018704],
    [13.8149751, 52.4038538], [13.8147832, 52.4046215], [13.813951, 52.4049964],
    [13.8097711, 52.4049881], [13.8093825, 52.404052], [13.8065934, 52.4039092],
    [13.8065604, 52.4031275], [13.8041711, 52.4030866], [13.8041175, 52.4012974],
    [13.8037237, 52.4011155], [13.8021048, 52.4010421], [13.7890939, 52.4010025],
    [13.7890128, 52.4027394], [13.7872794, 52.4028543], [13.7868393, 52.4021933],
    [13.7867413, 52.4003316],
]


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in ("mcc", "mnc", "obs"):
                row[key] = int(raw[key])
            for key in ("glat", "glon"):
                row[key] = float(raw[key])
            row["duration_days"] = (
                datetime.fromisoformat(raw["last_seen"])
                - datetime.fromisoformat(raw["first_seen"])
            ).total_seconds() / 86400
            rows.append(row)
    return rows


def polygon_rings(geometry: dict) -> list[list[list[float]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [polygon[0] for polygon in geometry["coordinates"]]
    return []


def draw_world(ax: plt.Axes, features: list[dict]) -> None:
    ax.set_facecolor("#edf3f5")
    for feature in features:
        for ring in polygon_rings(feature["geometry"]):
            xy = np.asarray(ring)
            if xy.ndim != 2 or len(xy) < 3:
                continue
            ax.add_patch(
                Polygon(
                    xy,
                    closed=True,
                    facecolor="#f3efe7",
                    edgecolor="#aaa39a",
                    linewidth=0.12,
                    rasterized=True,
                    zorder=0,
                )
            )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 82)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#aaa39a")
        spine.set_linewidth(0.45)


def draw_region(ax: plt.Axes, features: list[dict], extent: tuple[float, float, float, float]) -> None:
    draw_world(ax, features)
    west, east, south, north = extent
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1 / math.cos(math.radians((south + north) / 2)))
    ax.grid(color="#ffffff", linewidth=0.55, alpha=0.9, zorder=1)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_xticks(np.linspace(west, east, 3))
    ax.set_yticks(np.linspace(south, north, 3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{abs(value):.2f}°{'W' if value < 0 else 'E'}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{abs(value):.2f}°{'S' if value < 0 else 'N'}"))


def human_number(value: float, _position=None) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def make_figure(rows_by_category: dict[str, list[dict[str, object]]], output: Path, preview: Path | None) -> None:
    with WORLD.open(encoding="utf-8") as handle:
        world_features = json.load(handle)["features"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(15.0, 8.5))
    grid = fig.add_gridspec(2, 12, height_ratios=(0.82, 1.18), hspace=0.52, wspace=0.48)
    ax_totals = fig.add_subplot(grid[0, 0:4])
    ax_tech = fig.add_subplot(grid[0, 4:8])
    ax_rank = fig.add_subplot(grid[0, 8:12])
    ax_world = fig.add_subplot(grid[1, 0:3])
    ax_testing = fig.add_subplot(grid[1, 3:6])
    ax_unassigned = fig.add_subplot(grid[1, 6:9])
    ax_private = fig.add_subplot(grid[1, 9:12])
    fig.suptitle("Testing, unassigned, and private MCCs form distinct populations", fontsize=14, fontweight="bold", y=0.985)

    # A. Census totals.
    identities = np.array([len(rows_by_category[c]) for c in CATEGORIES])
    observations = np.array([sum(int(r["obs"]) for r in rows_by_category[c]) for c in CATEGORIES])
    x = np.arange(len(CATEGORIES))
    width = 0.34
    bars1 = ax_totals.bar(x - width / 2, identities, width, color=[CATEGORY_COLORS[c] for c in CATEGORIES], label="Cell identities")
    bars2 = ax_totals.bar(x + width / 2, observations, width, color=[CATEGORY_COLORS[c] for c in CATEGORIES], alpha=0.34, hatch="///", label="Database observations")
    ax_totals.set_yscale("log")
    ax_totals.set_ylim(900, 6_000_000)
    ax_totals.set_xticks(x, CATEGORIES)
    ax_totals.set_ylabel("Count (log scale)")
    ax_totals.set_title("A. Identity and observation census", fontsize=10)
    ax_totals.grid(axis="y", linewidth=0.35, alpha=0.45)
    ax_totals.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=2)
    for bar, value in zip(bars1, identities, strict=True):
        ax_totals.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:,}", ha="center", va="bottom", fontsize=7.5)
    for bar, value in zip(bars2, observations, strict=True):
        ax_totals.text(bar.get_x() + bar.get_width() / 2, value / 1.22, f"{value:,}", ha="center", va="top", fontsize=7.5)

    # B. Technology composition.
    y = np.arange(len(CATEGORIES))
    left = np.zeros(len(CATEGORIES))
    for technology in TECHNOLOGIES:
        counts = np.array([sum(r["cell_type"] == technology for r in rows_by_category[c]) for c in CATEGORIES])
        shares = counts / identities * 100
        bars = ax_tech.barh(y, shares, left=left, height=0.56, color=TECH_COLORS[technology], label=TECH_LABELS[technology])
        for i, (bar, share, count) in enumerate(zip(bars, shares, counts, strict=True)):
            if share >= 3.0:
                ax_tech.text(left[i] + share / 2, bar.get_y() + bar.get_height() / 2, f"{share:.0f}%\n{count:,}", ha="center", va="center", fontsize=7, color="white" if technology != "gsm" else "#222222")
        left += shares
    ax_tech.set_yticks(y, CATEGORIES)
    ax_tech.invert_yaxis()
    ax_tech.set_xlim(0, 100)
    ax_tech.set_xlabel("Share of cell identities")
    ax_tech.set_title("B. Technology mix", fontsize=10)
    ax_tech.grid(axis="x", linewidth=0.35, alpha=0.4)
    ax_tech.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.27))

    # C. Rank-size distribution of PLMNs.
    for category in CATEGORIES:
        counts = Counter((int(r["mcc"]), int(r["mnc"])) for r in rows_by_category[category])
        ranked = np.array(sorted(counts.values(), reverse=True))
        ax_rank.plot(np.arange(1, len(ranked) + 1), ranked, color=CATEGORY_COLORS[category], linewidth=1.8, label=f"{category} ({len(ranked)} PLMNs)")
        largest_plmn, largest_count = counts.most_common(1)[0]
        ax_rank.scatter([1], [largest_count], color=CATEGORY_COLORS[category], s=24, zorder=3)
        ax_rank.annotate(
            f"{largest_plmn[0]:03d}/{largest_plmn[1]}: {largest_count:,}",
            (1, largest_count),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.2,
            color=CATEGORY_COLORS[category],
        )
    ax_rank.set_xscale("log")
    ax_rank.set_yscale("log")
    ax_rank.set_xlim(0.9, 1000)
    ax_rank.set_ylim(0.8, 40_000)
    ax_rank.set_xlabel("PLMN rank by number of cell identities (log scale)")
    ax_rank.set_ylabel("Cell identities (log scale)")
    ax_rank.set_title("C. PLMN concentration", fontsize=10)
    ax_rank.grid(which="both", linewidth=0.3, alpha=0.35)
    ax_rank.legend(frameon=False, fontsize=8, loc="upper right")
    ax_rank.yaxis.set_major_formatter(FuncFormatter(human_number))

    # D. Combined geographic coverage.
    draw_world(ax_world, world_features)
    for category in ("Unassigned", "Testing", "Private"):
        rows = rows_by_category[category]
        ax_world.scatter(
            [float(r["glon"]) for r in rows],
            [float(r["glat"]) for r in rows],
            s=3.0,
            color=CATEGORY_COLORS[category],
            alpha=0.32,
            linewidths=0,
            rasterized=True,
            label=category,
            zorder=2,
        )
    ax_world.set_title("D. Global distribution", fontsize=10)
    handles, labels = ax_world.get_legend_handles_labels()
    legend_items = dict(zip(labels, handles, strict=True))
    ax_world.legend(
        [legend_items[c] for c in CATEGORIES], CATEGORIES, frameon=False,
        fontsize=6.2, markerscale=2.0, handletextpad=0.25, columnspacing=0.65,
        loc="lower left", ncol=3,
    )
    ax_world.text(0.99, 0.01, "Natural Earth", transform=ax_world.transAxes, ha="right", va="bottom", fontsize=5.5, color="#666666")

    # E. Testing case: six LTE identities reverse-geocode to the same laboratory site.
    porton = [
        r for r in rows_by_category["Testing"]
        if int(r["mcc"]) == 1 and int(r["mnc"]) == 1
        and 51.13 < float(r["glat"]) < 51.14 and -1.71 < float(r["glon"]) < -1.69
    ]
    draw_region(ax_testing, world_features, (-1.710, -1.694, 51.1300, 51.1400))
    ax_testing.add_patch(Polygon(PORTON_BOUNDARY, closed=True, facecolor=CATEGORY_COLORS["Testing"], alpha=0.13, edgecolor=CATEGORY_COLORS["Testing"], linewidth=1.1, zorder=2))
    ax_testing.scatter(
        [float(r["glon"]) for r in porton], [float(r["glat"]) for r in porton],
        s=[20 + math.sqrt(int(r["obs"])) * 2.2 for r in porton], color=CATEGORY_COLORS["Testing"],
        edgecolor="white", linewidth=0.5, alpha=0.82, zorder=3,
    )
    ax_testing.text(0.03, 0.04, "001/1 LTE · 6 identities · 688 observations", transform=ax_testing.transAxes, fontsize=6.7, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2}, zorder=4)
    ax_testing.set_title("E. Testing: Dstl Porton Down", fontsize=10)

    # F. Unassigned case: an alias of ordinary Bahamian PLMN 364/49.
    bahamas = [
        r for r in rows_by_category["Unassigned"]
        if int(r["mcc"]) == 123 and int(r["mnc"]) == 456
    ]
    draw_region(ax_unassigned, world_features, (-79.5, -76.7, 24.45, 27.0))
    ax_unassigned.scatter(
        [float(r["glon"]) for r in bahamas], [float(r["glat"]) for r in bahamas],
        s=8, color=CATEGORY_COLORS["Unassigned"], edgecolor="white", linewidth=0.2,
        alpha=0.68, rasterized=True, zorder=3,
    )
    ax_unassigned.text(0.03, 0.04, "123/456 LTE · 83 exact IDs also occur under 364/49", transform=ax_unassigned.transAxes, fontsize=6.7, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2}, zorder=4)
    ax_unassigned.set_title("F. Unassigned alias: The Bahamas", fontsize=10)

    # G. Private case: NR identities distributed across an industrial site.
    tesla = [
        r for r in rows_by_category["Private"]
        if int(r["mcc"]) == 999 and int(r["mnc"]) == 40 and r["cell_type"] == "nr"
        and 52.38 < float(r["glat"]) < 52.41 and 13.77 < float(r["glon"]) < 13.82
    ]
    draw_region(ax_private, world_features, (13.784, 13.817, 52.388, 52.406))
    ax_private.add_patch(Polygon(TESLA_BOUNDARY, closed=True, facecolor=CATEGORY_COLORS["Private"], alpha=0.12, edgecolor=CATEGORY_COLORS["Private"], linewidth=1.1, zorder=2))
    ax_private.scatter(
        [float(r["glon"]) for r in tesla], [float(r["glat"]) for r in tesla],
        s=22, color=CATEGORY_COLORS["Private"], edgecolor="white", linewidth=0.45,
        alpha=0.82, zorder=3,
    )
    ax_private.text(0.03, 0.04, "999/40 NR · 20 identities · 8,865 observations", transform=ax_private.transAxes, fontsize=6.7, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2}, zorder=4)
    ax_private.set_title("G. Private: Tesla Gigafactory Berlin", fontsize=10)

    fig.text(0.995, 0.008, "Boundaries: Natural Earth; site outlines: © OpenStreetMap contributors", ha="right", va="bottom", fontsize=6, color="#666666")
    fig.subplots_adjust(top=0.91, bottom=0.07, left=0.055, right=0.99)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FIGS / "test_mccs_overview.pdf")
    parser.add_argument("--preview", type=Path, default=FIGS / "test_mccs_overview.png")
    args = parser.parse_args()
    rows = {category: load_rows(path) for category, path in FILES.items()}
    make_figure(rows, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
