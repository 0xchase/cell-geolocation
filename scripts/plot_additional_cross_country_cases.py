#!/usr/bin/env python3
"""Render compact maps for additional 25 km-buffered MCC case studies."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_helpers import (
    ADMIN1_GEOJSON,
    COUNTRIES_GEOJSON,
    TILE_ATTRIBUTION,
    add_osm_basemap,
    draw_geojson_layer,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "out-of-country" / "additional-cases"
FIGS = ROOT / "paper" / "figs"
MAP_UNITS = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
BASEMAP = "carto_voyager_nolabels_retina"

PALETTE = {
    "AE": "#D55E00", "SA": "#E69F00", "OM": "#009E73",
    "SO": "#7A3E9D", "NE": "#56B4E9", "DZ": "#CC79A7",
    "HK": "#7A3E9D", "CN": "#D55E00", "MO": "#0072B2",
    "AM": "#7A3E9D", "OTHER": "#666666",
}

LABELS = {
    "AE": "UAE", "SA": "Saudi Arabia", "OM": "Oman", "SO": "Somalia",
    "NE": "Niger", "DZ": "Algeria", "HK": "Hong Kong", "CN": "China",
    "MO": "Macao", "AM": "Armenia", "OTHER": "Other MCCs",
}

APPENDIX_CASES = [
    dict(key="yemen-foreign", title="(a) Yemen", bbox=(42.7, 52.7, 12.3, 17.4),
         countries={"YE", "SA", "OM", "DJ", "ER", "SO"}, zoom=8,
         cities=[("Sana'a", 44.21, 15.37), ("Marib", 45.33, 15.46),
                 ("Aden", 45.03, 12.79), ("Mukalla", 49.13, 14.54)]),
    dict(key="myanmar-foreign", title="(b) Myanmar", bbox=(95.5, 101.7, 18.8, 27.8),
         countries={"MM", "CN", "IN", "TH", "LA", "BD"}, zoom=8,
         cities=[("Yangon", 96.16, 16.84), ("Mandalay", 96.08, 21.98),
                 ("Lashio", 97.75, 22.94), ("Myitkyina", 97.40, 25.38)]),
    dict(key="azerbaijan-armenia", title="(c) Azerbaijan: Armenian MCC", bbox=(44.5, 50.0, 38.7, 42.0),
         countries={"AZ", "AM", "GE", "IR", "TR"}, zoom=8,
         cities=[("Yerevan", 44.51, 40.18), ("Khankendi", 46.75, 39.82),
                 ("Baku", 49.87, 40.41)]),
]


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def read_case(key: str) -> list[dict[str, str]]:
    path = DATA / f"{key}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def setup_map(ax: plt.Axes, spec: dict) -> None:
    bbox = spec["bbox"]
    ax.set_facecolor("#dceaf2")
    drawn = add_osm_basemap(ax, bbox, zoom=spec["zoom"], source=BASEMAP, alpha=0.96, zorder=0)
    if not drawn:
        draw_geojson_layer(ax, COUNTRIES_GEOJSON, bbox, countries=spec["countries"],
                           facecolor="#f5f1e8", edgecolor="#69635c", linewidth=0.4, zorder=0)
    if ADMIN1_GEOJSON.exists():
        draw_geojson_layer(ax, ADMIN1_GEOJSON, bbox, countries=spec["countries"],
                           facecolor="none", edgecolor="#8b837a", linewidth=0.22, alpha=0.65, zorder=2)
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect(1 / math.cos(math.radians((bbox[2] + bbox[3]) / 2)), adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#817a72"); spine.set_linewidth(0.42)


def categories(key: str, rows: list[dict[str, str]]) -> list[str]:
    homes = [r["mcc_country_iso"] for r in rows]
    if key == "yemen-foreign":
        order = ["AE", "SA", "OM", "SO", "NE", "DZ"]
        return [x for x in order if x in homes] + (["OTHER"] if any(x not in order for x in homes) else [])
    order = ["HK", "CN", "MO", "AM"]
    return [x for x in order if x in homes]


def draw_case(ax: plt.Axes, spec: dict) -> None:
    rows = read_case(spec["key"])
    setup_map(ax, spec)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    cats = categories(spec["key"], rows)
    for row in rows:
        cat = row["mcc_country_iso"]
        if spec["key"] == "yemen-foreign" and cat not in cats:
            cat = "OTHER"
        grouped[cat].append(row)
    for cat in reversed(cats):
        rr = grouped[cat]
        counts = np.asarray([int(r["cells"]) for r in rr], dtype=float)
        ax.scatter(
            [float(r["lon"]) for r in rr], [float(r["lat"]) for r in rr],
            s=0.7 + 1.0 * np.log10(counts + 1), marker="s", color=PALETTE[cat],
            alpha=0.88, edgecolor="none", rasterized=True, zorder=3,
        )
    if MAP_UNITS.exists():
        draw_geojson_layer(ax, MAP_UNITS, spec["bbox"], countries=spec["countries"],
                           facecolor="none", edgecolor="#443f3a", linewidth=0.52,
                           alpha=0.95, zorder=4.5)
    for label, lon, lat in spec["cities"]:
        if not (spec["bbox"][0] <= lon <= spec["bbox"][1] and spec["bbox"][2] <= lat <= spec["bbox"][3]):
            continue
        ax.plot(lon, lat, "o", ms=1.4, color="#27231f", zorder=5.1)
        txt = ax.annotate(label, (lon, lat), xytext=(1.8, 1.4), textcoords="offset points",
                          fontsize=4.2, weight="semibold", color="#27231f", zorder=5.2)
        txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white")])
    total = sum(int(r["cells"]) for r in rows)
    ax.set_title(f'{spec["title"]}  ({total:,})', fontsize=7.2, weight="semibold", pad=2.5)
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=3.2,
                      markerfacecolor=PALETTE[c], markeredgecolor="none", label=LABELS[c]) for c in cats]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.035),
              ncol=min(3, len(handles)), frameon=False, fontsize=4.5,
              handletextpad=0.3, columnspacing=0.7)


def render_appendix(output: Path) -> None:
    """Lay out the three retained cases at their natural map aspect ratios."""
    by_key = {spec["key"]: spec for spec in APPENDIX_CASES}
    fig = plt.figure(figsize=(7.0, 5.2))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(1.45, 0.75), height_ratios=(0.86, 1.14),
        left=0.025, right=0.985, bottom=0.055, top=0.965,
        wspace=0.10, hspace=0.24,
    )
    axes = [
        (fig.add_subplot(gs[0, 0]), by_key["yemen-foreign"]),
        (fig.add_subplot(gs[:, 1]), by_key["myanmar-foreign"]),
        (fig.add_subplot(gs[1, 0]), by_key["azerbaijan-armenia"]),
    ]
    for ax, spec in axes:
        draw_case(ax, spec)
    attribution = TILE_ATTRIBUTION[BASEMAP].replace(r"\copyright{}", "©")
    fig.text(0.985, 0.008, attribution, ha="right", fontsize=4.5, color="#555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output.relative_to(ROOT)}")


def main() -> None:
    style()
    render_appendix(FIGS / "additional_cross_country_appendix.pdf")


if __name__ == "__main__":
    main()
