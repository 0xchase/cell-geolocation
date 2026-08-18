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
    "JM": "#D55E00", "MC": "#7A3E9D", "RU": "#C43C39",
    "AE": "#D55E00", "SA": "#E69F00", "OM": "#009E73",
    "SO": "#7A3E9D", "NE": "#56B4E9", "DZ": "#CC79A7",
    "HK": "#7A3E9D", "CN": "#D55E00", "MO": "#0072B2",
    "IL": "#D55E00", "TR": "#D55E00", "CY": "#0072B2",
    "AM": "#7A3E9D", "OTHER": "#666666",
}

LABELS = {
    "JM": "Jamaican MCC", "MC": "Monaco MCC", "RU": "Russian MCC",
    "AE": "UAE", "SA": "Saudi Arabia", "OM": "Oman", "SO": "Somalia",
    "NE": "Niger", "DZ": "Algeria", "HK": "Hong Kong", "CN": "China",
    "MO": "Macao", "IL": "Israel", "TR": "Türkiye", "CY": "Cyprus",
    "AM": "Armenia", "OTHER": "Other MCCs",
}

MAJOR = [
    dict(key="caribbean-jamaica", title="(a) Caribbean: Jamaican MCC",
         bbox=(-79.0, -58.0, 10.5, 26.5), countries=set(), zoom=5,
         cities=[("Port-au-Prince", -72.31, 18.54), ("Kingston", -76.79, 17.97),
                 ("Bridgetown", -59.62, 13.10)]),
    dict(key="kosovo-monaco", title="(b) Kosovo: Monaco MCC",
         bbox=(20.0, 21.9, 41.8, 43.3), countries={"XK", "RS", "AL", "MK", "ME"}, zoom=8,
         cities=[("Pristina", 21.17, 42.66), ("Prizren", 20.74, 42.21)]),
]

SECONDARY = [
    dict(key="yemen-foreign", title="(a) Yemen", bbox=(41.5, 54.7, 12.0, 18.8),
         countries={"YE", "SA", "OM", "DJ", "ER", "SO"}, zoom=6,
         cities=[("Sana'a", 44.21, 15.37), ("Marib", 45.33, 15.46),
                 ("Aden", 45.03, 12.79), ("Mukalla", 49.13, 14.54)]),
    dict(key="myanmar-foreign", title="(b) Myanmar", bbox=(92.0, 102.6, 9.5, 28.8),
         countries={"MM", "CN", "IN", "TH", "LA", "BD"}, zoom=5,
         cities=[("Yangon", 96.16, 16.84), ("Mandalay", 96.08, 21.98),
                 ("Lashio", 97.75, 22.94), ("Myitkyina", 97.40, 25.38)]),
    dict(key="cyprus-turkiye", title="(c) Cyprus–Türkiye", bbox=(31.8, 35.0, 34.5, 36.7),
         countries={"CY", "TR"}, zoom=7,
         cities=[("Nicosia", 33.38, 35.19), ("Kyrenia", 33.32, 35.34),
                 ("Mersin", 34.63, 36.81)]),
    dict(key="azerbaijan-armenia", title="(d) Azerbaijan: Armenian MCC", bbox=(44.3, 50.6, 38.3, 42.3),
         countries={"AZ", "AM", "GE", "IR", "TR"}, zoom=7,
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
    order = ["JM", "MC", "RU", "HK", "CN", "MO", "IL", "TR", "CY", "AM"]
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


def render(specs: list[dict], output: Path, figsize: tuple[float, float], shape: tuple[int, int],
           width_ratios: list[float] | None = None) -> None:
    fig, axes = plt.subplots(*shape, figsize=figsize, squeeze=False)
    if width_ratios is not None:
        plt.close(fig)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(*shape, width_ratios=width_ratios, left=0.025, right=0.985,
                              bottom=0.12, top=0.94, wspace=0.10, hspace=0.28)
        axes = np.asarray([[fig.add_subplot(gs[r, c]) for c in range(shape[1])] for r in range(shape[0])])
    else:
        fig.subplots_adjust(left=0.025, right=0.985, bottom=0.10, top=0.94, wspace=0.10, hspace=0.24)
    flat = list(axes.flat)
    for ax, spec in zip(flat, specs, strict=False):
        draw_case(ax, spec)
    for ax in flat[len(specs):]: ax.set_visible(False)
    attribution = TILE_ATTRIBUTION[BASEMAP].replace(r"\copyright{}", "©")
    fig.text(0.985, 0.012, attribution, ha="right", fontsize=4.5, color="#555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output.relative_to(ROOT)}")


def main() -> None:
    style()
    render(MAJOR, FIGS / "additional_cross_country_major.pdf", (7.0, 2.45), (1, 2), [1.55, 0.95])
    render(SECONDARY, FIGS / "additional_cross_country_secondary.pdf", (7.0, 4.45), (2, 2))


if __name__ == "__main__":
    main()
