#!/usr/bin/env python3
"""Render the full-width Yemen foreign-PLMN diagnostic figure from CSVs."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plot_helpers import TILE_ATTRIBUTION, add_osm_basemap, draw_geojson_layer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "out-of-country" / "yemen"
FIGS = ROOT / "paper" / "figs"
ADMIN0 = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"
ADMIN1 = ROOT / "data" / "reference" / "ne_10m_admin_1_yemen.geojson"
OUTPUT = FIGS / "yemen_foreign_plmn_diagnostic.pdf"
BASEMAP = "opentopomap"
BBOX = (42.3, 53.3, 12.0, 18.7)
CORE_BBOX = (44.25, 50.05, 13.35, 16.15)

COLORS = {
    "Gulf": "#E23D28",
    "Horn of Africa": "#8A3FA0",
    "North/West Africa": "#008FC4",
    "Other foreign": "#777777",
    "Yemen MCC": "#555555",
}
EDGES = {
    "Gulf": "#8F2114",
    "Horn of Africa": "#4D1C61",
    "North/West Africa": "#004D78",
    "Other foreign": "#4D4D4D",
}

COUNTRY_ORDER = ["AE", "SA", "OM", "SO", "NE", "DZ", "ER", "DJ"]
COUNTRY_SHORT = {
    "AE": "UAE", "SA": "Saudi Arabia", "OM": "Oman", "SO": "Somalia",
    "NE": "Niger", "DZ": "Algeria", "ER": "Eritrea", "DJ": "Djibouti",
    "YE": "Yemen", "OTHER": "Other",
}
GOV_ORDER = ["Marib", "Shabwah", "Hadramawt", "Aden", "Al Mahrah", "Abyan", "Hajjah", "Saada"]
CITIES = [
    ("Sana'a", 44.21, 15.37), ("Marib", 45.33, 15.46),
    ("Ataq", 46.83, 14.54), ("Aden", 45.03, 12.79),
    ("Mukalla", 49.13, 14.54), ("Al Ghaydah", 52.18, 16.21),
]


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(rf"$\bf{{({label})}}$ {title}", loc="left", pad=3.0)


def setup_map(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float] = BBOX,
    *,
    zoom: int = 8,
    admin1_width: float = 0.38,
) -> None:
    ax.set_facecolor("#dceaf2")
    add_osm_basemap(
        ax, bbox, zoom=zoom, source=BASEMAP, alpha=0.92,
        grayscale=True, grayscale_brightness=1.02, grayscale_contrast=1.08,
        zorder=0,
    )
    draw_geojson_layer(ax, ADMIN1, bbox, facecolor="none", edgecolor="#746f68",
                       linewidth=admin1_width, alpha=0.82, zorder=2.0)
    draw_geojson_layer(ax, ADMIN0, bbox, countries={"YE"}, facecolor="none",
                       edgecolor="#373431", linewidth=0.75, alpha=0.96, zorder=2.2)
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    mean_lat = (bbox[2] + bbox[3]) / 2
    ax.set_aspect(1 / math.cos(math.radians(mean_lat)), adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#706a63"); spine.set_linewidth(0.45)


def draw_identity_points(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    local_size: float,
    foreign_size: float,
) -> None:
    # Plot the identity-level CSV directly.  The previous version collapsed
    # the 29,897 Yemen-MCC identities into three contour lines, which hid most
    # of the sample.  Exact-coordinate overplotting is negligible (six rows in
    # the complete 34,209-row file), so each row remains a visible observation.
    local = [r for r in rows if r["is_foreign"] == "0"]
    ax.scatter(
        [float(r["lon"]) for r in local], [float(r["lat"]) for r in local],
        s=local_size, c="#252A31", edgecolors="none", alpha=0.34,
        rasterized=True, zorder=2.45,
    )
    foreign = [r for r in rows if r["is_foreign"] == "1"]
    order = ["Other foreign", "North/West Africa", "Horn of Africa", "Gulf"]
    for family in order:
        rr = [r for r in foreign if r["family"] == family]
        ax.scatter(
            [float(r["lon"]) for r in rr], [float(r["lat"]) for r in rr],
            s=foreign_size, c=COLORS[family], edgecolors=EDGES[family],
            linewidths=0.18, alpha=0.88, rasterized=True, zorder=3.0,
        )


def draw_city_labels(ax: plt.Axes, bbox: tuple[float, float, float, float], *, fontsize: float) -> None:
    for label, lon, lat in CITIES:
        if not (bbox[0] <= lon <= bbox[1] and bbox[2] <= lat <= bbox[3]):
            continue
        ax.plot(lon, lat, "o", ms=1.35, color="#25221F", zorder=4.2)
        txt = ax.annotate(label, (lon, lat), xytext=(2.0, 1.2), textcoords="offset points",
                          fontsize=fontsize, weight="semibold", color="#25221F", zorder=4.3)
        txt.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])


def draw_map(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    setup_map(ax)
    draw_identity_points(ax, rows, local_size=0.68, foreign_size=3.8)
    draw_city_labels(ax, BBOX, fontsize=5.4)

    # The country-wide panel establishes coverage; this inset makes the main
    # Marib--Shabwah--Hadramawt concentration readable without discarding the
    # geographically sparse observations elsewhere in Yemen.
    inset = ax.inset_axes([0.525, 0.055, 0.445, 0.365], zorder=5)
    setup_map(inset, CORE_BBOX, zoom=9, admin1_width=0.32)
    core = [r for r in rows if CORE_BBOX[0] <= float(r["lon"]) <= CORE_BBOX[1]
            and CORE_BBOX[2] <= float(r["lat"]) <= CORE_BBOX[3]]
    draw_identity_points(inset, core, local_size=0.42, foreign_size=3.0)
    draw_city_labels(inset, CORE_BBOX, fontsize=4.1)
    inset.set_title("Marib--Shabwah--Hadramawt detail", loc="left", fontsize=4.5,
                    pad=1.2, color="#25221F",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8})
    ax.add_patch(Rectangle(
        (CORE_BBOX[0], CORE_BBOX[2]), CORE_BBOX[1] - CORE_BBOX[0],
        CORE_BBOX[3] - CORE_BBOX[2], fill=False, edgecolor="#25221F",
        linewidth=0.5, linestyle=(0, (2, 1.5)), alpha=0.8, zorder=4.5,
    ))

    local = [r for r in rows if r["is_foreign"] == "0"]
    foreign = [r for r in rows if r["is_foreign"] == "1"]
    handles = [
        Line2D([0], [0], marker=".", linestyle="", markersize=3.6,
               color=COLORS["Yemen MCC"], alpha=0.70, label=f"Yemen MCC ({len(local):,})"),
        *[Line2D([0], [0], marker="o", linestyle="", markersize=3.6,
                 markerfacecolor=COLORS[f], markeredgecolor=EDGES[f], markeredgewidth=0.35,
                 label=f"{f} ({sum(r['family'] == f for r in foreign):,})")
          for f in ["Gulf", "Horn of Africa", "North/West Africa", "Other foreign"]],
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=5.1,
              frameon=True, framealpha=0.88, facecolor="white", edgecolor="#B8B2AA",
              handlelength=1.2, handletextpad=0.45, columnspacing=0.8, borderpad=0.35)
    panel_title(ax, "a", "All 34,209 geolocated identities")


def draw_country_bars(ax: plt.Axes, summary: list[dict[str, str]]) -> None:
    by_iso = {r["home_iso"]: r for r in summary if r["is_foreign"] == "1"}
    items = [(iso, int(by_iso[iso]["cells"]), by_iso[iso]["family"]) for iso in COUNTRY_ORDER]
    used = set(COUNTRY_ORDER)
    other = sum(int(r["cells"]) for r in summary if r["is_foreign"] == "1" and r["home_iso"] not in used)
    items.append(("OTHER", other, "Other foreign"))
    labels = [COUNTRY_SHORT[i[0]] for i in items]
    values = [i[1] for i in items]
    colors = [COLORS[i[2]] for i in items]
    y = np.arange(len(items))
    ax.barh(y, values, color=colors, height=0.66, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlim(0, 1100); ax.set_xticks([0, 500, 1000])
    ax.set_xlabel("Distinct cell identities", labelpad=1.5)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.45, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 18, yi, f"{value:,}", va="center", fontsize=5.7)
    panel_title(ax, "b", "Foreign MCC composition")


def draw_matrix(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    foreign = Counter()
    local = Counter()
    for r in rows:
        gov, iso, cells = r["governorate"], r["home_iso"], int(r["cells"])
        if r["is_foreign"] == "1":
            col = iso if iso in COUNTRY_ORDER else "OTHER"
            foreign[(gov, col)] += cells
        else:
            local[gov] += cells
    columns = COUNTRY_ORDER + ["OTHER"]
    for yi, gov in enumerate(GOV_ORDER):
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#F4F1EC", zorder=0)
        for xi, iso in enumerate(columns):
            value = foreign[(gov, iso)]
            if not value:
                continue
            family = "Other foreign"
            if iso != "OTHER":
                if iso in {"AE", "SA", "OM"}: family = "Gulf"
                elif iso in {"SO", "ER", "DJ"}: family = "Horn of Africa"
                elif iso in {"NE", "DZ"}: family = "North/West Africa"
            size = 8 + 105 * math.sqrt(value / 700)
            ax.scatter(xi, yi, s=size, color=COLORS[family], edgecolor=EDGES[family],
                       linewidth=0.35, alpha=0.92, zorder=2)
            if value >= 180:
                ax.text(xi, yi, str(value), ha="center", va="center", fontsize=4.5,
                        color="white", weight="bold", zorder=3)
        total_foreign = sum(foreign[(gov, iso)] for iso in columns)
        denominator = total_foreign + local[gov]
        share = 100 * total_foreign / denominator if denominator else 0
        ax.text(len(columns) + 0.25, yi, f"{share:.0f}%", ha="center", va="center",
                fontsize=5.8, weight="semibold" if share >= 50 else "normal")
    ax.axvline(len(columns) - 0.45, color="#B9B4AC", linewidth=0.6)
    ax.set_xlim(-0.6, len(columns) + 0.8); ax.set_ylim(len(GOV_ORDER) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(GOV_ORDER)), GOV_ORDER)
    ax.set_xticks(np.arange(len(columns)) + 0.0,
                  [COUNTRY_SHORT.get(x, x) if x != "OTHER" else "Other" for x in columns],
                  rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(length=0, pad=1.5)
    ax.spines[:].set_visible(False)
    ax.text(len(columns) + 0.25, -0.78, "Foreign\nshare", ha="center", va="bottom", fontsize=5.5)
    panel_title(ax, "c", "Co-occurrence by governorate")


def draw_lac(ax: plt.Axes, summary: list[dict[str, str]]) -> None:
    by_iso = {r["home_iso"]: r for r in summary}
    order = COUNTRY_ORDER + ["YE"]
    values = [100 * float(by_iso[iso]["lac_294xx_share"]) for iso in order]
    colors = [COLORS[by_iso[iso]["family"]] for iso in order]
    x = np.arange(len(order))
    ax.vlines(x, 0, values, colors=colors, linewidth=1.4)
    ax.scatter(x, values, s=13, c=colors, edgecolor="white", linewidth=0.35, zorder=2)
    ax.set_ylim(0, 106); ax.set_yticks([0, 50, 100]); ax.set_ylabel("LAC 294xx (%)", labelpad=1)
    ax.set_xticks(x, order)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1)
    ax.set_title("Shared identifier signature", loc="left", fontsize=6.7, pad=2)


def weighted_ecdf(rows: list[dict[str, str]], population: str) -> tuple[np.ndarray, np.ndarray]:
    rr = sorted((int(r["span_days"]), int(r["cells"])) for r in rows if r["population"] == population)
    x = np.asarray([r[0] for r in rr], dtype=float)
    counts = np.asarray([r[1] for r in rr], dtype=float)
    return x, np.cumsum(counts) / counts.sum()


def draw_lifetime(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    populations = [
        ("Foreign-coded GSM", COLORS["Gulf"], "Foreign (median 86 d)"),
        ("Yemen-coded GSM", COLORS["Yemen MCC"], "Yemen (median 441 d)"),
    ]
    for population, color, label in populations:
        x, y = weighted_ecdf(rows, population)
        ax.step(x, y, where="post", color=color, linewidth=1.25, label=label)
    ax.axvspan(80, 95, color=COLORS["Gulf"], alpha=0.10, linewidth=0)
    ax.set_xlim(0, 900); ax.set_ylim(0, 1.01)
    ax.set_xticks([0, 300, 600, 900]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Observed lifespan (days)", labelpad=1); ax.set_ylabel("ECDF", labelpad=1)
    ax.grid(color="#DDDDDD", linewidth=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=5.1, loc="lower right", handlelength=1.5, borderpad=0.1)
    ax.set_title("Synchronized lifetimes", loc="left", fontsize=6.7, pad=2)


def render(output: Path = OUTPUT) -> None:
    style()
    map_rows = read_csv("yemen-cell-identities.csv")
    summary = read_csv("yemen-country-summary.csv")
    gov_rows = read_csv("yemen-governorate-country.csv")
    lifetime = read_csv("yemen-lifetime-distribution.csv")

    fig = plt.figure(figsize=(7.0, 4.75), constrained_layout=False)
    outer = fig.add_gridspec(
        2, 2, width_ratios=(1.72, 0.82), height_ratios=(1.24, 0.94),
        left=0.055, right=0.985, bottom=0.095, top=0.96, wspace=0.20, hspace=0.30,
    )
    ax_map = fig.add_subplot(outer[0, 0])
    ax_bar = fig.add_subplot(outer[0, 1])
    ax_matrix = fig.add_subplot(outer[1, 0])
    diagnostics = outer[1, 1].subgridspec(2, 1, hspace=0.62)
    ax_lac = fig.add_subplot(diagnostics[0, 0])
    ax_life = fig.add_subplot(diagnostics[1, 0])

    draw_map(ax_map, map_rows)
    draw_country_bars(ax_bar, summary)
    draw_matrix(ax_matrix, gov_rows)
    draw_lac(ax_lac, summary)
    draw_lifetime(ax_life, lifetime)
    fig.text(0.692, 0.438, r"$\bf{(d)}$ Identifier and temporal diagnostics", fontsize=8)
    attribution = TILE_ATTRIBUTION[BASEMAP].replace(r"\copyright{}", "©")
    fig.text(0.985, 0.018, attribution + "; boundaries: Natural Earth", ha="right",
             fontsize=4.4, color="#666666")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600)
    fig.savefig(output.with_suffix(".png"), dpi=450)
    plt.close(fig)
    print(f"[figure] {output.relative_to(ROOT)}")


def main() -> None:
    render()


if __name__ == "__main__":
    main()
