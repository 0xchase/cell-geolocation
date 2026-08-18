#!/usr/bin/env python3
"""Render twenty-four audited anomalous-MCC defense sites over detailed imagery."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import contextily as cx
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, shape


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "test-mccs"
OUT_OF_COUNTRY_DATA = ROOT / "data" / "out-of-country"
FIGS = ROOT / "paper" / "figs"
FACILITY_CACHE = ROOT / ".cache" / "osm_facilities"

CATEGORY_COLORS = {
    "Testing": "#ef476f",
    "Unassigned": "#ff9f1c",
    "Private": "#2ec4b6",
    "Cross-country": "#8f5bd7",
}
TECH_MARKERS = {"gsm": "s", "lte": "o", "nr": "^"}


def load_rows(filename: str, directory: Path = DATA) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (directory / filename).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for key in ("mcc", "mnc", "lac", "cid", "obs"):
                row[key] = int(raw[key])
            for key in ("glat", "glon"):
                row[key] = float(raw[key])
            rows.append(row)
    return rows


def select(
    rows_by_category: dict[str, list[dict[str, object]]],
    networks: list[tuple[str, int, int]],
    bbox: tuple[float, float, float, float],
) -> list[dict[str, object]]:
    west, east, south, north = bbox
    selected: list[dict[str, object]] = []
    for category, mcc, mnc in networks:
        for row in rows_by_category[category]:
            if (
                row["mcc"] == mcc
                and row["mnc"] == mnc
                and west <= float(row["glon"]) <= east
                and south <= float(row["glat"]) <= north
            ):
                selected.append(row)
    return selected


def osm_geometry(osm_id: str | None):
    if osm_id is None:
        return None
    FACILITY_CACHE.mkdir(parents=True, exist_ok=True)
    cache = FACILITY_CACHE / f"{osm_id}.geojson"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        query = urlencode({"osm_ids": osm_id, "format": "geojson", "polygon_geojson": 1})
        request = Request(
            f"https://nominatim.openstreetmap.org/lookup?{query}",
            headers={"User-Agent": "cell-geolocation-paper/0.1 (research plotting)"},
        )
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.loads(response.read())
            cache.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            return None
    features = payload.get("features", [])
    if not features:
        return None
    return shape(features[0]["geometry"])


def polygons(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return []


def draw_facility(ax: plt.Axes, geometry) -> None:
    if geometry is None:
        return
    for polygon in polygons(geometry):
        x, y = polygon.exterior.xy
        ax.fill(x, y, facecolor="#ffd166", alpha=0.13, edgecolor="none", zorder=2)
        line = ax.plot(x, y, color="#ffd166", linewidth=1.25, zorder=3)[0]
        line.set_path_effects([path_effects.Stroke(linewidth=2.8, foreground="white"), path_effects.Normal()])


def grouped(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["category"]), str(row["cell_type"]))
        groups.setdefault(key, []).append(row)
    return groups


def proximity_clusters(
    rows: list[dict[str, object]],
    bbox: tuple[float, float, float, float],
    threshold: float = 0.035,
) -> list[list[dict[str, object]]]:
    """Group identities that are indistinguishable at the panel's display scale."""
    west, east, south, north = bbox
    xspan = max(east - west, 1e-9)
    yspan = max(north - south, 1e-9)
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(rows)):
        left_x = (float(rows[left]["glon"]) - west) / xspan
        left_y = (float(rows[left]["glat"]) - south) / yspan
        for right in range(left + 1, len(rows)):
            right_x = (float(rows[right]["glon"]) - west) / xspan
            right_y = (float(rows[right]["glat"]) - south) / yspan
            if math.hypot(left_x - right_x, left_y - right_y) <= threshold:
                union(left, right)

    clusters: dict[int, list[dict[str, object]]] = {}
    for index, row in enumerate(rows):
        clusters.setdefault(find(index), []).append(row)
    return list(clusters.values())


def annotate_multiplicity(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    bbox: tuple[float, float, float, float],
) -> None:
    """Label visually overlapping identities without moving their coordinates."""
    # Dense cohorts already read unmistakably as multi-identity clouds, and
    # repeated badges would hide the infrastructure beneath them.
    if len(rows) > 30:
        return
    for cluster in proximity_clusters(rows, bbox):
        if len(cluster) < 2:
            continue
        x = sum(float(row["glon"]) for row in cluster) / len(cluster)
        y = sum(float(row["glat"]) for row in cluster) / len(cluster)
        ax.annotate(
            f"×{len(cluster)}",
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=5.5,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "edgecolor": "#ffd166", "linewidth": 0.7, "alpha": 0.88},
            zorder=9,
        )


def footer(rows: list[dict[str, object]]) -> str:
    counts = Counter((str(r["category"]), int(r["mcc"]), int(r["mnc"])) for r in rows)
    combined: dict[tuple[str, int], tuple[list[int], int]] = {}
    for (category, mcc, mnc), count in sorted(counts.items()):
        mncs, total = combined.get((category, mcc), ([], 0))
        combined[(category, mcc)] = ([*mncs, mnc], total + count)
    labels = []
    for (category, mcc), (mncs, count) in combined.items():
        mnc_label = str(mncs[0]) if len(mncs) == 1 else "{" + ",".join(map(str, mncs)) + "}"
        labels.append(f"{mcc:03d}/{mnc_label} {category.lower()} n={count}")
    networks = "; ".join(labels)
    return f"{networks} · {sum(int(r['obs']) for r in rows):,} obs"


def add_scale_bar(ax: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    west, east, south, north = bbox
    mid_lat = (south + north) / 2
    width_km = (east - west) * 111.32 * math.cos(math.radians(mid_lat))
    target = width_km * 0.22
    candidates = (0.2, 0.5, 1, 2, 5, 10, 20, 50)
    distance = max(value for value in candidates if value <= max(target, 0.2))
    lon_span = distance / (111.32 * math.cos(math.radians(mid_lat)))
    x1 = east - (east - west) * 0.055
    x0 = x1 - lon_span
    y = north - (north - south) * 0.065
    line = ax.plot([x0, x1], [y, y], color="white", linewidth=2.1, zorder=8)[0]
    line.set_path_effects([path_effects.Stroke(linewidth=3.6, foreground="black"), path_effects.Normal()])
    text = ax.text((x0 + x1) / 2, y - (north - south) * 0.018, f"{distance:g} km", color="white", fontsize=5.5, ha="center", va="top", zorder=8)
    text.set_path_effects([path_effects.Stroke(linewidth=1.8, foreground="black"), path_effects.Normal()])


def fit_bbox_to_panel(
    bbox: tuple[float, float, float, float],
    panel_ratio: float = 1.15,
) -> tuple[float, float, float, float]:
    """Expand a lon/lat bbox so every geographic panel occupies equal space."""
    west, east, south, north = bbox
    mid_lon = (west + east) / 2
    mid_lat = (south + north) / 2
    xspan = east - west
    yspan = north - south
    cos_lat = max(math.cos(math.radians(mid_lat)), 0.25)
    current_ratio = xspan * cos_lat / yspan
    if current_ratio < panel_ratio:
        xspan = panel_ratio * yspan / cos_lat
    else:
        yspan = xspan * cos_lat / panel_ratio
    return (
        mid_lon - xspan / 2,
        mid_lon + xspan / 2,
        mid_lat - yspan / 2,
        mid_lat + yspan / 2,
    )


def draw_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    bbox: tuple[float, float, float, float],
    title: str,
    osm_id: str | None,
    *,
    zoom: int,
    footer_text: str | None = None,
    title_fontsize: float = 7.15,
    footer_fontsize: float = 5.25,
) -> None:
    west, east, south, north = bbox
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_facecolor("#33443d")
    try:
        cx.add_basemap(
            ax,
            source=cx.providers.Esri.WorldImagery,
            crs="EPSG:4326",
            zoom=zoom,
            attribution=False,
            reset_extent=True,
            interpolation="bilinear",
        )
        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.PositronOnlyLabels,
            crs="EPSG:4326",
            zoom=min(zoom, 15),
            attribution=False,
            reset_extent=True,
            alpha=0.88,
        )
    except Exception:
        # The data and facility outline remain readable if a tile service is
        # temporarily unavailable; contextily also caches successful requests.
        pass

    draw_facility(ax, osm_geometry(osm_id))
    for (category, technology), group in grouped(rows).items():
        sizes = np.clip(np.sqrt([int(r["obs"]) for r in group]) * 6.0, 30, 78)
        ax.scatter(
            [float(r["glon"]) for r in group],
            [float(r["glat"]) for r in group],
            s=sizes,
            marker=TECH_MARKERS.get(technology, "o"),
            color=CATEGORY_COLORS[category],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.85,
            zorder=6,
        )
    annotate_multiplicity(ax, rows, bbox)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1 / max(math.cos(math.radians((south + north) / 2)), 0.25), adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#202020")
        spine.set_linewidth(0.65)
    ax.set_title(title, fontsize=title_fontsize, pad=3, fontweight="bold", linespacing=1.18)
    ax.text(
        0.012,
        0.014,
        footer_text or footer(rows),
        transform=ax.transAxes,
        fontsize=footer_fontsize,
        color="white",
        ha="left",
        va="bottom",
        bbox={"facecolor": "black", "edgecolor": "white", "linewidth": 0.25, "alpha": 0.68, "pad": 1.15},
        zorder=8,
    )
    add_scale_bar(ax, bbox)


def make_figure(output: Path, preview: Path | None) -> None:
    rows_by_category = {
        "Testing": load_rows("testing.csv"),
        "Unassigned": load_rows("unassigned.csv"),
        "Private": load_rows("private.csv"),
        "Cross-country": [
            *load_rows("fort_bragg_mcc553.csv", OUT_OF_COUNTRY_DATA),
            *load_rows("defense_additions.csv", OUT_OF_COUNTRY_DATA),
        ],
    }

    # bbox is (west, east, south, north); OSM IDs are authoritative mapped
    # facility/training-area polygons returned by Nominatim.  Extents are kept
    # tight around the observations to expose roads, buildings, and ranges.
    specs = [
        ("A. Khojaly · Military Base\nKhojaly District · Azerbaijan", [("Cross-country", 283, 4)], (46.788, 46.806, 39.884, 39.900), "W429004217", 15),
        ("B. Gozha · Training Ground\nGrodno Region · Belarus", [("Cross-country", 246, 1), ("Cross-country", 246, 2), ("Cross-country", 246, 8)], (23.900, 24.055, 53.830, 53.925), "W314965353", 13),
        ("C. Sanya · Heliport\nHainan · China", [("Testing", 1, 0)], (109.451, 109.473, 18.279, 18.298), "W469039710", 15),
        ("D. DGA Landes · Missile Test Range\nNouvelle-Aquitaine · France", [("Unassigned", 9, 9)], (-1.275, -1.210, 44.405, 44.457), "R3234655", 14),
        ("E. Hohenfels · Training Area\nBavaria · Germany", [("Testing", 1, 1)], (11.785, 11.925, 49.185, 49.285), "W229977366", 13),
        ("F. Naliya · Air Force Station\nGujarat · India", [("Unassigned", 123, 45)], (68.860, 68.923, 23.202, 23.244), "W138755258", 14),
        ("G. IRGC Command · Military Headquarters\nTehran Province · Iran", [("Cross-country", 400, 1)], (51.492, 51.519, 35.675, 35.698), "W506037515", 15),
        ("H. Hamamatsu · Air Base\nShizuoka · Japan", [("Private", 999, 1)], (137.684, 137.723, 34.739, 34.760), "W42722174", 15),
        ("I. Niamey · Air Base\nNiamey · Niger", [("Testing", 1, 1)], (2.075, 2.205, 13.475, 13.550), "W871876288", 13),
        ("J. Rena · Army Camp\nInnlandet · Norway", [("Testing", 1, 1)], (11.335, 11.515, 61.105, 61.270), "W962221904", 12),
        ("K. Ukrainka · Air Base\nAmur Oblast · Russia", [("Cross-country", 460, 0)], (128.414, 128.501, 51.140, 51.194), "W366468209", 14),
        ("L. Chkalovsky · Military Compound\nMoscow Oblast · Russia", [("Cross-country", 255, 3)], (38.132, 38.162, 55.881, 55.899), "W445080855", 15),
        ("M. Copehill Down · Training Area\nEngland · United Kingdom", [("Testing", 1, 1)], (-2.070, -1.890, 51.160, 51.245), "W258574467", 12),
        ("N. Porton Down · Defense Laboratory\nEngland · United Kingdom", [("Testing", 1, 1)], (-1.711, -1.692, 51.1295, 51.1402), "W28121567", 15),
        ("O. Berdyansk · Airfield\nZaporizhzhia Oblast · Ukraine", [("Cross-country", 250, 94), ("Cross-country", 250, 96), ("Cross-country", 250, 97)], (36.731, 36.795, 46.792, 46.837), "W506925303", 14),
        ("P. Fort Huachuca · Army Base\nArizona · United States", [("Testing", 1, 1), ("Unassigned", 111, 94)], (-110.455, -110.285, 31.345, 31.605), "R2262740", 12),
        ("Q. Camp Roberts · National Guard Training Site\nCalifornia · United States", [("Testing", 1, 1)], (-120.875, -120.665, 35.690, 35.785), "R317716", 12),
        ("R. China Lake · Naval Air Weapons Station\nCalifornia · United States", [("Testing", 1, 1)], (-117.620, -117.560, 35.605, 35.655), "R317710", 14),
        ("S. Eglin · Air Force Base\nFlorida · United States", [("Unassigned", 111, 111)], (-86.645, -86.565, 30.615, 30.685), "R533644", 14),
        ("T. Stockbridge · Experimental Facility\nNew York · United States", [("Testing", 1, 1)], (-75.663, -75.640, 43.0250, 43.0445), "W1289150743", 15),
        ("U. Fort Bragg · Army Base\nNorth Carolina · United States", [("Testing", 1, 1), ("Cross-country", 553, 96), ("Cross-country", 553, 97), ("Cross-country", 553, 987)], (-79.390, -79.210, 35.025, 35.175), "R176940", 13),
        ("V. Grand Forks · Air Force Base\nNorth Dakota · United States", [("Testing", 1, 10)], (-97.500, -97.300, 47.890, 48.065), "W556280798", 12),
        ("W. Fort Pickett · Maneuver Training Center\nVirginia · United States", [("Testing", 1, 1)], (-78.020, -77.835, 36.960, 37.125), "R4251895", 12),
        ("X. Quantico · Marine Corps Base\nVirginia · United States", [("Private", 999, 99)], (-77.580, -77.500, 38.545, 38.615), "R2586049", 14),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(6, 4, figsize=(13.8, 16.65))
    fig.subplots_adjust(left=0.025, right=0.992, top=0.950, bottom=0.040, wspace=0.11, hspace=0.29)
    fig.suptitle(
        "Anomalous MCC identities at defense laboratories, test ranges, and military bases",
        fontsize=13.3,
        fontweight="bold",
        y=0.985,
    )
    for ax, (title, networks, selection_bbox, osm_id, zoom) in zip(axes.flat, specs, strict=True):
        rows = select(rows_by_category, networks, selection_bbox)
        draw_panel(ax, rows, fit_bbox_to_panel(selection_bbox), title, osm_id, zoom=zoom)

    category_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="white", markersize=6, label=category)
        for category, color in CATEGORY_COLORS.items()
    ]
    tech_handles = [
        plt.Line2D([0], [0], marker=TECH_MARKERS[tech], linestyle="", color="#555555", markerfacecolor="#777777", markersize=5.5, label=label)
        for tech, label in (("gsm", "GSM/UMTS"), ("lte", "LTE"), ("nr", "NR"))
    ]
    facility_handle = plt.Line2D([0], [0], color="#ffd166", linewidth=2.0, label="Mapped facility boundary")
    fig.legend(
        handles=category_handles + tech_handles + [facility_handle],
        loc="lower left",
        bbox_to_anchor=(0.025, 0.006),
        ncol=8,
        frameon=False,
        fontsize=6.25,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    fig.text(
        0.99,
        0.010,
        "Imagery © Esri and contributors · boundaries/labels © OpenStreetMap contributors, © CARTO",
        ha="right",
        va="bottom",
        fontsize=5.55,
        color="#666666",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FIGS / "military_defense_mcc_maps.pdf")
    parser.add_argument("--preview", type=Path, default=FIGS / "military_defense_mcc_maps.png")
    args = parser.parse_args()
    make_figure(args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
