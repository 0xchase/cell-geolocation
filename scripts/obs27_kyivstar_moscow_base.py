#!/usr/bin/env python3
"""Map the anomalous Kyivstar (UA MCC 255/03) cell that surfaces on the fence of
Russian military unit v/ch 51424 near Sverdlovsky, Moscow oblast.

Ukrainian-operator cells do appear inside Russia (153 of them in this dataset),
but almost all sit in the Kursk salient occupied during the Aug-2024 incursion.
This one is the outlier: ~488 km from the nearest point of Ukraine, deep in the
Russian interior, ~150 m off the eastern perimeter of a military unit adjacent
to the Chkalovsky military airfield, seen for a single observation on
2024-05-17 -- months before the Kursk incursion.

Left panel: national-scale locator showing how far the cell sits inside Russia.
Right panel: zoomed OpenStreetMap view of the cell against the v/ch 51424
footprint and the neighbouring Chkalovsky airfield beacons.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon

from plot_helpers import add_osm_basemap, setup_context_map
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
MILITARY_POLY = DATA_ROOT / "enrichment" / "military_poly.jsonl"
PLOTS = ROOT / "plots"
OUTPUT_DPI = 900
PREVIEW_DPI = 450

# The anomalous cell (Kyivstar, UA MCC 255 / MNC 03).
CELL = dict(mcc=255, mnc=3, lac=53213, cid=43289, cell_type="gsm", lat=55.891422, lon=38.1535)
BASE_NAME = "в/ч 51424"
NEIGHBOUR_BASES = (
    "Дальний приводной радиомаяк аэродрома Чкаловский",
    "Дальний приводной радиомаяк № 301",
)

KYIV = (50.4501, 30.5234)          # Kyivstar's home network
UA_BORDER_PT = (52.316, 33.819)    # nearest point of the Ukrainian border
KM_TO_UKRAINE = 488

CELL_COLOR = "#c02d3c"
BASE_COLOR = "#2f6f9f"
PATH_COLOR = "#7f1d2d"


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def great_circle_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi = math.radians(la2 - la1)
    dl = math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def load_base_rings(names: tuple[str, ...]) -> dict[str, list[list[list[float]]]]:
    """Return {name: [ring, ...]} for the requested military polygons."""
    wanted = set(names)
    out: dict[str, list[list[list[float]]]] = {}
    with open(MILITARY_POLY) as f:
        for line in f:
            obj = json.loads(line)
            name = obj.get("name", "")
            if name not in wanted:
                continue
            rings: list[list[list[float]]] = []

            def collect(node):
                if (
                    isinstance(node, list)
                    and node
                    and isinstance(node[0], list)
                    and node[0]
                    and isinstance(node[0][0], (int, float))
                ):
                    rings.append(node)
                elif isinstance(node, list):
                    for child in node:
                        collect(child)

            collect(obj["poly"])
            out[name] = rings
            if set(out) >= wanted:
                break
    return out


def nearest_ring_distance_m(pt: tuple[float, float], rings: list[list[list[float]]]) -> float:
    """Approximate min distance (metres) from a point to any polygon edge."""
    lat, lon = pt
    mlat = 111_320.0
    mlon = 111_320.0 * math.cos(math.radians(lat))
    px, py = lon * mlon, lat * mlat
    best = math.inf
    for ring in rings:
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
            ax, ay = x1 * mlon, y1 * mlat
            bx, by = x2 * mlon, y2 * mlat
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
            cx, cy = ax + t * dx, ay + t * dy
            best = min(best, math.hypot(px - cx, py - cy))
    return best


def load_data() -> dict:
    cell = CELL
    # Raw observations of the anomalous cell (there is only one, but query it so
    # the figure is regenerable straight from the database).
    obs = ch_df(
        f"""
        SELECT lat, lon, timestamp
        FROM cell.geos
        WHERE mcc = {cell['mcc']} AND mnc = {cell['mnc']}
          AND lac = {cell['lac']} AND cid = {cell['cid']}
        ORDER BY timestamp
        """
    )
    # Every Ukrainian-operator cell physically inside Russia, for context in the
    # locator panel (the Kursk salient cluster vs this Moscow-oblast outlier).
    ua_in_ru = ch_df(
        """
        SELECT glat AS lat, glon AS lon, state
        FROM cell.summary_full
        WHERE mcc = 255 AND country_osm = 'Россия'
          AND glat != 0 AND glon != 0
        """
    )
    rings = load_base_rings((BASE_NAME, *NEIGHBOUR_BASES))
    return dict(obs=obs, ua_in_ru=ua_in_ru, rings=rings)


def draw_locator(ax: plt.Axes, ua_in_ru: pd.DataFrame) -> None:
    bbox = (26.0, 44.0, 48.0, 58.5)  # lon/lat: Ukraine east to Moscow
    setup_context_map(
        ax,
        bbox,
        countries={"Ukraine", "UA", "Russia", "Russian Federation", "RU", "Belarus", "BY"},
    )
    # Kursk-salient cluster vs the Moscow outlier.
    if not ua_in_ru.empty:
        far = ua_in_ru[ua_in_ru["lat"] > 54]
        near = ua_in_ru[ua_in_ru["lat"] <= 54]
        ax.scatter(near["lon"], near["lat"], s=9, color="#8a8f97", alpha=0.55,
                   edgecolor="none", zorder=3, label="Other UA cells in Russia (mostly Kursk salient)")

    # Great-circle-ish link from Kyiv through the border to the cell.
    ax.plot([KYIV[1], UA_BORDER_PT[1], CELL["lon"]], [KYIV[0], UA_BORDER_PT[0], CELL["lat"]],
            color=PATH_COLOR, linewidth=1.1, linestyle="--", alpha=0.8, zorder=3.5)
    ax.scatter([KYIV[1]], [KYIV[0]], marker="*", s=90, color="#174d73",
               edgecolor="white", linewidth=0.4, zorder=5)
    ax.annotate("Kyiv\n(Kyivstar home network)", (KYIV[1], KYIV[0]),
                textcoords="offset points", xytext=(6, -18), fontsize=6.5, color="#174d73")
    ax.scatter([CELL["lon"]], [CELL["lat"]], marker="^", s=95, color=CELL_COLOR,
               edgecolor="white", linewidth=0.5, zorder=6)
    ax.annotate(f"Kyivstar cell\nat в/ч 51424\n≈{KM_TO_UKRAINE} km inside Russia",
                (CELL["lon"], CELL["lat"]), textcoords="offset points", xytext=(-4, 8),
                ha="right", fontsize=6.8, color=CELL_COLOR, fontweight="bold")
    ax.set_title("A  Where it surfaces: deep in the Russian interior", fontsize=9, loc="left")


def draw_zoom(ax: plt.Axes, rings: dict, fence_m: float) -> None:
    half_lon, half_lat = 0.028, 0.017
    bbox = (
        CELL["lon"] - half_lon, CELL["lon"] + half_lon,
        CELL["lat"] - half_lat, CELL["lat"] + half_lat,
    )
    drew_tiles = add_osm_basemap(ax, bbox, zoom=14, alpha=0.9, grayscale=True, zorder=0)
    if not drew_tiles:
        ax.set_facecolor("#eef1f4")

    for name, ring_set in rings.items():
        is_base = name == BASE_NAME
        for i, ring in enumerate(ring_set):
            patch = MplPolygon(
                [(lon, lat) for lon, lat in ring],
                closed=True,
                facecolor=(BASE_COLOR if is_base else "#6d6f74"),
                edgecolor=(BASE_COLOR if is_base else "#4a4c50"),
                linewidth=1.4 if is_base else 0.8,
                alpha=0.30 if is_base else 0.18,
                zorder=2,
            )
            ax.add_patch(patch)
        # Label at the polygon centroid of the first ring.
        r0 = ring_set[0]
        clat = sum(p[1] for p in r0) / len(r0)
        clon = sum(p[0] for p in r0) / len(r0)
        label = "в/ч 51424\n(military unit)" if is_base else "Chkalovsky airfield beacon"
        ax.annotate(label, (clon, clat), fontsize=6.2,
                    color=(BASE_COLOR if is_base else "#3f4145"),
                    ha="center", va="center", zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 1.0})

    ax.scatter([CELL["lon"]], [CELL["lat"]], marker="^", s=130, color=CELL_COLOR,
               edgecolor="white", linewidth=0.7, zorder=6)
    ax.annotate(
        f"Kyivstar 255/03\nLAC {CELL['lac']} / CID {CELL['cid']}\nGSM · seen once 2024-05-17\n≈{fence_m:.0f} m off the fence",
        (CELL["lon"], CELL["lat"]), textcoords="offset points", xytext=(10, 10),
        fontsize=6.6, color=CELL_COLOR, fontweight="bold", zorder=6,
        bbox={"facecolor": "white", "edgecolor": CELL_COLOR, "linewidth": 0.5, "alpha": 0.85, "pad": 2.0},
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", fontsize=7)
    ax.set_ylabel("Latitude", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("B  On the fence line of the base (Sverdlovsky, Moscow oblast)", fontsize=9, loc="left")


def make_figure(data: dict, output: Path, preview: Path | None) -> None:
    rings = data["rings"]
    base_rings = rings.get(BASE_NAME, [])
    fence_m = nearest_ring_distance_m((CELL["lat"], CELL["lon"]), base_rings) if base_rings else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    draw_locator(axes[0], data["ua_in_ru"])
    draw_zoom(axes[1], rings, fence_m)

    n_obs = 0 if data["obs"].empty else len(data["obs"])
    fig.suptitle(
        "A Ukrainian Kyivstar cell surfaces on a Russian military base near Moscow",
        fontsize=12, fontweight="bold", y=0.99,
    )
    note = (
        f"Kyivstar (Ukraine, MCC 255 / MNC 03) GSM cell LAC {CELL['lac']} / CID {CELL['cid']}, "
        f"geolocated at {CELL['lat']:.4f}°N, {CELL['lon']:.4f}°E from {n_obs} Apple crowdsourced observation(s) "
        f"on 2024-05-17 — ~{fence_m:.0f} m off the eastern perimeter of Russian military unit в/ч 51424, "
        f"~{KM_TO_UKRAINE} km from the nearest point of Ukraine and months before the Aug-2024 Kursk incursion. "
        "A single-observation foreign-PLMN cell deep in the adversary interior is the signature of captured/redeployed "
        "telecom kit, an IMSI-catcher cloning the operator, or a spoofing artifact — the data fixes where and what, not which."
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=7.2, wrap=True)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.13, wspace=0.12)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=OUTPUT_DPI, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "obs27_kyivstar_moscow_base.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
