#!/usr/bin/env python3
"""Generate a four-panel figure for test/unassigned MCCs at military sites."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from io import StringIO
from pathlib import Path

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helpers import COUNTRIES_GEOJSON, add_osm_basemap, draw_geojson_layer
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"
MIL_POLY = DATA_ROOT / "enrichment" / "military_poly.jsonl"

COLORS = {
    "ITU test (001)": "#b23a48",
    "Private (999)": "#2f6f9f",
    "India 123/45": "#4f7f52",
    "Other unassigned": "#c9743a",
}

SITE_SHORT = {
    "United Nations Buffer Zone": "UN Buffer Zone",
    "Dstl Porton Down": "Dstl Porton Down",
    "Copehill Down Training Area": "Copehill Down",
    "Stockbridge Experimental Facility": "Stockbridge Experimental",
    "Camp Roberts": "Camp Roberts",
    "Fort Huachuca": "Fort Huachuca",
    "Fort Carson": "Fort Carson",
    "Marine Corps Base Quantico": "MCB Quantico",
    "Naval Air Warfare Center China Lake": "China Lake",
    "Karwar Naval Base - INS Kadamba": "Karwar / INS Kadamba",
    "Indian Naval Academy Ezhimala": "Indian Naval Academy",
    "Ramgarh Test Range": "Ramgarh Test Range",
    "Leh Kushok Bakula Rimpochee Airport": "Leh airport",
    "Sheikh ul-Alam International Airport": "Srinagar airport",
    "Truppenübungsplatz Hohenfels": "Hohenfels",
    "陸上自衛隊 福島駐屯地": "JGSDF Fukushima",
    "Base Aérienne 101 de Niamey": "Niamey Air Base",
}

MAP_REGIONS = [
    ("United States", (-126, -64, 24, 50), ["Stockbridge Experimental", "Fort Huachuca", "China Lake"]),
    ("United Kingdom", (-8, 3, 49, 56), ["Dstl Porton Down", "Copehill Down"]),
    ("India", (67, 98, 6, 36), ["Karwar / INS Kadamba", "Srinagar airport", "Leh airport"]),
]

ZOOM_BASES = [
    "Dstl Porton Down",
    "Copehill Down Training Area",
    "Stockbridge Experimental Facility",
    "Camp Roberts",
    "Fort Huachuca",
    "Naval Air Warfare Center China Lake",
    "Truppenübungsplatz Hohenfels",
    "Fort Carson",
    "Marine Corps Base Quantico",
    "Karwar Naval Base - INS Kadamba",
    "Ramgarh Test Range",
    "Indian Naval Academy Ezhimala",
]


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def code_class(mcc: int) -> str:
    if mcc == 1:
        return "ITU test (001)"
    if mcc == 999:
        return "Private (999)"
    if mcc == 123:
        return "India 123/45"
    return "Other unassigned"


def ascii_label(label: str) -> str:
    cleaned = str(label).encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or "non-ASCII site"


def walk_rings(obj) -> list[list[tuple[float, float]]]:
    if (
        isinstance(obj, list)
        and obj
        and isinstance(obj[0], list)
        and len(obj[0]) == 2
        and all(isinstance(v, (int, float)) for v in obj[0])
    ):
        return [[(float(lon), float(lat)) for lon, lat in obj]]
    if isinstance(obj, list):
        rings: list[list[tuple[float, float]]] = []
        for item in obj:
            rings.extend(walk_rings(item))
        return rings
    return []


def load_military_polygons(base_names: list[str]) -> dict[str, list[list[tuple[float, float]]]]:
    wanted = set(base_names)
    rings = {name: [] for name in base_names}
    with open(MIL_POLY) as f:
        for line in f:
            row = json.loads(line)
            name = row.get("name", "")
            if name in wanted:
                rings[name].extend(walk_rings(row["poly"]))
    return rings


def zoom_bbox(points: pd.DataFrame, rings: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs = points["lon"].dropna().astype(float).tolist()
    ys = points["lat"].dropna().astype(float).tolist()
    for ring in rings:
        xs.extend(lon for lon, _ in ring)
        ys.extend(lat for _, lat in ring)
    if not xs or not ys:
        return (-180, 180, -80, 80)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = max(xmax - xmin, 0.035)
    height = max(ymax - ymin, 0.035)
    pad_x = max(width * 0.16, 0.012)
    pad_y = max(height * 0.16, 0.012)
    return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)


def osm_zoom_for_bbox(bbox: tuple[float, float, float, float]) -> int:
    span = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    if span > 1.0:
        return 9
    if span > 0.45:
        return 10
    if span > 0.18:
        return 11
    return 12


def fit_bbox_to_panel(
    bbox: tuple[float, float, float, float],
    panel_ratio: float,
) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = bbox
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    xspan = xmax - xmin
    yspan = ymax - ymin
    geographic_aspect = 1 / max(math.cos(math.radians(ymid)), 0.25)
    target_ratio = panel_ratio * geographic_aspect
    if xspan / yspan < target_ratio:
        xspan = yspan * target_ratio
    else:
        yspan = xspan / target_ratio
    return (xmid - xspan / 2, xmid + xspan / 2, ymid - yspan / 2, ymid + yspan / 2)


def load_data() -> dict[str, pd.DataFrame]:
    condition = "mcc = 1 OR mcc = 999 OR (mcc BETWEEN 100 AND 199) OR (mcc BETWEEN 800 AND 899)"
    counts = ch_df(
        f"""
        SELECT mcc, count() AS cells, uniqExact(base) AS bases, uniqExact(country_iso) AS countries
        FROM cell.mil_cells
        WHERE cid > 0 AND ({condition})
        GROUP BY mcc
        ORDER BY cells DESC
        """
    )
    counts["class"] = counts["mcc"].map(lambda m: code_class(int(m)))

    sites = ch_df(
        f"""
        SELECT
            base,
            any(country) AS country,
            mcc,
            count() AS cells,
            uniqExact((mnc, lac, cid, cell_type)) AS uniq_cells,
            min(first_seen) AS first_seen,
            max(last_seen) AS last_seen,
            avg(glat) AS lat,
            avg(glon) AS lon
        FROM cell.mil_cells
        WHERE cid > 0 AND ({condition})
        GROUP BY base, mcc
        ORDER BY cells DESC, base
        LIMIT 60
        """
    )
    sites["class"] = sites["mcc"].map(lambda m: code_class(int(m)))
    sites["site_label"] = sites["base"].map(SITE_SHORT).fillna(sites["base"]).map(ascii_label)
    sites["first_seen"] = pd.to_datetime(sites["first_seen"])
    sites["last_seen"] = pd.to_datetime(sites["last_seen"])
    sites["span_days"] = (sites["last_seen"] - sites["first_seen"]).dt.total_seconds() / 86400

    zoom_base_literals = ", ".join(sql_string(base) for base in ZOOM_BASES)
    facility_cells = ch_df(
        f"""
        SELECT
            base,
            any(country) AS country,
            mcc,
            mnc,
            lac,
            cid,
            cell_type,
            min(first_seen) AS first_seen,
            max(last_seen) AS last_seen,
            avg(glat) AS lat,
            avg(glon) AS lon
        FROM cell.mil_cells
        WHERE cid > 0 AND ({condition}) AND base IN ({zoom_base_literals})
        GROUP BY base, mcc, mnc, lac, cid, cell_type
        ORDER BY base, mcc, mnc, lac, cid
        """
    )
    facility_cells["class"] = facility_cells["mcc"].map(lambda m: code_class(int(m)))
    facility_cells["site_label"] = facility_cells["base"].map(SITE_SHORT).fillna(facility_cells["base"]).map(ascii_label)
    facility_cells["first_seen"] = pd.to_datetime(facility_cells["first_seen"])
    facility_cells["last_seen"] = pd.to_datetime(facility_cells["last_seen"])

    return {"counts": counts, "sites": sites, "facility_cells": facility_cells}


def draw_facility_zoom_panel(ax: plt.Axes, facility_cells: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_title("E. Local zooms show the anomalous MCCs inside named military facilities", pad=4)
    polygons = load_military_polygons(ZOOM_BASES)
    cols = 4
    rows = 3
    gap_x = 0.018
    gap_y = 0.058
    top_pad = 0.05
    w = (1 - gap_x * (cols - 1)) / cols
    h = (1 - top_pad - gap_y * (rows - 1)) / rows

    for i, base in enumerate(ZOOM_BASES):
        col = i % cols
        row = i // cols
        x = col * (w + gap_x)
        y = 1 - top_pad - h - row * (h + gap_y)
        iax = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(x, y, w, h),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        points = facility_cells[facility_cells["base"] == base].copy()
        rings = polygons.get(base, [])
        bbox = fit_bbox_to_panel(zoom_bbox(points, rings), 2.35)
        iax.set_facecolor("#dceaf2")
        add_osm_basemap(iax, bbox, zoom=osm_zoom_for_bbox(bbox), alpha=0.62, grayscale=True, zorder=0)

        for ring in rings:
            patch = Polygon(
                ring,
                closed=True,
                facecolor="#efe2c6",
                edgecolor="#5f554d",
                linewidth=0.65,
                alpha=0.56,
                zorder=1.5,
            )
            iax.add_patch(patch)

        if not points.empty:
            for klass, rows_for_class in points.groupby("class"):
                iax.scatter(
                    rows_for_class["lon"],
                    rows_for_class["lat"],
                    s=38,
                    color=COLORS[klass],
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=4,
                )
            mccs = "/".join(str(int(m)) for m in sorted(points["mcc"].unique()))
            subtitle = f"{len(points):,} cells; MCC {mccs}"
        else:
            subtitle = "no selected cells"

        label = SITE_SHORT.get(base, base)
        label = ascii_label(label)
        iax.text(
            0.02,
            0.98,
            f"{label}\n{subtitle}",
            transform=iax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#252525",
            bbox={"facecolor": "white", "edgecolor": "#9b948a", "linewidth": 0.25, "alpha": 0.86, "pad": 1.0},
            zorder=6,
        )
        iax.set_xlim(bbox[0], bbox[1])
        iax.set_ylim(bbox[2], bbox[3])
        mid_lat = (bbox[2] + bbox[3]) / 2
        iax.set_aspect(1 / max(math.cos(math.radians(mid_lat)), 0.25), adjustable="box")
        iax.set_xticks([])
        iax.set_yticks([])
        for spine in iax.spines.values():
            spine.set_color("#625b53")
            spine.set_linewidth(0.55)


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    counts = data["counts"].copy()
    sites = data["sites"].copy()
    facility_cells = data["facility_cells"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(13.4, 13.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.35])
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )
    zoom_ax = fig.add_subplot(gs[2, :])
    fig.suptitle(
        "Test and unassigned MCCs appear at military R&D, training, and private-network sites",
        fontsize=14,
        fontweight="bold",
    )

    # A. Code-family counts.
    ax = axes[0, 0]
    counts = counts.sort_values("cells", ascending=True)
    ax.barh(counts["class"], counts["cells"], color=[COLORS[c] for c in counts["class"]])
    for patch, cells, bases in zip(ax.patches, counts["cells"], counts["bases"], strict=False):
        ax.text(cells + 0.7, patch.get_y() + patch.get_height() / 2, f"{cells:,} cells; {bases} sites", va="center", fontsize=8)
    ax.set_title("A. Military polygons contain test/private MCC families")
    ax.set_xlabel("Cells inside military polygons")
    ax.set_ylabel("")
    ax.set_xlim(0, counts["cells"].max() * 1.35)

    # B. Top sites, with confounders visible.
    ax = axes[0, 1]
    top = sites.sort_values("cells", ascending=False).head(14).sort_values("cells", ascending=True)
    ax.barh(top["site_label"], top["cells"], color=[COLORS[c] for c in top["class"]])
    for patch, cells, mcc in zip(ax.patches, top["cells"], top["mcc"], strict=False):
        ax.text(cells + 0.2, patch.get_y() + patch.get_height() / 2, f"{cells}; MCC {mcc}", va="center", fontsize=7)
    ax.set_title("B. Top sites include both R&D sites and known confounders")
    ax.set_xlabel("Cells")
    ax.set_ylabel("")
    ax.set_xlim(0, top["cells"].max() * 1.35)

    # C. Regional context maps.
    ax = axes[1, 0]
    ax.axis("off")
    ax.set_title("C. Labeled sites sit in recognizable military geographies", pad=2)
    positions = [(0.00, 0.08, 0.48, 0.82), (0.52, 0.52, 0.46, 0.38), (0.52, 0.08, 0.46, 0.38)]
    for (region_title, bbox, labels), pos in zip(MAP_REGIONS, positions, strict=True):
        iax = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=pos,
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        iax.set_facecolor("#dceaf2")
        draw_geojson_layer(
            iax,
            COUNTRIES_GEOJSON,
            bbox,
            facecolor="#f5f1e8",
            edgecolor="#8a8176",
            linewidth=0.45,
            alpha=1.0,
            zorder=0,
        )
        rows = sites[
            sites["lon"].between(bbox[0], bbox[1])
            & sites["lat"].between(bbox[2], bbox[3])
        ]
        if not rows.empty:
            iax.scatter(
                rows["lon"],
                rows["lat"],
                s=24 + rows["cells"] * 10,
                color=[COLORS[c] for c in rows["class"]],
                alpha=0.84,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        for label in labels:
            label_rows = rows[rows["site_label"] == label]
            if not label_rows.empty:
                r = label_rows.iloc[0]
                iax.text(
                    r["lon"],
                    r["lat"],
                    label,
                    fontsize=6.5,
                    ha="center",
                    va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "#9b948a", "linewidth": 0.25, "alpha": 0.78, "pad": 0.8},
                    zorder=4,
                )
        iax.set_xlim(bbox[0], bbox[1])
        iax.set_ylim(bbox[2], bbox[3])
        iax.set_aspect("equal", adjustable="box")
        iax.set_xticks([])
        iax.set_yticks([])
        iax.set_title(region_title, fontsize=8, pad=1.5)
        for spine in iax.spines.values():
            spine.set_color("#8a8176")
            spine.set_linewidth(0.55)

    # D. Ephemerality and recurrence.
    ax = axes[1, 1]
    plot_sites = sites[sites["cells"] <= 6].copy()
    sns.scatterplot(
        data=plot_sites,
        x="cells",
        y="span_days",
        hue="class",
        size="uniq_cells",
        sizes=(30, 180),
        palette=COLORS,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("D. Sightings are sparse per site but persist for months")
    ax.set_xlabel("Cells at site for code family")
    ax.set_ylabel("First-to-last span (days; symlog)")
    ax.legend(title="", loc="upper left", frameon=True, fontsize=8)

    draw_facility_zoom_panel(zoom_ax, facility_cells)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS / "obs13_military_test_mccs.pdf",
        help="PDF output path.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
