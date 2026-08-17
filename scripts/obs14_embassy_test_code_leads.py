#!/usr/bin/env python3
"""Generate the embassy-area test/unassigned MCC lead figure."""

from __future__ import annotations

import argparse
import json
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import seaborn as sns

from plot_helpers import COUNTRIES_GEOJSON, add_osm_basemap, _lonlat_to_tile, draw_geojson_layer
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"
EMB_POLY = DATA_ROOT / "enrichment" / "embassy_poly.jsonl"

COLORS = {
    "All embassy-area cells": "#8b8b8b",
    "Test/private/unassigned": "#b23a48",
    "Single-cell candidates": "#d08c2f",
    "Zero-hour candidates": "#2f6f9f",
}

MCC_COLORS = {
    "001 test": "#b23a48",
    "999 private": "#2f6f9f",
    "123/45 family": "#4f7f52",
    "Other unassigned": "#c9743a",
}

SITE_SHORT = {
    "Korean Cultural Center NY": "Korean Cultural Center, NY",
    "Honorair Consulaat van Bhutan": "Bhutan honorary consulate, The Hague",
    "Embassy of Australia": "Australia embassy, Harare",
    "Embassy of Belgium": "Belgium embassy, Makati",
    "Italian Vice-Consulate in Birmingham": "Italian consulate, Birmingham",
    "Embassy of Ecuador": "Ecuador embassy, Canberra",
    "韩国驻沈阳总领事馆": "Korea consulate, Shenyang",
    "Ambassade d'Angola - Ambassade van Angola": "Angola embassy, Uccle",
    "U.S. Embassy in Nassau": "U.S. Embassy, Nassau",
    "Embassy of the United Mexican States": "Mexican Embassy, Belmopan",
    "Indian Consulate": "Indian Consulate, Belmopan",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def mcc_class(mcc: int) -> str:
    if mcc == 1:
        return "001 test"
    if mcc == 999:
        return "999 private"
    if mcc == 123:
        return "123/45 family"
    return "Other unassigned"


def ascii_label(value: str) -> str:
    cleaned = str(value).encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or "non-ASCII embassy"


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


def load_embassy_polygons(names: list[str]) -> dict[str, list[list[tuple[float, float]]]]:
    wanted = set(names)
    rings = {name: [] for name in names}
    with open(EMB_POLY) as f:
        for line in f:
            row = json.loads(line)
            name = row.get("name", "")
            if name in wanted:
                rings[name].extend(walk_rings(row["poly"]))
    return rings


def ring_bounds(ring: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [lon for lon, _ in ring]
    ys = [lat for _, lat in ring]
    return (min(xs), max(xs), min(ys), max(ys))


def ring_intersects_bbox(
    ring: list[tuple[float, float]],
    bbox: tuple[float, float, float, float],
) -> bool:
    rxmin, rxmax, rymin, rymax = ring_bounds(ring)
    xmin, xmax, ymin, ymax = bbox
    return rxmax >= xmin and rxmin <= xmax and rymax >= ymin and rymin <= ymax


def local_bbox(row: pd.Series, rings: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    del rings
    lon = float(row["lon"])
    lat = float(row["lat"])
    # Use a fixed neighborhood window so a malformed or overly broad site
    # polygon cannot turn the one-off detail panel back into a global map.
    lon_span = 0.050
    lat_span = 0.034
    return (lon - lon_span / 2, lon + lon_span / 2, lat - lat_span / 2, lat + lat_span / 2)


def local_embassy_rings(
    rings: list[list[tuple[float, float]]],
    bbox: tuple[float, float, float, float],
) -> list[list[tuple[float, float]]]:
    local: list[list[tuple[float, float]]] = []
    for ring in rings:
        if len(ring) < 3 or not ring_intersects_bbox(ring, bbox):
            continue
        rxmin, rxmax, rymin, rymax = ring_bounds(ring)
        if (rxmax - rxmin) > 0.10 or (rymax - rymin) > 0.10:
            continue
        local.append(ring)
    return local


def osm_zoom(bbox: tuple[float, float, float, float]) -> int:
    for zoom in (14, 13, 12):
        xmin, xmax, ymin, ymax = bbox
        x0, y1 = _lonlat_to_tile(xmin, ymin, zoom)
        x1, y0 = _lonlat_to_tile(xmax, ymax, zoom)
        tile_count = (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
        if tile_count <= 24:
            return zoom
    return 12


def load_data() -> dict[str, pd.DataFrame]:
    condition = "mcc = 1 OR mcc = 999 OR (mcc BETWEEN 100 AND 199) OR (mcc BETWEEN 800 AND 899)"
    scope = ch_df(
        f"""
        SELECT label, cells, embassies
        FROM
        (
            SELECT 'All embassy-area cells' AS label, count() AS cells, uniqExact(emb) AS embassies
            FROM cell.emb_cells
            WHERE cid > 0
            UNION ALL
            SELECT 'Test/private/unassigned' AS label, count() AS cells, uniqExact(emb) AS embassies
            FROM cell.emb_cells
            WHERE cid > 0 AND ({condition})
            UNION ALL
            SELECT 'Single-cell candidates' AS label, count() AS cells, uniqExact(emb) AS embassies
            FROM
            (
                SELECT emb, mcc, mnc, cell_type, count() AS group_cells
                FROM cell.emb_cells
                WHERE cid > 0 AND ({condition})
                GROUP BY emb, mcc, mnc, cell_type
            )
            WHERE group_cells = 1
            UNION ALL
            SELECT 'Zero-hour candidates' AS label, count() AS cells, uniqExact(emb) AS embassies
            FROM
            (
                SELECT emb, mcc, mnc, cell_type, min(first_seen) AS first_seen, max(last_seen) AS last_seen
                FROM cell.emb_cells
                WHERE cid > 0 AND ({condition})
                GROUP BY emb, mcc, mnc, cell_type
            )
            WHERE dateDiff('hour', first_seen, last_seen) = 0
        )
        ORDER BY cells DESC
        """
    )

    mccs = ch_df(
        f"""
        SELECT mcc, count() AS cells, uniqExact(emb) AS embassies, uniqExact(country_iso) AS host_countries
        FROM cell.emb_cells
        WHERE cid > 0 AND ({condition})
        GROUP BY mcc
        ORDER BY cells DESC
        """
    )
    mccs["class"] = mccs["mcc"].map(lambda m: mcc_class(int(m)))

    events = ch_df(
        f"""
        SELECT *, dateDiff('hour', first_seen, last_seen) AS span_hrs
        FROM
        (
            SELECT
                emb,
                any(country_iso) AS host,
                any(city) AS city,
                any(emb_country) AS emb_country,
                mcc,
                mnc,
                cell_type,
                count() AS cells,
                min(first_seen) AS first_seen,
                max(last_seen) AS last_seen,
                uniqExact((mcc,mnc,lac,cid,cell_type)) AS uniq_cells,
                avg(glat) AS lat,
                avg(glon) AS lon
            FROM cell.emb_cells
            WHERE cid > 0 AND ({condition})
            GROUP BY emb, mcc, mnc, cell_type
        )
        ORDER BY cells ASC, span_hrs ASC, first_seen
        """
    )
    events["class"] = events["mcc"].map(lambda m: mcc_class(int(m)))
    events["site_label"] = events["emb"].map(SITE_SHORT).fillna(events["emb"]).map(ascii_label)
    events["first_seen"] = pd.to_datetime(events["first_seen"])
    events["last_seen"] = pd.to_datetime(events["last_seen"])
    candidates = (
        events[(events["span_hrs"] == 0) & (events["uniq_cells"].le(2))]
        .sort_values(["first_seen", "site_label", "mcc", "mnc"])
        .head(8)
        .copy()
    )
    candidates["map_title"] = candidates["site_label"].str.replace(", ", "\n", n=1, regex=False)

    return {"scope": scope, "mccs": mccs, "events": events, "candidates": candidates}


def draw_candidate_map(
    ax: plt.Axes,
    row: pd.Series,
    rings: list[list[tuple[float, float]]],
    panel_label: str,
) -> None:
    bbox = local_bbox(row, rings)
    site_rings = local_embassy_rings(rings, bbox)
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(
        ax,
        COUNTRIES_GEOJSON,
        bbox,
        facecolor="#f5f1e8",
        edgecolor="#8a8176",
        linewidth=0.35,
        alpha=1.0,
        zorder=0,
    )
    used_osm = add_osm_basemap(ax, bbox, zoom=osm_zoom(bbox), alpha=0.78, zorder=0.2)
    for ring in site_rings:
        ax.add_patch(
            Polygon(
                ring,
                closed=True,
                facecolor="#f4e7c9",
                edgecolor="#62594f",
                linewidth=0.7,
                alpha=0.68,
                zorder=2,
            )
        )
    klass = row["class"]
    ax.scatter(
        [row["lon"]],
        [row["lat"]],
        s=58,
        color=MCC_COLORS[klass],
        edgecolor="white",
        linewidth=0.7,
        alpha=0.95,
        zorder=5,
    )
    ax.text(
        0.02,
        0.98,
        f"{panel_label}. {row['map_title']}\nMCC {int(row['mcc'])}/{int(row['mnc'])} {str(row['cell_type']).upper()}\n{row['first_seen'].date()} | {int(row['uniq_cells'])} cell",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color="#262626",
        bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.3, "alpha": 0.86, "pad": 1.0},
        zorder=6,
    )
    ax.text(
        0.02,
        0.02,
        "OSM" if used_osm else "Natural Earth",
        transform=ax.transAxes,
        fontsize=5.5,
        color="#5b5b5b",
        zorder=7,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#7a7065")
        spine.set_linewidth(0.55)


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    scope = data["scope"].copy()
    mccs = data["mccs"].copy()
    events = data["events"].copy()
    candidates = data["candidates"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.04)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(15.8, 12.2), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=[0.88, 0.95, 1.0, 1.0])
    ax_scope = fig.add_subplot(gs[0, 0:2])
    ax_events = fig.add_subplot(gs[0, 2:4])
    ax_mccs = fig.add_subplot(gs[1, 0:2])
    ax_world = fig.add_subplot(gs[1, 2:4])
    map_axes = [fig.add_subplot(gs[2 + i // 4, i % 4]) for i in range(8)]
    fig.suptitle(
        "Embassy-area test MCCs surface surveillance leads, but not proof",
        fontsize=14,
        fontweight="bold",
    )

    ax = ax_scope
    order = ["All embassy-area cells", "Test/private/unassigned", "Single-cell candidates", "Zero-hour candidates"]
    scope["label"] = pd.Categorical(scope["label"], order, ordered=True)
    scope = scope.sort_values("label", ascending=False)
    ax.barh(scope["label"].astype(str), scope["cells"], color=[COLORS[x] for x in scope["label"].astype(str)])
    ax.set_xscale("log")
    ax.set_title("A. Candidate events are tiny relative to the embassy corpus")
    ax.set_xlabel("Cells or grouped events (log scale)")
    ax.set_ylabel("")
    for patch, cells, embassies in zip(ax.patches, scope["cells"], scope["embassies"], strict=False):
        ax.text(cells * 1.12, patch.get_y() + patch.get_height() / 2, f"{cells:,}; {embassies:,} sites", va="center", fontsize=8)
    ax.set_xlim(0.7, scope["cells"].max() * 5)

    ax = ax_events
    plot = candidates.sort_values("first_seen").copy()
    plot["date_num"] = plot["first_seen"].map(mdates.date2num)
    sns.scatterplot(
        data=plot,
        x="first_seen",
        y="site_label",
        hue="class",
        style="cell_type",
        s=90,
        palette=MCC_COLORS,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    for _, row in plot.iterrows():
        ax.text(
            row["first_seen"] + pd.Timedelta(days=6),
            row["site_label"],
            f"{int(row['mcc'])}/{int(row['mnc'])}",
            va="center",
            fontsize=6.8,
        )
    ax.set_title("B. Low-observation-count cells (NB: obs = crawl cadence)")
    ax.set_xlabel("First observed")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="", loc="lower right", frameon=True, fontsize=7)

    ax = ax_mccs
    plot_mccs = mccs.sort_values("cells", ascending=True)
    ax.barh(plot_mccs["mcc"].astype(str), plot_mccs["cells"], color=[MCC_COLORS[c] for c in plot_mccs["class"]])
    ax.set_title("C. MCC 001 dominates; the rest are a thin tail")
    ax.set_xlabel("Cells near diplomatic sites")
    ax.set_ylabel("MCC")
    for patch, cells, sites in zip(ax.patches, plot_mccs["cells"], plot_mccs["embassies"], strict=False):
        ax.text(cells + 0.45, patch.get_y() + patch.get_height() / 2, f"{cells}; {sites} sites", va="center", fontsize=7.5)
    ax.set_xlim(0, plot_mccs["cells"].max() * 1.45)

    ax = ax_world
    bbox = (-180, 180, -55, 75)
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(
        ax,
        COUNTRIES_GEOJSON,
        bbox,
        facecolor="#f5f1e8",
        edgecolor="#8a8176",
        linewidth=0.28,
        alpha=1.0,
        zorder=0,
    )
    map_events = events[events["cells"].le(2)].copy()
    ax.scatter(
        map_events["lon"],
        map_events["lat"],
        s=36 + map_events["span_hrs"].clip(upper=1200) / 18,
        color=[MCC_COLORS[c] for c in map_events["class"]],
        alpha=0.86,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    for label in [
        "Korean Cultural Center, NY",
        "Bhutan honorary consulate, The Hague",
        "Australia embassy, Harare",
        "Korea consulate, Shenyang",
        "U.S. Embassy, Nassau",
    ]:
        rows = map_events[map_events["site_label"] == label]
        if rows.empty:
            continue
        row = rows.iloc[0]
        ax.text(
            row["lon"],
            row["lat"],
            label.split(",")[0],
            fontsize=6.2,
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#9b948a", "linewidth": 0.25, "alpha": 0.74, "pad": 0.7},
            zorder=4,
        )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("auto")
    ax.set_title("D. Leads are geographically scattered, not one systematic campaign")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linewidth=0.28, color="white", alpha=0.75)

    polygons = load_embassy_polygons(candidates["emb"].tolist())
    for i, (_, row) in enumerate(candidates.reset_index(drop=True).iterrows()):
        draw_candidate_map(map_axes[i], row, polygons.get(row["emb"], []), chr(ord("E") + i))

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
        default=PLOTS / "obs14_embassy_test_code_leads.pdf",
        help="PDF output path.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
