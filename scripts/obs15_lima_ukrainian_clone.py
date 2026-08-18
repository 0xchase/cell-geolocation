#!/usr/bin/env python3
"""Generate a four-panel figure for Ukrainian cell identities replayed in Lima."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helpers import add_osm_basemap, draw_geojson_layer
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"
COUNTRIES_GEOJSON = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"

# Home-network extent used throughout (matches the 'Ukraine home' classifier).
UA_BBOX = (22.0, 41.0, 44.0, 53.0)  # xmin, xmax, ymin, ymax

OPERATOR = {1: "Vodafone UA", 6: "lifecell", 7: "3Mob/other", 3: "Kyivstar/other"}
PALETTE = {
    "Vodafone UA": "#b23a48",
    "lifecell": "#2f6f9f",
    "3Mob/other": "#c9743a",
    "Kyivstar/other": "#4f7f52",
    "MNC 702": "#8e6aa7",
    "MNC 707": "#7d7d7d",
}


def setup_basic_map(ax: plt.Axes, bbox: tuple[float, float, float, float]) -> None:
    """Draw geographic context from the reference data checked into this project."""
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(
        ax, COUNTRIES_GEOJSON, bbox,
        facecolor="#f5f1e8", edgecolor="#8a8176", linewidth=0.35, zorder=0,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_data() -> dict[str, pd.DataFrame]:
    americas = ch_df(
        """
        SELECT round(lat, 0) AS lat_bin, round(lon, 0) AS lon_bin,
               count() AS obs, uniqExact((mnc,lac,cid,cell_type)) AS cells
        FROM cell.geos
        WHERE mcc = 255 AND cid > 0 AND lon BETWEEN -120 AND -30 AND lat BETWEEN -56 AND 50
        GROUP BY lat_bin, lon_bin
        ORDER BY obs DESC
        LIMIT 20
        """
    )
    lima = ch_df(
        """
        SELECT mnc, lac, cid, cell_type, lat, lon, timestamp
        FROM cell.geos
        WHERE mcc = 255 AND cid > 0
          AND lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11
        ORDER BY timestamp
        """
    )
    clone_counts = ch_df(
        """
        WITH lima AS
        (
            SELECT DISTINCT mnc, lac, cid, cell_type
            FROM cell.geos
            WHERE mcc = 255 AND cid > 0
              AND lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11
        )
        SELECT
            multiIf(lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11, 'Lima replay',
                    lon BETWEEN 22 AND 41 AND lat BETWEEN 44 AND 53, 'Ukraine home',
                    'Elsewhere') AS location,
            count() AS obs,
            uniqExact((mnc,lac,cid,cell_type)) AS identities
        FROM cell.geos
        WHERE mcc = 255 AND cid > 0 AND (mnc, lac, cid, cell_type) IN lima
        GROUP BY location
        ORDER BY obs DESC
        """
    )
    timeline = ch_df(
        """
        WITH lima AS
        (
            SELECT DISTINCT mnc, lac, cid, cell_type
            FROM cell.geos
            WHERE mcc = 255 AND cid > 0
              AND lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11
        )
        -- Aggregated to monthly counts. The corrected table holds ~375k rows for
        -- this identity set (vs 135 in the deduplicated snapshot), so plotting raw
        -- observations is neither readable nor renderable. Monthly counts also
        -- expose the replay's onset, which the old data could not show.
        SELECT toStartOfMonth(timestamp) AS month,
               multiIf(lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11, 'Lima replay',
                       lon BETWEEN 22 AND 41 AND lat BETWEEN 44 AND 53, 'Ukraine home',
                       'Elsewhere') AS location,
               count() AS obs,
               uniqExact((lac, cid, cell_type)) AS identities
        FROM cell.geos
        WHERE mcc = 255 AND cid > 0 AND (mnc, lac, cid, cell_type) IN lima
        GROUP BY month, location
        ORDER BY month, location
        """
    )
    # Panel E: where the Lima identity set reports *at home*. One point per
    # identity (not per observation) — there are ~372k home observations, which
    # would be unreadable and would bloat the vector output.
    ua_home_pts = ch_df(
        """
        WITH lima AS
        (
            SELECT DISTINCT mnc, lac, cid, cell_type
            FROM cell.geos
            WHERE mcc = 255 AND cid > 0
              AND lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11
        )
        -- Aliases must not shadow the source columns used in WHERE, hence the
        -- avg_ prefix, renamed to lat/lon after the fact.
        SELECT mnc, lac, cid, cell_type,
               avg(lat) AS avg_lat, avg(lon) AS avg_lon, count() AS obs
        FROM cell.geos
        WHERE mcc = 255 AND cid > 0 AND (mnc, lac, cid, cell_type) IN lima
          AND lon BETWEEN 22 AND 41 AND lat BETWEEN 44 AND 53
        GROUP BY mnc, lac, cid, cell_type
        """
    ).rename(columns={"avg_lat": "lat", "avg_lon": "lon"}
    )
    paired = ch_df(
        """
        WITH lima AS
        (
            SELECT DISTINCT mnc, lac, cid, cell_type
            FROM cell.geos
            WHERE mcc = 255 AND cid > 0
              AND lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11
        )
        SELECT mnc, lac, cid, cell_type,
               countIf(lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11) AS lima_obs,
               countIf(lon BETWEEN 22 AND 41 AND lat BETWEEN 44 AND 53) AS ukraine_obs
        FROM cell.geos
        WHERE mcc = 255 AND cid > 0 AND (mnc, lac, cid, cell_type) IN lima
        GROUP BY mnc, lac, cid, cell_type
        HAVING lima_obs > 0 AND ukraine_obs > 0
        ORDER BY ukraine_obs DESC, lima_obs DESC
        """
    )
    # Full home-network footprint: every MCC 255 report inside Ukraine, binned to
    # a 0.1-degree grid, so panel E can show the real countrywide distribution as
    # a density backdrop for comparison with the collapsed Lima point cloud.
    ukraine_all = ch_df(
        f"""
        SELECT round(lat, 1) AS lat_bin, round(lon, 1) AS lon_bin, count() AS obs
        FROM cell.geos
        WHERE mcc = 255 AND cid > 0
          AND lon BETWEEN {UA_BBOX[0]} AND {UA_BBOX[1]}
          AND lat BETWEEN {UA_BBOX[2]} AND {UA_BBOX[3]}
          AND NOT (lat = 0 AND lon = 0)
        GROUP BY lat_bin, lon_bin
        ORDER BY obs DESC
        """
    )

    for df in [lima, timeline, paired, ua_home_pts]:
        if "mnc" in df:
            df["operator"] = df["mnc"].map(OPERATOR).fillna(df["mnc"].map(lambda x: f"MNC {x}"))
    lima["timestamp"] = pd.to_datetime(lima["timestamp"])
    # `timeline` is now aggregated to months server-side, so it carries `month`
    # rather than per-observation timestamps.
    timeline["month"] = pd.to_datetime(timeline["month"])

    return {
        "americas": americas,
        "lima": lima,
        "clone_counts": clone_counts,
        "timeline": timeline,
        "paired": paired,
        "ua_home_pts": ua_home_pts,
        "ukraine_all": ukraine_all,
    }


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    americas = data["americas"].copy()
    lima = data["lima"].copy()
    clone_counts = data["clone_counts"].copy()
    timeline = data["timeline"].copy()
    paired = data["paired"].copy()
    ua_home_pts = data["ua_home_pts"].copy()
    ukraine_all = data["ukraine_all"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.04)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(12.0, 13.4), constrained_layout=True)
    axd = fig.subplot_mosaic(
        [
            ["A", "A"],
            ["B", "E"],
            ["C", "D"],
        ]
    )
    fig.suptitle("Ukrainian cell identities are replayed in a single Lima neighborhood", fontsize=14, fontweight="bold")

    ax = axd["A"]
    bbox = (-120, -30, -56, 50)
    setup_basic_map(ax, bbox)
    ax.scatter(
        americas["lon_bin"],
        americas["lat_bin"],
        s=18 + americas["cells"].clip(upper=105) * 2.2,
        color="#9c3d46",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    if not americas.empty:
        top = americas.iloc[0]
        ax.text(top["lon_bin"], top["lat_bin"] + 5, f"Lima\n{top['cells']:,} identities", ha="center", fontsize=8, fontweight="bold")
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("A. Americas-wide scan finds one Ukrainian hotspot")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax = axd["B"]
    lima_bbox = (-77.0575, -77.0425, -12.0465, -12.0330)
    used_tiles = add_osm_basemap(
        ax, lima_bbox, zoom=15, alpha=1.0, grayscale=False, source="carto_voyager"
    )
    if not used_tiles:
        setup_basic_map(ax, lima_bbox)
    sns.scatterplot(
        data=lima,
        x="lon",
        y="lat",
        hue="operator",
        palette=PALETTE,
        s=35,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.45,
        ax=ax,
        zorder=4,
    )
    center_lon, center_lat = lima["lon"].mean(), lima["lat"].mean()
    ax.scatter([center_lon], [center_lat], marker="x", s=80, color="black", linewidth=1.4, zorder=5)
    ax.text(center_lon, center_lat + 0.0018, "Jiron Mancora cluster", ha="center", fontsize=7.4, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0}, zorder=6)
    # Derived from the data rather than hardcoded: the old title's "109" was the
    # count from the deduplicated snapshot.
    ax.set_title(f"B. {len(lima):,} reports collapse into one Lima neighborhood cluster")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="", loc="lower right", frameon=True, fontsize=8)
    ax.text(0.02, 0.02, "Map data (c) OpenStreetMap contributors", transform=ax.transAxes, fontsize=5.8, color="#555", zorder=7)

    ax = axd["E"]
    setup_basic_map(ax, UA_BBOX)
    # Backdrop: real countrywide MCC 255 report density (0.1-deg bins, log-scaled).
    if not ukraine_all.empty:
        weights = np.log10(ukraine_all["obs"].clip(lower=1) + 1)
        ax.scatter(
            ukraine_all["lon_bin"],
            ukraine_all["lat_bin"],
            s=4 + weights * 6,
            c="#6f6f6f",
            alpha=0.28,
            edgecolor="none",
            zorder=2.5,
        )
    # Overlay: the *same* Lima replay identities, plotted where they report in
    # Ukraine — spread across the whole country rather than a single point.
    ua_home = ua_home_pts
    if not ua_home.empty:
        sns.scatterplot(
            data=ua_home,
            x="lon",
            y="lat",
            hue="operator",
            palette=PALETTE,
            s=42,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.45,
            ax=ax,
            legend=False,
            zorder=4,
        )
    ax.set_xlim(UA_BBOX[0], UA_BBOX[1])
    ax.set_ylim(UA_BBOX[2], UA_BBOX[3])
    ax.set_title("E. In Ukraine, the same identities spread across the whole country")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.text(
        0.98,
        0.03,
        f"grey: all MCC 255 report density\ncolored: {len(ua_home)} Lima-clone reports at home",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "linewidth": 0.4, "alpha": 0.8, "pad": 2},
        zorder=6,
    )

    ax = axd["C"]
    location_order = ["Lima replay", "Ukraine home", "Elsewhere"]
    clone_counts["location"] = pd.Categorical(clone_counts["location"], location_order, ordered=True)
    clone_counts = clone_counts.sort_values("location", ascending=False)
    # Colour by label, not by row order, so panels C and D agree.
    loc_palette = {"Lima replay": "#b23a48", "Ukraine home": "#2f6f9f", "Elsewhere": "#8b8b8b"}
    ax.barh(
        clone_counts["location"].astype(str),
        clone_counts["obs"],
        color=[loc_palette.get(str(l), "#8b8b8b") for l in clone_counts["location"]],
    )
    for patch, obs, ids in zip(ax.patches, clone_counts["obs"], clone_counts["identities"], strict=False):
        ax.text(obs * 1.15, patch.get_y() + patch.get_height() / 2, f"{obs:,} obs; {ids:,} IDs", va="center", fontsize=8)
    # Log axis: home observations now outnumber Lima ones ~127:1, so a linear
    # scale renders the Lima bar invisible. (In the deduplicated snapshot the
    # ratio looked like 1:4 the other way, because dedup kept only the most
    # recent row per cell and the Lima reports were the more recent ones.)
    ax.set_xscale("log")
    ax.set_title("C. Home observations outnumber the Lima replay ~127:1")
    ax.set_xlabel("Observations for the Lima identity set (log)")
    ax.set_ylabel("")
    ax.set_xlim(1, clone_counts["obs"].max() * 4)

    ax = axd["D"]
    timeline["month"] = pd.to_datetime(timeline["month"])
    loc_colors = {"Ukraine home": "#2f6f9f", "Lima replay": "#b23a48", "Elsewhere": "#8b8b8b"}
    for location, grp in timeline.groupby("location"):
        grp = grp.sort_values("month")
        ax.plot(
            grp["month"], grp["identities"],
            marker="o", markersize=3.5, linewidth=1.6,
            color=loc_colors.get(str(location), "#8b8b8b"), label=str(location),
        )
    # The replay is absent for the first 19 months of the record and then ramps,
    # so a log axis is needed to show home and Lima on the same panel.
    ax.set_yscale("log")
    ax.set_ylabel("Distinct identities seen (log)")
    ax.set_title("D. The Lima replay switches on in mid-2025, then ramps")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="", loc="lower left", frameon=True, fontsize=8)
    onset = timeline[(timeline["location"] == "Lima replay") & (timeline["identities"] >= 50)]
    if not onset.empty:
        first = onset["month"].min()
        ax.axvline(first, color="#b23a48", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"ramp begins\n{first:%b %Y}",
            xy=(first, ax.get_ylim()[1] * 0.35),
            fontsize=7.5, color="#b23a48", ha="right", va="top",
        )
    if not paired.empty:
        ax.text(
            0.98,
            0.95,
            f"{len(paired)} identities seen in both places",
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
    parser.add_argument("--output", type=Path, default=PLOTS / "obs15_lima_ukrainian_clone.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    data = load_data()
    exports = {
        "lima_replay_observations.csv": data["lima"],
        "lima_replay_location_counts.csv": data["clone_counts"],
        "lima_replay_monthly_locations.csv": data["timeline"],
        "lima_replay_paired_identities.csv": data["paired"],
        "lima_replay_home_identities.csv": data["ua_home_pts"],
        "lima_ukraine_reference_density.csv": data["ukraine_all"],
    }
    data_output = ROOT / "data" / "spoofing"
    data_output.mkdir(parents=True, exist_ok=True)
    for filename, frame in exports.items():
        frame.to_csv(data_output / filename, index=False, date_format="%Y-%m-%d %H:%M:%S")
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
