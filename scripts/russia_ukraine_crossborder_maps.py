#!/usr/bin/env python3
"""Map Russian cells in Ukraine and Ukrainian cells in Russia, with movement paths."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_helpers import ADMIN1_GEOJSON, COUNTRIES_GEOJSON, draw_geojson_layer, setup_context_map
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"
OUTPUT_DPI = 900
PREVIEW_DPI = 450

RU_IN_UA = "#b23a48"
UA_IN_RU = "#2f6f9f"
RU_PATH = "#7f1d2d"
UA_PATH = "#174d73"
YEARLY_BBOX = (22.0, 42.5, 44.0, 53.5)


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_data() -> dict[str, pd.DataFrame]:
    ru_in_ua = ch_df(
        """
        SELECT
            glat AS lat,
            glon AS lon,
            mnc,
            region,
            city
        FROM cell.summary_full
        WHERE
            cid > 0
            AND mcc = 250
            AND country_iso = 'UA'
            AND glat BETWEEN 44.0 AND 53.0
            AND glon BETWEEN 22.0 AND 42.0
            AND glat != 0
            AND glon != 0
        """
    )

    ua_in_ru = ch_df(
        """
        SELECT
            glat AS lat,
            glon AS lon,
            mnc,
            region,
            city
        FROM cell.summary_full
        WHERE
            cid > 0
            AND mcc = 255
            AND country_iso = 'RU'
            AND glat BETWEEN 40.0 AND 62.0
            AND glon BETWEEN 27.0 AND 145.0
            AND glat != 0
            AND glon != 0
        """
    )

    ru_paths = ch_df(
        """
        WITH cross AS
        (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE cid > 0 AND mcc = 250 AND country_iso = 'UA'
        ),
        moving AS
        (
            SELECT
                g.mcc,
                g.mnc,
                g.lac,
                g.cid,
                g.cell_type,
                greatCircleDistance(min(lon), min(lat), max(lon), max(lat)) / 1000 AS km
            FROM cell.geos AS g
            INNER JOIN cross USING (mcc, mnc, lac, cid, cell_type)
            WHERE g.cid > 0 AND NOT (lat = 0 AND lon = 0)
            GROUP BY g.mcc, g.mnc, g.lac, g.cid, g.cell_type
            HAVING count() > 1 AND km > 10
            ORDER BY km DESC
            LIMIT 16
        )
        SELECT
            concat(toString(g.mcc), '/', toString(g.mnc), '/', toString(g.lac), '/', toString(g.cid)) AS tower_id,
            g.mcc,
            g.mnc,
            g.lac,
            g.cid,
            g.cell_type,
            m.km,
            g.lat,
            g.lon,
            g.timestamp
        FROM cell.geos AS g
        INNER JOIN moving AS m USING (mcc, mnc, lac, cid, cell_type)
        WHERE g.cid > 0 AND NOT (lat = 0 AND lon = 0)
        ORDER BY tower_id, timestamp
        """
    )

    ua_paths = ch_df(
        """
        WITH cross AS
        (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE cid > 0 AND mcc = 255 AND country_iso = 'RU'
        ),
        moving AS
        (
            SELECT
                g.mcc,
                g.mnc,
                g.lac,
                g.cid,
                g.cell_type,
                greatCircleDistance(min(lon), min(lat), max(lon), max(lat)) / 1000 AS km
            FROM cell.geos AS g
            INNER JOIN cross USING (mcc, mnc, lac, cid, cell_type)
            WHERE g.cid > 0 AND NOT (lat = 0 AND lon = 0)
            GROUP BY g.mcc, g.mnc, g.lac, g.cid, g.cell_type
            HAVING count() > 1 AND km > 0
            ORDER BY km DESC
            LIMIT 8
        )
        SELECT
            concat(toString(g.mcc), '/', toString(g.mnc), '/', toString(g.lac), '/', toString(g.cid)) AS tower_id,
            g.mcc,
            g.mnc,
            g.lac,
            g.cid,
            g.cell_type,
            m.km,
            g.lat,
            g.lon,
            g.timestamp
        FROM cell.geos AS g
        INNER JOIN moving AS m USING (mcc, mnc, lac, cid, cell_type)
        WHERE g.cid > 0 AND NOT (lat = 0 AND lon = 0)
        ORDER BY tower_id, timestamp
        """
    )

    yearly = ch_df(
        """
        WITH
        ru_cross AS
        (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE cid > 0 AND mcc = 250 AND country_iso = 'UA'
        ),
        ua_cross AS
        (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM cell.summary_full
            WHERE cid > 0 AND mcc = 255 AND country_iso = 'RU'
        )
        SELECT
            year,
            group_name,
            avg_lat AS lat,
            avg_lon AS lon,
            obs
        FROM
        (
            SELECT
                toYear(g.timestamp) AS year,
                'Russian MCC 250 in Ukraine' AS group_name,
                g.mcc, g.mnc, g.lac, g.cid, g.cell_type,
                avg(g.lat) AS avg_lat,
                avg(g.lon) AS avg_lon,
                count() AS obs
            FROM cell.geos AS g
            INNER JOIN ru_cross USING (mcc, mnc, lac, cid, cell_type)
            WHERE
                g.cid > 0
                AND g.lat BETWEEN 44.0 AND 53.5
                AND g.lon BETWEEN 22.0 AND 42.5
                AND NOT (g.lat = 0 AND g.lon = 0)
            GROUP BY year, g.mcc, g.mnc, g.lac, g.cid, g.cell_type
            UNION ALL
            SELECT
                toYear(g.timestamp) AS year,
                'Ukrainian MCC 255 in Russia' AS group_name,
                g.mcc, g.mnc, g.lac, g.cid, g.cell_type,
                avg(g.lat) AS avg_lat,
                avg(g.lon) AS avg_lon,
                count() AS obs
            FROM cell.geos AS g
            INNER JOIN ua_cross USING (mcc, mnc, lac, cid, cell_type)
            WHERE
                g.cid > 0
                AND g.lat BETWEEN 44.0 AND 53.5
                AND g.lon BETWEEN 22.0 AND 42.5
                AND NOT (g.lat = 0 AND g.lon = 0)
            GROUP BY year, g.mcc, g.mnc, g.lac, g.cid, g.cell_type
        )
        ORDER BY year, group_name
        """
    )

    for df in [ru_paths, ua_paths]:
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    return {
        "ru_in_ua": ru_in_ua,
        "ua_in_ru": ua_in_ru,
        "ru_paths": ru_paths,
        "ua_paths": ua_paths,
        "yearly": yearly,
    }


def draw_paths(ax: plt.Axes, paths: pd.DataFrame, color: str, *, min_label_km: float) -> None:
    for _, path in paths.groupby("tower_id", sort=False):
        path = path.sort_values("timestamp")
        if len(path) < 2:
            continue
        ax.plot(path["lon"], path["lat"], color=color, linewidth=0.85, alpha=0.62, zorder=5)
        ax.scatter(path["lon"].iloc[0], path["lat"].iloc[0], marker="o", s=16, color=color, edgecolor="white", linewidth=0.35, zorder=6)
        ax.scatter(path["lon"].iloc[-1], path["lat"].iloc[-1], marker=">", s=28, color=color, edgecolor="white", linewidth=0.35, zorder=7)
        km = float(path["km"].iloc[0])
        if km >= min_label_km:
            ax.text(
                path["lon"].iloc[-1],
                path["lat"].iloc[-1],
                f"{km:.0f} km",
                fontsize=6.5,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
                zorder=8,
            )


def setup_fill_map(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    *,
    countries: set[str],
    admin_names: set[str] | None = None,
    label_points: list[tuple[str, float, float]] | None = None,
) -> None:
    xmin, xmax, ymin, ymax = bbox
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(
        ax,
        COUNTRIES_GEOJSON,
        bbox,
        countries=countries,
        facecolor="#f5f1e8",
        edgecolor="#69635c",
        linewidth=0.50,
        alpha=1.0,
        zorder=0,
    )
    draw_geojson_layer(
        ax,
        ADMIN1_GEOJSON,
        bbox,
        countries=countries,
        facecolor="none",
        edgecolor="#a69d91",
        linewidth=0.30,
        alpha=0.85,
        zorder=0.8,
    )
    if admin_names:
        draw_geojson_layer(
            ax,
            ADMIN1_GEOJSON,
            bbox,
            admin_names=admin_names,
            facecolor="#ede2cf",
            edgecolor="#514b45",
            linewidth=0.65,
            alpha=0.70,
            zorder=0.9,
        )
    if label_points:
        for label, lon, lat in label_points:
            ax.text(
                lon,
                lat,
                label,
                fontsize=7,
                color="#3f3a35",
                ha="center",
                va="center",
                zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 1.0},
            )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("auto")
    ax.grid(True, linewidth=0.3, color="#ffffff", alpha=0.65)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    ru_in_ua = data["ru_in_ua"].copy()
    ua_in_ru = data["ua_in_ru"].copy()
    ru_paths = data["ru_paths"].copy()
    ua_paths = data["ua_paths"].copy()
    yearly = data["yearly"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(15.7, 8.6), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.12, 0.82], hspace=0.32, wspace=0.16)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.91, bottom=0.19)
    fig.suptitle("Cross-border PLMN geolocations and movement traces: Russia and Ukraine", fontsize=14, fontweight="bold")

    # Ukraine-focused panel.
    ax = fig.add_subplot(gs[0, 0:2])
    setup_context_map(
        ax,
        (22.0, 42.0, 44.0, 53.0),
        countries={"UA", "RU", "BY", "MD", "PL", "RO", "SK", "HU"},
        admin_names={"Crimea", "Donets'k", "Luhans'k", "Zaporizhzhya", "Kherson", "Kharkiv", "Sumy"},
        label_points=[
            ("Kyiv", 30.52, 50.45),
            ("Donetsk", 37.8, 48.0),
            ("Luhansk", 39.3, 48.57),
            ("Crimea", 34.1, 45.1),
            ("Russia", 39.5, 51.3),
        ],
    )
    ax.set_aspect("auto")
    ax.scatter(
        ru_in_ua["lon"],
        ru_in_ua["lat"],
        s=2.0,
        color=RU_IN_UA,
        alpha=0.24,
        linewidth=0,
        antialiaseds=False,
        rasterized=True,
        label=f"Russian MCC 250 in Ukraine ({len(ru_in_ua):,})",
        zorder=3,
    )
    ua_near_ukraine = ua_in_ru[
        ua_in_ru["lon"].between(22.0, 42.0)
        & ua_in_ru["lat"].between(44.0, 53.0)
    ]
    ax.scatter(
        ua_near_ukraine["lon"],
        ua_near_ukraine["lat"],
        s=10,
        color=UA_IN_RU,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
        zorder=4,
    )
    ax.set_aspect("auto")
    draw_paths(ax, ru_paths, RU_PATH, min_label_km=50)
    draw_paths(ax, ua_paths, UA_PATH, min_label_km=5)
    ax.set_title("A. Ukraine map: Russian-network cells geolocated in Ukraine")

    # Russia-focused panel.
    ax = fig.add_subplot(gs[0, 2:4])
    setup_fill_map(
        ax,
        (27.0, 145.0, 40.0, 62.0),
        countries={"RU", "UA", "BY", "KZ", "MN", "CN", "GE", "AZ", "FI", "EE", "LV", "LT", "PL"},
        admin_names={"Rostov", "Belgorod", "Bryansk", "Kursk", "Moskovskaya", "Tatarstan", "Pskov"},
        label_points=[
            ("Moscow", 37.62, 55.75),
            ("Rostov", 39.72, 47.24),
            ("Belgorod", 36.58, 50.60),
            ("Ukraine", 31.0, 49.0),
            ("Russia", 70.0, 56.0),
        ],
    )
    ax.scatter(
        ru_in_ua["lon"],
        ru_in_ua["lat"],
        s=1.4,
        color=RU_IN_UA,
        alpha=0.14,
        linewidth=0,
        antialiaseds=False,
        rasterized=True,
        zorder=3,
    )
    ax.scatter(
        ua_in_ru["lon"],
        ua_in_ru["lat"],
        s=18,
        color=UA_IN_RU,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
        label=f"Ukrainian MCC 255 in Russia ({len(ua_in_ru):,})",
        zorder=4,
    )
    draw_paths(ax, ru_paths, RU_PATH, min_label_km=50)
    draw_paths(ax, ua_paths, UA_PATH, min_label_km=5)
    ax.set_title("B. Russia map: Ukrainian-network cells geolocated in Russia")

    inset_bbox = (32.0, 41.0, 46.0, 53.5)
    iax = inset_axes(ax, width="38%", height="58%", loc="lower left", borderpad=1.2)
    setup_fill_map(
        iax,
        inset_bbox,
        countries={"RU", "UA", "BY"},
        admin_names={"Rostov", "Belgorod", "Bryansk", "Kursk", "Sumy", "Chernihiv"},
        label_points=[
            ("Kursk", 36.2, 51.7),
            ("Belgorod", 36.6, 50.6),
            ("Ukraine", 34.0, 49.2),
        ],
    )
    western_ua = ua_in_ru[
        ua_in_ru["lon"].between(inset_bbox[0], inset_bbox[1])
        & ua_in_ru["lat"].between(inset_bbox[2], inset_bbox[3])
    ]
    iax.scatter(
        western_ua["lon"],
        western_ua["lat"],
        s=20,
        color=UA_IN_RU,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.30,
        zorder=4,
    )
    draw_paths(iax, ua_paths, UA_PATH, min_label_km=5)
    iax.set_title("Western Russia detail", fontsize=7.5, pad=1)
    iax.tick_params(labelsize=6)
    iax.set_xlabel("")
    iax.set_ylabel("")

    # C-F. Yearly theater maps from raw observations.
    years = sorted(int(y) for y in yearly["year"].unique())
    for i, year in enumerate(years):
        ax = fig.add_subplot(gs[1, i])
        setup_fill_map(
            ax,
            YEARLY_BBOX,
            countries={"UA", "RU", "BY", "MD", "PL", "RO", "SK", "HU"},
            admin_names={"Crimea", "Donets'k", "Luhans'k", "Zaporizhzhya", "Kherson", "Kharkiv", "Sumy", "Rostov", "Belgorod", "Bryansk", "Kursk"},
            label_points=[
                ("Kyiv", 30.52, 50.45),
                ("Donbas", 38.1, 48.4),
                ("Crimea", 34.1, 45.1),
                ("Russia", 39.4, 51.6),
            ],
        )
        year_points = yearly[yearly["year"] == year]
        ru_year = year_points[year_points["group_name"] == "Russian MCC 250 in Ukraine"]
        ua_year = year_points[year_points["group_name"] == "Ukrainian MCC 255 in Russia"]
        if not ru_year.empty:
            ax.scatter(
                ru_year["lon"],
                ru_year["lat"],
                s=1.25,
                color=RU_IN_UA,
                alpha=0.30,
                linewidth=0,
                antialiaseds=False,
                rasterized=True,
                zorder=3,
            )
        if not ua_year.empty:
            ax.scatter(
                ua_year["lon"],
                ua_year["lat"],
                s=9,
                color=UA_IN_RU,
                alpha=0.84,
                edgecolor="white",
                linewidth=0.25,
                rasterized=True,
                zorder=4,
            )
        panel = chr(ord("C") + i)
        ax.set_title(f"{panel}. Seen in {year}\nRussian-in-UA {len(ru_year):,}; Ukrainian-in-RU {len(ua_year):,}", fontsize=8.5)
        ax.tick_params(labelsize=6.2)
        if i > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.set_xlabel("Longitude", fontsize=7)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"Russian MCC 250 in Ukraine ({len(ru_in_ua):,})", markerfacecolor=RU_IN_UA, markersize=6, alpha=0.65),
        plt.Line2D([0], [0], marker="o", color="w", label=f"Ukrainian MCC 255 in Russia ({len(ua_in_ru):,})", markerfacecolor=UA_IN_RU, markersize=6),
        plt.Line2D([0], [0], color=RU_PATH, label="Movement trace: Russian-ID cell", linewidth=1.2),
        plt.Line2D([0], [0], color=UA_PATH, label="Movement trace: Ukrainian-ID cell", linewidth=1.2),
        plt.Line2D([0], [0], marker=">", color="#555", label="Path endpoint", linestyle="None", markersize=6),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.072), ncols=3, frameon=True, fontsize=8)

    note = (
        f"Paths: top {ru_paths['tower_id'].nunique() if not ru_paths.empty else 0} moving Russian-ID cross-border cells "
        f"and top {ua_paths['tower_id'].nunique() if not ua_paths.empty else 0} moving Ukrainian-ID cross-border cells. "
        "Yearly panels use raw observations averaged per identity inside the Ukraine/western-Russia theater."
    )
    fig.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=OUTPUT_DPI, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=PREVIEW_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PLOTS / "russia_ukraine_crossborder_maps.pdf")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
