#!/usr/bin/env python3
"""Generate a four-panel figure for Crimea operator substitution."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_helpers import setup_context_map
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"
STALE_CUTOFF = "2025-06-01"

COLORS = {
    "Russia MCC 250": "#b23a48",
    "Ukraine MCC 255": "#2f6f9f",
    "Other MCCs": "#8a8f98",
}

CITY_LABELS = {
    "Симферополь": "Simferopol",
    "Севастополь": "Sevastopol",
    "Керчь": "Kerch",
    "город Ялта": "Yalta",
    "Евпатория": "Yevpatoria",
    "Феодосия": "Feodosia",
    "Алушта": "Alushta",
    "Судак": "Sudak",
    "Саки": "Saky",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def load_data() -> dict[str, pd.DataFrame]:
    top_mccs = ch_df(
        """
        SELECT
            multiIf(
                mcc = 250, 'Russia MCC 250',
                mcc = 255, 'Ukraine MCC 255',
                concat('MCC ', toString(mcc))
            ) AS mcc_label,
            mcc,
            count() AS cells
        FROM cell.summary
        WHERE
            cid > 0
            AND glat BETWEEN 44.3 AND 46.2
            AND glon BETWEEN 32.5 AND 36.7
        GROUP BY mcc_label, mcc
        ORDER BY cells DESC
        LIMIT 8
        """
    )

    control = ch_df(
        f"""
        SELECT
            scope,
            multiIf(mcc = 250, 'Russia MCC 250', 'Ukraine MCC 255') AS mcc_label,
            mcc,
            count() AS cells,
            countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count() AS stale_frac
        FROM (
            SELECT
                multiIf(
                    glat BETWEEN 44.3 AND 46.2 AND glon BETWEEN 32.5 AND 36.7,
                    'Crimea bbox',
                    country_iso = 'UA'
                        AND NOT (glat BETWEEN 44.3 AND 46.2 AND glon BETWEEN 32.5 AND 36.7),
                    'Ukraine outside Crimea',
                    'Other'
                ) AS scope,
                mcc,
                last_seen
            FROM cell.summary_full
            WHERE cid > 0 AND mcc IN (250, 255)
        )
        WHERE scope != 'Other'
        GROUP BY scope, mcc_label, mcc
        ORDER BY scope, mcc
        """
    )

    city_literals = ", ".join(sql_string(c) for c in CITY_LABELS)
    cities = ch_df(
        f"""
        SELECT
            city,
            countIf(mcc = 250) AS russian,
            countIf(mcc = 255) AS ukrainian
        FROM cell.summary_full
        WHERE
            cid > 0
            AND mcc IN (250, 255)
            AND city IN ({city_literals})
            AND (
                region IN ('Crimea', 'Sevastopol')
                OR (glat BETWEEN 44.3 AND 46.2 AND glon BETWEEN 32.5 AND 36.7)
            )
        GROUP BY city
        ORDER BY russian DESC
        """
    )
    cities["city_label"] = cities["city"].map(CITY_LABELS)

    map_bins = ch_df(
        f"""
        SELECT
            round(glat, 2) AS lat,
            round(glon, 2) AS lon,
            count() AS cells,
            countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count() AS stale_frac
        FROM cell.summary
        WHERE
            cid > 0
            AND mcc = 250
            AND glat BETWEEN 44.3 AND 46.2
            AND glon BETWEEN 32.5 AND 36.7
            AND glat != 0
            AND glon != 0
        GROUP BY lat, lon
        HAVING cells >= 5
        """
    )

    ukrainian_point = ch_df(
        """
        SELECT glat AS lat, glon AS lon, first_seen, last_seen
        FROM cell.summary
        WHERE
            cid > 0
            AND mcc = 255
            AND glat BETWEEN 44.3 AND 46.2
            AND glon BETWEEN 32.5 AND 36.7
        """
    )

    return {
        "top_mccs": top_mccs,
        "control": control,
        "cities": cities,
        "map_bins": map_bins,
        "ukrainian_point": ukrainian_point,
    }


def annotate_barh(ax: plt.Axes, values: pd.Series, labels: pd.Series, pad: float) -> None:
    for patch, value, label in zip(ax.patches, values, labels, strict=False):
        ax.text(
            value + pad,
            patch.get_y() + patch.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )


def annotate_barh_log(ax: plt.Axes, values: pd.Series, labels: pd.Series) -> None:
    for patch, value, label in zip(ax.patches, values, labels, strict=False):
        ax.text(
            max(value * 1.08, value + 0.2),
            patch.get_y() + patch.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )


def make_figure(data: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    top_mccs = data["top_mccs"].copy()
    control = data["control"].copy()
    cities = data["cities"].copy()
    map_bins = data["map_bins"].copy()
    ukrainian_point = data["ukrainian_point"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.7), constrained_layout=True)
    fig.suptitle(
        "Crimea: cellular operator substitution after annexation",
        fontsize=14,
        fontweight="bold",
    )

    # A. Coordinate-based headline: top MCCs in the Crimea peninsula bbox.
    ax = axes[0, 0]
    top_mccs = top_mccs.sort_values("cells", ascending=True)
    bar_colors = [
        COLORS.get(label, COLORS["Other MCCs"])
        for label in top_mccs["mcc_label"]
    ]
    ax.barh(top_mccs["mcc_label"], top_mccs["cells"], color=bar_colors)
    annotate_barh_log(ax, top_mccs["cells"], top_mccs["cells"].map(lambda x: f"{x:,}"))
    ax.set_title("A. Top MCCs in tight Crimea coordinate bbox")
    ax.set_xlabel("Cells")
    ax.set_ylabel("")
    ax.set_xscale("log")
    ax.set_xlim(0.8, top_mccs["cells"].max() * 2.2)
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x):,}" if x < 1000 else f"{int(x / 1000):,}k")

    # B. Control: Ukrainian networks are present elsewhere in Ukraine.
    ax = axes[0, 1]
    scope_order = ["Crimea bbox", "Ukraine outside Crimea"]
    sns.barplot(
        data=control,
        x="scope",
        y="cells",
        hue="mcc_label",
        order=scope_order,
        palette=COLORS,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("B. Ukraine MCC 255 is healthy outside Crimea")
    ax.set_xlabel("")
    ax.set_ylabel("Cells (log scale)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x):,}" if x < 1000 else f"{int(x / 1000):,}k")
    ax.legend(title="", loc="lower left", frameon=True)
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height) or height <= 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height * 1.18,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    # C. Named Crimean cities: no Ukrainian MCC remains in the city-level view.
    ax = axes[1, 0]
    cities = cities.sort_values("russian", ascending=True)
    ax.barh(cities["city_label"], cities["russian"], color=COLORS["Russia MCC 250"], label="Russia MCC 250")
    ax.scatter(
        cities["ukrainian"],
        cities["city_label"],
        color=COLORS["Ukraine MCC 255"],
        marker="x",
        s=38,
        label="Ukraine MCC 255",
        zorder=3,
    )
    annotate_barh(ax, cities["russian"], cities["russian"].map(lambda x: f"{x:,}"), cities["russian"].max() * 0.012)
    ax.set_title("C. Named Crimean cities retain no Ukrainian cells")
    ax.set_xlabel("Cells")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x / 1000):,}k")
    ax.set_xlim(0, cities["russian"].max() * 1.18)
    ax.legend(title="", loc="lower right", frameon=True)

    # D. Spatial footprint across the peninsula.
    ax = axes[1, 1]
    setup_context_map(
        ax,
        (32.5, 36.7, 44.3, 46.2),
        countries={"UA", "RU"},
        admin_names={"Crimea", "Sevastopol"},
        label_points=[
            ("Simferopol", 34.10, 44.95),
            ("Sevastopol", 33.52, 44.62),
            ("Kerch", 36.47, 45.35),
            ("Black Sea", 33.0, 45.75),
            ("Sea of Azov", 35.7, 45.85),
        ],
    )
    sizes = (map_bins["cells"].clip(upper=500) ** 0.5) * 4
    sc = ax.scatter(
        map_bins["lon"],
        map_bins["lat"],
        c=map_bins["stale_frac"],
        s=sizes,
        cmap="rocket_r",
        vmin=0,
        vmax=1,
        alpha=0.72,
        linewidths=0,
    )
    if not ukrainian_point.empty:
        ax.scatter(
            ukrainian_point["lon"],
            ukrainian_point["lat"],
            color=COLORS["Ukraine MCC 255"],
            marker="*",
            s=130,
            edgecolor="white",
            linewidth=0.7,
            label="only Ukrainian MCC 255 cell",
            zorder=5,
        )
        ax.legend(title="", loc="lower left", frameon=True, fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(f"Russian cells stale before {STALE_CUTOFF}")
    ax.set_title("D. Peninsula-wide Russian footprint")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS / "obs03_crimea_operator_substitution.pdf",
        help="PDF output path.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
