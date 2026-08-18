#!/usr/bin/env python3
"""Map the three foreign-PLMN identities at Golden Sun Sky/Jinshui."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

from military_defense_maps import CATEGORY_COLORS, draw_panel, fit_bbox_to_panel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity" / "golden_sun_sky_40406_cells.csv"
FIGS = ROOT / "paper" / "figs"

SITE_LON = 103.5544370
SITE_LAT = 10.5742531
SELECTION_BBOX = (103.5516, 103.5567, 10.5722, 10.5774)


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "category": "Cross-country",
                    "mcc": int(raw["mcc"]),
                    "mnc": int(raw["mnc"]),
                    "lac": int(raw["lac"]),
                    "cid": int(raw["cid"]),
                    "cell_type": raw["cell_type"],
                    "glat": float(raw["latitude"]),
                    "glon": float(raw["longitude"]),
                    "obs": int(raw["raw_observations"]),
                }
            )
    return rows


def make_figure(output: Path, preview: Path | None) -> None:
    rows = load_rows()
    bbox = fit_bbox_to_panel(SELECTION_BBOX)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 3.05))
    fig.subplots_adjust(left=0.025, right=0.992, top=0.92, bottom=0.12)

    draw_panel(
        ax,
        rows,
        bbox,
        "Golden Sun Sky/Jinshui · Documented scam complex\nSihanoukville · Cambodia",
        None,
        zoom=17,
        footer_text="Foreign PLMN: India 404/06 · n=3 · 272 obs",
        title_fontsize=6.1,
        footer_fontsize=4.4,
    )

    ax.scatter(
        [SITE_LON],
        [SITE_LAT],
        s=46,
        marker="o",
        facecolor="#ffd166",
        edgecolor="white",
        linewidth=1.0,
        zorder=8,
    )
    label = ax.annotate(
        "Golden Sun Sky",
        (SITE_LON, SITE_LAT),
        xytext=(5, -6),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=5.0,
        fontweight="bold",
        color="white",
        zorder=9,
    )
    label.set_path_effects(
        [path_effects.Stroke(linewidth=1.8, foreground="black"), path_effects.Normal()]
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=CATEGORY_COLORS["Cross-country"],
            markeredgecolor="white",
            markersize=5.5,
            label="Foreign GSM identity",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#ffd166",
            markeredgecolor="white",
            markersize=5.5,
            label="Mapped complex",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.025, 0.012),
        ncol=2,
        frameon=False,
        fontsize=4.5,
        handletextpad=0.3,
        columnspacing=0.65,
    )
    fig.text(
        0.99,
        0.014,
        "Imagery © Esri and contributors · labels © OpenStreetMap contributors, © CARTO",
        ha="right",
        va="bottom",
        fontsize=3.15,
        color="#666666",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGS / "golden_sun_sky_foreign_plmn.pdf",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=FIGS / "golden_sun_sky_foreign_plmn.png",
    )
    args = parser.parse_args()
    make_figure(args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
