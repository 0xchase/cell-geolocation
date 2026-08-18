#!/usr/bin/env python3
"""Render twelve audited geopolitical or institutional defense-site cases."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

from military_defense_maps import (
    CATEGORY_COLORS,
    TECH_MARKERS,
    draw_panel,
    fit_bbox_to_panel,
    load_rows,
    select,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "out-of-country"
FIGS = ROOT / "paper" / "figs"

MCC_COUNTRIES = {
    230: "Czech Republic",
    246: "Lithuania",
    250: "Russia",
    255: "Ukraine",
    280: "Cyprus",
    283: "Armenia",
    284: "Bulgaria",
    286: "Türkiye",
    294: "North Macedonia",
    363: "Aruba",
    400: "Azerbaijan",
    416: "Jordan",
    420: "Saudi Arabia",
    424: "UAE",
    426: "Bahrain",
    428: "Mongolia",
    460: "China",
    525: "Singapore",
    553: "Tuvalu",
    602: "Egypt",
    606: "Libya",
}


def compact_footer(rows: list[dict[str, object]]) -> str:
    categories = {str(row["category"]) for row in rows}
    mccs = sorted({int(row["mcc"]) for row in rows})
    if categories != {"Cross-country"}:
        networks = sorted({(int(row["mcc"]), int(row["mnc"])) for row in rows})
        label = ", ".join(f"{mcc:03d}/{mnc}" for mcc, mnc in networks)
        category = "/".join(sorted(categories))
        return f"{category} MCC: {label} · n={len(rows):,} · {sum(int(row['obs']) for row in rows):,} obs"
    if len(mccs) == 1:
        source = f"Foreign MCC: {MCC_COUNTRIES[mccs[0]]} ({mccs[0]:03d})"
        return f"{source} · n={len(rows):,} · {sum(int(row['obs']) for row in rows):,} obs"
    else:
        countries = ", ".join(f"{MCC_COUNTRIES[mcc]} ({mcc:03d})" for mcc in mccs)
        source = "Foreign MCCs: " + countries
        wrapped = "\n".join(textwrap.wrap(source, width=54, subsequent_indent="  "))
        return f"{wrapped}\nn={len(rows):,} · {sum(int(row['obs']) for row in rows):,} obs"


def make_figure(output: Path, preview: Path | None) -> None:
    rows_by_category = {
        "Cross-country": load_rows("geopolitical_defense_cases.csv", DATA),
        "Testing": load_rows("testing.csv"),
        "Unassigned": load_rows("unassigned.csv"),
    }

    # Ordered by observed country. Bboxes remain close to the relevant
    # infrastructure while retaining every selected identity at each site.
    specs = [
        ("A. Khojaly · Military Base\nKhojaly District · Azerbaijan", [("Cross-country", 283, 4)], (46.788, 46.806, 39.884, 39.900), "W429004217", 15),
        ("B. DGA Landes · Missile Test Range\nNouvelle-Aquitaine · France", [("Unassigned", 9, 9)], (-1.275, -1.210, 44.405, 44.457), "R3234655", 14),
        ("C. IRGC Command · Military Headquarters\nTehran Province · Iran", [("Cross-country", 400, 1)], (51.492, 51.519, 35.675, 35.698), "W506037515", 15),
        ("D. Muthanna · Regiment Command\nMuthanna Governorate · Iraq", [("Cross-country", 460, 0)], (45.301, 45.320, 31.260, 31.279), "W1308481475", 15),
        ("E. Yellow Line · Military Demarcation Line\nGaza Strip · Palestine", [("Cross-country", mcc, mnc) for mcc, mncs in {280: (1,), 284: (3,), 286: (1, 2, 3), 416: (1,), 420: (4,), 424: (2,), 426: (6,), 428: (28,), 602: (1, 2, 3, 4, 11), 606: (0, 1)}.items() for mnc in mncs], (34.205, 34.535, 31.210, 31.555), None, 11),
        ("F. Ukrainka · Air Base\nAmur Oblast · Russia", [("Cross-country", 460, 0)], (128.414, 128.501, 51.140, 51.194), "W366468209", 14),
        ("G. Chkalovsky · Military Unit\nMoscow Oblast · Russia", [("Cross-country", 255, 3)], (38.132, 38.162, 55.881, 55.899), "W445080855", 15),
        ("H. Dübendorf · Military Airfield\nZürich · Switzerland", [("Cross-country", 294, 1)], (8.620, 8.674, 47.389, 47.413), "W180859413", 14),
        ("I. Porton Down · Defence Laboratory\nEngland · United Kingdom", [("Testing", 1, 1)], (-1.711, -1.692, 51.1295, 51.1402), "W28121567", 15),
        ("J. Muscatatuck · Urban Training Center\nIndiana · United States", [("Cross-country", 230, 1), ("Cross-country", 283, 508), ("Cross-country", 283, 518), ("Cross-country", 426, 1), ("Cross-country", 525, 3), ("Cross-country", 525, 5)], (-85.548, -85.517, 39.037, 39.061), "W966102529", 15),
        ("K. Fort Bragg · Army Base\nNorth Carolina · United States", [("Cross-country", 553, 96), ("Cross-country", 553, 97), ("Cross-country", 553, 987)], (-79.390, -79.210, 35.025, 35.175), "R176940", 13),
        ("L. Stockbridge · Experimental Facility\nNew York · United States", [("Testing", 1, 1)], (-75.663, -75.640, 43.0250, 43.0445), "W1289150743", 15),
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
    fig, axes = plt.subplots(3, 4, figsize=(13.8, 8.65))
    fig.subplots_adjust(left=0.025, right=0.992, top=0.910, bottom=0.080, wspace=0.11, hspace=0.30)
    fig.suptitle(
        "Geopolitically and institutionally significant anomalous MCC identities at military sites",
        fontsize=13.3,
        fontweight="bold",
        y=0.975,
    )

    for ax, (title, networks, selection_bbox, osm_id, zoom) in zip(axes.flat, specs, strict=True):
        rows = select(rows_by_category, networks, selection_bbox)
        if not rows:
            raise RuntimeError(f"No identities selected for {title}")
        draw_panel(
            ax,
            rows,
            fit_bbox_to_panel(selection_bbox),
            title,
            osm_id,
            zoom=zoom,
            footer_text=compact_footer(rows),
        )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CATEGORY_COLORS["Cross-country"], markeredgecolor="white", markersize=6, label="Foreign MCC identity"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CATEGORY_COLORS["Testing"], markeredgecolor="white", markersize=6, label="Testing MCC identity"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CATEGORY_COLORS["Unassigned"], markeredgecolor="white", markersize=6, label="Unassigned MCC identity"),
        *[
            plt.Line2D([0], [0], marker=TECH_MARKERS[tech], linestyle="", color="#555555", markerfacecolor="#777777", markersize=5.5, label=label)
            for tech, label in (("gsm", "GSM/UMTS"), ("lte", "LTE"), ("nr", "NR"))
        ],
        plt.Line2D([0], [0], color="#ffd166", linewidth=2.0, label="Mapped facility boundary"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.025, 0.014), ncol=7, frameon=False, fontsize=6.2, handletextpad=0.35, columnspacing=0.8)
    fig.text(0.99, 0.018, "Imagery © Esri and contributors · boundaries/labels © OpenStreetMap contributors, © CARTO", ha="right", va="bottom", fontsize=5.55, color="#666666")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FIGS / "geopolitical_defense_mcc_maps.pdf")
    parser.add_argument("--preview", type=Path, default=FIGS / "geopolitical_defense_mcc_maps.png")
    args = parser.parse_args()
    make_figure(args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
