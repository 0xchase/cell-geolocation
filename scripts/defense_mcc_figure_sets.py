#!/usr/bin/env python3
"""Render deduplicated main-paper and appendix defense-site MCC map sets."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from military_defense_maps import (
    CATEGORY_COLORS,
    DATA,
    OUT_OF_COUNTRY_DATA,
    TECH_MARKERS,
    draw_panel,
    fit_bbox_to_panel,
    load_rows,
    select,
)


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"


def compact_grouped_footer(rows: list[dict[str, object]]) -> str:
    """Keep multi-PLMN labels legible in reduced multi-panel figures."""
    categories = "/".join(sorted({str(row["category"]) for row in rows}))
    networks: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        networks[int(row["mcc"])].add(int(row["mnc"]))
    labels = []
    for mcc, mncs in sorted(networks.items()):
        ordered = sorted(mncs)
        mnc_label = str(ordered[0]) if len(ordered) == 1 else "{" + ",".join(map(str, ordered)) + "}"
        labels.append(f"{mcc:03d}/{mnc_label}")
    return f"{categories}: {', '.join(labels)} · n={len(rows):,} · {sum(int(row['obs']) for row in rows):,} obs"


def spec(
    title: str,
    place: str,
    networks: list[tuple[str, int, int]],
    bbox: tuple[float, float, float, float],
    osm_id: str | None,
    zoom: int,
):
    return (title, place, networks, bbox, osm_id, zoom)


# The four main-paper testing/unassigned cases requested by the authors.
MAIN_ANOMALOUS = [
    spec("DGA Landes · Missile Test Range", "Nouvelle-Aquitaine · France", [("Unassigned", 9, 9)], (-1.275, -1.210, 44.405, 44.457), "R3234655", 14),
    spec("Idaho National Laboratory · Wireless Test Range", "Idaho · United States", [("Testing", 1, 1), ("Unassigned", 103, 10)], (-113.140, -112.920, 43.515, 43.570), None, 13),
    spec("Porton Down · Defence Laboratory", "England · United Kingdom", [("Testing", 1, 1)], (-1.711, -1.692, 51.1295, 51.1402), "W28121567", 15),
    spec("Stockbridge · Experimental Facility", "New York · United States", [("Testing", 1, 1)], (-75.663, -75.640, 43.0250, 43.0445), "W1289150743", 15),
]


# Commercial sites where unusual MCC activity can disclose the existence,
# extent, or persistence of private infrastructure and sensitive testing.
# Ordered by country, then site.
MAIN_COMMERCIAL = [
    spec("Cadia · Gold and Copper Mine", "New South Wales · Australia", [("Private", 999, 50), ("Private", 999, 99)], (148.975, 149.065, -33.560, -33.445), None, 13),
    spec("Tesla Gigafactory Berlin · Automobile Factory", "Brandenburg · Germany", [("Private", 999, 40)], (13.783, 13.819, 52.387, 52.407), "W775978799", 15),
    spec("Applus+ IDIADA · Automotive Proving Ground", "Catalonia · Spain", [("Private", 999, 10)], (1.493, 1.541, 41.257, 41.275), "W33790449", 14),
    spec("Point Beach · Nuclear Power Plant", "Wisconsin · United States", [("Private", 999, 40)], (-87.541, -87.532, 44.276, 44.284), "W109774222", 16),
]


# Remaining testing, unassigned, and private cases, deduplicated by site.
APPENDIX_ANOMALOUS = [
    spec("Sanya · Heliport", "Hainan · China", [("Testing", 1, 0)], (109.451, 109.473, 18.279, 18.298), "W469039710", 15),
    spec("Hohenfels · Training Area", "Bavaria · Germany", [("Testing", 1, 1)], (11.785, 11.925, 49.185, 49.285), "W229977366", 13),
    spec("Naliya · Air Force Station", "Gujarat · India", [("Unassigned", 123, 45)], (68.860, 68.923, 23.202, 23.244), "W138755258", 14),
    spec("Hamamatsu · Air Base", "Shizuoka · Japan", [("Private", 999, 1)], (137.684, 137.723, 34.739, 34.760), "W42722174", 15),
    spec("Niamey · Air Base", "Niamey · Niger", [("Testing", 1, 1)], (2.075, 2.205, 13.475, 13.550), "W871876288", 13),
    spec("Rena · Army Camp", "Innlandet · Norway", [("Testing", 1, 1)], (11.335, 11.515, 61.105, 61.270), "W962221904", 12),
    spec("Copehill Down · Training Area", "England · United Kingdom", [("Testing", 1, 1)], (-2.070, -1.890, 51.160, 51.245), "W258574467", 12),
    spec("Fort Huachuca · Army Base", "Arizona · United States", [("Testing", 1, 1), ("Unassigned", 111, 94)], (-110.455, -110.285, 31.345, 31.605), "R2262740", 12),
    spec("Camp Roberts · National Guard Training Site", "California · United States", [("Testing", 1, 1)], (-120.875, -120.665, 35.690, 35.785), "R317716", 12),
    spec("China Lake · Naval Air Weapons Station", "California · United States", [("Testing", 1, 1)], (-117.620, -117.560, 35.605, 35.655), "R317710", 14),
    spec("Eglin · Air Force Base", "Florida · United States", [("Unassigned", 111, 111)], (-86.645, -86.565, 30.615, 30.685), "R533644", 14),
    spec("Grand Forks · Air Force Base", "North Dakota · United States", [("Testing", 1, 10)], (-97.500, -97.300, 47.890, 48.065), "W556280798", 12),
    spec("Fort Pickett · Maneuver Training Center", "Virginia · United States", [("Testing", 1, 1)], (-78.020, -77.835, 36.960, 37.125), "R4251895", 12),
    spec("Quantico · Marine Corps Base", "Virginia · United States", [("Private", 999, 99)], (-77.580, -77.500, 38.545, 38.615), "R2586049", 14),
]


# The four main-paper wrong-country cases requested by the authors.
MAIN_FOREIGN = [
    spec("IRGC Command · Military Headquarters", "Tehran Province · Iran", [("Cross-country", 400, 1)], (51.492, 51.519, 35.675, 35.698), "W506037515", 15),
    spec("Chkalovsky · Military Unit", "Moscow Oblast · Russia", [("Cross-country", 255, 3)], (38.132, 38.162, 55.881, 55.899), "W445080855", 15),
    spec("Muscatatuck · Urban Training Center", "Indiana · United States", [("Cross-country", 230, 1), ("Cross-country", 283, 508), ("Cross-country", 283, 518), ("Cross-country", 426, 1), ("Cross-country", 525, 3), ("Cross-country", 525, 5)], (-85.548, -85.517, 39.037, 39.061), "W966102529", 15),
    spec("Fort Bragg · Army Base", "North Carolina · United States", [("Cross-country", 553, 96), ("Cross-country", 553, 97), ("Cross-country", 553, 987)], (-79.390, -79.210, 35.025, 35.175), "R176940", 13),
]


# Remaining wrong-country cases from the two source grids, deduplicated by site.
APPENDIX_FOREIGN = [
    spec("Khojaly · Military Base", "Khojaly District · Azerbaijan", [("Cross-country", 283, 4)], (46.788, 46.806, 39.884, 39.900), "W429004217", 15),
    spec("Gozha · Training Ground", "Grodno Region · Belarus", [("Cross-country", 246, 1), ("Cross-country", 246, 2), ("Cross-country", 246, 8)], (23.900, 24.055, 53.830, 53.925), "W314965353", 13),
    spec("Muthanna · Regiment Command", "Muthanna Governorate · Iraq", [("Cross-country", 460, 0)], (45.301, 45.320, 31.260, 31.279), "W1308481475", 15),
    spec("Yellow Line · Military Demarcation Line", "Gaza Strip · Palestine", [("Cross-country", mcc, mnc) for mcc, mncs in {280: (1,), 284: (3,), 286: (1, 2, 3), 416: (1,), 420: (4,), 424: (2,), 426: (6,), 428: (28,), 602: (1, 2, 3, 4, 11), 606: (0, 1)}.items() for mnc in mncs], (34.205, 34.535, 31.210, 31.555), None, 11),
    spec("Ukrainka · Air Base", "Amur Oblast · Russia", [("Cross-country", 460, 0)], (128.414, 128.501, 51.140, 51.194), "W366468209", 14),
    spec("Dübendorf · Military Airfield", "Zürich · Switzerland", [("Cross-country", 294, 1)], (8.620, 8.674, 47.389, 47.413), "W180859413", 14),
    spec("Berdyansk · Airfield", "Zaporizhzhia Oblast · Ukraine", [("Cross-country", 250, 94), ("Cross-country", 250, 96), ("Cross-country", 250, 97)], (36.731, 36.795, 46.792, 46.837), "W506925303", 14),
]


def load_all_rows():
    cross_country: dict[tuple[object, ...], dict[str, object]] = {}
    for row in [
        *load_rows("geopolitical_defense_cases.csv", OUT_OF_COUNTRY_DATA),
        *load_rows("defense_additions.csv", OUT_OF_COUNTRY_DATA),
    ]:
        key = tuple(row[field] for field in ("mcc", "mnc", "lac", "cid", "cell_type", "glat", "glon"))
        cross_country[key] = row
    return {
        "Testing": load_rows("testing.csv"),
        "Unassigned": load_rows("unassigned.csv"),
        "Private": load_rows("private.csv"),
        "Cross-country": list(cross_country.values()),
    }


def legend_handles(categories: set[str]):
    order = ("Testing", "Unassigned", "Private", "Cross-country")
    labels = {"Cross-country": "Foreign MCC", "Testing": "Testing", "Unassigned": "Unassigned", "Private": "Private"}
    category_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CATEGORY_COLORS[category], markeredgecolor="white", markersize=6, label=labels[category])
        for category in order if category in categories
    ]
    tech_handles = [
        plt.Line2D([0], [0], marker=TECH_MARKERS[tech], linestyle="", color="#555555", markerfacecolor="#777777", markersize=5.5, label=label)
        for tech, label in (("gsm", "GSM/UMTS"), ("lte", "LTE"), ("nr", "NR"))
    ]
    return [*category_handles, *tech_handles, plt.Line2D([0], [0], color="#ffd166", linewidth=2.0, label="Mapped facility boundary")]


def render(specs, output: Path, preview: Path, title: str, rows: int, cols: int = 4) -> None:
    rows_by_category = load_all_rows()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    column_figure = cols <= 2
    column_quad = rows == 2 and cols == 2
    width = 3.35 if column_figure else 13.8
    height = (3.05 if rows == 1 and cols == 1 else 3.62) if column_figure else 2.82 * rows + 0.75
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    bottom = 0.12 if rows == 1 and cols == 1 else (0.105 if column_quad else 0.065)
    top = 0.92 if rows == 1 and cols == 1 else (0.94 if column_quad else 0.91)
    fig.subplots_adjust(left=0.025, right=0.992, top=top, bottom=bottom, wspace=0.08 if column_figure else 0.11, hspace=0.17 if column_figure else 0.31)
    if title:
        fig.suptitle(title, fontsize=7.4 if column_quad else 13.3, fontweight="bold", y=0.975)
    categories: set[str] = set()
    for index, (ax, panel) in enumerate(zip(axes.flat, specs)):
        panel_title, place, networks, bbox, osm_id, zoom = panel
        selected = select(rows_by_category, networks, bbox)
        if not selected:
            raise RuntimeError(f"No identities selected for {panel_title}")
        categories.update(str(row["category"]) for row in selected)
        draw_panel(
            ax,
            selected,
            fit_bbox_to_panel(bbox),
            f"{chr(65 + index) + '. ' if len(specs) > 1 else ''}{panel_title}\n{place}",
            osm_id,
            zoom=zoom,
            footer_text=compact_grouped_footer(selected),
            title_fontsize=6.1 if rows == 1 and cols == 1 else (4.7 if column_quad else 7.15),
            footer_fontsize=4.4 if rows == 1 and cols == 1 else (3.65 if column_quad else 5.25),
        )
    for ax in axes.flat[len(specs):]:
        ax.set_visible(False)
    handles = legend_handles(categories)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.025, 0.006), ncol=3 if column_figure else len(handles), frameon=False, fontsize=4.5 if column_figure else 6.2, handletextpad=0.3, columnspacing=0.55 if column_figure else 0.8)
    fig.text(0.99, 0.010, "Imagery © Esri and contributors · boundaries/labels © OpenStreetMap contributors, © CARTO", ha="right", va="bottom", fontsize=3.25 if column_figure else 5.55, color="#666666")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Appendix grids use a double-width source canvas and are reduced by about
    # half in the paper; column quads are exported at their final physical size.
    pdf_dpi = 300 if column_figure else 180
    fig.savefig(output, dpi=pdf_dpi, bbox_inches="tight")
    fig.savefig(preview, dpi=180 if column_figure else 160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    jobs = [
        (MAIN_ANOMALOUS, FIGS / "military_defense_mcc_maps.pdf", FIGS / "military_defense_mcc_maps.png", "", 2, 2),
        (MAIN_COMMERCIAL, FIGS / "tesla_gigafactory_private_mcc_map.pdf", FIGS / "tesla_gigafactory_private_mcc_map.png", "", 2, 2),
        (MAIN_FOREIGN, FIGS / "geopolitical_defense_mcc_maps.pdf", FIGS / "geopolitical_defense_mcc_maps.png", "", 2, 2),
        (APPENDIX_ANOMALOUS, FIGS / "test_mcc_defense_appendix_maps.pdf", FIGS / "test_mcc_defense_appendix_maps.png", "Additional testing, unassigned, and private MCC identities at defense-related facilities", math.ceil(len(APPENDIX_ANOMALOUS) / 4), 4),
        (APPENDIX_FOREIGN, FIGS / "foreign_mcc_defense_appendix_maps.pdf", FIGS / "foreign_mcc_defense_appendix_maps.png", "Additional foreign MCC identities at military sites", math.ceil(len(APPENDIX_FOREIGN) / 4), 4),
    ]
    for specs, output, preview, title, rows, cols in jobs:
        render(specs, output, preview, title, rows, cols)
        print(output)


if __name__ == "__main__":
    main()
