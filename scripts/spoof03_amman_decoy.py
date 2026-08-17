#!/usr/bin/env python3
"""The Queen Alia decoy: a GNSS-spoofing campaign seen through a cell database.

From 2024-09-05, cells belonging to Israeli (MCC 425), Lebanese (415) and Syrian
(417) operators begin appearing at a single coordinate in Jordan -- 31.717 N,
35.999 E, the runway complex of Queen Alia International Airport, ~70 km from
the nearest Israeli territory and ~200 km from the Galilee panhandle where most
of the affected cells actually live.

The mechanism is that Apple's Cellular Positioning System derives each cell's
coordinate from crowdsourced handset GNSS fixes. A handset whose receiver has
been spoofed to a decoy reports every cell it can hear as being at that decoy,
so the decoy accumulates cells that are physically nowhere near it.

Four independent properties separate this from the ordinary explanations:
  * onset  - all three foreign MCCs begin within 27 minutes of each other on a
             single day, after ten months of the dataset showing none of them
  * decay  - the campaign fades through 2025 and stops; positioning artifacts
             and Apple centroid fallbacks do not switch off
  * point  - 32% of displaced observations land on ONE coordinate, versus 0.5%
             for the genuinely-local Jordanian cells in the same box
  * source - the displaced cells' home positions cluster on the Israel-Lebanon
             border, not on Tel Aviv/Ben Gurion as a traveller artifact requires
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_helpers import COUNTRIES_GEOJSON, draw_geojson_layer
from ch_remote import ch_df

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"

DECOY_LAT, DECOY_LON = 31.717, 35.999          # Queen Alia International Airport
BOX = (31.6, 31.85, 35.9, 36.1)                 # lat_lo, lat_hi, lon_lo, lon_hi

MCC_NAME = {415: "Lebanon (415)", 416: "Jordan (416)", 417: "Syria (417)",
            420: "Saudi (420)", 425: "Israel/PS (425)", 901: "Intl (901)"}
MCC_COLOR = {415: "#c9743a", 416: "#8b8b8b", 417: "#4f7f52",
             420: "#8e6aa7", 425: "#b23a48", 901: "#2f6f9f"}


def load_data() -> dict[str, pd.DataFrame]:
    box = f"lat BETWEEN {BOX[0]} AND {BOX[1]} AND lon BETWEEN {BOX[2]} AND {BOX[3]}"

    composition = ch_df(f"""
        SELECT mcc, count() AS obs, uniqExact((mnc,lac,cid)) AS cells,
               min(timestamp) AS first, max(timestamp) AS last
        FROM cell.geos
        WHERE mcc IN (415,416,417,420,425,901) AND cid>0 AND {box}
        GROUP BY mcc ORDER BY cells DESC""")

    timeline = ch_df(f"""
        SELECT toStartOfMonth(timestamp) AS month, mcc,
               uniqExact((mnc,lac,cid)) AS cells
        FROM cell.geos
        WHERE mcc IN (415,417,425) AND cid>0 AND {box}
        GROUP BY month, mcc ORDER BY month""")

    # Concentration control: local infrastructure vs displaced foreign cells.
    concentration = ch_df(f"""
        SELECT mcc,
               countIf(abs(lat-{DECOY_LAT})<0.002 AND abs(lon-{DECOY_LON})<0.002) AS at_point,
               count() AS obs_in_box,
               uniqExact((round(lat,3),round(lon,3))) AS distinct_coords
        FROM cell.geos WHERE mcc IN (416,425) AND cid>0 AND {box}
        GROUP BY mcc""")

    # Where the displaced Israeli cells actually live.
    sources = ch_df(f"""
        WITH decoyed AS (
          SELECT DISTINCT mnc,lac,cid FROM cell.geos
          WHERE mcc=425 AND cid>0 AND {box}
        )
        SELECT round(lat,2) AS lat, round(lon,2) AS lon,
               uniqExact((mnc,lac,cid)) AS cells
        FROM cell.geos
        WHERE mcc=425 AND cid>0 AND (mnc,lac,cid) IN decoyed
          AND lat BETWEEN 29.4 AND 33.45 AND lon BETWEEN 34.2 AND 35.95
        GROUP BY lat, lon""")

    # Landing scatter at the decoy: displaced (425) against local (416).
    landing = ch_df(f"""
        SELECT mcc, round(lat,3) AS lat, round(lon,3) AS lon, count() AS obs
        FROM cell.geos WHERE mcc IN (416,425) AND cid>0 AND {box}
        GROUP BY mcc, lat, lon""")

    timeline["month"] = pd.to_datetime(timeline["month"])
    return {"composition": composition, "timeline": timeline,
            "concentration": concentration, "sources": sources, "landing": landing}


def make_figure(d: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    comp, timeline = d["composition"].copy(), d["timeline"].copy()
    conc, sources, landing = d["concentration"].copy(), d["sources"].copy(), d["landing"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.02)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), constrained_layout=True)
    fig.suptitle("A GNSS-spoofing campaign recorded in a cell-location database: "
                 "the Queen Alia decoy", fontsize=13.5, fontweight="bold")

    # A. Regional geometry: source region -> decoy.
    ax = axes[0, 0]
    bbox = (33.6, 37.4, 29.2, 34.6)
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(ax, COUNTRIES_GEOJSON, bbox, facecolor="#f5f1e8",
                       edgecolor="#69635c", linewidth=0.5, zorder=0)
    if not sources.empty:
        ax.scatter(sources["lon"], sources["lat"], s=6 + sources["cells"] * 0.35,
                   color="#b23a48", alpha=0.55, edgecolor="none", zorder=3,
                   label="home of displaced cells")
        # Arrow from the dominant source concentration to the decoy.
        top = sources.sort_values("cells", ascending=False).iloc[0]
        ax.annotate("", xy=(DECOY_LON, DECOY_LAT), xytext=(top["lon"], top["lat"]),
                    arrowprops={"arrowstyle": "-|>", "color": "#333", "lw": 1.6,
                                "linestyle": "--", "shrinkA": 3, "shrinkB": 6}, zorder=4)
    ax.scatter([DECOY_LON], [DECOY_LAT], marker="X", s=210, color="#111",
               edgecolor="white", linewidth=1.2, zorder=6, label="decoy (QAIA)")
    ax.text(DECOY_LON + 0.12, DECOY_LAT - 0.28, "Queen Alia Int'l\n31.717, 35.999",
            fontsize=7.6, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4}, zorder=7)
    for name, la, lo in [("Beirut", 33.89, 35.50), ("Damascus", 33.51, 36.29),
                         ("Tel Aviv", 32.08, 34.78), ("Amman", 31.95, 35.93)]:
        ax.text(lo, la, name, fontsize=6.8, color="#3f3a35", ha="center", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5, "pad": 0.8})
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", fontsize=7.0, frameon=True, framealpha=0.85)
    ax.set_title("A. Displaced cells live on the Israel-Lebanon border,\n"
                 "not near the decoy", fontsize=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # B. Synchronised onset.
    ax = axes[0, 1]
    for mcc, grp in timeline.groupby("mcc"):
        grp = grp.sort_values("month")
        ax.plot(grp["month"], grp["cells"], marker="o", markersize=3.4, linewidth=1.7,
                color=MCC_COLOR.get(int(mcc), "#333"), label=MCC_NAME.get(int(mcc), str(mcc)))
    ax.axvline(pd.Timestamp("2024-09-05"), color="#111", linestyle="--", linewidth=1.1)
    ax.annotate("2024-09-05\nall three MCCs begin\nwithin 27 minutes",
                xy=(pd.Timestamp("2024-09-05"), ax.get_ylim()[1] * 0.62),
                xytext=(8, 0), textcoords="offset points", fontsize=7.4, va="top",
                bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "alpha": 0.85, "pad": 2})
    ax.set_yscale("log")
    ax.set_title("B. Foreign cells appear at the decoy on a single day", fontsize=10)
    ax.set_ylabel("Distinct cells at decoy (log)"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7.4, frameon=True)

    # C. Concentration control.
    ax = axes[1, 0]
    conc["pct"] = 100 * conc["at_point"] / conc["obs_in_box"]
    conc["label"] = conc["mcc"].map({416: "Jordan (416)\nlocal infrastructure",
                                     425: "Israel (425)\ndisplaced"})
    bars = ax.bar(conc["label"], conc["pct"],
                  color=[MCC_COLOR.get(int(m), "#333") for m in conc["mcc"]], width=0.55)
    for bar, row in zip(bars, conc.itertuples(), strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.9,
                f"{row.pct:.1f}%\n{row.distinct_coords:,} distinct coords",
                ha="center", fontsize=8)
    ax.set_ylim(0, max(conc["pct"]) * 1.35)
    ax.set_ylabel("Observations at the single decoy point (%)")
    ax.set_title("C. Displaced cells pile onto one coordinate;\nlocal cells do not", fontsize=10)

    # D. Landing pattern at the airport.
    ax = axes[1, 1]
    for mcc, color, label, z in [(416, "#8b8b8b", "Jordan (416) local", 2),
                                 (425, "#b23a48", "Israel (425) displaced", 3)]:
        sub = landing[landing["mcc"] == mcc]
        ax.scatter(sub["lon"], sub["lat"], s=3 + sub["obs"] ** 0.5 * 0.9,
                   color=color, alpha=0.22 if mcc == 416 else 0.55,
                   edgecolor="none", label=label, zorder=z)
    ax.scatter([DECOY_LON], [DECOY_LAT], marker="X", s=150, color="#111",
               edgecolor="white", linewidth=1.0, zorder=6)
    ax.set_xlim(BOX[2], BOX[3]); ax.set_ylim(BOX[0], BOX[1])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=7.4, frameon=True)
    # The westward "comet tail" is a direct consequence of the mechanism: Apple's
    # estimate is a weighted mean of true and spoofed reports, so a cell sits on
    # the line between its real home (west, in Israel) and the decoy, at a
    # distance set by what fraction of its reports were spoofed. The tail
    # therefore points back at the spoofed population.
    ax.annotate("tail points back toward Israel:\nestimates are a mixture of\ntrue and spoofed reports",
                xy=(35.955, 31.726), xytext=(35.905, 31.79), fontsize=7.0,
                arrowprops={"arrowstyle": "->", "color": "#111", "lw": 0.9},
                bbox={"facecolor": "white", "edgecolor": "#bdb7ae", "alpha": 0.88, "pad": 2},
                zorder=8)
    ax.set_title("D. Local cells map the airport; displaced cells\ncollapse toward one point",
                 fontsize=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=PLOTS / "spoof03_amman_decoy.pdf")
    p.add_argument("--preview", type=Path, default=None)
    a = p.parse_args()
    make_figure(load_data(), a.output, a.preview)
    print(a.output)


if __name__ == "__main__":
    main()
