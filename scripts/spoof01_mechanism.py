#!/usr/bin/env python3
"""How GNSS spoofing enters a crowdsourced cell-location database, and evidence
that the recorded corruption really is a mixture of true and spoofed reports.

The platform's coordinate for a cell is an aggregate of GNSS fixes contributed by
handsets that heard it. If a fraction w of contributing fixes are spoofed to a
decoy D and the rest are honest at the true location H, then a linear aggregate
lands at (1-w)*H + w*D -- i.e. ON THE SEGMENT between the tower's real position
and the decoy.

That is a falsifiable prediction, and it is what the Queen Alia campaign shows:
displaced estimates lie along the home->decoy axis with small perpendicular
error, rather than scattering isotropically around the decoy (which is what a
"positions are randomly wrong here" artifact would produce).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ch_remote import ch_df

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"

DECOY_LAT, DECOY_LON = 31.717, 35.999
BOX = (31.6, 31.85, 35.9, 36.1)


def load_data() -> pd.DataFrame:
    box = f"lat BETWEEN {BOX[0]} AND {BOX[1]} AND lon BETWEEN {BOX[2]} AND {BOX[3]}"
    return ch_df(f"""
        WITH decoyed AS (
            SELECT DISTINCT mnc, lac, cid FROM cell.geos
            WHERE mcc=425 AND cid>0 AND {box}
        ),
        pos AS (
            SELECT mnc, lac, cid, round(lat,3) AS la, round(lon,3) AS lo, count() AS n
            FROM cell.geos
            WHERE mcc=425 AND cid>0 AND (mnc,lac,cid) IN decoyed AND NOT(lat=0 AND lon=0)
            GROUP BY mnc, lac, cid, la, lo
        ),
        home AS (
            SELECT mnc, lac, cid, argMax(la,n) AS hla, argMax(lo,n) AS hlo, sum(n) AS tot
            FROM pos GROUP BY mnc, lac, cid
        )
        SELECT p.la AS la, p.lo AS lo, p.n AS n,
               h.hla AS hla, h.hlo AS hlo, h.tot AS tot
        FROM pos AS p INNER JOIN home AS h
          ON p.mnc=h.mnc AND p.lac=h.lac AND p.cid=h.cid
        WHERE h.hla < 33.5 AND h.hlo < 35.95""")   # home genuinely in Israel/PS


def project(df: pd.DataFrame) -> pd.DataFrame:
    """Along-/cross-track coordinates relative to each cell's home->decoy axis."""
    kx = 111.32 * np.cos(np.radians(df["hla"]))       # km per degree lon
    ky = 110.57                                        # km per degree lat
    vx = (DECOY_LON - df["hlo"]) * kx
    vy = (DECOY_LAT - df["hla"]) * ky
    L = np.hypot(vx, vy)
    px = (df["lo"] - df["hlo"]) * kx
    py = (df["la"] - df["hla"]) * ky
    df = df.copy()
    df["home_decoy_km"] = L
    df["along"] = (px * vx + py * vy) / np.where(L == 0, np.nan, L)
    df["cross"] = (px * vy - py * vx) / np.where(L == 0, np.nan, L)
    df["along_frac"] = df["along"] / df["home_decoy_km"]
    return df[df["home_decoy_km"] > 20]


def make_figure(df: pd.DataFrame, output: Path, preview: Path | None) -> None:
    d = project(df)
    disp = d[d["along_frac"] > 0.05]                   # displaced away from home

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.02)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    fig.suptitle("Spoofed handsets drag the platform's tower estimates along the "
                 "home-to-decoy axis", fontsize=13, fontweight="bold")

    # A. Mechanism schematic.
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # Geometry: tower top-left, decoy bottom-right, estimate on the segment.
    ax.plot([2.2, 8.4], [8.2, 2.4], color="#b23a48", linewidth=1.5,
            linestyle="--", zorder=1)
    ax.add_patch(plt.Circle((2.2, 8.2), 0.30, color="#2f6f9f", zorder=4))
    ax.text(2.2, 8.95, "tower\ntrue position $H$", ha="center", fontsize=8.2)
    ax.add_patch(plt.Circle((8.4, 2.4), 0.30, color="#b23a48", zorder=4))
    ax.text(8.4, 1.55, "decoy $D$", ha="center", fontsize=8.2, color="#b23a48")
    ax.add_patch(plt.Circle((5.3, 5.3), 0.28, facecolor="white", edgecolor="#b23a48",
                            linewidth=2.0, linestyle=":", zorder=4))
    ax.text(5.75, 5.75, "recorded estimate\n$(1-w)H + wD$", ha="left", fontsize=8.2,
            color="#b23a48", fontweight="bold")
    # Handset well clear of the segment, on the lower left.
    ax.add_patch(plt.Circle((1.1, 4.4), 0.24, color="#333", zorder=4))
    ax.text(1.1, 3.75, "handset", ha="center", fontsize=8.0)
    ax.annotate("", xy=(2.0, 7.9), xytext=(1.15, 4.7),
                arrowprops={"arrowstyle": "-|>", "color": "#555", "lw": 1.3})
    ax.text(0.95, 6.3, "hears tower;\ncontributes its\nown GNSS fix",
            fontsize=7.6, color="#555", ha="left", va="center")
    ax.annotate("", xy=(1.45, 4.4), xytext=(4.4, 3.1),
                arrowprops={"arrowstyle": "-|>", "color": "#b23a48", "lw": 1.4,
                            "linestyle": "--"})
    ax.text(4.6, 2.95, "spoofer replaces\nthat fix with $D$", fontsize=7.8,
            color="#b23a48", ha="left")
    ax.set_title("A. Mechanism", fontsize=10.5)

    # B. The prediction: displacement lies along the axis, not scattered.
    ax = axes[1]
    ax.scatter(disp["along_frac"], disp["cross"], s=2 + np.sqrt(disp["n"]) * 1.1,
               color="#b23a48", alpha=0.32, edgecolor="none")
    ax.axhline(0, color="#333", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Position along home$\\rightarrow$decoy axis\n(0 = tower's true home, 1 = decoy)")
    ax.set_ylabel("Perpendicular offset from axis (km)")
    ax.set_ylim(-60, 60)
    ax.set_xlim(-0.05, 1.25)
    med_abs = disp["cross"].abs().median()
    med_disp = disp["home_decoy_km"].median()
    ax.set_title("B. Estimates track the axis\n"
                 f"median |offset| {med_abs:.2f} km over {med_disp:.0f} km displacement",
                 fontsize=10.5)

    # C. Continuum of displacement: the corruption is graded, not binary.
    ax = axes[2]
    ax.hist(disp["along_frac"].clip(0, 1.2), bins=44, weights=disp["n"],
            color="#b23a48", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Fraction of the way from true home to decoy")
    ax.set_ylabel("Observations (log)")
    ax.set_title("C. A graded mixture, not a jump\n"
                 "(intermediate estimates are common)", fontsize=10.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=PLOTS / "spoof01_mechanism.pdf")
    p.add_argument("--preview", type=Path, default=None)
    a = p.parse_args()
    make_figure(load_data(), a.output, a.preview)
    print(a.output)


if __name__ == "__main__":
    main()
