#!/usr/bin/env python3
"""Temporal structure of the detected spoofing campaigns.

A decoy that is an artifact of the positioning system runs for as long as the
system does. A decoy produced by a transmitter starts when the transmitter is
switched on and stops when it is switched off. This figure shows that the
detected sites overwhelmingly behave like the latter: they have datable onsets,
finite durations, and in several cases onsets shared across multiple operators
and countries within hours.

Onset is measured as the first time each contributing cell is seen at the decoy.
A sharp campaign has a steep cumulative-onset curve; an artifact accumulates
cells gradually over the whole observation period.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ch_remote import ch_df

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"

SPREAD_COHERENT_KM = 600.0
DIST_DISPLACED_KM = 25.0
TOP_N = 8


def load_data() -> dict[str, pd.DataFrame]:
    decoys = ch_df(f"""
        SELECT a.plat AS plat, a.plon AS plon, a.plat/100 AS lat, a.plon/100 AS lon,
               a.cells AS cells, round(a.med_km) AS med_km,
               a.t_start AS t_start, a.t_end AS t_end,
               dateDiff('day', a.t_start, a.t_end) AS days,
               g.cc AS cc, g.city AS city
        FROM cell.attractors AS a
        LEFT JOIN cell.coord_geo AS g ON a.plat=g.klat AND a.plon=g.klon
        WHERE a.src_spread_km <= {SPREAD_COHERENT_KM}
          AND a.med_km >= {DIST_DISPLACED_KM}
          AND a.cells >= 40
        ORDER BY a.cells DESC""")

    top = decoys.head(TOP_N)
    if top.empty:
        return {"decoys": decoys, "onsets": pd.DataFrame()}
    pairs = " OR ".join(f"(plat={int(r.plat)} AND plon={int(r.plon)})"
                        for r in top.itertuples())
    onsets = ch_df(f"""
        SELECT plat, plon, toStartOfWeek(first_seen) AS week, count() AS cells
        FROM cell.displaced
        WHERE {pairs}
        GROUP BY plat, plon, week ORDER BY plat, plon, week""")
    onsets["week"] = pd.to_datetime(onsets["week"])
    for c in ("t_start", "t_end"):
        decoys[c] = pd.to_datetime(decoys[c])
    return {"decoys": decoys, "onsets": onsets, "top": top}


def label_of(r) -> str:
    place = str(r.city) if isinstance(r.city, str) and r.city.strip() else ""
    cc = str(r.cc).upper() if isinstance(r.cc, str) and r.cc.strip() else "?"
    return f"{place[:18] + ' ' if place else ''}({cc}) {r.lat:.2f},{r.lon:.2f}"


def make_figure(d: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    decoys, onsets = d["decoys"].copy(), d["onsets"].copy()
    top = d.get("top", decoys.head(TOP_N)).copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.02)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    fig.suptitle("Detected decoys behave like transmitters, not like artifacts",
                 fontsize=13.5, fontweight="bold")

    # A. Active windows.
    ax = axes[0, 0]
    tt = top.sort_values("t_start")
    for i, r in enumerate(tt.itertuples()):
        ax.plot([r.t_start, r.t_end], [i, i], linewidth=6.5, color="#b23a48",
                solid_capstyle="butt", alpha=0.85)
        ax.text(r.t_end, i, f"  {int(r.cells):,}", va="center", fontsize=7.2)
    ax.set_yticks(range(len(tt)))
    ax.set_yticklabels([label_of(r) for r in tt.itertuples()], fontsize=7.4)
    ax.set_title("A. Active window of the largest decoys\n(label: distinct cells)",
                 fontsize=10.4)
    ax.tick_params(axis="x", rotation=30)

    # B. Cumulative onset: sharp campaigns vs gradual accumulation.
    ax = axes[0, 1]
    for r in top.itertuples():
        sub = onsets[(onsets["plat"] == r.plat) & (onsets["plon"] == r.plon)]
        if sub.empty:
            continue
        sub = sub.sort_values("week")
        ax.plot(sub["week"], sub["cells"].cumsum() / sub["cells"].sum(),
                linewidth=1.5, alpha=0.85, label=label_of(r))
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Cumulative fraction of the decoy's cells")
    ax.set_title("B. Onsets are steps, not ramps", fontsize=10.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=6.2, frameon=True, loc="lower right")

    # C. Size against duration.
    ax = axes[1, 0]
    ax.scatter(decoys["days"].clip(lower=1), decoys["cells"],
               s=14, color="#b23a48", alpha=0.5, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Campaign duration (days, log)")
    ax.set_ylabel("Distinct displaced cells (log)")
    ax.set_title(f"C. {len(decoys):,} decoys: size vs duration", fontsize=10.4)

    # D. When campaigns start.
    ax = axes[1, 1]
    starts = decoys["t_start"].dropna()
    if not starts.empty:
        ax.hist(starts, bins=32, color="#b23a48", alpha=0.85)
    ax.set_ylabel("Decoys beginning")
    ax.set_title("D. Onset dates across the observation window", fontsize=10.4)
    ax.tick_params(axis="x", rotation=30)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=PLOTS / "spoof04_campaigns.pdf")
    p.add_argument("--preview", type=Path, default=None)
    a = p.parse_args()
    make_figure(load_data(), a.output, a.preview)
    print(a.output)


if __name__ == "__main__":
    main()
