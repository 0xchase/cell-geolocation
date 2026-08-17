#!/usr/bin/env python3
"""How much of the location service is corrupted, by how much, and for how long.

The security question is not only "was there spoofing" but "what did it do to a
production positioning system". A cell whose published coordinate has been pulled
toward a decoy will mislocate any device that relies on it, including devices
whose own GNSS is perfectly healthy — the error propagates from the spoofed
population to everyone else in the same cells.

Panels:
  A  magnitude distribution of displacement
  B  how much of the corpus is affected, by detector class
  C  persistence: how long a displaced cell stays displaced
  D  corrupted-observation share over time
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
DIST_DISPLACED_KM = 12.0   # matches the census classifier floor


def load_data() -> dict[str, pd.DataFrame]:
    mag = ch_df("""
        SELECT floor(log10(greatest(km, 1)) * 4) / 4 AS logkm,
               sum(obs) AS obs, count() AS rows
        FROM cell.displaced GROUP BY logkm ORDER BY logkm""")

    totals = ch_df("""
        SELECT
          (SELECT sum(total_obs) FROM cell.cellhome)                AS all_obs,
          (SELECT count() FROM cell.cellhome)                       AS all_cells,
          (SELECT sum(obs) FROM cell.displaced)                     AS disp_obs,
          (SELECT uniqExact((mcc,mnc,lac,cid,cell_type)) FROM cell.displaced) AS disp_cells,
          (SELECT sum(obs) FROM cell.displaced WHERE km > 100)      AS far_obs,
          (SELECT uniqExact((mcc,mnc,lac,cid,cell_type)) FROM cell.displaced WHERE km > 100) AS far_cells""")

    persist = ch_df("""
        SELECT least(dateDiff('day', first_seen, last_seen), 900) AS days, count() AS cells
        FROM cell.displaced WHERE km > 25 GROUP BY days ORDER BY days""")

    monthly = ch_df("""
        SELECT toStartOfMonth(first_seen) AS month,
               sum(obs) AS disp_obs,
               uniqExact((mcc,mnc,lac,cid,cell_type)) AS disp_cells
        FROM cell.displaced WHERE km > 25
        GROUP BY month ORDER BY month""")
    monthly["month"] = pd.to_datetime(monthly["month"])
    return {"mag": mag, "totals": totals, "persist": persist, "monthly": monthly}


def make_figure(d: dict[str, pd.DataFrame], output: Path, preview: Path | None) -> None:
    mag, totals = d["mag"].copy(), d["totals"].iloc[0]
    persist, monthly = d["persist"].copy(), d["monthly"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.02)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)
    fig.suptitle("Corruption of the location service: magnitude, extent, persistence",
                 fontsize=13.4, fontweight="bold")

    # A. Magnitude.
    ax = axes[0, 0]
    ax.bar(10 ** mag["logkm"], mag["obs"], width=10 ** mag["logkm"] * 0.55,
           color="#b23a48", alpha=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(DIST_DISPLACED_KM, color="#333", linestyle="--", linewidth=1.0)
    ax.text(DIST_DISPLACED_KM * 1.1, ax.get_ylim()[1] * 0.25,
            f"{DIST_DISPLACED_KM:.0f} km\nclassifier floor", fontsize=7.2)
    ax.set_xlabel("Displacement from the cell's own home (km, log)")
    ax.set_ylabel("Observations (log)")
    ax.set_title("A. How wrong the published coordinate is", fontsize=10.4)

    # B. Extent.
    ax = axes[0, 1]
    labels = ["any displacement\n(>10 km)", "gross displacement\n(>100 km)"]
    cell_pct = [100 * totals["disp_cells"] / totals["all_cells"],
                100 * totals["far_cells"] / totals["all_cells"]]
    obs_pct = [100 * totals["disp_obs"] / totals["all_obs"],
               100 * totals["far_obs"] / totals["all_obs"]]
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w/2, cell_pct, w, color="#b23a48", label="% of cells")
    ax.bar(x + w/2, obs_pct, w, color="#2f6f9f", label="% of observations")
    for xi, (c, o) in enumerate(zip(cell_pct, obs_pct, strict=False)):
        ax.text(xi - w/2, c, f"{c:.2f}%\n({totals['disp_cells' if xi==0 else 'far_cells']:,})",
                ha="center", va="bottom", fontsize=7.4)
        ax.text(xi + w/2, o, f"{o:.2f}%", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_yscale("log")
    ax.set_ylabel("Share of the corpus (%, log)")
    ax.legend(fontsize=7.6)
    ax.set_title("B. Rare per cell, but a large absolute population", fontsize=10.4)

    # C. Persistence.
    ax = axes[1, 0]
    if not persist.empty:
        cum = persist["cells"].cumsum() / persist["cells"].sum()
        ax.plot(persist["days"], cum, linewidth=1.9, color="#b23a48")
        for q in (0.5, 0.9):
            idx = int(np.searchsorted(cum.values, q))
            if idx < len(persist):
                dv = int(persist["days"].iloc[idx])
                ax.axhline(q, color="#999", linewidth=0.7, linestyle=":")
                ax.annotate(f"{int(q*100)}% within {dv} d", xy=(dv, q), fontsize=7.4,
                            xytext=(6, -10), textcoords="offset points")
    ax.set_xlabel("Days between first and last displaced observation")
    ax.set_ylabel("Cumulative fraction of displaced cells")
    ax.set_title("C. Most corruption is transient, a tail is not", fontsize=10.4)

    # D. Over time.
    ax = axes[1, 1]
    if not monthly.empty:
        ax.plot(monthly["month"], monthly["disp_cells"], marker="o", markersize=3.2,
                linewidth=1.6, color="#b23a48")
        ax.set_yscale("log")
    ax.set_ylabel("Cells newly displaced >25 km (log)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("D. Onset of new corruption over the study period", fontsize=10.4)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=PLOTS / "spoof05_contamination.pdf")
    p.add_argument("--preview", type=Path, default=None)
    a = p.parse_args()
    make_figure(load_data(), a.output, a.preview)
    print(a.output)


if __name__ == "__main__":
    main()
