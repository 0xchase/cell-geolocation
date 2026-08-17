#!/usr/bin/env python3
"""Controls: the confounders this detector must survive, and the tests that kill them.

"Many cells at one coordinate" is a weak observation on its own. This figure is
the argument that the census is not a pile of artifacts, and it is deliberately
organised around the things that could go wrong rather than around the result.

  A  anisotropy separates a decoy's directional source field from the isotropic
     scatter produced by dense-area positioning error (the dominant confounder,
     162 rejected sites, most in Japan). Note it does NOT separate decoys from
     local aggregation, which is also directional -- the displacement floor does
     that. Each test is reported against the confounder it actually removes.
  B  the same two sites as source-bearing distributions -- a decoy draws from one
     bearing, scatter draws from all of them
  C  positive/negative controls: displaced cells pile onto one coordinate while
     genuinely-local cells in the same box do not
  D  a negative result that follows from the method's physics: regions dominated
     by jamming rather than spoofing produce no decoys, because a denied receiver
     contributes no fix at all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ch_remote import ch_df
from census2 import load as load_attr, cluster, summarise, classify

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"

DECOY = ("STATIC DECOY", "STATIC DECOY (cross-border)")
AMMAN = (3168, 3178, 3595, 3605)          # plat/plon bounds of the QAIA decoy
JAPAN = (3605, 3620, 13940, 13960)        # a representative Japanese scatter site


def bearings(box) -> pd.DataFrame:
    lo_la, hi_la, lo_lo, hi_lo = box
    return ch_df(f"""
        SELECT round(degrees(atan2(plon - hlon, plat - hlat))) AS bearing,
               sum(obs) AS obs
        FROM cell.displaced
        WHERE plat BETWEEN {lo_la} AND {hi_la} AND plon BETWEEN {lo_lo} AND {hi_lo}
        GROUP BY bearing ORDER BY bearing""")


def load_data() -> dict:
    sites = summarise(cluster(load_attr()))
    sites["klass"] = sites.apply(classify, axis=1)

    conc = ch_df("""
        SELECT mcc,
               countIf(abs(lat-31.717)<0.002 AND abs(lon-35.999)<0.002) AS at_point,
               count() AS obs_in_box
        FROM cell.geos
        WHERE mcc IN (416,425) AND cid>0
          AND lat BETWEEN 31.6 AND 31.85 AND lon BETWEEN 35.9 AND 36.1
        GROUP BY mcc""")

    # Baltic states + Kaliningrad: documented interference is overwhelmingly
    # jamming. Count decoys detected on their territory.
    baltic = sites[sites["site_iso"].isin(["FI", "EE", "LV", "LT", "PL"])]
    levant = sites[sites["site_iso"].isin(["JO", "IL", "LB", "SY"])]
    return {"sites": sites, "conc": conc,
            "amman_b": bearings(AMMAN), "japan_b": bearings(JAPAN),
            "baltic": baltic, "levant": levant}


def make_figure(d: dict, output: Path, preview: Path | None) -> None:
    sites, conc = d["sites"], d["conc"]

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.02)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(12.4, 8.6), constrained_layout=True)
    axd = fig.subplot_mosaic([["A", "B", "B"], ["C", "D", "D"]])
    fig.suptitle("Controls: what could produce this signal without spoofing, and why it doesn't",
                 fontsize=13.2, fontweight="bold")

    # A. Anisotropy by class.
    ax = axd["A"]
    groups = [("decoys", sites[sites["klass"].isin(DECOY)], "#b23a48"),
              ("isotropic\nscatter", sites[sites["klass"] == "isotropic scatter"], "#9ecae1"),
              ("local\naggregation", sites[sites["klass"] == "local aggregation"], "#b9b9b9")]
    ax.boxplot([g[1]["anisotropy"].dropna() for g in groups],
               tick_labels=[g[0] for g in groups], showfliers=False,
               patch_artist=True,
               boxprops={"facecolor": "#ddd"}, medianprops={"color": "#111"})
    for i, (_, sub, c) in enumerate(groups, start=1):
        v = sub["anisotropy"].dropna()
        ax.scatter(np.random.normal(i, 0.06, len(v)), v, s=4, color=c, alpha=0.35, zorder=3)
    ax.axhline(0.60, color="#333", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Source anisotropy  (1 = one bearing, 0 = all bearings)")
    ax.set_title("A. Directionality rejects scatter\n(distance, not this test, rejects aggregation)",
                 fontsize=9.6)

    # B. Bearing distributions for the two exemplars.
    ax = axd["B"]
    for lbl, df, color in [("Queen Alia decoy (anisotropy 0.96)", d["amman_b"], "#b23a48"),
                           ("Japan 36.13,139.50 (anisotropy 0.45)", d["japan_b"], "#2f6f9f")]:
        if df.empty:
            continue
        w = df["obs"] / df["obs"].sum()
        ax.plot(df["bearing"], w.rolling(9, center=True, min_periods=1).mean(),
                linewidth=1.9, color=color, label=lbl)
    ax.set_xlim(-180, 180)
    ax.set_xlabel("Bearing from the displaced cell's home to the site (deg)")
    ax.set_ylabel("Share of displaced observations")
    ax.legend(fontsize=7.6, frameon=True)
    ax.set_title("B. A decoy pulls from one bearing; scatter pulls from all of them",
                 fontsize=10.2)

    # C. Concentration control at the decoy point.
    ax = axd["C"]
    conc = conc.copy()
    conc["pct"] = 100 * conc["at_point"] / conc["obs_in_box"]
    lab = {416: "Jordan (416)\nlocal", 425: "Israel (425)\ndisplaced"}
    col = {416: "#8b8b8b", 425: "#b23a48"}
    ax.bar([lab[m] for m in conc["mcc"]], conc["pct"],
           color=[col[m] for m in conc["mcc"]], width=0.55)
    for x, v in enumerate(conc["pct"]):
        ax.text(x, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_ylim(0, max(conc["pct"]) * 1.3)
    ax.set_ylabel("Observations on the decoy point (%)")
    ax.set_title("C. Same box, same period:\nonly the displaced population piles up",
                 fontsize=10.2)

    # D. Negative control: jamming-dominated regions yield no decoys.
    ax = axd["D"]
    rows = [("Baltic + Poland\n(jamming-dominated)", d["baltic"]),
            ("Levant\n(spoofing-dominated)", d["levant"])]
    x = np.arange(len(rows)); w = 0.36
    tot = [len(r[1]) for r in rows]
    dec = [int(r[1]["klass"].isin(DECOY).sum()) for r in rows]
    ax.bar(x - w/2, tot, w, color="#b9b9b9", label="candidate sites")
    ax.bar(x + w/2, dec, w, color="#b23a48", label="classified as decoys")
    for xi, (t, dd) in enumerate(zip(tot, dec, strict=False)):
        ax.text(xi - w/2, t, str(t), ha="center", va="bottom", fontsize=8.5)
        ax.text(xi + w/2, dd, str(dd), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8.6)
    ax.set_ylabel("Sites")
    ax.legend(fontsize=7.8)
    ax.set_title("D. Denial produces no fix, hence no decoy:\nthe method's blind spot, confirmed",
                 fontsize=10.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=PLOTS / "spoof06_controls.pdf")
    p.add_argument("--preview", type=Path, default=None)
    a = p.parse_args()
    make_figure(load_data(), a.output, a.preview)
    print(a.output)


if __name__ == "__main__":
    main()
