#!/usr/bin/env python3
"""Global atlas of GNSS-spoofing decoys detected in a crowdsourced cell database.

Runs the detector worldwide, agglomerates attractor squares into physical sites,
and classifies each. "Many cells at one coordinate" has several very different
causes; three measured quantities separate them:

  med_km         how far the site sits from the contributing cells' homes
  src_spread_km  how tightly those homes cluster (one region vs worldwide)
  lead_days      how long the platform had already known those cells before they
                 appeared here -- near zero means we merely crawled the region
                 for the first time, which is not a switch-on

  spoofing decoy    coherent source region, >12 km away, long lead time
  equipment/test    cell identities radiated by test gear; sources worldwide
  local aggregation site is essentially inside the source region
  dense venue/DAS   excluded upstream: those cells are *home* here, so they never
                    enter the displaced set at all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helpers import COUNTRIES_GEOJSON, draw_geojson_layer
from census2 import load as load_attractors, cluster, summarise, classify

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"

CLASS_COLOR = {
    "STATIC DECOY (cross-border)": "#8c1d2e",
    "STATIC DECOY": "#b23a48",
    "diffuse degradation": "#c9743a",
    "isotropic scatter": "#9ecae1",
    "identity replay": "#2f6f9f",
    "equipment/test": "#4f7f52",
    "local aggregation": "#b9b9b9",
    "persistent artifact": "#d9c98a",
    "crawl-onset (indeterminate)": "#7d7d7d",
}
DECOY_CLASSES = ("STATIC DECOY", "STATIC DECOY (cross-border)")


def load_data() -> pd.DataFrame:
    """Site-level census: attractor squares agglomerated into physical sites."""
    sites = summarise(cluster(load_attractors()))
    sites["klass"] = sites.apply(classify, axis=1)
    return sites


def make_figure(attr: pd.DataFrame, output: Path, preview: Path | None) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.02)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(12.4, 9.4), constrained_layout=True)
    axd = fig.subplot_mosaic([["A", "A"], ["B", "C"]],
                             height_ratios=[1.25, 1.0])
    fig.suptitle("Crowdsourced cell databases record GNSS spoofing worldwide",
                 fontsize=13.8, fontweight="bold")

    spoof = attr[attr["klass"].isin(DECOY_CLASSES)]

    # A. World atlas.
    ax = axd["A"]
    bbox = (-180, 180, -60, 78)
    ax.set_facecolor("#dceaf2")
    draw_geojson_layer(ax, COUNTRIES_GEOJSON, bbox, facecolor="#f5f1e8",
                       edgecolor="#9c948a", linewidth=0.25, zorder=0)
    for klass, color in CLASS_COLOR.items():
        sub = attr[attr["klass"] == klass]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"],
                   s=8 + np.sqrt(sub["cells"]) * 2.4,
                   color=color, alpha=0.9 if klass in DECOY_CLASSES else 0.45,
                   edgecolor="white", linewidth=0.35,
                   zorder=4 if klass in DECOY_CLASSES else 2,
                   label=f"{klass} (n={len(sub)})")
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9, ncols=2)
    ax.set_title(f"A. {len(attr):,} candidate sites; {len(spoof):,} static decoys "
                 f"({(attr['klass']=='STATIC DECOY (cross-border)').sum()} cross-border)",
                 fontsize=10.5)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # B. The classifier itself.
    ax = axd["B"]
    for klass, color in CLASS_COLOR.items():
        sub = attr[attr["klass"] == klass]
        if sub.empty:
            continue
        ax.scatter(sub["concentration"].clip(lower=0.005), sub["med_km"].clip(lower=1),
                   s=6 + np.sqrt(sub["cells"]) * 1.6, color=color,
                   alpha=0.65, edgecolor="none", label=klass)
    ax.axvline(0.10, color="#333", linestyle="--", linewidth=1.0)
    ax.axhline(12, color="#333", linestyle="--", linewidth=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Concentration: share of the site's displaced cells\non its single peak square (log)")
    ax.set_ylabel("Displacement: median home-to-attractor (km, log)")
    ax.set_title("B. Sharp decoys separate from diffuse displacement fields", fontsize=10.5)
    ax.legend(fontsize=6.4, frameon=True, loc="upper left", ncols=2)

    # C. Largest decoys.
    ax = axd["C"]
    top = spoof.nlargest(14, "cells").sort_values("cells")
    labels = [f"{r.lat:.2f}, {r.lon:.2f}" for r in top.itertuples()]
    ax.barh(labels, top["cells"], color="#b23a48")
    for patch, r in zip(ax.patches, top.itertuples(), strict=False):
        ax.text(r.cells * 1.04, patch.get_y() + patch.get_height() / 2,
                f"{int(r.cells):,}  ({int(r.days)}d)", va="center", fontsize=7.2)
    ax.set_xscale("log")
    ax.set_xlim(top["cells"].min() * 0.7, top["cells"].max() * 4)
    ax.set_xlabel("Distinct displaced cells (log)")
    ax.set_title("C. Largest decoys, with campaign duration", fontsize=10.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=PLOTS / "spoof02_global_atlas.pdf")
    p.add_argument("--preview", type=Path, default=None)
    a = p.parse_args()
    make_figure(load_data(), a.output, a.preview)
    print(a.output)


if __name__ == "__main__":
    main()
