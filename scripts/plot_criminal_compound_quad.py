#!/usr/bin/env python3
"""Render full-colour imagery panels for the four primary compound clusters."""

from pathlib import Path
import math

import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pyproj import Transformer
from PIL import Image
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity"
FIGS = ROOT / "paper" / "figs"
COLORS = {"KA03": "#c23b4a", "TD01H": "#2b6f9e", "BA01": "#2f8150", "KA01": "#77519a"}


def imagery(ax, bbox, zoom):
    west, east, south, north = bbox
    ax.set_facecolor("#35463f")
    try:
        image, extent = cx.bounds2img(west, south, east, north, zoom=zoom,
                                      source=cx.providers.Esri.WorldImagery, ll=True)
        transform = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        x0, y0 = transform.transform(extent[0], extent[2])
        x1, y1 = transform.transform(extent[1], extent[3])
        ax.imshow(image, extent=(x0, x1, y0, y1), origin="upper", interpolation="bilinear", zorder=0)
    except Exception as exc:
        print(f"imagery tile error: {exc}")
    ax.set_xlim(west, east); ax.set_ylim(south, north)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#202020"); spine.set_linewidth(.65)


def scale_bar(ax, bbox):
    west, east, south, north = bbox
    mid = (south + north) / 2
    width_km = (east - west) * 111.32 * math.cos(math.radians(mid))
    km = max(v for v in (.1, .2, .5, 1, 2) if v <= max(width_km * .23, .1))
    span = km / (111.32 * math.cos(math.radians(mid)))
    x1 = east - .055 * (east - west); x0 = x1 - span; y = north - .07 * (north - south)
    line = ax.plot([x0, x1], [y, y], color="white", lw=2)[0]
    line.set_path_effects([path_effects.Stroke(linewidth=3.4, foreground="black"), path_effects.Normal()])
    txt = ax.text((x0+x1)/2, y-.02*(north-south), f"{km:g} km", color="white", fontsize=5, ha="center", va="top")
    txt.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground="black"), path_effects.Normal()])


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    ids = pd.read_csv(DATA / "cambodia_40401_lac0_identities.csv")
    monthly = pd.read_csv(DATA / "cambodia_40401_cluster_monthly.csv")
    enrichment = pd.read_csv(DATA / "cambodia_40401_compound_enrichment.csv")
    sites = pd.read_csv(DATA / "cambodia_compounds_86.csv").set_index("site_id")
    specs = [("KA03", "Bokor Mountain", 14), ("KA01", "Bokor Mountain", 15),
             ("BA01", "Bavet", 15), ("TD01H", "Koh Kong", 16)]
    # A compact 2×2 one-column item, matching the paper's facility quad style.
    fig = plt.figure(figsize=(3.35, 3.55))
    gs = fig.add_gridspec(2, 2, hspace=.28, wspace=.10, left=.035, right=.985, top=.965, bottom=.035)

    for index, (site, place, zoom) in enumerate(specs):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        s = sites.loc[site]; group = ids[ids.cluster.eq(site)]
        lat, lon = float(s.latitude), float(s.longitude)
        # A compact square window exposes buildings, roads, and the cell cloud.
        half = .0045 if site != "TD01H" else .0025
        bbox = (lon-half, lon+half, lat-half, lat+half)
        imagery(ax, bbox, zoom)
        ax.scatter(group.glon, group.glat, s=24, color=COLORS[site], edgecolor="white", linewidth=.55, alpha=.95, zorder=5)
        ax.scatter([lon], [lat], marker="x", s=36, color="white", linewidth=1.0, zorder=7)
        ax.scatter([lon], [lat], marker="x", s=22, color="#171b1c", linewidth=.7, zorder=8)
        obs = int(group.obs.sum())
        ax.set_title(f"{site} · {place}\n404/01 · TAC-0 · {len(group)} IDs · {obs:,} obs.", fontsize=5.7, fontweight="bold", pad=2)
        ax.text(.012, .014, "cross: compound centre", transform=ax.transAxes, fontsize=4.1, color="white", bbox={"facecolor":"black", "alpha":.62, "pad":1.0})
        scale_bar(ax, bbox)
    fig.text(.985, .012, "Imagery © Esri and contributors", ha="right", va="bottom", fontsize=4.2, color="#666")
    fig.savefig(FIGS / "criminal_compound_quad.pdf", bbox_inches="tight")
    png_path = FIGS / "criminal_compound_quad.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    # pdflatex can mishandle RGBA raster images; flatten to an RGB PNG for paper inclusion.
    Image.open(png_path).convert("RGB").save(png_path)


if __name__ == "__main__":
    main()
