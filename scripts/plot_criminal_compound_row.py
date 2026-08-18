#!/usr/bin/env python3
"""Render the seven Cambodia compound panels as one grouped row."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from plot_criminal_compound_quad import imagery, scale_bar, COLORS
from plot_criminal_compound_additional import SUSPECTED

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity"
FIGS = ROOT / "paper" / "figs"


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42})
    ids = pd.read_csv(DATA / "cambodia_40401_lac0_identities.csv")
    sites = pd.read_csv(DATA / "cambodia_compounds_86.csv").set_index("site_id")
    specs = [
        ("KA03", "Bokor Mountain", 14, "confirmed"),
        ("KA01", "Bokor Mountain", 15, "confirmed"),
        ("BA01", "Bavet", 15, "confirmed"),
        ("TD01H", "Koh Kong", 16, "confirmed"),
        ("BM03", "", 15, "confirmed"),
        ("CT12", "", 15, "suspicious"),
        ("BA23", "", 15, "suspicious"),
    ]
    colors = {"confirmed": "#2f8150", "suspicious": "#d27b27"}
    fig, axes = plt.subplots(1, 7, figsize=(7.0, 1.55), gridspec_kw={"wspace": 0.012})
    fig.subplots_adjust(left=.002, right=.998, bottom=.01, top=.78)
    for ax, (site, place, zoom, status) in zip(axes, specs):
        group = ids[ids.cluster.eq(site)]
        if site in sites.index:
            lat, lon = float(sites.loc[site, "latitude"]), float(sites.loc[site, "longitude"])
        else:
            lat, lon = SUSPECTED[site]
        if site == "BA23":
            lat, lon = float(group.glat.mean()), float(group.glon.mean())
        half = .0045 if site not in {"TD01H", "CT12", "BA23"} else (.0025 if site == "TD01H" else .0035)
        bbox = (lon-half, lon+half, lat-half, lat+half)
        imagery(ax, bbox, zoom)
        ax.scatter(group.glon, group.glat, s=14, color=COLORS.get(site, "#4e79a7"),
                   edgecolor="white", linewidth=.4, alpha=.95, zorder=5)
        ax.scatter([lon], [lat], marker="x", s=18, color="white", linewidth=.7, zorder=7)
        ax.scatter([lon], [lat], marker="x", s=12, color="#171b1c", linewidth=.5, zorder=8)
        obs = int(group.obs.sum())
        # Keep the row legible at one-column width; the shared caption gives
        # the common PLMN/TAC configuration.
        ax.set_title(f"{site}\n{len(group)} IDs · {obs:,} obs.", fontsize=4.7,
                     fontweight="bold", pad=1.2)
        ax.text(.015, .015, "cross: centre", transform=ax.transAxes, fontsize=3.2,
                color="white", bbox={"facecolor": "black", "alpha": .62, "pad": .5})
        scale_bar(ax, bbox)
        for spine in ax.spines.values():
            spine.set_color(colors[status]); spine.set_linewidth(1.4)
    fig.text(.285, .985, "CONFIRMED SCAM COMPOUNDS", ha="center", va="top", fontsize=5.8,
             fontweight="bold", color=colors["confirmed"])
    fig.text(.855, .985, "SUSPICIOUS LOCATIONS", ha="center", va="top", fontsize=5.8,
             fontweight="bold", color=colors["suspicious"])
    fig.add_artist(plt.Line2D([.713, .713], [.03, .90], transform=fig.transFigure,
                              color="#777", linewidth=.7, linestyle="--"))
    fig.text(.998, .002, "Imagery © Esri and contributors", ha="right", va="bottom", fontsize=3.2, color="#666")
    pdf = FIGS / "criminal_compound_row.pdf"
    png = FIGS / "criminal_compound_row.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    Image.open(png).convert("RGB").save(png)


if __name__ == "__main__":
    main()
