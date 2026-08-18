#!/usr/bin/env python3
"""Render the additional 404/01 TAC-0 compound-associated clusters."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from plot_criminal_compound_quad import imagery, scale_bar

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity"
FIGS = ROOT / "paper" / "figs"
COLORS = {"CT12": "#d27b27", "BM03": "#4e79a7", "BA23": "#8b6bb1"}
SUSPECTED = {"CT12": (10.9784, 105.0417), "BA23": (11.0752, 106.1512)}
SUSPECTED_SITES = [
    ("BA04", 11.0742, 106.1640), ("BA07", 11.0680, 106.1698), ("BA10", 11.0676, 106.1614),
    ("BA19", 11.0788, 106.1638), ("BA21", 11.0607, 106.1791), ("BA22", 11.0623, 106.1776),
    ("BA23", 11.0752, 106.1512), ("BT01", 13.1974, 102.4016), ("CH02", 14.3405, 104.0587),
    ("CT06", 10.9594, 105.0607), ("CT07", 10.9555, 105.0594), ("CT09", 10.9564, 105.0599),
    ("CT10", 10.9536, 105.0593), ("CT11", 10.9420, 105.0543), ("CT12", 10.9784, 105.0417),
    ("KCC01", 11.8855, 104.7094), ("PO02", 13.6521, 102.5582), ("PO11", 13.6635, 102.5562),
    ("PP02", 11.4530, 104.8561), ("PP11", 11.4107, 104.6514), ("PP14", 11.5985, 104.8683),
    ("PP21", 11.6048, 104.8691), ("PSP06", 12.9227, 102.4968), ("SI05", 10.5964, 103.6315),
    ("SI07", 10.6099, 103.5349), ("SI10", 10.6162, 103.5652), ("SI14", 10.6323, 103.5068),
    ("SI17", 10.6246, 103.5009), ("SI18", 10.6250, 103.4977), ("SI23", 10.6150, 103.5271),
    ("SI31", 10.6058, 103.5287), ("SI38", 10.6214, 103.5126), ("SI41", 10.6212, 103.5019),
    ("SV01", 10.9829, 106.1954), ("SV02", 10.9292, 106.1441), ("TBK01", 11.6419, 105.8159),
]


def main():
    ids = pd.read_csv(DATA / "cambodia_40401_lac0_identities.csv")
    sites = pd.read_csv(DATA / "cambodia_compounds_86.csv").set_index("site_id")
    specs = [("CT12", "suspected", 15), ("BM03", "confirmed", 15), ("BA23", "suspected", 15)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.35))
    fig.subplots_adjust(left=.025, right=.995, top=.88, bottom=.025, hspace=.16, wspace=.035)
    axes = axes.ravel()
    for ax, (site, status, zoom) in zip(axes[:3], specs):
        group = ids[ids.cluster.eq(site)]
        if site in sites.index:
            s = sites.loc[site]
            lat, lon = float(s.latitude), float(s.longitude)
        else:
            lat, lon = SUSPECTED[site]
        if site == "BA23":
            # The suspected registry coordinate is ~12.7 km from the lone
            # observed identity; center this panel on the observed point.
            lat, lon = float(group.glat.mean()), float(group.glon.mean())
        half = .0035
        bbox = (lon-half, lon+half, lat-half, lat+half)
        imagery(ax, bbox, zoom)
        ax.scatter(group.glon, group.glat, s=26, color=COLORS[site], edgecolor="white", linewidth=.55, alpha=.95, zorder=5)
        ax.scatter([lon], [lat], marker="x", s=36, color="white", linewidth=1.0, zorder=7)
        ax.scatter([lon], [lat], marker="x", s=22, color="#171b1c", linewidth=.7, zorder=8)
        obs = int(group.obs.sum())
        ax.set_title(f"{site} · {status}\n404/01 · TAC-0 · {len(group)} IDs · {obs:,} obs.", fontsize=6.0, fontweight="bold", pad=2)
        ax.text(.012, .014, "cross: compound centre", transform=ax.transAxes, fontsize=4.2, color="white", bbox={"facecolor":"black", "alpha":.62, "pad":1.0})
        scale_bar(ax, bbox)
        ax.set_box_aspect(1)
    # National overview: every registry location, with confirmed and suspected
    # entries separated by marker shape.
    ax = axes[3]
    bbox = (102.0, 108.0, 10.0, 15.0)
    imagery(ax, bbox, 7)
    ax.scatter(sites.longitude, sites.latitude, s=8, marker="x", color="#d1495b", linewidth=.45, alpha=.8, zorder=4)
    suspected = pd.DataFrame(SUSPECTED_SITES, columns=["site_id", "latitude", "longitude"])
    ax.scatter(suspected.longitude, suspected.latitude, s=8, marker="+", color="#f2b134", linewidth=.55, alpha=.9, zorder=5)
    ax.set_title("Cambodia-wide compound registry", fontsize=5.6, fontweight="bold", pad=2)
    ax.text(.02, .03, "× confirmed   + suspected", transform=ax.transAxes, fontsize=4.0, color="white", bbox={"facecolor":"black", "alpha":.62, "pad":1.0})
    ax.set_box_aspect(1)
    pdf = FIGS / "criminal_compound_additional.pdf"
    png = FIGS / "criminal_compound_additional.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=180)
    Image.open(png).convert("RGB").save(png)


if __name__ == "__main__":
    main()
