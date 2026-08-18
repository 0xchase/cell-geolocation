#!/usr/bin/env python3
"""Render the Cambodia-wide compound registry map as a standalone figure."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from plot_criminal_compound_quad import imagery
from plot_criminal_compound_additional import SUSPECTED_SITES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity"
FIGS = ROOT / "paper" / "figs"


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
    sites = pd.read_csv(DATA / "cambodia_compounds_86.csv")
    suspected = pd.DataFrame(SUSPECTED_SITES, columns=["site_id", "latitude", "longitude"])
    fig, ax = plt.subplots(figsize=(3.35, 3.0))
    fig.subplots_adjust(left=.02, right=.99, top=.96, bottom=.02)
    imagery(ax, (102.0, 108.0, 10.0, 15.0), 7)
    ax.scatter(sites.longitude, sites.latitude, s=16, marker="x", color="#d1495b",
               linewidth=.65, alpha=.85, zorder=4, label="Confirmed")
    ax.scatter(suspected.longitude, suspected.latitude, s=18, marker="+", color="#f2b134",
               linewidth=.8, alpha=.95, zorder=5, label="Suspicious")
    ax.set_title("Cambodia-wide compound registry", fontsize=8, fontweight="bold", pad=3)
    ax.legend(loc="lower left", fontsize=6, framealpha=.8, handlelength=1.2)
    ax.set_box_aspect(1)
    fig.text(.998, .002, "Imagery © Esri and contributors", ha="right", va="bottom", fontsize=4, color="#666")
    pdf = FIGS / "criminal_compound_map.pdf"
    png = FIGS / "criminal_compound_map.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    Image.open(png).convert("RGB").save(png)


if __name__ == "__main__":
    main()
