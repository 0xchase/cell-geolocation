#!/usr/bin/env python3
"""Plot the Cambodia foreign-PLMN compound-cluster evidence."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from plot_helpers import add_osm_basemap


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity"
FIGS = ROOT / "paper" / "figs"

COLORS = {
    "KA03": "#c23b4a",
    "TD01H": "#2b6f9e",
    "BA01": "#2f8150",
    "KA01": "#77519a",
    "CT12": "#d27b27",
    "other": "#9aa2a6",
}
INK = "#242a2d"
MUTED = "#667177"


def basemap(ax, bbox, zoom):
    ax.set_facecolor("#e8eef0")
    add_osm_basemap(
        ax, bbox, zoom=zoom, source="carto_light", alpha=0.88,
        grayscale=True, grayscale_brightness=1.04, grayscale_contrast=0.92,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#7a8387")
        spine.set_linewidth(0.5)


def plot_points(ax, identities, *, zoomed=False):
    for cluster, group in identities.groupby("cluster"):
        color = COLORS.get(cluster, COLORS["other"])
        size = 18 if zoomed else 8
        ax.scatter(
            group.glon, group.glat, s=size, color=color,
            edgecolor="white", linewidth=0.35, alpha=0.90,
            zorder=5, label=cluster,
        )


def add_cluster_labels(ax, sites, *, zoomed=False):
    labels = ["KA03", "KA01", "KA04", "TD01H", "BA01", "CT12"]
    for site in labels:
        row = sites[sites.site_id.eq(site)]
        if row.empty:
            continue
        x, y = float(row.longitude.iloc[0]), float(row.latitude.iloc[0])
        if zoomed and not (103.99 <= x <= 104.07 and 10.61 <= y <= 10.69):
            continue
        ax.scatter([x], [y], marker="x", s=18 if zoomed else 10,
                   color=INK, linewidth=0.7, zorder=7)
        ax.annotate(site, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=5.5 if zoomed else 4.5, color=INK,
                    fontweight="bold", zorder=8)


def make_figure(output: Path, preview: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    ids = pd.read_csv(DATA / "cambodia_40401_lac0_identities.csv")
    monthly = pd.read_csv(DATA / "cambodia_40401_cluster_monthly.csv")
    enrichment = pd.read_csv(DATA / "cambodia_40401_compound_enrichment.csv")
    confirmed = pd.read_csv(DATA / "cambodia_compounds_86.csv")
    suspected = pd.DataFrame([
        ("BA04", 11.0742, 106.1640), ("BA07", 11.0680, 106.1698),
        ("BA10", 11.0676, 106.1614), ("BA19", 11.0788, 106.1638),
        ("BA21", 11.0607, 106.1791), ("BA22", 11.0623, 106.1776),
        ("BA23", 11.0752, 106.1512), ("BT01", 13.1974, 102.4016),
        ("CH02", 14.3405, 104.0587), ("CT06", 10.9594, 105.0607),
        ("CT07", 10.9555, 105.0594), ("CT09", 10.9564, 105.0599),
        ("CT10", 10.9536, 105.0593), ("CT11", 10.9420, 105.0543),
        ("CT12", 10.9784, 105.0417), ("KCC01", 11.8855, 104.7094),
        ("PO02", 13.6521, 102.5582), ("PO11", 13.6635, 102.5562),
        ("PP02", 11.4530, 104.8561), ("PP11", 11.4107, 104.6514),
        ("PP14", 11.5985, 104.8683), ("PP21", 11.6048, 104.8691),
        ("PSP06", 12.9227, 102.4968), ("SI05", 10.5964, 103.6315),
        ("SI07", 10.6099, 103.5349), ("SI10", 10.6162, 103.5652),
        ("SI14", 10.6323, 103.5068), ("SI17", 10.6246, 103.5009),
        ("SI18", 10.6250, 103.4977), ("SI23", 10.6150, 103.5271),
        ("SI31", 10.6058, 103.5287), ("SI38", 10.6214, 103.5126),
        ("SI41", 10.6212, 103.5019), ("SV01", 10.9829, 106.1954),
        ("SV02", 10.9292, 106.1441), ("TBK01", 11.6419, 105.8159),
    ], columns=["site_id", "latitude", "longitude"])

    fig = plt.figure(figsize=(7.05, 7.0))
    gs = fig.add_gridspec(3, 2, hspace=0.34, wspace=0.16,
                          left=0.035, right=0.985, top=0.965, bottom=0.045)

    # (a) National view.
    ax = fig.add_subplot(gs[0, 0])
    basemap(ax, (102.0, 108.0, 10.0, 15.0), 7)
    ax.set_title("(a) Cambodia-wide distribution", loc="left", fontsize=7, fontweight="bold", pad=3)
    ax.scatter(confirmed.longitude, confirmed.latitude, s=4, marker="x",
               color="#6d767b", alpha=0.55, linewidth=0.35, zorder=2)
    ax.scatter(suspected.longitude, suspected.latitude, s=5, marker="+",
               color="#b7a16b", alpha=0.65, linewidth=0.45, zorder=2)
    plot_points(ax, ids)
    add_cluster_labels(ax, pd.concat([confirmed, suspected]), zoomed=False)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[k], markersize=4, label=k)
        for k in ["KA03", "TD01H", "BA01", "KA01", "CT12"]
    ], loc="lower left", fontsize=4.7, frameon=True, framealpha=0.88,
              borderpad=0.35, handletextpad=0.25, labelspacing=0.2, ncol=2)

    # (b) Bokor zoom.
    ax = fig.add_subplot(gs[0, 1])
    basemap(ax, (103.99, 104.07, 10.61, 10.69), 13)
    ax.set_title("(b) Bokor Mountain cluster", loc="left", fontsize=7, fontweight="bold", pad=3)
    local_ids = ids[(ids.glon.between(103.99, 104.07)) & (ids.glat.between(10.61, 10.69))]
    plot_points(ax, local_ids, zoomed=True)
    local_sites = pd.concat([confirmed, suspected])
    add_cluster_labels(ax, local_sites, zoomed=True)
    # 250 m reference circles (longitude scale corrected at this latitude).
    for site, color in [("KA03", COLORS["KA03"]), ("KA01", COLORS["KA01"])]:
        row = confirmed[confirmed.site_id.eq(site)].iloc[0]
        ax.add_patch(Circle((row.longitude, row.latitude), 250 / (111320 * np.cos(np.radians(row.latitude))),
                            fill=False, color=color, linewidth=0.8, linestyle="--", alpha=0.9, zorder=6))
    ax.text(0.02, 0.03, "dashed circles: 250 m", transform=ax.transAxes, fontsize=4.7, color=MUTED)

    # (c) Temporal cluster growth and attrition.
    ax = fig.add_subplot(gs[1, :])
    monthly.month = pd.to_datetime(monthly.month)
    for cluster in ["KA03", "TD01H", "BA01", "KA01"]:
        group = monthly[monthly.cluster.eq(cluster)].sort_values("month")
        if group.empty:
            continue
        ax.plot(group.month, group.active_identities, color=COLORS[cluster], linewidth=1.35,
                marker="o", markersize=2.2, label=cluster)
    ax.set_title("(c) Number of active 404/01–TAC-0 identities by month", loc="left", fontsize=7, fontweight="bold", pad=3)
    ax.set_ylabel("active identities")
    ax.grid(axis="y", color="#d7dcde", linewidth=0.45)
    ax.tick_params(axis="both", labelsize=6)
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=5.7, handlelength=1.6)
    ax.spines[["top", "right"]].set_visible(False)

    # (d) Enrichment and compact cluster table.
    ax = fig.add_subplot(gs[2, 0])
    x = np.arange(len(enrichment))
    width = 0.34
    ax.bar(x - width / 2, enrichment.observed_identities, width, color="#c23b4a", label="observed")
    ax.bar(x + width / 2, enrichment.matched_expected, width, color="#b9c0c3", label="matched expectation")
    ax.set_xticks(x, [f"≤{int(r)} m" for r in enrichment.radius_m])
    ax.set_ylabel("identities")
    ax.set_title("(d) Spatial enrichment vs. matched Cambodian cells", loc="left", fontsize=7, fontweight="bold", pad=3)
    ax.grid(axis="y", color="#d7dcde", linewidth=0.45)
    ax.tick_params(axis="both", labelsize=5.7)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    for i, p in enumerate(enrichment.one_sided_p):
        ax.text(i, enrichment.observed_identities.iloc[i] + 1.1,
                f"p={p:.1e}", ha="center", va="bottom", fontsize=4.8, color=INK)

    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    ax.set_title("(e) Primary site-confined clusters", loc="left", fontsize=7, fontweight="bold", pad=3)
    rows = [
        ["KA03", "confirmed", "20", "4,031", "47–129"],
        ["TD01H", "confirmed", "6", "1,317", "67–106"],
        ["BA01", "confirmed", "3", "361", "131–155"],
        ["KA01", "confirmed", "2", "292", "184–243"],
        ["CT12", "suspected", "4", "456", "31–136"],
    ]
    table = ax.table(cellText=rows, colLabels=["site", "status", "IDs", "obs.", "m"],
                     cellLoc="center", colLoc="center", loc="upper left", bbox=[0, 0.15, 1, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd1d3")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#eef1f2")
            cell.set_text_props(weight="bold", color=INK)
        elif col == 0:
            cell.set_text_props(weight="bold", color=COLORS.get(rows[row - 1][0], INK))
    ax.text(0, 0.06, "m = nearest distance to site center; IDs and observations are database counts.",
            transform=ax.transAxes, fontsize=4.4, color=MUTED)

    fig.text(0.985, 0.012, "Basemap © OpenStreetMap contributors, © CARTO", ha="right", va="bottom", fontsize=3.5, color="#666")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure(FIGS / "criminal_compound_clusters.pdf", FIGS / "criminal_compound_clusters.png")
