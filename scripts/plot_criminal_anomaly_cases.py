#!/usr/bin/env python3
"""Plot the three retained criminal-activity cellular anomalies."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_helpers import add_osm_basemap


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rogue-bts-detector"
FIGS = ROOT / "paper" / "figs"

INK = "#252a2e"
MUTED = "#687178"
FOREIGN = "#b33d4a"
MACAU = "#7a55a5"
SHAN = "#2e6f9e"
LAND = "#eef1ed"


def square_bbox(lon: np.ndarray, lat: np.ndarray, padding: float = 1.22) -> tuple[float, float, float, float]:
    lon_mid = float((lon.min() + lon.max()) / 2)
    lat_mid = float((lat.min() + lat.max()) / 2)
    cos_lat = max(math.cos(math.radians(lat_mid)), 0.25)
    width = float(lon.max() - lon.min()) * 111.32 * cos_lat
    height = float(lat.max() - lat.min()) * 111.32
    span = max(width, height, 8.0) * padding
    return (
        lon_mid - span / (2 * 111.32 * cos_lat),
        lon_mid + span / (2 * 111.32 * cos_lat),
        lat_mid - span / (2 * 111.32),
        lat_mid + span / (2 * 111.32),
    )


def zoom_for(bbox: tuple[float, float, float, float]) -> int:
    span = max((bbox[1] - bbox[0]) * 111.32, (bbox[3] - bbox[2]) * 111.32)
    if span > 180:
        return 7
    if span > 50:
        return 10
    return 14


def setup_map(ax: plt.Axes, bbox: tuple[float, float, float, float], title: str) -> None:
    ax.set_facecolor(LAND)
    add_osm_basemap(
        ax, bbox, zoom=zoom_for(bbox), source="carto_voyager",
        alpha=0.88, grayscale=True, grayscale_brightness=1.03,
        grayscale_contrast=1.04,
    )
    ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
    ax.set_box_aspect(1); ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="bold", fontsize=7, pad=3)
    for spine in ax.spines.values():
        spine.set_color("#777d80"); spine.set_linewidth(0.55)


def lacs_strip(ax: plt.Axes, lacs: list[int], xmin: int, xmax: int, title: str, color: str, *, pattern: bool = False) -> None:
    present = set(int(x) for x in lacs)
    xs = np.arange(xmin, xmax + 1)
    vals = np.array([x in present for x in xs], dtype=float)
    ax.imshow(vals[np.newaxis, :], aspect="auto", interpolation="nearest",
              extent=(xmin - 0.5, xmax + 0.5, 0, 1), cmap="Greys", vmin=0, vmax=1)
    ax.set_xlim(xmin - 0.5, xmax + 0.5); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.set_xticks([xmin, xmax]); ax.tick_params(axis="x", labelsize=4, length=2, pad=1)
    ax.set_title(title, fontsize=5.3, loc="left", color=MUTED, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.35); spine.set_color("#9ba1a4")
    # A second row shows the observed LAC sequence for the patterned family.
    if pattern:
        ax.text(0.99, 0.15, "arithmetic LAC spacing", transform=ax.transAxes,
                ha="right", va="center", fontsize=4.2, color=FOREIGN)


def make_figure(output: Path, preview: Path | None) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })
    shan = pd.read_csv(DATA / "eastern_shan_foreign_plmn_positions.csv")
    raw = pd.read_csv(DATA / "static_criminal_anomaly_raw_history.csv")
    este = raw[raw.case_label.eq("estepona_kw_419_02_gsm_cid1971")]
    krug = raw[raw.case_label.eq("krugersdorp_zm_645_02_gsm_cid2730")]

    fig = plt.figure(figsize=(7.05, 3.45))
    outer = fig.add_gridspec(1, 3, wspace=0.18, left=0.018, right=0.995, top=0.92, bottom=0.12)
    panels = [
        ("Eastern Shan · foreign LTE system", shan.longitude.to_numpy(), shan.latitude.to_numpy()),
        ("Estepona · Kuwait-coded family", este.lon.to_numpy(), este.lat.to_numpy()),
        ("Krugersdorp · Zambia-coded family", krug.lon.to_numpy(), krug.lat.to_numpy()),
    ]
    axes = []
    for i, (title, lon, lat) in enumerate(panels):
        gs = outer[i].subgridspec(2, 1, height_ratios=[5.4, 1], hspace=0.20)
        ax = fig.add_subplot(gs[0]); strip = fig.add_subplot(gs[1]); axes.append(ax)
        setup_map(ax, square_bbox(lon, lat), title)
        if i == 0:
            hk = shan[(shan.mcc == 454) & (shan.mnc == 3)]
            mo = shan[(shan.mcc == 455) & (shan.mnc == 1)]
            ax.scatter(hk.longitude, hk.latitude, s=2.0, c=SHAN, alpha=0.28, linewidths=0, rasterized=True)
            ax.scatter(mo.longitude, mo.latitude, s=3.2, c=MACAU, alpha=0.62, linewidths=0, rasterized=True)
            # Emphasize the seven independently extracted moving identities.
            moving = pd.read_csv(DATA / "case_raw_history.csv.gz", compression="gzip")
            moving = moving[moving.case_label.eq("eastern_shan_hk_454_03_lte_lac12596")]
            for _, g in moving.groupby(["mcc", "mnc", "lac", "cid", "cell_type"]):
                g = g.sort_values("timestamp")
                ax.plot(g.lon, g.lat, color=FOREIGN, linewidth=0.45, alpha=0.58, zorder=4)
                ax.scatter(g.lon, g.lat, s=3.0, color=FOREIGN, edgecolor="white", linewidth=0.15, zorder=5)
            strip.set_xlim(0, 1); strip.set_ylim(0, 1); strip.axis("off")
            strip.add_patch(plt.Rectangle((0.01, 0.30), 0.72, 0.38, color=SHAN, alpha=0.85))
            strip.add_patch(plt.Rectangle((0.74, 0.30), 0.25, 0.38, color=MACAU, alpha=0.85))
            strip.text(0.37, 0.49, "Hong Kong 454/03 · 939 CIDs · TAC 12596", ha="center", va="center", fontsize=4.35, color="white", fontweight="bold")
            strip.text(0.865, 0.49, "455/01 · 185 CIDs", ha="center", va="center", fontsize=4.0, color="white", fontweight="bold")
            strip.text(0.01, 0.05, "Distinct foreign-coded LTE identities in the regional footprint", ha="left", va="bottom", fontsize=4.1, color=MUTED)
        elif i == 1:
            ax.scatter(este.lon, este.lat, s=5.2, color=FOREIGN, alpha=0.34, linewidth=0, zorder=4)
            ax.scatter([este.lon.mean()], [este.lat.mean()], s=14, facecolor="none", edgecolor=INK, linewidth=0.7, zorder=5)
            lacs = sorted(pd.read_csv(DATA / "foreign_far_candidate_members.csv").query("phenomenon_id == 'FBS-06'").lac.unique())
            lacs_strip(strip, lacs, 1, 99, "Kuwait 419/02 · CID 1971 · 83 of 99 LACs", FOREIGN)
        else:
            ax.scatter(krug.lon, krug.lat, s=5.2, color=FOREIGN, alpha=0.34, linewidth=0, zorder=4)
            ax.scatter([krug.lon.mean()], [krug.lat.mean()], s=14, facecolor="none", edgecolor=INK, linewidth=0.7, zorder=5)
            lacs = sorted(pd.read_csv(DATA / "foreign_far_candidate_members.csv").query("phenomenon_id == 'FBS-07'").lac.unique())
            strip.clear()
            strip.plot(np.arange(len(lacs)), lacs, color=FOREIGN, linewidth=0.8, marker="o", markersize=1.8)
            strip.set_title("Zambia 645/02 · CID 2730 · 80 LACs in a patterned sequence", fontsize=5.3, loc="left", color=MUTED, pad=1)
            strip.set_xlabel("Sorted identity rank", fontsize=4.0, labelpad=1)
            strip.set_ylabel("LAC", fontsize=4.0, labelpad=1)
            strip.tick_params(axis="both", labelsize=4, length=2, pad=1)
            strip.grid(axis="y", color="#d5d9da", linewidth=0.35)
            for spine in strip.spines.values():
                spine.set_linewidth(0.35); spine.set_color("#9ba1a4")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SHAN, markersize=4, alpha=0.7, label="Hong Kong-coded LTE positions"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MACAU, markersize=4, alpha=0.7, label="Macao-coded LTE positions"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FOREIGN, markersize=4, alpha=0.7, label="Static-family positions / sampled paths"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.015), ncol=3,
               frameon=False, fontsize=4.6, handletextpad=0.25, columnspacing=0.8)
    fig.text(0.995, 0.018, "Basemap © OpenStreetMap contributors, © CARTO", ha="right", va="bottom", fontsize=3.4, color="#666")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure(FIGS / "criminal_anomaly_case_studies.pdf", FIGS / "criminal_anomaly_case_studies.png")
