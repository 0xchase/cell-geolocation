#!/usr/bin/env python3
"""Create the satellite/D2C analysis figure rows from derived local CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "satellites"
FIGS = ROOT / "paper" / "figs"
BOUNDARIES = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"

COHORT_COLORS = {
    "early_v1": "#2878b5",
    "late_v2": "#7b3294",
    "nonmatching": "#c4572e",
}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    pdf = FIGS / f"{name}.pdf"
    png = FIGS / f"{name}.png"
    fig.savefig(pdf, dpi=360, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"[figure] {pdf}")


def world_geometries():
    collection = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    return [shape(feature["geometry"]) for feature in collection["features"]]


def cohort_order(phenotypes: pd.DataFrame) -> list[str]:
    frame = phenotypes[phenotypes["implementation_cohort"].isin(COHORT_COLORS)].copy()
    frame["cohort_rank"] = frame["implementation_cohort"].map({"early_v1": 0, "late_v2": 1, "nonmatching": 2})
    frame["evidence_rank"] = frame["evidence_tier"].map({"linked_candidate": 0, "unresolved_signature": 1, "assignment_conflict": 2}).fillna(3)
    frame = frame.sort_values(["cohort_rank", "evidence_rank", "bulk_onset", "plmn"])
    return frame["plmn"].tolist()


def latest_points(plmns: list[str]) -> pd.DataFrame:
    frames = []
    for plmn in plmns:
        path = DATA / "cells" / f"{plmn}.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path, compression="gzip", usecols=["latest_lat", "latest_lon"])
        valid = (
            frame["latest_lat"].between(-90, 90)
            & frame["latest_lon"].between(-180, 180)
            & ~((frame["latest_lat"] == 0) & (frame["latest_lon"] == 0))
        )
        frames.append(frame.loc[valid])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["latest_lat", "latest_lon"])


def coverage_points(plmns: list[str], phenotypes: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    assigned = phenotypes.set_index("plmn")["country_iso"].to_dict()
    points = coverage[coverage["plmn"].isin(plmns)].copy()
    points["assigned_country_iso"] = points["plmn"].map(assigned)
    return points[points["country_iso"] == points["assigned_country_iso"]]


def starlink_global_map(phenotypes: pd.DataFrame, coverage: pd.DataFrame) -> None:
    inventory = pd.read_csv(DATA / "plmn_inventory.csv")
    linked_starlink = phenotypes[
        phenotypes["attribution_confidence"].eq("source_linked_plus_fingerprint")
        & phenotypes["likely_system"].fillna("").str.contains("Starlink")
    ]
    starlink_coverage = coverage_points(linked_starlink["plmn"].tolist(), phenotypes, coverage)
    starlink_direct = latest_points(
        inventory.loc[inventory["assignee"].fillna("").str.contains("Starlink|SpaceX", case=False), "plmn"].tolist()
    )

    categories = [
        {
            "label": "Starlink / SpaceX",
            "color": "#6a3d9a",
            "marker": "o",
            "frames": [
                starlink_coverage.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]],
                starlink_direct.rename(columns={"latest_lat": "lat", "latest_lon": "lon"})[["lat", "lon"]],
            ],
        },
        {"label": "Telstra D2D candidate", "color": "#1b9e77", "marker": "o", "coverage": ["505-11"]},
        {"label": "Spark NZ D2D candidate", "color": "#66a61e", "marker": "o", "coverage": ["530-02"]},
        {
            "label": "Unattributed D2C-like",
            "color": "#e6ab02",
            "marker": "o",
            "coverage": phenotypes.loc[phenotypes["attribution_confidence"].eq("fingerprint_only"), "plmn"].tolist(),
        },
        {"label": "China SatNet", "color": "#d95f02", "marker": "^", "latest": ["460-04"]},
        {"label": "Bureau 1440 conflict", "color": "#7570b3", "marker": "^", "latest": ["901-70"]},
        {"label": "Intelsat", "color": "#e7298a", "marker": "^", "latest": ["901-94"]},
        {"label": "Sateliot", "color": "#a6761d", "marker": "^", "latest": ["901-97"]},
        {"label": "Skylo", "color": "#1f78b4", "marker": "^", "latest": ["901-98"]},
        {"label": "Thuraya", "color": "#b15928", "marker": "^", "latest": ["901-05"]},
        {"label": "AeroMobile", "color": "#17becf", "marker": "x", "latest": ["901-14"]},
        {"label": "OnAir / SITA", "color": "#9467bd", "marker": "x", "latest": ["901-15"]},
        {"label": "MCP maritime", "color": "#2ca02c", "marker": "x", "latest": ["901-12"]},
        {"label": "WMS maritime", "color": "#8c564b", "marker": "x", "latest": ["901-18"]},
    ]

    for category in categories:
        frames = category.get("frames", [])
        if "coverage" in category:
            frame = coverage_points(category["coverage"], phenotypes, coverage)
            frames.append(frame.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
        if "latest" in category:
            frame = latest_points(category["latest"])
            frames.append(frame.rename(columns={"latest_lat": "lat", "latest_lon": "lon"})[["lat", "lon"]])
        category["points"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["lat", "lon"])

    fig = plt.figure(figsize=(7.0, 3.00))
    projection = ccrs.Robinson()
    plate = ccrs.PlateCarree()
    ax = fig.add_axes([0.01, 0.25, 0.98, 0.73], projection=projection)
    ax.set_global()
    ax.set_facecolor("#e8f1f5")
    ax.add_geometries(world_geometries(), crs=plate, facecolor="#f4f0e6", edgecolor="#89847c", linewidth=0.18)
    for category in categories:
        part = category["points"]
        if part.empty:
            continue
        marker = category["marker"]
        ax.scatter(
            part["lon"], part["lat"], s=1.15 if marker == "o" else 4.0,
            c=category["color"], marker=marker, alpha=0.48 if marker == "o" else 0.82,
            linewidths=0.35 if marker == "x" else 0, rasterized=True,
            transform=plate, zorder=2 if marker == "o" else 3,
        )
    handles = []
    for category in categories:
        if category["points"].empty:
            continue
        marker = category["marker"]
        handles.append(Line2D(
            [0], [0], marker=marker, linestyle="none", markersize=3.4,
            markerfacecolor=category["color"] if marker != "x" else "none",
            markeredgecolor=category["color"], markeredgewidth=0.65,
            label=category["label"],
        ))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=4,
              frameon=True, facecolor="white", edgecolor="#bbbbbb", fontsize=5.5,
              columnspacing=0.8, handletextpad=0.25, borderpad=0.35)
    ax.set_title("Database-inferred satellite-associated cellular observations", pad=2)
    ax.spines["geo"].set_linewidth(0.4)
    ax.spines["geo"].set_edgecolor("#777777")
    save(fig, "satellite_world_map")


def identifier_fingerprint(phenotypes: pd.DataFrame, tracks: pd.DataFrame) -> None:
    order = cohort_order(phenotypes)
    tracks = tracks[tracks["plmn"].isin(order)]
    high_range = np.arange(11072, 11749)
    high_index = {value: i for i, value in enumerate(high_range)}
    high = np.zeros((len(order), len(high_range)), dtype=np.uint8)
    low = np.zeros((len(order), 256), dtype=np.uint8)
    for row_index, plmn in enumerate(order):
        part = tracks[tracks["plmn"] == plmn]
        for value in part["enodeb_id"].unique():
            if value in high_index:
                high[row_index, high_index[value]] = 1
        low[row_index, part["cell_slot"].unique().astype(int)] = 1

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.62), gridspec_kw={"width_ratios": [2.55, 1.0], "wspace": 0.09})
    cmap = mpl.colors.ListedColormap(["#f1eee7", "#542788"])
    axes[0].imshow(high, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    axes[1].imshow(low, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("(a) eNodeB-ID occupancy")
    axes[1].set_title("(b) Cell-slot occupancy")
    axes[0].set_xticks([0, 168, 338, 507, 676], ["11,072", "11,240", "11,410", "11,579", "11,748"])
    axes[1].set_xticks([0, 63, 127, 191, 255])
    axes[0].set_yticks(range(len(order)), order)
    axes[1].set_yticks([])
    axes[0].set_xlabel("High 20 bits of LTE ECI")
    axes[1].set_xlabel("Low 8 bits")
    phenotype_index = phenotypes.set_index("plmn")
    for i, plmn in enumerate(order):
        color = COHORT_COLORS[phenotype_index.loc[plmn, "implementation_cohort"]]
        axes[0].get_yticklabels()[i].set_color(color)
    for ax in axes:
        ax.tick_params(length=2, pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.35)
    save(fig, "satellite_identifier_fingerprint")


def similarity_heatmap(phenotypes: pd.DataFrame, similarity: pd.DataFrame) -> None:
    order = cohort_order(phenotypes)
    index = {plmn: i for i, plmn in enumerate(order)}
    matrices = {"enodeb_containment": np.full((len(order), len(order)), np.nan),
                "eci_containment": np.full((len(order), len(order)), np.nan)}
    for _, row in similarity.iterrows():
        if row["plmn_a"] not in index or row["plmn_b"] not in index:
            continue
        i, j = index[row["plmn_a"]], index[row["plmn_b"]]
        for key, matrix in matrices.items():
            matrix[i, j] = matrix[j, i] = row[key]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), gridspec_kw={"wspace": 0.10})
    cmap = LinearSegmentedColormap.from_list("containment", ["#f5f1e8", "#b2abd2", "#542788"])
    images = []
    for ax, (key, title) in zip(axes, [("enodeb_containment", "(a) eNodeB-ID containment"), ("eci_containment", "(b) Complete-ECI containment")]):
        image = ax.imshow(matrices[key], vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
        images.append(image)
        ax.set_title(title)
        ax.set_xticks(range(len(order)), order, rotation=90)
        ax.set_yticks(range(len(order)), order if ax is axes[0] else [])
        ax.tick_params(length=0, pad=1)
    cbar = fig.colorbar(images[-1], ax=axes, fraction=0.025, pad=0.018)
    cbar.set_label("Intersection / smaller set")
    save(fig, "satellite_identifier_similarity")


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values.dropna().to_numpy(dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values) if len(values) else np.array([])


def eci_reuse(phenotypes: pd.DataFrame, tracks: pd.DataFrame) -> None:
    frame = phenotypes[phenotypes["implementation_cohort"].isin(["early_v1", "late_v2"])].copy()
    frame = frame.sort_values("lte_tac_eci_rows", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.52), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.30})

    y = np.arange(len(frame))
    axes[0].barh(y, frame["lte_tac_eci_rows"], color="#c2b3d8", label="TAC–ECI rows")
    axes[0].barh(y, frame["unique_lte_ecis"], color="#542788", label="Unique PLMN–ECIs")
    axes[0].set_yticks(y, frame["plmn"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Count (log scale)")
    axes[0].set_title("(a) Corrected identity counts")
    axes[0].legend(loc="lower right", frameon=False)

    highlight = ["302-723", "234-02", "401-04", "530-13", "255-707"]
    palette = dict(zip(highlight, ["#2166ac", "#762a83", "#1b7837", "#d6604d", "#b2182b"]))
    for plmn in highlight:
        part = tracks[tracks["plmn"] == plmn]
        x, yy = ecdf(part["n_tacs"])
        axes[1].step(x, yy, where="post", color=palette[plmn], label=plmn, linewidth=1.15)
        x, yy = ecdf(part["latest_tac_span_km"])
        axes[2].step(x, yy, where="post", color=palette[plmn], label=plmn, linewidth=1.15)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.9, None)
    axes[1].set_xlabel("TACs per ECI (log)")
    axes[1].set_ylabel("Fraction of ECIs")
    axes[1].set_title("(b) TAC multiplicity")
    axes[2].set_xscale("symlog", linthresh=1)
    axes[2].set_xlabel("Latest-position span (km)")
    axes[2].set_title("(c) ECI geographic reuse")
    axes[2].legend(loc="lower right", frameon=False, ncol=1)
    for ax in axes[1:]:
        ax.set_ylim(0, 1.01)
        ax.grid(axis="y", linewidth=0.3, color="#dddddd")
    save(fig, "satellite_eci_reuse")


def visibility_timeline(phenotypes: pd.DataFrame, daily: pd.DataFrame) -> None:
    order = cohort_order(phenotypes)
    daily = daily[daily["plmn"].isin(order)].copy()
    start = pd.Timestamp("2024-05-01")
    end = daily["date"].max()
    dates = pd.date_range(start, end, freq="D")
    matrix = np.full((len(order), len(dates)), np.nan)
    for row_index, plmn in enumerate(order):
        part = daily[daily["plmn"] == plmn].set_index("date")
        values = part["active_lte_ecis"].reindex(dates)
        peak = values.max()
        if pd.notna(peak) and peak > 0:
            matrix[row_index] = values / peak

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.62), gridspec_kw={"width_ratios": [4.7, 1.0], "wspace": 0.12})
    cmap = LinearSegmentedColormap.from_list("visibility", ["#f4f0e7", "#c2a5cf", "#542788"])
    cmap.set_bad("white")
    extent = [mdates.date2num(dates[0]), mdates.date2num(dates[-1]), len(order) - 0.5, -0.5]
    image = axes[0].imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1, extent=extent)
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_title("(a) Daily crawler-visible LTE ECIs, normalized within PLMN")
    axes[0].tick_params(length=2, pad=1)
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.018, pad=0.012)
    cbar.ax.tick_params(labelsize=5.5, pad=1)

    stats = phenotypes.set_index("plmn").reindex(order)
    axes[1].scatter(stats["exact_once_day_fraction"], range(len(order)), s=10,
                    c=[COHORT_COLORS.get(value, "#777777") for value in stats["implementation_cohort"]], linewidths=0)
    axes[1].axvline(1, color="#555555", linewidth=0.5)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(len(order) - 0.5, -0.5)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Days with one query\nper active TAC–cell")
    axes[1].set_title("(b) Collection cadence")
    axes[1].grid(axis="x", linewidth=0.3, color="#dddddd")
    save(fig, "satellite_visibility_timeline")


def add_country_geometries(ax, plate, extent):
    ax.set_extent(extent, crs=plate)
    ax.set_facecolor("#e8f1f5")
    ax.add_geometries(world_geometries(), crs=plate, facecolor="#f4f0e6", edgecolor="#77736c", linewidth=0.30)
    ax.coastlines(resolution="10m", linewidth=0.25, color="#77736c")


def tac_geography(coverage: pd.DataFrame) -> None:
    plate = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.52), subplot_kw={"projection": plate}, gridspec_kw={"wspace": 0.08})
    panels = [
        (axes[0], ["440-55", "440-25", "440-26"], [127.0, 147.0, 29.0, 46.5], "(a) Japan: three partner PLMNs"),
        (axes[1], ["530-13", "530-02", "530-07"], [165.0, 179.5, -48.5, -33.0], "(b) New Zealand: reused TAC geography"),
    ]
    colors = ["#2166ac", "#b2182b", "#1b7837"]
    markers = ["o", "^", "s"]
    for ax, plmns, extent, title in panels:
        add_country_geometries(ax, plate, extent)
        for plmn, color, marker in zip(plmns, colors, markers):
            part = coverage[coverage["plmn"] == plmn]
            ax.scatter(part["longitude"], part["latitude"], s=8, marker=marker, facecolors="none",
                       edgecolors=color, linewidths=0.55, alpha=0.70, transform=plate, label=plmn,
                       rasterized=True, zorder=3)
        ax.set_title(title)
        ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#bbbbbb", ncol=len(plmns))
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#cccccc", alpha=0.7)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {"size": 5.5}
        gl.ylabel_style = {"size": 5.5}
    save(fig, "satellite_tac_geography")


def identifier_occupancy(phenotypes: pd.DataFrame, tracks: pd.DataFrame) -> None:
    late = phenotypes[phenotypes["implementation_cohort"] == "late_v2"]["plmn"].tolist()
    frame = tracks[tracks["plmn"].isin(late)][["plmn", "eci", "enodeb_id"]].drop_duplicates()
    enodeb = frame[["plmn", "enodeb_id"]].drop_duplicates().groupby("enodeb_id")["plmn"].nunique()
    eci = frame.groupby("eci")["plmn"].nunique()
    core = set(enodeb[enodeb >= max(2, len(late) - 3)].index)
    per_plmn = []
    for plmn in late:
        values = set(frame.loc[frame["plmn"] == plmn, "enodeb_id"])
        per_plmn.append({"plmn": plmn, "core_containment": len(values & core) / len(core) if core else np.nan,
                         "enodebs": len(values), "ecis": frame.loc[frame["plmn"] == plmn, "eci"].nunique()})
    per_plmn = pd.DataFrame(per_plmn).sort_values("core_containment")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.4, 1.0, 1.35], "wspace": 0.30})
    axes[0].plot(enodeb.index, enodeb.values, linestyle="none", marker=".", markersize=2.2, color="#542788", rasterized=True)
    axes[0].set_xlim(11060, 11760)
    axes[0].set_ylim(0, len(late) + 0.5)
    axes[0].set_xlabel("eNodeB ID")
    axes[0].set_ylabel("Later-cohort PLMNs containing ID")
    axes[0].set_title("(a) Shared logical-node core")

    counts = eci.value_counts().sort_index()
    axes[1].bar(counts.index, counts.values, color="#8073ac", width=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("PLMNs sharing complete ECI")
    axes[1].set_ylabel("ECIs (log scale)")
    axes[1].set_title("(b) Full-ECI reuse")

    axes[2].barh(range(len(per_plmn)), per_plmn["core_containment"], color="#542788")
    axes[2].set_yticks(range(len(per_plmn)), per_plmn["plmn"])
    axes[2].set_xlim(0, 1.02)
    axes[2].set_xlabel("Fraction of shared core present")
    axes[2].set_title("(c) Sparse-population attribution")
    for ax in axes:
        ax.grid(axis="y", linewidth=0.3, color="#dddddd")
    save(fig, "satellite_identifier_occupancy")


def main() -> None:
    configure()
    phenotypes = pd.read_csv(DATA / "plmn_phenotypes.csv", parse_dates=["first_seen", "last_seen", "bulk_onset"])
    tracks = pd.read_csv(DATA / "ecgi_tracks.csv.gz", compression="gzip", usecols=[
        "plmn", "eci", "enodeb_id", "cell_slot", "n_tacs", "latest_tac_span_km"
    ])
    coverage = pd.read_csv(DATA / "tac_coverage.csv.gz", compression="gzip")
    similarity = pd.read_csv(DATA / "identifier_similarity.csv")
    daily = pd.read_csv(DATA / "daily_coverage.csv", parse_dates=["date"])
    starlink_global_map(phenotypes, coverage)
    identifier_fingerprint(phenotypes, tracks)
    similarity_heatmap(phenotypes, similarity)
    eci_reuse(phenotypes, tracks)
    visibility_timeline(phenotypes, daily)
    tac_geography(coverage)
    identifier_occupancy(phenotypes, tracks)


if __name__ == "__main__":
    main()
