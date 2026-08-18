#!/usr/bin/env python3
"""Plot the requested coordinated-coordinate-displacement examples.

All inputs are compact, auditable CSV extracts; this script does not query or
modify the remote database.  It produces two event atlases and one mechanism
diagnostic for the three previously unresolved destination attractors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from spoofing_category_overview import load_world, setup_map


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"
HQ = DATA / "high_quality"
FIGS = ROOT / "paper" / "figs"
WORLD = ROOT / "data" / "reference" / "ne_10m_admin_0_map_units.geojson"

BLUE = "#286a91"
RED = "#b5443c"
GREEN = "#3f7d5a"
PURPLE = "#76528b"
GOLD = "#b17a24"
INK = "#292724"
MUTED = "#716e68"
GRID = "#d8d5cf"


@dataclass(frozen=True)
class Event:
    number: str
    title: str
    subtitle: str
    source_lat: float
    source_lon: float
    dest_lat: float
    dest_lon: float
    identities: int
    onset: pd.Timestamp
    home_after: float
    color: str


def save(fig: plt.Figure, stem: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def route_bbox(event: Event, pad_fraction: float = 0.13) -> tuple[float, float, float, float]:
    west, east = sorted([event.source_lon, event.dest_lon])
    south, north = sorted([event.source_lat, event.dest_lat])
    xpad = max(0.18, (east - west) * pad_fraction)
    ypad = max(0.14, (north - south) * pad_fraction)
    return west - xpad, east + xpad, south - ypad, north + ypad


def route_panel(ax: plt.Axes, rings, event: Event, *, labels: bool = True) -> None:
    setup_map(ax, rings, route_bbox(event), equal=False)
    arrow = FancyArrowPatch(
        (event.source_lon, event.source_lat), (event.dest_lon, event.dest_lat),
        arrowstyle="-|>", mutation_scale=8, connectionstyle="arc3,rad=.10",
        linewidth=1.35, color=event.color, alpha=.86, zorder=3,
    )
    ax.add_patch(arrow)
    ax.scatter(event.source_lon, event.source_lat, s=23, color=BLUE,
               edgecolor="white", linewidth=.45, zorder=4)
    ax.scatter(event.dest_lon, event.dest_lat, s=30, marker="X", color=RED,
               edgecolor="white", linewidth=.45, zorder=4)
    if labels:
        ax.annotate("home", (event.source_lon, event.source_lat), xytext=(3, 3),
                    textcoords="offset points", fontsize=5.1, color=INK)
        ax.annotate("destination", (event.dest_lon, event.dest_lat), xytext=(3, -7),
                    textcoords="offset points", fontsize=5.1, color=INK)
    ax.set_title(f"{event.number}  {event.title}", loc="left", fontsize=7.2,
                 fontweight="bold", pad=3)
    ax.text(.01, .01, event.subtitle, transform=ax.transAxes, fontsize=5.4,
            color=MUTED, va="bottom", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=1.2))
    ax.tick_params(labelsize=4.7)


def event_members() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "all_event_members.csv")
    for c in ["onset_ts", "home_first_seen", "home_last_seen", "t_first_away", "t_last_away"]:
        frame[c] = pd.to_datetime(frame[c], errors="coerce")
    return frame


def regional_events(frame: pd.DataFrame) -> list[tuple[Event, pd.DataFrame]]:
    specs = [
        ("1", "Vladimir mass displacement", "17--19 Oct 2024 · 560 identities · 5 PLMNs",
         ["561_403_2024-10-17", "561_403_2024-10-18", "561_403_2024-10-19"],
         56.10, 40.30, 56.14, 40.94, GREEN),
        ("2", "St Petersburg reassignment", "2 May 2026 · 125 LTE identities · 5 PLMNs",
         [f"600_30{x}_2026-05-02" for x in range(3, 7)],
         60.04, 30.45, 60.11, 31.30, PURPLE),
        ("7a", "Udmurtia regional event", "5 Dec 2025 · 18 identities · 4 PLMNs",
         ["564_537_2025-12-05"], 56.40, 53.70, 56.15, 54.28, GOLD),
        ("7b", "Krasnodar repeated event", "23 Nov 2024 & 1 Apr 2025 · 26 identity-events",
         ["458_401_2024-11-23", "458_401_2025-04-01"],
         45.80, 40.10, 45.625, 40.80, RED),
    ]
    output = []
    for number, title, subtitle, ids, slat, slon, dlat, dlon, color in specs:
        group = frame[frame.event_id.isin(ids)].copy()
        unique = group.drop_duplicates(["mcc", "mnc", "lac", "cid", "cell_type"])
        home_after = unique.home_last_seen.ge(unique.onset_ts + pd.Timedelta(days=7)).mean()
        output.append((Event(number, title, subtitle, slat, slon, dlat, dlon,
                             len(unique), group.onset_ts.min(), home_after, color), group))
    return output


def onset_panel(ax: plt.Axes, event: Event, members: pd.DataFrame) -> None:
    times = members.drop_duplicates(["mcc", "mnc", "lac", "cid", "cell_type"])["onset_ts"]
    if event.number == "1":
        bins = pd.date_range("2024-10-17 15:30", "2024-10-19 20:30", freq="1h")
        label = "onset hour (UTC)"
    elif event.number == "7b":
        days = times.dt.floor("D").value_counts().sort_index()
        ax.bar(np.arange(len(days)), days.values, color=event.color, width=.58)
        ax.set_xticks(np.arange(len(days)), [d.strftime("%b %Y") for d in days.index], fontsize=5)
        label = "onset date"
        bins = None
    else:
        start = times.min().floor("h") - pd.Timedelta(minutes=15)
        end = times.max().ceil("h") + pd.Timedelta(minutes=15)
        bins = pd.date_range(start, end + pd.Timedelta(minutes=15), freq="15min")
        label = "onset time (UTC)"
    if bins is not None:
        ax.hist(times, bins=bins, color=event.color, edgecolor="white", linewidth=.3)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.set_xlabel(label, fontsize=5.2)
    ax.set_ylabel("identities", fontsize=5.2)
    ax.tick_params(labelsize=4.7, length=2)
    ax.grid(axis="y", color=GRID, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    termination = 100 * (1 - event.home_after)
    ax.text(.98, .96, f"home active >7 d: {event.home_after:.0%}\n"
            f"home terminated: {termination:.0f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.3, color=INK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=1.1))


def regional_figure(rings) -> None:
    events = regional_events(event_members())
    fig = plt.figure(figsize=(7.15, 7.45))
    outer = fig.add_gridspec(2, 2, hspace=.30, wspace=.20)
    for index, (event, members) in enumerate(events):
        cell = outer[index // 2, index % 2].subgridspec(
            2, 1, height_ratios=[1.25, .58], hspace=.23
        )
        route_panel(fig.add_subplot(cell[0]), rings, event)
        onset_panel(fig.add_subplot(cell[1]), event, members)
    fig.suptitle("Regional coordinated coordinate-displacement campaigns",
                 fontsize=9.3, fontweight="bold", x=.07, ha="left", y=.995)
    save(fig, "coordinated_displacement_regional")


def long_range_events() -> list[tuple[Event, pd.DataFrame, str]]:
    method = pd.read_csv(HQ / "method2_members.csv")
    method["onset_ts"] = pd.to_datetime(method.onset_ts)
    vietnam = method[(method.onset_day == "2024-09-03") & method.src_lat10.isin([214, 215])
                     & (method.dest_lat5 == 2265) & (method.dest_lon5 == 10460)].copy()
    caucasus = method[(method.onset_day == "2025-06-20") & (method.src_lat10 == 448)
                      & (method.src_lon10 == 414) & (method.dest_lat5 == 4425)].copy()
    all_members = event_members()
    shaanxi = all_members[all_members.event_id.eq("385_1098_2026-02-12")].copy()
    dynamic = pd.read_csv(DATA / "dynamic_validated_members.csv")
    dynamic = dynamic[dynamic.campaign_id.eq("DYN1")].copy()
    dynamic["month"] = pd.to_datetime(dynamic.month)
    events = [
        (Event("3", "Northern Vietnam jump--return", "3 Sep 2024 · 11 identities · 3 PLMNs",
               21.49, 105.11, 22.66, 104.615, 11, vietnam.onset_ts.min(), 1.0, GREEN),
         vietnam, "method"),
        (Event("4", "China Mobile destination switch", "Feb 2025 · 15/16 shared identities · 1,202 km",
               32.30, 106.80, 21.70, 109.20, 15, pd.Timestamp("2025-02-01"), np.nan, PURPLE),
         dynamic, "dynamic"),
        (Event("5", "North Caucasus to NE China", "20 Jun 2025 · 5 LTE identities · exact onset",
               44.83, 41.48, 44.27, 129.82, 5, caucasus.onset_ts.min(), 1.0, RED),
         caucasus, "method"),
        (Event("6", "Northern to southern Shaanxi", "12 Feb 2026 · 9 LTE identities · 4 PLMNs",
               38.52, 109.87, 33.15, 107.19, 9, shaanxi.onset_ts.min(), 5 / 9, GOLD),
         shaanxi, "event"),
    ]
    return events


def long_diagnostic(ax: plt.Axes, event: Event, members: pd.DataFrame, kind: str) -> None:
    exact_onset = False
    if kind == "dynamic":
        monthly = members.groupby(["month", "dest_lat10", "dest_lon10"])[
            ["mcc", "mnc", "lac", "cid", "cell_type"]
        ].size().rename("rows").reset_index()
        for (lat, lon), g in monthly.groupby(["dest_lat10", "dest_lon10"]):
            label = f"{lat/10:.1f}°N, {lon/10:.1f}°E"
            ax.plot(g.month, g.rows, marker="o", ms=3, lw=1.1, label=label)
        ax.legend(fontsize=4.6, frameon=False, loc="upper left")
        ax.set_ylabel("identity-months", fontsize=5.2)
        ax.set_xlabel("destination assignment", fontsize=5.2)
    else:
        times = pd.to_datetime(members.onset_ts)
        if times.nunique() == 1:
            exact_onset = True
            ax.scatter(times, np.arange(1, len(times) + 1), s=12, color=event.color)
            ax.set_xlim(times.iloc[0] - pd.Timedelta(minutes=8),
                        times.iloc[0] + pd.Timedelta(minutes=8))
            ax.set_ylabel("identity", fontsize=5.2)
        else:
            span = max(pd.Timedelta(minutes=15), times.max() - times.min())
            freq = "15min" if span < pd.Timedelta(hours=6) else "1h"
            bins = pd.date_range(times.min().floor(freq), times.max().ceil(freq) + pd.Timedelta(freq), freq=freq)
            ax.hist(times, bins=bins, color=event.color, edgecolor="white", linewidth=.3)
            ax.set_ylabel("identities", fontsize=5.2)
        if event.number == "3":
            note = "all return home\n91% home-interval overlap"
        elif event.number == "5":
            note = "all retain home\nall exact at 10:22:26 UTC"
        else:
            note = "5/9 retain home\nlower-confidence common change"
        ax.text(.98, .96, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=5.2, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=1.1))
        ax.set_xlabel("onset time (UTC)", fontsize=5.2)
    if exact_onset:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y") if kind == "dynamic"
                                     else mdates.DateFormatter("%H:%M"))
    ax.tick_params(labelsize=4.7, length=2)
    ax.grid(axis="y", color=GRID, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)


def long_range_figure(rings) -> None:
    fig = plt.figure(figsize=(7.15, 7.45))
    outer = fig.add_gridspec(2, 2, hspace=.30, wspace=.20)
    for index, (event, members, kind) in enumerate(long_range_events()):
        cell = outer[index // 2, index % 2].subgridspec(
            2, 1, height_ratios=[1.25, .58], hspace=.23
        )
        route_panel(fig.add_subplot(cell[0]), rings, event)
        long_diagnostic(fig.add_subplot(cell[1]), event, members, kind)
    fig.suptitle("Long-range coordinated coordinate-displacement campaigns",
                 fontsize=9.3, fontweight="bold", x=.07, ha="left", y=.995)
    save(fig, "coordinated_displacement_long_range")


def attractor_figure(rings) -> None:
    summary = pd.read_csv(DATA / "unresolved_attractor_summary.csv").set_index("attractor")
    identities = pd.read_csv(DATA / "unresolved_attractor_identity_audit.csv")
    batches = pd.read_csv(DATA / "unresolved_attractor_batches.csv")
    batches["timestamp"] = pd.to_datetime(batches.timestamp)
    positions = pd.read_csv(DATA / "chengde_cid22812_positions.csv")
    specs = [
        ("chengde", "A  Chengde / Kuancheng", 40.62, 118.48, "CID/LAC aliasing", GREEN),
        ("valdai", "B  Demyansk / Valdai", 57.67, 32.53, "batched constellation replay", PURPLE),
        ("smolensk_npp", "C  Smolensk NPP", 54.17, 33.23, "batched constellation replay", RED),
    ]
    fig = plt.figure(figsize=(7.15, 5.1))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, .82], hspace=.30, wspace=.25)
    for col, (key, title, dlat, dlon, mechanism, color) in enumerate(specs):
        group = identities[identities.attractor.eq(key)]
        ax = fig.add_subplot(grid[0, col])
        west = min(group.home_lon.min(), dlon)
        east = max(group.home_lon.max(), dlon)
        south = min(group.home_lat.min(), dlat)
        north = max(group.home_lat.max(), dlat)
        xpad, ypad = max(1.0, (east-west)*.07), max(.7, (north-south)*.10)
        setup_map(ax, rings, (west-xpad, east+xpad, south-ypad, north+ypad))
        ax.scatter(group.home_lon, group.home_lat, s=6, alpha=.42, color=BLUE,
                   linewidth=0, label="plurality home")
        ax.scatter(dlon, dlat, s=38, marker="X", color=color, edgecolor="white",
                   linewidth=.5, zorder=4, label="attractor")
        ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", pad=3)
        ax.tick_params(labelsize=4.7, length=2)
        row = summary.loc[key]
        ax.text(.02, .02, f"{int(row.identities)} identities · {int(row.distinct_cids)} CIDs\n"
                f"{int(row.plmns)} PLMNs · {int(row.destination_observations):,} destination obs",
                transform=ax.transAxes, fontsize=5.2, color=INK, va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=1.2))

        bx = fig.add_subplot(grid[1, col])
        if key == "chengde":
            top = positions.head(8).sort_values("identities")
            labels = [f"{r.lat:.2f}, {r.lon:.2f}" for r in top.itertuples()]
            colors = [color if (r.lat == dlat and r.lon == dlon) else "#8ca5b2" for r in top.itertuples()]
            bx.barh(np.arange(len(top)), top.identities, color=colors)
            bx.set_yticks(np.arange(len(top)), labels, fontsize=4.4)
            bx.set_xlabel("LACs sharing GSM CID 22812", fontsize=5.2)
            bx.text(.98, .05, "16,592 LACs nationwide\n3,912 reported positions",
                    transform=bx.transAxes, ha="right", va="bottom", fontsize=5.1)
        else:
            bg = batches[batches.attractor.eq(key)]
            significant = bg[bg.identities.ge(2)]
            bx.scatter(significant.timestamp, significant.identities,
                       s=7 + significant.identities * 1.1, color=color, alpha=.65,
                       linewidth=0)
            bx.set_ylabel("identities at exact timestamp", fontsize=5.2)
            bx.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            bx.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
            top_sources = group.nlargest(min(30, len(group)), "destination_observations")
            source_name = "Krasny Sulin" if key == "valdai" else "Penza"
            bx.text(.02, .96, f"compact {source_name} constellation\n"
                    f"largest exact batch: {int(row.largest_exact_batch)} identities",
                    transform=bx.transAxes, ha="left", va="top", fontsize=5.1)
        bx.set_title(mechanism, loc="left", fontsize=6, color=color, pad=2)
        bx.tick_params(labelsize=4.6, length=2)
        bx.grid(axis="x" if key == "chengde" else "y", color=GRID, linewidth=.5)
        bx.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mechanism audit of three large destination attractors",
                 fontsize=9.3, fontweight="bold", x=.07, ha="left", y=.995)
    save(fig, "unresolved_attractor_mechanisms")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    rings = load_world(WORLD)
    regional_figure(rings)
    long_range_figure(rings)
    attractor_figure(rings)
    print("wrote coordinated displacement figures")


if __name__ == "__main__":
    main()
