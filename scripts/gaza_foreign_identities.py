#!/usr/bin/env python3
"""Gaza: foreign network identities in the Strip -- where they sit, and when.

Replaces the (identity x year) density grid, which spent a full page on 16 maps
of which 10 were near-empty, to say two things that do not need a map each:
foreign identities sit in one small patch, and they exist only in 2024. Density
encoded as colour is also simply hard to read; magnitude read off a log axis is
not. This version is ~1/5 of a page.

Two panels, each answering the question it is actually good at:

* **(a) where** -- one map over the whole corpus. Each cell is a dot at its
  last known position; a cell that relocated further than MOVE_KM also gets a
  segment back to where it started, so a mark is never placed somewhere the
  cell was never seen. An earlier version plotted the *mean* of each cell's
  positions, which for the 38-54 percent of cells that move is a coordinate
  that was never observed. Domestic MCC 425 is drawn faint as context so the
  reader can tell the cluster is a specific place rather than simply where
  everyone is; Egypt and the non-adjacent foreign identities are drawn on top.
* **(b) when / how many** -- distinct cells per month on a log axis. Relative
  magnitude is position, not colour, so a 3-cell series and a 3,000-cell series
  are readable in the same frame.

The panels share one colour assignment, so panel (b)'s labels double as panel
(a)'s legend and no legend box is needed. Palette is the validated categorical
set (dataviz `validate_palette.js`, light, all-pairs): worst CVD separation
9.0 dE, worst normal-vision 17.6 dE, all passing. The single contrast warning
(Egypt's amber vs the surface) is discharged by the direct labels, which is the
relief that check asks for.

**The April 2024 step is ours, not Gaza's.** Every series jumps that month
(Israeli 1,267 -> 3,717; Palestinian 4 -> 2,305; Egypt 25 -> 230; non-adjacent
foreign 0 -> 388) because the crawl's scope expanded -- a large set of PLMNs
carry a first-seen of 2024-04-03. The onset of foreign identities therefore
cannot be dated from this corpus, and the chart marks the step so it is not read
as an event on the ground. What is not an artifact is the decay: non-adjacent
foreign falls 389 -> 103 across 2024 and then sits at 3-4 every month from Jan
2025 to Jun 2026, while the other three series continue at full strength. (The
Apr 2026 dip in the Israeli line is a one-month crawl gap, not a change.)

Distinct cells, never `obs`: `obs` counts our API poll cadence rather than
sightings, so an observation-weighted series would chart our own crawl schedule.

Zeros are gaps, not points, on a log axis -- Palestinian PLMNs are absent from
the corpus until 2024 (425/05 first seen globally 2024-01-01, 425/06 2024-03-10)
and the non-adjacent foreign series is genuinely 0 in Mar 2024.

Output split, and why it differs from `ukraine_progression.py`: the map is a
raster (basemap plus thousands of marks) so it is written as a bare PNG with no
typography, exactly like the Ukraine panels. The timeline is a dozen vector
strokes and its text -- rasterising it would throw away crispness for no size
win -- so it is written as a PDF, in a serif face so it sits with the paper's
Times. All panel labels and the caption are set in LaTeX; `--emit-tex` prints
the figure body.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from PIL import Image as PILImage

from plot_helpers import setup_context_map, add_osm_basemap, draw_geojson_layer
from plot_helpers import ADMIN1_GEOJSON, TILE_ATTRIBUTION
from gaza_country_identity import (
    CELLS_QUERY,
    GAZA,
    MONTHS,
    _cached,
    audit_mccs,
    classify,
)


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"

MAP_DPI = 300
MAP_IN = 1.61            # square; 0.23\linewidth at 7.0in
TL_W, TL_H = 5.10, 1.61  # timeline; 0.73\linewidth

# One colour per identity, shared by both panels. Validated all-pairs.
SERIES = [
    ("il", "Israeli",             "#2f6f9f"),
    ("ps", "Palestinian",         "#b23a48"),
    ("eg", "Egypt",               "#E69F00"),
    ("ff", "Non-adjacent foreign", "#009E73"),
]
COLOR = {k: c for k, _, c in SERIES}

# Non-country codes, excluded so they are not counted as foreign operators.
# Kept in sync with NON_COUNTRY_MCC in the companion module.
EXCLUDED_MCC = (1, 69, 526, 901, 999)

# Raster basemap. Not plain OSM: those tiles label this area in Arabic and
# Hebrew, which is wrong for an English paper, and the labels are baked into the
# raster at whatever size the tile renderer chose. The no-labels styles supply
# roads, built-up areas and coastline only, and the place names are set here
# instead at a size picked for the printed panel.
#
# Voyager rather than Positron, and drawn at full opacity: Positron is a
# deliberately near-monochrome style and at alpha 0.95 the result read as washed
# out -- the Mediterranean in particular came through as flat grey rather than
# water. Voyager keeps land, water and roads distinguishable while staying well
# below the plotted marks in chroma.
# Tiles are cached under .cache/tiles_<source>/, so this stays offline-capable
# after the first run and falls back to the vector basemap if the CDN is down.
BASEMAP = "carto_voyager_nolabels"
BASEMAP_ZOOM = 12
BASEMAP_ALPHA = 1.0

# Kept to the three the strip is usually described by; more than this collides
# at 1.61in. Coordinates are the town centres.
CITIES = [
    ("Gaza City", 34.466, 31.507),
    ("Khan Yunis", 34.303, 31.342),
    ("Rafah", 34.256, 31.288),
]

# The month the crawl's scope expanded; every series steps here.
CRAWL_STEP = pd.Timestamp("2024-04-01")

# A cell is not a point. Apple returns an estimate per poll, so a cell that is
# re-sited shows up as a sequence of positions -- and plotting the mean of that
# sequence, as the earlier version did, puts the mark somewhere the cell was
# never seen. These two constants separate real relocation from estimator noise:
#
# SNAP_DEG collapses positions onto a ~200 m lattice before the track is built.
# Median per-cell spread is 0.22 km, i.e. below Apple's own positional
# uncertainty, and the raw sequences are bimodal -- half the cells have one or
# two positions, the rest jitter on nearly every poll (p90 = 451 distinct
# positions). Without the snap the map is a hairball of noise.
#
# MOVE_KM is then the span a cell must cover across its snapped stops before a
# line is drawn instead of a dot. 6,826 of 17,842 cells clear it.
SNAP_DEG = 0.002
MOVE_KM = 1.0
# A position within PLACE_KM of a place's running centroid belongs to that
# place. Tied to MOVE_KM on purpose: if a displacement under MOVE_KM is not
# a move, then two positions closer than MOVE_KM are not two places. Loosening
# it to 0.5 km triples the vertex count without resolving anything the map can
# show at this size (median places per mover 6 -> 14).
PLACE_KM = MOVE_KM

# Tracks are drawn only for the identities this figure is about. Domestic MCC
# 425 supplies 6,572 movers -- at panel size they cover the map in a hairball
# and bury both the basemap and the foreign cluster, and "Israeli cells are
# re-sited a lot" is a separate result that belongs with the moving-base-station
# figure, not here. Domestic cells stay as faint context dots.
TRACK_GROUPS = {"eg", "ff"}
KM_PER_DEG_LAT = 110.9
KM_PER_DEG_LON = 95.2  # at 31.4 N

# Every position each Gaza cell was ever seen at -- not only the in-bbox ones,
# so a track that leaves the frame is drawn leaving it rather than being
# truncated at the boundary and read as if it stopped there. Bucketed by year,
# because the moving panel is built per year: a cell's itinerary has to be
# clustered from that year's positions alone, not sliced out of an all-time
# track afterwards.
TRACKS_QUERY = f"""
SELECT g.mcc AS mcc, g.mnc AS mnc, g.lac AS lac, g.cid AS cid,
       g.cell_type AS cell_type, toYear(g.timestamp) AS yr,
       g.lat AS lat, g.lon AS lon,
       min(g.timestamp) AS t0, count() AS obs
FROM cell.geos AS g
INNER JOIN (
    SELECT DISTINCT mcc, mnc, lac, cid, cell_type
    FROM cell.geos
    WHERE lat BETWEEN {GAZA[2]} AND {GAZA[3]}
      AND lon BETWEEN {GAZA[0]} AND {GAZA[1]}
      AND cid > 0 AND NOT (lat = 0 AND lon = 0)
) AS s USING (mcc, mnc, lac, cid, cell_type)
WHERE g.cid > 0 AND NOT (g.lat = 0 AND g.lon = 0)
GROUP BY mcc, mnc, lac, cid, cell_type, yr, lat, lon
"""

MONTHLY_QUERY = f"""
SELECT toStartOfMonth(timestamp) AS mo,
  uniqExactIf((mcc,mnc,lac,cid,cell_type), mcc=425 AND mnc IN (5,6))     AS ps,
  uniqExactIf((mcc,mnc,lac,cid,cell_type), mcc=425 AND mnc NOT IN (5,6)) AS il,
  uniqExactIf((mcc,mnc,lac,cid,cell_type), mcc=602)                      AS eg,
  uniqExactIf((mcc,mnc,lac,cid,cell_type),
              mcc NOT IN (425,602,{','.join(str(m) for m in EXCLUDED_MCC)})) AS ff
FROM cell.geos
WHERE lat BETWEEN {GAZA[2]} AND {GAZA[3]}
  AND lon BETWEEN {GAZA[0]} AND {GAZA[1]}
  AND cid > 0 AND NOT (lat = 0 AND lon = 0)
GROUP BY mo ORDER BY mo
"""


def add_grp(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the four-way identity key used by both panels."""
    df = df.copy()
    df["grp"] = np.select(
        [(df["mcc"] == 425) & df["mnc"].isin((5, 6)),
         df["mcc"] == 425,
         df["mcc"] == 602,
         df["mcc"].isin(EXCLUDED_MCC)],
        ["ps", "il", "eg", "nc"], default="ff")
    return df


def _serif() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def place_sequence(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cluster one cell's time-ordered positions into an ordered list of places.

    Sequential stay-point clustering: a position within PLACE_KM of the current
    place's running centroid joins it, otherwise it opens a new place. Returns
    an (n_places, 2) array of lon/lat centroids in visit order.

    A grid snap would be cheaper but wrong here -- a cell jittering across a
    cell boundary would oscillate between two grid squares forever and manufacture
    movement that is not there. Clustering against a running centroid has no
    boundaries to straddle.
    """
    out_la = [lat[0]]
    out_lo = [lon[0]]
    n = 1
    for i in range(1, lat.shape[0]):
        d = np.hypot((lat[i] - out_la[-1]) * KM_PER_DEG_LAT,
                     (lon[i] - out_lo[-1]) * KM_PER_DEG_LON)
        if d <= PLACE_KM:
            n += 1
            out_la[-1] += (lat[i] - out_la[-1]) / n
            out_lo[-1] += (lon[i] - out_lo[-1]) / n
        else:
            out_la.append(lat[i])
            out_lo.append(lon[i])
            n = 1
    return np.column_stack([out_lo, out_la])


def build_tracks(tracks: pd.DataFrame, year: int | None = None) -> tuple[dict, dict, dict]:
    """Split cells into a stationary population and a moving one.

    Each cell's history is reduced to an ordered sequence of *places* (see
    `place_sequence`), and a cell is a mover when it has at least two places and
    its two furthest-apart places are more than MOVE_KM apart. Movers are drawn
    as the polyline through that sequence.

    Three things this gets right that the previous first-stop-to-last-stop
    segment did not, all of which are common rather than edge cases:

    * **Three or more places.** 79%% of movers visit 3+ places. A single segment
      skipped every intermediate place and could be drawn straight across ground
      the cell was never on.
    * **Revisits.** Positions used to be de-duplicated *globally* per cell, so a
      return to a place already seen was deleted outright and A->B->A collapsed
      to A->B before anything was drawn. Only *consecutive* duplicates are
      collapsed now, so a round trip survives as three vertices.
    * **Round trips reading as stationary.** A cell that came back to where it
      started had first and last positions nearly coincident, so it drew a
      near-zero-length segment while still being counted as a mover: 876 of
      6,921 (12.7%%) drew under 0.5 km, and half of all movers drew less than
      half their true extent (median drawn/actual extent 0.506).

    Positions are snapped to SNAP_DEG first, which is the estimator-jitter floor
    -- median per-cell spread is 0.22 km and a cell can carry hundreds of
    distinct positions (p90 = 451).

    With `year` set, the itinerary is clustered from that year's positions
    alone. Slicing an all-time track by year afterwards would be wrong: it would
    inherit places the cell only visited in other years, and a cell that sat
    still all year would still show its neighbours' segments.
    """
    key = ["mcc", "mnc", "lac", "cid", "cell_type"]
    t = tracks if year is None else tracks[tracks["yr"] == year]
    t = t.copy()
    t["t0"] = pd.to_datetime(t["t0"])
    t["slat"] = (t["lat"] / SNAP_DEG).round() * SNAP_DEG
    t["slon"] = (t["lon"] / SNAP_DEG).round() * SNAP_DEG
    t = t.sort_values(key + ["t0"], kind="stable")

    # Collapse CONSECUTIVE duplicates only. A global de-duplication would drop
    # revisits and flatten every round trip -- see the docstring.
    same_cell = (t[key].shift() == t[key]).all(axis=1)
    same_pos = (t["slat"].shift() == t["slat"]) & (t["slon"].shift() == t["slon"])
    t = t[~(same_cell & same_pos)]

    gid = pd.factorize(pd.MultiIndex.from_frame(t[key]))[0]
    lat = t["slat"].to_numpy()
    lon = t["slon"].to_numpy()
    grp = t["grp"].to_numpy()
    bounds = np.flatnonzero(np.diff(gid)) + 1

    # `all` holds every place a cell was found at; `last` holds one entry per
    # cell. Which one is drawn depends on whether the group's itinerary is drawn
    # too -- see make_map.
    static: dict = {"all": {k: [] for k, _, _ in SERIES},
                    "last": {k: [] for k, _, _ in SERIES}, "n": 0}
    moving: dict = {"all": {k: [] for k, _, _ in SERIES},
                    "last": {k: [] for k, _, _ in SERIES}, "n": 0}
    lines: dict[str, list] = {k: [] for k, _, _ in SERIES}

    for sl in np.split(np.arange(gid.shape[0]), bounds):
        seq = place_sequence(lat[sl], lon[sl])
        g = grp[sl[0]]
        if seq.shape[0] >= 2:
            dx = (seq[:, 0][:, None] - seq[:, 0][None, :]) * KM_PER_DEG_LON
            dy = (seq[:, 1][:, None] - seq[:, 1][None, :]) * KM_PER_DEG_LAT
            extent = float(np.hypot(dx, dy).max())
        else:
            extent = 0.0
        # A dot goes at *every* place the cell was found at, not only the last
        # one: the polyline is the itinerary, the dots are the observations it
        # connects. With one endpoint dot a 6-place track read as a single
        # sighting joined by an unexplained zig-zag.
        pop = moving if (seq.shape[0] >= 2 and extent > MOVE_KM) else static
        pop["all"][g].append(seq)
        pop["last"][g].append(seq[-1][None, :])
        pop["n"] += 1
        if pop is moving:
            lines[g].append(seq)

    def to_arr(v):
        return np.vstack(v) if v else np.empty((0, 2))

    for pop in (static, moving):
        for which in ("all", "last"):
            pop[which] = {k: to_arr(v) for k, v in pop[which].items()}
    return static, moving, lines


# Per-identity mark weights. Domestic MCC 425 is context in both panels: faint
# and small, so the foreign marks stay legible on top of it.
STYLE = [
    # key   dot size  dot alpha  edge  line width  line alpha
    ("il",  0.30,     0.16,      0.0,  0.14,       0.16),
    ("ps",  0.30,     0.16,      0.0,  0.14,       0.16),
    ("eg",  1.5,      0.90,      0.10, 0.26,       0.70),
    ("ff",  1.7,      0.90,      0.10, 0.30,       0.80),
]


def make_map(pop: dict, lines: dict, path: Path) -> None:
    """Bare map: basemap plus marks, no typography. Empty `lines` for panel (a).

    Groups whose itinerary is drawn get a dot at every place, so the polyline
    reads as a route joining sightings. Context groups get one dot per cell:
    their tracks are not drawn, so their intermediate places carry no
    information here and at 57k marks they turn the panel into a wash.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(MAP_IN, MAP_IN))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor("#dceaf2")
    drawn = BASEMAP and add_osm_basemap(ax, GAZA, zoom=BASEMAP_ZOOM,
                                        alpha=BASEMAP_ALPHA, grayscale=False,
                                        zorder=0, source=BASEMAP)
    if drawn:
        # The strip's own boundary, stroked once and crisply on top of the raster.
        draw_geojson_layer(ax, ADMIN1_GEOJSON, GAZA, admin_names={"Gaza Strip"},
                           facecolor="none", edgecolor="#333333", linewidth=0.7,
                           alpha=0.95, zorder=2)
        for name, lon, lat in CITIES:
            ax.plot([lon], [lat], marker="o", ms=0.9, color="#2b2b2b",
                    markeredgewidth=0, zorder=6)
            ax.annotate(
                name, (lon, lat), xytext=(1.8, 1.3), textcoords="offset points",
                fontsize=3.4, color="#1a1a1a", zorder=6,
                path_effects=[pe.withStroke(linewidth=0.7, foreground="white")])
    else:
        # Tile server unreachable and nothing cached: fall back to the vector base.
        setup_context_map(ax, GAZA, countries={"PS", "IL", "EG"},
                          admin_names={"Gaza Strip"}, high_res=True)

    for key, size, alpha, edge, lw, lalpha in STYLE:
        rows = pop["all" if key in TRACK_GROUPS else "last"].get(key)
        if rows is not None and len(rows):
            ax.scatter(rows[:, 0], rows[:, 1], s=size, c=COLOR[key],
                       alpha=alpha, linewidth=edge,
                       edgecolors="white" if edge else "none",
                       rasterized=True, zorder=3 if size < 1 else 4)
        segs = lines.get(key) if key in TRACK_GROUPS else None
        if segs is not None and len(segs):
            # Lines clip to the frame, so a cell that also appears far away is
            # drawn leaving the map rather than silently dropped.
            ax.add_collection(LineCollection(
                segs, colors=COLOR[key], linewidths=lw, alpha=lalpha,
                capstyle="round", joinstyle="round", rasterized=True,
                zorder=3.5 if size < 1 else 4.5))

    ax.set_xlim(GAZA[0], GAZA[1])
    ax.set_ylim(GAZA[2], GAZA[3])
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=MAP_DPI, pad_inches=0)
    plt.close(fig)
    # No alpha channel -> the embedded PDF image carries no soft mask.
    with PILImage.open(path) as im:
        im.convert("RGB").save(path, optimize=True)
    w, h = PILImage.open(path).size
    print(f"  {path.name}  {w}x{h}px  {path.stat().st_size / 1e6:.2f} MB")


def make_timeline(monthly: pd.DataFrame, path: Path) -> None:
    """Distinct cells per month, log axis, direct-labelled. Vector PDF."""
    _serif()
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.8)
    _serif()

    m = monthly.copy()
    m["mo"] = pd.to_datetime(m["mo"])

    fig = plt.figure(figsize=(TL_W, TL_H))
    ax = fig.add_axes((0.048, 0.165, 0.73, 0.79))

    for key, label, color in SERIES:
        y = m[key].astype(float).to_numpy()
        # Zeros are gaps on a log axis, not points: a corpus gap (Palestinian
        # before 2024) and a genuine zero (foreign, Mar 2024).
        ax.plot(m["mo"], np.where(y > 0, y, np.nan), color=color, linewidth=1.5,
                solid_capstyle="round", zorder=3)
        last = y[-1]
        ax.annotate(f" {label} ({int(last):,})",
                    xy=(m["mo"].iloc[-1], last), xytext=(3, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=6.0, color="#2b2b2b", zorder=5)

    # The April 2024 step is ours: every series jumps when the crawl's scope
    # expanded. Marked so it is not read as an event on the ground.
    ax.axvline(CRAWL_STEP, color="#7a7a7a", linewidth=0.9, linestyle=(0, (3, 2)),
               zorder=2)
    # Set horizontally in the clear band above the Israeli line, to the right of
    # the rule: rotated and on the rule it collided with the step itself.
    ax.annotate("crawl scope expands", xy=(CRAWL_STEP, 5800), xytext=(3, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=5.4, color="#5f5f5f", zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(0.7, 9000)
    ax.set_xlim(m["mo"].iloc[0] - pd.Timedelta(days=20), m["mo"].iloc[-1])
    # No axis label: the LaTeX panel header names the quantity.
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(4, 7, 10)))
    ax.tick_params(axis="both", labelsize=6.2)
    ax.tick_params(axis="x", which="minor", length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}  {TL_W}x{TL_H}in  {path.stat().st_size / 1e6:.2f} MB")


def year_label(yr: int, monthly: pd.DataFrame) -> str:
    """Label a year, flagging the partial ones by the months actually crawled."""
    mo = pd.to_datetime(monthly["mo"])
    m = mo[mo.dt.year == yr]
    if m.empty:
        return str(yr)
    lo, hi = int(m.dt.month.min()), int(m.dt.month.max())
    return str(yr) if (lo, hi) == (1, 12) else f"{yr} ({MONTHS[lo]}--{MONTHS[hi]})"


def emit_tex(static_path: Path, tl_path: Path, year_panels: list, n_static: int,
             monthly: pd.DataFrame) -> str:
    """Figure body: bare panels, every piece of typography set in LaTeX.

    Each \\includegraphics sits alone on its line with no further brace, because
    the Makefile scrapes prerequisites with a regex whose filename class does
    not exclude `}` -- a trailing brace from an enclosing group gets swallowed
    into the captured name and make then tries to build `<name>.png}.eps`.
    """
    tex = lambda n: f"{n:,}".replace(",", "{,}")
    w = f"{0.92 / len(year_panels):.3f}"
    lines = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{cc}",
        rf"\small\textbf{{(a) Fixed position ({tex(n_static)})}} &",
        r"\small\textbf{(b) Distinct cells per month (log scale)} \\",
        rf"\includegraphics[width=0.23\linewidth]{{{static_path.name}}} &",
        rf"\includegraphics[width=0.73\linewidth]{{{tl_path.name}}} \\",
        r"\end{tabular}",
        r"",
        r"\smallskip",
        rf"\begin{{tabular}}{{{'c' * len(year_panels)}}}",
        rf"\multicolumn{{{len(year_panels)}}}{{c}}{{\small\textbf{{(c) Relocated $>$1\,km, by year}}}} \\",
        " & ".join(rf"\small {year_label(y, monthly)}" for y, _, _ in year_panels) + r" \\",
    ]
    for i, (_y, path, _n) in enumerate(year_panels):
        sep = "&" if i < len(year_panels) - 1 else r"\\"
        lines.append(rf"\includegraphics[width={w}\linewidth]{{{path.name}}} {sep}")
    lines.append(" & ".join(rf"\scriptsize {tex(n)} cells" for _y, _p, n in year_panels)
                 + r" \\")
    lines += [
        r"\end{tabular}",
        r"",
        # Required by the tile licence; the only figure text besides the caption.
        rf"{{\scriptsize {TILE_ATTRIBUTION[BASEMAP]}}}" if BASEMAP else "",
        r"    \caption{Foreign network identities in the Gaza Strip cluster at its",
        r"    southern end and are confined to 2024, while Palestinian, Israeli and",
        r"    Egyptian identities persist throughout.}",
        r"    \label{fig:gaza-foreign-identities}",
        r"\end{figure*}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGS / "gaza_foreign",
                        help="path stem: <stem>_map.png and <stem>_timeline.pdf")
    parser.add_argument("--preview", type=Path, default=None,
                        help="write a stitched preview here for eyeballing")
    parser.add_argument("--emit-tex", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    cells = classify(_cached("cells_by_year", CELLS_QUERY, args.refresh_cache))
    audit_mccs(cells)

    tracks = add_grp(_cached("tracks", TRACKS_QUERY, args.refresh_cache))
    tracks = tracks[tracks["grp"] != "nc"]
    monthly = _cached("monthly_by_group", MONTHLY_QUERY, args.refresh_cache)

    stem = args.output.with_suffix("")
    static_path = stem.parent / f"{stem.name}_static.png"
    tl_path = stem.parent / f"{stem.name}_timeline.pdf"

    t0 = time.time()
    # Panel (a) is all-time: a cell counts as fixed only if it never relocated.
    static, _, _ = build_tracks(tracks)
    n_static = static["n"]
    make_map(static, {}, static_path)

    year_panels = []
    for yr in sorted(int(y) for y in tracks["yr"].unique()):
        _, moving, lines = build_tracks(tracks, year=yr)
        path = stem.parent / f"{stem.name}_moving_{yr}.png"
        make_map(moving, lines, path)
        drawn = sum(len(lines[k]) for k in TRACK_GROUPS)
        print(f"    {yr}: {moving['n']:,} moved, {drawn:,} tracks drawn "
              + ", ".join(f"{k}={len(moving['last'][k]):,}" for k, _, _ in SERIES))
        year_panels.append((yr, path, moving["n"]))
    make_timeline(monthly, tl_path)
    print(f"[render] {time.time() - t0:.1f}s")

    if args.preview is not None:
        import subprocess
        subprocess.run(["/Users/chase/Research/cell-geolocation/venv/bin/python", "-c",
                        f"import pymupdf;d=pymupdf.open('{tl_path}');"
                        f"d[0].get_pixmap(dpi={MAP_DPI}).save('{args.preview}')"],
                       check=False)
        tl = PILImage.open(args.preview)
        mp = PILImage.open(map_path)
        h = max(tl.height, mp.height)
        sheet = PILImage.new("RGB", (mp.width + tl.width + 10, h), "white")
        sheet.paste(mp, (0, 0))
        sheet.paste(tl, (mp.width + 10, 0))
        sheet.save(args.preview)
        print(f"[preview] {args.preview}  {sheet.size[0]}x{sheet.size[1]}px")

    if args.emit_tex:
        print("\n" + "=" * 72 + "\n"
              + emit_tex(static_path, tl_path, year_panels, n_static, monthly))


if __name__ == "__main__":
    main()
