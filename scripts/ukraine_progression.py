#!/usr/bin/env python3
"""Ukraine progression: density of distinct Russian- and Ukrainian-operator cells
across Ukraine, one panel per year, against the documented front.

Rewritten from the earlier two-colour occupancy version, which drew every bin
that held any observation at near-full opacity and so read as flat territory.
This one measures *how many distinct cell identities* sit in each bin, which is
both the more informative quantity and the more defensible one:

* Observation counts track crawl cadence -- corpus volume swings ~4x month to
  month -- so the old panels partly measured how hard we polled. A distinct-ID
  count does not care how many times an identity was seen.
* Density is carried by a sequential ramp (light -> dark, one hue per operator,
  one shared log scale across all panels), so cities separate from countryside
  instead of everything saturating to the same block of colour.

Layout follows gaza_country_identity_density.py: bare map PNGs with no
typography, composed into a tabular by LaTeX so all type is vector and in the
paper's own font. `--emit-tex` prints the figure body.

Identity totals under each panel come from a separate whole-panel query, not
from summing the per-bin counts -- an identity seen in two bins is two bin-level
uniques but one cell, so the sums would overcount.

Caveats: 2023 (Nov--Dec) and 2026 (Jan--Jun) are partial years, marked in the
column heads; and cells are geocoded to Apple's returned position, so a bin
holds identities Apple places there, not verified tower sites.

Bins are square in kilometres (see CELL_KM) rather than in degrees, and the
panels are square, so the maps carry their true geographic proportions instead
of plate carree's longitude stretch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage

from plot_helpers import add_osm_basemap, draw_geojson_layer, ADMIN1_GEOJSON, TILE_ATTRIBUTION
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
FRONTLINE_GEOJSON = DATA_ROOT / "enrichment" / "frontline_by_year.geojson"
# preamble.tex sets \graphicspath{{./figs/}}.
FIGS = ROOT / "paper" / "figs"
CACHE = ROOT / ".cache" / "ukraine_progression"

# Floor only. The effective DPI is raised in make_panels if a panel would
# otherwise be narrower than NBINS pixels, since a nearest-neighbour resample
# below one pixel per bin silently discards data.
PANEL_DPI_MIN = 300

# Bins are square in kilometres, not in degrees. A degree of longitude is only
# cos(lat) as long as a degree of latitude, so the old 0.02-degree bin was
# 2.22 km tall and 1.47 km wide. Latitude and longitude steps are therefore
# derived separately from one target edge length, at a reference latitude in the
# middle of the map.
#
# The single reference latitude is the one approximation: a bin is exactly
# CELL_KM wide at LAT_MID and drifts to +2.9% at the south edge and -3.0% at the
# north. A projected CRS would remove that, at the cost of reprojecting the
# basemap too.
R_KM = 6371.0088
KM_PER_DEG_LAT = 2 * math.pi * R_KM / 360
CELL_KM = 1.0
# Cropped from the west: unoccupied central and western Ukraine carried no
# front and dominated the frame. The eastern edge stays at the Russian border
# so the whole occupied east, Luhansk included, is kept.
#
# 31.4E is as far east as the western edge can go. The documented front runs to
# 31.51E at the Dnipro mouth below Kherson, so anything past that starts
# clipping the front itself. That caps the crop at 12% rather than 20%, and it
# puts Kyiv (30.52E) and Odesa (30.73E) outside the frame.
LON0, LON1 = 31.4, 40.2
LAT_MID = 48.93
DLAT = CELL_KM / KM_PER_DEG_LAT
DLON = CELL_KM / (KM_PER_DEG_LAT * math.cos(math.radians(LAT_MID)))
# Square grid, so the panel is square once it holds an NBINS x NBINS array. The
# latitude span follows from the bin count rather than being chosen, which is
# what makes the panel square; it lands on roughly 45.6-52.2N, covering Odesa,
# Kyiv and the full length of the front.
NBINS = round((LON1 - LON0) / DLON)
_LAT_SPAN = NBINS * DLAT
BBOX = (LON0, LON1, LAT_MID - _LAT_SPAN / 2, LAT_MID + _LAT_SPAN / 2)

NCOLS = 3                          # 2024-2026 in a single row
# 2023 is Nov-Dec only: two months against twelve, which is not comparable on a
# shared scale and made the first panel look like a collapse in coverage.
DROP_YEARS = {2023}
FIG_W = 7.0                        # USENIX two-column \linewidth, in inches

# Sequential, single hue per operator, light -> dark and monotonic in lightness.
# Identity is carried by the row label and the ramp hue; magnitude by lightness.
RU_COLOR = "#c02a3c"
UA_COLOR = "#2f6f9f"
# MNC 255-707 is Kyivstar's Starlink Direct-to-Cell network, a separate PLMN
# issued for the satellite segment (ITU OB #1322, 15 Aug 2025; commercial launch
# 24 Nov 2025). It is not terrestrial infrastructure and its footprint does not
# respect the front, so folding it into the Ukrainian ramp made the 2026 panel
# read as restored ground coverage inside occupied territory. Third hue chosen
# with the palette validator: it is the only candidate tested that clears CVD
# separation and the normal-vision floor against both existing hues.
SAT_COLOR = "#762a83"

# Raster basemap, matching the Gaza figures. Zoom 8 gives ~180 px per degree
# against the ~80 px per degree the panel actually renders at, so the tiles are
# ~2x oversampled -- crisp without pulling thousands of tiles for a bbox this
# wide (zoom 12, the Gaza setting, would be ~8,000 tiles here).
BASEMAP = "carto_voyager_nolabels"
BASEMAP_ZOOM = 8
BASEMAP_ALPHA = 1.0
# Near-black, so the front reads against both the warm and the cool ramp.
FRONT_COLOR = "#141414"

OPERATORS = [("ru", "Russian terrestrial (MCC 250)"),
             ("ua", "Ukrainian terrestrial (MCC 255)"),
             ("sat", "Kyivstar Direct-to-Cell (255-707)")]

# Both ramps start well above white. sns.light_palette bottoms out near #f2f0f0,
# which is indistinguishable from the beige land (#f5f1e8) and the pale water
# (#dceaf2) underneath, so the sparsest bins read as empty basemap. Cutting the
# bottom 30% puts the lightest step at a clearly tinted colour while keeping the
# ramp monotonic in lightness.
RAMP_FLOOR = 0.30


# Bins are cut server-side at DLAT x DLON. Distinct counts cannot be re-binned
# client-side: an identity spanning two fine bins would be counted twice when
# the fine bins are merged.
#
# The server returns integer bin *indices*, not binned coordinates. Returning a
# binned coordinate and re-deriving the index here put 85% of bins in the wrong
# column: the step has no exact binary form, so (coord-origin)/step lands on
# 0.9999... and int() truncates it into the previous bin, collapsing alternate
# columns into their neighbour and printing a crosshatch across every panel.
GRID_QUERY = f"""
SELECT toYear(g.timestamp) AS yr,
       toInt32(floor((g.lat - {BBOX[2]:.10f}) / {DLAT:.10f})) AS iy,
       toInt32(floor((g.lon - {BBOX[0]:.10f}) / {DLON:.10f})) AS ix,
       uniqExactIf((g.mcc, g.mnc, g.lac, g.cid, g.cell_type), g.mcc = 250) AS ru,
       uniqExactIf((g.mcc, g.mnc, g.lac, g.cid, g.cell_type), g.mcc = 255 AND g.mnc != 707) AS ua,
       uniqExactIf((g.mcc, g.mnc, g.lac, g.cid, g.cell_type), g.mcc = 255 AND g.mnc = 707) AS sat
FROM cell.geos AS g
INNER JOIN (SELECT mcc, mnc, lac, cid, cell_type FROM cell.summary_full
            WHERE country_iso = 'UA') AS s
    USING (mcc, mnc, lac, cid, cell_type)
WHERE g.mcc IN (250, 255) AND g.cid > 0 AND NOT (g.lat = 0 AND g.lon = 0)
  AND g.lat BETWEEN {BBOX[2]} AND {BBOX[3]}
  AND g.lon BETWEEN {BBOX[0]} AND {BBOX[1]}
GROUP BY yr, iy, ix
HAVING ru + ua + sat > 0
"""

# Whole-panel distinct identities, counted once over the panel rather than
# summed from bins.
TOTALS_QUERY = f"""
SELECT toYear(g.timestamp) AS yr,
       uniqExactIf((g.mcc, g.mnc, g.lac, g.cid, g.cell_type), g.mcc = 250) AS ru,
       uniqExactIf((g.mcc, g.mnc, g.lac, g.cid, g.cell_type), g.mcc = 255 AND g.mnc != 707) AS ua,
       uniqExactIf((g.mcc, g.mnc, g.lac, g.cid, g.cell_type), g.mcc = 255 AND g.mnc = 707) AS sat,
       min(toMonth(g.timestamp)) AS m0, max(toMonth(g.timestamp)) AS m1
FROM cell.geos AS g
INNER JOIN (SELECT mcc, mnc, lac, cid, cell_type FROM cell.summary_full
            WHERE country_iso = 'UA') AS s
    USING (mcc, mnc, lac, cid, cell_type)
WHERE g.mcc IN (250, 255) AND g.cid > 0 AND NOT (g.lat = 0 AND g.lon = 0)
  AND g.lat BETWEEN {BBOX[2]} AND {BBOX[3]}
  AND g.lon BETWEEN {BBOX[0]} AND {BBOX[1]}
GROUP BY yr ORDER BY yr
"""

MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _cached(name: str, query: str, refresh: bool) -> pd.DataFrame:
    """Fetch `query`, caching by its own text.

    obs29 had no cache. A sibling figure did, and silently rebuilt itself from
    CSVs written during the corrupted run -- so the key is the query text and
    provenance is printed rather than assumed.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.{hashlib.sha256(query.encode()).hexdigest()[:16]}.pkl"
    if path.exists() and not refresh:
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"[cache] hit {path.name} (written {stamp})")
        return pd.read_pickle(path)
    print(f"[cache] {'refresh forced' if refresh else 'miss'} -- querying cell.geos for {name} ...")
    t0 = time.time()
    df = _remote_ch_df(query)
    print(f"[cache] {len(df):,} rows in {time.time() - t0:.1f}s -> {path.name}")
    df.to_pickle(path)
    return df


def load_fronts() -> dict[int, list[list[list[float]]]]:
    fc = json.loads(FRONTLINE_GEOJSON.read_text())
    out: dict[int, list[list[list[float]]]] = {}
    for feat in fc["features"]:
        g = feat["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        out[int(feat["properties"]["year"])] = [[[p[0], p[1]] for p in line] for line in lines]
    return out


def year_label(yr: int, totals: pd.DataFrame) -> str:
    row = totals[totals["yr"] == yr]
    if row.empty:
        return str(yr)
    m0, m1 = int(row["m0"].iloc[0]), int(row["m1"].iloc[0])
    if (m0, m1) == (1, 12):
        return str(yr)
    return f"{yr} ({MONTHS[m0]}--{MONTHS[m1]})"


def bin_counts(sub: pd.DataFrame, col: str) -> np.ndarray:
    """Distinct identities per CELL_KM bin, as a square (NBINS, NBINS) array."""
    nlon = nlat = NBINS
    if sub.empty:
        return np.zeros((nlat, nlon))
    ix = np.clip(sub["ix"].to_numpy(), 0, nlon - 1)
    iy = np.clip(sub["iy"].to_numpy(), 0, nlat - 1)
    flat = iy * nlon + ix
    return np.bincount(flat, weights=sub[col].to_numpy(),
                       minlength=nlat * nlon).reshape(nlat, nlon).astype(float)


def _bare_axes(fig: plt.Figure) -> plt.Axes:
    """Axes filling the canvas edge to edge, with no typography or furniture."""
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor("#dceaf2")
    # Raster tiles rather than the Natural Earth vector fill. At this extent the
    # vector basemap was one flat beige polygon per country, so terrain, rivers
    # and the coastline carried no detail; the tiles also removed the need to
    # enumerate every land neighbour to stop them rendering as sea.
    drawn = BASEMAP and add_osm_basemap(ax, BBOX, zoom=BASEMAP_ZOOM,
                                        alpha=BASEMAP_ALPHA, grayscale=False,
                                        zorder=0, source=BASEMAP)
    if drawn:
        # Oblast boundaries stroked once, crisply, over the raster.
        draw_geojson_layer(ax, ADMIN1_GEOJSON, BBOX, countries={"UA"},
                           facecolor="none", edgecolor="#8a8279", linewidth=0.25,
                           alpha=0.8, zorder=2)
    ax.set_xlim(BBOX[0], BBOX[1])
    ax.set_ylim(BBOX[2], BBOX[3])
    ax.set_aspect("auto")
    # Hide ticks/spines individually rather than set_axis_off(), which also
    # suppresses the axes patch -- and that patch is the water colour.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def _flatten(path: Path) -> tuple[int, int]:
    """Drop the alpha channel so the embedded PDF image carries no soft mask."""
    with PILImage.open(path) as im:
        im.convert("RGB").save(path, optimize=True)
    return PILImage.open(path).size


def cmaps() -> dict[str, object]:
    out = {}
    for key, colour in (("ru", RU_COLOR), ("ua", UA_COLOR), ("sat", SAT_COLOR)):
        base = sns.light_palette(colour, as_cmap=True)
        out[key] = LinearSegmentedColormap.from_list(
            key, base(np.linspace(RAMP_FLOOR, 1.0, 256)))
    return out


def panel_rgba(grids: dict, yr: int, vmax: float, cm: dict) -> np.ndarray:
    """All three networks on one panel: hue = whichever has the most cells in the
    bin, lightness = that network's count on the shared log scale.

    Blending hues by share instead would put a muddy mixture at every contested
    bin and would collide with lightness, which already carries magnitude.
    """
    stack = np.stack([grids[(k, yr)] for k, _ in OPERATORS])      # (3, nlat, nlon)
    win = stack.argmax(0)
    val = stack.max(0)
    seen = stack.sum(0) > 0
    t = np.log10(np.clip(val, 1.0, vmax)) / math.log10(vmax)

    rgba = np.zeros(val.shape + (4,))
    for i, (key, _label) in enumerate(OPERATORS):
        m = win == i
        rgba[m] = cm[key](t[m])
    rgba[..., 3] = np.where(seen, 1.0, 0.0)
    return rgba


def make_panels(grid: pd.DataFrame, fronts: dict[int, list], stem: Path
                ) -> tuple[dict, float, list[int]]:
    """Write one bare map PNG per (operator, year); return paths, vmax, years."""
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    years = sorted(int(y) for y in grid["yr"].unique() if int(y) not in DROP_YEARS)
    dropped = sorted(int(y) for y in grid["yr"].unique() if int(y) in DROP_YEARS)
    if dropped:
        print(f"  dropped partial year(s): {', '.join(map(str, dropped))}")

    grids = {(key, yr): bin_counts(grid[grid["yr"] == yr], key)
             for key, _label in OPERATORS for yr in years}
    # One scale across every panel, so a shade means the same count everywhere.
    # Topped out at a high percentile rather than the maximum. A few bins hold
    # thousands of identities -- 1,721 squares inside this bbox are attractors in
    # cell.attractors, the largest pulling 1,404 displaced cells to 48.70N 39.10E
    # -- and scaling to those pushed the median bin (3 identities) into the
    # bottom eighth of the ramp, which is what made every panel look washed out.
    nz = np.concatenate([g[g > 0].ravel() for g in grids.values()])
    vmax = float(max(10.0, np.ceil(np.percentile(nz, 99.5) / 100.0) * 100.0))
    print(f"  scale: median {np.median(nz):.0f}, p99.5 {np.percentile(nz, 99.5):.0f}, "
          f"max {nz.max():.0f} -> vmax {vmax:g} ({(nz > vmax).sum()} bins clipped)")
    cm = cmaps()

    # Square panel holding a square array: every bin renders square. And since
    # the latitude span was chosen as lon_span * cos(LAT_MID), the map carries
    # its true geographic proportions rather than plate carree's stretch.
    panel_w = panel_h = FIG_W / NCOLS
    dpi = max(PANEL_DPI_MIN, math.ceil(NBINS / panel_w))

    stem.parent.mkdir(parents=True, exist_ok=True)
    out: dict[int, dict] = {}
    for yr in years:
        fig = plt.figure(figsize=(panel_w, panel_h))
        ax = _bare_axes(fig)
        # Zero-total bins get alpha 0, so they show the basemap rather than the
        # ramp's lightest step.
        ax.imshow(panel_rgba(grids, yr, vmax, cm), origin="lower",
                  extent=BBOX, interpolation="nearest", aspect="auto", zorder=3)
        if yr in fronts:
            for line in fronts[yr]:
                ax.plot([p[0] for p in line], [p[1] for p in line],
                        color=FRONT_COLOR, linewidth=1.2, alpha=0.9,
                        solid_capstyle="round", zorder=6)

        path = stem.parent / f"{stem.name}_{yr}.png"
        fig.savefig(path, dpi=dpi, pad_inches=0)
        plt.close(fig)
        w, h = _flatten(path)
        out[yr] = {"path": path}
        peaks = "  ".join(f"{k}={int(grids[(k, yr)].max())}" for k, _ in OPERATORS)
        print(f"  {path.name}  {w}x{h}px  {path.stat().st_size / 1e6:.2f} MB  peak/bin {peaks}")
    return out, vmax, years


def make_colorbars(stem: Path, vmax: float) -> None:
    """Write one bare gradient strip per operator, no ticks or labels.

    Position along a strip is linear in log(count) -- exactly what LogNorm does
    -- so LaTeX can place a decade tick at log10(v)/log10(vmax).
    """
    for key, cmap in cmaps().items():
        fig = plt.figure(figsize=(3.2, 0.10))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect="auto", cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        path = stem.parent / f"{stem.name}_cbar_{key}.png"
        fig.savefig(path, dpi=PANEL_DPI_MIN, pad_inches=0)
        plt.close(fig)
        w, h = _flatten(path)
        print(f"  {path.name}  {w}x{h}px  (log 1 -> {vmax:g})")


def emit_tex(panels: dict, years: list[int], totals: pd.DataFrame,
             stem: Path, vmax: float) -> str:
    """Build the figure body: a tabular of bare PNGs with LaTeX typography."""
    name = stem.name
    colw = f"{0.98 / NCOLS:.3f}"

    lines = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{1pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{tabular}{" + "c" * NCOLS + "}",
    ]
    # Every \includegraphics is emitted alone on its line, ending in `}%` with no
    # further brace. The Makefile scrapes prerequisites with
    # `s/.*includegraphics(\[.+\])?\{([^\{]*)\}.*/\2/p`, and `[^\{]` does not
    # exclude `}` -- a second closing brace later on the same line (from an
    # enclosing \shortstack) gets swallowed into the captured filename, and make
    # then tries to build `<name>.png}.eps`.
    for r in range(0, len(years), NCOLS):
        row = years[r:r + NCOLS]
        lines.append(" & ".join(
            rf"\small\textbf{{{year_label(y, totals)}}}" for y in row) + r" \\")
        for i, y in enumerate(row):
            n = {k: int(totals.loc[totals["yr"] == y, k].iloc[0]) for k, _ in OPERATORS}
            note = (rf"\scriptsize {n['ru']:,} RU \textbar{{}} {n['ua']:,} UA"
                    rf" \textbar{{}} {n['sat']:,} D2C").replace(",", "{,}")
            sep = "&" if i < len(row) - 1 else r"\\"
            lines += [
                r"\shortstack{%",
                rf"\includegraphics[width={colw}\linewidth]{{{panels[y]['path'].name}}}%",
                rf"\\[-1pt] {note}}} {sep}",
            ]
    lines.append(r"\end{tabular}")

    # Decade ticks at log10(v)/log10(vmax) along the strip, as fractions of
    # \linewidth rather than a \newlength, which would break on a second compile
    # of the same float.
    cb = 0.30
    span = math.log10(vmax)
    prev = 0.0
    ticks = []
    for k in range(0, int(math.floor(span)) + 1):
        frac = k / span
        ticks.append(rf"\hspace*{{{(frac - prev) * cb:.4f}\linewidth}}"
                     rf"\makebox[0pt][c]{{\scriptsize ${{10}}^{{{k}}}$}}")
        prev = frac

    lines += [
        r"",
        r"\smallskip",
        r"\noindent\makebox[\linewidth][c]{%",
        r"\begin{tabular}{r@{\hspace{3pt}}l}",
    ] + [
        line
        for key, label in OPERATORS
        for line in (rf"\scriptsize {label} &",
                     rf"\includegraphics[width={cb}\linewidth,height=5pt]{{{name}_cbar_{key}.png}}%",
                     r"\\")
    ] + [
        r"& \makebox[" + f"{cb}" + r"\linewidth][l]{" + "".join(ticks) + r"}",
        r"\end{tabular}%",
        r"}",
        r"\par\vspace{2pt}",
        r"{\scriptsize Distinct cells per " + f"{CELL_KM:g}" +
        r"\,km square bin (log scale, shared by all three networks and all "
        + f"{len(years)}" + r" panels); each bin takes the hue of whichever"
        r" network has the most cells in it.\quad"
        r"\rule[0.15em]{1.4em}{1pt}~Documented front}",
        r"",
        # Required by the tile licence.
        (rf"{{\scriptsize {TILE_ATTRIBUTION[BASEMAP]}}}" if BASEMAP else ""),
        r"",
        r"    \caption{Density of distinct Russian, Ukrainian and Kyivstar",
        r"    Direct-to-Cell satellite cells across Ukraine, by year, on one",
        r"    shared scale, against the documented front.}",
        r"    \label{fig:ukraine-progression}",
        r"\end{figure*}",
    ]
    return "\n".join(lines)


def contact_sheet(panels: dict, years: list[int], path: Path) -> None:
    """Stitch the panels into one image, purely so the result can be eyeballed."""
    ims = [PILImage.open(panels[y]["path"]) for y in years]
    w, h = ims[0].size
    pad = 8
    nrows = (len(years) + NCOLS - 1) // NCOLS
    sheet = PILImage.new("RGB", (NCOLS * w + (NCOLS - 1) * pad,
                                 nrows * h + (nrows - 1) * pad), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % NCOLS) * (w + pad), (i // NCOLS) * (h + pad)))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGS / "ukraine_progression")
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true",
                        help="re-query cell.geos instead of reading the cached grid")
    parser.add_argument("--emit-tex", action="store_true",
                        help="print the LaTeX figure body for casestudies.tex")
    args = parser.parse_args()

    grid = _cached("ids_by_year", GRID_QUERY, args.refresh_cache)
    totals = _cached("id_totals_by_year", TOTALS_QUERY, args.refresh_cache)
    stem = args.output.with_suffix("")

    t0 = time.time()
    panels, vmax, years = make_panels(grid, load_fronts(), stem)
    make_colorbars(stem, vmax)
    if args.preview is not None:
        contact_sheet(panels, years, args.preview)
    print(f"[render] {len(panels)} panels in {time.time() - t0:.1f}s  (vmax={vmax:g}/bin)")
    if args.emit_tex:
        print("\n" + "=" * 72 + "\n" + emit_tex(panels, years, totals, stem, vmax))


if __name__ == "__main__":
    main()
