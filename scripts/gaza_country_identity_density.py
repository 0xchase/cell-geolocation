#!/usr/bin/env python3
"""Gaza: spatial density of cells by the country identity their PLMN claims,
emitted as one bare PNG per (identity, year) cell plus a bare colourbar strip.

Companion to `gaza_country_identity.py`, which plots the same data as a
categorical scatter. That version cannot show density, for a structural reason:
colour is spent on identity, so the only remaining density cue is opacity
stacking, which is not readable as a quantity -- and with group sizes spanning
16,672 (Israeli) to 102 (Turkiye), the large groups saturate into a solid mass
while the small ones collapse into a single blob.

Here each channel does exactly one job:

* **identity -> grid row**   (4 rows)
* **year -> grid column**    (4 columns)
* **count -> colour**, one sequential light->dark ramp, shared by all 16 panels
  and log-scaled, so a bin in the foreign row is directly comparable to a bin in
  the Israeli row.

Because the scale is shared and absolute, the rows are legitimately unequal --
that inequality is the point. Per-panel normalisation would make the foreign row
look as busy as the Israeli one and is not done.

Like `ukraine_progression.py`, the PNGs carry map content only: no titles,
ticks, row labels, counts, legend or margins. Every piece of typography is set
in LaTeX, so it comes out in the paper's own font as real vector text and the
raster carries nothing but pixels. `--emit-tex` prints the figure body to paste
into the paper. Each image is fully opaque (no alpha channel) so the PDF embeds
it as a plain image with no soft mask.

Binning is 0.02 deg (~2.2 km lat, ~1.9 km lon) and the quantity binned is
**distinct cells**, not observations: `obs` counts our API poll cadence rather
than sightings, so an observation-weighted density would be a map of our own
crawl schedule. At 0.01 deg the Israeli row averages ~4 cells/bin, so a large
share of bins hold 0 or 1 and the panel salt-and-peppers -- masked bins let the
basemap through and the eye reads texture instead of density.

Rows. Egypt keeps its own row because it is the only foreign identity with a
land border on the bbox, so it is the control: whatever explains the others has
to explain why Egypt behaves differently. The remaining foreign identities
(Libya, Saudi Arabia, Turkiye, Jordan, Cyprus, UAE, Bulgaria, US, Mongolia,
Niger, Greece, Poland, Lebanon, Bahrain) share the fourth row. MCCs that name no
country -- 001 test, 901 international/shared, and the unassigned 069 and 526 --
are excluded and reported at run time rather than being counted as foreign
operators; the paper treats those separately. See `MCC_ISO` in the companion
module for why the mapping is hardcoded instead of read from `geo.mcc_iso`.

Coverage caveats, both encoded on the figure rather than left to the reader:

* Palestinian PLMNs (425/05, 425/06) do not enter the corpus until 2024-01-01
  and 2024-03-10 respectively, globally -- so the 2023 cell of the Palestinian
  row is a coverage gap, not an empty Gaza, and is hatched and labelled as one.
* The Apr 2024 arrival of foreign identities is confounded with a crawl-scope
  expansion (many PLMNs share a 2024-04-03 first-seen). Their departure is not:
  the 2024 -> 2025 contrast is between two fully covered years.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from PIL import Image as PILImage

from plot_helpers import setup_context_map
from gaza_country_identity import (
    CELLS_QUERY,
    COVERAGE_QUERY,
    GAZA,
    NON_COUNTRY_MCC,
    PS_CORPUS_YEAR,
    _cached,
    audit_mccs,
    classify,
    year_label,
)


ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"

PANEL_DPI = 300
FIG_W = 7.0        # two-column \linewidth, in inches
NCOLS = 4
GRID_DEG = 0.02    # see module docstring

# Grid rows: key, LaTeX row label. Egypt is the border-adjacent control.
ROWS = [
    ("PS", r"Palestinian\\425/05--06"),
    ("IL", r"Israeli\\425, other MNC"),
    ("EG", r"Egypt\\602"),
    ("FF", r"Non-adjacent\\foreign"),
]
DOMESTIC = {"PS", "IL", "EG"}

# Sequential, single hue, light -> dark, monotonic in lightness. Warm, so it
# separates from the beige land and pale blue sea of the basemap.
CMAP = sns.light_palette("#7a1f2b", as_cmap=True)


def to_rows(cells: pd.DataFrame) -> pd.DataFrame:
    """Fold every non-domestic country into the shared foreign row."""
    cells = cells.copy()
    cells["row"] = cells["country"].where(cells["country"].isin(DOMESTIC), "FF")
    return cells


def bin_counts(sub: pd.DataFrame) -> np.ndarray:
    """Distinct cells per GRID_DEG bin, as an (nlat, nlon) array."""
    nlon = int(round((GAZA[1] - GAZA[0]) / GRID_DEG))
    nlat = int(round((GAZA[3] - GAZA[2]) / GRID_DEG))
    if sub.empty:
        return np.zeros((nlat, nlon))
    ix = np.clip(((sub["avg_lon"].to_numpy() - GAZA[0]) / GRID_DEG).astype(int), 0, nlon - 1)
    iy = np.clip(((sub["avg_lat"].to_numpy() - GAZA[2]) / GRID_DEG).astype(int), 0, nlat - 1)
    flat = iy * nlon + ix
    return np.bincount(flat, minlength=nlat * nlon).reshape(nlat, nlon).astype(float)


def _bare_axes(fig: plt.Figure) -> plt.Axes:
    """Axes filling the canvas edge to edge, with no typography or furniture."""
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    setup_context_map(ax, GAZA, countries={"PS", "IL", "EG"},
                      admin_names={"Gaza Strip"})
    ax.set_xlim(GAZA[0], GAZA[1])
    ax.set_ylim(GAZA[2], GAZA[3])
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


def make_panels(cells: pd.DataFrame, stem: Path) -> tuple[dict, float]:
    """Write one bare map PNG per (identity, year) and return paths and vmax."""
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    years = sorted(int(y) for y in cells["yr"].unique())

    grids = {(key, yr): bin_counts(cells[(cells["row"] == key) & (cells["yr"] == yr)])
             for key, _ in ROWS for yr in years}
    vmax = max(2.0, max(g.max() for g in grids.values()))
    norm = LogNorm(vmin=1, vmax=vmax)

    lon_span, lat_span = GAZA[1] - GAZA[0], GAZA[3] - GAZA[2]
    panel_w = FIG_W / NCOLS
    panel_h = panel_w * (lat_span / lon_span)

    stem.parent.mkdir(parents=True, exist_ok=True)
    out: dict[tuple[str, int], dict] = {}
    for key, _label in ROWS:
        for yr in years:
            g = grids[(key, yr)]
            fig = plt.figure(figsize=(panel_w, panel_h))
            ax = _bare_axes(fig)
            ax.imshow(np.ma.masked_where(g == 0, g), origin="lower",
                      extent=(GAZA[0], GAZA[1], GAZA[2], GAZA[3]),
                      cmap=CMAP, norm=norm, interpolation="nearest",
                      aspect="auto", zorder=3)

            n = int(g.sum())
            gap = key == "PS" and yr < PS_CORPUS_YEAR and n == 0
            if gap:
                # Not an empty Gaza -- these PLMNs are not in the corpus yet.
                ax.add_patch(plt.Rectangle(
                    (GAZA[0], GAZA[2]), lon_span, lat_span, facecolor="#9a9a9a",
                    edgecolor="none", alpha=0.30, hatch="///", zorder=4))

            path = stem.parent / f"{stem.name}_{key}_{yr}.png"
            fig.savefig(path, dpi=PANEL_DPI, pad_inches=0)
            plt.close(fig)
            w, h = _flatten(path)
            out[(key, yr)] = {"path": path, "n": n, "gap": gap}
            print(f"  {path.name}  {w}x{h}px  {path.stat().st_size / 1e6:.2f} MB  n={n:,}")
    return out, vmax


def make_colorbar(stem: Path, vmax: float) -> Path:
    """Write a bare horizontal gradient strip, no ticks or labels.

    Position along the strip is linear in log(count) -- exactly what LogNorm
    does -- so the LaTeX side can place a decade tick at log10(v)/log10(vmax).
    """
    fig = plt.figure(figsize=(3.2, 0.10))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(np.linspace(0, 1, 512).reshape(1, -1), aspect="auto", cmap=CMAP)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    path = stem.parent / f"{stem.name}_cbar.png"
    fig.savefig(path, dpi=PANEL_DPI, pad_inches=0)
    plt.close(fig)
    w, h = _flatten(path)
    print(f"  {path.name}  {w}x{h}px  (log 1 -> {vmax:g})")
    return path


def emit_tex(panels: dict, years: list[int], coverage: pd.DataFrame,
             stem: Path, vmax: float) -> str:
    """Build the figure body: a tabular of bare PNGs with LaTeX typography."""
    name = stem.name
    colw = f"{0.98 / NCOLS:.3f}"

    head = " & ".join(rf"\small\textbf{{{year_label(y, coverage).replace('–', '--')}}}"
                      for y in years)
    lines = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{1pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{tabular}{r" + "c" * NCOLS + "}",
        " & " + head + r" \\",
    ]
    # Every \includegraphics is emitted alone on its line, ending in `}%` with
    # no further brace. The Makefile scrapes figure prerequisites with
    # `s/.*includegraphics(\[.+\])?\{([^\{]*)\}.*/\2/p`, and `[^\{]` does not
    # exclude `}` -- so a second closing brace later on the same line (from an
    # enclosing \shortstack or \makebox) gets swallowed into the captured
    # filename, and make then tries to build `<name>.png}.eps`. Trailing `%`
    # keeps the line breaks from inserting spurious spaces.
    for key, label in ROWS:
        lines.append(rf"\rotatebox{{90}}{{\scriptsize\shortstack{{{label}}}}} &")
        for i, y in enumerate(years):
            p = panels[(key, y)]
            note = r"\scriptsize\itshape not yet in corpus" if p["gap"] \
                else rf"\scriptsize {p['n']:,}".replace(",", "{,}")
            sep = "&" if i < len(years) - 1 else r"\\"
            lines += [
                r"\shortstack{%",
                rf"\includegraphics[width={colw}\linewidth]{{{p['path'].name}}}%",
                rf"\\[-1pt] {note}}} {sep}",
            ]
    lines.append(r"\end{tabular}")

    # Decade ticks placed at log10(v)/log10(vmax) along the strip. Expressed as
    # fractions of \linewidth rather than a \newlength, which would break on a
    # second compile of the same float.
    cb = 0.46  # colourbar width, in \linewidth
    decades = [10 ** k for k in range(0, int(math.floor(math.log10(vmax))) + 1)]
    span = math.log10(vmax)
    prev = 0.0
    ticks = []
    for v in decades:
        frac = math.log10(v) / span
        ticks.append(rf"\hspace*{{{(frac - prev) * cb:.4f}\linewidth}}"
                     rf"\makebox[0pt][c]{{\scriptsize ${{10}}^{{{int(math.log10(v))}}}$}}")
        prev = frac
    lines += [
        r"",
        r"\smallskip",
        r"\noindent\makebox[\linewidth][c]{%",
        rf"\includegraphics[width={cb}\linewidth,height=5pt]{{{name}_cbar.png}}%",
        r"}",
        r"\par\nobreak\vspace{1pt}",
        rf"\noindent\makebox[\linewidth][c]{{\makebox[{cb}\linewidth][l]{{"
        + "".join(ticks) + r"}}",
        r"\par\vspace{1pt}",
        r"{\scriptsize Distinct cells per " + f"{GRID_DEG:g}" +
        r"\textdegree{} bin (log scale, shared by all "
        + f"{len(ROWS) * len(years)}" + r" panels)}",
        r"",
        r"    \caption{Density of cells in the Gaza Strip by the country identity",
        r"    their \ac{PLMN} claims, by year, on one shared scale, with",
        r"    non-adjacent foreign identities confined to 2024.}",
        r"    \label{fig:gaza-country-identity}",
        r"\end{figure*}",
    ]
    return "\n".join(lines)


def contact_sheet(panels: dict, years: list[int], path: Path) -> None:
    """Stitch the panels into one image, purely so the result can be eyeballed.

    Not referenced by the paper -- LaTeX composes the real figure.
    """
    ims = {k: PILImage.open(v["path"]) for k, v in panels.items()}
    w, h = next(iter(ims.values())).size
    pad = 8
    sheet = PILImage.new("RGB", (NCOLS * w + (NCOLS - 1) * pad,
                                 len(ROWS) * h + (len(ROWS) - 1) * pad), "white")
    for r, (key, _) in enumerate(ROWS):
        for c, y in enumerate(years):
            sheet.paste(ims[(key, y)], (c * (w + pad), r * (h + pad)))
    sheet.save(path)
    print(f"[preview] {path}  {sheet.size[0]}x{sheet.size[1]}px")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=FIGS / "gaza_density",
                        help="path stem; per-panel PNGs are <stem>_<row>_<year>.png")
    parser.add_argument("--preview", type=Path, default=None,
                        help="write a stitched contact sheet here for eyeballing")
    parser.add_argument("--emit-tex", action="store_true",
                        help="print the LaTeX figure body to paste into the paper")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    cells = classify(_cached("cells_by_year", CELLS_QUERY, args.refresh_cache))
    coverage = _cached("coverage_by_year", COVERAGE_QUERY, args.refresh_cache)
    audit_mccs(cells)

    nc = cells[cells["country"] == "NC"]
    if not nc.empty:
        detail = ", ".join(f"{m} ({NON_COUNTRY_MCC[m]}): {n}"
                           for m, n in nc.groupby("mcc").size().items())
        print(f"[classify] excluding {len(nc)} cells on non-country MCCs -- {detail}")
        cells = cells[cells["country"] != "NC"]

    cells = to_rows(cells)
    print(cells.groupby(["yr", "row"]).size().unstack(fill_value=0).to_string())

    years = sorted(int(y) for y in cells["yr"].unique())
    t0 = time.time()
    stem = args.output.with_suffix("")
    panels, vmax = make_panels(cells, stem)
    make_colorbar(stem, vmax)
    if args.preview is not None:
        contact_sheet(panels, years, args.preview)
    print(f"[render] {len(panels)} panels in {time.time() - t0:.1f}s")

    if args.emit_tex:
        print("\n" + "=" * 72 + "\n" + emit_tex(panels, years, coverage, stem, vmax))


if __name__ == "__main__":
    main()
