#!/usr/bin/env python3
"""Gaza: cell observations by the country identity their PLMN claims, one panel
per year (2023-2026).

Production figure, run against the corrected 63.3B-row `cell.geos`.

What the figure shows: MCC 425 is shared by Israeli and Palestinian operators,
so "country identity" is resolved at MNC granularity for 425 (05/06 = Jawwal and
Ooredoo are Palestinian, the rest Israeli) and at MCC granularity for everyone
else. Plotted that way, the Gaza bbox turns out to contain a burst of
*non-adjacent* foreign identities -- Libya, Saudi Arabia, Turkiye, Jordan,
Cyprus -- almost entirely confined to 2024. Egypt is present throughout and is
the one foreign identity with a land border on the bbox.

These are not a positional artifact: every foreign cell carries a distinct
position (2,074 unique latitudes across 2,166 cells, with offsets from any
candidate 0.001/0.005/0.01-deg lattice uniformly distributed), rather than
stacking on a single default coordinate.

Two coverage caveats are encoded in the figure rather than left to the reader:

* **Palestinian PLMNs are absent from the corpus before 2024.** MCC 425/05
  (Jawwal) is first seen globally on 2024-01-01 and 425/06 (Ooredoo) on
  2024-03-10 -- anywhere in the world, not merely in Gaza. The 2023 panel
  therefore says nothing about Palestinian service in Gaza, and is labelled to
  say so. Without that label it reads as a total network collapse.
* **The April 2024 onset is confounded with a crawl-scope expansion.** A large
  set of PLMNs (602/01, 606/01, 286/01, 416/01, 280/01, 425/28, 425/29, ...)
  all carry a first-seen of 2024-04-03, so the *arrival* of foreign identities
  cannot be dated from this corpus. Their *departure* can: monthly distinct
  non-adjacent foreign cells run 380-386 in Apr-May 2024, decay to 103 by Dec
  2024, and sit at exactly 3 from Jan 2025 through Jun 2026 -- while Palestinian
  and Egyptian coverage of the same bbox continues at 700-3,000 and 26-216 cells
  per month. The 2024-vs-2025 contrast is between two fully covered years and is
  not confounded; that is the comparison the panels support.

Design notes:

* Authored at final print width (7.0in) so `\\includegraphics[width=\\linewidth]`
  in a two-column `figure*` scales 1:1 and type renders at its true point size.
* Colours are the validated categorical set (dataviz `validate_palette.js`,
  light mode, all-pairs): 6 hues pass the lightness, chroma and normal-vision
  checks. Worst-pair CVD separation lands in the 6-8 band, which is legal only
  with secondary encoding, so **every country also gets its own marker shape**
  -- identity is never carried by colour alone. A 7th+ country folds into a grey
  "Other" rather than inventing a hue, per the same rule.
* The two MCC 425 groups are the bulk of the data (12.6k cells in 2024 against
  ~1.7k foreign), so they are drawn first, small and semi-transparent; foreign
  identities are drawn on top, larger and with a white edge. Without that the
  entire finding sits underneath the domestic scatter.
* Cached under `.cache/gaza_country_identity/`, keyed by query text so edits
  invalidate it automatically. `--refresh-cache` forces a re-read.

Caveat: observation volume is crawl-biased and 2023 (Nov-Dec) and 2026 (Jan-Jun)
are partial years -- panel titles say which months. Read position, not density.
Cell counts in the legend are distinct cells seen that year, which is a
population count and does not inherit the crawl-cadence bias that `obs` does.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helpers import setup_context_map
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
# Production figures live where LaTeX looks for them: preamble.tex sets
# \graphicspath{{./figs/}}.
FIGS = ROOT / "paper" / "figs"
CACHE = ROOT / ".cache" / "gaza_country_identity"

OUTPUT_DPI = 300
PREVIEW_DPI = 130

GAZA = (34.2, 34.6, 31.2, 31.6)  # xmin, xmax, ymin, ymax
NCOLS = 4
FIG_W = 7.0  # two-column \linewidth, in inches

# Validated categorical palette + a distinct marker per country (secondary
# encoding, required because worst-pair CVD separation is in the 6-8 band).
# Order is fixed and never cycled; anything past it folds into OTHER.
COUNTRIES = [
    # key            label                       colour     marker  size  z
    ("PS", "Palestinian (425/05-06)", "#b23a48", "o", 1.6, 3.0),
    ("IL", "Israeli (425, other MNC)", "#2f6f9f", "o", 1.6, 3.1),
    ("EG", "Egypt (602)", "#E69F00", "s", 6.5, 4.0),
    ("LY", "Libya (606)", "#009E73", "^", 8.0, 4.1),
    ("SA", "Saudi Arabia (420)", "#CC79A7", "D", 5.5, 4.2),
    ("TR", "Türkiye (286)", "#56B4E9", "v", 8.0, 4.3),
]
# "Other" is a residual, not a finding, so it is deliberately the least
# prominent foreign mark: smallest, and drawn beneath the named countries.
OTHER = ("XX", "Other foreign", "#6f6f6f", "X", 4.5, 3.6)

# MCC 425/05 and /06 do not enter the corpus until 2024; a year with no
# Palestinian cells is a coverage gap, not an observation. See module docstring.
PS_CORPUS_YEAR = 2024

# Authoritative MCC -> ISO map. Deliberately NOT read from `geo.mcc_iso`, which
# is incomplete in exactly the way that matters here: it has no row for MCC 425
# at all (so Israel and Palestine resolve to NULL), lists only 424 for the UAE
# (missing 430 and 431) and only 312/314/315/316 for the US (missing 310, 311
# and 313 -- and 311 is one of the MCCs actually observed in this bbox).
#
# Every country is given its *complete* ITU allocation, not just the codes seen
# today, so a future crawl that picks up a sibling MCC is classified rather than
# silently swept into the catch-all. Israel and Palestine share MCC 425 and are
# separated by MNC below; Egypt (602) and the rest each hold a single code.
MCC_ISO = {
    425: "IL_PS",                          # Israel + Palestine, split by MNC
    602: "EG", 606: "LY", 420: "SA", 286: "TR", 416: "JO", 280: "CY",
    415: "LB", 417: "SY", 426: "BH", 427: "QA", 419: "KW", 418: "IQ",
    421: "YE", 422: "OM", 432: "IR",
    424: "AE", 430: "AE", 431: "AE",       # UAE holds three codes
    310: "US", 311: "US", 312: "US", 313: "US",
    314: "US", 315: "US", 316: "US",       # US holds seven
    202: "GR", 284: "BG", 260: "PL", 428: "MN", 614: "NE",
}

# Codes that do not denote a country at all. Lumping these in with foreign
# operators would double-count them against the paper's separate treatment of
# testing/internal/non-existent MCCs, so they get their own key and are reported
# rather than hidden inside "other".
NON_COUNTRY_MCC = {
    1: "test (ITU 001)",
    999: "internal/private (ITU 999)",
    901: "international/shared",
    69: "unassigned by ITU",
    526: "unassigned by ITU",
}

PALESTINIAN_MNC = {5, 6}

MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


CELLS_QUERY = f"""
SELECT toYear(timestamp) AS yr, mcc, mnc, lac, cid, cell_type,
       avg(lat) AS avg_lat, avg(lon) AS avg_lon, count() AS obs
FROM cell.geos
WHERE lat BETWEEN {GAZA[2]} AND {GAZA[3]}
  AND lon BETWEEN {GAZA[0]} AND {GAZA[1]}
  AND cid > 0 AND NOT (lat = 0 AND lon = 0)
GROUP BY yr, mcc, mnc, lac, cid, cell_type
"""

# Month span actually crawled in each year, so partial years can be labelled as
# such instead of being silently compared against full ones.
COVERAGE_QUERY = f"""
SELECT toYear(timestamp) AS yr,
       min(toMonth(timestamp)) AS m0, max(toMonth(timestamp)) AS m1
FROM cell.geos
WHERE lat BETWEEN {GAZA[2]} AND {GAZA[3]}
  AND lon BETWEEN {GAZA[0]} AND {GAZA[1]}
  AND cid > 0 AND NOT (lat = 0 AND lon = 0)
GROUP BY yr ORDER BY yr
"""


def _cached(name: str, query: str, refresh: bool) -> pd.DataFrame:
    """Fetch `query`, caching by its own text.

    A sibling figure once silently rebuilt itself from CSVs written during the
    corrupted pre-dedup-fix run, so the key is the query text and provenance is
    printed rather than assumed.
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


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Label each cell with the country identity its PLMN claims.

    Keys are ISO codes, plus `PS`/`IL` for the two halves of MCC 425, `NC` for
    codes that name no country, and `XX` for anything unmapped.
    """
    key = df["mcc"].map(MCC_ISO)
    key = key.where(~df["mcc"].isin(NON_COUNTRY_MCC), "NC")
    is_425 = df["mcc"] == 425
    key = key.where(~is_425, np.where(df["mnc"].isin(PALESTINIAN_MNC), "PS", "IL"))
    df = df.copy()
    df["country"] = key.fillna("XX")
    return df


def audit_mccs(df: pd.DataFrame) -> None:
    """Print the MCC -> classification mapping actually exercised by the data.

    Cheap insurance against the failure this map exists to prevent: an MCC that
    belongs to a named country landing in the catch-all because its code was
    never listed.
    """
    tab = (df.groupby(["mcc", "country"]).size().rename("cells").reset_index()
             .sort_values("cells", ascending=False))
    tab["note"] = tab["mcc"].map(NON_COUNTRY_MCC).fillna("")
    print("[mcc audit] MCCs present and how each was classified:")
    print(tab.to_string(index=False))
    unmapped = tab[tab["country"] == "XX"]
    if not unmapped.empty:
        print(f"[mcc audit] WARNING {len(unmapped)} unmapped MCC(s): "
              f"{sorted(unmapped['mcc'].tolist())}")
    else:
        print("[mcc audit] no unmapped MCCs")


def year_label(yr: int, coverage: pd.DataFrame) -> str:
    row = coverage[coverage["yr"] == yr]
    if row.empty:
        return str(yr)
    m0, m1 = int(row["m0"].iloc[0]), int(row["m1"].iloc[0])
    if (m0, m1) == (1, 12):
        return str(yr)
    return f"{yr} ({MONTHS[m0]}–{MONTHS[m1]})"


def make_figure(cells: pd.DataFrame, coverage: pd.DataFrame,
                stem: Path, preview: Path | None) -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold",
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    years = sorted(int(y) for y in cells["yr"].unique())
    specs = [*COUNTRIES, OTHER]

    lon_span, lat_span = GAZA[1] - GAZA[0], GAZA[3] - GAZA[2]
    panel_w = FIG_W / NCOLS
    # Panels are square (equal aspect on a square bbox); the constant is the
    # two-line titles, tick labels and the two-row legend strip.
    fig_h = panel_w * (lat_span / lon_span) + 0.95
    fig, axes = plt.subplots(1, NCOLS, figsize=(FIG_W, fig_h),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for k, yr in enumerate(years):
        ax = axes[k]
        setup_context_map(
            ax, GAZA,
            countries={"PS", "IL", "EG"},
            admin_names={"Gaza Strip"},
        )
        gy = cells[cells["yr"] == yr]
        for key, _label, color, marker, size, z in specs:
            rows = gy[gy["country"] == key]
            if rows.empty:
                continue
            domestic = key in ("PS", "IL")
            ax.scatter(
                rows["avg_lon"], rows["avg_lat"],
                s=size, c=color, marker=marker,
                alpha=0.55 if domestic else 0.95,
                linewidth=0 if domestic else 0.35,
                edgecolors="none" if domestic else "white",
                rasterized=True, zorder=z,
            )

        foreign = int((~gy["country"].isin(["PS", "IL"])).sum())
        ax.set_title(f"{year_label(yr, coverage)}\n{foreign:,} foreign-identity cells",
                     fontsize=8)

        # A year with no Palestinian cells predates their entry into the corpus.
        # Say so on the panel: unlabelled, it reads as a total network collapse.
        if yr < PS_CORPUS_YEAR and not (gy["country"] == "PS").any():
            ax.text(
                0.5, 0.035, "Palestinian PLMNs\nnot yet in corpus",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=5.6, color="#5a4a4a", linespacing=1.15, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "#b8a9a9",
                      "linewidth": 0.4, "alpha": 0.88, "pad": 1.6},
            )
        ax.set_xlim(GAZA[0], GAZA[1])
        ax.set_ylim(GAZA[2], GAZA[3])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)

    for k in range(len(years), len(axes)):
        axes[k].axis("off")

    # Legend carries the total cell count per identity, which doubles as the
    # "visible label" relief the contrast check asks for on the lighter hues.
    totals = cells.groupby("country").size()
    handles = [
        plt.Line2D([0], [0], linestyle="none", marker=marker, color=color,
                   markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.35,
                   markersize=5, label=f"{label} — {int(totals.get(key, 0)):,}")
        for key, label, color, marker, _s, _z in specs
        if int(totals.get(key, 0)) > 0
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4,
               frameon=True, fontsize=6.6, handletextpad=0.4, columnspacing=1.0)

    stem.parent.mkdir(parents=True, exist_ok=True)
    # Both formats: the PDF is what paper/*.tex includes (vector text, rasterized
    # scatter), the PNG is for quick viewing outside LaTeX.
    for ext in ("pdf", "png"):
        out = stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=OUTPUT_DPI)
        print(f"{out}  ({out.stat().st_size / 1e6:.2f} MB)")
    if preview is not None:
        fig.savefig(preview, dpi=PREVIEW_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGS / "gaza_country_identity",
                        help="output path without extension; .pdf and .png are both written")
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--refresh-cache", action="store_true",
                        help="re-query cell.geos instead of reading the cached extract")
    args = parser.parse_args()

    cells = classify(_cached("cells_by_year", CELLS_QUERY, args.refresh_cache))
    coverage = _cached("coverage_by_year", COVERAGE_QUERY, args.refresh_cache)
    audit_mccs(cells)

    # Codes that name no country are reported and dropped rather than shown as
    # foreign operators; the paper treats them in its own section.
    nc = int((cells["country"] == "NC").sum())
    if nc:
        print(f"[classify] dropping {nc} cells on non-country MCCs "
              f"{sorted(cells.loc[cells['country'] == 'NC', 'mcc'].unique())}")
        cells = cells[cells["country"] != "NC"]

    # Every country without its own slot folds into the fixed "Other" key.
    named = {s[0] for s in COUNTRIES}
    cells["country"] = cells["country"].where(cells["country"].isin(named), OTHER[0])

    breakdown = (cells.groupby(["yr", "country"]).size()
                 .unstack(fill_value=0).reindex(columns=[s[0] for s in [*COUNTRIES, OTHER]],
                                                fill_value=0))
    print(breakdown.to_string())

    t0 = time.time()
    make_figure(cells, coverage, args.output.with_suffix(""), args.preview)
    print(f"[render] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
