#!/usr/bin/env python3
"""S5: mixture-geometry mechanism test.

THE FALSIFIABLE PREDICTION
--------------------------
If the platform's published coordinate for a cell is an aggregate over
contributed GNSS fixes, and a fraction w of those fixes are spoofed to a decoy D
while the remainder are honest at the cell's true position H, then any linear
aggregate lands at

    P = (1 - w) H + w D

That is a point ON THE SEGMENT H->D. Two things follow, and both are testable:

  1. CROSS-TRACK OFFSET IS ZERO. The displaced estimate should have no
     systematic component perpendicular to the H->D axis. An artifact that
     merely made positions "wrong near D" -- a centroid fallback, a regional
     default, positional noise in a dense area -- would scatter estimates
     isotropically ABOUT D and produce cross-track offsets comparable to
     along-track ones.

  2. ALONG-TRACK IS A GRADED CONTINUUM. Because w varies per cell and over time
     (each cell has its own mix of honest and spoofed contributors), the
     along-track fraction should take intermediate values with mass throughout
     (0, 1) -- not cluster at the endpoints.

WHAT PROPERTY 2 BUYS: SPOOFING vs IDENTITY REPLAY
-------------------------------------------------
These two produce the same coarse signature -- "region A's cells appear at point
B" -- and the previous pipeline separated them with a distance heuristic, cutting
at 1,000 km on the reasoning that a spoofer sets a locally plausible position
while a replay device sits wherever it sits. That is a guess about intent, not a
measurement, and it misclassifies both a long-baseline spoof and a nearby replay.

The mixture geometry separates them physically. Under GNSS spoofing the cell is
real, stays where it is, and its published coordinate is a WEIGHTED MEAN -- so
intermediate positions are not merely possible, they are compulsory. Under
identity replay the cell identity is genuinely observed at the replay site by
honest receivers, so the platform sees two populations of true fixes and the
estimate sits at one place or the other. Replay is bimodal; spoofing is graded.

The discriminator is therefore the SHAPE of the along-track distribution, and it
requires no assumption about how far an attacker would plausibly move a target.

PROJECTION
----------
Positions are projected into a local equirectangular frame centred on H, with
longitude scaled by cos(lat). This is accurate to well under the 1.1 km grid for
the displacement ranges that matter here (tens to hundreds of km). For candidate
"displacements" of thousands of km -- which is the identity-replay regime -- the
planar frame is not valid, so along-track is computed from great-circle
distances instead and the planar cross-track is reported as NaN rather than as a
small number that would look like a passing result.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ch_remote import ch_df

SETTINGS = {"optimize_aggregation_in_order": 0, "max_threads": 12}

R_EARTH_KM = 6371.0
PLANAR_VALID_KM = 800.0   # beyond this the equirectangular frame is not trustworthy


def fetch(lat: float, lon: float, radius_deg: float,
          src_lat10: int | None = None, src_lon10: int | None = None) -> pd.DataFrame:
    """Away-observations landing near (lat, lon), with each cell's reference.

    SOURCE FILTERING IS NOT OPTIONAL FOR EVENT-LEVEL CLAIMS.

    Without it this returns everything that ever landed near the destination,
    which at any site with a substantial local population is dominated by that
    population rather than by the event. Measured: running this unfiltered on the
    Moscow->Crimea event (98 cells displaced 1,217 km) reported a median baseline
    of 14.13 km, because tens of thousands of LOCAL Crimean cells displaced ~14 km
    also land in the same box and outnumber the event 400 to 1.

    Passing src_lat10/src_lon10 restricts to cells whose reference lies in the
    event's source region, which is what makes the result a statement about the
    event. Omitting them characterises the SITE across its whole history, which
    is a different and usually less interesting question.
    """
    src = ""
    if src_lat10 is not None and src_lon10 is not None:
        src = (f" AND intDiv(rlat,10) = {src_lat10}"
               f" AND intDiv(rlon,10) = {src_lon10}")
    return ch_df(f"""
        SELECT plat, plon, rlat, rlon, obs, km, t_first, t_last,
               mcc, mnc, cell_type
        FROM spoof.away
        WHERE plat BETWEEN {int((lat - radius_deg) * 100)} AND {int((lat + radius_deg) * 100)}
          AND plon BETWEEN {int((lon - radius_deg) * 100)} AND {int((lon + radius_deg) * 100)}
          {src}
    """, settings=SETTINGS)


def decompose(df: pd.DataFrame, dlat: float, dlon: float) -> pd.DataFrame:
    """Along-/cross-track decomposition of each estimate on its own H->D axis.

    D is the event's decoy coordinate, shared by all cells. H is per-cell, so
    every cell gets its own axis -- which is the point: a shared destination with
    per-cell axes is a much stronger constraint than a shared axis would be.
    """
    h_lat, h_lon = df["rlat"].to_numpy() / 100.0, df["rlon"].to_numpy() / 100.0
    p_lat, p_lon = df["plat"].to_numpy() / 100.0, df["plon"].to_numpy() / 100.0

    cos0 = np.cos(np.radians(h_lat))
    # Vector H->D and H->P in the local planar frame centred on each cell's H.
    dx_d = np.radians(dlon - h_lon) * cos0 * R_EARTH_KM
    dy_d = np.radians(dlat - h_lat) * R_EARTH_KM
    dx_p = np.radians(p_lon - h_lon) * cos0 * R_EARTH_KM
    dy_p = np.radians(p_lat - h_lat) * R_EARTH_KM

    baseline = np.hypot(dx_d, dy_d)
    ok = baseline > 1e-6
    ux, uy = np.where(ok, dx_d / np.where(ok, baseline, 1), 0), np.where(ok, dy_d / np.where(ok, baseline, 1), 0)

    along = dx_p * ux + dy_p * uy          # km along H->D
    cross = np.abs(-dx_p * uy + dy_p * ux)  # km perpendicular

    out = df.copy()
    out["baseline_km"] = baseline
    out["along_km"] = along
    out["cross_km"] = np.where(baseline <= PLANAR_VALID_KM, cross, np.nan)
    # w: the implied spoofed fraction. 0 = cell at home, 1 = cell fully at decoy.
    out["w"] = np.where(ok, along / np.where(ok, baseline, 1), np.nan)
    return out


def report(out: pd.DataFrame, name: str) -> dict:
    w = np.repeat(out["w"].to_numpy(), out["obs"].to_numpy())
    cross = np.repeat(out["cross_km"].to_numpy(), out["obs"].to_numpy())
    along = np.repeat(out["along_km"].to_numpy(), out["obs"].to_numpy())
    base = np.repeat(out["baseline_km"].to_numpy(), out["obs"].to_numpy())

    fin = np.isfinite(cross)
    med_cross = float(np.median(cross[fin])) if fin.any() else float("nan")
    med_along = float(np.median(np.abs(along)))
    med_base = float(np.median(base))

    # Graded vs bimodal: share of mass at intermediate mixing fractions. A
    # weighted mean of two populations must put mass here; two distinct
    # populations of true observations cannot.
    inw = np.isfinite(w)
    mid = float(((w[inw] > 0.2) & (w[inw] < 0.8)).mean()) if inw.any() else float("nan")

    r = {
        "site": name,
        "obs": int(out["obs"].sum()),
        "cells": int(len(out)),
        "median_baseline_km": med_base,
        "median_along_km": med_along,
        "median_cross_km": med_cross,
        "cross_over_along_pct": 100.0 * med_cross / med_along if med_along else float("nan"),
        "mid_mass_frac": mid,
    }
    print(f"\n=== {name} ===")
    print(f"  cells {r['cells']:,}  observations {r['obs']:,}")
    print(f"  median baseline H->D      {med_base:9.2f} km")
    print(f"  median |along-track|      {med_along:9.2f} km")
    print(f"  median cross-track        {med_cross:9.3f} km "
          f"({r['cross_over_along_pct']:.2f}% of along-track)")
    verdict = ("GRADED -> weighted mixture (consistent with spoofing)"
               if mid > 0.25 else
               "BIMODAL -> NOT a mixture (identity replay, wholesale coordinate "
               "reassignment, or genuine relocation -- this test does not "
               "distinguish among those)")
    print(f"  mass at 0.2 < w < 0.8     {mid:9.3f}   {verdict}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True, help="decoy latitude")
    ap.add_argument("--lon", type=float, required=True, help="decoy longitude")
    ap.add_argument("--radius", type=float, default=0.05, help="degrees around decoy")
    ap.add_argument("--name", default="site")
    ap.add_argument("--src-lat10", type=int, default=None,
                    help="source region lat*10; REQUIRED for event-level claims")
    ap.add_argument("--src-lon10", type=int, default=None)
    args = ap.parse_args()

    df = fetch(args.lat, args.lon, args.radius, args.src_lat10, args.src_lon10)
    if df.empty:
        print("no away observations at that coordinate", file=sys.stderr)
        return 1
    report(decompose(df, args.lat, args.lon), args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
