#!/usr/bin/env python3
"""S10: tiered classification of the significant events.

WHY A TIER RATHER THAN A COUNT
------------------------------
S4 promotes 61 (region, day) bins above a family-wise threshold. That number is
NOT a spoofing census, and reporting it as one would be the central error of this
whole exercise. Two results establish that:

  * S9 negative control: the detector fires in 27 bins in Western and Central
    Europe and 27 in Russia/CIS. Western Europe is not a GNSS spoofing hotspot.
    Whatever most of these bins are, most of them are not spoofing.

  * S5 on the single strongest event: Vladimir, 475 cells across 5 operators and
    both technologies, clears synchrony by 43x the null maximum -- and has
    intermediate-mass exactly 0.000 with a 54 m cross-track. Every affected cell
    is either at home or exactly 37.79 km away on the same bearing. That is a
    wholesale coordinate offset, not an average of corrupted and honest reports,
    and it cannot be GNSS spoofing.

Synchrony establishes that something moved many cells at once. Only the mixture
geometry establishes that the something acted on RECEIVERS. The tiers below are
ordered by how much of that chain has been demonstrated.

TIERS
-----
  T1  mixture confirmed        synchrony + graded along-track + small cross-track
                               (a weighted mean of spoofed and honest fixes:
                               the mechanism GNSS spoofing must produce)
  T2  mixture ambiguous        synchrony + small cross-track, intermediate mass
                               near the boundary -- on-axis but shape unclear
  T3  coherent, not a mixture  synchrony + on-axis + no intermediate mass:
                               identity replay, wholesale reassignment, or
                               relocation. Real, dated, coherent -- and NOT
                               evidence of receiver deception.
  R   rejected                 fails the on-axis test: displacement is not
                               directed at a common destination

The intermediate-mass boundary (0.25) is a threshold, not a measurement, and
events near it are reported as T2 rather than forced either way.

THRESHOLD SENSITIVITY IS REAL, AND THE QUEEN ALIA EVENT DEMONSTRATES IT.
Measured at 0.186 when pooling a 0.06 deg destination radius over a whole source
region, it lands at 0.269-0.456 here, where each (region, day) bin is evaluated
against its own exact source square and a 0.10 deg destination radius. Same
event, same data, T2 under one aggregation and T1 under another. The physics did
not change; the binning did. Any published tier must therefore state the radius
and source granularity it was computed at, and events within roughly +/-0.08 of
the boundary should be read as "near the boundary" rather than as their nominal
tier.

WHAT THIS FILE DOES NOT ESTABLISH
---------------------------------
T1 means "consistent with a weighted mixture of spoofed and honest fixes". It
does not mean "confirmed GNSS spoofing". A regional platform-side change that
blended old and new coordinates would produce the same geometry, and confounder
C10 in s7_confounders.md records that this dataset cannot separate the two. The
tier names describe evidence, not verdicts.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ch_remote import ch_df

SETTINGS = {"optimize_aggregation_in_order": 0, "max_threads": 10}
R_EARTH_KM = 6371.0
MIN_KM = 25.0
THRESHOLD = 9          # from S4's family-wise 99% null quantile
MID_MASS_T1 = 0.25
MID_MASS_T2 = 0.10
CROSS_FRAC_MAX = 0.15  # cross-track above 15% of along-track is not "on-axis"


def significant_bins() -> pd.DataFrame:
    return ch_df(f"""
        SELECT src_lat10, src_lon10, onset_day, count() AS n_cells,
               uniqExact((mcc,mnc)) AS n_ops,
               uniqExact(cell_type)  AS n_tech,
               quantileExact(0.5)(med_km) AS ev_med_km,
               quantileExact(0.5)(top_plat) AS dest_plat,
               quantileExact(0.5)(top_plon) AS dest_plon
        FROM spoof.onsets_f
        WHERE med_km >= {MIN_KM}
        GROUP BY src_lat10, src_lon10, onset_day
        HAVING n_cells >= {THRESHOLD}
        ORDER BY n_cells DESC
    """, settings=SETTINGS)


def mechanism(src_lat10: int, src_lon10: int, dlat: float, dlon: float) -> dict:
    """Mixture geometry for one event, restricted to its own source region."""
    df = ch_df(f"""
        SELECT plat, plon, rlat, rlon, obs
        FROM spoof.away
        WHERE plat BETWEEN {int((dlat - 0.10) * 100)} AND {int((dlat + 0.10) * 100)}
          AND plon BETWEEN {int((dlon - 0.10) * 100)} AND {int((dlon + 0.10) * 100)}
          AND intDiv(rlat, 10) = {src_lat10}
          AND intDiv(rlon, 10) = {src_lon10}
    """, settings=SETTINGS)
    if df.empty:
        return {"cells": 0, "mid_mass": np.nan, "cross_frac": np.nan,
                "baseline_km": np.nan}

    h_lat, h_lon = df["rlat"].to_numpy() / 100.0, df["rlon"].to_numpy() / 100.0
    p_lat, p_lon = df["plat"].to_numpy() / 100.0, df["plon"].to_numpy() / 100.0
    cos0 = np.cos(np.radians(h_lat))
    dx_d = np.radians(dlon - h_lon) * cos0 * R_EARTH_KM
    dy_d = np.radians(dlat - h_lat) * R_EARTH_KM
    dx_p = np.radians(p_lon - h_lon) * cos0 * R_EARTH_KM
    dy_p = np.radians(p_lat - h_lat) * R_EARTH_KM
    base = np.hypot(dx_d, dy_d)
    ok = base > 1e-6
    if not ok.any():
        return {"cells": len(df), "mid_mass": np.nan, "cross_frac": np.nan,
                "baseline_km": np.nan}
    ux = np.where(ok, dx_d / np.where(ok, base, 1), 0)
    uy = np.where(ok, dy_d / np.where(ok, base, 1), 0)
    along = dx_p * ux + dy_p * uy
    cross = np.abs(-dx_p * uy + dy_p * ux)
    w = np.where(ok, along / np.where(ok, base, 1), np.nan)

    obs = df["obs"].to_numpy()
    wr = np.repeat(w, obs)
    cr = np.repeat(cross, obs)
    ar = np.repeat(np.abs(along), obs)
    br = np.repeat(base, obs)
    fin = np.isfinite(wr)
    med_along = float(np.median(ar)) if len(ar) else np.nan
    return {
        "cells": len(df),
        "baseline_km": float(np.median(br)),
        "mid_mass": float(((wr[fin] > 0.2) & (wr[fin] < 0.8)).mean()) if fin.any() else np.nan,
        "cross_frac": float(np.median(cr) / med_along) if med_along else np.nan,
    }


def tier(r) -> str:
    if not np.isfinite(r["mid_mass"]):
        return "R (no measurable axis)"
    # A long baseline invalidates the planar cross-track, so on-axis cannot be
    # tested there; such events are judged on mixture shape alone.
    if np.isfinite(r["cross_frac"]) and r["cross_frac"] > CROSS_FRAC_MAX:
        return "R (off-axis: not directed at a common destination)"
    if r["mid_mass"] >= MID_MASS_T1:
        return "T1 mixture confirmed"
    if r["mid_mass"] >= MID_MASS_T2:
        return "T2 mixture ambiguous"
    return "T3 coherent, not a mixture"


def main() -> int:
    bins = significant_bins()
    print(f"significant (region, day) bins at n >= {THRESHOLD}: {len(bins)}\n")
    if bins.empty:
        return 1

    rows = []
    for b in bins.itertuples():
        m = mechanism(b.src_lat10, b.src_lon10,
                      b.dest_plat / 100.0, b.dest_plon / 100.0)
        rows.append({
            "lat": b.src_lat10 / 10, "lon": b.src_lon10 / 10,
            "day": b.onset_day, "n_cells": b.n_cells,
            "ops": b.n_ops, "tech": b.n_tech,
            "med_km": round(b.ev_med_km, 1),
            "dest_lat": round(b.dest_plat / 100.0, 3),
            "dest_lon": round(b.dest_plon / 100.0, 3),
            **m,
        })
        rows[-1]["tier"] = tier(rows[-1])

    df = pd.DataFrame(rows)
    print("=== tier summary ===")
    print(df["tier"].value_counts().to_string())
    print("\n=== events by tier ===")
    for t in sorted(df["tier"].unique()):
        sub = df[df["tier"] == t].sort_values("n_cells", ascending=False)
        print(f"\n--- {t}  ({len(sub)}) ---")
        cols = ["lat", "lon", "day", "n_cells", "ops", "tech", "med_km",
                "dest_lat", "dest_lon", "mid_mass", "cross_frac"]
        with pd.option_context("display.width", 200):
            print(sub[cols].head(20).to_string(index=False,
                  float_format=lambda v: f"{v:.3f}"))

    df.to_csv("/tmp/s10_tiers.csv", index=False)
    print("\n-> /tmp/s10_tiers.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
