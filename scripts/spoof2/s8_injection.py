#!/usr/bin/env python3
"""S8: calibration by synthetic injection.

WHY THIS STAGE IS THE ONE THAT MATTERS
--------------------------------------
Every stage before this produces a COUNT. A count is not a measurement. Without
a sensitivity curve there is no way to say whether "61 significant bins" means
the world contains about 61 events or about 6,000, and no way for a reader to
know whether a null result in some region means no spoofing or no sensitivity.

Injection supplies the two numbers that turn a census into a measurement:

  * SENSITIVITY -- probability of recovering an event as a function of its size
    (number of affected cells), strength (mixing fraction w), and displacement.
    This is what licenses statements about what is NOT there, including the
    claim that the method is blind to jamming.
  * FALSE POSITIVE RATE -- how often the pipeline promotes a region where
    nothing was injected. S4's permutation null bounds the synchrony statistic
    specifically; this bounds the pipeline end to end.

WHAT THIS IMPLEMENTATION DOES AND DOES NOT EXERCISE
---------------------------------------------------
Being exact about this matters, because an injection test that silently skips
the stages most likely to fail is worse than none.

  EXERCISES: the detection statistic and its threshold -- i.e. whether an event
    of a given size and displacement clears the S4 family-wise threshold of 9
    cells in one (region, day), given the real background in that region.

  DOES NOT EXERCISE: S1 reference estimation or S2 away-detection. Synthetic
    cells are placed directly into the onset population with a known reference,
    so the test cannot detect the failure where a campaign is strong enough to
    CAPTURE a cell's reference and invert the sign. That is precisely the failure
    S1 was redesigned to prevent, so it must not be assumed away -- an end-to-end
    variant that rebuilds S1/S2 over cell.geos UNION synthetic observations is
    required before any sensitivity curve is published. It costs a full rebuild
    (~30 min) per configuration, hence a handful of anchors rather than a sweep.

  Also not exercised: S5 and S6. Synthetic cells are generated FROM the mixture
    model S5 tests for, so S5 would pass by construction -- a circularity, not a
    result. Sensitivity here is sensitivity of DETECTION, not of classification.

THE INJECTION MODEL
-------------------
For a chosen source region and decoy D, take n_cells real cells whose reference
lies in the region and place each at

    P = (1 - w) H + w D

with w drawn per cell from a Beta distribution rather than fixed, because real
campaigns show a spread of mixing fractions and a fixed w would make recovery
easier than it is. Onsets are spread over a ramp interval. Only cells whose
resulting displacement exceeds the 25 km detection floor can contribute, which is
itself part of what the sweep measures: at small w, most injected cells fall
below the floor and the event is invisible however many cells it touches.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ch_remote import ch_df

SETTINGS = {"optimize_aggregation_in_order": 0, "max_threads": 10}
R_EARTH_KM = 6371.0
MIN_KM = 25.0            # detection floor used by the S3/S4 statistic


def pick_cells(lat: float, lon: float, half_deg: float, n: int, seed: int) -> pd.DataFrame:
    df = ch_df(f"""
        SELECT rlat, rlon
        FROM spoof.cellref
        WHERE rlat BETWEEN {int((lat - half_deg) * 100)} AND {int((lat + half_deg) * 100)}
          AND rlon BETWEEN {int((lon - half_deg) * 100)} AND {int((lon + half_deg) * 100)}
          AND n_months >= 6
        LIMIT {max(n * 6, n)}
    """, settings=SETTINGS)
    if df.empty:
        return df
    rng = np.random.default_rng(seed)
    take = min(n, len(df))
    return df.iloc[rng.choice(len(df), size=take, replace=False)].reset_index(drop=True)


def background(lat: float, lon: float, half_deg: float) -> pd.DataFrame:
    """Real onsets already present in the region, by day, above the floor.

    Injection is measured against the REAL background rather than an empty
    region: an event has to clear the threshold on top of whatever that region
    ordinarily produces, and regions differ enormously in that respect.
    """
    return ch_df(f"""
        SELECT src_lat10, src_lon10, onset_day, count() AS n
        FROM spoof.onsets_f
        WHERE med_km >= {MIN_KM}
          AND src_lat10 BETWEEN {int((lat - half_deg) * 10)} AND {int((lat + half_deg) * 10)}
          AND src_lon10 BETWEEN {int((lon - half_deg) * 10)} AND {int((lon + half_deg) * 10)}
        GROUP BY src_lat10, src_lon10, onset_day
    """, settings=SETTINGS)


def synthesize(cells: pd.DataFrame, dlat: float, dlon: float,
               w_mean: float, w_conc: float, ramp_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = len(cells)
    a = max(w_mean * w_conc, 1e-3)
    b = max((1 - w_mean) * w_conc, 1e-3)
    w = rng.beta(a, b, size=n)

    h_lat = cells["rlat"].to_numpy() / 100.0
    h_lon = cells["rlon"].to_numpy() / 100.0
    p_lat = (1 - w) * h_lat + w * dlat
    p_lon = (1 - w) * h_lon + w * dlon

    cos0 = np.cos(np.radians(h_lat))
    km = np.hypot(np.radians(p_lon - h_lon) * cos0,
                  np.radians(p_lat - h_lat)) * R_EARTH_KM

    return pd.DataFrame({
        "src_lat10": (cells["rlat"].to_numpy() // 10).astype(int),
        "src_lon10": (cells["rlon"].to_numpy() // 10).astype(int),
        "day_offset": rng.integers(0, max(ramp_days, 1), size=n),
        "km": km,
        "w_true": w,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--half", type=float, default=0.05)
    ap.add_argument("--decoy-lat", type=float, required=True)
    ap.add_argument("--decoy-lon", type=float, required=True)
    ap.add_argument("--ramp-days", type=int, default=1)
    ap.add_argument("--w-conc", type=float, default=6.0)
    ap.add_argument("--threshold", type=int, required=True,
                    help="family-wise threshold from S4 (do not guess)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-cells", default="5,10,20,40,80,160,320")
    ap.add_argument("--sweep-w", default="0.05,0.1,0.25,0.5,0.9")
    ap.add_argument("--name", default="site")
    args = ap.parse_args()

    bg = background(args.lat, args.lon, args.half)
    bg_by_bin = {(r.src_lat10, r.src_lon10, r.onset_day): r.n for r in bg.itertuples()}
    bg_peak = max(bg_by_bin.values()) if bg_by_bin else 0
    print(f"=== S8 injection sensitivity: {args.name} ===")
    print(f"real background in region: {len(bg_by_bin)} occupied (region,day) bins, "
          f"peak {bg_peak} cells")
    print(f"S4 family-wise threshold: {args.threshold} cells\n")

    rows = []
    for n_cells in [int(x) for x in args.sweep_cells.split(",")]:
        for w_mean in [float(x) for x in args.sweep_w.split(",")]:
            hits, above_floor = 0, []
            for t in range(args.trials):
                seed = args.seed + 1000 * t + n_cells
                cells = pick_cells(args.lat, args.lon, args.half, n_cells, seed)
                if cells.empty:
                    print("no cells in source box", file=sys.stderr)
                    return 1
                syn = synthesize(cells, args.decoy_lat, args.decoy_lon,
                                 w_mean, args.w_conc, args.ramp_days, seed)
                # Only injected cells clearing the displacement floor are visible.
                vis = syn[syn["km"] >= MIN_KM]
                above_floor.append(len(vis))
                if vis.empty:
                    continue
                counts = vis.groupby(["src_lat10", "src_lon10", "day_offset"]).size()
                # Add the region's real background for the busiest matching bin.
                peak = int(counts.max()) + bg_peak
                hits += peak >= args.threshold
            rate = hits / args.trials
            rows.append({"n_cells": n_cells, "w_mean": w_mean,
                         "median_visible": float(np.median(above_floor)),
                         "recovery_rate": rate})
            print(f"  n_cells={n_cells:4d}  w={w_mean:4.2f}  "
                  f"visible(median)={np.median(above_floor):6.1f}  "
                  f"recovery={rate:5.0%}")

    out = pd.DataFrame(rows)
    out.to_csv("/tmp/s8_sensitivity.csv", index=False)
    print("\nsensitivity -> /tmp/s8_sensitivity.csv")
    print("SCOPE: detection only. Does NOT exercise S1 reference estimation,")
    print("S2 away-detection, or S5/S6 classification. See module docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
