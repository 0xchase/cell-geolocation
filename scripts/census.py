#!/usr/bin/env python3
"""Print the global spoofing census from cell.attractors, with classification.

Run after build_spoof_detector.sql completes. Reverse-geocodes each candidate
against cell.coord_geo so sites can be named rather than left as coordinates.
"""

from __future__ import annotations

import sys

import pandas as pd

from ch_remote import ch_df

SPREAD_COHERENT_KM = 600.0
DIST_DISPLACED_KM = 25.0


def classify(r) -> str:
    if r["src_spread_km"] > SPREAD_COHERENT_KM:
        return "equipment/test"
    if r["med_km"] < DIST_DISPLACED_KM:
        return "spillover/aggregation"
    return "SPOOF DECOY"


def main() -> int:
    min_cells = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    df = ch_df(f"""
        SELECT a.plat/100 AS lat, a.plon/100 AS lon,
               a.cells AS cells, a.obs AS obs, a.n_mcc AS n_mcc, a.top_mcc AS top_mcc,
               round(a.med_km) AS med_km, round(a.p90_km) AS p90_km,
               round(a.src_spread_km) AS src_spread_km,
               a.t_start AS t_start, a.t_end AS t_end,
               dateDiff('day', a.t_start, a.t_end) AS days,
               g.cc AS cc, g.city AS city, g.county AS county, g.country AS country
        FROM cell.attractors AS a
        LEFT JOIN cell.coord_geo AS g ON a.plat = g.klat AND a.plon = g.klon
        WHERE a.cells >= {min_cells}
        ORDER BY a.cells DESC""")
    if df.empty:
        print("no attractors")
        return 1
    df["klass"] = df.apply(classify, axis=1)

    print(f"total attractors (cells>={min_cells}): {len(df):,}")
    print(df["klass"].value_counts().to_string())
    print()
    for klass in ["SPOOF DECOY", "equipment/test", "spillover/aggregation"]:
        sub = df[df["klass"] == klass].head(22)
        if sub.empty:
            continue
        print(f"===== {klass} (top {len(sub)}) =====")
        cols = ["lat", "lon", "cells", "n_mcc", "top_mcc", "med_km",
                "src_spread_km", "days", "cc", "city", "t_start"]
        with pd.option_context("display.width", 250, "display.max_colwidth", 26):
            print(sub[cols].to_string(index=False))
        print()
    df.to_csv("/tmp/census.csv", index=False)
    print("full census -> /tmp/census.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
