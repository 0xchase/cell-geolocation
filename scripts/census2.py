#!/usr/bin/env python3
"""Site-level global census of GNSS-spoofing decoys.

Two corrections over the naive per-square census:

1. **Squares are not sites.** The platform's coordinate for a cell is an average,
   so cells dragged to a decoy scatter over several km around it. A single decoy
   therefore lights up a patch of adjacent 0.01-deg squares. We agglomerate
   adjacent squares into sites before counting anything.

2. **The displacement floor must not be 25 km.** Real campaigns place receivers
   on a nearby airport: Moscow handsets land at Sheremetyevo ~20 km away. A 25 km
   floor throws those out. We lower it to 12 km — still far beyond crowdsourced
   GNSS scatter and beyond the reach of any single cell — and lean on two other
   properties to exclude ordinary aggregation:

     - a **datable onset**: an artifact of the positioning system is present from
       the start of the record; a transmitter switches on.
     - **source coherence**: the contributing cells' homes form one region.
"""

from __future__ import annotations

import sys

import ast

import numpy as np
import pandas as pd

from ch_remote import ch_df

DATASET_START = pd.Timestamp("2023-11-04")
ONSET_GRACE_DAYS = 45          # onset must be later than this to count as a switch-on
LEAD_DAYS_MIN = 30             # cells must have been known this long before being displaced
CONCENTRATION_MIN = 0.10       # >=10% of a site's displaced cells on its peak square
ANISO_MIN = 0.60               # sources must lie in one direction, not surround the site
SPREAD_COHERENT_KM = 600.0
DIST_FLOOR_KM = 12.0
PEAK_RADIUS_DEG = 0.06         # a site owns squares within ~6 km of its peak
DIST_CEIL_KM = 1000.0          # beyond this, 'displacement' is identity replay, not a GNSS decoy


def parse_mccs(v):
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    try:
        return [int(x) for x in ast.literal_eval(str(v))]
    except Exception:
        return []


def load() -> pd.DataFrame:
    df = ch_df("""
        SELECT a.plat AS plat, a.plon AS plon, a.plat/100 AS lat, a.plon/100 AS lon,
               a.cells AS cells, a.obs AS obs, a.n_mcc AS n_mcc, a.top_mcc AS top_mcc,
               a.med_km AS med_km, a.p90_km AS p90_km, a.src_spread_km AS src_spread_km,
               a.src_lat AS src_lat, a.src_lon AS src_lon,
               a.t_start AS t_start, a.t_end AS t_end,
               ifNull(l.lead_days, 0) AS lead_days,
               ifNull(n.anisotropy, 0) AS anisotropy
        FROM cell.attractors AS a
        LEFT JOIN cell.attr_lead AS l ON a.plat=l.plat AND a.plon=l.plon
        LEFT JOIN cell.attr_aniso AS n ON a.plat=n.plat AND a.plon=n.plon
        WHERE a.cells >= 40""")
    df["top_mcc"] = df["top_mcc"].map(parse_mccs)
    # Country the site physically sits in, and the country each contributing MCC
    # belongs to. A decoy that pulls FOREIGN cells onto domestic soil is the
    # least ambiguous signature available: neither RF propagation nor a local
    # aggregation artifact can move another country's towers across a border.
    iso = ch_df("SELECT mcc, iso FROM geo.mcc_iso")
    mcc2iso = dict(zip(iso["mcc"], iso["iso"], strict=False))
    site_cc = ch_df("""
        SELECT a.plat AS plat, a.plon AS plon, ifNull(c.country_iso,'') AS site_iso
        FROM cell.attractors AS a
        LEFT JOIN cell.coord_a0 AS c ON a.plat=c.klat AND a.plon=c.klon
        WHERE a.cells >= 40""")
    df = df.merge(site_cc, on=["plat", "plon"], how="left")
    df["site_iso"] = df["site_iso"].fillna("")
    def is_foreign(r):
        if not r["site_iso"]:
            return False                     # site country unknown -> cannot judge
        for m in r["top_mcc"]:
            home = mcc2iso.get(m)
            if home and home != r["site_iso"]:
                return True                  # only count MCCs we can actually map
        return False

    df["foreign"] = df.apply(is_foreign, axis=1)
    return df


def cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Assign squares to sites by non-maximum suppression, not single linkage.

    Single-linkage agglomeration chains: in a metropolitan area the genuine decoy
    square is connected through a carpet of weak neighbouring squares to every
    other hit in the region, and the decoy disappears into a blob. (This is not
    hypothetical -- it swallowed the Queen Alia decoy into a 477-square, 97k-cell
    cluster spanning greater Amman.)

    Instead we treat a site as a LOCAL MAXIMUM of displaced-cell count: a square
    that no square within PEAK_RADIUS exceeds. Every square is then assigned to
    its nearest such peak. Sites cannot chain, and each keeps a well-defined
    centre whose share of the local population is meaningful.
    """
    df = df.sort_values("cells", ascending=False).reset_index(drop=True)
    r = int(PEAK_RADIUS_DEG * 100)
    taken: dict[tuple[int, int], int] = {}
    peaks: list[tuple[int, int, int]] = []      # (plat, plon, site_id)
    site_of = np.full(len(df), -1, dtype=int)

    for i, row in df.iterrows():
        la, lo = int(row.plat), int(row.plon)
        # Claim by an existing peak if we fall inside its radius.
        best, bestd = -1, None
        for pla, plo, sid in peaks:
            d = max(abs(pla - la), abs(plo - lo))
            if d <= r and (bestd is None or d < bestd):
                best, bestd = sid, d
        if best >= 0:
            site_of[i] = best
        else:
            sid = len(peaks)
            peaks.append((la, lo, sid))
            site_of[i] = sid
    df["site"] = site_of
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("site")
    out = pd.DataFrame({
        "squares": g.size(),
        "cells": g["cells"].sum(),
        "obs": g["obs"].sum(),
        "n_mcc": g["n_mcc"].max(),
        "med_km": g.apply(lambda s: np.average(s["med_km"], weights=s["cells"]), include_groups=False),
        "src_spread_km": g.apply(lambda s: np.average(s["src_spread_km"], weights=s["cells"]), include_groups=False),
        "lat": g.apply(lambda s: np.average(s["lat"], weights=s["cells"]), include_groups=False),
        "lon": g.apply(lambda s: np.average(s["lon"], weights=s["cells"]), include_groups=False),
        "t_start": g["t_start"].min(),
        "t_end": g["t_end"].max(),
        "mccs": g["top_mcc"].apply(lambda x: sorted({m for lst in x for m in lst})[:6]),
        "n_src_mcc": g["top_mcc"].apply(lambda x: len({m for lst in x for m in lst})),
        "lead_days": g.apply(lambda s: np.average(s["lead_days"], weights=s["cells"]), include_groups=False),
        # Concentration separates a STATIC DECOY (one coordinate absorbs most of
        # the displaced population) from DIFFUSE DEGRADATION (the displacement is
        # smeared over hundreds of squares, as partial jamming would produce).
        "peak_cells": g["cells"].max(),
        "anisotropy": g.apply(lambda s: np.average(s["anisotropy"], weights=s["cells"]), include_groups=False),
        "foreign": g["foreign"].max(),
        "site_iso": g["site_iso"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else ""),
    }).reset_index(drop=True)
    out["days"] = (pd.to_datetime(out["t_end"]) - pd.to_datetime(out["t_start"])).dt.days
    out["onset_day"] = (pd.to_datetime(out["t_start"]) - DATASET_START).dt.days
    out["concentration"] = out["peak_cells"] / out["cells"]
    return out


def classify(r) -> str:
    if r["src_spread_km"] > SPREAD_COHERENT_KM:
        return "equipment/test"
    if r["med_km"] < DIST_FLOOR_KM:
        return "local aggregation"
    if r["onset_day"] < ONSET_GRACE_DAYS:
        return "persistent artifact"
    # If the contributing cells were first seen at the same moment they appeared
    # here, we simply crawled the region for the first time -- not a switch-on.
    if r["lead_days"] < LEAD_DAYS_MIN:
        return "crawl-onset (indeterminate)"
    # A static decoy concentrates the displaced population onto one coordinate.
    # Displacement smeared over hundreds of squares is a different phenomenon --
    # consistent with degraded (jammed) fixes rather than a coherent false fix.
    # A decoy drags cells from ONE direction. If the contributing cells surround
    # the site instead, the displacement is ordinary positional scatter in a
    # dense area -- the dominant confounder in Japan, where the platform's
    # estimates for indoor/dense deployments wander 10-20 km in every direction.
    if r["anisotropy"] < ANISO_MIN:
        return "isotropic scatter"
    # Cross-border displacement is the strongest evidence and is not subject to
    # the resolution dilution that weakens the concentration measure.
    if r["foreign"]:
        return "STATIC DECOY (cross-border)"
    if r["concentration"] < CONCENTRATION_MIN:
        return "diffuse degradation"
    # A GNSS spoofer sets a locally plausible position. A "displacement" of
    # thousands of km is better explained by a transmitter re-broadcasting
    # another region's cell identities: there the position is correct and the
    # IDENTITY is false, which is the opposite failure and is not GNSS spoofing.
    if r["med_km"] > DIST_CEIL_KM:
        return "identity replay"
    return "STATIC DECOY"


def main() -> int:
    raw = load()
    if raw.empty:
        print("no attractors"); return 1
    sites = summarise(cluster(raw))
    sites["klass"] = sites.apply(classify, axis=1)
    sites = sites.sort_values("cells", ascending=False)

    print(f"attractor squares: {len(raw):,}   ->   sites: {len(sites):,}")
    print(sites["klass"].value_counts().to_string())
    print()
    d = sites[sites["klass"].str.startswith("STATIC DECOY")]
    print(f"=== STATIC DECOY sites: {len(d):,} "
          f"({d['cells'].sum():,} cells, {d['obs'].sum():,} observations) ===")
    cols = ["lat", "lon", "site_iso", "squares", "cells", "mccs", "med_km",
            "concentration", "anisotropy", "lead_days", "t_start"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 30):
        print(d.head(28)[cols].to_string(index=False))
    sites.to_csv("/tmp/sites.csv", index=False)
    print("\nall sites -> /tmp/sites.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
