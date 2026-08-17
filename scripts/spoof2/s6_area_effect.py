#!/usr/bin/env python3
"""S6: area-effect consistency tests.

THE IDEA
--------
GNSS spoofing corrupts the HANDSET's receiver, not the network. A spoofed phone
reports every cell it can hear at the decoy, regardless of which operator runs
that cell or which radio technology it uses. Spoofing is therefore an
INDISCRIMINATE AREA EFFECT, and that indiscriminacy is measurable: the mix of
technologies and operators among affected cells should match the mix present in
the affected area, up to sampling noise.

Anything that is NOT an area effect breaks this. A misconfigured distributed
antenna system, an operator's bad location backhaul, a vendor's test equipment,
a single mislabelled base station -- all of these are specific to one operator or
one technology, and all of them show a mix that departs sharply from the local
baseline.

This is a genuinely free discriminator: `cell_type` and `mcc`/`mnc` are already
on every row of every table, no new scan is needed, and it tests a property that
follows directly from the physical mechanism rather than from a tuned threshold.

WHAT IT DOES AND DOES NOT SHOW
------------------------------
Consistency with the local baseline is NECESSARY for spoofing but not
sufficient -- a region-wide platform artifact would also be indiscriminate. It is
used here to REJECT operator- and equipment-specific explanations, and it is
reported against exactly that confounder and no other. The tests that separate
spoofing from a platform artifact are S4 (synchrony: an artifact has no
switch-on) and S5 (mixture geometry: an artifact scatters isotropically).

THE STATISTIC
-------------
G-test (likelihood-ratio chi-square) of affected counts against baseline
proportions. G is preferred over Pearson's chi-square because these tables have
many low-count categories in the tail (operators with a handful of cells), where
G is better behaved. Categories with an expected count below 5 are pooled into
"other" before testing, which is the standard remedy and is reported so the
pooling is visible rather than silent.

A LOW G (high p) SUPPORTS the area-effect reading. That inverts the usual
reading of a significance test, so the direction is stated explicitly wherever it
is reported: we are looking for FAILURE TO REJECT the hypothesis that the
affected population is a random draw from the local one.
"""

from __future__ import annotations

import argparse
import sys

import math

import numpy as np
import pandas as pd

from ch_remote import ch_df


def _chi2_sf(x: float, k: int) -> float:
    """Upper tail of the chi-square distribution, Q(k/2, x/2).

    Implemented here rather than imported: the project venv has numpy and pandas
    but not scipy, and a single special function is not worth a dependency. This
    is the standard series/continued-fraction pair for the regularised incomplete
    gamma function (Numerical Recipes 6.2), which is exact to double precision
    over the range these G statistics reach.
    """
    if x <= 0 or k <= 0:
        return 1.0
    a, xx = k / 2.0, x / 2.0
    gln = math.lgamma(a)
    if xx < a + 1.0:
        # Series expansion for P(a, x); Q = 1 - P.
        ap, term, total = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            term *= xx / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-xx + a * math.log(xx) - gln))
    # Continued fraction for Q(a, x) directly.
    tiny = 1e-300
    b, c, d = xx + 1.0 - a, 1.0 / tiny, 1.0 / (xx + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-xx + a * math.log(xx) - gln)

SETTINGS = {"optimize_aggregation_in_order": 0, "max_threads": 12}


def affected(lat: float, lon: float, half_deg: float, d0: str, d1: str) -> pd.DataFrame:
    """Cells whose REFERENCE lies in the source box and which were displaced in window."""
    return ch_df(f"""
        SELECT r.mcc AS mcc, r.mnc AS mnc, a.cell_type AS cell_type, count() AS n
        FROM spoof.cell_away AS a
        INNER JOIN spoof.cellref AS r
          ON a.mcc=r.mcc AND a.mnc=r.mnc AND a.lac=r.lac
         AND a.cid=r.cid AND a.cell_type=r.cell_type
        WHERE r.rlat BETWEEN {int((lat - half_deg) * 100)} AND {int((lat + half_deg) * 100)}
          AND r.rlon BETWEEN {int((lon - half_deg) * 100)} AND {int((lon + half_deg) * 100)}
          AND a.t_first_away >= toDateTime('{d0}')
          AND a.t_first_away <  toDateTime('{d1}')
        GROUP BY mcc, mnc, cell_type
    """, settings=SETTINGS)


def baseline(lat: float, lon: float, half_deg: float, d0: str, d1: str) -> pd.DataFrame:
    """MATCHED TEMPORAL CONTROL: displaced cells in the SAME box, OUTSIDE the window.

    An earlier version of this function used the local cell census
    (cell.summary_full) on the reasoning that spoofing is indiscriminate and
    should therefore mirror the infrastructure present. That reasoning is wrong,
    and the data falsified it: at Sheremetyevo the affected population is 76.2%
    LTE against a census that is 31.1% LTE, and restricting the census to S0
    candidates moves it only to 36.5% -- so candidate selection explains almost
    none of the gap.

    The premise was too naive. Spoofing is indiscriminate over WHAT THE HANDSET
    HEARS AND REPORTS, not over what is deployed. The devices contributing GNSS
    fixes are modern smartphones that camp on LTE and report LTE neighbours far
    more readily than legacy GSM, so the reported mix is skewed toward LTE before
    any attacker is involved. Comparing against a census measures that reporting
    propensity, not the event.

    A matched control absorbs it: the same region's displaced-cell mix outside
    the event window is subject to identical reporting biases, identical
    candidate selection, and identical local deployment. What remains is whether
    THIS event drew from a different population than the region's ordinary
    displacement does -- which is the question actually worth asking.
    """
    return ch_df(f"""
        SELECT r.mcc AS mcc, r.mnc AS mnc, a.cell_type AS cell_type, count() AS n
        FROM spoof.cell_away AS a
        INNER JOIN spoof.cellref AS r
          ON a.mcc=r.mcc AND a.mnc=r.mnc AND a.lac=r.lac
         AND a.cid=r.cid AND a.cell_type=r.cell_type
        WHERE r.rlat BETWEEN {int((lat - half_deg) * 100)} AND {int((lat + half_deg) * 100)}
          AND r.rlon BETWEEN {int((lon - half_deg) * 100)} AND {int((lon + half_deg) * 100)}
          AND NOT (a.t_first_away >= toDateTime('{d0}')
                   AND a.t_first_away < toDateTime('{d1}'))
        GROUP BY mcc, mnc, cell_type
    """, settings=SETTINGS)


def g_test(obs: pd.Series, base: pd.Series, label: str) -> dict:
    """G-test of obs against proportions implied by base, pooling small expectations."""
    keys = sorted(set(obs.index) | set(base.index))
    o = np.array([obs.get(k, 0) for k in keys], dtype=float)
    b = np.array([base.get(k, 0) for k in keys], dtype=float)
    if b.sum() == 0 or o.sum() == 0:
        return {"test": label, "G": np.nan, "df": 0, "p": np.nan, "note": "empty"}

    e = b / b.sum() * o.sum()
    small = e < 5
    n_pooled = int(small.sum())
    if n_pooled:
        o = np.append(o[~small], o[small].sum())
        e = np.append(e[~small], e[small].sum())
    keep = e > 0
    o, e = o[keep], e[keep]
    if len(o) < 2:
        return {"test": label, "G": np.nan, "df": 0, "p": np.nan,
                "note": "fewer than 2 usable categories"}

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(o > 0, o * np.log(o / e), 0.0)
    G = float(2 * terms.sum())
    df = len(o) - 1
    p = float(_chi2_sf(G, df))
    return {"test": label, "G": G, "df": df, "p": p,
            "categories": len(o), "pooled_into_other": n_pooled}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True, help="source region centre")
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--half", type=float, default=0.5, help="half-width in degrees")
    ap.add_argument("--from", dest="d0", required=True, help="YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--to", dest="d1", required=True)
    ap.add_argument("--name", default="site")
    args = ap.parse_args()

    aff = affected(args.lat, args.lon, args.half, args.d0, args.d1)
    bas = baseline(args.lat, args.lon, args.half, args.d0, args.d1)
    if aff.empty:
        print("no affected cells in that box/window", file=sys.stderr)
        return 1

    print(f"=== S6 area-effect consistency: {args.name} ===")
    print(f"affected cells {int(aff['n'].sum()):,}   "
          f"matched control (same box, outside window) {int(bas['n'].sum()):,}")
    print("\nReading: HIGH p SUPPORTS the area-effect (spoofing) reading -- we are")
    print("looking for failure to reject 'affected is a random draw from local'.")
    print("LOW p indicates an operator- or technology-specific cause instead.\n")

    rows = [
        g_test(aff.groupby("cell_type")["n"].sum(),
               bas.groupby("cell_type")["n"].sum(), "technology (gsm/lte/nr)"),
        g_test(aff.groupby("mcc")["n"].sum(),
               bas.groupby("mcc")["n"].sum(), "country (mcc)"),
        g_test(aff.groupby(["mcc", "mnc"])["n"].sum(),
               bas.groupby(["mcc", "mnc"])["n"].sum(), "operator (mcc,mnc)"),
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n-- technology detail (share of population) --")
    at = aff.groupby("cell_type")["n"].sum()
    bt = bas.groupby("cell_type")["n"].sum()
    det = pd.DataFrame({
        "affected": at, "affected_pct": 100 * at / at.sum(),
        "baseline": bt, "baseline_pct": 100 * bt / bt.sum(),
    }).fillna(0)
    print(det.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
