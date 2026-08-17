#!/usr/bin/env python3
"""S4: permutation null for the S3 detection statistic.

WHAT IS BEING CALIBRATED
------------------------
The statistic that survived S3 is, for each (source region, day):

    n = number of cells whose FIRST observed displacement fell on that day,
        counting only cells that
          - were observed within 2 days before that onset   (C13 precision), and
          - were displaced at least MIN_KM from their reference

Both restrictions matter and the null must respect them, because they change the
population being permuted. An earlier draft of this script calibrated raw
spoof.onsets, which is a different and much larger population -- that null would
have been quietly wrong rather than loudly wrong.

THE NULL HYPOTHESIS
-------------------
H0: cells were displaced independently of one another, at times governed only by
    when we happened to look at them.

Rejecting H0 is exactly the claim "these cells moved TOGETHER", which is the
signature of an area effect -- one transmitter reaching many cells at once -- and
is what separates spoofing from per-cell noise.

WHY PERMUTATION AND NOT A CLOSED FORM
-------------------------------------
The sampling process is bursty in three independent ways: global crawl volume
swings ~8x month to month, regional coverage starts and stops, and per-cell
cadence varies. A Poisson null assumes all three away and would call every crawl
burst an event. That failure is not hypothetical -- it is what the first version
of this pipeline did, and what the previous analysis of this dataset did when it
reported an "early-2026 crawl sweep" that never existed.

Each cell's onset is redrawn from ITS OWN observed sampling distribution: a month
chosen with probability proportional to how often we sampled that cell that month
(spoof.cellmonth.obs), then a uniform day within the month. This preserves real
observability and destroys only cross-cell coincidence, which is the quantity
under test.

MULTIPLE TESTING
----------------
There are ~10^5 regions x ~10^3 days of opportunity. Each round therefore records
the GLOBAL MAXIMUM over all regions and days, and the upper quantile of that
distribution is a family-wise threshold: a region exceeding it is significant
after accounting for every comparison made anywhere in the world. Deliberately
conservative, which is the right trade for a published census.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ch_remote import ch_df

# ch_remote defaults optimize_aggregation_in_order=1, which serialises these
# aggregations onto ~2 cores. Override everywhere in this script.
SETTINGS = {"optimize_aggregation_in_order": 0, "max_threads": 10}

MIN_KM = 25.0            # displacement floor for the detection population
MAX_PRECISION_DAYS = 2   # C13: onset must be bracketed this tightly


def load_population() -> pd.DataFrame:
    """The cells whose onsets constitute the statistic."""
    return ch_df(f"""
        SELECT o.mcc AS mcc, o.mnc AS mnc, o.lac AS lac, o.cid AS cid,
               o.cell_type AS cell_type,
               o.src_lat10 AS src_lat10, o.src_lon10 AS src_lon10,
               o.onset_day AS onset_day
        FROM spoof.onsets_f AS o
        WHERE o.med_km >= {MIN_KM}
    """, settings=SETTINGS)


def load_cadence(pop: pd.DataFrame) -> pd.DataFrame:
    """Per-cell monthly sampling weights, restricted to the population."""
    return ch_df(f"""
        SELECT m.mcc AS mcc, m.mnc AS mnc, m.lac AS lac, m.cid AS cid,
               m.cell_type AS cell_type, m.month AS month, m.obs AS obs
        FROM spoof.cellmonth AS m
        WHERE (m.mcc, m.mnc, m.lac, m.cid, m.cell_type) IN (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM spoof.onsets_f WHERE med_km >= {MIN_KM}
        )
    """, settings=SETTINGS)


def build_sampler(pop: pd.DataFrame, cad: pd.DataFrame):
    """Flatten per-cell month weights into arrays for vectorised sampling.

    Returns (cell_index_of_row, day_offset_of_row, weight_of_row, region_of_cell)
    where rows are (cell, month) pairs sorted by cell.
    """
    key = ["mcc", "mnc", "lac", "cid", "cell_type"]
    pop = pop.copy()
    pop["cell_ix"] = np.arange(len(pop))

    cad = cad.merge(pop[key + ["cell_ix"]], on=key, how="inner")
    cad = cad.sort_values("cell_ix", kind="stable").reset_index(drop=True)

    month = pd.to_datetime(cad["month"])
    epoch = pd.Timestamp("2023-11-01")
    cad["month_start"] = (month - epoch).dt.days
    cad["month_len"] = (month + pd.offsets.MonthBegin(1) - month).dt.days

    region = pop["src_lat10"].to_numpy().astype(np.int64) * 100000 + pop["src_lon10"].to_numpy()
    return cad, region, epoch


def one_round(cad: pd.DataFrame, region: np.ndarray, rng: np.random.Generator) -> int:
    """Redraw every cell's onset from its own cadence; return the global max."""
    # Exponential race: argmin over a cell's months of -ln(u)/w picks a month
    # with probability proportional to w. Vectorised across all cells at once.
    w = cad["obs"].to_numpy(dtype=np.float64)
    keys = -np.log(rng.random(len(w)) + 1e-300) / np.maximum(w, 1e-9)

    cell_ix = cad["cell_ix"].to_numpy()
    n_cells = int(cell_ix.max()) + 1
    best = np.full(n_cells, np.inf)
    np.minimum.at(best, cell_ix, keys)
    chosen = keys == best[cell_ix]
    # Ties are vanishingly unlikely with continuous keys, but guard anyway by
    # taking the first winner per cell.
    first = np.zeros(n_cells, dtype=np.int64)
    idx = np.flatnonzero(chosen)
    first[cell_ix[idx]] = idx

    start = cad["month_start"].to_numpy()[first]
    length = cad["month_len"].to_numpy()[first]
    day = start + (rng.random(n_cells) * length).astype(np.int64)

    # (region, day) -> count, then the maximum. Composite key avoids a 2-D bin.
    composite = region.astype(np.int64) * 2000 + day
    _, counts = np.unique(composite, return_counts=True)
    return int(counts.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="/tmp/s4_null.csv")
    args = ap.parse_args()

    print("loading detection population ...", flush=True)
    pop = load_population()
    print(f"  cells: {len(pop):,}   regions: {pop.groupby(['src_lat10','src_lon10']).ngroups:,}")
    if pop.empty:
        print("empty population", file=sys.stderr)
        return 1

    print("loading per-cell cadence ...", flush=True)
    cad_raw = load_cadence(pop)
    print(f"  (cell, month) rows: {len(cad_raw):,}")

    cad, region, _ = build_sampler(pop, cad_raw)

    obs_counts = pop.groupby(["src_lat10", "src_lon10", "onset_day"]).size()
    observed_max = int(obs_counts.max())
    top = obs_counts.sort_values(ascending=False).head(10)
    print(f"\nobserved global max in one (region, day): {observed_max:,}")

    rng = np.random.default_rng(args.seed)
    maxima = np.array([one_round(cad, region, rng) for _ in range(args.rounds)])

    thr95, thr99, thr999 = (float(np.quantile(maxima, q)) for q in (0.95, 0.99, 0.999))
    p = (int((maxima >= observed_max).sum()) + 1) / (len(maxima) + 1)

    print(f"\nnull over {args.rounds} rounds (cadence-weighted resample)")
    print(f"  null global max: median {np.median(maxima):.0f}  "
          f"95% {thr95:.0f}  99% {thr99:.0f}  99.9% {thr999:.0f}  max {maxima.max():.0f}")
    print(f"  observed {observed_max:,}  ->  family-wise p {p:.4g}")
    print(f"\nDETECTION THRESHOLD (family-wise 99%): n >= {int(np.ceil(thr99))} cells "
          f"in one (region, day)")

    print("\nobserved (region, day) bins above the 99% threshold:")
    sig = obs_counts[obs_counts >= thr99].sort_values(ascending=False)
    print(f"  {len(sig):,} bins")
    out = sig.reset_index()
    out.columns = ["src_lat10", "src_lon10", "onset_day", "n_cells"]
    out["lat"] = out["src_lat10"] / 10
    out["lon"] = out["src_lon10"] / 10
    print(out.head(25).to_string(index=False))

    pd.DataFrame({"round": np.arange(len(maxima)), "global_max": maxima}).to_csv(
        args.out, index=False)
    out.to_csv("/tmp/s4_significant.csv", index=False)
    print(f"\nnull -> {args.out}   significant bins -> /tmp/s4_significant.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
