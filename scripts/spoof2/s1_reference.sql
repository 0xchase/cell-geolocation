-- S1: Robust, time-resolved reference position.
--
-- The quantity every later stage measures against is "where the platform
-- normally believed this cell was". Getting that estimator right is the single
-- most load-bearing choice in the methodology, because a displacement is
-- defined relative to it, and a bad reference inverts the sign of the result.
--
-- Why not the all-time plurality position (the previous approach).
--
--   The old cell.cellhome took argMax(position, obs) over a cell's entire
--   history. That is a plurality, not a majority, and we measured the
--   consequence directly: 18,229,474 cells (14.5% of the corpus) have a home
--   square holding under 50% of their observations, and 7,606,227 hold under
--   35%. For those cells a sustained, heavily-sampled campaign can hold more
--   observations than the truth does, and the estimator silently adopts the
--   DECOY as home -- after which the cell's real position is what gets reported
--   as "displaced". 1,680,791 cells are homed on a square that is itself an
--   attractor, which is where that failure would live.
--
--   The reason it fails is that it counts observations, and the crawl is what
--   produces observations. Sampling intensity is not evidence about the world.
--
-- The estimator used here.
--
--   Two-level, so that time is the unit of evidence rather than sample count:
--
--     level 1  within each calendar month, take the MODAL 0.01-deg square
--              (the platform's consensus for that cell that month)
--     level 2  across months, take the COORDINATE-WISE MEDIAN of those modes,
--              each month weighted EQUALLY regardless of how often we sampled it
--
--   Equal weighting per month is the entire point. It makes the reference
--   resistant to a campaign that captures a minority of the RECORD's months,
--   however densely we happened to crawl during it. The breakdown point is 50%
--   of observed months per coordinate: an attacker must hold the cell for more
--   than half the months we observed it to move the reference at all.
--
--   Naming: this is the coordinate-wise (marginal) median, applied to latitude
--   and longitude separately. It is not the geometric median, which minimises
--   summed Euclidean distance and has no closed form in SQL. The marginal
--   median is what gives the 50%-per-coordinate breakdown point we rely on;
--   at these scales the difference between the two is far below the 1.1 km
--   grid resolution and is not worth an iterative solver.
--
-- Cells with few observed months have a fragile reference, so n_months is
-- carried and downstream stages impose a minimum rather than trusting it here.

-- ---------------------------------------------------------------------------
-- S1.1  Per-cell, per-month modal position.
--
-- Also the per-cell crawl cadence (the `obs` column), which is what S0.3 would
-- otherwise have cost a second full scan to produce, and which S4's permutation
-- null needs in order to redistribute onsets within each cell's own sampling
-- pattern.
--
-- Single pass over cell.geos: the inner aggregation counts observations per
-- (cell, month, square), the outer picks the modal square per (cell, month).
CREATE TABLE IF NOT EXISTS spoof.cellmonth
ENGINE = MergeTree ORDER BY (mcc, mnc, lac, cid, cell_type, month)
AS
SELECT
    mcc, mnc, lac, cid, cell_type, month,
    argMax(plat, c) AS mlat,      -- modal square for this cell this month
    argMax(plon, c) AS mlon,
    max(c)          AS mode_obs,  -- observations at the mode
    sum(c)          AS obs,       -- crawl cadence for this cell this month
    count()         AS n_pos      -- distinct squares this month
FROM
(
    SELECT
        mcc, mnc, lac, cid, cell_type,
        toStartOfMonth(timestamp)  AS month,
        toInt32(round(lat * 100))  AS plat,
        toInt32(round(lon * 100))  AS plon,
        count()                    AS c
    FROM cell.geos
    WHERE (mcc, mnc, lac, cid, cell_type) IN (
        SELECT mcc, mnc, lac, cid, cell_type FROM spoof.candidates
    )
    GROUP BY mcc, mnc, lac, cid, cell_type, month, plat, plon
)
GROUP BY mcc, mnc, lac, cid, cell_type, month;

-- ---------------------------------------------------------------------------
-- S1.2  Per-cell reference position: coordinate-wise median over monthly modes.
CREATE TABLE IF NOT EXISTS spoof.cellref
ENGINE = MergeTree ORDER BY (mcc, mnc, lac, cid, cell_type)
AS
SELECT
    mcc, mnc, lac, cid, cell_type,
    toInt32(round(quantileExact(0.5)(mlat))) AS rlat,
    toInt32(round(quantileExact(0.5)(mlon))) AS rlon,
    count()      AS n_months,
    sum(obs)     AS obs,
    min(month)   AS m_first,
    max(month)   AS m_last
FROM spoof.cellmonth
GROUP BY mcc, mnc, lac, cid, cell_type;

-- ---------------------------------------------------------------------------
-- S1.3  Reference stability, and the sign-inversion audit.
--
-- stab_frac is the fraction of a cell's observed months whose mode sits within
-- 2 km of the reference. A stable cell with an episode has stab_frac near 1
-- (the episode occupies a minority of months, which is what makes it an
-- episode). stab_frac near 0.5 means the cell spends about half its life in
-- each of two places -- either a genuine relocation or a campaign long enough
-- to threaten the estimator, and in both cases the reference is not trustworthy
-- and the cell must be handled explicitly rather than silently.
--
-- This table is also the audit that the old pipeline could not perform: it
-- measures directly how many cells the plurality estimator would have gotten
-- wrong, instead of assuming the number is small.
CREATE TABLE IF NOT EXISTS spoof.cellref_stability
ENGINE = MergeTree ORDER BY (mcc, mnc, lac, cid, cell_type)
AS
SELECT
    m.mcc AS mcc, m.mnc AS mnc, m.lac AS lac, m.cid AS cid,
    m.cell_type AS cell_type,
    count() AS n_months,
    countIf(greatCircleDistance(m.mlon / 100, m.mlat / 100,
                                r.rlon / 100, r.rlat / 100) / 1000 <= 2.0)
        / count() AS stab_frac,
    max(greatCircleDistance(m.mlon / 100, m.mlat / 100,
                            r.rlon / 100, r.rlat / 100) / 1000) AS max_month_km
FROM spoof.cellmonth AS m
INNER JOIN spoof.cellref AS r
    ON m.mcc = r.mcc AND m.mnc = r.mnc AND m.lac = r.lac
   AND m.cid = r.cid AND m.cell_type = r.cell_type
GROUP BY mcc, mnc, lac, cid, cell_type;
