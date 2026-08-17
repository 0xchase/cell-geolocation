-- S0: Observation model and exposure normalization.
--
-- The crawl is not a uniform sampler. Monthly observation volume ranges from
-- 703M (2024-02) to 5.29B (2024-08), an 8x swing, and 2026-04 holds only 597M.
-- Any temporal statistic computed as a raw count therefore measures crawl
-- intensity as much as it measures the world. This is not hypothetical: the
-- previous analysis of this dataset reported an "early-2026 crawl sweep" that
-- did not exist -- it was an artifact of counting without a denominator.
--
-- Everything downstream divides by one of these two exposure tables.
--
--   exposure_sq   -- how much crawl attention a GRID SQUARE received per month.
--                    Denominator for landing rates (how often observations fall
--                    at a candidate decoy), because a displaced observation
--                    lands away from its cell's home.
--   exposure_cell -- how often each CELL was sampled per month. Denominator for
--                    per-cell episode statistics and for the S4 permutation
--                    null, which must preserve each cell's own cadence.
--
-- Grid convention matches the rest of the project: plat = round(lat*100),
-- plon = round(lon*100), i.e. 0.01 deg ~ 1.1 km.

CREATE DATABASE IF NOT EXISTS spoof;

-- ---------------------------------------------------------------------------
-- S0.1  Candidate cells.
--
-- A cell whose position estimates never spread cannot host a displacement
-- episode, so the population is prefiltered on the bounding box already carried
-- by cell.summary_full. This costs no scan of cell.geos.
--
-- The threshold is 5 km, NOT the 10 km used by the earlier pipeline. 10 km is
-- defensible as a *displacement* threshold but wrong as a *prefilter*, because
-- a partially spoofed cell moves less than the decoy distance: if a fraction w
-- of reports are dragged to a decoy at distance D, the platform's estimate moves
-- only w*D. A campaign capturing 20% of reports to a decoy 50 km away moves the
-- estimate 10 km and sits exactly on the old cutoff. 5 km recovers the 6.61M
-- cells in the 5-10 km band that the old filter could never see.
--
-- This is a recall-oriented superset. Rejection happens downstream on measured
-- quantities, not here. S8 (synthetic injection) measures what it still misses.
--
-- Population at 5 km: 9,456,471 cells / 5.39B observations.
CREATE TABLE IF NOT EXISTS spoof.candidates
ENGINE = MergeTree ORDER BY (mcc, mnc, lac, cid, cell_type)
AS
SELECT
    mcc, mnc, lac, cid, cell_type,
    obs, n_pos, first_seen, last_seen,
    glat, glon,
    greatCircleDistance(lon_min, lat_min, lon_max, lat_max) / 1000 AS bbox_km
FROM cell.summary_full
WHERE greatCircleDistance(lon_min, lat_min, lon_max, lat_max) / 1000 > 5;

-- ---------------------------------------------------------------------------
-- S0.2  Per-square monthly exposure (all cells, not just candidates).
--
-- Must cover the whole corpus: the denominator for "how unusual is it that N
-- cells landed here this month" is the total crawl attention at that square,
-- which includes observations from non-candidate cells.
CREATE TABLE IF NOT EXISTS spoof.exposure_sq
ENGINE = MergeTree ORDER BY (plat, plon, month)
AS
SELECT
    toInt32(round(lat * 100))  AS plat,
    toInt32(round(lon * 100))  AS plon,
    toStartOfMonth(timestamp)  AS month,
    count()                    AS obs
FROM cell.geos
GROUP BY plat, plon, month;

-- ---------------------------------------------------------------------------
-- S0.3  Per-cell monthly exposure -- NOT built here.
--
-- The per-cell sampling cadence that S4's permutation null needs is exactly the
-- `obs` column of spoof.cellmonth, which S1 must build anyway to compute each
-- cell's monthly modal position. Materialising it separately would spend a
-- second full pass over the 63.34B-row cell.geos for a column we already get
-- for free. See s1_reference.sql.
