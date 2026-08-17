-- S3b: the synchrony statistic, normalised against independent exposure.
--
-- Replaces spoof.sync, whose `excess` normalised each region only against its
-- own mean daily onset rate and was therefore dominated by days on which the
-- CRAWLER was unusually busy (see s0b_exposure_daily.sql: 2024-04-03 produced
-- 207,522 onsets across 68,086 regions).
--
-- THE MODEL
--
--   expected(r, d) = onset_rate(r) * exposure(r, d)
--
-- where onset_rate(r) = region_onsets(r) / region_exposure(r) is the region's
-- own onsets per unit of sampling attention, and exposure(r, d) is observations
-- of that region's cells on that day. Both come from spoof.exposure_region_day,
-- built from cell.geos independently of the onsets -- unlike the provisional
-- global-factor correction used to preview this ranking, which divided onsets by
-- a quantity derived from onsets and was therefore circular.
--
--   excess(r, d) = n_onsets(r, d) / expected(r, d)
--
-- LEAD TIME (confounder C2)
--
-- A cell cannot be seen displaced before it is seen at all, so a region entering
-- the crawl for the first time manufactures a synchronised onset. lead_days is
-- the mean interval between a contributing cell first being KNOWN
-- (cellref.m_first) and its first away observation. Short lead means first
-- contact, not switch-on; such events are labelled rather than silently dropped,
-- because 2023-11-04 -- day one of the record -- is the second largest onset day
-- globally and must be visibly excluded rather than quietly missing.
--
-- WHY THIS IS BUILT AS STEPS RATHER THAN ONE QUERY
--
-- Two attempts as a single statement with CTEs both died at 10 GiB in
-- FillingRightJoinSide. ClickHouse's hash join materialises the right side, and
-- the exposure aggregate is 354.9M rows; reordering the joins by hand did not
-- help, because the planner materialises the CTE regardless of the written
-- order. Rather than keep guessing at the planner, the large input is reduced to
-- the rows that can possibly matter FIRST, as a real table.
--
-- The reduction is exact, not a sample: only (region, day) pairs that actually
-- contain an onset can appear in the output, and there are 4.14M of those
-- against 354.9M exposure rows. Everything joined afterwards is small.
--
-- exposure_region_day is a SummingMergeTree, so every read must aggregate --
-- background merges are not guaranteed complete.

-- Step 1: exposure restricted to (region, day) pairs that have onsets.
DROP TABLE IF EXISTS spoof.exp_onsetday;
CREATE TABLE spoof.exp_onsetday
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10, day)
AS
SELECT src_lat10, src_lon10, day, sum(obs) AS obs
FROM spoof.exposure_region_day
WHERE (src_lat10, src_lon10, day) IN (
    SELECT src_lat10, src_lon10, onset_day FROM spoof.onsets
)
GROUP BY src_lat10, src_lon10, day;

-- Step 2: total exposure per region (419k rows, pure aggregation, no join).
DROP TABLE IF EXISTS spoof.exp_region;
CREATE TABLE spoof.exp_region
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10)
AS
SELECT src_lat10, src_lon10, sum(obs) AS region_obs
FROM spoof.exposure_region_day
GROUP BY src_lat10, src_lon10;

-- Step 3: per (region, day) onset aggregate, with lead time.
DROP TABLE IF EXISTS spoof.ons_day;
CREATE TABLE spoof.ons_day
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10, onset_day)
AS
SELECT
    o.src_lat10 AS src_lat10, o.src_lon10 AS src_lon10,
    o.onset_day AS onset_day,
    count()                      AS n_onsets,
    uniqExact(o.mcc)             AS n_mcc,
    uniqExact((o.mcc, o.mnc))    AS n_operators,
    uniqExact(o.cell_type)       AS n_tech,
    quantileExact(0.5)(o.med_km) AS med_km,
    quantileExact(0.5)(o.max_km) AS max_km,
    avg(dateDiff('day', r.m_first, o.onset_ts)) AS lead_days
FROM spoof.onsets AS o
INNER JOIN spoof.cellref AS r
    ON o.mcc = r.mcc AND o.mnc = r.mnc AND o.lac = r.lac
   AND o.cid = r.cid AND o.cell_type = r.cell_type
GROUP BY src_lat10, src_lon10, onset_day;

-- Step 4: region onset totals (419k rows).
DROP TABLE IF EXISTS spoof.ons_region;
CREATE TABLE spoof.ons_region
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10)
AS
SELECT src_lat10, src_lon10, sum(n_onsets) AS region_onsets
FROM spoof.ons_day
GROUP BY src_lat10, src_lon10;

-- Step 5: assemble. Every input here is at most 4.14M rows.
DROP TABLE IF EXISTS spoof.sync2;
CREATE TABLE spoof.sync2
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10, onset_day)
AS
SELECT
    d.src_lat10 AS src_lat10,
    d.src_lon10 AS src_lon10,
    d.onset_day AS onset_day,
    d.n_onsets AS n_onsets,
    d.n_mcc AS n_mcc,
    d.n_operators AS n_operators,
    d.n_tech AS n_tech,
    d.med_km AS med_km,
    d.max_km AS max_km,
    d.lead_days AS lead_days,
    e.obs AS day_obs,
    rr.region_obs AS region_obs,
    ro.region_onsets AS region_onsets,
    (ro.region_onsets / rr.region_obs) * e.obs AS expected,
    d.n_onsets / greatest((ro.region_onsets / rr.region_obs) * e.obs, 1e-9) AS excess
FROM spoof.ons_day AS d
INNER JOIN spoof.exp_onsetday AS e
    ON d.src_lat10 = e.src_lat10 AND d.src_lon10 = e.src_lon10
   AND d.onset_day = e.day
INNER JOIN spoof.exp_region AS rr
    ON d.src_lat10 = rr.src_lat10 AND d.src_lon10 = rr.src_lon10
INNER JOIN spoof.ons_region AS ro
    ON d.src_lat10 = ro.src_lat10 AND d.src_lon10 = ro.src_lon10;
