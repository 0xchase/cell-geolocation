-- S3: Events.
--
-- S3 and S4 are the same operation, and separating them was an error in my
-- original stage list. An "event" is not a thing you find first and then test
-- for synchrony: an event IS a synchrony spike. Defining it any other way
-- reintroduces exactly the bias this methodology exists to remove -- you would
-- be picking clusters by spatial concentration and only then asking about time,
-- which is the old attractor pipeline with extra steps.
--
-- So the detection statistic is defined here, and s4_null.sql supplies its null
-- distribution.
--
-- THE UNIT: (source region, onset time bucket).
--
--   NOT (destination, time). Anchoring on the destination is what limited the
--   old pipeline to point decoys. A spoofer that walks its decoy, alternates
--   between decoys, or induces a coherent offset field produces no concentrated
--   destination at all, yet its SOURCE region and its TIMING are as sharp as
--   any static decoy's -- because the source region is just "wherever the
--   transmitter's signal reaches", and the timing is when it was switched on.
--
--   Anchoring on the source therefore detects the transmitter rather than the
--   decoy, which is both more general and closer to the physical claim we want
--   to make. Destination structure is recovered afterwards (S3.3) and becomes a
--   descriptive property of an event -- point-like, fan, or field -- rather than
--   a precondition for finding it.
--
-- Source region granularity is 0.1 deg (~11 km), coarser than the 0.01 deg
-- working grid, because the affected population is everything within radio
-- range of the transmitter, which is a regional not a per-square quantity.

-- ---------------------------------------------------------------------------
-- S3.1  Per-cell onset, attributed to a source region.
--
-- A cell contributes ONE onset: the first time we observed it away from its
-- reference. Cells with no away observations do not appear, but they are not
-- discarded -- S3.2 needs them as the denominator, and they come from
-- spoof.candidates.
CREATE TABLE IF NOT EXISTS spoof.onsets
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10, onset_day)
AS
SELECT
    a.mcc AS mcc, a.mnc AS mnc, a.lac AS lac, a.cid AS cid,
    a.cell_type AS cell_type,
    -- Source region: the cell's own reference position, coarsened to ~11 km.
    intDiv(r.rlat, 10) AS src_lat10,
    intDiv(r.rlon, 10) AS src_lon10,
    toDate(a.t_first_away) AS onset_day,
    a.t_first_away AS onset_ts,
    a.away_obs AS away_obs,
    a.med_km   AS med_km,
    a.max_km   AS max_km,
    a.top_plat AS top_plat,
    a.top_plon AS top_plon
FROM spoof.cell_away AS a
INNER JOIN spoof.cellref AS r
    ON a.mcc = r.mcc AND a.mnc = r.mnc AND a.lac = r.lac
   AND a.cid = r.cid AND a.cell_type = r.cell_type
-- A reference built from too few months is not a reliable baseline, so the
-- displacement measured against it is not either.
WHERE r.n_months >= 3;

-- ---------------------------------------------------------------------------
-- S3.2  The synchrony statistic.
--
-- For each (source region, day): how many cells had their onset that day,
-- against how many cells in that region COULD have had an onset observed that
-- day. The denominator is the whole point -- a spike in onsets on a day when
-- the crawler happened to sweep the region is not an event, and counting
-- without normalising is precisely the error that produced the phantom
-- "early-2026 crawl sweep" in the previous analysis of this dataset.
--
-- excess is the observed onset count divided by the region's own mean daily
-- onset rate over the period it was actually being sampled. It is a descriptive
-- effect size; s4_null.sql converts it into a p-value by permutation, because
-- the sampling process is bursty and no closed-form null is trustworthy here.
CREATE TABLE IF NOT EXISTS spoof.sync
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10, onset_day)
AS
WITH
    region_totals AS (
        SELECT
            src_lat10, src_lon10,
            count()                       AS region_onsets,
            uniqExact(onset_day)           AS active_days,
            min(onset_day)                AS d_first,
            max(onset_day)                AS d_last
        FROM spoof.onsets
        GROUP BY src_lat10, src_lon10
    )
SELECT
    o.src_lat10 AS src_lat10,
    o.src_lon10 AS src_lon10,
    o.onset_day AS onset_day,
    count()                        AS n_onsets,
    uniqExact(o.mcc)                AS n_mcc,
    uniqExact((o.mcc, o.mnc))       AS n_operators,
    uniqExact(o.cell_type)          AS n_tech,
    quantileExact(0.5)(o.med_km)   AS med_km,
    -- Destination concentration: share of this day's cells landing on the single
    -- most-used destination square. Describes the event's morphology; it does
    -- not gate detection.
    max(dest_n) / count()          AS dest_concentration,
    t.region_onsets                AS region_onsets,
    t.d_last - t.d_first + 1       AS region_span_days,
    -- Effect size: today's onsets vs this region's mean daily onset rate.
    count() / (t.region_onsets / greatest(t.d_last - t.d_first + 1, 1)) AS excess
FROM
(
    SELECT *,
           count() OVER (PARTITION BY src_lat10, src_lon10, onset_day,
                                      top_plat, top_plon) AS dest_n
    FROM spoof.onsets
) AS o
INNER JOIN region_totals AS t
    ON o.src_lat10 = t.src_lat10 AND o.src_lon10 = t.src_lon10
GROUP BY src_lat10, src_lon10, onset_day, region_onsets, d_first, d_last;
