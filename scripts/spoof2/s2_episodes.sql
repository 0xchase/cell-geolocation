-- S2: Displacement episodes.
--
-- The atomic unit of this methodology. An EPISODE is one cell, one time
-- interval, during which the platform's estimate for that cell sat away from
-- the cell's reference position (S1) and then, usually, returned.
--
-- Why an episode rather than a landing square.
--
--   The previous pipeline's unit was the attractor: a 0.01-deg square that
--   accumulated displaced observations from >=25 distinct cells. That unit has
--   a built-in blind spot -- it can only see displacement that CONCENTRATES.
--   A spoofer that walks the decoy, alternates between decoys, or produces a
--   coherent offset field rather than a point never accumulates 25 cells in any
--   single square, and is structurally invisible no matter how strong it is.
--   The old census reported 214 sites as "directional displacement field",
--   which is that blind spot showing up as an unresolved residual category.
--
--   An episode carries WHEN, which a position histogram has already discarded.
--   Time is what makes spoofing separable: the signature of an area effect is
--   that many cells move TOGETHER (S4), and that test cannot even be stated in
--   a time-collapsed representation.
--
-- Coarse-to-fine.
--
--   Exact per-observation segmentation of 5.39B candidate observations requires
--   a window function partitioned by cell and ordered by time, i.e. an external
--   sort of the whole candidate corpus, on a 16-core box shared with Nominatim
--   and another user. That cost is not warranted globally, because the vast
--   majority of candidate cells are ordinary scatter that no later stage will
--   promote.
--
--   So: stage A is a pure aggregation over the full candidate population and
--   costs one streaming pass. Stage B (s2b_refine.sql) applies exact windowed
--   segmentation only to cells implicated in candidate events after S3, where
--   precise onset times actually change a conclusion.
--
--   The cost of the approximation is stated exactly: stage A merges repeat
--   visits by one cell to the SAME square into a single row, so a cell that is
--   displaced, recovers, and is displaced again to the same place reads as one
--   long visit. n_days_spanned vs obs makes those detectable, and stage B
--   resolves them. Stage A never merges visits to DIFFERENT squares, so the
--   fan-shaped partial displacement the mechanism predicts stays resolved.

-- ---------------------------------------------------------------------------
-- S2.1  Away observations, aggregated per (cell, destination square).
--
-- "Away" is 5 km from the S1 reference, matching the candidate prefilter. This
-- is deliberately a LOW bar: its only job is to exclude observations sitting on
-- the cell's own consensus position. Magnitude thresholds that carry
-- interpretive weight are applied downstream on measured quantities, not here,
-- so that the threshold can be varied in S8 without rebuilding this table.
CREATE TABLE IF NOT EXISTS spoof.away
ENGINE = MergeTree ORDER BY (plat, plon, mcc, mnc, lac, cid, cell_type)
AS
SELECT
    g.mcc AS mcc, g.mnc AS mnc, g.lac AS lac, g.cid AS cid,
    g.cell_type AS cell_type,
    toInt32(round(g.lat * 100)) AS plat,
    toInt32(round(g.lon * 100)) AS plon,
    count()             AS obs,
    min(g.timestamp)    AS t_first,
    max(g.timestamp)    AS t_last,
    any(r.rlat)         AS rlat,
    any(r.rlon)         AS rlon,
    -- Displacement of this destination square from the cell's own reference.
    any(greatCircleDistance(g.lon, g.lat, r.rlon / 100, r.rlat / 100)) / 1000 AS km
FROM cell.geos AS g
INNER JOIN spoof.cellref AS r
    ON g.mcc = r.mcc AND g.mnc = r.mnc AND g.lac = r.lac
   AND g.cid = r.cid AND g.cell_type = r.cell_type
WHERE greatCircleDistance(g.lon, g.lat, r.rlon / 100, r.rlat / 100) / 1000 > 5.0
GROUP BY mcc, mnc, lac, cid, cell_type, plat, plon;

-- ---------------------------------------------------------------------------
-- S2.2  Per-cell away summary.
--
-- Onset (t_first_away) is the quantity S4's synchrony test consumes. Note it is
-- the first time we OBSERVED the cell away, which is an upper bound on the true
-- onset, bounded below by the previous observation of that cell. S4 must
-- account for that interval, which is why spoof.cellmonth carries per-cell
-- cadence: a cell sampled twice a month has a two-week onset uncertainty and
-- cannot contribute the same evidence as one sampled hourly.
CREATE TABLE IF NOT EXISTS spoof.cell_away
ENGINE = MergeTree ORDER BY (mcc, mnc, lac, cid, cell_type)
AS
SELECT
    mcc, mnc, lac, cid, cell_type,
    sum(obs)        AS away_obs,
    count()         AS away_squares,
    min(t_first)    AS t_first_away,
    max(t_last)     AS t_last_away,
    max(km)         AS max_km,
    -- Observation-weighted median displacement: the typical distance this cell
    -- was displaced, not the distance of its single furthest outlier.
    quantileExactWeighted(0.5)(km, obs) AS med_km,
    argMax(plat, obs) AS top_plat,   -- the square this cell spent most away-time at
    argMax(plon, obs) AS top_plon,
    max(obs)          AS top_obs
FROM spoof.away
GROUP BY mcc, mnc, lac, cid, cell_type;
