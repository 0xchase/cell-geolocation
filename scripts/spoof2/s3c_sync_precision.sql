-- S3c: the synchrony statistic over PRECISION-FILTERED onsets.
--
-- Supersedes spoof.sync2. Same exposure model; the difference is which onsets
-- are allowed to testify.
--
-- An onset is admitted only if we saw the cell within MAX_PRECISION_DAYS before
-- it (spoof.onset_precision, confounder C13). A cell last observed 40 days
-- earlier is compatible with the displacement having begun on any of 40 days and
-- cannot support a claim about one particular day; admitting it lets crawl gaps
-- masquerade as coordinated switch-ons.
--
-- Measured effect of the filter on the input population:
--   5.62M cells have an onset
--   3.41M have any prior observation at all   (2.21M were ALREADY displaced the
--                                              first time we ever saw them --
--                                              first contact, excluded here by
--                                              construction, which subsumes C2)
--   2.98M have precision <= 2 days            (87.6% of those with a prior obs)
--
-- The 2-day cut is set against the corpus median inter-observation gap of 1.03
-- days: it admits cells sampled at roughly the normal cadence and rejects those
-- behind a gap.

DROP TABLE IF EXISTS spoof.onsets_f;
CREATE TABLE spoof.onsets_f
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10, onset_day)
AS
SELECT o.*
FROM spoof.onsets AS o
INNER JOIN spoof.onset_precision AS p
    ON o.mcc = p.mcc AND o.mnc = p.mnc AND o.lac = p.lac
   AND o.cid = p.cid AND o.cell_type = p.cell_type
WHERE p.precision_days <= 2;

DROP TABLE IF EXISTS spoof.ons_day_f;
CREATE TABLE spoof.ons_day_f
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
FROM spoof.onsets_f AS o
INNER JOIN spoof.cellref AS r
    ON o.mcc = r.mcc AND o.mnc = r.mnc AND o.lac = r.lac
   AND o.cid = r.cid AND o.cell_type = r.cell_type
GROUP BY src_lat10, src_lon10, onset_day;

DROP TABLE IF EXISTS spoof.ons_region_f;
CREATE TABLE spoof.ons_region_f
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10)
AS
SELECT src_lat10, src_lon10, sum(n_onsets) AS region_onsets
FROM spoof.ons_day_f GROUP BY src_lat10, src_lon10;

DROP TABLE IF EXISTS spoof.sync3;
CREATE TABLE spoof.sync3
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
FROM spoof.ons_day_f AS d
INNER JOIN spoof.exp_onsetday AS e
    ON d.src_lat10 = e.src_lat10 AND d.src_lon10 = e.src_lon10
   AND d.onset_day = e.day
INNER JOIN spoof.exp_region AS rr
    ON d.src_lat10 = rr.src_lat10 AND d.src_lon10 = rr.src_lon10
INNER JOIN spoof.ons_region_f AS ro
    ON d.src_lat10 = ro.src_lat10 AND d.src_lon10 = ro.src_lon10;
