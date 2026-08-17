CREATE TABLE spoof.sync
ENGINE = MergeTree ORDER BY (src_lat10, src_lon10, onset_day)
AS
SELECT
    a.src_lat10 AS src_lat10,
    a.src_lon10 AS src_lon10,
    a.onset_day AS onset_day,
    a.n_onsets AS n_onsets,
    a.n_mcc AS n_mcc,
    a.n_operators AS n_operators,
    a.n_tech AS n_tech,
    a.med_km AS med_km,
    b.dest_concentration AS dest_concentration,
    t.region_onsets AS region_onsets,
    t.span_days AS span_days,
    a.n_onsets / (t.region_onsets / greatest(t.span_days, 1)) AS excess
FROM
(
    SELECT src_lat10, src_lon10, onset_day,
           count()                        AS n_onsets,
           uniqExact(mcc)                 AS n_mcc,
           uniqExact((mcc, mnc))          AS n_operators,
           uniqExact(cell_type)           AS n_tech,
           quantileExact(0.5)(med_km)     AS med_km
    FROM spoof.onsets
    GROUP BY src_lat10, src_lon10, onset_day
) AS a
INNER JOIN
(
    SELECT src_lat10, src_lon10, onset_day, max(c) / sum(c) AS dest_concentration
    FROM
    (
        SELECT src_lat10, src_lon10, onset_day, top_plat, top_plon, count() AS c
        FROM spoof.onsets
        GROUP BY src_lat10, src_lon10, onset_day, top_plat, top_plon
    )
    GROUP BY src_lat10, src_lon10, onset_day
) AS b
    ON a.src_lat10 = b.src_lat10 AND a.src_lon10 = b.src_lon10
   AND a.onset_day = b.onset_day
INNER JOIN
(
    SELECT src_lat10, src_lon10,
           count() AS region_onsets,
           dateDiff('day', min(onset_day), max(onset_day)) + 1 AS span_days
    FROM spoof.onsets
    GROUP BY src_lat10, src_lon10
) AS t
    ON a.src_lat10 = t.src_lat10 AND a.src_lon10 = t.src_lon10;
