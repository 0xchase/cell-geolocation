-- Per-cell summary rebuilt from the corrected (append-only MergeTree) cell.geos.
--
-- The old pipeline's summary carried only first_seen/last_seen/obs/latest-position,
-- because the deduplicated source had no real observation history to summarise.
-- The corrected source averages ~503 observations per cell, so this version also
-- carries the first position and the observation bounding box, which is what the
-- movement/clone analyses (obs15, obs17, obs21, all_moving_tower_thresholds) need.
-- Materialising it once avoids re-scanning 63B rows for every figure.
--
-- GROUP BY is exactly the table's ORDER BY key, so optimize_aggregation_in_order
-- lets ClickHouse stream the aggregation instead of building a 126M-group hash table.

CREATE TABLE IF NOT EXISTS cell.summary
ENGINE = MergeTree
ORDER BY (mcc, mnc, lac, cid, cell_type)
AS
SELECT
    mcc,
    mnc,
    lac,
    cid,
    cell_type,
    min(timestamp)                          AS first_seen,
    max(timestamp)                          AS last_seen,
    count()                                 AS obs,
    argMax(lat, timestamp)                  AS glat,      -- latest position (old `glat` semantics)
    argMax(lon, timestamp)                  AS glon,
    argMin(lat, timestamp)                  AS first_lat, -- earliest position
    argMin(lon, timestamp)                  AS first_lon,
    min(lat)                                AS lat_min,
    max(lat)                                AS lat_max,
    min(lon)                                AS lon_min,
    max(lon)                                AS lon_max,
    uniqExact((round(lat, 2), round(lon, 2))) AS n_pos    -- distinct ~1km positions
FROM cell.geos
GROUP BY mcc, mnc, lac, cid, cell_type
SETTINGS optimize_aggregation_in_order = 1;
