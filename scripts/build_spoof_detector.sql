-- GNSS-spoofing detector over the per-cell position histogram (cell.cellpos).
--
-- Mechanism. Apple's Cellular Positioning System estimates a cell's coordinate
-- from crowdsourced handset GNSS fixes. A handset whose GNSS receiver has been
-- spoofed to a decoy coordinate D reports every cell it can hear as being at D.
-- Apple's estimate for those cells is dragged toward D, and our crawler records
-- the result. Spoofing is therefore visible as: real, locally-homed cells
-- acquiring a minority of observations at a distant, sharply-defined point.
--
-- Jamming is largely invisible by contrast: denial of service yields no fix, so
-- no (corrupted) report is generated at all.

-- 1. Home position: the grid square holding the plurality of a cell's observations.
CREATE TABLE IF NOT EXISTS cell.cellhome
ENGINE = MergeTree ORDER BY (mcc, mnc, lac, cid, cell_type)
AS
SELECT
    mcc, mnc, lac, cid, cell_type,
    argMax(plat, obs) AS hlat,
    argMax(plon, obs) AS hlon,
    max(obs)          AS home_obs,
    sum(obs)          AS total_obs,
    count()           AS n_pos
FROM cell.cellpos
GROUP BY mcc, mnc, lac, cid, cell_type
SETTINGS optimize_aggregation_in_order = 1;

-- 2. Displaced observations: a cell seen >10 km from its own home square.
--    10 km is well beyond ordinary crowdsourced GNSS scatter and beyond the
--    reach of a single terrestrial cell, so a displaced row cannot be explained
--    by a handset genuinely hearing that cell from where it claims to be.
CREATE TABLE IF NOT EXISTS cell.displaced
ENGINE = MergeTree ORDER BY (plat, plon)
AS
SELECT
    p.mcc AS mcc, p.mnc AS mnc, p.lac AS lac, p.cid AS cid, p.cell_type AS cell_type,
    p.plat AS plat, p.plon AS plon,
    p.obs AS obs, p.first_seen AS first_seen, p.last_seen AS last_seen,
    h.hlat AS hlat, h.hlon AS hlon,
    h.total_obs AS total_obs, h.home_obs AS home_obs,
    greatCircleDistance(p.plon / 100, p.plat / 100, h.hlon / 100, h.hlat / 100) / 1000 AS km
FROM cell.cellpos AS p
INNER JOIN cell.cellhome AS h
    ON p.mcc = h.mcc AND p.mnc = h.mnc AND p.lac = h.lac
   AND p.cid = h.cid AND p.cell_type = h.cell_type
WHERE greatCircleDistance(p.plon / 100, p.plat / 100, h.hlon / 100, h.hlat / 100) / 1000 > 10;

-- 3. Attractors: grid squares that collect displaced observations from many
--    distinct cells. Features are chosen to separate the four confounders that
--    also produce "many cells at one point":
--      spoofing decoy  - sources form ONE coherent region, attractor is a sharp
--                        point tens-to-hundreds of km away, time-bounded
--      equipment lab   - sources scattered worldwide (huge src_spread_km)
--      border spillover- attractor within ~30 km of the source centroid
--      indoor DAS      - the point is those cells' actual home (few displaced)
-- 3a. Per-attractor aggregate plus the centroid of its source homes.
CREATE TABLE IF NOT EXISTS cell.attr_base
ENGINE = MergeTree ORDER BY (plat, plon)
AS
SELECT
    plat, plon,
    uniqExact((mcc, mnc, lac, cid, cell_type)) AS cells,
    sum(obs)                                   AS obs,
    uniqExact(mcc)                             AS n_mcc,
    topK(4)(mcc)                               AS top_mcc,
    min(first_seen)                            AS t_start,
    max(last_seen)                             AS t_end,
    avg(hlat) / 100                            AS src_lat,
    avg(hlon) / 100                            AS src_lon,
    quantile(0.5)(km)                          AS med_km,
    quantile(0.9)(km)                          AS p90_km
FROM cell.displaced
GROUP BY plat, plon
HAVING cells >= 25;

-- 3b. Source spread: how tightly the contributing cells' homes cluster around
--     their own centroid. Small => one coherent spoofed region. Large => the
--     "sources" are unrelated places worldwide, i.e. equipment-test leakage.
CREATE TABLE IF NOT EXISTS cell.attractors
ENGINE = MergeTree ORDER BY (plat, plon)
AS
SELECT
    b.plat AS plat, b.plon AS plon,
    b.cells AS cells, b.obs AS obs, b.n_mcc AS n_mcc, b.top_mcc AS top_mcc,
    b.t_start AS t_start, b.t_end AS t_end,
    b.src_lat AS src_lat, b.src_lon AS src_lon,
    b.med_km AS med_km, b.p90_km AS p90_km,
    s.src_spread_km AS src_spread_km
FROM cell.attr_base AS b
INNER JOIN
(
    SELECT
        d.plat AS plat, d.plon AS plon,
        quantile(0.9)(greatCircleDistance(d.hlon / 100, d.hlat / 100,
                                          b2.src_lon, b2.src_lat) / 1000) AS src_spread_km
    FROM cell.displaced AS d
    INNER JOIN cell.attr_base AS b2 ON d.plat = b2.plat AND d.plon = b2.plon
    GROUP BY d.plat, d.plon
) AS s ON b.plat = s.plat AND b.plon = s.plon;
