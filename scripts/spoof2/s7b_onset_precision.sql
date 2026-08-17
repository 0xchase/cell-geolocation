-- S7b / confounder C13: ONSET PRECISION.
--
-- THE HOLE THIS FILLS
--
-- spoof.onsets records t_first_away: the first time we OBSERVED a cell away from
-- its reference. That is an upper bound on when it actually moved, and the bound
-- is only as tight as the interval since we last looked at that cell.
--
-- The lead-time control (C2) requires a cell to have been KNOWN for >=30 days
-- before its onset, which removes cells discovered by a crawl reaching a region
-- for the first time. It does not remove cells that were known for years but
-- unobserved for a stretch, and that gap is real in this dataset:
--
--     2026-04   596,577,221 observations   <- crawl gap
--     2026-05 2,306,338,558 observations   <- resumption
--
-- A cell displaced during April is first SEEN displaced on 1 May. Thousands of
-- such cells across unrelated regions produce a synchronised onset on the
-- resumption day that survives both the exposure denominator and the lead-time
-- filter. Measured: 2026-05-01 carries 67,163 onsets across 45,525 regions at a
-- median excess of 62. 2023-11-04 reaches a median excess of 179.
--
-- Neither is a physical event. Both are the crawl restarting.
--
-- THE CONTROL
--
-- precision_days = t_first_away - (last observation of that cell strictly before
-- t_first_away). It is the width of the window within which the displacement
-- actually began.
--
-- A cell whose previous sighting was 40 days earlier cannot testify that
-- anything happened on a particular DAY -- its onset is compatible with any of 40
-- days. Day-resolution synchrony must therefore be computed only over cells
-- whose precision is comparable to the bucket width. Given a median inter-
-- observation gap of 1.03 days across the corpus, the natural cut is a small
-- number of days; it is left as a parameter rather than baked in, because S8
-- must be able to vary it.
--
-- This subsumes C2 rather than supplementing it: a cell seen for the first time
-- has no previous observation at all and is excluded by construction (no row).
--
-- COST
--
-- One pass over cell.geos restricted to the 5.62M cells that have an onset,
-- taking the max timestamp below each cell's own onset. The per-cell threshold
-- makes this a join rather than a constant filter, so it cannot be pushed into
-- the primary key; it is a full scan and takes roughly as long as the S2 build.

DROP TABLE IF EXISTS spoof.onset_precision;

CREATE TABLE spoof.onset_precision
ENGINE = MergeTree ORDER BY (mcc, mnc, lac, cid, cell_type)
AS
SELECT
    a.mcc AS mcc, a.mnc AS mnc, a.lac AS lac, a.cid AS cid,
    a.cell_type AS cell_type,
    a.t_first_away AS t_first_away,
    max(g.timestamp) AS t_prev,
    dateDiff('day', max(g.timestamp), a.t_first_away) AS precision_days
FROM cell.geos AS g
INNER JOIN spoof.cell_away AS a
    ON g.mcc = a.mcc AND g.mnc = a.mnc AND g.lac = a.lac
   AND g.cid = a.cid AND g.cell_type = a.cell_type
WHERE g.timestamp < a.t_first_away
GROUP BY mcc, mnc, lac, cid, cell_type, t_first_away;
