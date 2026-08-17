-- S0b: DAILY per-region exposure. Added after the monthly table proved
-- insufficient, and the reason it was insufficient is worth recording.
--
-- WHAT WENT WRONG WITH MONTHLY EXPOSURE
--
-- S3's first output ranked (region, day) bins by an `excess` that normalised
-- each region against its OWN mean daily onset rate. That controls for regions
-- being differently busy. It does not control for the whole crawl being
-- differently busy on a given DAY, and the top of the ranking was consequently
-- dominated by dates that are impossible as physical events:
--
--     2024-04-03   207,522 onsets across 68,086 regions worldwide
--     2023-11-04   166,080 onsets across 78,147 regions  (day 1 of the dataset)
--     2026-05-01    67,163 onsets across 45,525 regions
--
-- A terrestrial GNSS transmitter cannot displace cells in 68,086 regions on
-- five continents on one day. These are properties of our crawler and of the
-- platform, not of the world: April 2024 is precisely when monthly crawl volume
-- stepped from 796M to 2.43B observations.
--
-- Monthly exposure cannot see this, because the confound lives entirely inside
-- a month. Hence a daily table.
--
-- THE DENOMINATOR
--
-- For each (source region, day): how many observations of CANDIDATE cells
-- homed in that region did we collect. This is the sampling attention paid to
-- the population that could have produced an onset there that day. An onset
-- count is only interpretable against it.
--
-- Region granularity matches spoof.onsets (0.1 deg, ~11 km), keyed on the
-- cell's REFERENCE position rather than the observation's position -- the
-- question is "how hard were we looking at the cells that live here", and a
-- displaced observation of a local cell still counts as looking at that cell.
--
-- Observation count only, deliberately no distinct-cell column: there are
-- ~4x10^8 (region, day) groups (418,713 regions x ~970 days) and a cardinality
-- state per group does not fit. uniqExact hit the 54 GiB server cap; uniq()
-- died at 16.76 GiB in the spill-merge. Observation count needs one UInt64 per
-- group and answers the question a denominator has to answer -- how hard were
-- we looking -- rather than how many distinct things we saw.
--
-- WHY THIS IS CHUNKED
--
-- Even with a bare count(), a single-shot GROUP BY over 4x10^8 groups OOMs at
-- 18.63 GiB during the spill-merge, because the merge must hold the distinct
-- keys of each spilled bucket. cell.geos is ordered by (mcc, mnc, lac, cid,
-- cell_type), so filtering on an MCC RANGE is a primary-key range scan rather
-- than a full pass. Each chunk therefore touches a fraction of the data AND
-- spans only the regions belonging to those MCCs, which is what actually bounds
-- the group count.
--
-- Chunks straddle regions (several MCCs can share an 11 km square, e.g. at
-- borders), so each chunk emits a PARTIAL aggregate for such regions.
-- SummingMergeTree adds them on merge, making the result exact regardless of
-- how chunks are cut. Reads must still use sum(obs) with GROUP BY, or FINAL,
-- since background merges are not guaranteed to have completed.
--
-- Chunk edges follow the candidate population measured per 50-MCC bucket: the
-- mass sits in 200-750, with 250-300 (1.97M cells) and 400-500 (2.8M) the
-- heaviest, so those are cut finer.

DROP TABLE IF EXISTS spoof.exposure_region_day;

CREATE TABLE spoof.exposure_region_day
(
    src_lat10 Int32,
    src_lon10 Int32,
    day       Date,
    obs       UInt64
)
ENGINE = SummingMergeTree(obs)
ORDER BY (src_lat10, src_lon10, day);
