# GNSS-spoofing detection: methodology v2

Built from scratch against the corrected `cell.geos` (63.34B rows, Nov 2023 –
Jul 2026). Does not read or modify any table from the earlier pipeline
(`cell.cellhome`, `cell.displaced`, `cell.attractors`, `cell.attr_*`), which are
left intact.

Everything lives in the ClickHouse database `spoof` on `nominatim.cybre.io`.

## Run order

| # | file | builds | cost |
|---|---|---|---|
| S0 | `s0_exposure.sql` | `candidates`, `exposure_sq` | ~9 min |
| S0b | `s0b_exposure_daily.sql` + `s0b_chunks.sh` | `exposure_region_day` | ~12 min, 27 chunks |
| S1 | `s1_reference.sql` | `cellmonth`, `cellref`, `cellref_stability` | ~21 min |
| S2 | `s2_episodes.sql` | `away`, `cell_away` | ~11 min |
| S3 | `s3_events.sql` | `onsets`, `sync` | fast |
| S3b | `s3b_sync_corrected.sql` | `exp_onsetday`, `exp_region`, `ons_day`, `ons_region`, `sync2` | fast |
| S7b | `s7b_onset_precision.sql` | `onset_precision` | ~9 min |
| S3c | `s3c_sync_precision.sql` | `onsets_f`, `ons_day_f`, `ons_region_f`, `sync3` | fast |
| S4 | `s4_null.py` | permutation null, detection threshold | ~1 min |
| S5 | `s5_mechanism.py` | per-event mixture geometry | seconds |
| S6 | `s6_area_effect.py` | composition G-test | seconds |
| S8 | `s8_injection.py` | sensitivity curve | ~2 min |
| S10 | `s10_classify.py` | tiered classification | ~3 min |

`s7_confounders.md` and `s11_external_validation.md` are documents, not code.

## Checksums

Any rebuild must reproduce these exactly. The first three are independent
derivations of the same quantity from `cell.geos` and must agree.

```
cellref.obs              = 5,388,929,553
cellmonth.obs            = 5,388,929,553
exposure_region_day.obs  = 5,388,929,553
exposure_sq.obs          = 63,342,786,323   (all cells, not just candidates)
```

Row counts:

```
candidates            9,455,648        onsets              5,377,936
exposure_sq         521,151,986        onsets_f            2,880,515
cellmonth           173,654,190        onset_precision     3,406,752
cellref               9,455,648        sync3               2,335,808
cellref_stability     9,455,648        exposure_region_day 354,895,266
away                 34,665,797        exp_region            464,924
cell_away             5,621,105
```

Note `cell.summary_full.obs` sums to 5,389,363,622 over the candidate set —
434,069 more (0.008%) than a live `cell.geos` scan. Use geos-derived values.

## Results as of this run

- Detection threshold (S4, family-wise 99%): **n ≥ 9 cells in one (region, day)**.
  Null global max over 500 rounds: median 6, 99% 9, absolute max 11. Observed
  max 475.
- **61 significant bins.** Classified (S10): **20 T1** (mixture confirmed),
  **5 T2** (ambiguous), **34 T3** (coherent but not a mixture), **2 rejected**.
- Sensitivity (S8, 165 km decoy, quiet region): ~20 cells at w≥0.5, ~40 at
  w≥0.25, ~80 at w≥0.10. **Never detected at w=0.05 at any size** — 5% of
  reports spoofed moves the estimate ~8 km, below the 25 km floor.
- Contamination (S12): 5,621,105 cells (4.46%) displaced >5 km; 379,977 (0.30%)
  >25 km; 126,546 (0.10%) >100 km. 441.67M of 63.35B observations (0.70%).

## Gotchas that cost time here

1. **`ch_remote.py` sets `optimize_aggregation_in_order=1` by default.** That
   serialises these aggregations onto ~2 cores. Every script here overrides it.
2. **`uniqExact`/`uniq` per group dies above ~10^8 groups.** Killed three builds
   (54 GiB cap, then 16.76 GiB and 18.63 GiB in the spill-merge). Use `count()`,
   or chunk.
3. **Chunk on `mcc` ranges, not time or space.** `cell.geos` is ordered by
   `(mcc, mnc, lac, cid, cell_type)`, so an MCC range is a primary-key range
   scan. Chunks straddle regions at borders; `SummingMergeTree` reconciles them.
4. **ClickHouse hash joins materialise the RIGHT side.** A 354.9M-row table on
   the right dies in `FillingRightJoinSide`. Reduce the large input to a real
   table first rather than trying to out-guess the planner with CTE ordering.
5. **Double quotes are identifiers, not strings.** `query_id="x"` in a wait loop
   silently never matches.
6. **`KILL QUERY WHERE query LIKE ...` matches other users' queries.** Kill by
   `query_id`. This session killed a bystander's query that way.
7. **Aggregate aliases shadow source columns** — `quantileExact(0.5)(med_km) AS
   med_km` makes `WHERE med_km >= 25` resolve to the aggregate.

## What is not done

- **S8 end-to-end anchors.** The sensitivity curve exercises detection only. It
  does not exercise S1 reference estimation or S2 away-detection, so it cannot
  see the reference-capture failure S1 exists to prevent. Required before
  publishing any sensitivity number.
- **S11 external validation.** Predictions are frozen in
  `s11_external_validation.md`; the checking has not been done.
- **S6 is unresolved.** It rejects the area-effect hypothesis for Sheremetyevo
  even against a matched temporal control (76.2% LTE vs 32.9%), which conflicts
  with S5 calling the same event a clean mixture. The likely confound is that the
  control is dominated by chronic small-displacement GSM; it should be matched on
  displacement magnitude as well as region and time.
- **C8 and C10 remain uncontrollable** (see `s7_confounders.md`): operators
  legitimately present in contested territory, and a platform-side change
  confined to one region.
