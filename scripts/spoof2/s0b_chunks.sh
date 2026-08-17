#!/usr/bin/env bash
# S0b chunked loader. See s0b_exposure_daily.sql for why this is chunked.
#
# Each chunk is a primary-key range scan on mcc, aggregated to (region, day) and
# inserted into a SummingMergeTree, which reconciles regions that straddle chunk
# boundaries. Chunks are sequential on purpose: the box is shared with Nominatim
# and another user, and running them concurrently would recreate the memory
# pressure the chunking exists to avoid.
set -uo pipefail

CH="clickhouse-client --password password --max_threads 10 \
    --max_bytes_before_external_group_by 4000000000 \
    --max_memory_usage 14000000000"

# Edges chosen from the per-50-MCC candidate census; heavy buckets cut finer.
EDGES=(0 210 220 230 240 250 260 270 290 300 310 330 350 400 415 425 440 450 460 470 500 520 600 640 660 700 720 1000)

for ((i = 0; i < ${#EDGES[@]} - 1; i++)); do
  LO=${EDGES[$i]}
  HI=${EDGES[$((i + 1))]}
  printf '[%s] chunk mcc [%s,%s) ... ' "$(date +%H:%M:%S)" "$LO" "$HI"
  $CH --query_id "s0b_chunk_${LO}_${HI}" --query "
    INSERT INTO spoof.exposure_region_day
    SELECT intDiv(r.rlat,10) AS src_lat10,
           intDiv(r.rlon,10) AS src_lon10,
           toDate(g.timestamp) AS day,
           count() AS obs
    FROM cell.geos AS g
    INNER JOIN spoof.cellref AS r
      ON g.mcc=r.mcc AND g.mnc=r.mnc AND g.lac=r.lac
     AND g.cid=r.cid AND g.cell_type=r.cell_type
    WHERE g.mcc >= ${LO} AND g.mcc < ${HI}
    GROUP BY src_lat10, src_lon10, day" 2>&1 | tail -2
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "CHUNK FAILED mcc [${LO},${HI}) -- stopping"
    exit 1
  fi
  echo "ok"
done

echo "ALL CHUNKS COMPLETE"
clickhouse-client --password password --query "
  SELECT formatReadableQuantity(count()) AS rows_before_merge,
         sum(obs) AS total_obs
  FROM spoof.exposure_region_day"
