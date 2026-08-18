#!/usr/bin/env python3
"""Read-only global screens for six additional spoofing hypotheses.

The queries deliberately use the existing summary tables instead of modifying
ClickHouse.  They are expensive and are therefore only executed with
``--refresh``.  Detailed follow-up tables in ``data/spoofing`` retain the
results used in the accompanying audit; this script regenerates the global
screens and cadence controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "spoofing"


DYNAMIC_MONTHLY_SQL = """
SELECT
    intDiv(r.rlat, 10) AS src_lat10,
    intDiv(r.rlon, 10) AS src_lon10,
    m.month,
    intDiv(m.mlat, 10) AS dest_lat10,
    intDiv(m.mlon, 10) AS dest_lon10,
    count() AS identities,
    uniqExact(m.mcc) AS n_mcc,
    uniqExact((m.mcc,m.mnc)) AS n_operators,
    sum(m.mode_obs) AS observations
FROM spoof.cellmonth AS m
INNER JOIN spoof.cellref AS r USING (mcc,mnc,lac,cid,cell_type)
WHERE r.n_months >= 3
  AND m.mode_obs >= 2
  AND greatCircleDistance(r.rlon/100, r.rlat/100, m.mlon/100, m.mlat/100) >= 25000
GROUP BY src_lat10,src_lon10,m.month,dest_lat10,dest_lon10
HAVING identities >= 8
ORDER BY src_lat10,src_lon10,m.month,identities DESC
"""


JUMP_RETURN_SQL = """
WITH ordered AS (
    SELECT m.*, r.rlat, r.rlon,
           lagInFrame(month) OVER w AS prev_month,
           leadInFrame(month) OVER w AS next_month,
           lagInFrame(mlat) OVER w AS prev_lat,
           lagInFrame(mlon) OVER w AS prev_lon,
           leadInFrame(mlat) OVER w AS next_lat,
           leadInFrame(mlon) OVER w AS next_lon
    FROM spoof.cellmonth AS m
    INNER JOIN spoof.cellref AS r USING (mcc,mnc,lac,cid,cell_type)
    WHERE r.n_months >= 3 AND m.mode_obs >= 2
    WINDOW w AS (
      PARTITION BY mcc,mnc,lac,cid,cell_type
      ORDER BY month ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    )
), jumps AS (
    SELECT * FROM ordered
    WHERE dateDiff('month',prev_month,month)=1
      AND dateDiff('month',month,next_month)=1
      AND greatCircleDistance(rlon/100,rlat/100,mlon/100,mlat/100)>=25000
      AND greatCircleDistance(rlon/100,rlat/100,prev_lon/100,prev_lat/100)<=5000
      AND greatCircleDistance(rlon/100,rlat/100,next_lon/100,next_lat/100)<=5000
)
SELECT intDiv(rlat,10) src_lat10,intDiv(rlon,10) src_lon10,
       month AS away_month,intDiv(mlat,10) dest_lat10,intDiv(mlon,10) dest_lon10,
       count() identities,uniqExact(cid) distinct_cids,
       uniqExact((mcc,mnc)) operators,sum(mode_obs) observations
FROM jumps
GROUP BY src_lat10,src_lon10,away_month,dest_lat10,dest_lon10
HAVING identities>=5
ORDER BY identities DESC
"""


INTERVAL_OVERLAP_SQL = """
SELECT d.plat,d.plon,d.plat/100 AS destination_lat,d.plon/100 AS destination_lon,
       count() AS overlapping_identities,uniqExact(d.cid) AS distinct_cids,
       uniqExact(d.mcc) AS n_mcc,uniqExact((d.mcc,d.mnc)) AS n_operators,
       sum(d.obs) AS away_observations,sum(h.obs) AS home_observations,
       median(d.km) AS median_km,
       median(dateDiff('day',greatest(d.first_seen,h.first_seen),
                            least(d.last_seen,h.last_seen))) AS median_interval_overlap_days,
       min(greatest(d.first_seen,h.first_seen)) AS overlap_start,
       max(least(d.last_seen,h.last_seen)) AS overlap_end
FROM cell.displaced AS d
INNER JOIN cell.cellpos AS h
  ON d.mcc=h.mcc AND d.mnc=h.mnc AND d.lac=h.lac AND d.cid=h.cid
 AND d.cell_type=h.cell_type AND d.hlat=h.plat AND d.hlon=h.plon
WHERE d.km>=100 AND d.obs>=2 AND h.obs>=2
  AND greatest(d.first_seen,h.first_seen)<=least(d.last_seen,h.last_seen)
GROUP BY d.plat,d.plon
HAVING overlapping_identities>=5
ORDER BY overlapping_identities DESC
"""


IDENTIFIER_SUFFIX_SQL = """
WITH local AS (
  SELECT hlat AS plat,hlon AS plon,mcc,mnc,cell_type,
         count() AS local_identities,
         groupUniqArray(bitAnd(toUInt64(cid),255)) AS suffix8,
         groupUniqArray(bitAnd(toUInt64(cid),4095)) AS suffix12,
         groupUniqArray(bitAnd(toUInt64(cid),65535)) AS suffix16
  FROM cell.cellhome
  GROUP BY plat,plon,mcc,mnc,cell_type
)
SELECT d.plat,d.plon,d.mcc,d.mnc,toString(d.cell_type) AS cell_type,
       count() AS displaced_identities,uniqExact(d.cid) AS distinct_cids,
       any(l.local_identities) AS local_identities,
       length(any(l.suffix8)) AS local_suffix8_count,
       length(any(l.suffix12)) AS local_suffix12_count,
       length(any(l.suffix16)) AS local_suffix16_count,
       countIf(has(l.suffix8,bitAnd(toUInt64(d.cid),255))) AS matches8,
       countIf(has(l.suffix12,bitAnd(toUInt64(d.cid),4095))) AS matches12,
       countIf(has(l.suffix16,bitAnd(toUInt64(d.cid),65535))) AS matches16,
       sum(d.obs) AS observations,median(d.km) AS median_km,
       matches8/displaced_identities AS match8_fraction,
       matches12/displaced_identities AS match12_fraction,
       matches16/displaced_identities AS match16_fraction,
       match16_fraction/(local_suffix16_count/65536) AS suffix16_enrichment
FROM cell.displaced AS d
INNER JOIN local AS l USING (plat,plon,mcc,mnc,cell_type)
WHERE d.km>=25 AND d.obs>=2
GROUP BY d.plat,d.plon,d.mcc,d.mnc,d.cell_type
HAVING displaced_identities>=5 AND matches16>=2
ORDER BY matches16 DESC,displaced_identities DESC
"""


ENDPOINT_MINUTE_SQL = """
SELECT toStartOfMinute(first_seen) AS first_seen_minute,plat,plon,
       count() AS identities,uniqExact(cid) AS distinct_cids,
       uniqExact((mcc,mnc)) AS operators,uniqExact(mcc) AS mccs,
       sum(obs) AS displaced_observations
FROM cell.displaced
WHERE km>=25 AND obs>=2 AND plat BETWEEN -9000 AND 9000
  AND plon BETWEEN -18000 AND 18000
  AND NOT (abs(plat)<=100 AND abs(plon)<=100)
GROUP BY first_seen_minute,plat,plon
HAVING identities>=5
ORDER BY identities DESC
LIMIT 10000
"""


GLOBAL_BATCH_SQL = """
SELECT first_seen,count() AS identities,uniqExact(cid) AS distinct_cids,
       uniqExact((mcc,mnc)) AS operators,uniqExact(mcc) AS mccs,
       sum(obs) AS observations
FROM cell.summary
GROUP BY first_seen
HAVING identities>=50
ORDER BY identities DESC
LIMIT 10000
"""


CADENCE_SQL = {
    "timestamp_cadence_all.csv": """
        SELECT toSecond(first_seen) second,toMinute(first_seen) minute,
               count() identities,uniqExact(cid) distinct_cids,sum(obs) observations
        FROM cell.summary GROUP BY second,minute ORDER BY minute,second
    """,
    "timestamp_cadence_displaced.csv": """
        SELECT toSecond(first_seen) second,toMinute(first_seen) minute,
               count() identities,uniqExact(cid) distinct_cids,sum(obs) observations
        FROM cell.displaced WHERE km>=25 AND obs>=2
        GROUP BY second,minute ORDER BY minute,second
    """,
}


QUERIES = {
    "dynamic_monthly_screen.csv": DYNAMIC_MONTHLY_SQL,
    "jump_return_group_audit.csv": JUMP_RETURN_SQL,
    "concurrent_interval_endpoints.csv": INTERVAL_OVERLAP_SQL,
    "identifier_suffix_audit_raw.csv": IDENTIFIER_SUFFIX_SQL,
    "timestamp_endpoint_minute_batches.csv": ENDPOINT_MINUTE_SQL,
    "timestamp_global_first_seen_batches.csv": GLOBAL_BATCH_SQL,
    **CADENCE_SQL,
}


def provenance_audit() -> pd.DataFrame:
    fields = ch_df("""
      SELECT database,table,name,type FROM system.columns
      WHERE database IN ('cell','spoof') AND (
        positionCaseInsensitive(name,'source')>0 OR
        positionCaseInsensitive(name,'crawl')>0 OR
        positionCaseInsensitive(name,'provider')>0 OR
        positionCaseInsensitive(name,'provenance')>0 OR
        positionCaseInsensitive(name,'collector')>0)
      ORDER BY database,table,position
    """)
    if fields.empty:
        return pd.DataFrame([{
            "table": "cell.geos",
            "available_fields": "mcc;mnc;lac;cid;cell_type;lat;lon;timestamp",
            "provenance_fields_found": 0,
            "interpretation": "collector/provider attribution is impossible from stored schema",
        }])
    fields["provenance_fields_found"] = len(fields)
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true",
        help="run the long read-only ClickHouse scans",
    )
    args = parser.parse_args()
    if not args.refresh:
        parser.error("pass --refresh to run the long read-only scans")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, sql in QUERIES.items():
        print(f"querying {filename}", flush=True)
        ch_df(sql).to_csv(OUTPUT / filename, index=False)
    provenance_audit().to_csv(OUTPUT / "provenance_schema_audit.csv", index=False)


if __name__ == "__main__":
    main()
