#!/usr/bin/env python3
"""Audit three large unresolved moving-cell destination clusters.

The local moving-identity census supplies the candidate keys and plurality-home
coordinates.  Raw observations are then queried through ``ch_remote``, whose
ClickHouse connection is forced to read-only mode.  The resulting compact CSVs
support both the paper figure and the mechanism assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
MOVING = ROOT / "data" / "moving-mccs" / "identities.csv.zst"
OUTPUT = ROOT / "data" / "spoofing"
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]


@dataclass(frozen=True)
class Attractor:
    key: str
    name: str
    lat: float
    lon: float


ATTRACTORS = (
    Attractor("chengde", "Kuancheng / Chengde", 40.62, 118.48),
    Attractor("valdai", "Zaozerye / Demyansk", 57.67, 32.53),
    Attractor("smolensk_npp", "Smolensk Nuclear Power Plant", 54.17, 33.23),
)


def candidates(spec: Attractor) -> pd.DataFrame:
    source = f"read_csv_auto('{MOVING}', header=true)"
    return duckdb.connect().execute(f"""
        SELECT mcc,mnc,lac,cid,cell_type,max_span_km,
               home_lat,home_lon,total_observations
        FROM {source}
        WHERE max_span_km >= 100
          AND ((endpoint_a_lat={spec.lat} AND endpoint_a_lon={spec.lon})
            OR (endpoint_b_lat={spec.lat} AND endpoint_b_lon={spec.lon}))
    """).fetchdf()


def reference_tuples(frame: pd.DataFrame) -> str:
    return ",".join(
        "tuple(toUInt16({}),toUInt16({}),toUInt32({}),toInt64({}),'{}',"
        "toFloat64({:.6f}),toFloat64({:.6f}))".format(
            int(row.mcc), int(row.mnc), int(row.lac), int(row.cid),
            row.cell_type, float(row.home_lat), float(row.home_lon),
        )
        for row in frame.itertuples(index=False)
    )


def identity_tuples(frame: pd.DataFrame) -> str:
    return ",".join(
        f"({int(row.mcc)},{int(row.mnc)},{int(row.lac)},"
        f"{int(row.cid)},'{row.cell_type}')"
        for row in frame.itertuples(index=False)
    )


def refs_sql(frame: pd.DataFrame) -> str:
    return f"""
      WITH refs AS (
        SELECT tupleElement(x,1) AS mcc,tupleElement(x,2) AS mnc,
               tupleElement(x,3) AS lac,tupleElement(x,4) AS cid,
               tupleElement(x,5) AS cell_type,tupleElement(x,6) AS home_lat,
               tupleElement(x,7) AS home_lon
        FROM (SELECT arrayJoin([{reference_tuples(frame)}]) AS x)
      )
    """


def fetch_identity_audit(spec: Attractor, frame: pd.DataFrame) -> pd.DataFrame:
    keys = identity_tuples(frame)
    result = ch_df(refs_sql(frame) + f"""
      SELECT g.mcc,g.mnc,g.lac,g.cid,toString(g.cell_type) AS cell_type,
             countIf(greatCircleDistance(g.lon,g.lat,r.home_lon,r.home_lat)<=3000)
                 AS home_observations,
             countIf(greatCircleDistance(g.lon,g.lat,{spec.lon},{spec.lat})<=3000)
                 AS destination_observations,
             minIf(g.timestamp,greatCircleDistance(
                 g.lon,g.lat,r.home_lon,r.home_lat)<=3000) AS home_first,
             maxIf(g.timestamp,greatCircleDistance(
                 g.lon,g.lat,r.home_lon,r.home_lat)<=3000) AS home_last,
             minIf(g.timestamp,greatCircleDistance(
                 g.lon,g.lat,{spec.lon},{spec.lat})<=3000) AS destination_first,
             maxIf(g.timestamp,greatCircleDistance(
                 g.lon,g.lat,{spec.lon},{spec.lat})<=3000) AS destination_last
      FROM cell.geos AS g
      INNER JOIN refs AS r ON g.mcc=r.mcc AND g.mnc=r.mnc AND g.lac=r.lac
        AND g.cid=r.cid AND toString(g.cell_type)=r.cell_type
      PREWHERE (g.mcc,g.mnc,g.lac,g.cid,g.cell_type) IN ({keys})
      GROUP BY g.mcc,g.mnc,g.lac,g.cid,cell_type
    """, settings={"max_threads": 6})
    result.insert(0, "attractor", spec.key)
    return result.merge(frame[KEY + ["home_lat", "home_lon", "max_span_km"]], on=KEY)


def fetch_batches(spec: Attractor, frame: pd.DataFrame) -> pd.DataFrame:
    keys = identity_tuples(frame)
    raw = ch_df(f"""
      SELECT timestamp,mcc,mnc,lac,cid,toString(cell_type) AS cell_type
      FROM cell.geos
      PREWHERE (mcc,mnc,lac,cid,cell_type) IN ({keys})
      WHERE greatCircleDistance(lon,lat,{spec.lon},{spec.lat})<=3000
      ORDER BY timestamp,mcc,mnc,lac,cid,cell_type
    """, settings={"max_threads": 6})
    grouped = raw.groupby("timestamp", as_index=False).agg(
        identities=("cid", "size"),
        distinct_cids=("cid", "nunique"),
        plmns=("mnc", "nunique"),
    )
    grouped.insert(0, "attractor", spec.key)
    return grouped


def fetch_same_day(spec: Attractor, frame: pd.DataFrame) -> pd.DataFrame:
    keys = identity_tuples(frame)
    return ch_df(refs_sql(frame) + f"""
      SELECT g.mcc,g.mnc,g.lac,g.cid,toString(g.cell_type) AS cell_type,
             toDate(g.timestamp) AS day,
             countIf(greatCircleDistance(g.lon,g.lat,r.home_lon,r.home_lat)<=3000)
                 AS home_observations,
             countIf(greatCircleDistance(g.lon,g.lat,{spec.lon},{spec.lat})<=3000)
                 AS destination_observations
      FROM cell.geos AS g
      INNER JOIN refs AS r ON g.mcc=r.mcc AND g.mnc=r.mnc AND g.lac=r.lac
        AND g.cid=r.cid AND toString(g.cell_type)=r.cell_type
      PREWHERE (g.mcc,g.mnc,g.lac,g.cid,g.cell_type) IN ({keys})
      WHERE greatCircleDistance(g.lon,g.lat,r.home_lon,r.home_lat)<=3000
         OR greatCircleDistance(g.lon,g.lat,{spec.lon},{spec.lat})<=3000
      GROUP BY g.mcc,g.mnc,g.lac,g.cid,cell_type,day
    """, settings={"max_threads": 6})


def chengde_cid_population() -> tuple[pd.DataFrame, pd.DataFrame]:
    population = ch_df("""
      SELECT mnc,countDistinct(lac) AS lacs,
             countDistinct((plat,plon)) AS positions,sum(obs) AS observations,
             min(first_seen) AS first_seen,max(last_seen) AS last_seen
      FROM cell.cellpos
      WHERE mcc=460 AND mnc IN (0,1) AND cid=22812 AND cell_type='gsm'
      GROUP BY mnc ORDER BY mnc
    """)
    positions = ch_df("""
      SELECT plat/100 AS lat,plon/100 AS lon,
             countDistinct((mnc,lac)) AS identities,sum(obs) AS observations,
             min(first_seen) AS first_seen,max(last_seen) AS last_seen
      FROM cell.cellpos
      WHERE mcc=460 AND mnc IN (0,1) AND cid=22812 AND cell_type='gsm'
      GROUP BY plat,plon ORDER BY identities DESC LIMIT 50
    """)
    return population, positions


def summarize(
    spec: Attractor,
    candidates_frame: pd.DataFrame,
    identity_frame: pd.DataFrame,
    batches: pd.DataFrame,
    day_frame: pd.DataFrame,
) -> dict[str, object]:
    dates = ["home_first", "home_last", "destination_first", "destination_last"]
    for column in dates:
        identity_frame[column] = pd.to_datetime(identity_frame[column], errors="coerce")
    valid = identity_frame[
        identity_frame.home_observations.gt(0)
        & identity_frame.destination_observations.gt(0)
    ]
    dual_days = day_frame[
        day_frame.home_observations.gt(0)
        & day_frame.destination_observations.gt(0)
    ]
    return {
        "attractor": spec.key,
        "destination_name": spec.name,
        "destination_lat": spec.lat,
        "destination_lon": spec.lon,
        "identities": len(candidates_frame),
        "distinct_cids": candidates_frame.cid.nunique(),
        "distinct_lacs": candidates_frame.lac.nunique(),
        "plmns": candidates_frame[["mcc", "mnc"]].drop_duplicates().shape[0],
        "home_observations": int(valid.home_observations.sum()),
        "destination_observations": int(valid.destination_observations.sum()),
        "home_after_first_destination": int(
            valid.home_last.ge(valid.destination_first).sum()
        ),
        "home_after_last_destination": int(
            valid.home_last.gt(valid.destination_last).sum()
        ),
        "same_day_identities": dual_days[KEY].drop_duplicates().shape[0],
        "destination_timestamps": len(batches),
        "destination_observations_in_batches_ge2": int(
            batches.loc[batches.identities.ge(2), "identities"].sum()
        ),
        "destination_observations_in_batches_ge5": int(
            batches.loc[batches.identities.ge(5), "identities"].sum()
        ),
        "largest_exact_batch": int(batches.identities.max()),
        "first_destination": valid.destination_first.min(),
        "last_destination": valid.destination_last.max(),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    identity_outputs = []
    batch_outputs = []
    summaries = []
    for spec in ATTRACTORS:
        frame = candidates(spec)
        identity_frame = fetch_identity_audit(spec, frame)
        batches = fetch_batches(spec, frame)
        day_frame = fetch_same_day(spec, frame)
        identity_outputs.append(identity_frame)
        batch_outputs.append(batches)
        summaries.append(summarize(spec, frame, identity_frame, batches, day_frame))
        print(f"{spec.key}: {len(frame)} identities, {int(batches.identities.sum())} destination observations")

    pd.concat(identity_outputs, ignore_index=True).to_csv(
        OUTPUT / "unresolved_attractor_identity_audit.csv", index=False
    )
    pd.concat(batch_outputs, ignore_index=True).to_csv(
        OUTPUT / "unresolved_attractor_batches.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(
        OUTPUT / "unresolved_attractor_summary.csv", index=False
    )
    population, positions = chengde_cid_population()
    population.to_csv(OUTPUT / "chengde_cid22812_population.csv", index=False)
    positions.to_csv(OUTPUT / "chengde_cid22812_positions.csv", index=False)


if __name__ == "__main__":
    main()
