#!/usr/bin/env python3
"""High-specificity, read-only searches for nonphysical cell coordinates.

Implements the three strongest proposed detectors:

1. exact contemporaneous home/away observations for one full cell identity;
2. synchronized multi-cell change points with persistence and return controls;
3. the observable necessary condition for receiver-batch common-mode motion.

Method 3 cannot be identified fully because the database has no receiver,
device, scan, submission, collector, or provider identifier.  Its output is
therefore explicitly an upper-bound audit, never a receiver attribution.

The global method-1 query reads the large raw table twice and is intentionally
gated behind ``--refresh``.  ClickHouse access is forced to ``readonly=2`` by
``ch_remote.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "spoofing" / "high_quality"
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]
EVENT_KEY = ["src_lat10", "src_lon10", "onset_day", "dest_lat5", "dest_lon5"]
EARTH_KM = 6371.0

MIN_REFERENCE_MONTHS = 4
MIN_REFERENCE_STABILITY = 0.70
HOME_RADIUS_KM = 5.0
DUAL_MIN_DISTANCE_KM = 500.0
DUAL_MAX_GAP_SECONDS = 3600
CHANGE_MIN_DISTANCE_KM = 100.0
CHANGE_MIN_CIDS = 5
DESTINATION_BIN_HUNDREDTHS = 5


METHOD1_WINDOWS_SQL = f"""
WITH candidates AS (
  SELECT mcc,mnc,lac,cid,cell_type,any(hlat) AS hlat,any(hlon) AS hlon
  FROM cell.displaced
  WHERE km>={DUAL_MIN_DISTANCE_KM} AND obs>=2
    AND NOT (abs(plat)<=1 AND abs(plon)<=1)
  GROUP BY mcc,mnc,lac,cid,cell_type
), relevant AS (
  SELECT g.mcc,g.mnc,g.lac,g.cid,g.cell_type,g.timestamp,g.lat,g.lon,
         c.hlat,c.hlon,
         greatCircleDistance(g.lon,g.lat,c.hlon/100,c.hlat/100) AS distance_m
  FROM cell.geos AS g
  INNER JOIN candidates AS c USING (mcc,mnc,lac,cid,cell_type)
  WHERE (distance_m<={HOME_RADIUS_KM * 1000} OR
         distance_m>={DUAL_MIN_DISTANCE_KM * 1000})
    AND g.lat BETWEEN -90 AND 90 AND g.lon BETWEEN -180 AND 180
    AND NOT (abs(g.lat)<=0.01 AND abs(g.lon)<=0.01)
), windows AS (
  SELECT *,toStartOfInterval(timestamp,INTERVAL 2 HOUR,
           toDateTime('1970-01-01 00:00:00')) AS window_start,0 AS window_shift
  FROM relevant
  UNION ALL
  SELECT *,toStartOfInterval(timestamp,INTERVAL 2 HOUR,
           toDateTime('1970-01-01 01:00:00')) AS window_start,1 AS window_shift
  FROM relevant
), groups AS (
 SELECT mcc,mnc,lac,cid,cell_type,hlat,hlon,window_start,window_shift,
        countIf(distance_m<={HOME_RADIUS_KM * 1000}) AS home_obs,
        countIf(distance_m>={DUAL_MIN_DISTANCE_KM * 1000}) AS away_obs,
        groupArrayIf(toUInt32(timestamp),distance_m<={HOME_RADIUS_KM * 1000}) AS home_ts,
        groupArrayIf(toUInt32(timestamp),distance_m>={DUAL_MIN_DISTANCE_KM * 1000}) AS away_ts,
        argMaxIf(lat,timestamp,distance_m>={DUAL_MIN_DISTANCE_KM * 1000}) AS away_lat,
        argMaxIf(lon,timestamp,distance_m>={DUAL_MIN_DISTANCE_KM * 1000}) AS away_lon,
        maxIf(distance_m,distance_m>={DUAL_MIN_DISTANCE_KM * 1000})/1000 AS max_distance_km,
        minIf(timestamp,distance_m>={DUAL_MIN_DISTANCE_KM * 1000}) AS first_away,
        maxIf(timestamp,distance_m>={DUAL_MIN_DISTANCE_KM * 1000}) AS last_away
 FROM windows
 GROUP BY mcc,mnc,lac,cid,cell_type,hlat,hlon,window_start,window_shift
 HAVING home_obs>0 AND away_obs>0
)
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
       hlat/100 AS home_lat,hlon/100 AS home_lon,
       window_start,window_shift,home_obs,away_obs,away_lat,away_lon,
       max_distance_km,first_away,last_away,
       arrayMin(arrayMap(a -> arrayMin(arrayMap(
         h -> abs(toInt64(a)-toInt64(h)),home_ts)),away_ts)) AS min_gap_seconds
FROM groups
WHERE min_gap_seconds<={DUAL_MAX_GAP_SECONDS}
"""


METHOD2_EVENTS_SQL = f"""
SELECT o.src_lat10,o.src_lon10,o.onset_day,
       intDiv(o.top_plat,{DESTINATION_BIN_HUNDREDTHS})
         *{DESTINATION_BIN_HUNDREDTHS} AS dest_lat5,
       intDiv(o.top_plon,{DESTINATION_BIN_HUNDREDTHS})
         *{DESTINATION_BIN_HUNDREDTHS} AS dest_lon5,
       count() AS identities,uniqExact(o.cid) AS distinct_cids,
       uniqExact((o.mcc,o.mnc)) AS operators,
       uniqExact(o.cell_type) AS technologies,uniqExact(o.mcc) AS mccs,
       sum(o.away_obs) AS away_observations,
       median(o.med_km) AS median_displacement_km,
       min(o.onset_ts) AS first_onset,max(o.onset_ts) AS last_onset,
       median(st.stab_frac) AS median_reference_stability,
       min(st.stab_frac) AS min_reference_stability
FROM spoof.onsets_f AS o
INNER JOIN spoof.cellref_stability AS st USING (mcc,mnc,lac,cid,cell_type)
INNER JOIN spoof.cellref AS r USING (mcc,mnc,lac,cid,cell_type)
WHERE o.med_km>={CHANGE_MIN_DISTANCE_KM}
  AND st.stab_frac>={MIN_REFERENCE_STABILITY}
  AND r.n_months>={MIN_REFERENCE_MONTHS}
  AND NOT (abs(o.top_plat)<=1 AND abs(o.top_plon)<=1)
GROUP BY o.src_lat10,o.src_lon10,o.onset_day,dest_lat5,dest_lon5
HAVING distinct_cids>={CHANGE_MIN_CIDS}
ORDER BY identities DESC
"""


METHOD3_SQL = f"""
WITH rows AS (
 SELECT o.mcc AS mcc,o.mnc AS mnc,o.lac AS lac,o.cid AS cid,
        o.cell_type AS cell_type,o.onset_ts AS onset_ts,
        a.plat AS plat,a.plon AS plon,r.rlat AS rlat,r.rlon AS rlon,a.km AS km,
        intDiv(a.plat,{DESTINATION_BIN_HUNDREDTHS})
          *{DESTINATION_BIN_HUNDREDTHS} AS dest_lat5,
        intDiv(a.plon,{DESTINATION_BIN_HUNDREDTHS})
          *{DESTINATION_BIN_HUNDREDTHS} AS dest_lon5
 FROM spoof.onsets_f o
 INNER JOIN spoof.away a ON o.mcc=a.mcc AND o.mnc=a.mnc AND o.lac=a.lac
   AND o.cid=a.cid AND o.cell_type=a.cell_type AND a.t_first=o.onset_ts
 INNER JOIN spoof.cellref r ON o.mcc=r.mcc AND o.mnc=r.mnc AND o.lac=r.lac
   AND o.cid=r.cid AND o.cell_type=r.cell_type
 INNER JOIN spoof.cellref_stability st ON o.mcc=st.mcc AND o.mnc=st.mnc
   AND o.lac=st.lac AND o.cid=st.cid AND o.cell_type=st.cell_type
 WHERE a.km>={CHANGE_MIN_DISTANCE_KM}
   AND st.stab_frac>={MIN_REFERENCE_STABILITY}
   AND r.n_months>={MIN_REFERENCE_MONTHS}
   AND NOT (abs(a.plat)<=1 AND abs(a.plon)<=1)
), centers AS (
 SELECT onset_ts,dest_lat5,dest_lon5,median(rlat) AS center_rlat,
        median(rlon) AS center_rlon,median(plat) AS center_plat,
        median(plon) AS center_plon
 FROM rows GROUP BY onset_ts,dest_lat5,dest_lon5
 HAVING uniqExact(cid)>={CHANGE_MIN_CIDS}
)
SELECT x.onset_ts,x.dest_lat5,x.dest_lon5,count() AS identities,
       uniqExact(x.cid) AS distinct_cids,
       uniqExact((x.mcc,x.mnc)) AS operators,uniqExact(x.mcc) AS mccs,
       uniqExact(x.cell_type) AS technologies,
       c.center_rlat/100 AS source_lat,c.center_rlon/100 AS source_lon,
       c.center_plat/100 AS destination_lat,c.center_plon/100 AS destination_lon,
       quantileExact(0.9)(greatCircleDistance(x.rlon/100,x.rlat/100,
         c.center_rlon/100,c.center_rlat/100)/1000) AS source_radius_p90_km,
       quantileExact(0.9)(greatCircleDistance(x.plon/100,x.plat/100,
         c.center_plon/100,c.center_plat/100)/1000) AS destination_radius_p90_km,
       median(x.km) AS median_displacement_km
FROM rows x INNER JOIN centers c ON x.onset_ts=c.onset_ts
 AND x.dest_lat5=c.dest_lat5 AND x.dest_lon5=c.dest_lon5
GROUP BY x.onset_ts,x.dest_lat5,x.dest_lon5,c.center_rlat,c.center_rlon,
         c.center_plat,c.center_plon
ORDER BY distinct_cids DESC,onset_ts
"""


def haversine(lat1, lon1, lat2, lon2):
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def sql_identity_tuples(frame: pd.DataFrame) -> str:
    return ",".join(
        f"({int(r.mcc)},{int(r.mnc)},{int(r.lac)},{int(r.cid)},'{r.cell_type}')"
        for r in frame[KEY].drop_duplicates().itertuples(index=False)
    )


def robust_references(ids: pd.DataFrame) -> pd.DataFrame:
    tuples = sql_identity_tuples(ids)
    return ch_df(f"""
      SELECT r.mcc,r.mnc,r.lac,r.cid,toString(r.cell_type) AS cell_type,
             r.rlat/100 AS robust_ref_lat,r.rlon/100 AS robust_ref_lon,
             r.n_months,s.stab_frac,s.max_month_km
      FROM spoof.cellref r INNER JOIN spoof.cellref_stability s
        USING (mcc,mnc,lac,cid,cell_type)
      WHERE (r.mcc,r.mnc,r.lac,r.cid,toString(r.cell_type)) IN ({tuples})
    """)


def strict_method1_ids(windows: pd.DataFrame) -> pd.DataFrame:
    refs = robust_references(windows)
    merged = windows.merge(refs, on=KEY, how="left")
    merged["screen_to_robust_ref_km"] = haversine(
        merged.home_lat, merged.home_lon,
        merged.robust_ref_lat, merged.robust_ref_lon,
    )
    keep = merged[
        (merged.stab_frac >= MIN_REFERENCE_STABILITY)
        & (merged.n_months >= MIN_REFERENCE_MONTHS)
        & (merged.screen_to_robust_ref_km <= HOME_RADIUS_KM)
    ]
    return keep[KEY].drop_duplicates().merge(refs, on=KEY)


def fetch_method1_raw(ids: pd.DataFrame) -> pd.DataFrame:
    key_tuples = sql_identity_tuples(ids)
    reference_tuples = ",".join(
        "tuple(toUInt16({}),toUInt16({}),toUInt32({}),toInt64({}),'{}',"
        "toFloat64({}),toFloat64({}))".format(
            int(r.mcc), int(r.mnc), int(r.lac), int(r.cid), r.cell_type,
            r.robust_ref_lat, r.robust_ref_lon,
        )
        for r in ids.itertuples(index=False)
    )
    return ch_df(f"""
      WITH refs AS (
       SELECT tupleElement(x,1) mcc,tupleElement(x,2) mnc,
              tupleElement(x,3) lac,tupleElement(x,4) cid,
              tupleElement(x,5) cell_type,tupleElement(x,6) ref_lat,
              tupleElement(x,7) ref_lon
       FROM (SELECT arrayJoin([{reference_tuples}]) x)
      )
      SELECT g.mcc,g.mnc,g.lac,g.cid,toString(g.cell_type) cell_type,
             g.timestamp,g.lat,g.lon,r.ref_lat,r.ref_lon,
             greatCircleDistance(g.lon,g.lat,r.ref_lon,r.ref_lat)/1000 distance_km,
             if(distance_km<={HOME_RADIUS_KM},'home','away') location_class
      FROM cell.geos g INNER JOIN refs r ON g.mcc=r.mcc AND g.mnc=r.mnc
       AND g.lac=r.lac AND g.cid=r.cid AND toString(g.cell_type)=r.cell_type
      WHERE (g.mcc,g.mnc,g.lac,g.cid,toString(g.cell_type)) IN ({key_tuples})
       AND (distance_km<={HOME_RADIUS_KM} OR distance_km>={DUAL_MIN_DISTANCE_KM})
       AND NOT (abs(g.lat)<=0.01 AND abs(g.lon)<=0.01)
      ORDER BY g.mcc,g.mnc,g.lac,g.cid,g.cell_type,g.timestamp
    """, settings={"max_threads": 6, "max_memory_usage": 30_000_000_000})


def exact_pairs(raw: pd.DataFrame) -> pd.DataFrame:
    raw["timestamp"] = pd.to_datetime(raw.timestamp)
    rows = []
    for identity, group in raw.groupby(KEY, sort=False):
        home = group[group.location_class.eq("home")].sort_values("timestamp")
        away = group[group.location_class.eq("away")].sort_values("timestamp")
        if home.empty or away.empty:
            continue
        home_seconds = home.timestamp.values.astype("datetime64[s]").astype("int64")
        away_seconds = away.timestamp.values.astype("datetime64[s]").astype("int64")
        index = np.searchsorted(home_seconds, away_seconds)
        left = np.clip(index - 1, 0, len(home_seconds) - 1)
        right = np.clip(index, 0, len(home_seconds) - 1)
        nearest = np.where(
            abs(home_seconds[left] - away_seconds) <= abs(home_seconds[right] - away_seconds),
            left, right,
        )
        gaps = abs(home_seconds[nearest] - away_seconds)
        for away_i in np.flatnonzero(gaps <= DUAL_MAX_GAP_SECONDS):
            a = away.iloc[away_i]
            h = home.iloc[nearest[away_i]]
            rows.append(dict(zip(KEY, identity, strict=True)) | {
                "home_timestamp": h.timestamp, "away_timestamp": a.timestamp,
                "gap_seconds": int(gaps[away_i]),
                "home_lat": h.lat, "home_lon": h.lon,
                "away_lat": a.lat, "away_lon": a.lon,
                "distance_km": a.distance_km,
                "reference_lat": a.ref_lat, "reference_lon": a.ref_lon,
            })
    result = pd.DataFrame(rows)
    return result.drop_duplicates(KEY + ["away_timestamp", "away_lat", "away_lon"])


def summarize_method1(pairs: pd.DataFrame, references: pd.DataFrame) -> None:
    pairs = pairs.sort_values(KEY + ["away_timestamp"])
    pairs.to_csv(OUTPUT / "method1_exact_dual_pairs.csv", index=False)
    working = pairs.assign(day=pairs.away_timestamp.dt.date)
    summary = working.groupby(KEY).agg(
        dual_away_observations=("away_timestamp", "size"),
        dual_days=("day", "nunique"), first_dual=("away_timestamp", "min"),
        last_dual=("away_timestamp", "max"), min_gap_seconds=("gap_seconds", "min"),
        median_gap_seconds=("gap_seconds", "median"),
        max_distance_km=("distance_km", "max"),
    ).reset_index().merge(references, on=KEY, how="left")
    summary.to_csv(OUTPUT / "method1_identity_summary.csv", index=False)

    working["dest_lat5"] = (working.away_lat * 20).round().astype(int) * 5
    working["dest_lon5"] = (working.away_lon * 20).round().astype(int) * 5
    endpoint_rows = []
    for (lat5, lon5), group in working.groupby(["dest_lat5", "dest_lon5"]):
        identities = group.drop_duplicates(KEY)
        center_lat = identities.reference_lat.median()
        center_lon = identities.reference_lon.median()
        endpoint_rows.append({
            "dest_lat5": lat5, "dest_lon5": lon5,
            "dual_observations": len(group), "unique_identities": len(identities),
            "operators": identities[["mcc", "mnc"]].drop_duplicates().shape[0],
            "mccs": identities.mcc.nunique(), "days": group.day.nunique(),
            "first_dual": group.away_timestamp.min(),
            "last_dual": group.away_timestamp.max(),
            "min_gap_seconds": group.gap_seconds.min(),
            "median_gap_seconds": group.gap_seconds.median(),
            "median_distance_km": group.distance_km.median(),
            "source_lat": center_lat, "source_lon": center_lon,
            "source_radius_p90_km": np.quantile(haversine(
                identities.reference_lat, identities.reference_lon,
                center_lat, center_lon,
            ), 0.9),
            "destination_lat": group.away_lat.median(),
            "destination_lon": group.away_lon.median(),
        })
    pd.DataFrame(endpoint_rows).sort_values(
        ["unique_identities", "dual_observations"], ascending=False
    ).to_csv(OUTPUT / "method1_endpoint_summary.csv", index=False)

    batches = working.assign(
        away_minute=working.away_timestamp.dt.floor("min")
    ).groupby(["away_minute", "dest_lat5", "dest_lon5"]).agg(
        identities=("cid", "size"), distinct_cids=("cid", "nunique"),
        operators=("mnc", "nunique"), mccs=("mcc", "nunique"),
        median_distance_km=("distance_km", "median"),
        min_gap_seconds=("gap_seconds", "min"),
    ).reset_index()
    batches[batches.distinct_cids >= 2].sort_values(
        ["distinct_cids", "away_minute"], ascending=[False, True]
    ).to_csv(OUTPUT / "method1_common_minute_batches.csv", index=False)


def method2_event_tuples(events: pd.DataFrame) -> str:
    return ",".join(
        f"({int(r.src_lat10)},{int(r.src_lon10)},toDate('{r.onset_day}'),"
        f"{int(r.dest_lat5)},{int(r.dest_lon5)})"
        for r in events.itertuples(index=False)
    )


def fetch_method2_members(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = method2_event_tuples(events)
    members = ch_df(f"""
      SELECT o.mcc AS mcc,o.mnc AS mnc,o.lac AS lac,o.cid AS cid,
             toString(o.cell_type) AS cell_type,o.src_lat10,o.src_lon10,
             o.onset_day,o.onset_ts,
             intDiv(o.top_plat,{DESTINATION_BIN_HUNDREDTHS})
               *{DESTINATION_BIN_HUNDREDTHS} AS dest_lat5,
             intDiv(o.top_plon,{DESTINATION_BIN_HUNDREDTHS})
               *{DESTINATION_BIN_HUNDREDTHS} AS dest_lon5,
             o.top_plat,o.top_plon,o.away_obs,o.med_km,o.max_km,
             r.rlat,r.rlon,r.n_months AS reference_months,
             st.stab_frac AS reference_stability
      FROM spoof.onsets_f o
      INNER JOIN spoof.cellref r ON o.mcc=r.mcc AND o.mnc=r.mnc
       AND o.lac=r.lac AND o.cid=r.cid AND o.cell_type=r.cell_type
      INNER JOIN spoof.cellref_stability st ON o.mcc=st.mcc AND o.mnc=st.mnc
       AND o.lac=st.lac AND o.cid=st.cid AND o.cell_type=st.cell_type
      WHERE o.med_km>={CHANGE_MIN_DISTANCE_KM}
       AND st.stab_frac>={MIN_REFERENCE_STABILITY}
       AND r.n_months>={MIN_REFERENCE_MONTHS}
       AND (o.src_lat10,o.src_lon10,o.onset_day,
            intDiv(o.top_plat,{DESTINATION_BIN_HUNDREDTHS})
              *{DESTINATION_BIN_HUNDREDTHS},
            intDiv(o.top_plon,{DESTINATION_BIN_HUNDREDTHS})
              *{DESTINATION_BIN_HUNDREDTHS}) IN ({selected})
    """, settings={"max_threads": 4})
    identity_tuples = sql_identity_tuples(members)
    away = ch_df(f"""
      SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
             plat,plon,obs,t_first,t_last,km
      FROM spoof.away
      WHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({identity_tuples})
    """, settings={"max_threads": 4})
    positions = ch_df(f"""
      SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
             plat,plon,obs,first_seen,last_seen
      FROM cell.cellpos
      WHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({identity_tuples})
    """, settings={"max_threads": 4})
    return members, away, positions


def enrich_method2_members(
    members: pd.DataFrame, away: pd.DataFrame, positions: pd.DataFrame,
) -> pd.DataFrame:
    members["onset_ts"] = pd.to_datetime(members.onset_ts)
    away[["t_first", "t_last"]] = away[["t_first", "t_last"]].apply(pd.to_datetime)
    positions[["first_seen", "last_seen"]] = positions[
        ["first_seen", "last_seen"]
    ].apply(pd.to_datetime)
    rows = []
    for member in members.itertuples(index=False):
        identity = tuple(getattr(member, column) for column in KEY)
        away_mask = np.ones(len(away), dtype=bool)
        home_mask = np.ones(len(positions), dtype=bool)
        for column, value in zip(KEY, identity, strict=True):
            away_mask &= away[column].eq(value)
            home_mask &= positions[column].eq(value)
        destination = away[
            away_mask
            & (np.trunc(away.plat / DESTINATION_BIN_HUNDREDTHS).astype(int)
               * DESTINATION_BIN_HUNDREDTHS == member.dest_lat5)
            & (np.trunc(away.plon / DESTINATION_BIN_HUNDREDTHS).astype(int)
               * DESTINATION_BIN_HUNDREDTHS == member.dest_lon5)
        ]
        home = positions[
            home_mask & positions.plat.eq(member.rlat) & positions.plon.eq(member.rlon)
        ]
        if destination.empty:
            continue
        first_destination = destination.t_first.min()
        last_destination = destination.t_last.max()
        first_home = home.first_seen.min() if len(home) else pd.NaT
        last_home = home.last_seen.max() if len(home) else pd.NaT
        rows.append(member._asdict() | {
            "destination_obs": int(destination.obs.sum()),
            "destination_first": first_destination,
            "destination_last": last_destination,
            "destination_squares": len(destination),
            "home_obs": int(home.obs.sum()) if len(home) else 0,
            "home_first": first_home, "home_last": last_home,
            "home_seen_after_onset": bool(
                pd.notna(last_home) and last_home > member.onset_ts
            ),
            "returned_after_destination": bool(
                pd.notna(last_home) and last_home > last_destination
            ),
            "home_interval_overlaps_destination": bool(
                pd.notna(last_home) and first_home <= last_destination
                and last_home >= first_destination
            ),
            "destination_span_days": (
                last_destination - first_destination
            ).total_seconds() / 86400,
        })
    return pd.DataFrame(rows)


def summarize_method2(members: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    members.to_csv(OUTPUT / "method2_members.csv", index=False)
    grouped = members.groupby(EVENT_KEY)
    events = grouped.agg(
        identities=("cid", "size"), distinct_cids=("cid", "nunique"),
        operators=("mnc", "nunique"), technologies=("cell_type", "nunique"),
        mccs=("mcc", "nunique"), first_onset=("onset_ts", "min"),
        last_onset=("onset_ts", "max"),
        destination_observations=("destination_obs", "sum"),
        median_displacement_km=("med_km", "median"),
        median_reference_stability=("reference_stability", "median"),
        home_after_fraction=("home_seen_after_onset", "mean"),
        return_fraction=("returned_after_destination", "mean"),
        home_overlap_fraction=("home_interval_overlaps_destination", "mean"),
        median_destination_span_days=("destination_span_days", "median"),
    ).reset_index()
    events["onset_span_hours"] = (
        events.last_onset - events.first_onset
    ).dt.total_seconds() / 3600
    geometry = []
    for key, group in grouped:
        destination_lat = group.top_plat.median() / 100
        destination_lon = group.top_plon.median() / 100
        source_lat = group.rlat.median() / 100
        source_lon = group.rlon.median() / 100
        geometry.append(dict(zip(EVENT_KEY, key, strict=True)) | {
            "destination_lat": destination_lat,
            "destination_lon": destination_lon,
            "destination_radius_p90_km": np.quantile(haversine(
                group.top_plat / 100, group.top_plon / 100,
                destination_lat, destination_lon,
            ), 0.9),
            "source_lat": source_lat, "source_lon": source_lon,
            "source_radius_p90_km": np.quantile(haversine(
                group.rlat / 100, group.rlon / 100, source_lat, source_lon,
            ), 0.9),
        })
    events = events.merge(pd.DataFrame(geometry), on=EVENT_KEY)
    events["passes_full_change_point_test"] = (
        (events.distinct_cids >= CHANGE_MIN_CIDS)
        & (events.destination_radius_p90_km <= 5)
        & (events.onset_span_hours <= 24)
        & (events.home_after_fraction >= 0.8)
        & (events.return_fraction >= 0.5)
        & (events.home_overlap_fraction >= 0.8)
    )
    events.to_csv(OUTPUT / "method2_event_audit.csv", index=False)

    passed = events[events.passes_full_change_point_test].reset_index(drop=True)
    parent = list(range(len(passed)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    for left in range(len(passed)):
        for right in range(left + 1, len(passed)):
            if abs((pd.Timestamp(passed.onset_day[left])
                    - pd.Timestamp(passed.onset_day[right])).days) > 45:
                continue
            source_distance = haversine(
                passed.source_lat[left], passed.source_lon[left],
                passed.source_lat[right], passed.source_lon[right],
            )
            destination_distance = haversine(
                passed.destination_lat[left], passed.destination_lon[left],
                passed.destination_lat[right], passed.destination_lon[right],
            )
            if source_distance <= 100 and destination_distance <= 20:
                union(left, right)
    passed["component"] = [root(index) for index in range(len(passed))]
    campaign_rows = []
    for component, event_group in passed.groupby("component"):
        selected = members.merge(event_group[EVENT_KEY], on=EVENT_KEY, how="inner")
        identities = selected[KEY].drop_duplicates()
        weights = event_group.distinct_cids
        source_lat = np.average(event_group.source_lat, weights=weights)
        source_lon = np.average(event_group.source_lon, weights=weights)
        destination_lat = np.average(event_group.destination_lat, weights=weights)
        destination_lon = np.average(event_group.destination_lon, weights=weights)
        mirrored = (
            abs(source_lat - destination_lat) < 0.1
            and abs(source_lon + destination_lon) < 0.15
        )
        integer_destination = (
            abs(destination_lat - round(destination_lat)) < 0.01
            and abs(destination_lon - round(destination_lon)) < 0.01
        )
        repeated_digit = (
            abs(destination_lat - 22.22) < 0.02
            and abs(destination_lon - 55.56) < 0.02
        )
        near_null = abs(destination_lat) < 0.1 and abs(destination_lon) < 0.1
        artifact = (
            "near-null coordinate" if near_null else
            "longitude sign inversion" if mirrored else
            "round-coordinate placeholder" if integer_destination else
            "repeated-digit synthetic-coordinate candidate" if repeated_digit else
            "none"
        )
        campaign_rows.append({
            "first_onset": event_group.onset_day.min(),
            "last_onset": event_group.onset_day.max(),
            "event_bins": len(event_group), "identities": len(identities),
            "distinct_cids": identities.cid.nunique(),
            "operators": identities[["mcc", "mnc"]].drop_duplicates().shape[0],
            "technologies": identities.cell_type.nunique(),
            "source_lat": source_lat, "source_lon": source_lon,
            "destination_lat": destination_lat,
            "destination_lon": destination_lon,
            "median_displacement_km": np.average(
                event_group.median_displacement_km, weights=weights
            ),
            "home_after_fraction": selected.home_seen_after_onset.mean(),
            "return_fraction": selected.returned_after_destination.mean(),
            "home_overlap_fraction": selected.home_interval_overlaps_destination.mean(),
            "destination_observations": selected.destination_obs.sum(),
            "artifact_control": artifact,
        })
    campaigns = pd.DataFrame(campaign_rows).sort_values("identities", ascending=False)
    campaigns.insert(0, "campaign_id", [f"CP{i + 1:02d}" for i in range(len(campaigns))])
    campaigns.to_csv(OUTPUT / "method2_campaigns.csv", index=False)
    return events, campaigns


def audit_method3(frame: pd.DataFrame) -> pd.DataFrame:
    frame["necessary_common_mode_signature"] = (
        (frame.source_radius_p90_km <= 50)
        & (frame.destination_radius_p90_km <= 5)
    )
    mirrored = (
        (abs(frame.source_lat - frame.destination_lat) < 0.05)
        & (abs(frame.source_lon + frame.destination_lon) < 0.10)
    )
    round_placeholder = (
        (abs(frame.destination_lat - frame.destination_lat.round()) < 1e-9)
        & (abs(frame.destination_lon - frame.destination_lon.round()) < 1e-9)
    )
    frame["obvious_coordinate_artifact"] = mirrored | round_placeholder
    frame["common_mode_upper_bound"] = (
        frame.necessary_common_mode_signature & ~frame.obvious_coordinate_artifact
    )
    frame["receiver_identifiable"] = False
    return frame


def run_refresh() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("method 1: global shifted-window screen", flush=True)
    windows = ch_df(METHOD1_WINDOWS_SQL, settings={
        "max_threads": 8, "max_memory_usage": 40_000_000_000,
    })
    strict_ids = strict_method1_ids(windows)
    raw = fetch_method1_raw(strict_ids)
    summarize_method1(exact_pairs(raw), strict_ids)

    print("method 2: synchronized change-point event screen", flush=True)
    event_universe = ch_df(METHOD2_EVENTS_SQL, settings={"max_threads": 4})
    event_universe.to_csv(OUTPUT / "method2_event_universe.csv", index=False)
    member_base, away, positions = fetch_method2_members(event_universe)
    summarize_method2(enrich_method2_members(member_base, away, positions))

    print("method 3: exact-onset common-mode upper bound", flush=True)
    audit_method3(ch_df(METHOD3_SQL, settings={"max_threads": 6})).to_csv(
        OUTPUT / "method3_exact_onset_batch_audit.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="run the long read-only remote searches")
    args = parser.parse_args()
    if not args.refresh:
        parser.error("pass --refresh to run the long read-only searches")
    run_refresh()


if __name__ == "__main__":
    main()
