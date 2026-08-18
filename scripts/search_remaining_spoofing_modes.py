#!/usr/bin/env python3
"""Exhaustive follow-up screens for the remaining spoofing hypotheses.

All remote access uses ``ch_remote.py`` and is therefore forced to
ClickHouse ``readonly=2``.  Large global screens operate on server-side
position summaries; raw observations are fetched only for the finite set of
survivors identified by those screens.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "spoofing" / "remaining_search"
HQ = ROOT / "data" / "spoofing" / "high_quality"
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]
EARTH_KM = 6371.0088


MOVING_ROWS_SQL = """
WITH candidates AS (
  SELECT a.mcc AS cmcc,a.mnc AS cmnc,a.lac AS clac,a.cid AS ccid,a.cell_type AS ctype
  FROM spoof.away a
  INNER JOIN spoof.cellref r USING (mcc,mnc,lac,cid,cell_type)
  INNER JOIN spoof.cellref_stability s USING (mcc,mnc,lac,cid,cell_type)
  WHERE a.cid>0 AND a.km>=25 AND a.obs>=2 AND r.n_months>=4 AND s.stab_frac>=0.7
  GROUP BY a.mcc,a.mnc,a.lac,a.cid,a.cell_type
  HAVING uniqExact((a.plat,a.plon))>=4
     AND dateDiff('day',min(a.t_first),max(a.t_first))>=1
)
SELECT a.mcc AS mcc,a.mnc AS mnc,a.lac AS lac,a.cid AS cid,toString(a.cell_type) cell_type,
       a.plat AS plat,a.plon AS plon,a.obs AS obs,a.t_first AS t_first,
       a.t_last AS t_last,a.km AS km,r.rlat AS rlat,r.rlon AS rlon,
       r.n_months AS n_months,s.stab_frac AS stab_frac
FROM spoof.away a
INNER JOIN candidates c ON a.mcc=c.cmcc AND a.mnc=c.cmnc AND a.lac=c.clac
 AND a.cid=c.ccid AND a.cell_type=c.ctype
INNER JOIN spoof.cellref r ON a.mcc=r.mcc AND a.mnc=r.mnc AND a.lac=r.lac
 AND a.cid=r.cid AND a.cell_type=r.cell_type
INNER JOIN spoof.cellref_stability s ON a.mcc=s.mcc AND a.mnc=s.mnc AND a.lac=s.lac
 AND a.cid=s.cid AND a.cell_type=s.cell_type
WHERE a.cid>0 AND a.km>=25 AND a.obs>=2
ORDER BY a.mcc,a.mnc,a.lac,a.cid,a.cell_type,a.t_first,a.plat,a.plon
"""


SHORT_RANGE_EVENTS_SQL = """
SELECT intDiv(h.hlat,10) src_lat10,intDiv(h.hlon,10) src_lon10,
       toDate(p.first_seen) onset_day,p.plat,p.plon,
       uniqExact((p.mcc,p.mnc,p.lac,p.cid,p.cell_type)) identities,
       uniqExact(p.cid) distinct_cids,uniqExact((p.mcc,p.mnc)) operators,
       uniqExact(p.cell_type) technologies,sum(p.obs) observations,
       median(greatCircleDistance(p.plon/100,p.plat/100,
                                  h.hlon/100,h.hlat/100)/1000) median_km,
       min(p.first_seen) first_seen,max(p.last_seen) last_seen,
       median(s.stab_frac) median_reference_stability
FROM cell.cellpos p
INNER JOIN cell.cellhome h USING (mcc,mnc,lac,cid,cell_type)
INNER JOIN spoof.cellref r USING (mcc,mnc,lac,cid,cell_type)
INNER JOIN spoof.cellref_stability s USING (mcc,mnc,lac,cid,cell_type)
WHERE p.cid>0 AND p.obs>=2 AND r.n_months>=4 AND s.stab_frac>=0.7
  AND greatCircleDistance(p.plon/100,p.plat/100,
                          h.hlon/100,h.hlat/100)>=2000
  AND greatCircleDistance(p.plon/100,p.plat/100,
                          h.hlon/100,h.hlat/100)<25000
GROUP BY src_lat10,src_lon10,onset_day,p.plat,p.plon
HAVING distinct_cids>=5
ORDER BY distinct_cids DESC,observations DESC
"""


SMALL_COHORT_EVENTS_SQL = """
SELECT o.src_lat10,o.src_lon10,o.onset_day,
       intDiv(o.top_plat,5)*5 dest_lat5,intDiv(o.top_plon,5)*5 dest_lon5,
       count() identities,uniqExact(o.cid) distinct_cids,
       uniqExact((o.mcc,o.mnc)) operators,uniqExact(o.cell_type) technologies,
       sum(o.away_obs) away_observations,median(o.med_km) median_km,
       min(o.onset_ts) first_onset,max(o.onset_ts) last_onset,
       median(s.stab_frac) median_reference_stability
FROM spoof.onsets_f o
INNER JOIN spoof.cellref r USING (mcc,mnc,lac,cid,cell_type)
INNER JOIN spoof.cellref_stability s USING (mcc,mnc,lac,cid,cell_type)
WHERE o.cid>0 AND o.med_km>=100 AND r.n_months>=4 AND s.stab_frac>=0.7
  AND NOT(abs(o.top_plat)<=1 AND abs(o.top_plon)<=1)
GROUP BY o.src_lat10,o.src_lon10,o.onset_day,dest_lat5,dest_lon5
HAVING distinct_cids BETWEEN 1 AND 4
ORDER BY distinct_cids DESC,away_observations DESC
"""


KINEMATIC_SCREEN_SQL = """
SELECT d.mcc AS mcc,d.mnc AS mnc,d.lac AS lac,d.cid AS cid,toString(d.cell_type) cell_type,
       d.plat AS plat,d.plon AS plon,d.obs away_obs,d.first_seen away_first,d.last_seen away_last,
       d.hlat,d.hlon,h.obs home_obs,h.first_seen home_first,h.last_seen home_last,
       d.km AS km,r.n_months AS n_months,s.stab_frac AS stab_frac,
       dateDiff('second',greatest(d.first_seen,h.first_seen),
                         least(d.last_seen,h.last_seen)) interval_overlap_seconds
FROM cell.displaced d
INNER JOIN cell.cellpos h ON d.mcc=h.mcc AND d.mnc=h.mnc AND d.lac=h.lac
 AND d.cid=h.cid AND d.cell_type=h.cell_type AND d.hlat=h.plat AND d.hlon=h.plon
INNER JOIN spoof.cellref r ON d.mcc=r.mcc AND d.mnc=r.mnc AND d.lac=r.lac
 AND d.cid=r.cid AND d.cell_type=r.cell_type
INNER JOIN spoof.cellref_stability s ON d.mcc=s.mcc AND d.mnc=s.mnc AND d.lac=s.lac
 AND d.cid=s.cid AND d.cell_type=s.cell_type
WHERE d.cid>0 AND d.km>=100 AND d.km<500 AND d.obs>=2 AND h.obs>=2
 AND r.n_months>=4 AND s.stab_frac>=0.7
 AND greatest(d.first_seen,h.first_seen)<=least(d.last_seen,h.last_seen)
ORDER BY d.mcc,d.mnc,d.lac,d.cid,d.cell_type,d.km DESC
"""


MULTIDESTINATION_ROWS_SQL = """
WITH candidates AS (
  SELECT a.mcc AS cmcc,a.mnc AS cmnc,a.lac AS clac,a.cid AS ccid,a.cell_type AS ctype
  FROM spoof.away a
  INNER JOIN spoof.cellref r USING (mcc,mnc,lac,cid,cell_type)
  INNER JOIN spoof.cellref_stability s USING (mcc,mnc,lac,cid,cell_type)
  WHERE a.cid>0 AND a.km>=20 AND r.n_months>=4 AND s.stab_frac>=0.7
  GROUP BY a.mcc,a.mnc,a.lac,a.cid,a.cell_type
  HAVING uniqExact((a.plat,a.plon))>=3 AND countIf(a.obs>=2)>=3
)
SELECT a.mcc AS mcc,a.mnc AS mnc,a.lac AS lac,a.cid AS cid,toString(a.cell_type) cell_type,
       a.plat AS plat,a.plon AS plon,a.obs AS obs,a.t_first AS t_first,
       a.t_last AS t_last,a.km AS km,r.rlat AS rlat,r.rlon AS rlon,
       r.n_months AS n_months,s.stab_frac AS stab_frac
FROM spoof.away a
INNER JOIN candidates c ON a.mcc=c.cmcc AND a.mnc=c.cmnc AND a.lac=c.clac
 AND a.cid=c.ccid AND a.cell_type=c.ctype
INNER JOIN spoof.cellref r ON a.mcc=r.mcc AND a.mnc=r.mnc AND a.lac=r.lac
 AND a.cid=r.cid AND a.cell_type=r.cell_type
INNER JOIN spoof.cellref_stability s ON a.mcc=s.mcc AND a.mnc=s.mnc AND a.lac=s.lac
 AND a.cid=s.cid AND a.cell_type=s.cell_type
WHERE a.cid>0 AND a.km>=20 AND a.obs>=2
ORDER BY a.mcc,a.mnc,a.lac,a.cid,a.cell_type,a.t_first,a.plat,a.plon
"""


def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))


def normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.rsplit(".", 1)[-1] for column in frame.columns]
    return frame


def refresh_screens() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    queries = {
        "moving_position_screen.csv": MOVING_ROWS_SQL,
        "short_range_events.csv": SHORT_RANGE_EVENTS_SQL,
        "small_cohort_events.csv": SMALL_COHORT_EVENTS_SQL,
        "kinematic_interval_screen.csv": KINEMATIC_SCREEN_SQL,
        "multidestination_position_screen.csv": MULTIDESTINATION_ROWS_SQL,
    }
    for filename, query in queries.items():
        print(f"querying {filename}", flush=True)
        frame = ch_df(query, settings={"max_threads": 6, "max_execution_time": 7200})
        frame.to_csv(OUTPUT / filename, index=False)
        print(f"  {len(frame):,} rows", flush=True)
    mechanism_schema_audit().to_csv(OUTPUT / "mechanism_schema_audit.csv", index=False)


def track_metrics(group: pd.DataFrame) -> dict:
    points = group.sort_values("t_first").drop_duplicates(["plat", "plon"], keep="first")
    lat = points["plat"].to_numpy(dtype=float) / 100
    lon = points["plon"].to_numpy(dtype=float) / 100
    times = (
        pd.to_datetime(points["t_first"]) - pd.Timestamp("1970-01-01")
    ).dt.total_seconds().to_numpy(dtype=float)
    center_lat = float(np.mean(lat))
    xy = np.column_stack([
        (lon - np.mean(lon)) * 111.32 * math.cos(math.radians(center_lat)),
        (lat - np.mean(lat)) * 111.32,
    ])
    _, singular, vh = np.linalg.svd(xy, full_matrices=False)
    linearity = float(singular[0] ** 2 / max(np.sum(singular ** 2), 1e-12))
    projection = xy @ vh[0]
    projected = np.sort(projection)
    projected_span = float(projected[-1] - projected[0])
    max_projected_gap_fraction = (
        float(np.diff(projected).max() / projected_span)
        if len(projected) > 1 and projected_span > 0 else 1.0
    )
    time_rank = pd.Series(times).rank().to_numpy()
    projection_rank = pd.Series(projection).rank().to_numpy()
    time_correlation = float(abs(np.corrcoef(time_rank, projection_rank)[0, 1]))
    pairwise = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    span = float(np.max(pairwise))
    step = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]) if len(points) > 1 else np.array([])
    cumulative = float(np.sum(step))
    efficiency = float(haversine(lat[0], lon[0], lat[-1], lon[-1]) / cumulative) if cumulative else 0.0
    gaps = np.diff(np.sort(times)) / 86400
    return {
        "positions": len(points), "observations": int(points["obs"].sum()),
        "first_position": points["t_first"].min(), "last_position": points["t_first"].max(),
        "onset_span_days": float((times.max() - times.min()) / 86400),
        "max_gap_days": float(gaps.max()) if len(gaps) else 0.0,
        "track_span_km": span, "linearity": linearity,
        "absolute_spearman_time_projection": time_correlation,
        "max_projected_gap_fraction": max_projected_gap_fraction,
        "path_efficiency": efficiency,
        "start_lat": lat[0], "start_lon": lon[0], "end_lat": lat[-1], "end_lon": lon[-1],
        "source_lat": float(points["rlat"].iloc[0]) / 100,
        "source_lon": float(points["rlon"].iloc[0]) / 100,
    }


def analyze_moving() -> pd.DataFrame:
    raw = normalize_headers(pd.read_csv(
        OUTPUT / "moving_position_screen.csv", parse_dates=["t_first", "t_last"]
    ))
    rows = []
    for identity, group in raw.groupby(KEY, sort=False):
        metrics = track_metrics(group)
        rows.append(dict(zip(KEY, identity, strict=True)) | metrics)
    audit = pd.DataFrame(rows)
    audit["passes_geometry"] = (
        audit["positions"].ge(4) & audit["observations"].ge(8)
        & audit["track_span_km"].ge(25) & audit["linearity"].ge(0.85)
        & audit["absolute_spearman_time_projection"].ge(0.75)
        & audit["path_efficiency"].ge(0.55)
        & audit["max_projected_gap_fraction"].le(0.65)
        & audit["max_gap_days"].le(30)
    )
    audit.sort_values(
        ["passes_geometry", "positions", "track_span_km"], ascending=[False, False, False]
    ).to_csv(OUTPUT / "moving_track_audit.csv", index=False)
    return audit[audit["passes_geometry"]].copy()


def analyze_short_range() -> pd.DataFrame:
    events = pd.read_csv(OUTPUT / "short_range_events.csv", parse_dates=["onset_day", "first_seen", "last_seen"])
    events["distance_band"] = np.where(events["median_km"] < 10, "2-10 km", "10-25 km")
    events["multi_network"] = (events["operators"] >= 2) | (events["technologies"] >= 2)
    events["high_signal"] = (
        events["median_km"].ge(10) & events["distinct_cids"].ge(10)
        & events["multi_network"] & events["median_reference_stability"].ge(0.8)
    )
    repetitions = events.groupby(
        ["src_lat10", "src_lon10", "plat", "plon"], as_index=False
    ).agg(
        event_days=("onset_day", "nunique"), total_identities=("identities", "sum"),
        max_distinct_cids=("distinct_cids", "max"), first_event=("onset_day", "min"),
        last_event=("onset_day", "max"), median_km=("median_km", "median"),
        max_operators=("operators", "max"), high_signal_days=("high_signal", "sum"),
    )
    repetitions["repeated_high_signal"] = (
        repetitions["event_days"].ge(2) & repetitions["high_signal_days"].ge(1)
    )
    repetitions.sort_values(
        ["repeated_high_signal", "max_distinct_cids", "event_days"], ascending=False
    ).to_csv(OUTPUT / "short_range_repeated_destinations.csv", index=False)
    events.sort_values(
        ["high_signal", "distinct_cids", "observations"], ascending=False
    ).to_csv(OUTPUT / "short_range_event_audit.csv", index=False)
    return events[events["high_signal"]].copy()


def analyze_small_cohorts() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(OUTPUT / "small_cohort_events.csv", parse_dates=["onset_day", "first_onset", "last_onset"])
    events["multi_network"] = (events["operators"] >= 2) | (events["technologies"] >= 2)
    events["coordinated_candidate"] = (
        events["distinct_cids"].ge(2) & events["multi_network"]
        & events["away_observations"].ge(events["distinct_cids"] * 2)
    )
    events.sort_values(
        ["coordinated_candidate", "distinct_cids", "away_observations"], ascending=False
    ).to_csv(OUTPUT / "small_cohort_event_audit.csv", index=False)

    pairs = pd.read_csv(HQ / "method1_exact_dual_pairs.csv", parse_dates=["away_timestamp"])
    pairs = pairs[pairs["cid"] > 0].copy()
    pairs["dest_lat5"] = (pairs["away_lat"] * 20).round().astype(int) * 5
    pairs["dest_lon5"] = (pairs["away_lon"] * 20).round().astype(int) * 5
    endpoint = pairs.groupby(["dest_lat5", "dest_lon5"], as_index=False).agg(
        exact_pairs=("cid", "size"), identities=("cid", "nunique"),
        operators=("mnc", "nunique"), mccs=("mcc", "nunique"),
        first_exact_pair=("away_timestamp", "min"), last_exact_pair=("away_timestamp", "max"),
        min_gap_seconds=("gap_seconds", "min"), median_distance_km=("distance_km", "median"),
    )
    small = endpoint[endpoint["identities"].between(1, 4)].copy()
    small["repeated_exact_dual"] = small["exact_pairs"] > small["identities"]
    small.sort_values(
        ["repeated_exact_dual", "identities", "exact_pairs"], ascending=False
    ).to_csv(OUTPUT / "small_cohort_exact_dual_endpoints.csv", index=False)
    return events[events["coordinated_candidate"]].copy(), small


def analyze_multidestination() -> pd.DataFrame:
    raw = normalize_headers(pd.read_csv(
        OUTPUT / "multidestination_position_screen.csv", parse_dates=["t_first", "t_last"]
    ))
    rows = []
    for identity, group in raw.groupby(KEY, sort=False):
        group = group.sort_values("t_first").drop_duplicates(["plat", "plon"], keep="first")
        coords = group[["plat", "plon"]].to_numpy(dtype=int)
        sequence = [f"{lat}:{lon}" for lat, lon in coords]
        unique = len(set(sequence))
        revisits = max(len(sequence) - unique, 0)
        # Aggregated rows contain one interval per coordinate, so a true revisit
        # is conservatively identified when one destination's interval spans the
        # first appearance of at least two other destinations.
        interval_revisit = 0
        for row in group.itertuples(index=False):
            inside = group["t_first"].between(row.t_first, row.t_last, inclusive="neither").sum()
            interval_revisit = max(interval_revisit, int(inside))
        pairwise = haversine(
            coords[:, 0, None] / 100, coords[:, 1, None] / 100,
            coords[None, :, 0] / 100, coords[None, :, 1] / 100,
        )
        separated = int(np.sum(np.max(pairwise, axis=1) >= 20))
        rows.append(dict(zip(KEY, identity, strict=True)) | {
            "source_lat": group["rlat"].iloc[0] / 100,
            "source_lon": group["rlon"].iloc[0] / 100,
            "destinations": unique, "supported_observations": int(group["obs"].sum()),
            "destination_span_km": float(pairwise.max()),
            "separated_destinations": separated,
            "interval_nested_destinations": interval_revisit,
            "first_destination": group["t_first"].min(),
            "last_destination": group["t_last"].max(),
            "sequence_signature": "|".join(sequence),
        })
    audit = pd.DataFrame(rows)
    audit["cycle_candidate"] = (
        audit["destinations"].between(3, 12) & audit["separated_destinations"].ge(3)
        & audit["destination_span_km"].ge(50)
        & audit["interval_nested_destinations"].ge(2)
    )
    audit.sort_values(
        ["cycle_candidate", "destinations", "supported_observations"], ascending=False
    ).to_csv(OUTPUT / "multidestination_identity_audit.csv", index=False)
    candidates = audit[audit["cycle_candidate"]].copy()
    if not candidates.empty:
        candidates["src_lat10"] = np.trunc(candidates["source_lat"] * 10).astype(int)
        candidates["src_lon10"] = np.trunc(candidates["source_lon"] * 10).astype(int)
        cohorts = candidates.groupby(
            ["src_lat10", "src_lon10", "sequence_signature"], as_index=False
        ).agg(
            identities=("cid", "size"), distinct_cids=("cid", "nunique"),
            operators=("mnc", "nunique"), destinations=("destinations", "max"),
            observations=("supported_observations", "sum"),
            first_destination=("first_destination", "min"),
            last_destination=("last_destination", "max"),
        )
        cohorts.sort_values(["identities", "destinations"], ascending=False).to_csv(
            OUTPUT / "multidestination_cohorts.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(OUTPUT / "multidestination_cohorts.csv", index=False)
    return candidates


def mechanism_schema_audit() -> pd.DataFrame:
    columns = ch_df("""
      SELECT database,table,name,type
      FROM system.columns
      WHERE database IN ('cell','spoof')
      ORDER BY database,table,position
    """)
    patterns = {
        "receiver_or_device_id": ["receiver", "device", "handset", "client"],
        "submission_or_scan_id": ["submission", "scan", "batch", "report"],
        "collector_or_provider": ["collector", "provider", "source", "crawler"],
        "radio_measurement": ["rssi", "rsrp", "rsrq", "sinr", "arfcn", "pci", "timing_advance"],
        "gnss_measurement": ["satellite", "dop", "accuracy", "altitude", "speed", "bearing"],
    }
    rows = []
    for capability, needles in patterns.items():
        mask = columns["name"].str.lower().apply(lambda name: any(n in name for n in needles))
        matches = columns[mask]
        rows.append({
            "capability": capability, "matching_columns": len(matches),
            "columns": ";".join(
                f"{r.database}.{r.table}.{r.name}:{r.type}" for r in matches.itertuples(index=False)
            ),
            "identifiable": bool(len(matches)),
        })
    return pd.DataFrame(rows)


def write_manifest(
    moving: pd.DataFrame, short: pd.DataFrame, small: pd.DataFrame,
    exact_small: pd.DataFrame, multidest: pd.DataFrame,
) -> None:
    kinematic = normalize_headers(pd.read_csv(OUTPUT / "kinematic_interval_screen.csv"))
    schema = pd.read_csv(OUTPUT / "mechanism_schema_audit.csv")
    rows = [
        {"search": "raw-time moving destination", "screened_units": 16232,
         "strict_candidates": len(moving), "status": "awaiting raw-time validation",
         "interpretation": "geometry screen only; mobile infrastructure remains a confounder"},
        {"search": "short-range local displacement", "screened_units": len(pd.read_csv(OUTPUT / "short_range_events.csv")),
         "strict_candidates": len(short), "status": "event screen complete",
         "interpretation": "2-10 km remains compatible with coverage and positioning scatter"},
        {"search": "small cohorts", "screened_units": len(pd.read_csv(OUTPUT / "small_cohort_events.csv")),
         "strict_candidates": len(small), "status": "change-point and exact-dual screens complete",
         "interpretation": f"{len(exact_small)} exact-dual endpoints contain 1-4 identities"},
        {"search": "100-500 km kinematic impossibility", "screened_units": kinematic[KEY].drop_duplicates().shape[0],
         "strict_candidates": np.nan, "status": "awaiting raw timestamp pairing",
         "interpretation": "interval overlap is not exact simultaneity"},
        {"search": "irregular multi-destination cycling", "screened_units": 40113,
         "strict_candidates": len(multidest), "status": "awaiting raw-time validation",
         "interpretation": "aggregated position intervals conservatively identify revisits"},
        {"search": "mechanism attribution", "screened_units": len(schema),
         "strict_candidates": int(schema["identifiable"].sum()), "status": "schema audit complete",
         "interpretation": "missing receiver, submission, provenance, RF, and GNSS fields bound attribution"},
    ]
    pd.DataFrame(rows).to_csv(OUTPUT / "findings_manifest.csv", index=False)


def analyze() -> None:
    moving = analyze_moving()
    short = analyze_short_range()
    small, exact_small = analyze_small_cohorts()
    multidest = analyze_multidestination()
    write_manifest(moving, short, small, exact_small, multidest)
    print(json.dumps({
        "moving_geometry_candidates": len(moving),
        "short_range_high_signal_events": len(short),
        "small_coordinated_events": len(small),
        "small_exact_dual_endpoints": len(exact_small),
        "multidestination_cycle_candidates": len(multidest),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-screens", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.refresh_screens:
        refresh_screens()
    analyze()


if __name__ == "__main__":
    main()
