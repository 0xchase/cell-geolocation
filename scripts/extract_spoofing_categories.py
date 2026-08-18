#!/usr/bin/env python3
"""Extract and classify high-confidence spoofing-related activity.

The remote ClickHouse database is always opened with ``readonly=2``.  The
script first exports complete, auditable evidence tables and then derives six
category-specific CSVs locally:

* fixed GNSS-like decoys;
* moving GNSS-like decoys;
* alternating/multiple GNSS-like decoys;
* long-range cell-identity replay;
* bulk identity rebroadcast;
* coherent wholesale coordinate reassignment.

The category names describe observable behavior.  They do not resolve the two
mechanisms absent from this dataset: a regional platform-side coordinate change
can mimic receiver-location spoofing, and benign RF testing can mimic malicious
identity rebroadcast.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "spoofing"
HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")

MIN_EVENT_CELLS = 9
MIN_DISPLACEMENT_KM = 25.0
DEST_RADIUS_DEG = 0.10
MID_MASS_FIXED = 0.25
MID_MASS_AMBIGUOUS = 0.10
MAX_CROSS_FRACTION = 0.15
LONG_RANGE_KM = 800.0
MIN_REFERENCE_STABILITY = 0.50

MULTI_WINDOW_DAYS = 90
MULTI_GRID_DEG = 0.05
MULTI_MIN_CLUSTER_SHARE = 0.25
MULTI_MIN_CLUSTER_CELLS = 5
MULTI_MIN_SEPARATION_KM = 20.0
MULTI_MIN_SHARED_CELLS = 3

MOVING_MIN_DAYS = 4
MOVING_MIN_CELLS_PER_DAY = 3
MOVING_MIN_TOTAL_CELLS = 9
MOVING_MIN_SPAN_KM = 25.0
MOVING_MIN_LINEARITY = 0.85
MOVING_MIN_TIME_CORRELATION = 0.75
MOVING_MIN_REPEATED_CELLS = 3
MOVING_MIN_DISTINCT_SPATIAL_BINS = 4
MOVING_MAX_GAP_DAYS = 14

MULTI_MIN_BEARING_SEPARATION_DEG = 30.0
MULTI_MIN_SHARED_FRACTION = 0.30

EARTH_KM = 6371.0
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]


EVENTS_SQL = f"""
WITH events AS
(
    SELECT
        src_lat10, src_lon10, onset_day,
        count() AS n_cells,
        uniqExact(mcc) AS n_mcc,
        uniqExact((mcc,mnc)) AS n_operators,
        uniqExact(cell_type) AS n_technologies,
        quantileExact(0.5)(med_km) AS median_displacement_km,
        quantileExact(0.5)(max_km) AS median_max_displacement_km,
        quantileExact(0.5)(top_plat) AS destination_plat,
        quantileExact(0.5)(top_plon) AS destination_plon
    FROM spoof.onsets_f
    WHERE med_km >= {MIN_DISPLACEMENT_KM}
    GROUP BY src_lat10, src_lon10, onset_day
    HAVING n_cells >= {MIN_EVENT_CELLS}
)
SELECT
    concat(toString(e.src_lat10),'_',toString(e.src_lon10),'_',toString(e.onset_day)) AS event_id,
    e.src_lat10 / 10.0 AS source_lat,
    e.src_lon10 / 10.0 AS source_lon,
    e.onset_day,
    e.n_cells, e.n_mcc, e.n_operators, e.n_technologies,
    e.median_displacement_km, e.median_max_displacement_km,
    e.destination_plat / 100.0 AS destination_lat,
    e.destination_plon / 100.0 AS destination_lon,
    s.lead_days, s.day_obs, s.region_obs, s.region_onsets,
    s.expected AS exposure_expected_onsets,
    s.excess AS exposure_normalized_excess
FROM events AS e
LEFT JOIN spoof.sync3 AS s
    ON e.src_lat10=s.src_lat10 AND e.src_lon10=s.src_lon10
   AND e.onset_day=s.onset_day
ORDER BY e.n_cells DESC, e.onset_day, e.src_lat10, e.src_lon10
"""


MEMBERS_SQL = f"""
WITH sig AS
(
    SELECT src_lat10, src_lon10, onset_day
    FROM spoof.onsets_f
    WHERE med_km >= {MIN_DISPLACEMENT_KM}
    GROUP BY src_lat10, src_lon10, onset_day
    HAVING count() >= {MIN_EVENT_CELLS}
), members AS
(
    SELECT o.*
    FROM spoof.onsets_f AS o
    INNER JOIN sig AS s USING (src_lat10,src_lon10,onset_day)
    WHERE o.med_km >= {MIN_DISPLACEMENT_KM}
)
SELECT
    concat(toString(o.src_lat10),'_',toString(o.src_lon10),'_',toString(o.onset_day)) AS event_id,
    o.mcc AS mcc, o.mnc AS mnc, o.lac AS lac, o.cid AS cid,
    toString(o.cell_type) AS cell_type,
    o.src_lat10 / 10.0 AS source_region_lat,
    o.src_lon10 / 10.0 AS source_region_lon,
    o.onset_day AS onset_day, o.onset_ts AS onset_ts,
    o.away_obs AS away_obs, o.med_km AS med_km, o.max_km AS max_km,
    o.top_plat / 100.0 AS top_destination_lat,
    o.top_plon / 100.0 AS top_destination_lon,
    r.rlat / 100.0 AS reference_lat,
    r.rlon / 100.0 AS reference_lon,
    r.n_months AS reference_months,
    r.obs AS reference_input_observations,
    r.m_first AS reference_first_month,
    r.m_last AS reference_last_month,
    st.stab_frac AS reference_stability,
    st.max_month_km AS max_month_km,
    ca.away_squares AS away_squares,
    ca.t_first_away AS t_first_away,
    ca.t_last_away AS t_last_away,
    sum(hp.obs) AS home_position_observations,
    min(hp.first_seen) AS home_first_seen,
    max(hp.last_seen) AS home_last_seen
FROM members AS o
INNER JOIN spoof.cellref AS r USING (mcc,mnc,lac,cid,cell_type)
LEFT JOIN spoof.cellref_stability AS st USING (mcc,mnc,lac,cid,cell_type)
LEFT JOIN spoof.cell_away AS ca USING (mcc,mnc,lac,cid,cell_type)
LEFT JOIN cell.cellpos AS hp
    ON o.mcc=hp.mcc AND o.mnc=hp.mnc AND o.lac=hp.lac
   AND o.cid=hp.cid AND o.cell_type=hp.cell_type
   AND hp.plat=r.rlat AND hp.plon=r.rlon
GROUP BY ALL
ORDER BY 1,2,3,4,5,6
"""


AWAY_SQL = f"""
WITH sig AS
(
    SELECT src_lat10, src_lon10, onset_day
    FROM spoof.onsets_f
    WHERE med_km >= {MIN_DISPLACEMENT_KM}
    GROUP BY src_lat10, src_lon10, onset_day
    HAVING count() >= {MIN_EVENT_CELLS}
), members AS
(
    SELECT
        concat(toString(o.src_lat10),'_',toString(o.src_lon10),'_',toString(o.onset_day)) AS event_id,
        o.mcc,o.mnc,o.lac,o.cid,o.cell_type
    FROM spoof.onsets_f AS o
    INNER JOIN sig AS s USING (src_lat10,src_lon10,onset_day)
    WHERE o.med_km >= {MIN_DISPLACEMENT_KM}
)
SELECT
    m.event_id,
    a.mcc,a.mnc,a.lac,a.cid,toString(a.cell_type) AS cell_type,
    a.plat / 100.0 AS observed_lat,
    a.plon / 100.0 AS observed_lon,
    a.rlat / 100.0 AS reference_lat,
    a.rlon / 100.0 AS reference_lon,
    a.km AS displacement_km,
    a.obs AS observations,
    a.t_first, a.t_last
FROM spoof.away AS a
INNER JOIN members AS m USING (mcc,mnc,lac,cid,cell_type)
ORDER BY m.event_id,a.mcc,a.mnc,a.lac,a.cid,a.cell_type,a.t_first,a.plat,a.plon
"""


AXIS_SQL = f"""
WITH events AS
(
    SELECT
        src_lat10,src_lon10,onset_day,
        quantileExact(0.5)(top_plat) AS destination_plat,
        quantileExact(0.5)(top_plon) AS destination_plon
    FROM spoof.onsets_f
    WHERE med_km >= {MIN_DISPLACEMENT_KM}
    GROUP BY src_lat10,src_lon10,onset_day
    HAVING count() >= {MIN_EVENT_CELLS}
)
SELECT
    concat(toString(e.src_lat10),'_',toString(e.src_lon10),'_',toString(e.onset_day)) AS event_id,
    a.mcc,a.mnc,a.lac,a.cid,toString(a.cell_type) AS cell_type,
    a.plat / 100.0 AS observed_lat,
    a.plon / 100.0 AS observed_lon,
    a.rlat / 100.0 AS reference_lat,
    a.rlon / 100.0 AS reference_lon,
    a.km AS displacement_km,
    a.obs AS observations,
    a.t_first,a.t_last
FROM spoof.away AS a
INNER JOIN events AS e
    ON intDiv(a.rlat,10)=e.src_lat10 AND intDiv(a.rlon,10)=e.src_lon10
WHERE a.plat BETWEEN e.destination_plat-{int(DEST_RADIUS_DEG * 100)}
                 AND e.destination_plat+{int(DEST_RADIUS_DEG * 100)}
  AND a.plon BETWEEN e.destination_plon-{int(DEST_RADIUS_DEG * 100)}
                 AND e.destination_plon+{int(DEST_RADIUS_DEG * 100)}
ORDER BY event_id,a.mcc,a.mnc,a.lac,a.cid,a.cell_type,a.t_first,a.plat,a.plon
"""


ATTRACTORS_SQL = """
SELECT
    concat(toString(plat),'_',toString(plon)) AS site_id,
    plat / 100.0 AS destination_lat,
    plon / 100.0 AS destination_lon,
    cells, obs AS observations, n_mcc, arrayStringConcat(arrayMap(x -> toString(x),top_mcc),';') AS top_mcc,
    t_start, t_end, src_lat, src_lon,
    med_km AS median_displacement_km,
    p90_km AS p90_displacement_km,
    src_spread_km AS source_spread_km
FROM cell.attractors
WHERE cells >= 25 AND med_km >= 100 AND src_spread_km >= 1000
ORDER BY cells DESC,plat,plon
"""


ATTRACTOR_MEMBERS_SQL = """
WITH targets AS
(
    SELECT plat,plon
    FROM cell.attractors
    WHERE cells >= 25 AND med_km >= 100 AND src_spread_km >= 1000
)
SELECT
    concat(toString(d.plat),'_',toString(d.plon)) AS site_id,
    d.mcc,d.mnc,d.lac,d.cid,toString(d.cell_type) AS cell_type,
    d.plat / 100.0 AS destination_lat,
    d.plon / 100.0 AS destination_lon,
    d.hlat / 100.0 AS home_lat,
    d.hlon / 100.0 AS home_lon,
    d.km AS displacement_km,
    d.obs AS observations,
    d.first_seen,d.last_seen,
    d.total_obs,d.home_obs
FROM cell.displaced AS d
INNER JOIN targets AS t USING (plat,plon)
ORDER BY site_id,d.mcc,d.mnc,d.lac,d.cid,d.cell_type
"""


BASE_EXPORTS = {
    "all_synchronized_events.csv": EVENTS_SQL,
    "all_event_members.csv": MEMBERS_SQL,
    "all_event_away_points.csv": AWAY_SQL,
    "all_event_axis_points.csv": AXIS_SQL,
    "all_broad_source_attractors.csv": ATTRACTORS_SQL,
    "all_broad_source_attractor_members.csv": ATTRACTOR_MEMBERS_SQL,
}


def validate_select(sql: str) -> None:
    """Fail locally if a future edit accidentally adds mutating SQL."""
    upper = " " + " ".join(sql.upper().split()) + " "
    forbidden = (" CREATE ", " DROP ", " ALTER ", " INSERT ", " UPDATE ",
                 " DELETE ", " TRUNCATE ", " OPTIMIZE ", " KILL ")
    found = [word.strip() for word in forbidden if word in upper]
    if found:
        raise ValueError(f"refusing non-read-only SQL containing: {', '.join(found)}")
    if not upper.lstrip().startswith(("SELECT ", "WITH ")):
        raise ValueError("query must begin with SELECT or WITH")


def run_csv(name: str, sql: str, output: Path, *, refresh: bool) -> None:
    """Run one read-only ClickHouse query and atomically write CSV output."""
    if output.exists() and not refresh:
        print(f"[cached] {output.relative_to(ROOT)}", flush=True)
        return
    validate_select(sql)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    remote = (
        "clickhouse-client "
        f"--password {shlex.quote(PASSWORD)} --readonly 2 --max_threads 8 "
        "--max_execution_time 1800 --optimize_aggregation_in_order 0"
    )
    query = sql.strip().rstrip(";") + "\nFORMAT CSVWithNames\n"
    print(f"[query] {name} -> {output.relative_to(ROOT)}", flush=True)
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST, remote],
        input=query,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name} failed ({proc.returncode}):\n{proc.stderr.strip()}")
    tmp.write_text(proc.stdout, encoding="utf-8")
    tmp.replace(output)
    rows = max(0, proc.stdout.count("\n") - 1)
    print(f"[data] {rows:,} rows", flush=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + np.pi) % (2 * np.pi) - np.pi
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0, 1 - a)))


def initial_bearing(lat1, lon1, lat2, lon2):
    phi1 = np.radians(np.asarray(lat1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    dlambda = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return np.arctan2(y, x)


def weighted_quantile(values, weights, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return float("nan")
    values, weights = values[ok], weights[ok]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = q * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def axis_metrics(points: pd.DataFrame, dest_lat: float, dest_lon: float) -> dict:
    """Measure weighted home-to-destination mixture geometry on a sphere."""
    if points.empty:
        return {
            "destination_support_cells": 0,
            "destination_observations": 0,
            "median_baseline_km": np.nan,
            "median_along_km": np.nan,
            "median_cross_km": np.nan,
            "cross_fraction": np.nan,
            "intermediate_mass": np.nan,
        }
    hlat = points["reference_lat"].to_numpy(float)
    hlon = points["reference_lon"].to_numpy(float)
    plat = points["observed_lat"].to_numpy(float)
    plon = points["observed_lon"].to_numpy(float)
    weights = points["observations"].to_numpy(float)

    base = haversine_km(hlat, hlon, dest_lat, dest_lon)
    hp = haversine_km(hlat, hlon, plat, plon)
    theta_dest = initial_bearing(hlat, hlon, dest_lat, dest_lon)
    theta_point = initial_bearing(hlat, hlon, plat, plon)
    delta_bearing = np.arctan2(
        np.sin(theta_point - theta_dest), np.cos(theta_point - theta_dest)
    )
    delta_hp = hp / EARTH_KM
    cross_angle = np.arcsin(np.clip(np.sin(delta_hp) * np.sin(delta_bearing), -1, 1))
    along_angle = np.arctan2(
        np.sin(delta_hp) * np.cos(delta_bearing), np.cos(delta_hp)
    )
    cross = np.abs(cross_angle * EARTH_KM)
    along = along_angle * EARTH_KM
    w = np.divide(along, base, out=np.full_like(along, np.nan), where=base > 1e-9)
    finite_w = np.isfinite(w)
    intermediate = (
        float(weights[finite_w & (w > 0.2) & (w < 0.8)].sum() / weights[finite_w].sum())
        if finite_w.any() and weights[finite_w].sum() else np.nan
    )
    med_along = weighted_quantile(np.abs(along), weights, 0.5)
    med_cross = weighted_quantile(cross, weights, 0.5)
    return {
        "destination_support_cells": int(points[KEY].drop_duplicates().shape[0]),
        "destination_observations": int(weights.sum()),
        "median_baseline_km": weighted_quantile(base, weights, 0.5),
        "median_along_km": med_along,
        "median_cross_km": med_cross,
        "cross_fraction": med_cross / med_along if med_along > 0 else np.nan,
        "intermediate_mass": intermediate,
    }


def event_geometry(
    events: pd.DataFrame,
    members: pd.DataFrame,
    member_away: pd.DataFrame,
    axis_points: pd.DataFrame,
) -> pd.DataFrame:
    member_groups = {key: frame for key, frame in members.groupby("event_id", sort=False)}
    away_groups = {key: frame for key, frame in member_away.groupby("event_id", sort=False)}
    axis_groups = {key: frame for key, frame in axis_points.groupby("event_id", sort=False)}
    rows = []
    for event in events.itertuples(index=False):
        event_axis = axis_groups.get(event.event_id, axis_points.iloc[0:0])
        axis_near = event_axis[
            (event_axis["observed_lat"].sub(event.destination_lat).abs() <= DEST_RADIUS_DEG)
            & (event_axis["observed_lon"].sub(event.destination_lon).abs() <= DEST_RADIUS_DEG)
        ]
        axis = axis_metrics(axis_near, event.destination_lat, event.destination_lon)
        metrics = {
            "axis_support_cells": axis.pop("destination_support_cells"),
            "axis_observations": axis.pop("destination_observations"),
            **axis,
        }
        event_points = away_groups.get(event.event_id, member_away.iloc[0:0])
        member_near = event_points[
            (event_points["observed_lat"].sub(event.destination_lat).abs() <= DEST_RADIUS_DEG)
            & (event_points["observed_lon"].sub(event.destination_lon).abs() <= DEST_RADIUS_DEG)
        ]
        member_support = int(member_near[KEY].drop_duplicates().shape[0])
        metrics.update(
            {
                "event_member_destination_support_cells": member_support,
                "event_member_destination_support_fraction": member_support / event.n_cells,
            }
        )
        ev_members = member_groups.get(event.event_id, members.iloc[0:0]).copy()
        if not ev_members.empty:
            onset = pd.Timestamp(event.onset_day)
            home_first = pd.to_datetime(ev_members["home_first_seen"], errors="coerce")
            home_last = pd.to_datetime(ev_members["home_last_seen"], errors="coerce")
            stability = pd.to_numeric(ev_members["reference_stability"], errors="coerce")
            metrics.update(
                {
                    "stable_reference_fraction": float((stability >= MIN_REFERENCE_STABILITY).mean()),
                    "median_reference_stability": float(stability.median()),
                    "home_active_at_onset_fraction": float(((home_first <= onset) & (home_last >= onset)).mean()),
                    "home_seen_after_7d_fraction": float((home_last >= onset + pd.Timedelta(days=7)).mean()),
                }
            )
        else:
            metrics.update(
                {
                    "stable_reference_fraction": np.nan,
                    "median_reference_stability": np.nan,
                    "home_active_at_onset_fraction": np.nan,
                    "home_seen_after_7d_fraction": np.nan,
                }
            )
        rows.append({"event_id": event.event_id, **metrics})

    geometry = pd.DataFrame(rows)
    out = events.merge(geometry, on="event_id", how="left", validate="one_to_one")
    on_axis = out["cross_fraction"].le(MAX_CROSS_FRACTION)
    enough_support = (
        out["axis_support_cells"].ge(MIN_EVENT_CELLS)
        & out["event_member_destination_support_cells"].ge(
            np.maximum(5, np.ceil(out["n_cells"] * 0.50))
        )
    )
    out["geometry_class"] = np.select(
        [
            enough_support & on_axis & out["intermediate_mass"].ge(MID_MASS_FIXED),
            enough_support & on_axis & out["intermediate_mass"].ge(MID_MASS_AMBIGUOUS),
            enough_support & on_axis & out["intermediate_mass"].lt(MID_MASS_AMBIGUOUS),
            enough_support & ~on_axis,
        ],
        ["graded_mixture", "ambiguous_mixture", "coherent_not_mixture", "off_axis"],
        default="insufficient_destination_support",
    )
    out["operator_or_technology_diversity"] = (
        out["n_operators"].ge(2) | out["n_technologies"].ge(2)
    )
    return out


def destination_clusters(points: pd.DataFrame, n_event_cells: int) -> list[dict]:
    if points.empty:
        return []
    p = points.copy()
    p["grid_lat"] = (p["observed_lat"] / MULTI_GRID_DEG).round() * MULTI_GRID_DEG
    p["grid_lon"] = (p["observed_lon"] / MULTI_GRID_DEG).round() * MULTI_GRID_DEG
    min_cells = max(MULTI_MIN_CLUSTER_CELLS, math.ceil(n_event_cells * MULTI_MIN_CLUSTER_SHARE))
    clusters = []
    for (glat, glon), group in p.groupby(["grid_lat", "grid_lon"]):
        cells = set(map(tuple, group[KEY].drop_duplicates().to_numpy()))
        if len(cells) < min_cells:
            continue
        weights = group["observations"].to_numpy(float)
        clusters.append(
            {
                "lat": float(np.average(group["observed_lat"], weights=weights)),
                "lon": float(np.average(group["observed_lon"], weights=weights)),
                "cells": cells,
                "n_cells": len(cells),
                "observations": int(weights.sum()),
            }
        )
    return sorted(clusters, key=lambda c: (-c["n_cells"], -c["observations"]))


def alternating_candidates(events: pd.DataFrame, away: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    event_lookup = events.set_index("event_id")
    rows: list[dict] = []
    event_ids: set[str] = set()
    for event_id, all_points in away.groupby("event_id", sort=False):
        event = event_lookup.loc[event_id]
        onset = pd.Timestamp(event["onset_day"])
        first = pd.to_datetime(all_points["t_first"], errors="coerce")
        points = all_points[(first >= onset - pd.Timedelta(days=2)) &
                            (first <= onset + pd.Timedelta(days=MULTI_WINDOW_DAYS))]
        clusters = destination_clusters(points, int(event["n_cells"]))
        for i, a in enumerate(clusters):
            a_near = points[
                points["observed_lat"].sub(a["lat"]).abs().le(DEST_RADIUS_DEG)
                & points["observed_lon"].sub(a["lon"]).abs().le(DEST_RADIUS_DEG)
            ]
            am = axis_metrics(a_near, a["lat"], a["lon"])
            if not (am["intermediate_mass"] >= 0.20 and am["cross_fraction"] <= 0.20):
                continue
            for b in clusters[i + 1:]:
                separation = float(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]))
                shared = len(a["cells"] & b["cells"])
                shared_fraction = shared / min(a["n_cells"], b["n_cells"])
                if (
                    separation < MULTI_MIN_SEPARATION_KM
                    or shared < MULTI_MIN_SHARED_CELLS
                    or shared_fraction < MULTI_MIN_SHARED_FRACTION
                ):
                    continue
                shared_keys = a["cells"] & b["cells"]
                refs = points.drop_duplicates(KEY).copy()
                refs["_key"] = list(map(tuple, refs[KEY].to_numpy()))
                refs = refs[refs["_key"].isin(shared_keys)]
                bearing_a = initial_bearing(
                    refs["reference_lat"], refs["reference_lon"], a["lat"], a["lon"]
                )
                bearing_b = initial_bearing(
                    refs["reference_lat"], refs["reference_lon"], b["lat"], b["lon"]
                )
                bearing_delta = np.degrees(
                    np.abs(np.arctan2(np.sin(bearing_a-bearing_b), np.cos(bearing_a-bearing_b)))
                )
                median_bearing_separation = float(np.median(bearing_delta))
                if median_bearing_separation < MULTI_MIN_BEARING_SEPARATION_DEG:
                    continue
                b_near = points[
                    points["observed_lat"].sub(b["lat"]).abs().le(DEST_RADIUS_DEG)
                    & points["observed_lon"].sub(b["lon"]).abs().le(DEST_RADIUS_DEG)
                ]
                bm = axis_metrics(b_near, b["lat"], b["lon"])
                if not (bm["intermediate_mass"] >= 0.20 and bm["cross_fraction"] <= 0.20):
                    continue
                event_ids.add(event_id)
                rows.append(
                    {
                        "event_id": event_id,
                        "onset_day": event["onset_day"],
                        "source_lat": event["source_lat"],
                        "source_lon": event["source_lon"],
                        "event_cells": int(event["n_cells"]),
                        "destination_a_lat": a["lat"],
                        "destination_a_lon": a["lon"],
                        "destination_a_cells": a["n_cells"],
                        "destination_a_first_seen": pd.to_datetime(a_near["t_first"]).min(),
                        "destination_a_last_seen": pd.to_datetime(a_near["t_last"]).max(),
                        "destination_a_intermediate_mass": am["intermediate_mass"],
                        "destination_a_cross_fraction": am["cross_fraction"],
                        "destination_b_lat": b["lat"],
                        "destination_b_lon": b["lon"],
                        "destination_b_cells": b["n_cells"],
                        "destination_b_first_seen": pd.to_datetime(b_near["t_first"]).min(),
                        "destination_b_last_seen": pd.to_datetime(b_near["t_last"]).max(),
                        "destination_b_intermediate_mass": bm["intermediate_mass"],
                        "destination_b_cross_fraction": bm["cross_fraction"],
                        "shared_cells": shared,
                        "shared_fraction_of_smaller_cluster": shared_fraction,
                        "median_shared_cell_bearing_separation_deg": median_bearing_separation,
                        "destination_separation_km": separation,
                    }
                )
    columns = [
        "event_id", "onset_day", "source_lat", "source_lon", "event_cells",
        "destination_a_lat", "destination_a_lon", "destination_a_cells",
        "destination_a_first_seen", "destination_a_last_seen",
        "destination_a_intermediate_mass", "destination_a_cross_fraction",
        "destination_b_lat", "destination_b_lon", "destination_b_cells",
        "destination_b_first_seen", "destination_b_last_seen",
        "destination_b_intermediate_mass", "destination_b_cross_fraction",
        "shared_cells", "shared_fraction_of_smaller_cluster",
        "median_shared_cell_bearing_separation_deg", "destination_separation_km",
    ]
    return pd.DataFrame(rows, columns=columns), event_ids


def moving_candidates(events: pd.DataFrame, away: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    event_lookup = events.set_index("event_id")
    event_rows: list[dict] = []
    track_rows: list[dict] = []
    event_ids: set[str] = set()
    for event_id, all_points in away.groupby("event_id", sort=False):
        event = event_lookup.loc[event_id]
        onset = pd.Timestamp(event["onset_day"])
        p = all_points.copy()
        p["first_time"] = pd.to_datetime(p["t_first"], errors="coerce")
        p = p[(p["first_time"] >= onset - pd.Timedelta(days=2)) &
              (p["first_time"] <= onset + pd.Timedelta(days=MULTI_WINDOW_DAYS))]
        if p.empty:
            continue
        p["day"] = p["first_time"].dt.floor("D")
        p["grid_lat"] = (p["observed_lat"] / MULTI_GRID_DEG).round() * MULTI_GRID_DEG
        p["grid_lon"] = (p["observed_lon"] / MULTI_GRID_DEG).round() * MULTI_GRID_DEG

        daily = []
        for day, day_points in p.groupby("day"):
            choices = []
            for (glat, glon), group in day_points.groupby(["grid_lat", "grid_lon"]):
                cells = set(map(tuple, group[KEY].drop_duplicates().to_numpy()))
                choices.append((len(cells), int(group["observations"].sum()), glat, glon, group, cells))
            if not choices:
                continue
            n_cells, obs, _, _, group, cells = max(choices, key=lambda x: (x[0], x[1]))
            if n_cells < MOVING_MIN_CELLS_PER_DAY:
                continue
            weights = group["observations"].to_numpy(float)
            daily.append(
                {
                    "day": day,
                    "lat": float(np.average(group["observed_lat"], weights=weights)),
                    "lon": float(np.average(group["observed_lon"], weights=weights)),
                    "n_cells": n_cells,
                    "observations": obs,
                    "cells": cells,
                }
            )
        if len(daily) < MOVING_MIN_DAYS:
            continue
        daily.sort(key=lambda x: x["day"])
        lat = np.array([d["lat"] for d in daily])
        lon = np.array([d["lon"] for d in daily])
        lat0 = float(np.mean(lat))
        x = np.radians(lon - np.mean(lon)) * np.cos(np.radians(lat0)) * EARTH_KM
        y = np.radians(lat - np.mean(lat)) * EARTH_KM
        xy = np.column_stack([x, y])
        _, singular, vt = np.linalg.svd(xy, full_matrices=False)
        variance = singular ** 2
        linearity = float(variance[0] / variance.sum()) if variance.sum() else 0.0
        projection = xy @ vt[0]
        time_rank = pd.Series(range(len(daily)), dtype=float).rank().to_numpy()
        proj_rank = pd.Series(projection).rank().to_numpy()
        time_correlation = float(abs(np.corrcoef(time_rank, proj_rank)[0, 1]))
        pairwise = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        span_km = float(np.nanmax(pairwise))
        distinct_spatial_bins = len(
            set(zip((lat / MULTI_GRID_DEG).round(), (lon / MULTI_GRID_DEG).round(), strict=True))
        )
        gaps = np.diff(np.array([d["day"].to_datetime64() for d in daily])) / np.timedelta64(1, "D")
        max_gap_days = float(gaps.max()) if len(gaps) else 0.0
        union_cells = set().union(*(d["cells"] for d in daily))
        counts: dict[tuple, int] = {}
        for day in daily:
            for cell in day["cells"]:
                counts[cell] = counts.get(cell, 0) + 1
        repeated_cells = sum(n >= 2 for n in counts.values())
        if not (
            len(union_cells) >= MOVING_MIN_TOTAL_CELLS
            and span_km >= MOVING_MIN_SPAN_KM
            and linearity >= MOVING_MIN_LINEARITY
            and time_correlation >= MOVING_MIN_TIME_CORRELATION
            and repeated_cells >= MOVING_MIN_REPEATED_CELLS
            and distinct_spatial_bins >= MOVING_MIN_DISTINCT_SPATIAL_BINS
            and max_gap_days <= MOVING_MAX_GAP_DAYS
        ):
            continue
        event_ids.add(event_id)
        event_rows.append(
            {
                "event_id": event_id,
                "onset_day": event["onset_day"],
                "source_lat": event["source_lat"],
                "source_lon": event["source_lon"],
                "event_cells": int(event["n_cells"]),
                "track_days": len(daily),
                "track_cells": len(union_cells),
                "cells_repeated_on_track": repeated_cells,
                "distinct_spatial_bins": distinct_spatial_bins,
                "max_gap_days": max_gap_days,
                "track_span_km": span_km,
                "track_linearity": linearity,
                "absolute_time_projection_correlation": time_correlation,
            }
        )
        for rank, day in enumerate(daily, start=1):
            track_rows.append(
                {
                    "event_id": event_id,
                    "track_order": rank,
                    "day": day["day"].date(),
                    "destination_lat": day["lat"],
                    "destination_lon": day["lon"],
                    "cells": day["n_cells"],
                    "observations": day["observations"],
                }
            )
    event_columns = [
        "event_id", "onset_day", "source_lat", "source_lon", "event_cells",
        "track_days", "track_cells", "cells_repeated_on_track", "distinct_spatial_bins",
        "max_gap_days", "track_span_km",
        "track_linearity", "absolute_time_projection_correlation",
    ]
    track_columns = [
        "event_id", "track_order", "day", "destination_lat", "destination_lon",
        "cells", "observations",
    ]
    return (
        pd.DataFrame(event_rows, columns=event_columns),
        pd.DataFrame(track_rows, columns=track_columns),
        event_ids,
    )


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, date_format="%Y-%m-%d %H:%M:%S")
    tmp.replace(path)
    print(f"[write] {path.relative_to(ROOT)} ({len(frame):,} rows)", flush=True)


def subset_members(members: pd.DataFrame, event_ids: set[str]) -> pd.DataFrame:
    return members[members["event_id"].isin(event_ids)].copy()


def validate_evidence_tables(
    events: pd.DataFrame,
    members: pd.DataFrame,
    away: pd.DataFrame,
    axis_points: pd.DataFrame,
) -> None:
    expected_members = int(events["n_cells"].sum())
    member_keys = set(map(tuple, members[["event_id", *KEY]].to_numpy()))
    away_keys = set(map(tuple, away[["event_id", *KEY]].drop_duplicates().to_numpy()))
    if len(events) != 61:
        raise ValueError(f"expected 61 synchronized events, found {len(events)}")
    if len(members) != expected_members or len(member_keys) != expected_members:
        raise ValueError(
            f"event membership mismatch: expected {expected_members}, "
            f"found {len(members)} rows / {len(member_keys)} unique memberships"
        )
    if away_keys != member_keys:
        raise ValueError("away-position identities do not exactly match event memberships")
    unknown_axis_events = set(axis_points["event_id"]) - set(events["event_id"])
    if unknown_axis_events:
        raise ValueError(f"axis points contain unknown events: {sorted(unknown_axis_events)[:3]}")


def analyze(output: Path) -> None:
    events = pd.read_csv(output / "all_synchronized_events.csv", parse_dates=["onset_day"])
    members = pd.read_csv(
        output / "all_event_members.csv",
        parse_dates=[
            "onset_day", "onset_ts", "reference_first_month", "reference_last_month",
            "t_first_away", "t_last_away", "home_first_seen", "home_last_seen",
        ],
    )
    away = pd.read_csv(
        output / "all_event_away_points.csv", parse_dates=["t_first", "t_last"]
    )
    axis_points = pd.read_csv(
        output / "all_event_axis_points.csv", parse_dates=["t_first", "t_last"]
    )
    attractors = pd.read_csv(
        output / "all_broad_source_attractors.csv", parse_dates=["t_start", "t_end"]
    )
    attractor_members = pd.read_csv(
        output / "all_broad_source_attractor_members.csv",
        parse_dates=["first_seen", "last_seen"],
    )

    validate_evidence_tables(events, members, away, axis_points)

    audited = event_geometry(events, members, away, axis_points)
    if audited["event_member_destination_support_fraction"].max() > 1.0 + 1e-12:
        raise ValueError("event-member destination support exceeds 100%")
    write_frame(audited, output / "event_geometry_audit.csv")

    fixed = audited[
        audited["geometry_class"].eq("graded_mixture")
        & audited["operator_or_technology_diversity"]
        & audited["stable_reference_fraction"].ge(0.50)
    ].copy()
    fixed_ids = set(fixed["event_id"])

    identity = audited[
        audited["geometry_class"].eq("coherent_not_mixture")
        & audited["median_baseline_km"].ge(LONG_RANGE_KM)
        & ~(
            audited["destination_lat"].abs().lt(0.10)
            & audited["destination_lon"].abs().lt(0.10)
        )
        & audited["stable_reference_fraction"].ge(0.50)
        & audited["home_active_at_onset_fraction"].ge(0.50)
    ].copy()
    identity_ids = set(identity["event_id"])

    reassignment = audited[
        audited["geometry_class"].eq("coherent_not_mixture")
        & audited["median_baseline_km"].lt(LONG_RANGE_KM)
        & audited["operator_or_technology_diversity"]
        & audited["stable_reference_fraction"].ge(0.50)
    ].copy()
    reassignment_ids = set(reassignment["event_id"])

    moving, moving_track, moving_ids = moving_candidates(audited, away)
    alternating, alternating_ids = alternating_candidates(audited, away)

    bulk = attractors[
        attractors["n_mcc"].ge(3)
        & attractors["cells"].ge(25)
        & attractors["median_displacement_km"].ge(100)
        & attractors["source_spread_km"].ge(1000)
    ].copy()
    bulk_ids = set(bulk["site_id"])

    outputs = {
        "fixed_gnss_decoy_events.csv": fixed,
        "fixed_gnss_decoy_members.csv": subset_members(members, fixed_ids),
        "moving_gnss_decoy_events.csv": moving,
        "moving_gnss_decoy_track_points.csv": moving_track,
        "moving_gnss_decoy_members.csv": subset_members(members, moving_ids),
        "alternating_gnss_decoy_pairs.csv": alternating,
        "alternating_gnss_decoy_members.csv": subset_members(members, alternating_ids),
        "identity_replay_events.csv": identity,
        "identity_replay_members.csv": subset_members(members, identity_ids),
        "bulk_identity_rebroadcast_sites.csv": bulk,
        "bulk_identity_rebroadcast_members.csv": attractor_members[
            attractor_members["site_id"].isin(bulk_ids)
        ].copy(),
        "coordinate_reassignment_events.csv": reassignment,
        "coordinate_reassignment_members.csv": subset_members(members, reassignment_ids),
    }
    for filename, frame in outputs.items():
        write_frame(frame, output / filename)

    manifest = pd.DataFrame(
        [
            ("fixed_gnss_decoy", len(fixed), len(outputs["fixed_gnss_decoy_members.csv"]),
             "graded on-axis mixture; synchronized onset; diverse operator or technology; stable references",
             "strong receiver-location mixture; regional platform blending remains indistinguishable"),
            ("moving_gnss_decoy", len(moving), len(outputs["moving_gnss_decoy_members.csv"]),
             "time-ordered linear destination track from aggregated first-seen times; requires raw-time confirmation",
             "screen only until confirmed from raw observation timestamps"),
            ("alternating_gnss_decoy", len(alternating), len(outputs["alternating_gnss_decoy_members.csv"]),
             "two separated on-axis mixture destinations sharing at least three identities",
             "strong multiple-destination geometry; switching chronology remains interval-censored"),
            ("identity_replay", len(identity), len(outputs["identity_replay_members.csv"]),
             "synchronized endpoint-only displacement >=800 km with stable, concurrently active homes",
             "strong endpoint-clone signature; RF replay and platform reassignment remain distinguishable only with external evidence"),
            ("bulk_identity_rebroadcast", len(bulk), len(outputs["bulk_identity_rebroadcast_members.csv"]),
             "one destination receives >=25 identities from >=3 MCCs spread over >=1000 km",
             "strong rebroadcast-like concentration; benign versus malicious equipment is unknown"),
            ("coordinate_reassignment", len(reassignment), len(outputs["coordinate_reassignment_members.csv"]),
             "synchronized on-axis endpoint jump <800 km across operator or technology boundaries",
             "high confidence in coherent reassignment, not in its cause"),
        ],
        columns=[
            "category", "event_or_site_rows", "member_rows", "operational_definition",
            "interpretation_limit",
        ],
    )
    write_frame(manifest, output / "category_manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--refresh", action="store_true", help="rerun read-only database exports even when cached"
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="reuse existing base exports without SSH"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        for filename, sql in BASE_EXPORTS.items():
            run_csv(filename.removesuffix(".csv"), sql, args.output / filename, refresh=args.refresh)
    missing = [name for name in BASE_EXPORTS if not (args.output / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing base exports: {', '.join(missing)}")
    analyze(args.output)


if __name__ == "__main__":
    main()
