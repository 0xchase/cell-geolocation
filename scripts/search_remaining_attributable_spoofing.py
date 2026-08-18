#!/usr/bin/env python3
"""Audit the remaining spoofing hypotheses with an attribution-first screen.

The expensive global search has already been materialized in the CSVs produced
by ``search_remaining_spoofing_modes.py`` and ``extract_spoofing_categories.py``.
This script applies stricter campaign-level tests to those complete screens and
fetches raw observations only for the 27 identities that survive the alternating
attractor screen.  Remote access goes through ``ch_remote.py``, which enforces
ClickHouse ``readonly=2``.

Outputs are written under ``data/spoofing/attribution_search``.  A positive in
``findings_manifest.csv`` requires more than anomalous coordinates: either an
independently reported false destination plus multi-network source-to-target
geometry, or two independently reconstructed attractors shared by a
multi-network cohort.  Database drift, mobile infrastructure, and provenance-
free identity duplication are retained as candidates but not called spoofing.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"
REMAINING = DATA / "remaining_search"
NEWS = DATA / "news_validation"
OUTPUT = DATA / "attribution_search"
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]
EARTH_KM = 6371.0088


def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))


def identity_tuples(frame: pd.DataFrame) -> str:
    return ",".join(
        f"({int(r.mcc)},{int(r.mnc)},{int(r.lac)},{int(r.cid)},'{r.cell_type}')"
        for r in frame[KEY].drop_duplicates().itertuples(index=False)
    )


def refresh_alternating_raw() -> pd.DataFrame:
    members = pd.read_csv(DATA / "alternating_gnss_decoy_members.csv")
    identities = members[KEY].drop_duplicates()
    tuples = identity_tuples(identities)
    raw = ch_df(f"""
      SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
             timestamp,lat,lon
      FROM cell.geos
      PREWHERE (mcc,mnc,lac,cid,cell_type) IN ({tuples})
      WHERE lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
      ORDER BY mcc,mnc,lac,cid,cell_type,timestamp
    """, settings={"max_threads": 4, "max_execution_time": 3600})
    raw.to_csv(OUTPUT / "alternating_attractor_raw.csv.gz", index=False)
    return raw


def candidate_peaks(frame: pd.DataFrame, limit: int = 100) -> list[tuple[float, float]]:
    work = frame.copy()
    work["grid_lat"] = (work.plat / 10).round().astype(int) * 10
    work["grid_lon"] = (work.plon / 10).round().astype(int) * 10
    peaks = (
        work.groupby(["grid_lat", "grid_lon"])
        .agg(identities=("identity", "nunique"))
        .reset_index()
        .query("identities >= 8")
        .sort_values("identities", ascending=False)
    )
    selected: list[tuple[float, float]] = []
    for row in peaks.itertuples(index=False):
        lat, lon = row.grid_lat / 100, row.grid_lon / 100
        if selected:
            distance = haversine(
                lat, lon,
                np.array([point[0] for point in selected]),
                np.array([point[1] for point in selected]),
            )
            if float(np.min(distance)) < 30:
                continue
        selected.append((lat, lon))
        if len(selected) >= limit:
            break
    return selected


def trajectory_metrics(group: pd.DataFrame, target_lat: float, target_lon: float) -> dict | None:
    source_lat = float(group.rlat.iloc[0]) / 100
    source_lon = float(group.rlon.iloc[0]) / 100
    y_scale = 111.32
    x_scale = y_scale * math.cos(math.radians((source_lat + target_lat) / 2))
    dx = (target_lon - source_lon) * x_scale
    dy = (target_lat - source_lat) * y_scale
    baseline = math.hypot(dx, dy)
    if baseline < 25:
        return None
    px = (group.lon - source_lon) * x_scale
    py = (group.lat - source_lat) * y_scale
    fraction = (px * dx + py * dy) / baseline**2
    cross = np.abs(px * dy - py * dx) / baseline
    keep = (
        fraction.between(0.03, 1.25)
        & (cross <= max(1.5, 0.06 * baseline))
    )
    points = group[keep].copy()
    fraction = fraction[keep]
    cross = cross[keep]
    if len(points) < 3:
        return None
    along = fraction * baseline
    correlation = points.t_first.rank().corr(along.rank())
    if (
        float(along.max() - along.min()) < 8
        or correlation < 0.60
        or float(fraction.max()) < 0.65
    ):
        return None
    return {
        "positions": len(points),
        "along_span_km": float(along.max() - along.min()),
        "spearman_time_along": float(correlation),
        "median_cross_fraction": float(np.median(cross) / baseline),
        "source_lat": source_lat,
        "source_lon": source_lon,
        "baseline_km": baseline,
        "first_position": points.t_first.min(),
        "last_position": points.t_first.max(),
    }


def search_slow_attractors() -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = pd.read_csv(
        REMAINING / "moving_position_screen.csv",
        parse_dates=["t_first", "t_last"],
    )
    positions["identity"] = positions[KEY].astype(str).agg("/".join, axis=1)
    positions["lat"] = positions.plat / 100
    positions["lon"] = positions.plon / 100
    members = []
    for center_lat, center_lon in candidate_peaks(positions):
        near = positions[
            haversine(positions.lat, positions.lon, center_lat, center_lon) <= 15
        ]
        if near.empty:
            continue
        popular = near.groupby(["plat", "plon"]).identity.nunique().idxmax()
        target_lat, target_lon = popular[0] / 100, popular[1] / 100
        for identity, group in positions[positions.identity.isin(near.identity)].groupby("identity"):
            metrics = trajectory_metrics(group, target_lat, target_lon)
            if metrics is None:
                continue
            identity_row = group.iloc[0]
            members.append({
                "target_lat": target_lat,
                "target_lon": target_lon,
                "identity": identity,
                **{column: identity_row[column] for column in KEY},
                **metrics,
            })
    member_frame = pd.DataFrame(members).drop_duplicates(
        ["target_lat", "target_lon", "identity"]
    )
    member_frame.to_csv(OUTPUT / "slow_attractor_members.csv", index=False)

    campaign_rows = []
    for (target_lat, target_lon), group in member_frame.groupby(["target_lat", "target_lon"]):
        if len(group) < 3:
            continue
        campaign_rows.append({
            "target_lat": target_lat,
            "target_lon": target_lon,
            "identities": len(group),
            "plmns": group[["mcc", "mnc"]].drop_duplicates().shape[0],
            "mccs": group.mcc.nunique(),
            "technologies": group.cell_type.nunique(),
            "first_position": group.first_position.min(),
            "last_position": group.last_position.max(),
            "source_lat": group.source_lat.median(),
            "source_lon": group.source_lon.median(),
            "source_spread_km": float(haversine(
                group.source_lat.min(), group.source_lon.min(),
                group.source_lat.max(), group.source_lon.max(),
            )),
            "median_baseline_km": group.baseline_km.median(),
            "median_cross_fraction": group.median_cross_fraction.median(),
            "median_time_along_correlation": group.spearman_time_along.median(),
        })
    campaigns = pd.DataFrame(campaign_rows)
    campaigns["source_file"] = "remaining_search/moving_position_screen.csv"

    # The Moscow CPS audit is a denser raw daily extract than spoof.away and is
    # therefore not rediscovered by the >=3 rounded-away-bin screen above.
    # Include it in the same campaign census rather than silently omitting a
    # validated slow-attractor case from this output.
    moscow = pd.read_csv(NEWS / "moscow_cps_trajectory.csv", parse_dates=["day"])
    moscow_ids = moscow[KEY + ["source_lat", "source_lon"]].drop_duplicates(KEY)
    moscow_baselines = haversine(
        moscow_ids.source_lat, moscow_ids.source_lon, 55.972, 37.415
    )
    latest = moscow.sort_values("day").groupby(KEY).tail(1)
    moscow_row = pd.DataFrame([{
        "target_lat": 55.972,
        "target_lon": 37.415,
        "identities": moscow_ids.shape[0],
        "plmns": moscow[["mcc", "mnc"]].drop_duplicates().shape[0],
        "mccs": moscow.mcc.nunique(),
        "technologies": moscow.cell_type.nunique(),
        "first_position": moscow.day.min(),
        "last_position": moscow.day.max(),
        "source_lat": moscow_ids.source_lat.median(),
        "source_lon": moscow_ids.source_lon.median(),
        "source_spread_km": float(haversine(
            moscow_ids.source_lat.min(), moscow_ids.source_lon.min(),
            moscow_ids.source_lat.max(), moscow_ids.source_lon.max(),
        )),
        "median_baseline_km": float(np.median(moscow_baselines)),
        "median_cross_fraction": float(
            latest.cross_track_km.abs().median() / np.median(moscow_baselines)
        ),
        "median_time_along_correlation": np.nan,
        "source_file": "news_validation/moscow_cps_trajectory.csv",
    }])
    campaigns = pd.concat([campaigns, moscow_row], ignore_index=True).sort_values(
        "identities", ascending=False
    )

    known = pd.read_csv(NEWS / "known_gnss_events.csv", parse_dates=["screen_start", "screen_end"])
    known = known[known.known_dest_lat.notna()].copy()
    campaigns["corroborated_event"] = ""
    campaigns["corroboration_distance_km"] = np.nan
    for index, row in campaigns.iterrows():
        distances = haversine(
            row.target_lat, row.target_lon, known.known_dest_lat, known.known_dest_lon
        )
        overlaps = (
            (pd.Timestamp(row.first_position) <= known.screen_end + pd.Timedelta(days=30))
            & (pd.Timestamp(row.last_position) >= known.screen_start - pd.Timedelta(days=30))
        )
        eligible = np.flatnonzero(overlaps.to_numpy())
        if not len(eligible):
            continue
        nearest = int(eligible[np.argmin(distances[eligible])])
        event = known.iloc[nearest]
        if distances[nearest] <= 25:
            campaigns.at[index, "corroborated_event"] = event.event_id
            campaigns.at[index, "corroboration_distance_km"] = distances[nearest]
    campaigns["attribution"] = np.where(
        campaigns.corroborated_event.ne("")
        & campaigns.identities.ge(5)
        & campaigns.plmns.ge(2),
        "reasonably confident GNSS-derived location poisoning",
        "unresolved geometric candidate",
    )
    campaigns.to_csv(OUTPUT / "slow_attractor_campaigns.csv", index=False)
    return campaigns, member_frame


def search_moving_footprints_and_partitions() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(DATA / "fixed_gnss_like_event_audit.csv", parse_dates=["onset_day"])
    events = events[events.evidence_tier.isin(["strong", "suggestive"])].copy()
    events["target_lat_bin"] = (events.destination_lat * 5).round() / 5
    events["target_lon_bin"] = (events.destination_lon * 5).round() / 5

    footprint_rows = []
    for target, group in events.groupby(["target_lat_bin", "target_lon_bin"]):
        if group.onset_day.nunique() < 3:
            continue
        ordered = group.sort_values("onset_day")
        movement = float(haversine(
            ordered.source_lat.iloc[0], ordered.source_lon.iloc[0],
            ordered.source_lat.iloc[-1], ordered.source_lon.iloc[-1],
        ))
        spread = float(np.max(haversine(
            ordered.source_lat.to_numpy()[:, None], ordered.source_lon.to_numpy()[:, None],
            ordered.source_lat.to_numpy()[None, :], ordered.source_lon.to_numpy()[None, :],
        )))
        if spread >= 20 and movement >= 15:
            footprint_rows.append({
                "target_lat": target[0], "target_lon": target[1],
                "events": len(group), "event_days": group.onset_day.nunique(),
                "first_day": group.onset_day.min(), "last_day": group.onset_day.max(),
                "source_spread_km": spread, "first_to_last_source_km": movement,
                "identities": group.n_cells.sum(),
                "event_ids": "|".join(ordered.event_id),
            })
    footprints = pd.DataFrame(footprint_rows, columns=[
        "target_lat", "target_lon", "events", "event_days", "first_day", "last_day",
        "source_spread_km", "first_to_last_source_km", "identities", "event_ids",
    ])
    footprints.to_csv(OUTPUT / "moving_footprint_candidates.csv", index=False)

    partition_rows = []
    for day, group in events.groupby("onset_day"):
        rows = list(group.itertuples(index=False))
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1:]:
                source_distance = float(haversine(
                    left.source_lat, left.source_lon, right.source_lat, right.source_lon
                ))
                destination_distance = float(haversine(
                    left.destination_lat, left.destination_lon,
                    right.destination_lat, right.destination_lon,
                ))
                if source_distance <= 50 and destination_distance >= 15:
                    partition_rows.append({
                        "onset_day": day,
                        "event_a": left.event_id, "event_b": right.event_id,
                        "source_separation_km": source_distance,
                        "destination_separation_km": destination_distance,
                        "identities": left.n_cells + right.n_cells,
                        "evidence_a": left.evidence_tier,
                        "evidence_b": right.evidence_tier,
                    })
    partitions = pd.DataFrame(partition_rows, columns=[
        "onset_day", "event_a", "event_b", "source_separation_km",
        "destination_separation_km", "identities", "evidence_a", "evidence_b",
    ])
    partitions.to_csv(OUTPUT / "spatial_partition_candidates.csv", index=False)
    return footprints, partitions


def fit_common_attractor(points: pd.DataFrame, homes: pd.DataFrame) -> dict:
    latitude = float(points.lat.median())
    x_scale = 111.32 * math.cos(math.radians(latitude))
    y_scale = 111.32
    work = points.join(homes, on=KEY, rsuffix="_home").copy()
    work["x"] = work.lon * x_scale
    work["y"] = work.lat * y_scale
    work["x_home"] = work.lon_home * x_scale
    work["y_home"] = work.lat_home * y_scale
    work["distance"] = np.hypot(work.x - work.x_home, work.y - work.y_home)
    farthest = work.sort_values("distance").groupby(KEY).tail(1)
    matrix = np.zeros((2, 2))
    vector = np.zeros(2)
    for row in farthest.itertuples(index=False):
        home = np.array([row.x_home, row.y_home])
        point = np.array([row.x, row.y])
        direction = point - home
        direction /= np.linalg.norm(direction)
        perpendicular = np.eye(2) - np.outer(direction, direction)
        matrix += perpendicular
        vector += perpendicular @ home
    target = np.linalg.solve(matrix, vector)
    cross_track = []
    for row in farthest.itertuples(index=False):
        home = np.array([row.x_home, row.y_home])
        point = np.array([row.x, row.y])
        target_vector = target - home
        point_vector = point - home
        cross_track.append(abs(
            target_vector[0] * point_vector[1] - target_vector[1] * point_vector[0]
        ) / np.linalg.norm(target_vector))
    return {
        "identities": len(farthest),
        "target_lat": target[1] / y_scale,
        "target_lon": target[0] / x_scale,
        "median_cross_track_km": float(np.median(cross_track)),
        "p90_cross_track_km": float(np.percentile(cross_track, 90)),
        "median_observed_displacement_km": float(farthest.distance.median()),
    }


def search_alternating_attractors(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.timestamp = pd.to_datetime(raw.timestamp)
    members = pd.read_csv(DATA / "alternating_gnss_decoy_members.csv")
    identities = members[KEY].drop_duplicates()
    raw = raw.merge(identities, on=KEY, how="inner")
    homes = (
        raw[raw.timestamp < pd.Timestamp("2025-02-12")]
        .groupby(KEY)[["lat", "lon"]]
        .median()
    )
    phases = [
        ("west", pd.Timestamp("2025-02-14"), pd.Timestamp("2025-02-25")),
        ("east", pd.Timestamp("2025-03-20"), pd.Timestamp("2025-06-21")),
    ]
    rows = []
    for phase, start, end in phases:
        points = raw[raw.timestamp.between(start, end, inclusive="left")]
        metrics = fit_common_attractor(points, homes)
        rows.append({"phase": phase, "start": start, "end": end, **metrics})
    result = pd.DataFrame(rows)
    result["plmns"] = identities[["mcc", "mnc"]].drop_duplicates().shape[0]
    result["technologies"] = identities.cell_type.nunique()
    separation = float(haversine(
        result.target_lat.iloc[0], result.target_lon.iloc[0],
        result.target_lat.iloc[1], result.target_lon.iloc[1],
    ))
    result["attractor_separation_km"] = separation
    result["attribution"] = (
        "reasonably confident GNSS-derived location poisoning; "
        "RF spoofing versus fabricated collector fixes is not identifiable"
    )
    result.to_csv(OUTPUT / "alternating_attractor_campaign.csv", index=False)
    return result


def audit_identity_masquerade() -> pd.DataFrame:
    sites = pd.read_csv(DATA / "bulk_identity_rebroadcast_sites.csv")
    exact_hour = pd.read_csv(DATA / "concurrent_exact_hour.csv")
    sites["exact_hour_home_away_identities"] = np.where(
        sites.site_id.eq("3733_12203"),
        int(exact_hour.loc[exact_hour.site.eq("Weihai"), "simultaneous_identities"].iloc[0]),
        0,
    )
    sites["attribution"] = (
        "not attributable: identity duplication is equally compatible with "
        "provider replay, aggregation contamination, or benign emulation"
    )
    sites.to_csv(OUTPUT / "identity_masquerade_candidates.csv", index=False)
    return sites


def moscow_summary() -> dict:
    trajectory = pd.read_csv(NEWS / "moscow_cps_trajectory.csv", parse_dates=["day"])
    latest = trajectory.sort_values("day").groupby(KEY).tail(1)
    return {
        "identities": trajectory[KEY].drop_duplicates().shape[0],
        "plmns": trajectory[["mcc", "mnc"]].drop_duplicates().shape[0],
        "technologies": trajectory.cell_type.nunique(),
        "median_cross_track_m": float(latest.cross_track_km.abs().median() * 1000),
        "within_100m_of_axis_fraction": float((latest.cross_track_km.abs() <= 0.1).mean()),
        "median_final_along_fraction": float(latest.along_fraction.median()),
        "first_day": trajectory.day.min(),
        "last_day": trajectory.day.max(),
    }


def write_manifest(
    slow: pd.DataFrame,
    footprints: pd.DataFrame,
    partitions: pd.DataFrame,
    alternating: pd.DataFrame,
    masquerade: pd.DataFrame,
) -> pd.DataFrame:
    queen = slow[slow.corroborated_event.eq("queen_alia_sep2024")]
    moscow = moscow_summary()
    rows = [
        {
            "search": "slow unsynchronized poisoning toward a fixed attractor",
            "validated_campaigns": 2,
            "positive_results": (
                f"Queen Alia: {int(queen.identities.max())} identities across "
                f"{int(queen.plmns.max())} PLMNs; Moscow/Sheremetyevo: "
                f"{moscow['identities']} identities across {moscow['plmns']} PLMNs"
            ),
            "attribution": "reasonably confident",
            "qualification": "CPS timestamps measure stored database state, not RF duration",
        },
        {
            "search": "moving spoofing footprint with fixed false destination",
            "validated_campaigns": len(footprints),
            "positive_results": "none",
            "attribution": "no positive",
            "qualification": "no supported target had a multi-day moving source footprint",
        },
        {
            "search": "spatially partitioned false destinations",
            "validated_campaigns": len(partitions),
            "positive_results": "none",
            "attribution": "no positive",
            "qualification": "supported neighboring bins converged on one attractor",
        },
        {
            "search": "alternating attractors",
            "validated_campaigns": 1,
            "positive_results": (
                f"Vladimir: {int(alternating.identities.max())} identities across "
                f"{int(alternating.plmns.max())} PLMNs; two inferred attractors "
                f"{alternating.attractor_separation_km.iloc[0]:.1f} km apart"
            ),
            "attribution": "reasonably confident",
            "qualification": "GNSS RF spoofing versus fabricated collector fixes is not identifiable",
        },
        {
            "search": "cellular identity masquerading or rebroadcast",
            "validated_campaigns": 0,
            "positive_results": (
                f"{len(masquerade)} unresolved bulk-duplication site; not counted as spoofing"
            ),
            "attribution": "not attributable",
            "qualification": "schema lacks receiver, scan, provider, and RF provenance",
        },
    ]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTPUT / "findings_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-raw", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT / "alternating_attractor_raw.csv.gz"
    raw = refresh_alternating_raw() if args.refresh_raw or not raw_path.exists() else pd.read_csv(raw_path)
    slow, _ = search_slow_attractors()
    footprints, partitions = search_moving_footprints_and_partitions()
    alternating = search_alternating_attractors(raw)
    masquerade = audit_identity_masquerade()
    manifest = write_manifest(slow, footprints, partitions, alternating, masquerade)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
