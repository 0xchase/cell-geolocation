#!/usr/bin/env python3
"""Audit and group fixed GNSS-like displacement examples from local CSVs.

Unlike the original population-level screen, this audit measures the graded
home-to-destination corridor using only the cell identities in each synchronized
event.  It never connects to ClickHouse.  Run the database extraction separately
before this script when fresh source data are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extract_spoofing_categories import EARTH_KM, KEY, haversine_km, initial_bearing


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"

MIN_STABLE_REFERENCE_FRACTION = 0.50
MIN_DESTINATION_SUPPORT_FRACTION = 0.50
MIN_CORRIDOR_CELLS = 5
MIN_INTERMEDIATE_CELLS = 5
STRONG_INTERMEDIATE_SHARE = 0.25
SUGGESTIVE_INTERMEDIATE_SHARE = 0.15
MIN_AXIS_FRACTION = 0.15
MAX_AXIS_FRACTION = 1.15
MAX_CROSS_BASELINE_FRACTION = 0.15
INTERMEDIATE_LOW = 0.20
INTERMEDIATE_HIGH = 0.80
ENDPOINT_LOW = 0.80
GROUP_SOURCE_KM = 35.0
GROUP_DESTINATION_KM = 15.0


def corridor_points(
    points: pd.DataFrame, destination_lat: float, destination_lon: float
) -> pd.DataFrame:
    """Return event-member observations close to the home-to-destination axis."""
    if points.empty:
        return points.assign(axis_fraction=np.nan, cross_baseline_fraction=np.nan)
    hlat = points["reference_lat"].to_numpy(float)
    hlon = points["reference_lon"].to_numpy(float)
    plat = points["observed_lat"].to_numpy(float)
    plon = points["observed_lon"].to_numpy(float)
    baseline = haversine_km(hlat, hlon, destination_lat, destination_lon)
    displacement = haversine_km(hlat, hlon, plat, plon)
    destination_bearing = initial_bearing(hlat, hlon, destination_lat, destination_lon)
    point_bearing = initial_bearing(hlat, hlon, plat, plon)
    bearing_delta = np.arctan2(
        np.sin(point_bearing - destination_bearing),
        np.cos(point_bearing - destination_bearing),
    )
    angular_displacement = displacement / EARTH_KM
    cross_km = np.abs(
        np.arcsin(
            np.clip(
                np.sin(angular_displacement) * np.sin(bearing_delta), -1.0, 1.0
            )
        )
        * EARTH_KM
    )
    along_km = (
        np.arctan2(
            np.sin(angular_displacement) * np.cos(bearing_delta),
            np.cos(angular_displacement),
        )
        * EARTH_KM
    )
    out = points.copy()
    out["axis_fraction"] = np.divide(
        along_km,
        baseline,
        out=np.full_like(along_km, np.nan),
        where=baseline > 1e-9,
    )
    out["cross_baseline_fraction"] = np.divide(
        cross_km,
        baseline,
        out=np.full_like(cross_km, np.nan),
        where=baseline > 1e-9,
    )
    return out[
        out["axis_fraction"].between(MIN_AXIS_FRACTION, MAX_AXIS_FRACTION)
        & out["cross_baseline_fraction"].le(MAX_CROSS_BASELINE_FRACTION)
    ]


def corridor_metrics(points: pd.DataFrame, destination_lat: float, destination_lon: float) -> dict:
    corridor = corridor_points(points, destination_lat, destination_lon)
    intermediate = corridor[
        corridor["axis_fraction"].between(
            INTERMEDIATE_LOW, INTERMEDIATE_HIGH, inclusive="neither"
        )
    ]
    endpoint = corridor[
        corridor["axis_fraction"].between(
            ENDPOINT_LOW, MAX_AXIS_FRACTION, inclusive="both"
        )
    ]
    intermediate_obs = int(intermediate["observations"].sum())
    endpoint_obs = int(endpoint["observations"].sum())
    denominator = intermediate_obs + endpoint_obs
    return {
        "member_corridor_cells": int(corridor[KEY].drop_duplicates().shape[0]),
        "member_intermediate_cells": int(intermediate[KEY].drop_duplicates().shape[0]),
        "member_endpoint_cells": int(endpoint[KEY].drop_duplicates().shape[0]),
        "member_intermediate_observations": intermediate_obs,
        "member_endpoint_observations": endpoint_obs,
        "member_intermediate_share": intermediate_obs / denominator if denominator else 0.0,
    }


def evidence_tier(row: pd.Series) -> str:
    diverse = row["n_operators"] >= 2 or row["n_technologies"] >= 2
    core = (
        row["stable_reference_fraction"] >= MIN_STABLE_REFERENCE_FRACTION
        and row["event_member_destination_support_fraction"]
        >= MIN_DESTINATION_SUPPORT_FRACTION
        and row["member_corridor_cells"] >= MIN_CORRIDOR_CELLS
        and row["member_intermediate_cells"] >= MIN_INTERMEDIATE_CELLS
        and row["member_endpoint_cells"] >= MIN_CORRIDOR_CELLS
        and diverse
    )
    if core and row["member_intermediate_share"] >= STRONG_INTERMEDIATE_SHARE:
        return "strong"
    if core and row["member_intermediate_share"] >= SUGGESTIVE_INTERMEDIATE_SHARE:
        return "suggestive"
    return "not_supported"


def audit_events(events: pd.DataFrame, away: pd.DataFrame) -> pd.DataFrame:
    away_groups = {event_id: frame for event_id, frame in away.groupby("event_id")}
    rows = []
    for event in events.itertuples(index=False):
        metrics = corridor_metrics(
            away_groups.get(event.event_id, away.iloc[0:0]),
            event.destination_lat,
            event.destination_lon,
        )
        rows.append({"event_id": event.event_id, **metrics})
    out = events.merge(pd.DataFrame(rows), on="event_id", validate="one_to_one")
    out["evidence_tier"] = out.apply(evidence_tier, axis=1)
    return out


def candidate_components(candidates: pd.DataFrame) -> list[list[int]]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left):
            a, b = candidates.iloc[left], candidates.iloc[right]
            source_km = haversine_km(
                a.source_lat, a.source_lon, b.source_lat, b.source_lon
            )
            destination_km = haversine_km(
                a.destination_lat,
                a.destination_lon,
                b.destination_lat,
                b.destination_lon,
            )
            if source_km <= GROUP_SOURCE_KM and destination_km <= GROUP_DESTINATION_KM:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def group_examples(
    audited: pd.DataFrame,
    members: pd.DataFrame,
    away: pd.DataFrame,
) -> pd.DataFrame:
    candidates = audited[audited["evidence_tier"].ne("not_supported")].reset_index(drop=True)
    rows = []
    for sequence, indices in enumerate(candidate_components(candidates), start=1):
        qualifying = candidates.iloc[indices]
        destination_lat = float(
            np.average(qualifying["destination_lat"], weights=qualifying["n_cells"])
        )
        destination_lon = float(
            np.average(qualifying["destination_lon"], weights=qualifying["n_cells"])
        )
        source_lat = float(np.average(qualifying["source_lat"], weights=qualifying["n_cells"]))
        source_lon = float(np.average(qualifying["source_lon"], weights=qualifying["n_cells"]))
        source_distances = haversine_km(
            audited["source_lat"], audited["source_lon"], source_lat, source_lon
        )
        destination_distances = haversine_km(
            audited["destination_lat"],
            audited["destination_lon"],
            destination_lat,
            destination_lon,
        )
        related = audited[
            (source_distances <= GROUP_SOURCE_KM)
            & (destination_distances <= GROUP_DESTINATION_KM)
        ]
        event_ids = set(qualifying["event_id"])
        related_ids = set(related["event_id"])
        group_members = members[members["event_id"].isin(event_ids)].drop_duplicates(KEY)
        related_members = members[members["event_id"].isin(related_ids)].drop_duplicates(KEY)
        group_points = (
            away[away["event_id"].isin(event_ids)]
            .drop(columns="event_id")
            .drop_duplicates()
        )
        metrics = corridor_metrics(group_points, destination_lat, destination_lon)
        tier = "strong" if (qualifying["evidence_tier"] == "strong").any() else "suggestive"
        rows.append(
            {
                "example_id": f"fixed-displacement-{sequence:02d}",
                "evidence_tier": tier,
                "qualifying_event_bins": len(qualifying),
                "related_screened_event_bins": len(related),
                "first_onset": qualifying["onset_day"].min(),
                "last_onset": qualifying["onset_day"].max(),
                "related_first_onset": related["onset_day"].min(),
                "related_last_onset": related["onset_day"].max(),
                "source_lat": source_lat,
                "source_lon": source_lon,
                "destination_lat": destination_lat,
                "destination_lon": destination_lon,
                "median_displacement_km": float(qualifying["median_baseline_km"].median()),
                "qualifying_unique_cells": len(group_members),
                "related_unique_cells": len(related_members),
                "n_operators": int(group_members[["mcc", "mnc"]].drop_duplicates().shape[0]),
                "technologies": ";".join(sorted(group_members["cell_type"].unique())),
                "minimum_event_intermediate_share": float(
                    qualifying["member_intermediate_share"].min()
                ),
                "maximum_event_intermediate_share": float(
                    qualifying["member_intermediate_share"].max()
                ),
                **metrics,
                "qualifying_event_ids": ";".join(qualifying["event_id"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evidence_tier", "qualifying_unique_cells"],
        ascending=[True, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    events = pd.read_csv(args.data / "event_geometry_audit.csv", parse_dates=["onset_day"])
    members = pd.read_csv(args.data / "all_event_members.csv")
    away = pd.read_csv(args.data / "all_event_away_points.csv")
    audited = audit_events(events, away)
    examples = group_examples(audited, members, away)
    event_output = args.data / "fixed_gnss_like_event_audit.csv"
    example_output = args.data / "fixed_gnss_like_examples.csv"
    audited.to_csv(event_output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    examples.to_csv(example_output, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"{event_output}: {len(audited)} screened event bins")
    print(f"{example_output}: {len(examples)} strong or suggestive examples")


if __name__ == "__main__":
    main()
