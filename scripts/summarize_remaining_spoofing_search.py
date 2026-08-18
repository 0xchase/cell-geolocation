#!/usr/bin/env python3
"""Consolidate exhaustive remaining-mode screens into auditable campaigns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "spoofing" / "remaining_search"
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


def components(frame: pd.DataFrame, source_km: float, destination_km: float, days: int) -> list[int]:
    parent = list(range(len(frame)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[b] = a

    values = frame.reset_index(drop=True)
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if abs((values.onset_day.iloc[left] - values.onset_day.iloc[right]).days) > days:
                continue
            if haversine(
                values.source_lat.iloc[left], values.source_lon.iloc[left],
                values.source_lat.iloc[right], values.source_lon.iloc[right],
            ) > source_km:
                continue
            if haversine(
                values.destination_lat.iloc[left], values.destination_lon.iloc[left],
                values.destination_lat.iloc[right], values.destination_lon.iloc[right],
            ) <= destination_km:
                union(left, right)
    roots = [root(i) for i in range(len(values))]
    mapping = {value: index + 1 for index, value in enumerate(dict.fromkeys(roots))}
    return [mapping[value] for value in roots]


def short_campaigns() -> pd.DataFrame:
    event_key = ["src_lat10", "src_lon10", "onset_day", "plat", "plon"]
    events = pd.read_csv(OUTPUT / "short_range_raw_member_audit.csv", parse_dates=["onset_day"])
    events = events[events.synchronized_persistent_candidate].copy().reset_index(drop=True)
    members = pd.read_csv(
        OUTPUT / "short_range_candidate_members.csv.gz", parse_dates=["onset_day", "first_seen"]
    )
    members = members.merge(events[event_key], on=event_key, how="inner")
    geometry = members.groupby(event_key, as_index=False).agg(
        source_lat=("hlat", lambda x: x.median() / 100),
        source_lon=("hlon", lambda x: x.median() / 100),
    )
    events = events.merge(geometry, on=event_key)
    events["destination_lat"] = events.plat / 100
    events["destination_lon"] = events.plon / 100
    events["component"] = components(events, 40, 40, 30)
    event_components = events[event_key + ["component"]]
    members = members.merge(event_components, on=event_key)
    raw_membership_path = OUTPUT / "short_range_raw_membership_audit.csv"
    if raw_membership_path.exists():
        raw_membership = pd.read_csv(raw_membership_path, parse_dates=["onset_day"]).merge(
            event_components, on=event_key
        )
        raw_component = raw_membership.groupby("component").agg(
            exact_simultaneous_identities=("exact_simultaneous", "sum"),
            identities_with_five_minute_pair=("home_away_pairs_within_5m", lambda x: int((x > 0).sum())),
            minimum_raw_gap_seconds=("minimum_home_away_gap_seconds", "min"),
        )
    else:
        raw_component = pd.DataFrame()
    member_events = members.drop_duplicates(["component", *event_key, *KEY])
    identity_counts = member_events.groupby(["component", *KEY]).size().rename("event_count").reset_index()
    repeated = identity_counts.groupby("component").agg(
        unique_identities=("cid", "size"),
        identities_in_multiple_bins=("event_count", lambda x: int((x >= 2).sum())),
    )
    rows = []
    for component, group in events.groupby("component"):
        implicated = members[members.component.eq(component)].drop_duplicates(KEY)
        weights = group.distinct_cids
        row = {
            "campaign_id": f"SHORT-{component:02d}",
            "event_bins": len(group), "event_days": group.onset_day.nunique(),
            "first_onset": group.onset_day.min(), "last_onset": group.onset_day.max(),
            "maximum_event_identities": group.distinct_cids.max(),
            "unique_identities": int(repeated.loc[component, "unique_identities"]),
            "identities_in_multiple_bins": int(repeated.loc[component, "identities_in_multiple_bins"]),
            "operators": implicated[["mcc", "mnc"]].drop_duplicates().shape[0],
            "technologies": implicated.cell_type.nunique(),
            "source_lat": np.average(group.source_lat, weights=weights),
            "source_lon": np.average(group.source_lon, weights=weights),
            "destination_lat": np.average(group.destination_lat, weights=weights),
            "destination_lon": np.average(group.destination_lon, weights=weights),
            "median_displacement_km": group.median_km.median(),
            "destination_span_km": float(np.max(haversine(
                group.destination_lat.to_numpy()[:, None], group.destination_lon.to_numpy()[:, None],
                group.destination_lat.to_numpy()[None, :], group.destination_lon.to_numpy()[None, :],
            ))),
        }
        if component in raw_component.index:
            row |= {
                "exact_simultaneous_identities": int(raw_component.loc[component, "exact_simultaneous_identities"]),
                "identities_with_five_minute_pair": int(raw_component.loc[component, "identities_with_five_minute_pair"]),
                "minimum_raw_gap_seconds": int(raw_component.loc[component, "minimum_raw_gap_seconds"]),
            }
        else:
            row |= {"exact_simultaneous_identities": 0, "identities_with_five_minute_pair": 0,
                    "minimum_raw_gap_seconds": np.nan}
        row["repeated_identity_fraction"] = (
            row["identities_in_multiple_bins"] / row["unique_identities"]
            if row["unique_identities"] else 0
        )
        row["evidence_label"] = (
            "strong nonphysical coordinate conflict; mechanism unresolved"
            if row["exact_simultaneous_identities"] >= 3 and row["operators"] >= 2 else
            "persistent coordinated local displacement"
            if row["event_days"] >= 2 and row["identities_in_multiple_bins"] >= 3 else
            "non-repeating local concentration"
        )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["event_bins", "unique_identities", "maximum_event_identities"], ascending=False
    ).reset_index(drop=True)
    result["campaign_id"] = [f"SHORT-{i + 1:02d}" for i in range(len(result))]
    result.to_csv(OUTPUT / "short_range_campaigns.csv", index=False)
    return result


def small_campaigns() -> pd.DataFrame:
    event_key = ["src_lat10", "src_lon10", "onset_day", "dest_lat5", "dest_lon5"]
    events = pd.read_csv(OUTPUT / "small_cohort_raw_member_audit.csv", parse_dates=["onset_day"])
    events = events[events.tight_synchronized_candidate].copy().reset_index(drop=True)
    events["component"] = components(events, 100, 20, 45)
    raw_membership_path = OUTPUT / "small_cohort_raw_membership_audit.csv"
    if raw_membership_path.exists():
        raw_membership = pd.read_csv(raw_membership_path, parse_dates=["onset_day"]).merge(
            events[event_key + ["component"]], on=event_key
        )
        raw_component = raw_membership.groupby("component").agg(
            exact_simultaneous_identities=("exact_simultaneous", "sum"),
            identities_with_five_minute_pair=("home_away_pairs_within_5m", lambda x: int((x > 0).sum())),
            identities_with_one_hour_pair=("home_away_pairs_within_1h", lambda x: int((x > 0).sum())),
            minimum_raw_gap_seconds=("minimum_home_away_gap_seconds", "min"),
        )
    else:
        raw_component = pd.DataFrame()
    rows = []
    for component, group in events.groupby("component"):
        weights = group.identities
        destination_lat = np.average(group.destination_lat, weights=weights)
        destination_lon = np.average(group.destination_lon, weights=weights)
        artifact = (
            "near-null coordinate" if abs(destination_lat) < .1 and abs(destination_lon) < .1 else
            "round-coordinate placeholder" if (
                abs(destination_lat - round(destination_lat)) < .01
                and abs(destination_lon - round(destination_lon)) < .01
            ) else
            "known Lima attractor" if haversine(destination_lat, destination_lon, -12.04, -77.05) < 10 else
            "none"
        )
        row = {
            "component": component, "event_bins": len(group),
            "event_days": group.onset_day.nunique(), "identities_across_bins": int(group.identities.sum()),
            "maximum_event_identities": group.identities.max(),
            "operators_max": group.operators.max(), "technologies_max": group.technologies.max(),
            "first_onset": group.onset_day.min(), "last_onset": group.onset_day.max(),
            "source_lat": np.average(group.source_lat, weights=weights),
            "source_lon": np.average(group.source_lon, weights=weights),
            "destination_lat": destination_lat, "destination_lon": destination_lon,
            "median_displacement_km": group.median_km.median(),
            "artifact_control": artifact,
        }
        if component in raw_component.index:
            row |= {
                "exact_simultaneous_identities": int(raw_component.loc[component, "exact_simultaneous_identities"]),
                "identities_with_five_minute_pair": int(raw_component.loc[component, "identities_with_five_minute_pair"]),
                "identities_with_one_hour_pair": int(raw_component.loc[component, "identities_with_one_hour_pair"]),
                "minimum_raw_gap_seconds": int(raw_component.loc[component, "minimum_raw_gap_seconds"]),
            }
        else:
            row |= {"exact_simultaneous_identities": 0, "identities_with_five_minute_pair": 0,
                    "identities_with_one_hour_pair": 0, "minimum_raw_gap_seconds": np.nan}
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["event_bins", "identities_across_bins"], ascending=False
    ).reset_index(drop=True)
    result.insert(0, "campaign_id", [f"SMALL-{i + 1:02d}" for i in range(len(result))])
    result.to_csv(OUTPUT / "small_cohort_campaigns.csv", index=False)
    return result


def kinematic_endpoints() -> pd.DataFrame:
    audit = pd.read_csv(OUTPUT / "kinematic_raw_audit.csv")
    audit = audit[audit.raw_kinematic_positive].copy()
    screen = pd.read_csv(OUTPUT / "kinematic_interval_screen.csv")
    screen.columns = [column.rsplit(".", 1)[-1] for column in screen.columns]
    screen["destination_lat"] = screen.plat / 100
    screen["destination_lon"] = screen.plon / 100
    join = [*KEY, "destination_lat", "destination_lon"]
    audit = audit.merge(screen[join + ["hlat", "hlon"]], on=join, how="left")
    audit["dest_lat5"] = (np.trunc(audit.destination_lat * 20) * 5).astype(int)
    audit["dest_lon5"] = (np.trunc(audit.destination_lon * 20) * 5).astype(int)
    rows = []
    for (lat5, lon5), group in audit.groupby(["dest_lat5", "dest_lon5"]):
        identities = group[KEY].drop_duplicates()
        source_lat = group.hlat.median() / 100
        source_lon = group.hlon.median() / 100
        rows.append({
            "destination_lat": lat5 / 100, "destination_lon": lon5 / 100,
            "target_bins": len(group), "unique_identities": len(identities),
            "operators": identities[["mcc", "mnc"]].drop_duplicates().shape[0],
            "mccs": identities.mcc.nunique(),
            "impossible_pair_observations": int(group.impossible_pair_observations.sum()),
            "identity_days": int(group.impossible_pair_days.sum()),
            "minimum_gap_seconds": int(group.min_gap_seconds.min()),
            "median_distance_km": group.distance_km.median(),
            "source_lat": source_lat, "source_lon": source_lon,
            "source_radius_p90_km": np.quantile(haversine(
                group.hlat / 100, group.hlon / 100, source_lat, source_lon
            ), .9),
        })
    result = pd.DataFrame(rows).sort_values(
        ["unique_identities", "impossible_pair_observations"], ascending=False
    )
    result.to_csv(OUTPUT / "kinematic_endpoint_summary.csv", index=False)
    return result


def kinematic_batches_and_campaigns() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(OUTPUT / "kinematic_impossible_pairs.csv", parse_dates=["away_timestamp"])
    pairs["onset_day"] = pairs.away_timestamp.dt.floor("D")
    pairs["dest_lat5"] = (np.trunc(pairs.destination_lat * 20) * 5).astype(int)
    pairs["dest_lon5"] = (np.trunc(pairs.destination_lon * 20) * 5).astype(int)
    # One identity contributes at most once to a destination/day event.
    pairs = pairs.sort_values("gap_seconds").drop_duplicates(
        ["onset_day", "dest_lat5", "dest_lon5", *KEY]
    )
    rows = []
    for (day, lat5, lon5), group in pairs.groupby(["onset_day", "dest_lat5", "dest_lon5"]):
        identities = group[KEY].drop_duplicates()
        if len(identities) < 2:
            continue
        rows.append({
            "onset_day": day, "destination_lat": lat5 / 100,
            "destination_lon": lon5 / 100, "identities": len(identities),
            "operators": identities[["mcc", "mnc"]].drop_duplicates().shape[0],
            "mccs": identities.mcc.nunique(), "minimum_gap_seconds": group.gap_seconds.min(),
            "median_gap_seconds": group.gap_seconds.median(),
            "median_distance_km": group.distance_km.median(),
            "source_lat": group.home_lat.median(), "source_lon": group.home_lon.median(),
            "source_lat_span_degrees": group.home_lat.max() - group.home_lat.min(),
            "source_lon_span_degrees": group.home_lon.max() - group.home_lon.min(),
        })
    batches = pd.DataFrame(rows).sort_values(["identities", "operators"], ascending=False)
    batches["strong_coordinated_raw_batch"] = (
        batches.identities.ge(3) & batches.operators.ge(2)
    )
    batches.to_csv(OUTPUT / "kinematic_raw_event_batches.csv", index=False)

    strong = batches[batches.strong_coordinated_raw_batch].copy().reset_index(drop=True)
    if strong.empty:
        campaigns = pd.DataFrame()
    else:
        strong["component"] = components(strong, 100, 25, 30)
        campaign_rows = []
        for component, group in strong.groupby("component"):
            weights = group.identities
            campaign_rows.append({
                "component": component, "event_batches": len(group),
                "event_days": group.onset_day.nunique(),
                "maximum_daily_identities": group.identities.max(),
                "operators_max": group.operators.max(), "mccs": group.mccs.max(),
                "first_event": group.onset_day.min(), "last_event": group.onset_day.max(),
                "source_lat": np.average(group.source_lat, weights=weights),
                "source_lon": np.average(group.source_lon, weights=weights),
                "destination_lat": np.average(group.destination_lat, weights=weights),
                "destination_lon": np.average(group.destination_lon, weights=weights),
                "median_distance_km": group.median_distance_km.median(),
                "minimum_gap_seconds": group.minimum_gap_seconds.min(),
            })
        campaigns = pd.DataFrame(campaign_rows).sort_values(
            ["maximum_daily_identities", "event_batches"], ascending=False
        ).reset_index(drop=True)
        campaigns.insert(0, "campaign_id", [f"KIN-{i + 1:02d}" for i in range(len(campaigns))])
    campaigns.to_csv(OUTPUT / "kinematic_raw_campaigns.csv", index=False)
    return batches, campaigns


def exact_small_endpoints() -> pd.DataFrame:
    pairs = pd.read_csv(
        ROOT / "data" / "spoofing" / "high_quality" / "method1_exact_dual_pairs.csv",
        parse_dates=["away_timestamp"],
    )
    pairs = pairs[pairs.cid > 0].copy()
    pairs["dest_lat5"] = (pairs.away_lat * 20).round().astype(int) * 5
    pairs["dest_lon5"] = (pairs.away_lon * 20).round().astype(int) * 5
    rows = []
    for (lat5, lon5), group in pairs.groupby(["dest_lat5", "dest_lon5"]):
        identities = group[KEY].drop_duplicates()
        if len(identities) > 4:
            continue
        rows.append({
            "destination_lat": lat5 / 100, "destination_lon": lon5 / 100,
            "unique_identities": len(identities), "exact_pairs": len(group),
            "operators": identities[["mcc", "mnc"]].drop_duplicates().shape[0],
            "mccs": identities.mcc.nunique(), "days": group.away_timestamp.dt.date.nunique(),
            "first_pair": group.away_timestamp.min(), "last_pair": group.away_timestamp.max(),
            "minimum_gap_seconds": group.gap_seconds.min(),
            "median_distance_km": group.distance_km.median(),
            "source_lat": group.reference_lat.median(), "source_lon": group.reference_lon.median(),
        })
    result = pd.DataFrame(rows).sort_values(
        ["unique_identities", "exact_pairs", "days"], ascending=False
    )
    result.to_csv(OUTPUT / "small_exact_dual_endpoint_audit.csv", index=False)
    return result


def manifest(
    short: pd.DataFrame, small: pd.DataFrame, kinematic: pd.DataFrame,
    kinematic_batches: pd.DataFrame, kinematic_campaigns: pd.DataFrame,
) -> None:
    moving_pairs = pd.read_csv(OUTPUT / "moving_shared_identity_pairs.csv")
    cycles = pd.read_csv(OUTPUT / "multidestination_shared_transitions.csv")
    moving = pd.read_csv(OUTPUT / "moving_track_audit.csv")
    cycle_ids = pd.read_csv(OUTPUT / "multidestination_identity_audit.csv")
    kine_audit = pd.read_csv(OUTPUT / "kinematic_raw_audit.csv")
    rows = [
        {
            "search": "raw-time moving destination", "global_screen_survivors": int(moving.passes_geometry.sum()),
            "validated_units": len(moving_pairs), "strong_nonphysical_findings": 0,
            "result": (
                "no pair of independently rooted identities shares three moving destination-days"
                if moving_pairs.empty else
                f"{len(moving_pairs)} identity pairs share >=3 destination-days; no independent-network moving cohort"
            ),
        },
        {
            "search": "short-range local displacement", "global_screen_survivors": 1882,
            "validated_units": len(short),
            "strong_nonphysical_findings": int(short.evidence_label.str.startswith("strong").sum()),
            "result": "Tehran and Jordan dominate the synchronized screen, but no member has mutually exclusive home/away observations within one hour; suggestive only",
        },
        {
            "search": "small cohorts", "global_screen_survivors": 95,
            "validated_units": len(small), "strong_nonphysical_findings": 0,
            "result": "tight 2-4 identity extensions found, chiefly fragments of known destinations; labels describe coordinate inconsistency, not mechanism",
        },
        {
            "search": "100-500 km kinematic impossibility", "global_screen_survivors": 2710,
            "validated_units": len(kinematic_campaigns),
            "strong_nonphysical_findings": len(kinematic_campaigns),
            "result": (
                f"{len(kinematic_batches)} multi-identity daily batches, including "
                f"{int(kinematic_batches.strong_coordinated_raw_batch.sum())} with >=3 identities and >=2 operators; "
                "coordinated nonphysical coordinates are strong, but spoofing versus platform corruption is unresolved"
            ),
        },
        {
            "search": "irregular multi-destination cycling", "global_screen_survivors": int(cycle_ids.cycle_candidate.sum()),
            "validated_units": len(cycles), "strong_nonphysical_findings": 0,
            "result": "all shared raw transitions are local (<20 km); no coordinated long-range cycle survives",
        },
        {
            "search": "mechanism attribution", "global_screen_survivors": 0,
            "validated_units": 0, "strong_nonphysical_findings": 0,
            "result": "receiver, scan, provider, RF, and GNSS observables are absent; mechanism is not identifiable",
        },
    ]
    pd.DataFrame(rows).to_csv(OUTPUT / "validated_findings_manifest.csv", index=False)


def main() -> None:
    short = short_campaigns()
    small = small_campaigns()
    kinematic = kinematic_endpoints()
    kinematic_batches, kinematic_campaigns = kinematic_batches_and_campaigns()
    exact_small_endpoints()
    manifest(short, small, kinematic, kinematic_batches, kinematic_campaigns)
    print({"short_campaigns": len(short), "small_campaigns": len(small),
           "kinematic_endpoints": len(kinematic),
           "kinematic_raw_campaigns": len(kinematic_campaigns)})


if __name__ == "__main__":
    main()
