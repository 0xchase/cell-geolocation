#!/usr/bin/env python3
"""Derive corrected identity, coverage, similarity, and phenotype tables.

The input snapshots under ``data/satellites/cells`` contain one row for each
``(PLMN, TAC, cell ID, radio type)`` key in ``cell.summary_full``.  For LTE,
the same 28-bit ECI can occur under many TACs.  This script preserves those
intervals, but also emits a corrected ECI-level table so they are not counted
as independent cells.

No database writes are performed; this script reads only local CSV snapshots.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "satellites"
CELLS = DATA / "cells"


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance with wrapped longitude differences."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlat = lat2 - lat1
    dlon_deg = (np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float) + 180.0) % 360.0 - 180.0
    dlon = np.radians(dlon_deg)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def valid_coordinates(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["latest_lat"].between(-90, 90)
        & frame["latest_lon"].between(-180, 180)
        & ~((frame["latest_lat"] == 0) & (frame["latest_lon"] == 0))
    )


def semicolon_values(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna() if str(value)})
    return ";".join(values)


def mode_or_empty(series: pd.Series) -> str:
    series = series.dropna().astype(str)
    series = series[series != ""]
    return "" if series.empty else str(series.value_counts().index[0])


def write_gzip_frame(path: Path, frame: pd.DataFrame, first: bool) -> None:
    mode = "wt" if first else "at"
    with gzip.open(path, mode, encoding="utf-8", newline="", compresslevel=6) as output:
        frame.to_csv(output, index=False, header=first)


def candidate_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(DATA / "plmn_inventory.csv", dtype={"mcc": "Int64", "mnc": "Int64"})
    behavioral = inventory[inventory["selection_channel"].fillna("").str.contains("global_behavioral_screen")].copy()
    controls = inventory[inventory["evidence_tier"].eq("assignment_conflict")].copy()
    analysis = pd.concat([behavioral, controls], ignore_index=True).drop_duplicates("plmn")
    return inventory, analysis


def read_lte(plmn: str) -> pd.DataFrame:
    path = CELLS / f"{plmn}.csv.gz"
    frame = pd.read_csv(path, compression="gzip", low_memory=False)
    frame = frame[(frame["cell_type"] == "lte") & (frame["cid"] > 0)].copy()
    duplicate_source_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    frame.attrs["duplicate_source_rows"] = duplicate_source_rows
    frame["lac"] = frame["lac"].astype("int64")
    frame["cid"] = frame["cid"].astype("int64")
    frame["enodeb_id"] = frame["cid"] // 256
    frame["cell_slot"] = frame["cid"] % 256
    frame["first_seen"] = pd.to_datetime(frame["first_seen"], errors="coerce")
    frame["last_seen"] = pd.to_datetime(frame["last_seen"], errors="coerce")
    return frame


def derive_ecgi_tracks(plmn: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    located = frame.loc[valid_coordinates(frame)].copy()
    grouped = frame.groupby("cid", sort=True)
    tracks = grouped.agg(
        mcc=("mcc", "first"),
        mnc=("mnc", "first"),
        enodeb_id=("enodeb_id", "first"),
        cell_slot=("cell_slot", "first"),
        tac_eci_rows=("lac", "size"),
        n_tacs=("lac", "nunique"),
        observations=("observations", "sum"),
        first_seen=("first_seen", "min"),
        last_seen=("last_seen", "max"),
        position_count=("position_count", "sum"),
    ).reset_index().rename(columns={"cid": "eci"})

    if located.empty:
        tracks["located_tacs"] = 0
        tracks["latest_lat"] = np.nan
        tracks["latest_lon"] = np.nan
        tracks["latest_tac"] = pd.NA
        tracks["lat_min"] = np.nan
        tracks["lat_max"] = np.nan
        tracks["lon_min"] = np.nan
        tracks["lon_max"] = np.nan
        tracks["latest_tac_span_km"] = np.nan
        tracks["country_isos"] = ""
    else:
        location_summary = located.groupby("cid").agg(
            located_tacs=("lac", "nunique"),
            lat_min=("latest_lat", "min"),
            lat_max=("latest_lat", "max"),
            lon_min=("latest_lon", "min"),
            lon_max=("latest_lon", "max"),
        )
        latest = located.sort_values(["cid", "last_seen", "lac"]).drop_duplicates("cid", keep="last")
        latest = latest.set_index("cid")[["latest_lat", "latest_lon", "lac"]].rename(columns={"lac": "latest_tac"})
        countries = located.groupby("cid")["country_iso"].agg(semicolon_values).rename("country_isos")
        location_summary = location_summary.join(latest).join(countries)
        location_summary["latest_tac_span_km"] = haversine_km(
            location_summary["lat_min"],
            location_summary["lon_min"],
            location_summary["lat_max"],
            location_summary["lon_max"],
        )
        tracks = tracks.merge(location_summary.reset_index().rename(columns={"cid": "eci"}), on="eci", how="left")
        tracks["located_tacs"] = tracks["located_tacs"].fillna(0).astype(int)

    tracks.insert(0, "plmn", plmn)
    ordered = [
        "plmn", "mcc", "mnc", "eci", "enodeb_id", "cell_slot",
        "tac_eci_rows", "n_tacs", "located_tacs", "observations",
        "first_seen", "last_seen", "latest_tac", "latest_lat", "latest_lon",
        "lat_min", "lat_max", "lon_min", "lon_max", "latest_tac_span_km",
        "position_count", "country_isos",
    ]
    return tracks[ordered]


def derive_tac_coverage(plmn: str, frame: pd.DataFrame) -> pd.DataFrame:
    located = frame.loc[valid_coordinates(frame)].copy()
    if located.empty:
        return pd.DataFrame()
    coverage = located.groupby("lac", sort=True).agg(
        mcc=("mcc", "first"),
        mnc=("mnc", "first"),
        tac_eci_rows=("cid", "size"),
        unique_ecis=("cid", "nunique"),
        unique_enodebs=("enodeb_id", "nunique"),
        observations=("observations", "sum"),
        first_seen=("first_seen", "min"),
        last_seen=("last_seen", "max"),
        latitude=("latest_lat", "median"),
        longitude=("latest_lon", "median"),
        latitude_min=("latest_lat", "min"),
        latitude_max=("latest_lat", "max"),
        longitude_min=("latest_lon", "min"),
        longitude_max=("latest_lon", "max"),
        country_iso=("country_iso", mode_or_empty),
        country=("country", mode_or_empty),
        region=("region", mode_or_empty),
    ).reset_index().rename(columns={"lac": "tac"})
    coverage["tac_span_km"] = haversine_km(
        coverage["latitude_min"], coverage["longitude_min"],
        coverage["latitude_max"], coverage["longitude_max"],
    )
    coverage.insert(0, "plmn", plmn)
    return coverage


def interval_frame(plmn: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "mcc", "mnc", "lac", "cid", "enodeb_id", "cell_slot",
        "first_seen", "last_seen", "observations", "latest_lat", "latest_lon",
        "first_lat", "first_lon", "lat_min", "lat_max", "lon_min", "lon_max",
        "position_count", "country_iso", "country", "region", "state", "city",
    ]
    output = frame[columns].copy().rename(columns={"lac": "tac", "cid": "eci"})
    output.insert(0, "plmn", plmn)
    return output


def classify_cohort(frame: pd.DataFrame) -> tuple[str, float, str]:
    if frame.empty:
        return "unobserved", 0.0, ""
    highs = set(frame["enodeb_id"].unique())
    core_fraction = sum(11072 <= value <= 11748 for value in highs) / len(highs)
    slots = set(frame["cell_slot"].unique())
    missing = sorted(set(range(256)) - slots)
    last = frame["last_seen"].max()
    if core_fraction >= 0.80 and pd.notna(last) and last.date().isoformat() == "2025-04-23":
        cohort = "early_v1"
    elif core_fraction >= 0.80:
        cohort = "late_v2"
    else:
        cohort = "nonmatching"
    return cohort, core_fraction, ";".join(str(value) for value in missing)


def metric_row(meta: pd.Series, frame: pd.DataFrame, tracks: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    cohort, core_fraction, missing_slots = classify_cohort(frame)
    unique_ecis = int(frame["cid"].nunique()) if not frame.empty else 0
    tac_rows = len(frame)
    linked = meta.get("evidence_tier") == "linked_candidate"
    if cohort in {"early_v1", "late_v2"} and linked:
        attribution = "source_linked_plus_fingerprint"
    elif cohort in {"early_v1", "late_v2"}:
        attribution = "fingerprint_only"
    elif meta.get("evidence_tier") == "assignment_conflict":
        attribution = "assignment_conflict_control"
    else:
        attribution = "nonmatching_screen_result"

    if cohort in {"early_v1", "late_v2"}:
        phenotype = "national_partner_d2c_like"
    elif meta.get("evidence_tier") == "assignment_conflict":
        phenotype = "shared_code_or_test_control"
    else:
        phenotype = "terrestrial_private_or_test_candidate"

    if unique_ecis == 0:
        scale = "unobserved"
    elif unique_ecis < 1_000:
        scale = "very_sparse"
    elif unique_ecis < 10_000:
        scale = "sparse"
    else:
        scale = "national_scale"

    spans = tracks["latest_tac_span_km"].dropna() if not tracks.empty else pd.Series(dtype=float)
    tacs = tracks["n_tacs"] if not tracks.empty else pd.Series(dtype=float)
    return {
        "plmn": meta["plmn"],
        "mcc": meta.get("mcc", ""),
        "mnc": meta.get("mnc", ""),
        "assignee": meta.get("assignee", ""),
        "country_iso": meta.get("country_iso", ""),
        "evidence_tier": meta.get("evidence_tier", ""),
        "attribution_confidence": attribution,
        "network_phenotype": phenotype,
        "implementation_cohort": cohort,
        "deployment_scale": scale,
        "likely_system": meta.get("likely_system", ""),
        "duplicate_source_rows": int(frame.attrs.get("duplicate_source_rows", 0)),
        "lte_tac_eci_rows": tac_rows,
        "unique_lte_ecis": unique_ecis,
        "tac_fragmentation_factor": tac_rows / unique_ecis if unique_ecis else np.nan,
        "unique_enodebs": int(frame["enodeb_id"].nunique()) if not frame.empty else 0,
        "unique_cell_slots": int(frame["cell_slot"].nunique()) if not frame.empty else 0,
        "core_enodeb_fraction": core_fraction,
        "missing_cell_slots": missing_slots,
        "area_codes": int(frame["lac"].nunique()) if not frame.empty else 0,
        "located_area_codes": len(coverage),
        "multi_tac_eci_fraction": float((tacs > 1).mean()) if len(tacs) else np.nan,
        "median_tacs_per_eci": float(tacs.median()) if len(tacs) else np.nan,
        "median_eci_span_km": float(spans.median()) if len(spans) else np.nan,
        "p90_eci_span_km": float(spans.quantile(0.90)) if len(spans) else np.nan,
        "eci_span_over_100km_fraction": float((spans > 100).mean()) if len(spans) else np.nan,
        "median_tac_span_km": float(coverage["tac_span_km"].median()) if len(coverage) else np.nan,
        "first_seen": frame["first_seen"].min() if not frame.empty else pd.NaT,
        "last_seen": frame["last_seen"].max() if not frame.empty else pd.NaT,
    }


def build_similarity(sets: dict[str, dict[str, set[int]]], metadata: pd.DataFrame) -> pd.DataFrame:
    meta = metadata.set_index("plmn")
    rows = []
    plmns = sorted(sets)
    for i, left in enumerate(plmns):
        for right in plmns[i:]:
            row = {"plmn_a": left, "plmn_b": right}
            for key, prefix in [("enodeb", "enodeb"), ("eci", "eci"), ("tac", "tac")]:
                a, b = sets[left][key], sets[right][key]
                intersection = len(a & b)
                union = len(a | b)
                smaller = min(len(a), len(b))
                row[f"{prefix}_intersection"] = intersection
                row[f"{prefix}_jaccard"] = intersection / union if union else np.nan
                row[f"{prefix}_containment"] = intersection / smaller if smaller else np.nan
            row["cohort_a"] = meta.loc[left, "implementation_cohort"]
            row["cohort_b"] = meta.loc[right, "implementation_cohort"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_daily_coverage(phenotypes: pd.DataFrame) -> pd.DataFrame:
    daily = pd.read_csv(DATA / "daily_activity.csv", parse_dates=["date"])
    rename = {}
    if "active_identities" in daily:
        rename["active_identities"] = "active_tac_cell_rows"
    daily = daily.rename(columns=rename)
    if "active_lte_ecis" not in daily:
        daily["active_lte_ecis"] = np.nan
    if "active_lte_enodebs" not in daily:
        daily["active_lte_enodebs"] = np.nan
    daily["observations_per_active_tac_cell"] = daily["observations"] / daily["active_tac_cell_rows"].replace(0, np.nan)

    selected = set(phenotypes["plmn"])
    observed = daily[daily["plmn"].isin(selected)].copy()
    peak_column = "active_lte_ecis" if observed["active_lte_ecis"].notna().any() else "active_tac_cell_rows"
    onsets = []
    for plmn, group in observed.groupby("plmn"):
        peak = float(group[peak_column].max())
        threshold = max(100.0, peak * 0.01)
        qualifying = group[group[peak_column] >= threshold]
        onset = qualifying["date"].min() if not qualifying.empty else pd.NaT
        onsets.append({"plmn": plmn, "bulk_onset": onset, "bulk_threshold": threshold, "peak_daily_ecis": peak})
    onset_frame = pd.DataFrame(onsets)
    observed = observed.merge(onset_frame, on="plmn", how="left")
    observed["relative_daily_ecis"] = observed[peak_column] / observed["peak_daily_ecis"].replace(0, np.nan)

    expanded = []
    for plmn, group in observed.groupby("plmn"):
        calendar = pd.DataFrame({"date": pd.date_range(group["date"].min(), group["date"].max(), freq="D")})
        calendar["plmn"] = plmn
        calendar = calendar.merge(group, on=["plmn", "date"], how="left")
        calendar["observed_day"] = calendar["observations"].notna().astype(int)
        for column in ["mcc", "mnc", "bulk_onset", "bulk_threshold", "peak_daily_ecis"]:
            calendar[column] = calendar[column].ffill().bfill()
        expanded.append(calendar)
    observed = pd.concat(expanded, ignore_index=True)

    matching_plmns = set(
        phenotypes.loc[phenotypes["implementation_cohort"].isin(["early_v1", "late_v2"]), "plmn"]
    )
    observed["eligible_after_bulk_onset"] = (
        observed["plmn"].isin(matching_plmns) & (observed["date"] >= observed["bulk_onset"])
    ).astype(int)
    daily_status = observed.groupby("date").agg(
        eligible_candidate_plmns=("eligible_after_bulk_onset", "sum"),
        observed_candidate_plmns=("observed_day", lambda values: int(values.sum())),
        observed_eligible_plmns=("observed_day", lambda values: 0),
    )
    observed_eligible = observed.loc[observed["eligible_after_bulk_onset"].eq(1)].groupby("date")["observed_day"].sum()
    daily_status["observed_eligible_plmns"] = observed_eligible.reindex(daily_status.index).fillna(0).astype(int)
    daily_status["eligible_observation_fraction"] = (
        daily_status["observed_eligible_plmns"] / daily_status["eligible_candidate_plmns"].replace(0, np.nan)
    )
    daily_status["shared_collection_gap"] = (
        (daily_status["eligible_candidate_plmns"] >= 3)
        & (daily_status["eligible_observation_fraction"] < 0.20)
    ).astype(int)
    observed = observed.merge(daily_status.reset_index(), on="date", how="left")
    return observed.sort_values(["plmn", "date"])


def add_unobserved_phenotypes(inventory: pd.DataFrame, phenotypes: pd.DataFrame) -> pd.DataFrame:
    existing = set(phenotypes["plmn"])
    rows = []
    for _, meta in inventory.iterrows():
        if meta["plmn"] in existing:
            continue
        if meta["evidence_tier"] == "direct_assignment":
            phenotype = "direct_satellite_or_ntn_plmn"
            attribution = "exact_assignment"
        elif meta["figure_group"] == "onboard_control":
            phenotype = "onboard_or_hybrid_control"
            attribution = "source_linked_control"
        else:
            phenotype = "registry_lead"
            attribution = "source_registry"
        rows.append({
            "plmn": meta["plmn"], "mcc": meta.get("mcc", ""), "mnc": meta.get("mnc", ""),
            "assignee": meta.get("assignee", ""), "country_iso": meta.get("country_iso", ""),
            "evidence_tier": meta.get("evidence_tier", ""), "attribution_confidence": attribution,
            "network_phenotype": phenotype, "implementation_cohort": "not_applicable",
            "deployment_scale": "not_analyzed", "likely_system": meta.get("likely_system", ""),
        })
    return pd.concat([phenotypes, pd.DataFrame(rows)], ignore_index=True, sort=False)


def update_manifest() -> None:
    path = DATA / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    files = sorted(item for item in DATA.rglob("*") if item.is_file() and item.name != "manifest.json")
    manifest["identity_granularity"] = {
        "tac_ecgi_intervals": "one row per PLMN/TAC/ECI summary interval",
        "ecgi_tracks": "one row per PLMN/ECI after collapsing TAC reuse",
        "tac_coverage": "one row per PLMN/TAC with a median localized position",
    }
    manifest["files"] = [
        {"path": str(item.relative_to(DATA)), "bytes": item.stat().st_size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in files
    ]
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    inventory, analysis = candidate_inventory()
    intervals_path = DATA / "tac_ecgi_intervals.csv.gz"
    tracks_path = DATA / "ecgi_tracks.csv.gz"
    coverage_path = DATA / "tac_coverage.csv.gz"
    for path in (intervals_path, tracks_path, coverage_path):
        path.unlink(missing_ok=True)

    metric_rows = []
    sets: dict[str, dict[str, set[int]]] = {}
    first_interval = first_track = first_coverage = True
    for _, meta in analysis.iterrows():
        plmn = meta["plmn"]
        print(f"[analyze] {plmn}")
        frame = read_lte(plmn)
        tracks = derive_ecgi_tracks(plmn, frame)
        coverage = derive_tac_coverage(plmn, frame)
        intervals = interval_frame(plmn, frame)
        if not intervals.empty:
            write_gzip_frame(intervals_path, intervals, first_interval)
            first_interval = False
        if not tracks.empty:
            write_gzip_frame(tracks_path, tracks, first_track)
            first_track = False
        if not coverage.empty:
            write_gzip_frame(coverage_path, coverage, first_coverage)
            first_coverage = False
        metric_rows.append(metric_row(meta, frame, tracks, coverage))
        sets[plmn] = {
            "enodeb": set(frame["enodeb_id"].unique()),
            "eci": set(frame["cid"].unique()),
            "tac": set(frame["lac"].unique()),
        }

    phenotypes = pd.DataFrame(metric_rows)
    daily = build_daily_coverage(phenotypes)
    onset = daily.groupby("plmn", as_index=False).agg(
        bulk_onset=("bulk_onset", "first"),
        peak_daily_ecis=("peak_daily_ecis", "first"),
        median_observations_per_active_tac_cell=("observations_per_active_tac_cell", "median"),
        exact_once_day_fraction=("observations_per_active_tac_cell", lambda values: float(np.mean(np.isclose(values.dropna(), 1.0)))),
    )
    phenotypes = phenotypes.merge(onset, on="plmn", how="left")
    similarity = build_similarity(sets, phenotypes)
    all_phenotypes = add_unobserved_phenotypes(inventory, phenotypes)

    all_phenotypes.sort_values(["implementation_cohort", "plmn"]).to_csv(DATA / "plmn_phenotypes.csv", index=False)
    similarity.to_csv(DATA / "identifier_similarity.csv", index=False)
    daily.to_csv(DATA / "daily_coverage.csv", index=False, date_format="%Y-%m-%d")
    update_manifest()
    print(f"[done] {sum(row['lte_tac_eci_rows'] for row in metric_rows):,} TAC/ECI rows")
    print(f"[done] {sum(row['unique_lte_ecis'] for row in metric_rows):,} unique PLMN/ECIs")


if __name__ == "__main__":
    main()
