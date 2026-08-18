#!/usr/bin/env python3
"""Export per-identity evidence for the long-range destination case studies.

All database access goes through ``ch_remote``, whose shared ClickHouse
settings enforce read-only queries.  Adjacent hundredth-degree destination
squares are grouped according to the audited site components produced by
``analyze_destination_clusters.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ch_remote import ch_df
from extract_spoofing_categories import haversine_km


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spoofing"

IDENTITY = ["mcc", "mnc", "lac", "cid", "cell_type"]

SITE_META = {
    "2819_11322": ("Changsha", "Pakistan / Afghanistan", "coherent replay"),
    "3976_11639": ("Beijing", "Xinjiang", "coherent replay"),
    "3426_4726": ("Kermanshah", "Mashhad", "coherent replay"),
    "3059_11429": ("Wuhan", "Xinjiang", "coherent replay"),
    "2717_10007": ("Lijiang", "multiple Chinese regions", "mixed-source"),
    "4505_3398": ("Crimea", "multiple Russian regions", "mixed/broad-source"),
    "4581_12649": ("Harbin", "multiple Chinese regions", "broad-source"),
    "3111_11254": ("Jingmen endpoint", "Chifeng endpoint", "reciprocal swap"),
    "4367_11811": ("Chifeng endpoint", "Jingmen endpoint", "reciprocal swap"),
}


def selected_squares(attractors: pd.DataFrame) -> pd.DataFrame:
    selected = attractors[attractors["destination_site_id"].isin(SITE_META)].copy()
    reciprocal = selected["destination_site_id"].isin(["3111_11254", "4367_11811"])
    selected = selected[
        ~reciprocal | selected["evidence_class"].eq("reciprocal coordinate swap")
    ].copy()
    return selected


def query_members(squares: pd.DataFrame) -> pd.DataFrame:
    coordinates = ",".join(
        f"({int(row.plat)},{int(row.plon)})" for row in squares.itertuples(index=False)
    )
    raw = ch_df(
        f"""
        SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
               plat,plon,obs,first_seen,last_seen,
               hlat/100 AS home_lat,hlon/100 AS home_lon,
               total_obs,home_obs,km
        FROM cell.displaced
        WHERE (plat,plon) IN ({coordinates})
        """
    )
    lookup = squares.set_index(["plat", "plon"])["destination_site_id"]
    raw["site_id"] = [
        lookup.loc[(plat, plon)]
        for plat, plon in zip(raw["plat"], raw["plon"], strict=True)
    ]
    raw["first_seen"] = pd.to_datetime(raw["first_seen"])
    raw["last_seen"] = pd.to_datetime(raw["last_seen"])
    return raw


def aggregate_members(raw: pd.DataFrame, squares: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (site_id, *identity), group in raw.groupby(["site_id", *IDENTITY], sort=False):
        destination_rows = squares[squares["destination_site_id"].eq(site_id)]
        representative = destination_rows.sort_values(
            ["identities", "observations"], ascending=False
        ).iloc[0]
        destination_name, source_region, evidence_class = SITE_META[site_id]
        output.append({
            "site_id": site_id,
            "destination_name": destination_name,
            "source_region": source_region,
            "evidence_class": evidence_class,
            **dict(zip(IDENTITY, identity, strict=True)),
            "home_lat": float(group["home_lat"].median()),
            "home_lon": float(group["home_lon"].median()),
            "destination_lat": float(representative["destination_lat"]),
            "destination_lon": float(representative["destination_lon"]),
            "destination_squares": int(group[["plat", "plon"]].drop_duplicates().shape[0]),
            "observations": int(group["obs"].sum()),
            "total_observations": int(group["total_obs"].max()),
            "home_observations": int(group["home_obs"].max()),
            "first_seen": group["first_seen"].min(),
            "last_seen": group["last_seen"].max(),
            "median_displacement_km": float(group["km"].median()),
        })
    return pd.DataFrame(output).sort_values(
        ["evidence_class", "destination_name", "observations"],
        ascending=[True, True, False],
    )


def summarize_sites(members: pd.DataFrame) -> pd.DataFrame:
    output = []
    for site_id, group in members.groupby("site_id", sort=False):
        source_lat = float(group["home_lat"].median())
        source_lon = float(group["home_lon"].median())
        radii = haversine_km(
            group["home_lat"], group["home_lon"], source_lat, source_lon
        )
        output.append({
            "site_id": site_id,
            "destination_name": group["destination_name"].iloc[0],
            "source_region": group["source_region"].iloc[0],
            "evidence_class": group["evidence_class"].iloc[0],
            "destination_lat": group["destination_lat"].iloc[0],
            "destination_lon": group["destination_lon"].iloc[0],
            "source_lat": source_lat,
            "source_lon": source_lon,
            "identities": len(group),
            "observations": int(group["observations"].sum()),
            "median_displacement_km": float(group["median_displacement_km"].median()),
            "source_p90_radius_km": float(np.quantile(radii, 0.90)),
            "first_seen": group["first_seen"].min(),
            "last_seen": group["last_seen"].max(),
        })
    return pd.DataFrame(output).sort_values("identities", ascending=False)


def monthly_onsets(members: pd.DataFrame) -> pd.DataFrame:
    frame = members.copy()
    frame["month"] = frame["first_seen"].dt.to_period("M").dt.to_timestamp()
    return (
        frame.groupby(["site_id", "destination_name", "evidence_class", "month"])
        .agg(first_seen_identities=("cid", "size"), observations=("observations", "sum"))
        .reset_index()
        .sort_values(["site_id", "month"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    attractors = pd.read_csv(DATA / "global_long_range_attractors.csv")
    squares = selected_squares(attractors)
    members = aggregate_members(query_members(squares), squares)
    summary = summarize_sites(members)
    onsets = monthly_onsets(members)

    outputs = {
        "destination_cluster_members.csv": members,
        "destination_cluster_case_summary.csv": summary,
        "destination_cluster_monthly_onsets.csv": onsets,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output / filename, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"{filename}: {len(frame):,} rows")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
