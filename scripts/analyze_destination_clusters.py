#!/usr/bin/env python3
"""Audit exact long-range destination clusters missed by onset-based screening.

This read-only analysis works from ClickHouse's displaced-cell attractor tables.
It distinguishes coherent identity replay, reciprocal coordinate swaps,
broad-source rebroadcast/test sites, invalid coordinates, and unresolved rows.
The synchronized GNSS-mixture analysis is intentionally kept separate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ch_remote import ch_df
from extract_spoofing_categories import haversine_km


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "spoofing"

MIN_DISTANCE_KM = 800.0
MIN_IDENTITIES = 40
MIN_LEAD_DAYS = 30.0
MAX_COHERENT_SOURCE_SPREAD_KM = 600.0
RECIPROCAL_MATCH_KM = 30.0
DESTINATION_SITE_RADIUS_KM = 15.0


IDENTITY_COLUMNS = ["mcc", "mnc", "lac", "cid", "cell_type"]


ATTRACTORS_SQL = f"""
SELECT
    a.plat AS plat, a.plon AS plon,
    a.plat/100 AS destination_lat, a.plon/100 AS destination_lon,
    a.cells AS identities, a.obs AS observations, a.n_mcc, a.top_mcc,
    a.med_km AS median_displacement_km, a.p90_km AS p90_displacement_km,
    a.src_spread_km AS source_spread_km,
    a.src_lat AS source_lat, a.src_lon AS source_lon,
    a.t_start, a.t_end,
    ifNull(l.lead_days, 0) AS lead_days,
    ifNull(n.anisotropy, 0) AS anisotropy,
    ifNull(c.country_iso, '') AS destination_country
FROM cell.attractors AS a
LEFT JOIN cell.attr_lead AS l USING (plat,plon)
LEFT JOIN cell.attr_aniso AS n USING (plat,plon)
LEFT JOIN cell.coord_a0 AS c ON a.plat=c.klat AND a.plon=c.klon
WHERE a.med_km >= {MIN_DISTANCE_KM} AND a.cells >= 25
ORDER BY a.cells DESC, a.plat, a.plon
"""


def reciprocal_pairs(frame: pd.DataFrame) -> tuple[set[int], pd.DataFrame]:
    indices: set[int] = set()
    rows = []
    for left in range(len(frame)):
        a = frame.iloc[left]
        for right in range(left + 1, len(frame)):
            b = frame.iloc[right]
            source_a_to_b = float(
                haversine_km(a.source_lat, a.source_lon, b.destination_lat, b.destination_lon)
            )
            source_b_to_a = float(
                haversine_km(b.source_lat, b.source_lon, a.destination_lat, a.destination_lon)
            )
            if source_a_to_b <= RECIPROCAL_MATCH_KM and source_b_to_a <= RECIPROCAL_MATCH_KM:
                indices.update((left, right))
                rows.append(
                    {
                        "a_destination_lat": a.destination_lat,
                        "a_destination_lon": a.destination_lon,
                        "a_identities": a.identities,
                        "b_destination_lat": b.destination_lat,
                        "b_destination_lon": b.destination_lon,
                        "b_identities": b.identities,
                        "a_source_to_b_km": source_a_to_b,
                        "b_source_to_a_km": source_b_to_a,
                    }
                )
    return indices, pd.DataFrame(rows)


def add_destination_sites(frame: pd.DataFrame) -> pd.DataFrame:
    """Join adjacent hundredth-degree attractor squares into destination sites."""
    out = frame.copy().reset_index(drop=True)
    parent = list(range(len(out)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(out)):
        a = out.iloc[left]
        for right in range(left + 1, len(out)):
            b = out.iloc[right]
            distance = float(
                haversine_km(
                    a.destination_lat, a.destination_lon,
                    b.destination_lat, b.destination_lon,
                )
            )
            if distance <= DESTINATION_SITE_RADIUS_KM:
                union(left, right)

    roots = [root(index) for index in range(len(out))]
    representatives = {}
    for component in sorted(set(roots)):
        members = out.iloc[[i for i, value in enumerate(roots) if value == component]]
        representative = members.sort_values(
            ["identities", "observations"], ascending=False
        ).iloc[0]
        representatives[component] = f"{int(representative.plat)}_{int(representative.plon)}"
    out["destination_site_id"] = [representatives[value] for value in roots]
    return out


def classify(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = add_destination_sites(frame)
    reciprocal, pairs = reciprocal_pairs(out)
    labels = []
    for index, row in out.iterrows():
        if abs(row.destination_lat) < 0.10 and abs(row.destination_lon) < 0.10:
            label = "invalid coordinate"
        elif index in reciprocal:
            label = "reciprocal coordinate swap"
        elif row.identities < MIN_IDENTITIES:
            label = "lower-count review"
        elif row.lead_days < MIN_LEAD_DAYS:
            label = "crawl-onset / unresolved"
        elif row.source_spread_km > MAX_COHERENT_SOURCE_SPREAD_KM:
            label = "broad-source rebroadcast/test candidate"
        else:
            label = "coherent identity-replay candidate"
        labels.append(label)
    out["evidence_class"] = labels
    # Do not split a few apparently coherent squares out of a nearby, much
    # larger broad-source endpoint. This specifically prevents the Crimea site
    # from being presented as a clean replay campaign merely because some cell
    # subsets land in adjacent hundredth-degree squares.
    for _, indices in out.groupby("destination_site_id").groups.items():
        component_labels = set(out.loc[indices, "evidence_class"])
        if "broad-source rebroadcast/test candidate" in component_labels:
            coherent = out.loc[indices, "evidence_class"].eq(
                "coherent identity-replay candidate"
            )
            out.loc[np.asarray(indices)[coherent.to_numpy()], "evidence_class"] = (
                "mixed/broad-source destination site"
            )
    return out, pairs


def destination_name(latitude: float, longitude: float) -> tuple[str, str]:
    """Human-readable labels for the conservative coherent site set."""
    known = [
        (-12.04, -77.05, "Lima", "Ukraine"),
        (28.19, 113.22, "Changsha", "Pakistan / Afghanistan"),
        (27.17, 100.07, "Lijiang", "Guangdong"),
        (39.76, 116.39, "Beijing", "Xinjiang"),
        (34.26, 47.26, "Kermanshah", "Mashhad"),
        (30.59, 114.29, "Wuhan", "Xinjiang"),
    ]
    for target_lat, target_lon, destination, source in known:
        if float(haversine_km(latitude, longitude, target_lat, target_lon)) < 20:
            return destination, source
    return f"{latitude:.2f}, {longitude:.2f}", "source region"


def identity_replay_site_audit(candidates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate adjacent candidate squares, then re-test source coherence."""
    selected = candidates[candidates["evidence_class"].eq(
        "coherent identity-replay candidate"
    )]
    coordinates = ",".join(
        f"({int(row.plat)},{int(row.plon)})" for row in selected.itertuples(index=False)
    )
    if not coordinates:
        return pd.DataFrame()
    rows = ch_df(
        f"""
        SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
               plat,plon,obs,first_seen,last_seen,hlat/100 AS home_lat,
               hlon/100 AS home_lon,km
        FROM cell.displaced
        WHERE (plat,plon) IN ({coordinates})
        """
    )
    site_lookup = selected.set_index(["plat", "plon"])["destination_site_id"]
    rows["destination_site_id"] = [
        site_lookup.loc[(plat, plon)]
        for plat, plon in zip(rows["plat"], rows["plon"], strict=True)
    ]

    output = []
    for site_id, group in rows.groupby("destination_site_id"):
        site_squares = selected[selected["destination_site_id"].eq(site_id)]
        representative = site_squares.sort_values(
            ["identities", "observations"], ascending=False
        ).iloc[0]
        unique = group.drop_duplicates(IDENTITY_COLUMNS)
        source_lat = float(unique["home_lat"].median())
        source_lon = float(unique["home_lon"].median())
        radii = haversine_km(
            unique["home_lat"], unique["home_lon"], source_lat, source_lon
        )
        per_mcc = unique.groupby("mcc").size().sort_values(ascending=False)
        target_name, source_name = destination_name(
            representative.destination_lat, representative.destination_lon
        )
        output.append({
            "site_id": site_id,
            "destination_name": target_name,
            "source_region": source_name,
            "destination_lat": representative.destination_lat,
            "destination_lon": representative.destination_lon,
            "source_lat": source_lat,
            "source_lon": source_lon,
            "identities": len(unique),
            "observations": int(group["obs"].sum()),
            "destination_squares": len(site_squares),
            "n_mcc": int(unique["mcc"].nunique()),
            "top_mcc": ";".join(str(int(value)) for value in per_mcc.head(4).index),
            "median_displacement_km": float(unique["km"].median()),
            "source_p90_radius_km": float(np.quantile(radii, 0.90)),
            "first_seen": group["first_seen"].min(),
            "last_seen": group["last_seen"].max(),
            "minimum_lead_days": float(site_squares["lead_days"].min()),
            "site_evidence_class": (
                "coherent identity-replay candidate"
                if float(np.quantile(radii, 0.90)) <= MAX_COHERENT_SOURCE_SPREAD_KM
                else "mixed/broad-source destination site"
            ),
        })
    return pd.DataFrame(output).sort_values("identities", ascending=False)


def mcc_breakdown(candidates: pd.DataFrame) -> pd.DataFrame:
    selected = candidates[
        candidates["evidence_class"].isin(
            [
                "coherent identity-replay candidate",
                "reciprocal coordinate swap",
                "broad-source rebroadcast/test candidate",
            ]
        )
        & candidates["identities"].ge(MIN_IDENTITIES)
    ]
    coordinates = ",".join(
        f"({int(row.plat)},{int(row.plon)})" for row in selected.itertuples(index=False)
    )
    if not coordinates:
        return pd.DataFrame()
    return ch_df(
        f"""
        SELECT
            d.plat/100 AS destination_lat, d.plon/100 AS destination_lon, d.mcc,
            uniqExact((d.mnc,d.lac,d.cid,d.cell_type)) AS identities,
            sum(d.obs) AS displaced_observations,
            sum(d.total_obs) AS all_observations,
            min(d.first_seen) AS first_seen,
            max(d.last_seen) AS last_seen,
            avg(d.hlat)/100 AS source_lat,
            avg(d.hlon)/100 AS source_lon,
            quantileExact(0.5)(d.km) AS median_displacement_km
        FROM cell.displaced AS d
        WHERE (d.plat,d.plon) IN ({coordinates})
        GROUP BY d.plat,d.plon,d.mcc
        ORDER BY destination_lat,destination_lon,identities DESC
        """
    )


def lima_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    locations = ch_df(
        """
        WITH lima AS
        (
            SELECT DISTINCT mnc,lac,cid,cell_type
            FROM cell.geos
            WHERE mcc=255 AND cid>0
              AND lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11
        )
        SELECT
            multiIf(
                lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11, 'Lima replay',
                lon BETWEEN 22 AND 41 AND lat BETWEEN 44 AND 53, 'Ukraine home',
                'Elsewhere'
            ) AS location,
            count() AS observations,
            uniqExact((mnc,lac,cid,cell_type)) AS identities,
            min(timestamp) AS first_seen,
            max(timestamp) AS last_seen
        FROM cell.geos
        WHERE mcc=255 AND cid>0 AND (mnc,lac,cid,cell_type) IN lima
        GROUP BY location
        ORDER BY observations DESC
        """
    )
    timeline = ch_df(
        """
        WITH lima AS
        (
            SELECT DISTINCT mnc,lac,cid,cell_type
            FROM cell.geos
            WHERE mcc=255 AND cid>0
              AND lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11
        )
        SELECT toStartOfMonth(timestamp) AS month,
               count() AS observations,
               uniqExact((mnc,lac,cid,cell_type)) AS identities
        FROM cell.geos
        WHERE mcc=255 AND cid>0 AND (mnc,lac,cid,cell_type) IN lima
          AND lon BETWEEN -78 AND -76 AND lat BETWEEN -13 AND -11
        GROUP BY month
        ORDER BY month
        """
    )
    return locations, timeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    attractors, pairs = classify(ch_df(ATTRACTORS_SQL))
    site_audit = identity_replay_site_audit(attractors)
    broad_site_ids = set(site_audit.loc[
        site_audit["site_evidence_class"].eq("mixed/broad-source destination site"),
        "site_id",
    ])
    attractors.loc[
        attractors["destination_site_id"].isin(broad_site_ids)
        & attractors["evidence_class"].eq("coherent identity-replay candidate"),
        "evidence_class",
    ] = "mixed/broad-source destination site"
    replay_sites = site_audit[site_audit["site_evidence_class"].eq(
        "coherent identity-replay candidate"
    )].copy()
    breakdown = mcc_breakdown(attractors)
    lima_locations, lima_timeline = lima_evidence()

    outputs = {
        "global_long_range_attractors.csv": attractors,
        "reciprocal_destination_pairs.csv": pairs,
        "global_long_range_cluster_mccs.csv": breakdown,
        "destination_site_audit.csv": site_audit,
        "coherent_identity_replay_sites.csv": replay_sites,
        "lima_replay_locations.csv": lima_locations,
        "lima_replay_timeline.csv": lima_timeline,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output / filename, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"{filename}: {len(frame):,} rows")

    print("\nEvidence classes")
    print(attractors["evidence_class"].value_counts().to_string())
    interesting = attractors[
        ~attractors["evidence_class"].isin(
            ["lower-count review", "invalid coordinate", "crawl-onset / unresolved"]
        )
    ]
    print("\nHigh-interest exact destination squares")
    print(
        interesting[
            [
                "destination_lat", "destination_lon", "identities", "observations",
                "top_mcc", "median_displacement_km", "source_spread_km",
                "lead_days", "evidence_class",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
