#!/usr/bin/env python3
"""Build reproducible global cellular-anomaly lead tables.

The outputs are investigative leads, never classifications of criminal
activity.  Remote access is limited to read-only aggregate queries through
``ch_remote.py``; expensive global raw-history validation is performed only in
later follow-up for candidates surviving these screens.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity"
OCEAN = ROOT / "data" / "oceans" / "ocean_cell_positions.csv"
MOVING = ROOT / "data" / "moving-mccs" / "identities.csv.zst"
SPECIAL = ROOT / "data" / "test-mccs"
SITES = DATA / "scam_site_inventory.csv"
AMNESTY_TEXT = ROOT / ".cache" / "criminal-activity" / "amnesty_cambodia_2025.txt"
KEY = ["mcc", "mnc", "lac", "cid", "cell_type"]
EARTH_KM = 6371.0088

# This audit deliberately avoids duplicating the separate cartel search.
AMERICAS = {
    "AR", "BO", "BR", "BZ", "CL", "CO", "CR", "EC", "GF", "GT",
    "GY", "HN", "MX", "NI", "PA", "PE", "PY", "SR", "SV", "UY", "VE",
}


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    q = math.sin((lat2 - lat1) / 2) ** 2
    q += math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(q))


def clean_coordinate(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return lat.between(-85, 85) & lon.between(-180, 180) & ~(
        lat.abs().le(0.01) & lon.abs().le(0.01)
    )


def ocean_screens() -> None:
    rows = pd.read_csv(OCEAN)
    rows = rows[clean_coordinate(rows.lat, rows.lon) & rows.cid.ge(0)].copy()
    rows["key"] = list(map(tuple, rows[KEY].itertuples(index=False, name=None)))

    identity_rows: list[dict] = []
    for key, group in rows.groupby("key", sort=False):
        points = list(dict.fromkeys(zip(group.lat, group.lon)))
        max_span = 0.0
        for a, b in itertools.combinations(points, 2):
            max_span = max(max_span, haversine(a, b))
        identity_rows.append(
            {
                **dict(zip(KEY, key)),
                "offshore_position_bins": len(points),
                "offshore_observations": int(group.observations.sum()),
                "max_offshore_span_km": max_span,
                "minimum_distance_to_land_km": float(group.distance_to_land_km.min()),
                "maximum_distance_to_land_km": float(group.distance_to_land_km.max()),
                "first_seen": group.first_seen.min(),
                "last_seen": group.last_seen.max(),
            }
        )
    identities = pd.DataFrame(identity_rows)
    identities = identities.sort_values(
        ["max_offshore_span_km", "offshore_observations"], ascending=False
    )
    identities.to_csv(DATA / "ocean_identity_candidates.csv", index=False)

    at_coordinate: dict[tuple[float, float], list[tuple]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        at_coordinate[(row.lat, row.lon)].append(row.key)
    shared: dict[tuple[tuple, tuple], list[tuple[float, float]]] = defaultdict(list)
    for coordinate, keys in at_coordinate.items():
        unique = sorted(set(keys))
        # Exact-coordinate attractors with huge mixtures are handled by the
        # spoofing screen and would make this pair expansion quadratic.
        if len(unique) > 50:
            continue
        for pair in itertools.combinations(unique, 2):
            shared[pair].append(coordinate)

    pair_rows: list[dict] = []
    for (left, right), points in shared.items():
        if len(points) < 2:
            continue
        span = max(haversine(a, b) for a, b in itertools.combinations(points, 2))
        if span < 10:
            continue
        pair_rows.append(
            {
                **{f"left_{name}": value for name, value in zip(KEY, left)},
                **{f"right_{name}": value for name, value in zip(KEY, right)},
                "shared_position_bins": len(points),
                "shared_span_km": span,
                "same_plmn": int(left[:2] == right[:2]),
                "same_lac": int(left[:3] == right[:3]),
                "first_shared_lat": points[0][0],
                "first_shared_lon": points[0][1],
                "last_shared_lat": points[-1][0],
                "last_shared_lon": points[-1][1],
            }
        )
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["shared_span_km", "shared_position_bins"], ascending=False
        )
    pairs.to_csv(DATA / "ocean_comovement_pairs.csv", index=False)


def canonical_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    a = list(zip(frame.endpoint_a_lat.round(2), frame.endpoint_a_lon.round(2)))
    b = list(zip(frame.endpoint_b_lat.round(2), frame.endpoint_b_lon.round(2)))
    lo = [min(x, y) for x, y in zip(a, b)]
    hi = [max(x, y) for x, y in zip(a, b)]
    frame = frame.copy()
    frame["endpoint_1_lat"] = [x[0] for x in lo]
    frame["endpoint_1_lon"] = [x[1] for x in lo]
    frame["endpoint_2_lat"] = [x[0] for x in hi]
    frame["endpoint_2_lon"] = [x[1] for x in hi]
    return frame


def moving_bundle_screen() -> None:
    selected: list[pd.DataFrame] = []
    columns = None
    for chunk in pd.read_csv(MOVING, compression="zstd", chunksize=150_000):
        if columns is None:
            columns = list(chunk.columns)
        keep = chunk[
            chunk.max_span_km.ge(50)
            & chunk.total_observations.ge(3)
            & chunk.position_rows.le(500)
            & clean_coordinate(chunk.endpoint_a_lat, chunk.endpoint_a_lon)
            & clean_coordinate(chunk.endpoint_b_lat, chunk.endpoint_b_lon)
            & ~(
                chunk.endpoint_a_country_iso.isin(AMERICAS)
                | chunk.endpoint_b_country_iso.isin(AMERICAS)
            )
        ]
        selected.append(keep)
    candidates = canonical_endpoints(pd.concat(selected, ignore_index=True))
    group_columns = [
        "endpoint_1_lat", "endpoint_1_lon", "endpoint_2_lat", "endpoint_2_lon"
    ]
    grouped = (
        candidates.groupby(group_columns, dropna=False)
        .agg(
            identities=("cid", "size"),
            distinct_cids=("cid", "nunique"),
            operators=("mcc", lambda x: len(set(zip(x, candidates.loc[x.index, "mnc"])))),
            mccs=("mcc", "nunique"),
            technologies=("cell_type", lambda x: ";".join(sorted(set(x)))),
            median_span_km=("max_span_km", "median"),
            total_observations=("total_observations", "sum"),
            median_home_fraction=("home_fraction", "median"),
            first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"),
            example_mcc=("mcc", "first"),
            example_mnc=("mnc", "first"),
            example_lac=("lac", "first"),
            example_cid=("cid", "first"),
        )
        .reset_index()
    )
    grouped = grouped[
        grouped.identities.ge(2) | grouped.median_span_km.ge(250)
    ].copy()
    grouped["artifact_flag"] = np.where(
        (
            grouped.endpoint_1_lat.mod(1).abs().lt(1e-9)
            & grouped.endpoint_1_lon.mod(1).abs().lt(1e-9)
        )
        | (
            grouped.endpoint_2_lat.mod(1).abs().lt(1e-9)
            & grouped.endpoint_2_lon.mod(1).abs().lt(1e-9)
        ),
        "round-coordinate candidate",
        "",
    )
    grouped = grouped.sort_values(
        ["identities", "total_observations", "median_span_km"], ascending=False
    )
    grouped.to_csv(DATA / "moving_bundle_candidates.csv", index=False)


def special_mcc_screen() -> None:
    frames = []
    for name in ("private", "testing", "unassigned"):
        frame = pd.read_csv(SPECIAL / f"{name}.csv")
        frame["source_family"] = name
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[clean_coordinate(rows.glat, rows.glon) & rows.cid.ge(0)].copy()
    rows["lat_bin"] = rows.glat.round(2)
    rows["lon_bin"] = rows.glon.round(2)
    rows["lifespan_days"] = (
        pd.to_datetime(rows.last_seen) - pd.to_datetime(rows.first_seen)
    ).dt.total_seconds() / 86400
    grouped = (
        rows.groupby(
            ["source_family", "category", "subcategory", "lat_bin", "lon_bin",
             "country_iso", "country", "region", "city"],
            dropna=False,
        )
        .agg(
            identities=("cid", "size"),
            operators=("mnc", "nunique"),
            observations=("obs", "sum"),
            median_lifespan_days=("lifespan_days", "median"),
            multi_position_identities=("n_pos", lambda x: int((x > 1).sum())),
            first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"),
        )
        .reset_index()
    )
    grouped = grouped[
        grouped.identities.ge(2) | grouped.observations.ge(500)
    ].sort_values(["identities", "observations"], ascending=False)
    grouped.to_csv(DATA / "special_mcc_clusters.csv", index=False)


def scam_site_query() -> None:
    sites = pd.read_csv(SITES)
    tuples = ",\n".join(
        "tuple(%r,toFloat64(%s),toFloat64(%s),toFloat64(%s),%r)"
        % (r.site_id, r.latitude, r.longitude, r.radius_km, r.role)
        for r in sites.itertuples(index=False)
    )
    sql = f"""
WITH [{tuples}] AS sites
SELECT
    site.1 AS site_id,
    site.5 AS site_role,
    mcc,mnc,toString(cell_type) AS cell_type,
    count() AS identities,
    countIf(n_pos > 1) AS moving_identities,
    sum(obs) AS crawler_observations,
    min(first_seen) AS first_seen,
    max(last_seen) AS last_seen,
    countIf(cell.summary_full.first_seen >= toDateTime('2025-01-01')) AS first_seen_2025_or_later,
    countIf(mcc IN (1,991,999) OR mcc < 200 OR mcc BETWEEN 800 AND 900
            OR mcc BETWEEN 902 AND 999) AS special_mcc_identities
FROM cell.summary_full
ARRAY JOIN sites AS site
WHERE glat BETWEEN -85 AND 85 AND glon BETWEEN -180 AND 180
  AND greatCircleDistance(glon,glat,site.3,site.2) <= site.4 * 1000
GROUP BY site_id,site_role,mcc,mnc,cell_type
ORDER BY site_id,identities DESC
"""
    ch_df(sql).to_csv(DATA / "scam_site_cell_summary.csv", index=False)

    monthly_sql = f"""
WITH [{tuples}] AS sites
SELECT
    site.1 AS site_id, site.5 AS site_role,
    toStartOfMonth(cell.summary_full.first_seen) AS first_seen_month,
    count() AS identities,
    countIf(mcc IN (1,991,999) OR mcc < 200 OR mcc BETWEEN 800 AND 900
            OR mcc BETWEEN 902 AND 999) AS special_mcc_identities,
    countIf(mcc NOT IN (414,415,416,417,420,452,456,457,460,510,514,520))
        AS other_mcc_identities
FROM cell.summary_full
ARRAY JOIN sites AS site
WHERE glat BETWEEN -85 AND 85 AND glon BETWEEN -180 AND 180
  AND greatCircleDistance(glon,glat,site.3,site.2) <= site.4 * 1000
GROUP BY site_id,site_role,first_seen_month
ORDER BY site_id,first_seen_month
"""
    ch_df(monthly_sql).to_csv(DATA / "scam_site_first_seen_monthly.csv", index=False)


def amnesty_compounds() -> pd.DataFrame:
    """Extract the 53 verified Cambodia sites from Amnesty's report text.

    Two entries put the coordinates before the LOCATION label, so coordinates
    are read from the whole first 900 characters of each site section.
    """
    if not AMNESTY_TEXT.exists():
        raise FileNotFoundError(
            f"Missing {AMNESTY_TEXT}; convert Amnesty ASA 23/9447/2025 with "
            "pdftotext -layout first"
        )
    source = (
        "https://www.amnesty.org/en/wp-content/uploads/2025/06/"
        "ASA2394472025ENGLISH.pdf"
    )
    rows = []
    for section in AMNESTY_TEXT.read_text(errors="replace").split("\f"):
        match = re.match(r"\s*([A-Z]{2,4}\d+)\s*\n", section)
        if not match or "LOCATION" not in section[:900]:
            continue
        coordinate = re.search(
            r"(-?\d{1,2}\.\d{4}),\s*(-?\d{2,3}\.\d{4})", section[:900]
        )
        if not coordinate:
            continue
        site_id = match.group(1)
        rows.append(
            {
                "site_id": site_id,
                "latitude": float(coordinate.group(1)),
                "longitude": float(coordinate.group(2)),
                "site_status": "verified scam compound",
                "source_url": source,
                "source_note": "Amnesty International Annex I",
            }
        )
    compounds = pd.DataFrame(rows).drop_duplicates("site_id").sort_values("site_id")
    if len(compounds) != 53:
        raise ValueError(f"Expected 53 Amnesty sites, extracted {len(compounds)}")
    compounds.to_csv(DATA / "cambodia_verified_scam_compounds.csv", index=False)
    return compounds


def cambodia_compound_query(compounds: pd.DataFrame) -> None:
    tuples = ",\n".join(
        "tuple(%r,toFloat64(%s),toFloat64(%s))"
        % (r.site_id, r.latitude, r.longitude)
        for r in compounds.itertuples(index=False)
    )
    sql = f"""
WITH [{tuples}] AS sites
SELECT
    site.1 AS site_id,
    mcc,mnc,toString(cell_type) AS cell_type,cid,
    count() AS identities,
    uniqExact(lac) AS distinct_lacs,
    sum(obs) AS crawler_observations,
    countIf(n_pos > 1) AS moving_identities,
    min(first_seen) AS first_seen,
    max(last_seen) AS last_seen
FROM cell.summary_full
ARRAY JOIN sites AS site
WHERE glat BETWEEN 10 AND 15 AND glon BETWEEN 102 AND 107
  AND greatCircleDistance(glon,glat,site.3,site.2) <= 750
GROUP BY site_id,mcc,mnc,cell_type,cid
ORDER BY site_id,identities DESC
"""
    ch_df(sql).to_csv(DATA / "cambodia_compound_cell_summary_750m.csv", index=False)

    # One row per identity in the Sihanoukville urban footprint supports a
    # matched nearest-compound comparison rather than a city-wide proximity
    # anecdote.  The 456 baseline is local Cambodian cellular infrastructure.
    sql = """
SELECT
    mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
    glat,glon,obs,n_pos,first_seen,last_seen
FROM cell.summary_full
WHERE glat BETWEEN 10.55 AND 10.67 AND glon BETWEEN 103.47 AND 103.66
  AND ((mcc=460 AND mnc=0 AND cell_type='gsm' AND cid=10) OR mcc=456)
ORDER BY mcc,mnc,lac,cid,cell_type
"""
    urban = ch_df(sql)
    urban.to_csv(DATA / "sihanoukville_candidate_and_baseline_cells.csv", index=False)
    sihanoukville_proximity(urban, compounds)


def sihanoukville_raw_query() -> None:
    """Export raw histories for the city-wide China-Mobile/CID-10 anomaly."""
    identities = ch_df("""
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
       glat,glon,obs,n_pos,first_seen,last_seen
FROM cell.summary_full
WHERE mcc=460 AND mnc=0 AND cell_type='gsm' AND cid=10
  AND greatCircleDistance(glon,glat,103.52,10.63) <= 12000
ORDER BY lac
""")
    identities.to_csv(DATA / "sihanoukville_china_mobile_cid10_identities.csv", index=False)
    key_tuples = ",".join(
        f"tuple({r.mcc},{r.mnc},{r.lac},{r.cid},'{r.cell_type}')"
        for r in identities.itertuples(index=False)
    )
    raw = ch_df(f"""
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,lat,lon,timestamp
FROM cell.geos
WHERE tuple(mcc,mnc,lac,cid,cell_type) IN ({key_tuples})
ORDER BY mcc,mnc,lac,cid,cell_type,timestamp,lat,lon
""", settings={"max_threads": 6})
    raw.to_csv(DATA / "sihanoukville_china_mobile_cid10_raw.csv", index=False)


def china_plmn_scam_site_detail_query() -> None:
    """Export China-PLMN identities around non-Chinese scam-site screens."""
    sites = pd.read_csv(SITES)
    sites = sites[sites.role.eq("candidate") & sites.site_id.ne("laukkai")]
    tuples = ",\n".join(
        "tuple(%r,toFloat64(%s),toFloat64(%s),toFloat64(%s))"
        % (r.site_id, r.latitude, r.longitude, r.radius_km)
        for r in sites.itertuples(index=False)
    )
    detail = ch_df(f"""
WITH [{tuples}] AS sites
SELECT site.1 AS site_id,mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
       glat,glon,obs,n_pos,first_seen,last_seen
FROM cell.summary_full
ARRAY JOIN sites AS site
WHERE mcc=460 AND glat BETWEEN -85 AND 85 AND glon BETWEEN -180 AND 180
  AND greatCircleDistance(glon,glat,site.3,site.2) <= site.4 * 1000
ORDER BY site_id,mcc,mnc,lac,cid,cell_type
""")
    detail.to_csv(DATA / "china_plmn_scam_site_identities.csv", index=False)


def sihanoukville_proximity(urban: pd.DataFrame, compounds: pd.DataFrame) -> None:
    sites = compounds[compounds.site_id.str.startswith("SI")].copy()
    lat = np.radians(urban.glat.to_numpy())[:, None]
    lon = np.radians(urban.glon.to_numpy())[:, None]
    site_lat = np.radians(sites.latitude.to_numpy())[None, :]
    site_lon = np.radians(sites.longitude.to_numpy())[None, :]
    q = np.sin((site_lat - lat) / 2) ** 2
    q += np.cos(lat) * np.cos(site_lat) * np.sin((site_lon - lon) / 2) ** 2
    distances = EARTH_KM * 2 * np.arcsin(np.sqrt(q))

    urban = urban.copy()
    urban["nearest_verified_compound"] = sites.site_id.to_numpy()[distances.argmin(1)]
    urban["nearest_verified_compound_km"] = distances.min(1)
    urban["group"] = np.where(
        urban.mcc.eq(460)
        & urban.mnc.eq(0)
        & urban.cell_type.str.lower().eq("gsm")
        & urban.cid.eq(10),
        "China Mobile GSM CID 10 candidate",
        "Cambodian MCC 456 baseline",
    )
    urban.to_csv(DATA / "sihanoukville_nearest_compound_cells.csv", index=False)

    metrics = []
    for name, group in urban.groupby("group"):
        row = {
            "group": name,
            "identities": len(group),
            "median_nearest_km": group.nearest_verified_compound_km.median(),
        }
        for threshold in (0.25, 0.5, 0.75, 1.0):
            slug = str(threshold).replace(".", "p")
            row[f"within_{slug}km_identities"] = int(
                group.nearest_verified_compound_km.le(threshold).sum()
            )
            row[f"within_{slug}km_fraction"] = float(
                group.nearest_verified_compound_km.le(threshold).mean()
            )
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(
        DATA / "sihanoukville_compound_proximity_comparison.csv", index=False
    )


def global_repeated_cid_query() -> None:
    """Find dense deployments where many LACs reuse one CID.

    This can surface fabricated or rotating base-station identities, but is
    not crime-specific and may also describe legitimate network conventions.
    """
    sql = """
SELECT
    round(glat,1) AS lat_bin, round(glon,1) AS lon_bin,
    country_iso AS located_country_iso,
    mcc,mnc,toString(cell_type) AS cell_type,cid,
    count() AS identities, uniqExact(lac) AS distinct_lacs,
    sum(obs) AS crawler_observations,
    countIf(n_pos > 1) AS moving_identities,
    min(first_seen) AS first_seen, max(last_seen) AS last_seen
FROM cell.summary_full
WHERE glat BETWEEN -85 AND 85 AND glon BETWEEN -180 AND 180
  AND cid >= 0
GROUP BY lat_bin,lon_bin,located_country_iso,mcc,mnc,cell_type,cid
HAVING identities >= 15 AND distinct_lacs >= 15
ORDER BY identities DESC,crawler_observations DESC
LIMIT 50000
"""
    ch_df(sql).to_csv(DATA / "global_repeated_cid_clusters.csv", index=False)


def scam_compound_relocation_screen(compounds: pd.DataFrame) -> None:
    """Find identity endpoints close to two different verified compounds."""
    site_lat = np.radians(compounds.latitude.to_numpy())[None, :]
    site_lon = np.radians(compounds.longitude.to_numpy())[None, :]
    matches = []
    for chunk in pd.read_csv(MOVING, compression="zstd", chunksize=150_000):
        nearest = []
        nearest_distance = []
        for prefix in ("endpoint_a", "endpoint_b"):
            lat = np.radians(chunk[f"{prefix}_lat"].to_numpy())[:, None]
            lon = np.radians(chunk[f"{prefix}_lon"].to_numpy())[:, None]
            q = np.sin((site_lat - lat) / 2) ** 2
            q += np.cos(lat) * np.cos(site_lat) * np.sin((site_lon - lon) / 2) ** 2
            distance = EARTH_KM * 2 * np.arcsin(np.sqrt(q))
            nearest.append(compounds.site_id.to_numpy()[distance.argmin(1)])
            nearest_distance.append(distance.min(1))
        work = chunk.copy()
        work["endpoint_a_site"] = nearest[0]
        work["endpoint_a_site_km"] = nearest_distance[0]
        work["endpoint_b_site"] = nearest[1]
        work["endpoint_b_site_km"] = nearest_distance[1]
        keep = work[
            work.endpoint_a_site_km.le(0.75)
            & work.endpoint_b_site_km.le(0.75)
            & work.endpoint_a_site.ne(work.endpoint_b_site)
        ]
        if not keep.empty:
            matches.append(keep)
    output = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
    if not output.empty:
        output = output.sort_values(
            ["max_span_km", "total_observations"], ascending=False
        )
    output.to_csv(DATA / "cambodia_compound_relocation_candidates.csv", index=False)


def compound_relocation_raw_query() -> None:
    candidates = pd.read_csv(DATA / "cambodia_compound_relocation_candidates.csv")
    key_tuples = ",".join(
        f"tuple({r.mcc},{r.mnc},{r.lac},{r.cid},'{r.cell_type}')"
        for r in candidates[KEY].drop_duplicates().itertuples(index=False)
    )
    raw = ch_df(f"""
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,lat,lon,timestamp
FROM cell.geos
WHERE tuple(mcc,mnc,lac,cid,cell_type) IN ({key_tuples})
ORDER BY mcc,mnc,lac,cid,cell_type,timestamp,lat,lon
""", settings={"max_threads": 6})
    raw.to_csv(DATA / "cambodia_compound_relocation_raw.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-sites", action="store_true",
        help="refresh the read-only site aggregate from ClickHouse",
    )
    parser.add_argument(
        "--refresh-compounds", action="store_true",
        help="extract Amnesty compounds and refresh compound-level queries",
    )
    parser.add_argument(
        "--refresh-repeated-cid", action="store_true",
        help="refresh the global repeated-CID aggregate",
    )
    parser.add_argument(
        "--refresh-sihanoukville-raw", action="store_true",
        help="refresh raw histories for the Sihanoukville CID-10 lead",
    )
    parser.add_argument(
        "--refresh-china-plmn-sites", action="store_true",
        help="refresh China-PLMN details around non-Chinese scam-site screens",
    )
    parser.add_argument(
        "--refresh-compound-relocations", action="store_true",
        help="screen moving identities between verified Cambodia compounds",
    )
    parser.add_argument(
        "--refresh-compound-relocation-raw", action="store_true",
        help="refresh raw histories of Cambodia compound relocation leads",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    ocean_screens()
    moving_bundle_screen()
    special_mcc_screen()
    if args.refresh_sites or not (DATA / "scam_site_cell_summary.csv").exists():
        scam_site_query()
    if args.refresh_compounds:
        cambodia_compound_query(amnesty_compounds())
    if args.refresh_repeated_cid:
        global_repeated_cid_query()
    if args.refresh_sihanoukville_raw:
        sihanoukville_raw_query()
    if args.refresh_china_plmn_sites:
        china_plmn_scam_site_detail_query()
    if args.refresh_compound_relocations:
        scam_compound_relocation_screen(amnesty_compounds())
    if args.refresh_compound_relocation_raw:
        compound_relocation_raw_query()
    print(f"Wrote criminal-activity lead screens to {DATA}")


if __name__ == "__main__":
    main()
