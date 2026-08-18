#!/usr/bin/env python3
"""Detect cellular-identity families consistent with synthetic or rogue BTS use.

This is a lead detector, not a classifier of criminal activity.  It combines
signals that are observable in the Apple cellular-position corpus:

* a PLMN far outside its assigned geography;
* many LACs paired with one repeated/default CID in a small destination bin;
* identifiers anchored by a stable home population but displaced elsewhere;
* exact identities with supported, bimodal long-distance histories; and
* bundles of identities sharing the same distant endpoints.

The global raw table is never scanned by this program.  The broad family query
uses ``cell.summary_full`` and all movement work uses the existing local
``data/moving-mccs`` extraction.  Optional database refreshes go through
``ch_remote``, which enforces ClickHouse ``readonly=2``.

Scores rank investigations.  They are deliberately not probabilities and do
not establish that a transmitter existed, was unauthorized, or was criminal.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rogue-bts-detector"
REFERENCE = ROOT / "data" / "reference"
MOVING = ROOT / "data" / "moving-mccs" / "identities.csv.zst"
BUNDLES = ROOT / "data" / "criminal-activity" / "moving_bundle_candidates.csv"
EXISTING_REPEATED = (
    ROOT / "data" / "criminal-activity" / "global_repeated_cid_clusters.csv"
)
REPEATED_RAW = OUT / "repeated_cid_clusters_raw.csv"
REPEATED_SHARDS = OUT / "repeated-cid-shards"
EARTH_KM = 6371.0088
REPEATED_SHARD_COUNT = 32

# Values frequently used as defaults, sentinels, examples, or conspicuous
# human choices.  They add only a small amount to the score; repetition and
# geography must supply the substantive evidence.
DEFAULT_CIDS = {0, 1, 10, 44, 52, 123, 1234, 3584, 4113, 4376, 54321, 65535}


REPEATED_QUERY = r"""
SELECT
    floor(toFloat64(glat) * 20) / 20 AS dst_lat,
    floor(toFloat64(glon) * 20) / 20 AS dst_lon,
    topK(1)(country_iso)[1] AS corpus_country_iso,
    topK(1)(country)[1] AS corpus_country,
    topK(1)(region)[1] AS corpus_region,
    topK(1)(city)[1] AS corpus_city,
    mcc,mnc,toString(cell_type) AS cell_type,cid,
    count() AS identities,
    uniqExact(lac) AS distinct_lacs,
    sum(obs) AS crawler_observations,
    countIf(n_pos > 1) AS moving_identities,
    quantileTDigest(0.5)(obs) AS median_identity_observations,
    quantileTDigest(0.5)(dateDiff('day',first_seen,last_seen)) AS median_lifespan_days,
    min(first_seen) AS earliest_seen,
    max(last_seen) AS latest_seen
FROM cell.summary_full
WHERE glat BETWEEN -85 AND 85 AND glon BETWEEN -180 AND 180
  AND cid >= 0
  AND cityHash64(mcc,mnc,toString(cell_type),cid) % {shard_count} = {shard}
GROUP BY dst_lat,dst_lon,mcc,mnc,cell_type,cid
HAVING identities >= 5 AND distinct_lacs >= 5
ORDER BY identities DESC,crawler_observations DESC
"""


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    q = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(q))


def load_plmn_countries() -> tuple[dict[tuple[int, int], set[str]], dict[int, set[str]]]:
    table = pd.read_csv(REFERENCE / "mcc-mnc-table.csv", dtype={"MNC": str})
    table["mnc_numeric"] = pd.to_numeric(table.MNC, errors="coerce")
    table = table.dropna(subset=["mnc_numeric", "ISO"])
    table["iso"] = table.ISO.astype(str).str.upper()
    by_plmn: dict[tuple[int, int], set[str]] = {}
    for (mcc, mnc), group in table.groupby(["MCC", "mnc_numeric"]):
        by_plmn[(int(mcc), int(mnc))] = set(group.iso)
    by_mcc: dict[int, set[str]] = {
        int(mcc): set(group.iso) for mcc, group in table.groupby("MCC")
    }
    return by_plmn, by_mcc


class CountryGeometry:
    """Natural-Earth country lookup and approximate distance-to-home geometry."""

    def __init__(self) -> None:
        raw = json.loads((REFERENCE / "ne_10m_admin_0_map_units.geojson").read_text())
        pieces: dict[str, list] = defaultdict(list)
        self.named: list[tuple[str, str, object]] = []
        for feature in raw["features"]:
            props = feature["properties"]
            iso = props.get("ISO_A2_EH") or props.get("ISO_A2")
            if not iso or iso == "-99":
                continue
            iso = iso.upper()
            geom = shape(feature["geometry"])
            pieces[iso].append(geom)
            self.named.append((iso, props.get("NAME_EN") or props.get("NAME"), geom))
        self.by_iso = {iso: unary_union(geoms) for iso, geoms in pieces.items()}
        self._tree_geometries = [item[2] for item in self.named]
        self._tree = STRtree(self._tree_geometries)
        self._locate_cache: dict[tuple[float, float], tuple[str, str]] = {}
        self._distance_cache: dict[tuple[float, float, tuple[str, ...]], float] = {}

    def locate(self, lat: float, lon: float) -> tuple[str, str]:
        cache_key = (round(lat, 6), round(lon, 6))
        if cache_key in self._locate_cache:
            return self._locate_cache[cache_key]
        point = Point(lon, lat)
        for index in self._tree.query(point, predicate="intersects"):
            iso, name, geom = self.named[int(index)]
            if geom.covers(point):
                self._locate_cache[cache_key] = (iso, name)
                return iso, name
        self._locate_cache[cache_key] = ("", "")
        return "", ""

    def distance_km(self, lat: float, lon: float, homes: set[str]) -> float:
        cache_key = (round(lat, 6), round(lon, 6), tuple(sorted(homes)))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        point = Point(lon, lat)
        best = math.inf
        for iso in homes:
            geom = self.by_iso.get(iso)
            if geom is None:
                continue
            if geom.covers(point):
                self._distance_cache[cache_key] = 0.0
                return 0.0
            near = nearest_points(point, geom)[1]
            best = min(best, haversine(lat, lon, near.y, near.x))
        self._distance_cache[cache_key] = best
        return best


def assigned_countries(
    mcc: int,
    mnc: int,
    by_plmn: dict[tuple[int, int], set[str]],
    by_mcc: dict[int, set[str]],
) -> set[str]:
    exact = by_plmn.get((int(mcc), int(mnc)))
    if exact:
        return exact
    fallback = by_mcc.get(int(mcc), set())
    return fallback if len(fallback) == 1 else set()


def refresh_repeated() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    REPEATED_SHARDS.mkdir(parents=True, exist_ok=True)
    pieces = []
    for shard in range(REPEATED_SHARD_COUNT):
        path = REPEATED_SHARDS / f"part-{shard:02d}.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            query = REPEATED_QUERY.format(
                shard_count=REPEATED_SHARD_COUNT, shard=shard
            )
            frame = ch_df(
                query,
                settings={
                    "max_threads": 6,
                    "max_execution_time": 7200,
                    "max_bytes_before_external_group_by": 1_000_000_000,
                },
            )
            frame.to_csv(path, index=False)
        pieces.append(frame)
        print(
            f"repeated-CID shard {shard + 1}/{REPEATED_SHARD_COUNT}: "
            f"{len(frame):,} qualifying clusters",
            flush=True,
        )
    frame = pd.concat(pieces, ignore_index=True)
    frame.to_csv(REPEATED_RAW, index=False)
    return frame


def import_existing_repeated() -> pd.DataFrame:
    """Import the earlier comprehensive high-support global screen.

    That extraction used 0.1-degree bins and a minimum of 15 identities/LACs.
    It is a useful high-precision seed when a finer refresh would impose an
    excessive read-query aggregation load on the shared database.
    """
    frame = pd.read_csv(EXISTING_REPEATED).rename(
        columns={
            "lat_bin": "dst_lat",
            "lon_bin": "dst_lon",
            "located_country_iso": "corpus_country_iso",
            "first_seen": "earliest_seen",
            "last_seen": "latest_seen",
        }
    )
    for column in ["corpus_country", "corpus_region", "corpus_city"]:
        frame[column] = ""
    frame["median_identity_observations"] = np.nan
    frame["median_lifespan_days"] = np.nan
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPEATED_RAW, index=False)
    return frame


def ensure_host_geography(frame: pd.DataFrame, countries: CountryGeometry) -> pd.DataFrame:
    frame = frame.copy()
    host_iso, host_name = [], []
    for row in frame.itertuples(index=False):
        iso, name = countries.locate(float(row.dst_lat) + 0.025, float(row.dst_lon) + 0.025)
        corpus = str(getattr(row, "corpus_country_iso", "") or "").upper()
        if not iso and corpus not in {"", "NAN", "??"}:
            iso = corpus
            name = str(getattr(row, "corpus_country", "") or "")
        host_iso.append(iso)
        host_name.append(name)
    frame["host_country_iso"] = host_iso
    frame["host_country"] = host_name
    return frame


def score_repeated_families(frame: pd.DataFrame, countries: CountryGeometry) -> pd.DataFrame:
    by_plmn, by_mcc = load_plmn_countries()
    frame = ensure_host_geography(frame, countries)
    homes, distance = [], []
    for row in frame.itertuples(index=False):
        assigned = assigned_countries(row.mcc, row.mnc, by_plmn, by_mcc)
        homes.append(";".join(sorted(assigned)))
        distance.append(countries.distance_km(row.dst_lat + 0.025, row.dst_lon + 0.025, assigned))
    frame["assigned_country_iso"] = homes
    frame["distance_to_assigned_area_km"] = distance
    frame["foreign"] = (
        frame.host_country_iso.ne("")
        & frame.assigned_country_iso.ne("")
        & ~frame.apply(
            lambda r: r.host_country_iso in set(r.assigned_country_iso.split(";")), axis=1
        )
    )
    frame["far_outside"] = (
        frame.assigned_country_iso.ne("")
        & np.isfinite(frame.distance_to_assigned_area_km)
        & frame.distance_to_assigned_area_km.ge(200)
    )
    frame["lac_identity_fraction"] = frame.distinct_lacs / frame.identities
    frame["moving_fraction"] = frame.moving_identities / frame.identities

    family = ["mcc", "mnc", "cell_type", "cid"]
    totals = frame.groupby(family).identities.transform("sum")
    maxima = frame.groupby(family).identities.transform("max")
    frame["family_cluster_share"] = frame.identities / totals
    frame["is_largest_family_cluster"] = frame.identities.eq(maxima)
    frame["family_cluster_count"] = frame.groupby(family).identities.transform("size")

    # Ranking score.  High scores require geography plus structure; conspicuous
    # defaults alone cannot create a strong result.
    size_component = np.clip(np.log10(frame.identities) / 2.5, 0, 1) * 15
    frame["detector_score"] = (
        frame.foreign.astype(int) * 10
        + frame.far_outside.astype(int) * 25
        + size_component
        + frame.lac_identity_fraction.ge(0.95).astype(int) * 10
        + frame.family_cluster_share.ge(0.5).astype(int) * 10
        + frame.is_largest_family_cluster.astype(int) * 5
        + frame.cell_type.eq("gsm").astype(int) * 8
        + frame.cid.isin(DEFAULT_CIDS).astype(int) * 5
        + frame.moving_fraction.ge(0.1).astype(int) * 4
    )
    frame.loc[~frame.assigned_country_iso.ne(""), "detector_score"] -= 15
    frame.loc[frame.distance_to_assigned_area_km.lt(100), "detector_score"] -= 20

    def reasons(row: pd.Series) -> str:
        result = []
        if row.far_outside:
            result.append("far outside assigned PLMN geography")
        if row.lac_identity_fraction >= 0.95:
            result.append("one-CID/many-LAC topology")
        if row.family_cluster_share >= 0.5:
            result.append("dominant global cluster for exact PLMN/CID family")
        if row.cid in DEFAULT_CIDS:
            result.append("default/sentinel-like CID")
        if row.moving_fraction >= 0.1:
            result.append("multi-position identities")
        return "; ".join(result)

    frame["signals"] = frame.apply(reasons, axis=1)
    return frame.sort_values(
        ["detector_score", "identities", "crawler_observations"], ascending=False
    )


def score_moving_identities(countries: CountryGeometry) -> pd.DataFrame:
    by_plmn, by_mcc = load_plmn_countries()
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(MOVING, compression="zstd", chunksize=200_000):
        keep = chunk[
            chunk.max_span_km.ge(100)
            & chunk.total_observations.ge(6)
            & chunk.home_observations.ge(3)
            & chunk.displaced_from_home_observations.ge(3)
            & chunk.position_rows.between(2, 500)
        ].copy()
        if not keep.empty:
            selected.append(keep)
    frame = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if frame.empty:
        return frame

    away_lat, away_lon, away_country, home_distance = [], [], [], []
    assigned_iso = []
    for row in frame.itertuples(index=False):
        da = haversine(row.home_lat, row.home_lon, row.endpoint_a_lat, row.endpoint_a_lon)
        db = haversine(row.home_lat, row.home_lon, row.endpoint_b_lat, row.endpoint_b_lon)
        lat, lon = (
            (row.endpoint_a_lat, row.endpoint_a_lon)
            if da >= db else (row.endpoint_b_lat, row.endpoint_b_lon)
        )
        iso, _ = countries.locate(lat, lon)
        assigned = assigned_countries(row.mcc, row.mnc, by_plmn, by_mcc)
        away_lat.append(lat)
        away_lon.append(lon)
        away_country.append(iso)
        assigned_iso.append(";".join(sorted(assigned)))
        home_distance.append(countries.distance_km(lat, lon, assigned))
    frame["away_lat"] = away_lat
    frame["away_lon"] = away_lon
    frame["away_country_iso"] = away_country
    frame["assigned_country_iso"] = assigned_iso
    frame["away_distance_to_assigned_area_km"] = home_distance
    frame["supported_bimodal"] = (
        frame.assigned_country_iso.ne("")
        & np.isfinite(frame.away_distance_to_assigned_area_km)
        & frame.home_fraction.between(0.05, 0.95)
        & frame.position_rows.le(12)
        & frame.away_distance_to_assigned_area_km.ge(200)
    )
    frame["movement_score"] = (
        np.clip(np.log10(frame.max_span_km) / 4, 0, 1) * 25
        + np.clip(np.log10(frame.total_observations) / 3, 0, 1) * 10
        + frame.away_distance_to_assigned_area_km.ge(200).astype(int) * 25
        + frame.supported_bimodal.astype(int) * 20
        + frame.cell_type.eq("gsm").astype(int) * 8
        + frame.displaced_from_home_observations.ge(10).astype(int) * 5
    )
    return frame.sort_values(
        ["movement_score", "max_span_km", "total_observations"], ascending=False
    )


def score_bundles(countries: CountryGeometry) -> pd.DataFrame:
    if not BUNDLES.exists():
        return pd.DataFrame()
    frame = pd.read_csv(BUNDLES)
    frame = frame[
        frame.identities.ge(3)
        & frame.median_span_km.ge(100)
        & frame.total_observations.ge(12)
        & frame.artifact_flag.fillna("").eq("")
    ].copy()
    left_iso, right_iso = [], []
    for row in frame.itertuples(index=False):
        left_iso.append(countries.locate(row.endpoint_1_lat, row.endpoint_1_lon)[0])
        right_iso.append(countries.locate(row.endpoint_2_lat, row.endpoint_2_lon)[0])
    frame["endpoint_1_country_iso"] = left_iso
    frame["endpoint_2_country_iso"] = right_iso
    frame["cross_country"] = frame.endpoint_1_country_iso.ne(frame.endpoint_2_country_iso)
    frame["bundle_score"] = (
        np.clip(np.log10(frame.identities) / 2, 0, 1) * 25
        + np.clip(np.log10(frame.median_span_km) / 4, 0, 1) * 20
        + frame.cross_country.astype(int) * 15
        + frame.operators.eq(1).astype(int) * 10
        + frame.technologies.eq("gsm").astype(int) * 8
        + frame.median_home_fraction.between(0.1, 0.9).astype(int) * 10
    )
    return frame.sort_values(
        ["bundle_score", "identities", "median_span_km"], ascending=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-repeated",
        action="store_true",
        help="refresh the read-only global repeated-CID aggregate",
    )
    parser.add_argument(
        "--use-existing-aggregate",
        action="store_true",
        help="seed from the existing 0.1-degree, >=15-identity global screen",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    countries = CountryGeometry()
    if args.use_existing_aggregate:
        repeated = import_existing_repeated()
    elif args.refresh_repeated or not REPEATED_RAW.exists():
        repeated = refresh_repeated()
    else:
        repeated = pd.read_csv(REPEATED_RAW)

    scored = score_repeated_families(repeated, countries)
    scored.to_csv(OUT / "repeated_cid_family_scores.csv", index=False)

    moving = score_moving_identities(countries)
    moving.to_csv(OUT / "displaced_identity_scores.csv", index=False)

    bundles = score_bundles(countries)
    bundles.to_csv(OUT / "movement_bundle_scores.csv", index=False)

    manifest = {
        "purpose": "lead ranking; scores are not probabilities or rogue-BTS classifications",
        "source_tables": ["cell.summary_full (read-only aggregate)", "local moving-mccs extraction"],
        "repeated_clusters": int(len(scored)),
        "far_foreign_repeated_clusters": int((scored.foreign & scored.far_outside).sum()),
        "supported_displaced_identities": int(len(moving)),
        "supported_bimodal_identities": int(moving.supported_bimodal.sum()) if not moving.empty else 0,
        "movement_bundles": int(len(bundles)),
        "minimum_family_size": 15 if args.use_existing_aggregate else 5,
        "spatial_bin_degrees": 0.1 if args.use_existing_aggregate else 0.05,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
