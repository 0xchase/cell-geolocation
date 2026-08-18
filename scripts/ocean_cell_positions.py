#!/usr/bin/env python3
"""Export every cell/position observed at least 25 km from ocean shorelines.

The output unit is one cellular identity at one 0.01-degree coordinate. Raw
polls are represented by ``observations``, ``first_seen``, and ``last_seen``.
The remote ClickHouse database is always opened in read-only mode.

GSHHG 2.3.7 level-1 polygons define land versus ocean. Coastline vertices are
densified to at most 0.5 km spacing and indexed on the unit sphere. The stored
distance is a conservative lower bound: half the sampling interval is removed
from the nearest-sample distance before applying the 25 km threshold.
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import shapefile
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, shape


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "oceans"
COORDS = CACHE / "coord_geo.tsv.gz"
GSHHG = CACHE / "gshhg" / "GSHHS_shp" / "f" / "GSHHS_f_L1.shp"
OCEAN_KEYS = CACHE / "ocean_keys_25km.tsv.gz"
OUTPUT = ROOT / "data" / "oceans" / "ocean_cell_positions.csv"

HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
CH_PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")

EARTH_KM = 6371.0088
BUFFER_KM = 25.0
COAST_SAMPLE_KM = 0.5
POINT_CHUNK = 500_000


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_coordinate_keys() -> np.ndarray:
    log(f"Loading coordinate dictionary from {COORDS}")
    with gzip.open(COORDS, "rt", encoding="ascii") as stream:
        coords = np.loadtxt(stream, delimiter="\t", dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise RuntimeError(f"Unexpected coordinate dictionary shape: {coords.shape}")
    raw_count = len(coords)
    coords = np.unique(coords, axis=0)
    log(f"Loaded {raw_count:,} coordinate rows ({len(coords):,} unique keys)")
    return coords


def load_land():
    log(f"Loading GSHHG land polygons from {GSHHG}")
    reader = shapefile.Reader(str(GSHHG))
    polygons = []
    for item in reader.iterShapes():
        geom = shape(item.__geo_interface__)
        if not geom.is_empty:
            polygons.append(geom)
    land = MultiPolygon(polygons)
    shapely.prepare(land)
    log(f"Prepared {len(polygons):,} land polygons")
    return land


def classify_land(coords: np.ndarray, land_geometry) -> np.ndarray:
    """Return a mask for keys whose coordinate centers lie inside land."""
    land = np.zeros(len(coords), dtype=bool)
    for start in range(0, len(coords), POINT_CHUNK):
        stop = min(start + POINT_CHUNK, len(coords))
        # Boundary-inclusive so exact vertices such as the South Pole are land.
        land[start:stop] = shapely.intersects_xy(
            land_geometry,
            coords[start:stop, 1] / 100.0,
            coords[start:stop, 0] / 100.0,
        )
        # GSHHG's antimeridian split leaves the singular -90-degree coordinate
        # outside its polygon rings even though the South Pole is Antarctic land.
        land[start:stop] |= coords[start:stop, 0] == -9000
        if start == 0 or stop == len(coords) or stop % 5_000_000 == 0:
            log(f"Land classification: {stop:,}/{len(coords):,}")
    return land


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    cos_lat = np.cos(lat_r)
    return np.column_stack((cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)))


def coast_samples() -> np.ndarray:
    """Return unit-sphere samples along level-1 ocean/land boundaries."""
    log("Building a <=0.5 km coastline sample")
    reader = shapefile.Reader(str(GSHHG))
    chunks: list[np.ndarray] = []
    vertices = 0
    inserted = 0

    for item in reader.iterShapes():
        points = np.asarray(item.points, dtype=np.float64)
        parts = list(item.parts) + [len(points)]
        for begin, end in zip(parts, parts[1:]):
            ring = points[begin:end]
            if len(ring) < 2:
                continue
            chunks.append(ring)
            vertices += len(ring)

            lon1, lat1 = ring[:-1, 0], ring[:-1, 1]
            lon2, lat2 = ring[1:, 0], ring[1:, 1]
            dlon_deg = ((lon2 - lon1 + 180.0) % 360.0) - 180.0
            # GSHHG splits several land polygons at the antimeridian. Their
            # vertical closing edges are not coastlines and must not be sampled.
            seam = (np.abs(np.abs(lon1) - 180.0) < 1e-9) & (np.abs(np.abs(lon2) - 180.0) < 1e-9)
            lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
            dlon_r = np.radians(dlon_deg)
            a = np.sin((lat2_r - lat1_r) / 2.0) ** 2
            a += np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon_r / 2.0) ** 2
            distance = 2.0 * EARTH_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))

            for idx in np.flatnonzero((distance > COAST_SAMPLE_KM) & ~seam):
                pieces = int(math.ceil(float(distance[idx]) / COAST_SAMPLE_KM))
                fraction = np.arange(1, pieces, dtype=np.float64) / pieces
                local_lon = lon1[idx] + fraction * dlon_deg[idx]
                local_lon = ((local_lon + 180.0) % 360.0) - 180.0
                local_lat = lat1[idx] + fraction * (lat2[idx] - lat1[idx])
                chunks.append(np.column_stack((local_lon, local_lat)))
                inserted += pieces - 1

    lonlat = np.concatenate(chunks)
    xyz = lonlat_to_xyz(lonlat[:, 0], lonlat[:, 1])
    log(f"Coastline sample: {vertices:,} vertices + {inserted:,} inserted points")
    return xyz


def select_ocean_keys(coords: np.ndarray, land: np.ndarray, coast_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log("Building coastline nearest-neighbour index")
    tree = cKDTree(coast_xyz, compact_nodes=True, balanced_tree=True)
    water_idx = np.flatnonzero(~land)
    retained: list[np.ndarray] = []
    distances: list[np.ndarray] = []

    for start in range(0, len(water_idx), POINT_CHUNK):
        idx = water_idx[start:start + POINT_CHUNK]
        xyz = lonlat_to_xyz(coords[idx, 1] / 100.0, coords[idx, 0] / 100.0)
        chord, _ = tree.query(xyz, k=1, workers=-1)
        sampled_km = 2.0 * EARTH_KM * np.arcsin(np.minimum(1.0, chord / 2.0))
        lower_km = np.maximum(0.0, sampled_km - COAST_SAMPLE_KM / 2.0)
        keep = lower_km >= BUFFER_KM
        retained.append(idx[keep])
        distances.append(lower_km[keep].astype(np.float32))
        done = min(start + POINT_CHUNK, len(water_idx))
        if start == 0 or done == len(water_idx) or done % 5_000_000 == 0:
            log(f"Coast-distance classification: {done:,}/{len(water_idx):,}")

    return np.concatenate(retained), np.concatenate(distances)


def write_ocean_keys(coords: np.ndarray, idx: np.ndarray, distance: np.ndarray) -> None:
    log(f"Writing {len(idx):,} ocean keys to {OCEAN_KEYS}")
    OCEAN_KEYS.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OCEAN_KEYS, "wt", encoding="ascii", newline="") as stream:
        for row_idx, dist in zip(idx, distance, strict=True):
            stream.write(f"{coords[row_idx, 0]}\t{coords[row_idx, 1]}\t{dist:.3f}\n")


def export_cells() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    sql = """
SELECT
    g.mcc, g.mnc, g.lac, g.cid, g.cell_type,
    g.klat / 100.0 AS lat,
    g.klon / 100.0 AS lon,
    round(any(o.distance_to_land_km), 3) AS distance_to_land_km,
    count() AS observations,
    min(g.timestamp) AS first_seen,
    max(g.timestamp) AS last_seen
FROM
(
    SELECT
        mcc, mnc, lac, cid, cell_type, timestamp,
        toInt32(round(lat * 100)) AS klat,
        toInt32(round(lon * 100)) AS klon
    FROM cell.geos
    WHERE NOT (lat = 0 AND lon = 0)
) AS g
INNER JOIN ocean AS o USING (klat, klon)
GROUP BY g.mcc, g.mnc, g.lac, g.cid, g.cell_type, g.klat, g.klon
ORDER BY g.mcc, g.mnc, g.lac, g.cid, g.cell_type, g.klat, g.klon
FORMAT CSVWithNames
""".strip()
    remote_args = [
        "clickhouse-client",
        "--password", CH_PASSWORD,
        "--readonly", "1",
        "--max_threads", "12",
        "--max_execution_time", "7200",
        "--optimize_aggregation_in_order", "1",
        "--external",
        "--file", "-",
        "--name", "ocean",
        "--structure", "klat Int32, klon Int32, distance_to_land_km Float32",
        "--format", "TabSeparated",
        "--query", sql,
    ]
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST, shlex.join(remote_args)]
    log("Scanning cell.geos read-only and writing the final CSV")
    with gzip.open(OCEAN_KEYS, "rb") as keys, temporary.open("wb") as output:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        shutil.copyfileobj(keys, proc.stdin)
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
    if returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(stderr.decode("utf-8", errors="replace"))
    temporary.replace(OUTPUT)
    log(f"Wrote {OUTPUT}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reclassify", action="store_true", help="rebuild the cached 25 km ocean-coordinate list")
    parser.add_argument("--classify-only", action="store_true", help="stop after building the coordinate list")
    args = parser.parse_args()

    if args.reclassify or not OCEAN_KEYS.exists():
        if not COORDS.exists() or not GSHHG.exists():
            raise FileNotFoundError("Coordinate dictionary or GSHHG cache is missing")
        coords = load_coordinate_keys()
        land_geometry = load_land()
        land = classify_land(coords, land_geometry)
        coast_xyz = coast_samples()
        idx, distance = select_ocean_keys(coords, land, coast_xyz)
        write_ocean_keys(coords, idx, distance)

    if not args.classify_only:
        export_cells()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
