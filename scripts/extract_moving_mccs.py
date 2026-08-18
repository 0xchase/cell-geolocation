#!/usr/bin/env python3
"""Extract moving cell identities and their observed position histories.

The source ClickHouse connection is forced into readonly mode.  For every full
cell identity, the extraction computes the exact maximum great-circle distance
between every pair of observed ~1 km position bins in ``cell.cellpos``.  Every
identity with a span greater than 10 km is retained, along with all of its
position rows.  The endpoints are actual observations, never bounding-box
corners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "moving-mccs"
DEFAULT_HOST = "ckanipe@nominatim.cybre.io"

IDENTITY_KEY = "mcc, mnc, lac, cid, cell_type"

DISTANCE_BAND_SQL = """multiIf(
    max_span_km < 25, '10-25 km',
    max_span_km < 100, '25-100 km',
    max_span_km < 500, '100-500 km',
    max_span_km < 1000, '500-1,000 km',
    max_span_km < 5000, '1,000-5,000 km',
    max_span_km < 10000, '5,000-10,000 km',
    '10,000+ km'
)"""

CANDIDATES_CTE = f"""
position_groups AS
(
    SELECT
        {IDENTITY_KEY},
        groupArray(tuple(plat, plon)) AS points,
        count() AS position_rows,
        sum(obs) AS total_observations
    FROM cell.cellpos
    GROUP BY {IDENTITY_KEY}
),
spans AS
(
    SELECT
        *,
        arrayMax(arrayFlatten(arrayMap(a -> arrayMap(b -> tuple(
            greatCircleDistance(tupleElement(a, 2) / 100,
                                tupleElement(a, 1) / 100,
                                tupleElement(b, 2) / 100,
                                tupleElement(b, 1) / 100) / 1000,
            tupleElement(a, 1), tupleElement(a, 2),
            tupleElement(b, 1), tupleElement(b, 2)
        ), points), points))) AS farthest_pair
    FROM position_groups
),
candidates AS
(
    SELECT
        {IDENTITY_KEY},
        position_rows,
        total_observations,
        tupleElement(farthest_pair, 1) AS max_span_km,
        tupleElement(farthest_pair, 2) AS endpoint_a_plat,
        tupleElement(farthest_pair, 3) AS endpoint_a_plon,
        tupleElement(farthest_pair, 4) AS endpoint_b_plat,
        tupleElement(farthest_pair, 5) AS endpoint_b_plon,
        {DISTANCE_BAND_SQL} AS distance_band
    FROM spans
    WHERE max_span_km > 10
)
"""

BAND_SUMMARY_QUERY = f"""
WITH {CANDIDATES_CTE}
SELECT
    distance_band,
    count() AS identities,
    sum(position_rows) AS all_position_rows,
    sum(total_observations) AS all_raw_observations,
    min(max_span_km) AS minimum_span_km,
    max(max_span_km) AS maximum_span_km
FROM candidates AS c
GROUP BY distance_band
ORDER BY minimum_span_km
FORMAT CSVWithNames
"""

IDENTITIES_QUERY = f"""
WITH {CANDIDATES_CTE}
SELECT
    c.mcc AS mcc,
    c.mnc AS mnc,
    c.lac AS lac,
    c.cid AS cid,
    c.cell_type AS cell_type,
    c.distance_band AS distance_band,
    c.max_span_km AS max_span_km,
    c.endpoint_a_plat / 100 AS endpoint_a_lat,
    c.endpoint_a_plon / 100 AS endpoint_a_lon,
    c.endpoint_b_plat / 100 AS endpoint_b_lat,
    c.endpoint_b_plon / 100 AS endpoint_b_lon,
    ifNull(a_geo.country_iso, '') AS endpoint_a_country_iso,
    ifNull(b_geo.country_iso, '') AS endpoint_b_country_iso,
    h.hlat / 100 AS home_lat,
    h.hlon / 100 AS home_lon,
    ifNull(home_geo.country_iso, '') AS home_country_iso,
    c.total_observations AS total_observations,
    h.home_obs AS home_observations,
    h.home_obs / c.total_observations AS home_fraction,
    c.position_rows AS position_rows,
    ifNull(d.displaced_positions, 0) AS displaced_from_home_positions,
    ifNull(d.displaced_observations, 0) AS displaced_from_home_observations,
    ifNull(d.displaced_observations, 0) / c.total_observations AS displaced_from_home_fraction,
    s.first_seen AS first_seen,
    s.last_seen AS last_seen,
    dateDiff('day', s.first_seen, s.last_seen) AS active_days
FROM candidates AS c
INNER JOIN cell.cellhome AS h USING ({IDENTITY_KEY})
INNER JOIN cell.summary AS s USING ({IDENTITY_KEY})
LEFT JOIN cell.coord_a0 AS home_geo
    ON h.hlat = home_geo.klat AND h.hlon = home_geo.klon
LEFT JOIN cell.coord_a0 AS a_geo
    ON c.endpoint_a_plat = a_geo.klat AND c.endpoint_a_plon = a_geo.klon
LEFT JOIN cell.coord_a0 AS b_geo
    ON c.endpoint_b_plat = b_geo.klat AND c.endpoint_b_plon = b_geo.klon
LEFT JOIN
(
    SELECT
        {IDENTITY_KEY},
        count() AS displaced_positions,
        sum(obs) AS displaced_observations
    FROM cell.displaced
    GROUP BY {IDENTITY_KEY}
) AS d USING ({IDENTITY_KEY})
FORMAT CSVWithNames
"""

POSITIONS_QUERY = f"""
SELECT
    p.mcc AS mcc,
    p.mnc AS mnc,
    p.lac AS lac,
    p.cid AS cid,
    p.cell_type AS cell_type,
    c.distance_band AS distance_band,
    c.max_span_km AS max_span_km,
    p.plat / 100 AS lat,
    p.plon / 100 AS lon,
    p.obs AS observations,
    p.obs / c.total_observations AS observation_fraction,
    p.first_seen AS first_seen,
    p.last_seen AS last_seen
FROM cell.cellpos AS p
INNER JOIN candidates AS c USING ({IDENTITY_KEY})
FORMAT CSVWithNames
"""

EXTERNAL_CANDIDATE_STRUCTURE = (
    "mcc UInt16, mnc UInt16, lac UInt32, cid Int64, "
    "cell_type Enum8('gsm' = 1, 'cdma' = 2, 'lte' = 3, 'nr' = 4), "
    "distance_band String, max_span_km Float64, total_observations UInt64"
)

SOURCE_SNAPSHOT_QUERY = """
WITH
    (SELECT sum(rows) FROM system.parts
     WHERE active AND database = 'cell' AND table = 'geos') AS raw_rows,
    (SELECT sum(obs) FROM cell.cellpos) AS positioned_observations
SELECT
    raw_rows,
    sum(obs) AS summarized_raw_rows,
    countIf(cid > 0 AND NOT (lat_min = 0 AND lat_max = 0
                             AND lon_min = 0 AND lon_max = 0)) AS valid_identities,
    (SELECT sum(rows) FROM system.parts
     WHERE active AND database = 'cell' AND table = 'cellpos') AS position_rows,
    positioned_observations,
    raw_rows - positioned_observations AS excluded_unpositioned_observations,
    min(first_seen) AS first_observation,
    max(last_seen) AS last_observation
FROM cell.summary
FORMAT CSVWithNames
"""


def remote_command(host: str, password: str) -> list[str]:
    clickhouse = " ".join(
        [
            "clickhouse-client",
            "--password",
            shlex.quote(password),
            "--readonly 1",
            "--max_threads 12",
            "--max_execution_time 7200",
            "--receive_timeout 7200",
            "--send_timeout 7200",
        ]
    )
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        host,
        clickhouse,
    ]


def remote_external_command(host: str, password: str, query: str) -> list[str]:
    args = [
        "clickhouse-client",
        "--password", password,
        "--readonly", "1",
        "--max_threads", "12",
        "--max_execution_time", "7200",
        "--receive_timeout", "7200",
        "--send_timeout", "7200",
        "--external",
        "--file", "-",
        "--name", "candidates",
        "--structure", EXTERNAL_CANDIDATE_STRUCTURE,
        "--format", "TabSeparated",
        "--query", query.strip(),
    ]
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        host,
        " ".join(shlex.quote(arg) for arg in args),
    ]


def run_small_query(query: str, output: Path, host: str, password: str) -> None:
    """Run a small query and atomically save its uncompressed output."""
    tmp = output.with_suffix(output.suffix + ".part")
    proc = subprocess.run(
        remote_command(host, password),
        input=query.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip())
    tmp.write_bytes(proc.stdout)
    tmp.replace(output)


def run_compressed_query(query: str, output: Path, host: str, password: str) -> None:
    """Stream a ClickHouse query through zstd and atomically save it."""
    tmp = output.with_suffix(output.suffix + ".part")
    ssh_stderr = output.with_suffix(output.suffix + ".ssh.stderr")
    zstd_stderr = output.with_suffix(output.suffix + ".zstd.stderr")
    for stale in (tmp, ssh_stderr, zstd_stderr):
        stale.unlink(missing_ok=True)

    print(f"[extract] {output.name}", flush=True)
    with ssh_stderr.open("wb") as ssh_err, zstd_stderr.open("wb") as zstd_err:
        ssh_proc = subprocess.Popen(
            remote_command(host, password),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=ssh_err,
        )
        assert ssh_proc.stdin is not None
        assert ssh_proc.stdout is not None
        zstd_proc = subprocess.Popen(
            ["zstd", "-T0", "-12", "-f", "-o", str(tmp)],
            stdin=ssh_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=zstd_err,
        )
        ssh_proc.stdout.close()
        try:
            ssh_proc.stdin.write(query.encode())
            ssh_proc.stdin.close()
        except BrokenPipeError:
            pass
        zstd_status = zstd_proc.wait()
        ssh_status = ssh_proc.wait()

    if ssh_status or zstd_status:
        ssh_message = ssh_stderr.read_text(errors="replace").strip()
        zstd_message = zstd_stderr.read_text(errors="replace").strip()
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"extraction failed (ssh={ssh_status}, zstd={zstd_status})\n"
            f"ssh: {ssh_message}\nzstd: {zstd_message}"
        )

    subprocess.run(["zstd", "-q", "-t", str(tmp)], check=True)
    tmp.replace(output)
    ssh_stderr.unlink(missing_ok=True)
    zstd_stderr.unlink(missing_ok=True)
    print(f"[done] {output.name}: {output.stat().st_size:,} bytes", flush=True)


def build_candidate_index(identities: Path, output: Path, *, force: bool = False) -> None:
    """Derive the compact external ClickHouse join table from the identity CSV."""
    if output.exists() and output.stat().st_mtime >= identities.stat().st_mtime and not force:
        print(f"[skip] {output.name} is current")
        return
    tmp = output.with_suffix(output.suffix + ".part")
    tmp.unlink(missing_ok=True)
    decoder = subprocess.Popen(["zstd", "-dc", str(identities)], stdout=subprocess.PIPE)
    compressor = subprocess.Popen(
        ["zstd", "-T0", "-9", "-f", "-o", str(tmp)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    assert decoder.stdout is not None
    assert compressor.stdin is not None
    reader = csv.DictReader(io.TextIOWrapper(decoder.stdout, encoding="utf-8", newline=""))
    fields = [
        "mcc", "mnc", "lac", "cid", "cell_type", "distance_band",
        "max_span_km", "total_observations",
    ]
    try:
        for row in reader:
            compressor.stdin.write(("\t".join(row[field] for field in fields) + "\n").encode())
        compressor.stdin.close()
        compressor_status = compressor.wait()
        decoder_status = decoder.wait()
    except BaseException:
        compressor.kill()
        decoder.kill()
        tmp.unlink(missing_ok=True)
        raise
    if decoder_status or compressor_status:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"candidate index failed (decoder={decoder_status}, compressor={compressor_status})"
        )
    subprocess.run(["zstd", "-q", "-t", str(tmp)], check=True)
    tmp.replace(output)
    print(f"[done] {output.name}: {output.stat().st_size:,} bytes", flush=True)


def run_external_positions(
    query: str,
    candidate_index: Path,
    output: Path,
    host: str,
    password: str,
) -> None:
    """Stream candidates into ClickHouse as an external table and export paths."""
    tmp = output.with_suffix(output.suffix + ".part")
    ssh_stderr = output.with_suffix(output.suffix + ".ssh.stderr")
    zstd_stderr = output.with_suffix(output.suffix + ".zstd.stderr")
    for stale in (tmp, ssh_stderr, zstd_stderr):
        stale.unlink(missing_ok=True)
    print(f"[extract] {output.name} using {candidate_index.name}", flush=True)
    with ssh_stderr.open("wb") as ssh_err, zstd_stderr.open("wb") as zstd_err:
        decoder = subprocess.Popen(
            ["zstd", "-dc", str(candidate_index)],
            stdout=subprocess.PIPE,
            stderr=zstd_err,
        )
        assert decoder.stdout is not None
        ssh_proc = subprocess.Popen(
            remote_external_command(host, password, query),
            stdin=decoder.stdout,
            stdout=subprocess.PIPE,
            stderr=ssh_err,
        )
        decoder.stdout.close()
        assert ssh_proc.stdout is not None
        compressor = subprocess.Popen(
            ["zstd", "-T0", "-12", "-f", "-o", str(tmp)],
            stdin=ssh_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=zstd_err,
        )
        ssh_proc.stdout.close()
        compressor_status = compressor.wait()
        ssh_status = ssh_proc.wait()
        decoder_status = decoder.wait()
    if decoder_status or ssh_status or compressor_status:
        message = ssh_stderr.read_text(errors="replace").strip()
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "external position extraction failed "
            f"(decoder={decoder_status}, ssh={ssh_status}, zstd={compressor_status})\n{message}"
        )
    subprocess.run(["zstd", "-q", "-t", str(tmp)], check=True)
    tmp.replace(output)
    ssh_stderr.unlink(missing_ok=True)
    zstd_stderr.unlink(missing_ok=True)
    print(f"[done] {output.name}: {output.stat().st_size:,} bytes", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_dir: Path, host: str, password: str) -> None:
    run_small_query(SOURCE_SNAPSHOT_QUERY, output_dir / "source-snapshot.csv", host, password)
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.name in {"manifest.json", "README.md"} or not path.is_file():
            continue
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_host": host,
        "source_tables": [
            "cell.cellhome",
            "cell.cellpos",
            "cell.displaced",
            "cell.summary",
            "cell.coord_a0",
        ],
        "clickhouse_readonly": 1,
        "selection_metric": "exact maximum pairwise great-circle span between observed position bins",
        "minimum_span_km": 10,
        "distance_bands_km": [[10, 25], [25, 100], [100, 500], [500, 1000],
                              [1000, 5000], [5000, 10000], [10000, None]],
        "files": files,
    }
    tmp = output_dir / "manifest.json.part"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(output_dir / "manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default=os.environ.get("CELL_DB_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--only",
        choices=("all", "summary", "identities", "positions"),
        default="all",
        help="extract only one product (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="replace existing products")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get("CELL_DB_PASSWORD", "password")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    products = []
    if args.only in {"all", "summary"}:
        products.append((BAND_SUMMARY_QUERY, args.output_dir / "distance-bands.csv", False))
    if args.only in {"all", "identities"}:
        products.append((IDENTITIES_QUERY, args.output_dir / "identities.csv.zst", True))
    positions_requested = args.only in {"all", "positions"}

    for query, output, compressed in products:
        if output.exists() and not args.force:
            print(f"[skip] {output} exists (use --force to replace)")
            continue
        if compressed:
            run_compressed_query(query, output, args.host, password)
        else:
            run_small_query(query, output, args.host, password)
            print(f"[done] {output.name}: {output.stat().st_size:,} bytes")

    if positions_requested:
        identities = args.output_dir / "identities.csv.zst"
        if not identities.exists():
            raise RuntimeError("identities.csv.zst is required before extracting positions")
        candidate_index = args.output_dir / "candidate-index.tsv.zst"
        build_candidate_index(identities, candidate_index, force=args.force)
        positions = args.output_dir / "positions.csv.zst"
        if positions.exists() and not args.force:
            print(f"[skip] {positions} exists (use --force to replace)")
        else:
            run_external_positions(POSITIONS_QUERY, candidate_index, positions, args.host, password)

    write_manifest(args.output_dir, args.host, password)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
