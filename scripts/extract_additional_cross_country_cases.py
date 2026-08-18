#!/usr/bin/env python3
"""Export map-ready CSVs for additional 25 km-buffered MCC case studies.

This script reads the aggregate cache produced by ``out_of_country_mcc_matrix``
and applies exactly the same Natural Earth containment and distance-to-home-
territory tests.  It never writes to or queries the remote database.  Each CSV
contains 0.01-degree latest-position bins and distinct-identity counts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

from out_of_country_mcc_matrix import (
    BOUNDARIES,
    MCC_TABLE,
    CountryBoundaries,
    cache_path,
    load_mcc_reference,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "out-of-country" / "additional-cases"

def case_for(host: str, home: str) -> str | None:
    if not host:
        return None
    if home == "JM" and host != "JM":
        return "caribbean-jamaica"
    if (host, home) == ("XK", "MC"):
        return "kosovo-monaco"
    if (host, home) == ("GE", "RU"):
        return "georgia-russia"
    if host == "YE":
        return "yemen-foreign"
    if host == "MM" and home in {"HK", "CN", "MO"}:
        return "myanmar-foreign"
    if (host, home) == ("JO", "IL"):
        return "jordan-israel"
    if (host, home) in {("CY", "TR"), ("TR", "CY")}:
        return "cyprus-turkiye"
    if (host, home) == ("AZ", "AM"):
        return "azerbaijan-armenia"
    return None


def export_cases(output: Path, border_buffer_km: float) -> None:
    ref = load_mcc_reference(MCC_TABLE)
    source = cache_path(ref)
    if not source.exists():
        raise FileNotFoundError(
            f"missing {source}; run scripts/out_of_country_mcc_matrix.py first"
        )
    boundaries = CountryBoundaries(BOUNDARIES)
    selected: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    distance_cache: dict[tuple[str, float, float], float] = {}

    with gzip.open(source, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            home = row["home_iso"]
            lat, lon = float(row["lat"]), float(row["lon"])
            if not boundaries.has_geometry(home):
                continue
            host = boundaries.classify_host(lon, lat, row["host_raw"])
            case = case_for(host, home)
            if case is None or host == home or boundaries.contains(home, lon, lat):
                continue
            dkey = (home, lon, lat)
            if dkey not in distance_cache:
                distance_cache[dkey] = boundaries.distance_to_boundary_km(
                    home, lon, lat, border_buffer_km
                )
            distance = distance_cache[dkey]
            if distance <= border_buffer_km:
                continue
            selected[case].append({
                "located_iso": host,
                "mcc_country_iso": home,
                "mcc": int(row["mcc"]),
                "mnc": int(row["mnc"]),
                "lat": lat,
                "lon": lon,
                "cells": int(row["cells"]),
                "distance_to_home_km": round(distance, 2),
            })

    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "located_iso", "mcc_country_iso", "mcc", "mnc", "lat", "lon",
        "cells", "distance_to_home_km",
    ]
    for case in sorted(selected):
        path = output / f"{case}.csv"
        case_rows = sorted(
            selected[case],
            key=lambda r: (str(r["located_iso"]), str(r["mcc_country_iso"]),
                           float(r["lat"]), float(r["lon"]), int(r["mcc"]), int(r["mnc"])),
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(case_rows)
        total = sum(int(r["cells"]) for r in case_rows)
        print(f"[data] {path.relative_to(ROOT)}: {len(case_rows):,} bins, {total:,} identities")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--border-buffer-km", type=float, default=25.0)
    args = parser.parse_args()
    export_cases(args.output, args.border_buffer_km)


if __name__ == "__main__":
    main()
