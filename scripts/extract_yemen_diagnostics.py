#!/usr/bin/env python3
"""Export cell-level and aggregate CSVs for the Yemen PLMN diagnostic figure.

The remote ClickHouse query is explicitly read-only.  Foreign identities are
retained only when their latest position reverse-geocodes to Yemen and lies
farther than ``--border-buffer-km`` from the territory assigned to their MCC.
Yemen-coded identities provide the local spatial and lifetime comparison.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import math
import os
import shlex
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from out_of_country_mcc_matrix import (
    BOUNDARIES,
    MCC_TABLE,
    CountryBoundaries,
    NAME_OVERRIDES,
    home_iso_sql,
    load_mcc_reference,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "out-of-country" / "yemen"
HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
CH_PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")

GRID_DEGREES = 0.002

GOVERNORATES = {
    "أمانة العاصمة": "Sana'a City",
    "محافظة أبين": "Abyan",
    "محافظة إب": "Ibb",
    "محافظة البيضاء": "Al Bayda",
    "محافظة الجوف": "Al Jawf",
    "محافظة الحديدة": "Al Hudaydah",
    "محافظة الضالع": "Al Dhale'e",
    "محافظة المحويت": "Al Mahwit",
    "محافظة المهرة": "Al Mahrah",
    "محافظة تعز": "Taiz",
    "محافظة حجة": "Hajjah",
    "محافظة حضرموت": "Hadramawt",
    "محافظة ذمار": "Dhamar",
    "محافظة ريمة": "Raymah",
    "محافظة شبوة": "Shabwah",
    "محافظة صعدة": "Saada",
    "محافظة صنعاء": "Sana'a",
    "محافظة عدن": "Aden",
    "محافظة عمران": "Amran",
    "محافظة لحج": "Lahij",
    "محافظة مأرب": "Marib",
    "منطقة جازان": "Jizan (border)",
    "منطقة عسير": "Asir (border)",
}

FAMILY_ISOS = {
    "Gulf": {"AE", "SA", "OM", "QA", "BH"},
    "Horn of Africa": {"SO", "ER", "DJ", "ET"},
    "North/West Africa": {"NE", "DZ", "SD", "ML", "BF", "LY", "MR"},
}


def family_for(home_iso: str) -> str:
    if home_iso == "YE":
        return "Yemen MCC"
    for family, members in FAMILY_ISOS.items():
        if home_iso in members:
            return family
    return "Other foreign"


def query_cells() -> list[dict[str, str]]:
    ref = load_mcc_reference(MCC_TABLE)
    home = home_iso_sql(ref)
    sql = f"""
SELECT home_iso, mcc, mnc, lac, cid, cell_type, first_seen, last_seen,
       obs, glat, glon, state
FROM
(
    SELECT {home} AS home_iso, mcc, mnc, lac, cid, cell_type,
           first_seen, last_seen, obs, glat, glon, state
    FROM cell.summary_full
    WHERE country_iso = 'YE'
      AND cid > 0
      AND NOT (glat = 0 AND glon = 0)
)
WHERE home_iso != ''
ORDER BY home_iso, mcc, mnc, lac, cid, cell_type
FORMAT CSVWithNames
""".strip()
    remote = (
        "clickhouse-client "
        f"--password {shlex.quote(CH_PASSWORD)} --readonly 2 --max_threads 8 "
        "--max_execution_time 1800 --max_result_rows 0"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST, remote],
        input=sql,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode:
        raise RuntimeError(f"ClickHouse extraction failed ({proc.returncode}):\n{proc.stderr.strip()}")
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def country_name(home_iso: str, names: dict[str, str]) -> str:
    return NAME_OVERRIDES.get(home_iso, names.get(home_iso, home_iso))


def parse_span_days(row: dict[str, str]) -> int:
    first = datetime.fromisoformat(row["first_seen"])
    last = datetime.fromisoformat(row["last_seen"])
    return max(0, (last - first).days)


def bin_center(value: float) -> float:
    return round(round(value / GRID_DEGREES) * GRID_DEGREES, 3)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(output: Path, border_buffer_km: float) -> None:
    ref = load_mcc_reference(MCC_TABLE)
    boundaries = CountryBoundaries(BOUNDARIES)
    raw = query_cells()
    retained: list[dict] = []
    rejected_border = 0
    rejected_home = 0

    for row in raw:
        home_iso = row["home_iso"]
        lat, lon = float(row["glat"]), float(row["glon"])
        distance: float | None = None
        if home_iso != "YE":
            if not boundaries.has_geometry(home_iso):
                rejected_home += 1
                continue
            if boundaries.contains(home_iso, lon, lat):
                rejected_home += 1
                continue
            distance = boundaries.distance_to_boundary_km(home_iso, lon, lat, border_buffer_km)
            if distance <= border_buffer_km:
                rejected_border += 1
                continue
        span_days = parse_span_days(row)
        retained.append({
            "home_iso": home_iso,
            "home_country": country_name(home_iso, ref.names),
            "family": family_for(home_iso),
            "is_foreign": int(home_iso != "YE"),
            "mcc": int(row["mcc"]),
            "mnc": int(row["mnc"]),
            "lac": int(row["lac"]),
            "cid": int(row["cid"]),
            "cell_type": row["cell_type"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "span_days": span_days,
            "observations": int(row["obs"]),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "governorate": GOVERNORATES.get(row["state"], row["state"] or "Unresolved"),
            "lac_294xx": int(row["cell_type"] == "gsm" and 29400 <= int(row["lac"]) <= 29499),
            "distance_to_home_km": "" if distance is None or math.isinf(distance) else round(distance, 2),
            "border_buffer_km": border_buffer_km,
        })

    retained.sort(key=lambda r: (r["is_foreign"], r["home_iso"], r["mcc"], r["mnc"], r["lac"], r["cid"], r["cell_type"]))
    identity_fields = list(retained[0])
    write_csv(output / "yemen-cell-identities.csv", identity_fields, retained)

    map_counts: collections.Counter[tuple] = collections.Counter()
    for row in retained:
        map_counts[(bin_center(row["lat"]), bin_center(row["lon"]), row["home_iso"],
                    row["home_country"], row["family"], row["is_foreign"])] += 1
    map_rows = [
        {"lat": key[0], "lon": key[1], "home_iso": key[2], "home_country": key[3],
         "family": key[4], "is_foreign": key[5], "cells": cells,
         "grid_degrees": GRID_DEGREES, "border_buffer_km": border_buffer_km}
        for key, cells in sorted(map_counts.items())
    ]
    write_csv(output / "yemen-map-grid.csv", list(map_rows[0]), map_rows)

    gov_country: collections.Counter[tuple[str, str, str, str, int]] = collections.Counter()
    for row in retained:
        gov_country[(row["governorate"], row["home_iso"], row["home_country"],
                     row["family"], row["is_foreign"])] += 1
    gov_rows = [
        {"governorate": key[0], "home_iso": key[1], "home_country": key[2],
         "family": key[3], "is_foreign": key[4], "cells": cells}
        for key, cells in sorted(gov_country.items())
    ]
    write_csv(output / "yemen-governorate-country.csv", list(gov_rows[0]), gov_rows)

    by_country: dict[str, list[dict]] = collections.defaultdict(list)
    for row in retained:
        by_country[row["home_iso"]].append(row)
    country_rows: list[dict] = []
    for iso, rows in by_country.items():
        gsm = [r for r in rows if r["cell_type"] == "gsm"]
        country_rows.append({
            "home_iso": iso,
            "home_country": rows[0]["home_country"],
            "family": rows[0]["family"],
            "is_foreign": rows[0]["is_foreign"],
            "cells": len(rows),
            "gsm_cells": len(gsm),
            "lac_294xx_cells": sum(r["lac_294xx"] for r in gsm),
            "lac_294xx_share": round(sum(r["lac_294xx"] for r in gsm) / len(gsm), 5) if gsm else "",
            "median_span_days": round(statistics.median(r["span_days"] for r in rows), 1),
        })
    country_rows.sort(key=lambda r: (-r["cells"], r["home_iso"]))
    write_csv(output / "yemen-country-summary.csv", list(country_rows[0]), country_rows)

    lifetime_counts: collections.Counter[tuple[str, int]] = collections.Counter()
    for row in retained:
        if row["cell_type"] == "gsm":
            population = "Foreign-coded GSM" if row["is_foreign"] else "Yemen-coded GSM"
            lifetime_counts[(population, row["span_days"])] += 1
    lifetime_rows = [
        {"population": population, "span_days": days, "cells": cells}
        for (population, days), cells in sorted(lifetime_counts.items())
    ]
    write_csv(output / "yemen-lifetime-distribution.csv", list(lifetime_rows[0]), lifetime_rows)

    local = sum(not r["is_foreign"] for r in retained)
    foreign = sum(r["is_foreign"] for r in retained)
    print(f"[data] {foreign:,} foreign and {local:,} Yemen-coded identities")
    print(f"[filter] removed {rejected_border:,} within {border_buffer_km:g} km; {rejected_home:,} inside/missing home territory")
    for path in sorted(output.glob("*.csv")):
        print(f"[csv] {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--border-buffer-km", type=float, default=25.0)
    args = parser.parse_args()
    export(args.output, args.border_buffer_km)


if __name__ == "__main__":
    main()
