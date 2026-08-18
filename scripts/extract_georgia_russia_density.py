#!/usr/bin/env python3
"""Export 1 km Georgia/Russia density data for the Ukraine progression row.

The remote ClickHouse query is explicitly read-only. Russian MCC 250
identities must resolve to Georgia and lie farther than ``--border-buffer-km``
from Russian territory; Georgian MCC 282 identities provide the domestic
comparison layer.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import math
import os
import shlex
import subprocess
from pathlib import Path

from out_of_country_mcc_matrix import BOUNDARIES, CountryBoundaries


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "out-of-country" / "additional-cases"
HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
CH_PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")

R_KM = 6371.0088
KM_PER_DEG_LAT = 2 * math.pi * R_KM / 360
CELL_KM = 1.0
LON0, LON1 = 40.0, 46.4
LAT_MID = 42.1
DLAT = CELL_KM / KM_PER_DEG_LAT
DLON = CELL_KM / (KM_PER_DEG_LAT * math.cos(math.radians(LAT_MID)))
NBINS = round((LON1 - LON0) / DLON)
LAT_SPAN = NBINS * DLAT
BBOX = (LON0, LON1, LAT_MID - LAT_SPAN / 2, LAT_MID + LAT_SPAN / 2)


def query_cells() -> list[dict[str, str]]:
    sql = f"""
SELECT mcc, mnc, lac, cid, cell_type, glat, glon
FROM cell.summary_full
WHERE country_iso = 'GE'
  AND mcc IN (250, 282)
  AND cid > 0
  AND NOT (glat = 0 AND glon = 0)
  AND glat BETWEEN {BBOX[2]:.10f} AND {BBOX[3]:.10f}
  AND glon BETWEEN {BBOX[0]:.10f} AND {BBOX[1]:.10f}
ORDER BY mcc, mnc, lac, cid, cell_type
FORMAT CSVWithNames
""".strip()
    remote = (
        "clickhouse-client "
        f"--password {shlex.quote(CH_PASSWORD)} --readonly 2 --max_threads 8 "
        "--max_execution_time 1800 --max_result_rows 0"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST, remote],
        input=sql, capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode:
        raise RuntimeError(f"ClickHouse extraction failed ({proc.returncode}):\n{proc.stderr.strip()}")
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def export(output: Path, border_buffer_km: float) -> None:
    boundaries = CountryBoundaries(BOUNDARIES)
    counts: collections.Counter[tuple[int, int, str]] = collections.Counter()
    totals: collections.Counter[str] = collections.Counter()
    rejected_border = 0
    for row in query_cells():
        mcc = int(row["mcc"])
        lat, lon = float(row["glat"]), float(row["glon"])
        group = "russian_cells" if mcc == 250 else "georgian_cells"
        if mcc == 250:
            distance = boundaries.distance_to_boundary_km("RU", lon, lat, border_buffer_km)
            if boundaries.contains("RU", lon, lat) or distance <= border_buffer_km:
                rejected_border += 1
                continue
        ix = math.floor((lon - BBOX[0]) / DLON)
        iy = math.floor((lat - BBOX[2]) / DLAT)
        if 0 <= ix < NBINS and 0 <= iy < NBINS:
            counts[(iy, ix, group)] += 1
            totals[group] += 1

    grid_rows: list[dict] = []
    occupied = sorted({(iy, ix) for iy, ix, _ in counts})
    for iy, ix in occupied:
        grid_rows.append({
            "iy": iy, "ix": ix,
            "lat": round(BBOX[2] + (iy + 0.5) * DLAT, 6),
            "lon": round(BBOX[0] + (ix + 0.5) * DLON, 6),
            "russian_cells": counts[(iy, ix, "russian_cells")],
            "georgian_cells": counts[(iy, ix, "georgian_cells")],
            "cell_km": CELL_KM,
            "border_buffer_km": border_buffer_km,
        })
    write_csv(
        output / "georgia-russia-density-grid.csv",
        ["iy", "ix", "lat", "lon", "russian_cells", "georgian_cells", "cell_km", "border_buffer_km"],
        grid_rows,
    )
    summary_rows = [
        {"group": "Russian terrestrial (MCC 250)", "cells": totals["russian_cells"],
         "period": "collection-wide", "cell_km": CELL_KM, "border_buffer_km": border_buffer_km},
        {"group": "Georgian terrestrial (MCC 282)", "cells": totals["georgian_cells"],
         "period": "collection-wide", "cell_km": CELL_KM, "border_buffer_km": 0},
    ]
    write_csv(output / "georgia-russia-density-summary.csv", list(summary_rows[0]), summary_rows)
    print(f"[data] Russian {totals['russian_cells']:,}; Georgian {totals['georgian_cells']:,}; removed near border {rejected_border:,}")
    print(f"[csv] {(output / 'georgia-russia-density-grid.csv').relative_to(ROOT)}")
    print(f"[csv] {(output / 'georgia-russia-density-summary.csv').relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--border-buffer-km", type=float, default=25.0)
    args = parser.parse_args()
    export(args.output, args.border_buffer_km)


if __name__ == "__main__":
    main()
