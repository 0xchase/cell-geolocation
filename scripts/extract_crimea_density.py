#!/usr/bin/env python3
"""Export collection-wide 1 km Russian/Ukrainian density bins for Crimea.

The query is explicitly read-only and selects by a tight peninsula coordinate
extent rather than by reverse-geocoded country, avoiding a sovereignty label as
an input to the comparison.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "out-of-country" / "additional-cases"
HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
CH_PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")

R_KM = 6371.0088
KM_PER_DEG_LAT = 2 * math.pi * R_KM / 360
CELL_KM = 1.0
LON0, LON1 = 32.45, 36.75
LAT_MID = 45.25
DLAT = CELL_KM / KM_PER_DEG_LAT
DLON = CELL_KM / (KM_PER_DEG_LAT * math.cos(math.radians(LAT_MID)))
NBINS = round((LON1 - LON0) / DLON)
LAT_SPAN = NBINS * DLAT
BBOX = (LON0, LON1, LAT_MID - LAT_SPAN / 2, LAT_MID + LAT_SPAN / 2)
DATA_BBOX = (32.5, 36.7, 44.3, 46.2)


def query_grid() -> list[dict[str, str]]:
    sql = f"""
SELECT
    toInt32(floor((glat - {BBOX[2]:.10f}) / {DLAT:.10f})) AS iy,
    toInt32(floor((glon - {BBOX[0]:.10f}) / {DLON:.10f})) AS ix,
    countIf(mcc = 250) AS russian_cells,
    countIf(mcc = 255) AS ukrainian_cells
FROM cell.summary_full
WHERE cid > 0
  AND mcc IN (250, 255)
  AND NOT (glat = 0 AND glon = 0)
  AND glat BETWEEN {DATA_BBOX[2]} AND {DATA_BBOX[3]}
  AND glon BETWEEN {DATA_BBOX[0]} AND {DATA_BBOX[1]}
GROUP BY iy, ix
HAVING russian_cells + ukrainian_cells > 0
ORDER BY iy, ix
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


def export(output: Path) -> None:
    raw = query_grid()
    rows: list[dict] = []
    for row in raw:
        iy, ix = int(row["iy"]), int(row["ix"])
        if not (0 <= iy < NBINS and 0 <= ix < NBINS):
            continue
        rows.append({
            "iy": iy, "ix": ix,
            "lat": round(BBOX[2] + (iy + 0.5) * DLAT, 6),
            "lon": round(BBOX[0] + (ix + 0.5) * DLON, 6),
            "russian_cells": int(row["russian_cells"]),
            "ukrainian_cells": int(row["ukrainian_cells"]),
            "cell_km": CELL_KM,
        })
    grid_path = output / "crimea-russia-density-grid.csv"
    write_csv(grid_path, ["iy", "ix", "lat", "lon", "russian_cells", "ukrainian_cells", "cell_km"], rows)
    totals = {
        "Russian terrestrial (MCC 250)": sum(r["russian_cells"] for r in rows),
        "Ukrainian terrestrial (MCC 255)": sum(r["ukrainian_cells"] for r in rows),
    }
    summary = [
        {"group": group, "cells": cells, "period": "collection-wide", "cell_km": CELL_KM,
         "west": DATA_BBOX[0], "east": DATA_BBOX[1], "south": DATA_BBOX[2], "north": DATA_BBOX[3]}
        for group, cells in totals.items()
    ]
    summary_path = output / "crimea-russia-density-summary.csv"
    write_csv(summary_path, list(summary[0]), summary)
    print(f"[data] Russian {totals['Russian terrestrial (MCC 250)']:,}; Ukrainian {totals['Ukrainian terrestrial (MCC 255)']:,}")
    print(f"[csv] {grid_path.relative_to(ROOT)}")
    print(f"[csv] {summary_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()
