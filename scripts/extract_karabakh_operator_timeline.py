#!/usr/bin/env python3
"""Export quarterly active identities in a fixed Nagorno-Karabakh core box.

The coordinate box avoids using a disputed reverse-geocoded sovereignty label
as an input.  The remote ClickHouse query is explicitly read-only.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "data"
    / "out-of-country"
    / "contested-territories"
    / "karabakh-quarterly-active.csv"
)
HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
CH_PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")
BBOX = (46.55, 47.00, 39.65, 40.10)


def query_rows() -> list[dict[str, str]]:
    west, east, south, north = BBOX
    sql = f"""
WITH core AS (
    SELECT mcc, mnc, lac, cid, cell_type
    FROM cell.summary_full
    WHERE cid > 0
      AND glat BETWEEN {south} AND {north}
      AND glon BETWEEN {west} AND {east}
      AND mcc IN (283, 400)
)
SELECT
    toDate(toStartOfQuarter(g.timestamp)) AS quarter,
    if(g.mcc = 283, 'Armenian', 'Azerbaijani') AS network,
    uniqExact((g.mcc, g.mnc, g.lac, g.cid, g.cell_type)) AS active_cells
FROM cell.geos AS g
INNER JOIN core USING (mcc, mnc, lac, cid, cell_type)
WHERE NOT (g.lat = 0 AND g.lon = 0)
GROUP BY quarter, network
ORDER BY quarter, network
FORMAT CSVWithNames
""".strip()
    remote = (
        "clickhouse-client "
        f"--password {shlex.quote(CH_PASSWORD)} --readonly 2 --max_threads 8 "
        "--max_execution_time 1800 --optimize_aggregation_in_order 1"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST, remote],
        input=sql,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode:
        raise RuntimeError(
            f"ClickHouse extraction failed ({proc.returncode}):\n{proc.stderr.strip()}"
        )
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def export(output: Path) -> None:
    rows = query_rows()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["quarter", "network", "active_cells", "west", "east", "south", "north"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "quarter": row["quarter"],
                "network": row["network"],
                "active_cells": int(row["active_cells"]),
                "west": BBOX[0],
                "east": BBOX[1],
                "south": BBOX[2],
                "north": BBOX[3],
            })
    print(f"[csv] {output.relative_to(ROOT)} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()
