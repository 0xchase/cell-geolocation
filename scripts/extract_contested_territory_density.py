#!/usr/bin/env python3
"""Export 1 km density grids for Transnistria and Karabakh.

The queries read ``cell.summary_full`` through the project's read-only
ClickHouse wrapper.  Coordinate extents, rather than reverse-geocoded country
labels, define each panel so disputed sovereignty is not an input to the
comparison.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "out-of-country" / "contested-territories"
R_KM = 6371.0088
KM_PER_DEG_LAT = 2 * math.pi * R_KM / 360
CELL_KM = 1.0


@dataclass(frozen=True)
class Case:
    key: str
    lon0: float
    lon1: float
    lat_mid: float
    primary_sql: str
    secondary_sql: str
    primary_label: str
    secondary_label: str

    @property
    def dlat(self) -> float:
        return CELL_KM / KM_PER_DEG_LAT

    @property
    def dlon(self) -> float:
        return CELL_KM / (KM_PER_DEG_LAT * math.cos(math.radians(self.lat_mid)))

    @property
    def nbins(self) -> int:
        return round((self.lon1 - self.lon0) / self.dlon)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        lat_span = self.nbins * self.dlat
        return (
            self.lon0,
            self.lon1,
            self.lat_mid - lat_span / 2,
            self.lat_mid + lat_span / 2,
        )


CASES = {
    "transnistria": Case(
        key="transnistria",
        lon0=26.50,
        lon1=30.50,
        lat_mid=47.30,
        primary_sql="mcc=259 AND mnc=15",
        secondary_sql="mcc=259 AND mnc IN (1,2,5)",
        primary_label="Interdnestrcom (259/15)",
        secondary_label="Other Moldovan networks (259)",
    ),
    "karabakh": Case(
        key="karabakh",
        lon0=44.20,
        lon1=48.40,
        lat_mid=40.00,
        primary_sql="mcc=400",
        secondary_sql="mcc=283",
        primary_label="Azerbaijani networks (400)",
        secondary_label="Armenian networks (283)",
    ),
}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract(case: Case) -> None:
    west, east, south, north = case.bbox
    frame = ch_df(f"""
      SELECT toInt32(floor((glat-{south:.10f})/{case.dlat:.10f})) AS iy,
             toInt32(floor((glon-{west:.10f})/{case.dlon:.10f})) AS ix,
             countIf({case.primary_sql}) AS primary_cells,
             countIf({case.secondary_sql}) AS secondary_cells
      FROM cell.summary_full
      WHERE cid>0 AND NOT (glat=0 AND glon=0)
        AND glat BETWEEN {south:.10f} AND {north:.10f}
        AND glon BETWEEN {west:.10f} AND {east:.10f}
        AND (({case.primary_sql}) OR ({case.secondary_sql}))
      GROUP BY iy,ix
      HAVING primary_cells+secondary_cells>0
      ORDER BY iy,ix
    """, settings={"max_threads": 6})

    rows = []
    for row in frame.itertuples(index=False):
        iy, ix = int(row.iy), int(row.ix)
        if not (0 <= iy < case.nbins and 0 <= ix < case.nbins):
            continue
        rows.append({
            "iy": iy,
            "ix": ix,
            "lat": round(south + (iy + 0.5) * case.dlat, 6),
            "lon": round(west + (ix + 0.5) * case.dlon, 6),
            "primary_cells": int(row.primary_cells),
            "secondary_cells": int(row.secondary_cells),
            "cell_km": CELL_KM,
        })

    grid_path = OUTPUT / f"{case.key}-density-grid.csv"
    write_csv(
        grid_path,
        ["iy", "ix", "lat", "lon", "primary_cells", "secondary_cells", "cell_km"],
        rows,
    )
    totals = {
        case.primary_label: sum(row["primary_cells"] for row in rows),
        case.secondary_label: sum(row["secondary_cells"] for row in rows),
    }
    summary = [
        {
            "group": label,
            "cells": cells,
            "period": "collection-wide",
            "cell_km": CELL_KM,
            "west": west,
            "east": east,
            "south": south,
            "north": north,
        }
        for label, cells in totals.items()
    ]
    summary_path = OUTPUT / f"{case.key}-density-summary.csv"
    write_csv(summary_path, list(summary[0]), summary)
    print(f"[data] {case.key}: " + "; ".join(
        f"{label} {cells:,}" for label, cells in totals.items()
    ))


def main() -> None:
    for case in CASES.values():
        extract(case)


if __name__ == "__main__":
    main()
