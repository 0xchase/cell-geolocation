#!/usr/bin/env python3
"""Export publication inputs for the northern-Syria and West-Bank case studies.

All database access is read-only.  The resulting CSVs are deliberately more
aggregated than the underlying corpus: maps use counts of distinct cellular
identities per square kilometre bin, while timelines use distinct identities
per month.  Observation counts are never used because they measure crawler
polling cadence rather than network size.

The plotting companion reads only these CSVs, making every value used in a
figure inspectable under ``data/out-of-country/``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "out-of-country"
HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")

# Map extents are (west, east, south, north).  The Syria crop covers the
# Turkish-controlled northwest and the Kurdish northeast without spending half
# the panel on southern Syria.  The West Bank crop excludes Gaza.
SYRIA_BBOX = (35.50, 42.50, 35.00, 37.40)
WEST_BANK_BBOX = (34.82, 35.62, 31.30, 32.62)
GAZA_BBOX = (34.20, 34.60, 31.20, 31.60)
JORDAN_BBOX = (34.80, 36.20, 29.10, 33.00)

# Approximately square cells at each map's midpoint latitude.
SYRIA_DLAT, SYRIA_DLON = 0.03604, 0.04465   # about 4 km
WB_DLAT, WB_DLON = 0.00901, 0.01060         # about 1 km
SYRIA_FULL_DLAT, SYRIA_FULL_DLON = 0.004505, 0.00558  # about 500 m
WB_FULL_DLAT, WB_FULL_DLON = 0.0022525, 0.00265       # about 250 m
GAZA_FULL_DLAT, GAZA_FULL_DLON = 0.00112625, 0.001325  # about 125 m
JORDAN_FULL_DLAT, JORDAN_FULL_DLON = 0.0022525, 0.00265  # about 250 m


def run_csv(name: str, sql: str, output: Path, *, skip_existing: bool = False) -> None:
    """Run one ClickHouse query through SSH and atomically write its CSV."""
    if skip_existing and output.exists():
        print(f"[skip] {output.relative_to(ROOT)}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    remote = (
        "clickhouse-client "
        f"--password {shlex.quote(PASSWORD)} --readonly 2 --max_threads 8 "
        "--max_execution_time 1800 --optimize_aggregation_in_order 1"
    )
    query = sql.strip().rstrip(";") + "\nFORMAT CSVWithNames\n"
    print(f"[query] {name} -> {output.relative_to(ROOT)}", flush=True)
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST, remote],
        input=query,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode:
        raise RuntimeError(f"{name} failed ({proc.returncode}):\n{proc.stderr.strip()}")
    tmp.write_text(proc.stdout, encoding="utf-8")
    tmp.replace(output)
    rows = max(0, proc.stdout.count("\n") - 1)
    print(f"[data] {rows:,} rows", flush=True)


def map_query(
    bbox: tuple[float, float, float, float],
    dlat: float,
    dlon: float,
    identity_filter: str,
    observation_filter: str,
    count_expressions: list[tuple[str, str]],
) -> str:
    west, east, south, north = bbox
    counts = ",\n       ".join(
        f"uniqExactIf((g.mcc,g.mnc,g.lac,g.cid,g.cell_type), {condition}) AS {name}"
        for name, condition in count_expressions
    )
    nonzero = " + ".join(name for name, _ in count_expressions)
    return f"""
SELECT toYear(g.timestamp) AS year,
       toInt32(floor((g.lat - {south}) / {dlat})) AS iy,
       toInt32(floor((g.lon - {west}) / {dlon})) AS ix,
       {west} + (ix + 0.5) * {dlon} AS lon,
       {south} + (iy + 0.5) * {dlat} AS lat,
       {counts}
FROM cell.geos AS g
INNER JOIN
(
    SELECT mcc,mnc,lac,cid,cell_type
    FROM cell.summary_full
    WHERE cid > 0 AND {identity_filter}
) AS s USING (mcc,mnc,lac,cid,cell_type)
WHERE g.cid > 0 AND NOT (g.lat = 0 AND g.lon = 0)
  AND toYear(g.timestamp) BETWEEN 2024 AND 2026
  AND g.lon BETWEEN {west} AND {east}
  AND g.lat BETWEEN {south} AND {north}
  AND ({observation_filter})
GROUP BY year,iy,ix
HAVING {nonzero} > 0
ORDER BY year,iy,ix
"""


def temporal_query(
    bbox: tuple[float, float, float, float],
    identity_filter: str,
    observation_filter: str,
    count_expressions: list[tuple[str, str]],
    period: str,
) -> str:
    west, east, south, north = bbox
    counts = ",\n       ".join(
        f"uniqExactIf((g.mcc,g.mnc,g.lac,g.cid,g.cell_type), {condition}) AS {name}"
        for name, condition in count_expressions
    )
    period_expr = "toStartOfMonth(g.timestamp)" if period == "month" else "toYear(g.timestamp)"
    period_alias = "month" if period == "month" else "year"
    return f"""
SELECT {period_expr} AS {period_alias},
       {counts}
FROM cell.geos AS g
INNER JOIN
(
    SELECT mcc,mnc,lac,cid,cell_type
    FROM cell.summary_full
    WHERE cid > 0 AND {identity_filter}
) AS s USING (mcc,mnc,lac,cid,cell_type)
WHERE g.cid > 0 AND NOT (g.lat = 0 AND g.lon = 0)
  AND toYear(g.timestamp) BETWEEN 2024 AND 2026
  AND g.lon BETWEEN {west} AND {east}
  AND g.lat BETWEEN {south} AND {north}
  AND ({observation_filter})
GROUP BY {period_alias}
ORDER BY {period_alias}
"""


def aggregate_map_query(
    bbox: tuple[float, float, float, float],
    dlat: float,
    dlon: float,
    identity_filter: str,
    observation_filter: str,
    count_expressions: list[tuple[str, str]],
) -> str:
    """Map distinct identities once across the complete collection period."""
    west, east, south, north = bbox
    counts = ",\n       ".join(
        f"uniqExactIf((g.mcc,g.mnc,g.lac,g.cid,g.cell_type), {condition}) AS {name}"
        for name, condition in count_expressions
    )
    nonzero = " + ".join(name for name, _ in count_expressions)
    return f"""
SELECT toInt32(floor((g.lat - {south}) / {dlat})) AS iy,
       toInt32(floor((g.lon - {west}) / {dlon})) AS ix,
       {west} + (ix + 0.5) * {dlon} AS lon,
       {south} + (iy + 0.5) * {dlat} AS lat,
       {counts}
FROM cell.geos AS g
INNER JOIN
(
    SELECT mcc,mnc,lac,cid,cell_type
    FROM cell.summary_full
    WHERE cid > 0 AND {identity_filter}
) AS s USING (mcc,mnc,lac,cid,cell_type)
WHERE g.cid > 0 AND NOT (g.lat = 0 AND g.lon = 0)
  AND toYear(g.timestamp) BETWEEN 2024 AND 2026
  AND g.lon BETWEEN {west} AND {east}
  AND g.lat BETWEEN {south} AND {north}
  AND ({observation_filter})
GROUP BY iy,ix
HAVING {nonzero} > 0
ORDER BY iy,ix
"""


def latest_map_query(
    bbox: tuple[float, float, float, float],
    dlat: float,
    dlon: float,
    identity_filter: str,
    count_expressions: list[tuple[str, str]],
) -> str:
    """Map each distinct identity once, at its latest valid position."""
    west, east, south, north = bbox
    counts = ",\n       ".join(
        "uniqExactIf((s.mcc,s.mnc,s.lac,s.cid,s.cell_type), "
        f"{condition.replace('g.', 's.')}) AS {name}"
        for name, condition in count_expressions
    )
    nonzero = " + ".join(name for name, _ in count_expressions)
    return f"""
SELECT toInt32(floor((s.glat - {south}) / {dlat})) AS iy,
       toInt32(floor((s.glon - {west}) / {dlon})) AS ix,
       {west} + (ix + 0.5) * {dlon} AS lon,
       {south} + (iy + 0.5) * {dlat} AS lat,
       {counts}
FROM cell.summary_full AS s
WHERE s.cid > 0 AND NOT (s.glat = 0 AND s.glon = 0)
  AND s.glon BETWEEN {west} AND {east}
  AND s.glat BETWEEN {south} AND {north}
  AND ({identity_filter})
GROUP BY iy,ix
HAVING {nonzero} > 0
ORDER BY iy,ix
"""


def extract_syria(root: Path, *, skip_existing: bool = False) -> None:
    # Restrict the identity population by its latest reverse-geocoded country,
    # then retain every historical position inside the northern-Syria crop.
    identity_filter = (
        "country_iso = 'SY' AND (mcc IN (286,417,418) OR (mcc=460 AND mnc=0))"
    )
    groups = [
        ("syrian_cells", "g.mcc=417"),
        ("turkish_cells", "g.mcc=286"),
        ("iraqi_cells", "g.mcc=418"),
        ("china_cells", "g.mcc=460 AND g.mnc=0"),
    ]
    observation_filter = "g.mcc IN (286,417,418) OR (g.mcc=460 AND g.mnc=0)"
    run_csv(
        "northern Syria annual grid",
        map_query(SYRIA_BBOX, SYRIA_DLAT, SYRIA_DLON, identity_filter, observation_filter, groups),
        root / "northern-syria-grid-year.csv",
        skip_existing=skip_existing,
    )
    run_csv(
        "northern Syria collection-wide 1 km grid",
        latest_map_query(
            SYRIA_BBOX, SYRIA_FULL_DLAT, SYRIA_FULL_DLON,
            identity_filter, groups,
        ),
        root / "northern-syria-grid.csv",
        skip_existing=skip_existing,
    )
    run_csv(
        "northern Syria annual totals",
        temporal_query(SYRIA_BBOX, identity_filter, observation_filter, groups, "year"),
        root / "northern-syria-annual.csv",
        skip_existing=skip_existing,
    )
    run_csv(
        "northern Syria monthly totals",
        temporal_query(SYRIA_BBOX, identity_filter, observation_filter, groups, "month"),
        root / "northern-syria-monthly.csv",
        skip_existing=skip_existing,
    )


def extract_west_bank(root: Path, *, skip_existing: bool = False) -> None:
    # MCC 425 contains both Palestinian and Israeli networks.  Restrict the
    # latest-position map to identities whose latest resolved territory is
    # Palestine so the rectangular crop does not fill with surrounding Israel.
    identity_filter = "country_iso='PS' AND mcc IN (425,416,602,415,420,424,426,427,286,280,606)"
    groups = [
        ("palestinian_cells", "g.mcc=425 AND g.mnc IN (5,6)"),
        ("israeli_cells", "g.mcc=425 AND g.mnc NOT IN (5,6)"),
        ("egyptian_cells", "g.mcc=602"),
        ("jordanian_cells", "g.mcc=416"),
        ("other_foreign_cells", "g.mcc IN (415,420,424,426,427,286,280,606)"),
    ]
    observation_filter = "g.mcc IN (425,416,602,415,420,424,426,427,286,280,606)"
    run_csv(
        "West Bank annual grid",
        map_query(WEST_BANK_BBOX, WB_DLAT, WB_DLON, identity_filter, observation_filter, groups),
        root / "west-bank-grid-year.csv",
        skip_existing=skip_existing,
    )
    run_csv(
        "West Bank collection-wide 1 km grid",
        latest_map_query(
            WEST_BANK_BBOX, WB_FULL_DLAT, WB_FULL_DLON,
            identity_filter, groups,
        ),
        root / "west-bank-grid.csv",
        skip_existing=skip_existing,
    )
    run_csv(
        "West Bank annual totals",
        temporal_query(WEST_BANK_BBOX, identity_filter, observation_filter, groups, "year"),
        root / "west-bank-annual.csv",
        skip_existing=skip_existing,
    )
    run_csv(
        "West Bank monthly totals",
        temporal_query(WEST_BANK_BBOX, identity_filter, observation_filter, groups, "month"),
        root / "west-bank-monthly.csv",
        skip_existing=skip_existing,
    )

    west, east, south, north = WEST_BANK_BBOX
    run_csv(
        "West Bank operator composition",
        f"""
SELECT toYear(g.timestamp) AS year,g.mnc,
       uniqExact((g.mcc,g.mnc,g.lac,g.cid,g.cell_type)) AS cells
FROM cell.geos AS g
INNER JOIN
(
    SELECT mcc,mnc,lac,cid,cell_type
    FROM cell.summary_full
    WHERE cid > 0 AND {identity_filter}
) AS s USING (mcc,mnc,lac,cid,cell_type)
WHERE g.cid > 0 AND NOT (g.lat=0 AND g.lon=0)
  AND toYear(g.timestamp) BETWEEN 2024 AND 2026
  AND g.lon BETWEEN {west} AND {east}
  AND g.lat BETWEEN {south} AND {north}
  AND g.mcc=425
GROUP BY year,g.mnc
ORDER BY year,cells DESC
""",
        root / "west-bank-operator-year.csv",
        skip_existing=skip_existing,
    )


def extract_gaza(root: Path, *, skip_existing: bool = False) -> None:
    # Use the same latest-position population rule as the West Bank panel, but
    # retain every country-assigned MCC so the residual foreign group is not
    # restricted to a hand-selected list. Non-country/test MCCs are excluded.
    identity_filter = "country_iso='PS' AND mcc NOT IN (1,69,526,901,999)"
    groups = [
        ("palestinian_cells", "g.mcc=425 AND g.mnc IN (5,6)"),
        ("israeli_cells", "g.mcc=425 AND g.mnc NOT IN (5,6)"),
        ("egyptian_cells", "g.mcc=602"),
        ("libyan_cells", "g.mcc=606"),
        ("saudi_cells", "g.mcc=420"),
        ("turkish_cells", "g.mcc=286"),
        ("jordanian_cells", "g.mcc=416"),
        ("cypriot_cells", "g.mcc=280"),
        ("emirati_cells", "g.mcc=424"),
        ("other_country_cells", "g.mcc NOT IN (425,602,606,420,286,416,280,424,1,69,526,901,999)"),
    ]
    run_csv(
        "Gaza collection-wide latest-position 125 m grid",
        latest_map_query(
            GAZA_BBOX, GAZA_FULL_DLAT, GAZA_FULL_DLON,
            identity_filter, groups,
        ),
        root / "gaza-grid.csv",
        skip_existing=skip_existing,
    )


def extract_jordan(root: Path, *, skip_existing: bool = False) -> None:
    identity_filter = "country_iso='JO' AND mcc IN (416,425)"
    groups = [
        ("jordanian_cells", "g.mcc=416"),
        ("israeli_cells", "g.mcc=425"),
    ]
    run_csv(
        "Jordan collection-wide latest-position 250 m grid",
        latest_map_query(
            JORDAN_BBOX, JORDAN_FULL_DLAT, JORDAN_FULL_DLON,
            identity_filter, groups,
        ),
        root / "jordan-grid.csv",
        skip_existing=skip_existing,
    )


def extract_aliasing(root: Path, *, skip_existing: bool = False) -> None:
    alias = root / "aliasing"
    selected = {
        "DE": (262, "232,230,260,206,208,204,270,238,222"),
        "AT": (232, "216,230,231,293,262,222"),
    }
    unions = []
    for host, (local_mcc, foreign_mccs) in selected.items():
        unions.append(f"""
SELECT '{host}' AS located_iso,f.mcc,f.mnc,count() AS foreign_cells,
       countIf(foreign_cells_same_suffix > 0) AS identities_with_local_suffix,
       round(100 * identities_with_local_suffix / foreign_cells,2) AS matched_pct,
       round(quantileExactIf(0.5)(nearest_km,foreign_cells_same_suffix > 0),2) AS median_nearest_km,
       round(quantileExactIf(0.9)(nearest_km,foreign_cells_same_suffix > 0),2) AS p90_nearest_km
FROM
(
    SELECT f.mcc,f.mnc,f.lac,f.cid,f.cell_type,
           countIf(n.cid > 0) AS foreign_cells_same_suffix,
           minIf(greatCircleDistance(f.glon,f.glat,n.glon,n.glat)/1000,
                 n.cid > 0) AS nearest_km
    FROM cell.summary_full AS f
    LEFT JOIN cell.summary_full AS n
      ON f.lac=n.lac AND f.cid=n.cid AND f.cell_type=n.cell_type
     AND n.country_iso='{host}' AND n.mcc={local_mcc} AND n.cid>0
    WHERE f.country_iso='{host}' AND f.mcc IN ({foreign_mccs}) AND f.cid>0
    GROUP BY f.mcc,f.mnc,f.lac,f.cid,f.cell_type
) AS f
GROUP BY f.mcc,f.mnc
""")
    run_csv(
        "Germany/Austria local-identifier aliasing",
        "\nUNION ALL\n".join(unions) + "\nORDER BY located_iso,foreign_cells DESC",
        alias / "germany-austria-local-suffix-matches.csv",
        skip_existing=skip_existing,
    )

    run_csv(
        "Germany/Austria aliasing technology breakdown",
        """
SELECT country_iso AS located_iso,mcc,mnc,toString(cell_type) AS technology,
       count() AS cells,countIf(n_pos=1) AS single_position_cells,
       countIf(n_pos>1) AS multiple_position_cells,
       min(first_seen) AS first_seen,max(last_seen) AS last_seen
FROM cell.summary_full
WHERE cid>0 AND
 ((country_iso='DE' AND mcc IN (232,230,260,206,208,204,270,238,222)) OR
  (country_iso='AT' AND mcc IN (216,230,231,293,262,222)))
GROUP BY located_iso,mcc,mnc,technology
ORDER BY located_iso,cells DESC
""",
        alias / "germany-austria-foreign-plmn-technology.csv",
        skip_existing=skip_existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--case", choices=("all", "syria", "west-bank", "gaza", "jordan", "aliasing"), default="all"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="keep completed CSVs and query only missing outputs",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case in ("all", "syria"):
        extract_syria(args.output, skip_existing=args.skip_existing)
    if args.case in ("all", "west-bank"):
        extract_west_bank(args.output, skip_existing=args.skip_existing)
    if args.case in ("all", "gaza"):
        extract_gaza(args.output, skip_existing=args.skip_existing)
    if args.case in ("all", "jordan"):
        extract_jordan(args.output, skip_existing=args.skip_existing)
    if args.case in ("all", "aliasing"):
        extract_aliasing(args.output, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
