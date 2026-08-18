#!/usr/bin/env python3
"""Build the reproducible satellite/NTN candidate data snapshot.

The remote ClickHouse database is queried read-only.  The extraction combines
the literature registry in ``data/satellite`` with a global, operator-agnostic
screen for recent LTE PLMNs having the repeated CID structure seen in known
Starlink Direct-to-Cell populations.

Run from the repository root:

    uv run --with pandas scripts/extract_satellite_candidates.py

Identity-level CSVs are gzip-compressed and split by PLMN to keep individual
files manageable.  ``--refresh-cells`` replaces those generated snapshots.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD_REGISTRY = ROOT / "data" / "satellite"
OUTPUT = ROOT / "data" / "satellites"
CELLS = OUTPUT / "cells"
SNAPSHOT_DATE = "2026-08-17"

sys.path.insert(0, str(ROOT / "scripts"))
from ch_remote import (  # noqa: E402
    CH_PASSWORD,
    DEFAULT_SETTINGS,
    HOST,
    _SSH_OPTS,
    ch_df,
)


SCREEN_QUERY = r"""
SELECT
    mcc,
    mnc,
    countIf(cell_type = 'lte' AND cid > 0) AS lte_identities,
    sumIf(obs, cell_type = 'lte' AND cid > 0) AS lte_observations,
    uniqExactIf(lac, cell_type = 'lte' AND cid > 0) AS lte_tacs,
    uniqExactIf(intDiv(cid, 256), cell_type = 'lte' AND cid > 0) AS cid_high_values,
    uniqExactIf(bitAnd(cid, 255), cell_type = 'lte' AND cid > 0) AS cid_low_values,
    minIf(first_seen, cell_type = 'lte' AND cid > 0) AS first_seen,
    maxIf(last_seen, cell_type = 'lte' AND cid > 0) AS last_seen,
    countIf(cell_type != 'lte' AND cid > 0) AS non_lte_identities,
    uniqExactIf(country_iso, cell_type = 'lte' AND cid > 0 AND country_iso != '')
        AS located_countries
FROM cell.summary_full
GROUP BY mcc, mnc
HAVING lte_identities >= 10
ORDER BY lte_identities DESC, mcc, mnc
"""


@dataclass(frozen=True)
class BehavioralCandidate:
    mcc: int
    mnc: int
    mnc_length: int
    assignee: str
    country_iso: str
    country: str
    tier: str
    system: str
    relationship: str
    source_ids: str
    notes: str = ""

    @property
    def plmn(self) -> str:
        return f"{self.mcc}-{self.mnc:0{self.mnc_length}d}"


# Every PLMN returned by the global signature screen on the snapshot date.
# Tiering is deliberately provisional; the next analysis pass can revise it
# without rerunning the database extraction.
BEHAVIORAL_CANDIDATES = [
    BehavioralCandidate(302, 723, 3, "Rogers Communications", "CA", "Canada", "linked_candidate", "Starlink Direct to Cell", "documented partner; alternative Rogers MNC", "S008;S047"),
    BehavioralCandidate(401, 4, 2, "KaR-Tel LLC (Beeline)", "KZ", "Kazakhstan", "linked_candidate", "Starlink Direct to Cell", "documented partner; alternative KaR-Tel MNC", "S008;S051"),
    BehavioralCandidate(515, 1, 2, "Islacom / Globe group", "PH", "Philippines", "linked_candidate", "Starlink Mobile", "PLMN assignee is associated with documented partner Globe", "S007;S008"),
    BehavioralCandidate(530, 13, 2, "One New Zealand Group Limited", "NZ", "New Zealand", "linked_candidate", "Starlink Direct to Cell", "documented partner; dedicated One NZ MNC", "S008;S012;S046"),
    BehavioralCandidate(234, 2, 2, "Telefonica UK Limited (O2)", "GB", "United Kingdom", "linked_candidate", "Starlink Direct to Cell", "documented partner; alternative O2 MNC", "S007;S008"),
    BehavioralCandidate(440, 25, 2, "SoftBank Corp.", "JP", "Japan", "linked_candidate", "Starlink Direct to Cell", "documented partner; alternative SoftBank MNC", "S008;S048"),
    BehavioralCandidate(255, 707, 3, "Kyivstar PrJSC", "UA", "Ukraine", "linked_candidate", "Starlink Direct to Cell", "documented active partner; dedicated Kyivstar MNC", "S002;S008;S009"),
    BehavioralCandidate(440, 26, 2, "NTT DOCOMO Inc.", "JP", "Japan", "linked_candidate", "Starlink Direct to Cell", "documented partner; alternative DOCOMO MNC", "S008;S048"),
    BehavioralCandidate(730, 31, 2, "Entel Telefonia Movil S.A.", "CL", "Chile", "linked_candidate", "Starlink Direct to Cell", "documented partner; alternative Entel MNC", "S008;S052"),
    BehavioralCandidate(310, 210, 3, "T-Mobile USA", "US", "United States", "linked_candidate", "Starlink Direct to Cell", "documented active partner; alternative T-Mobile MNC", "S008;S057"),
    BehavioralCandidate(530, 2, 2, "Spark New Zealand / Telecom New Zealand", "NZ", "New Zealand", "linked_candidate", "Satellite D2D system unresolved", "documented D2D-partner family; legacy Spark MNC", "S007;S058", "CID structure is Starlink-like, but the primary partner source names Lynk."),
    BehavioralCandidate(440, 55, 2, "KDDI Corporation", "JP", "Japan", "linked_candidate", "Starlink Direct to Cell", "documented active partner; alternative KDDI MNC", "S008;S011;S048"),
    BehavioralCandidate(505, 11, 2, "Telstra Corporation Ltd", "AU", "Australia", "linked_candidate", "Satellite D2D system unresolved", "documented D2D-partner family; alternative Telstra MNC", "S007;S008;S058"),
    BehavioralCandidate(712, 5, 2, "Liberty Telecomunicaciones de Costa Rica", "CR", "Costa Rica", "linked_candidate", "Starlink Direct to Cell", "documented partner group; Liberty MNC", "S008;S055"),
    BehavioralCandidate(338, 110, 3, "Cable & Wireless Jamaica Ltd", "JM", "Jamaica", "linked_candidate", "Starlink Direct to Cell candidate", "Liberty Latin America operator family", "S007;S008", "Jamaica is not individually named by the partner page captured in the registry."),
    BehavioralCandidate(639, 5, 2, "Airtel Networks Kenya Limited", "KE", "Kenya", "linked_candidate", "Starlink Direct to Cell", "documented Airtel Africa partner group; alternative Airtel MNC", "S007;S008"),
    BehavioralCandidate(214, 16, 2, "R Cable y Telecomunicaciones Galicia", "ES", "Spain", "linked_candidate", "Starlink Direct to Cell", "documented MasOrange partner group; group MNC", "S007;S008"),
    BehavioralCandidate(714, 5, 2, "Cable & Wireless Panama S.A.", "PA", "Panama", "linked_candidate", "Starlink Direct to Cell", "documented Liberty partner group; alternative C&W MNC", "S007;S008"),
    BehavioralCandidate(630, 4, 2, "Unresolved (possible Airtel DRC family)", "CD", "Democratic Republic of the Congo", "unresolved_signature", "Unresolved", "behavioral signature only", "S007;S008", "No current primary assignment record was found for 630-04."),
    BehavioralCandidate(470, 8, 2, "Unresolved", "BD", "Bangladesh", "unresolved_signature", "Unresolved", "behavioral signature only", "S007;S008", "Potential relationship to a disclosed Banglalink partnership remains unverified."),
    BehavioralCandidate(460, 13, 2, "Unresolved", "CN", "China", "unresolved_signature", "Unresolved", "behavioral signature only", "S007", "More TACs, more CID high values, and a longer observation window than the dominant D2C pattern."),
    BehavioralCandidate(530, 7, 2, "Dense Air New Zealand Ltd", "NZ", "New Zealand", "unresolved_signature", "Unresolved", "behavioral signature only", "S007", "Small population sharing the 2025-04-23 cutoff of the early candidate wave."),
    BehavioralCandidate(645, 9, 2, "Unresolved", "ZM", "Zambia", "unresolved_signature", "Unresolved", "behavioral signature only", "S007;S008", "Potential relationship to a disclosed MTN Zambia partnership remains unverified."),
    BehavioralCandidate(310, 32, 3, "Unresolved", "US", "United States", "unresolved_signature", "Unresolved", "behavioral signature only", "S007", "Small population; assignment and satellite relationship unresolved."),
]


ADDITIONAL_ASSIGNMENTS = [
    {
        "plmn_id": "901-60", "mcc": "901", "mnc": "60", "mnc_length": "2", "plmn": "901-60",
        "assignee": "OneWeb", "country_iso": "", "scope": "International shared",
        "plmn_role": "satellite_operator", "assignment_status": "assigned", "valid_from": "2018-05-18",
        "valid_to": "", "scan_candidate": "1", "source_ids": "S053",
        "notes": "Direct satellite-operator assignment; no matching identities were found in this dataset.",
    },
    {
        "plmn_id": "901-70", "mcc": "901", "mnc": "70", "mnc_length": "2", "plmn": "901-70",
        "assignee": "Bureau 1440", "country_iso": "", "scope": "International shared",
        "plmn_role": "satellite_operator", "assignment_status": "assigned", "valid_from": "2026-02-23",
        "valid_to": "", "scan_candidate": "1", "source_ids": "S054",
        "notes": "Dataset observations predate this assignment and resemble private/test-network defaults; do not attribute automatically.",
    },
    {
        "plmn_id": "505-59", "mcc": "505", "mnc": "59", "mnc_length": "2", "plmn": "505-59",
        "assignee": "Starlink Internet Services Pte Ltd", "country_iso": "AU", "scope": "Australia",
        "plmn_role": "satellite_operator", "assignment_status": "assigned", "valid_from": "2024-03-18",
        "valid_to": "", "scan_candidate": "1", "source_ids": "S049",
        "notes": "Direct Starlink assignment in Australia.",
    },
    {
        "plmn_id": "505-60", "mcc": "505", "mnc": "60", "mnc_length": "2", "plmn": "505-60",
        "assignee": "Starlink Internet Services Pte Ltd", "country_iso": "AU", "scope": "Australia",
        "plmn_role": "satellite_operator", "assignment_status": "assigned", "valid_from": "2025-08-12",
        "valid_to": "", "scan_candidate": "1", "source_ids": "S050",
        "notes": "Direct Starlink assignment in Australia; no matching identities were found in this dataset.",
    },
]


ADDITIONAL_SOURCES = [
    ("S046", "International Telecommunication Union", "Operational Bulletin 1287", "2024-03-01", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1287-2024-OAS-PDF-E.pdf", "Assignment of 530-13 to One New Zealand"),
    ("S047", "International Telecommunication Union", "Operational Bulletin 1303", "2024-11-01", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1303-2024-OAS-PDF-E.pdf", "Assignment of 302-723 to Rogers"),
    ("S048", "International Telecommunication Union", "Operational Bulletin 1320", "2025-07-15", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1320-2025-OAS-PDF-E.pdf", "Assignments of 440-25 to SoftBank, 440-26 to NTT DOCOMO, and 440-55 to KDDI"),
    ("S049", "International Telecommunication Union", "Operational Bulletin 1288", "2024-03-15", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1288-2024-OAS-PDF-E.pdf", "Assignment of 505-59 to Starlink"),
    ("S050", "International Telecommunication Union", "Operational Bulletin 1321", "2025-08-01", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1321-2025-OAS-PDF-E.pdf", "Assignment of 505-60 to Starlink"),
    ("S051", "International Telecommunication Union", "Operational Bulletin 1326", "2025-10-15", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1326-2025-OAS-PDF-E.pdf", "Assignment of 401-04 to KaR-Tel"),
    ("S052", "International Telecommunication Union", "Operational Bulletin 1340", "2026-05-15", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1340-2026-OAS-PDF-E.pdf", "Assignment of 730-31 to Entel"),
    ("S053", "International Telecommunication Union", "Operational Bulletin 1147", "2018-06-01", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1147-2018-OAS-PDF-E.pdf", "Assignment of 901-60 to OneWeb"),
    ("S054", "International Telecommunication Union", "Operational Bulletin 1336", "2026-03-15", "https://www.itu.int/dms_pub/itu-t/opb/sp/T-SP-OB.1336-2026-OAS-PDF-E.pdf", "Assignment of 901-70 to Bureau 1440"),
    ("S055", "SUTEL Costa Rica", "Official numbering registry", "2026-03-26", "https://sutel.go.cr/sites/default/files/registro-numeracion-26-3-2026_0.pdf", "Assignment of 712-05 to Liberty"),
    ("S057", "T-Mobile", "T-Satellite beta open for all carriers", "2025-02-09", "https://www.t-mobile.com/news/network/t-mobile-starlink-beta-open-for-all-carriers", "T-Mobile/Starlink beta and fall 2024 emergency activation timing"),
    ("S058", "Lynk", "TPG Telecom looks to near-100 percent mobile coverage", "2025-03", "https://lynk.world/wp-content/uploads/2025/03/TPGTelecomlookstonear-100_mobilecoverage.pdf", "Names Telstra, Spark, Rogers, and Globe among Lynk partners"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_screening() -> pd.DataFrame:
    screen = ch_df(SCREEN_QUERY)
    screen["first_seen"] = pd.to_datetime(screen["first_seen"])
    screen["last_seen"] = pd.to_datetime(screen["last_seen"])
    screen["signature_match"] = (
        (screen["first_seen"] >= "2024-01-01")
        & (screen["cid_high_values"] >= 250)
        & (screen["cid_low_values"] >= 100)
    ).astype(int)
    screen["plmn"] = screen["mcc"].astype(str) + "-" + screen["mnc"].astype(str)
    columns = ["plmn"] + [c for c in screen.columns if c != "plmn"]
    screen[columns].to_csv(OUTPUT / "plmn_screening.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    return screen


def build_inventory() -> list[dict]:
    registry = read_csv(OLD_REGISTRY / "plmns.csv")
    by_plmn = {row["plmn"]: row for row in registry if row["plmn"]}
    for row in ADDITIONAL_ASSIGNMENTS:
        by_plmn[row["plmn"]] = row

    candidates = {candidate.plmn: candidate for candidate in BEHAVIORAL_CANDIDATES}
    rows: list[dict] = []
    for plmn, row in by_plmn.items():
        candidate = candidates.get(plmn)
        role = row["plmn_role"]
        if candidate:
            selection = "literature_registry;global_behavioral_screen"
            tier = candidate.tier
            system = candidate.system
            relationship = candidate.relationship
            figure_group = candidate.tier
            figure_include = 1
            source_ids = ";".join(dict.fromkeys((row.get("source_ids", "") + ";" + candidate.source_ids).strip(";").split(";")))
            notes = " ".join(filter(None, [row.get("notes", ""), candidate.notes])).strip()
        else:
            selection = "literature_registry"
            tier = {"satellite_operator": "direct_assignment", "trial": "direct_assignment", "mobility_control": "control"}.get(role, "literature_lead")
            system = ""
            relationship = ""
            figure_group = {"satellite_operator": "direct_assignment", "trial": "direct_assignment", "mobility_control": "onboard_control"}.get(role, "not_plotted")
            figure_include = int(role in {"satellite_operator", "trial", "dedicated_mno_segment", "mobility_control"})
            source_ids = row.get("source_ids", "")
            notes = row.get("notes", "")
            if plmn == "901-70":
                tier = "assignment_conflict"
                figure_group = "unresolved_signature"

        rows.append(
            {
                "plmn": plmn,
                "mcc": row.get("mcc", ""),
                "mnc": row.get("mnc", ""),
                "mnc_length": row.get("mnc_length", ""),
                "assignee": row.get("assignee", ""),
                "country_iso": row.get("country_iso", ""),
                "scope": row.get("scope", ""),
                "registry_role": role,
                "selection_channel": selection,
                "evidence_tier": tier,
                "likely_system": system,
                "relationship": relationship,
                "figure_group": figure_group,
                "figure_include": figure_include,
                "assignment_status": row.get("assignment_status", ""),
                "valid_from": row.get("valid_from", ""),
                "valid_to": row.get("valid_to", ""),
                "source_ids": source_ids,
                "notes": notes,
            }
        )

    for candidate in BEHAVIORAL_CANDIDATES:
        if candidate.plmn in by_plmn:
            continue
        rows.append(
            {
                "plmn": candidate.plmn,
                "mcc": candidate.mcc,
                "mnc": candidate.mnc,
                "mnc_length": candidate.mnc_length,
                "assignee": candidate.assignee,
                "country_iso": candidate.country_iso,
                "scope": candidate.country,
                "registry_role": "behavioral_candidate",
                "selection_channel": "global_behavioral_screen",
                "evidence_tier": candidate.tier,
                "likely_system": candidate.system,
                "relationship": candidate.relationship,
                "figure_group": candidate.tier,
                "figure_include": 1,
                "assignment_status": "assigned_or_observed",
                "valid_from": "",
                "valid_to": "",
                "source_ids": candidate.source_ids,
                "notes": candidate.notes,
            }
        )

    fieldnames = [
        "plmn", "mcc", "mnc", "mnc_length", "assignee", "country_iso", "scope",
        "registry_role", "selection_channel", "evidence_tier", "likely_system",
        "relationship", "figure_group", "figure_include", "assignment_status",
        "valid_from", "valid_to", "source_ids", "notes",
    ]
    rows.sort(key=lambda r: (str(r["mcc"]), int(r["mnc"]) if str(r["mnc"]).isdigit() else -1, r["plmn"]))
    write_csv(OUTPUT / "plmn_inventory.csv", rows, fieldnames)
    return rows


def build_sources() -> None:
    rows = read_csv(OLD_REGISTRY / "sources.csv")
    known = {row["source_id"] for row in rows}
    for source_id, publisher, title, published, url, scope in ADDITIONAL_SOURCES:
        if source_id not in known:
            rows.append(
                {
                    "source_id": source_id,
                    "publisher": publisher,
                    "title": title,
                    "published": published,
                    "url": url,
                    "accessed": SNAPSHOT_DATE,
                    "scope": scope,
                }
            )
    rows.sort(key=lambda row: int(row["source_id"][1:]))
    write_csv(OUTPUT / "sources.csv", rows, ["source_id", "publisher", "title", "published", "url", "accessed", "scope"])


def selected_rows(inventory: list[dict]) -> list[dict]:
    return [row for row in inventory if row["figure_include"] == 1 and str(row["mcc"]).isdigit() and str(row["mnc"]).isdigit()]


def stream_query_to_gzip(query: str, path: Path) -> None:
    remote_cmd = ["clickhouse-client", "--password", CH_PASSWORD]
    for key, value in DEFAULT_SETTINGS.items():
        remote_cmd += [f"--{key}", str(value)]
    proc = subprocess.Popen(
        ["ssh", *_SSH_OPTS, HOST, " ".join(remote_cmd)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    proc.stdin.write((query.strip().rstrip(";") + "\nFORMAT CSVWithNames\n").encode())
    proc.stdin.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=6) as output:
        shutil.copyfileobj(proc.stdout, output, length=1024 * 1024)
    error = proc.stderr.read().decode(errors="replace")
    returncode = proc.wait()
    if returncode:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"ClickHouse query failed for {path.name}:\n{error}")


def cell_query(row: dict) -> str:
    return f"""
SELECT
    mcc, mnc, lac, cid, cell_type,
    first_seen, last_seen, obs AS observations,
    glat AS latest_lat, glon AS latest_lon,
    first_lat, first_lon, lat_min, lat_max, lon_min, lon_max,
    n_pos AS position_count,
    country_iso, country, region, state, city
FROM cell.summary_full
WHERE mcc = {int(row['mcc'])} AND mnc = {int(row['mnc'])} AND cid > 0
ORDER BY lac, cid, cell_type
"""


def build_cells(inventory: list[dict], refresh: bool) -> None:
    CELLS.mkdir(parents=True, exist_ok=True)
    for row in selected_rows(inventory):
        path = CELLS / f"{row['plmn']}.csv.gz"
        if path.exists() and not refresh:
            print(f"[cells cached] {row['plmn']}")
            continue
        print(f"[cells query] {row['plmn']} {row['assignee']}")
        stream_query_to_gzip(cell_query(row), path)


def summary_query(rows: list[dict]) -> str:
    predicate = " OR ".join(f"(mcc = {int(row['mcc'])} AND mnc = {int(row['mnc'])})" for row in rows)
    return f"""
SELECT
    mcc, mnc,
    countIf(cid > 0) AS raw_summary_rows,
    uniqExactIf((lac, cid, cell_type), cid > 0) AS tac_cell_rows,
    uniqExactIf((cid, cell_type), cid > 0) AS unique_cell_ids,
    uniqExactIf((lac, cid), cid > 0 AND cell_type = 'lte') AS lte_tac_eci_rows,
    uniqExactIf(cid, cid > 0 AND cell_type = 'lte') AS unique_lte_ecis,
    countIf(cid > 0 AND cell_type = 'gsm') AS gsm_identities,
    countIf(cid > 0 AND cell_type = 'nr') AS nr_identities,
    countIf(cid > 0 AND cell_type = 'cdma') AS cdma_identities,
    sumIf(obs, cid > 0) AS observations,
    uniqExactIf(lac, cid > 0) AS area_codes,
    uniqExactIf(intDiv(cid, 256), cid > 0 AND cell_type = 'lte') AS lte_cid_high_values,
    uniqExactIf(bitAnd(cid, 255), cid > 0 AND cell_type = 'lte') AS lte_cid_low_values,
    minIf(first_seen, cid > 0) AS first_seen,
    maxIf(last_seen, cid > 0) AS last_seen,
    countIf(cid > 0 AND n_pos > 1) AS multi_position_identities,
    countIf(cid > 0 AND glat BETWEEN -90 AND 90 AND glon BETWEEN -180 AND 180
            AND NOT (glat = 0 AND glon = 0)) AS valid_latest_positions,
    uniqExactIf(country_iso, cid > 0 AND country_iso != '') AS located_countries
FROM cell.summary_full
WHERE {predicate}
GROUP BY mcc, mnc
ORDER BY tac_cell_rows DESC, mcc, mnc
"""


def build_summary(inventory: list[dict]) -> pd.DataFrame:
    selected = selected_rows(inventory)
    summary = ch_df(summary_query(selected))
    metadata = pd.DataFrame(selected)[["plmn", "mcc", "mnc", "assignee", "evidence_tier", "figure_group", "likely_system"]]
    metadata[["mcc", "mnc"]] = metadata[["mcc", "mnc"]].astype(int)
    summary = metadata.merge(summary, on=["mcc", "mnc"], how="left", validate="one_to_one")
    numeric = [c for c in summary.columns if c not in {"plmn", "assignee", "evidence_tier", "figure_group", "likely_system", "first_seen", "last_seen"}]
    summary[numeric] = summary[numeric].fillna(0)
    summary.to_csv(OUTPUT / "plmn_summary.csv", index=False)
    return summary


def build_daily(inventory: list[dict]) -> None:
    chunks: list[pd.DataFrame] = []
    for row in selected_rows(inventory):
        print(f"[daily query] {row['plmn']}")
        frame = ch_df(
            f"""
SELECT
    toDate(timestamp) AS date,
    count() AS observations,
    uniqExact((lac, cid, cell_type)) AS active_tac_cell_rows,
    uniqExactIf(cid, cid > 0 AND cell_type = 'lte') AS active_lte_ecis,
    uniqExactIf(intDiv(cid, 256), cid > 0 AND cell_type = 'lte') AS active_lte_enodebs,
    uniqExactIf(lac, cid > 0) AS active_area_codes
FROM cell.geos
WHERE mcc = {int(row['mcc'])} AND mnc = {int(row['mnc'])} AND cid > 0
GROUP BY date
ORDER BY date
"""
        )
        if frame.empty:
            continue
        frame.insert(0, "plmn", row["plmn"])
        frame.insert(1, "mcc", int(row["mcc"]))
        frame.insert(2, "mnc", int(row["mnc"]))
        chunks.append(frame)
    daily = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["plmn", "mcc", "mnc", "date", "observations", "active_tac_cell_rows", "active_lte_ecis", "active_lte_enodebs", "active_area_codes"])
    daily.to_csv(OUTPUT / "daily_activity.csv", index=False)


def build_manifest() -> None:
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "snapshot_date": SNAPSHOT_DATE,
        "database": "cell.summary_full and cell.geos on nominatim.cybre.io (read-only)",
        "files": [
            {
                "path": str(path.relative_to(OUTPUT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cells", action="store_true", help="replace existing per-PLMN cell CSVs")
    parser.add_argument("--skip-daily", action="store_true", help="skip the raw-observation daily aggregation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    screen = build_screening()
    actual = set(zip(screen.loc[screen["signature_match"] == 1, "mcc"], screen.loc[screen["signature_match"] == 1, "mnc"]))
    expected = {(candidate.mcc, candidate.mnc) for candidate in BEHAVIORAL_CANDIDATES}
    if actual != expected:
        raise RuntimeError(f"Global signature set changed: added={sorted(actual - expected)}, removed={sorted(expected - actual)}")
    inventory = build_inventory()
    build_sources()
    build_cells(inventory, args.refresh_cells)
    summary = build_summary(inventory)
    if not args.skip_daily:
        build_daily(inventory)
    build_manifest()
    print(f"[done] {len(screen):,} screened PLMNs; {len(expected)} signature matches; {int(summary['tac_cell_rows'].sum()):,} selected TAC/cell rows")


if __name__ == "__main__":
    main()
