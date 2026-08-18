#!/usr/bin/env python3
"""Ranked matrix of cells located outside their PLMN's geographic territory.

The unit is a distinct cell identity `(mcc,mnc,lac,cid,cell_type)`, never an
API observation. The remote database is queried read-only and reduced there to
counts per PLMN / latest 0.01-degree position before any rows are transferred.

Two independent country identities are retained:

* located country: the Natural Earth map unit containing the latest position;
* MCC country: the geographic area assigned to the MCC/MNC in the reference
  table under `data/reference/`.

`--border-buffer-km` excludes positions no farther than that distance outside
the claimed MCC territory. This is deliberately a distance to the *claimed
territory*, rather than a generic distance to any international border: it
removes the band in which coordinate error and ordinary RF spillover are most
plausible while retaining genuinely distant cross-country identities.

The raw aggregate is cached locally. Changing the border buffer only repeats
the inexpensive local geometry and plotting stages; use `--refresh-cache` when
the remote corpus or MCC reference has changed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"
MCC_TABLE = REFERENCE / "mcc-mnc-table.csv"
BOUNDARIES = REFERENCE / "ne_10m_admin_0_map_units.geojson"
CACHE_DIR = ROOT / ".cache" / "out_of_country_mcc"
DEFAULT_OUTPUT = ROOT / "paper" / "figs" / "out_of_country_mcc_matrix.pdf"

HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
CH_PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")

# Reverse-geocoder codes that Natural Earth expresses as ISO_A2_EH.
HOST_CODE_ALIASES = {
    "CN-TW": "TW",
    "FR-974": "RE",
    "FR-976": "YT",
}

# Obsolete Netherlands Antilles identities still occur in PLMN reference data.
# Natural Earth represents the current constituent territories separately.
GEOMETRY_ALIASES = {"AN": ("CW", "BQ", "SX")}

NAME_OVERRIDES = {
    "BO": "Bolivia",
    "BQ": "Caribbean Netherlands",
    "CD": "DR Congo",
    "CG": "Congo",
    "CI": "Côte d’Ivoire",
    "CN": "China",
    "CZ": "Czechia",
    "GB": "United Kingdom",
    "IR": "Iran",
    "IQ": "Iraq",
    "KP": "North Korea",
    "KR": "South Korea",
    "LA": "Laos",
    "MD": "Moldova",
    "PS": "Palestine",
    "RE": "Réunion",
    "RU": "Russia",
    "SY": "Syria",
    "TW": "Taiwan",
    "TZ": "Tanzania",
    "US": "United States",
    "VE": "Venezuela",
    "VN": "Vietnam",
    "XK": "Kosovo",
}


def _valid_iso(value: str | None) -> bool:
    return bool(value and len(value) == 2 and value.isalpha() and value != "-99")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class MccReference:
    plmn_iso: dict[tuple[int, int], str]
    mcc_iso: dict[int, str]
    names: dict[str, str]
    ambiguous_plmns: int
    ambiguous_mccs: dict[int, tuple[str, ...]]


def load_mcc_reference(path: Path) -> MccReference:
    """Load exact PLMN mappings and only unambiguous MCC fallbacks."""
    plmn_values: dict[tuple[int, int], set[str]] = collections.defaultdict(set)
    mcc_values: dict[int, set[str]] = collections.defaultdict(set)
    name_votes: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                mcc, mnc = int(row["MCC"]), int(row["MNC"])
            except (KeyError, TypeError, ValueError):
                continue
            iso = row.get("ISO", "").strip().upper()
            if not _valid_iso(iso):
                continue
            plmn_values[(mcc, mnc)].add(iso)
            mcc_values[mcc].add(iso)
            country = row.get("Country", "").strip()
            if country:
                name_votes[iso][country] += 1

    plmn_iso = {key: next(iter(values)) for key, values in plmn_values.items() if len(values) == 1}
    mcc_iso = {key: next(iter(values)) for key, values in mcc_values.items() if len(values) == 1}
    names = {iso: votes.most_common(1)[0][0] for iso, votes in name_votes.items()}
    ambiguous = {mcc: tuple(sorted(values)) for mcc, values in mcc_values.items() if len(values) > 1}
    return MccReference(
        plmn_iso=plmn_iso,
        mcc_iso=mcc_iso,
        names=names,
        ambiguous_plmns=sum(len(v) > 1 for v in plmn_values.values()),
        ambiguous_mccs=ambiguous,
    )


def home_iso_sql(ref: MccReference) -> str:
    """Return a ClickHouse expression resolving exact PLMN then MCC fallback."""
    plmns = sorted(ref.plmn_iso.items())
    mccs = sorted(ref.mcc_iso.items())
    plmn_keys = ",".join(str(mcc * 1000 + mnc) for (mcc, mnc), _ in plmns)
    plmn_isos = ",".join("'" + iso.replace("'", "''") + "'" for _, iso in plmns)
    mcc_keys = ",".join(str(mcc) for mcc, _ in mccs)
    mcc_isos = ",".join("'" + iso.replace("'", "''") + "'" for _, iso in mccs)
    return (
        "transform(toUInt32(mcc) * 1000 + toUInt32(mnc), "
        f"[{plmn_keys}], [{plmn_isos}], "
        f"transform(mcc, [{mcc_keys}], [{mcc_isos}], ''))"
    )


def extraction_sql(ref: MccReference) -> str:
    home = home_iso_sql(ref)
    # Keep territorial/special raw-host mismatches as candidates. Local polygon
    # containment removes false mismatches such as RE vs reverse-geocoder FR.
    host_normalized = "transform(country_iso, ['CN-TW'], ['TW'], country_iso)"
    return f"""
SELECT
    home_iso,
    mcc,
    mnc,
    round(glat, 2) AS lat,
    round(glon, 2) AS lon,
    country_iso AS host_raw,
    count() AS cells
FROM
(
    SELECT
        mcc, mnc, glat, glon, country_iso,
        {home} AS home_iso
    FROM cell.summary_full
    WHERE cid > 0 AND NOT (glat = 0 AND glon = 0)
)
WHERE home_iso != ''
  AND ({host_normalized} != home_iso OR country_iso IN ('', '??'))
GROUP BY home_iso, mcc, mnc, lat, lon, host_raw
ORDER BY home_iso, mcc, mnc, lat, lon, host_raw
FORMAT CSVWithNames
""".strip()


def cache_path(ref: MccReference) -> Path:
    digest = hashlib.sha256((home_iso_sql(ref) + _sha256(MCC_TABLE)).encode()).hexdigest()[:12]
    return CACHE_DIR / f"latest_position_candidates_{digest}.csv.gz"


def query_to_cache(sql: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    remote = (
        "clickhouse-client "
        f"--password {shlex.quote(CH_PASSWORD)} --readonly 2 --max_threads 8 "
        "--max_execution_time 1800 --max_result_rows 0"
    )
    proc = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", HOST, remote],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(sql)
    proc.stdin.close()
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", newline="") as out:
            for chunk in iter(lambda: proc.stdout.read(1024 * 1024), ""):
                out.write(chunk)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        rc = proc.wait()
        if rc:
            raise RuntimeError(f"ClickHouse extraction failed ({rc}):\n{stderr.strip()}")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        if proc.poll() is None:
            proc.terminate()
        raise


@dataclass
class MapFeature:
    iso: str
    name: str
    sovereign: str
    polygons: list[list[list[list[float]]]]
    bbox: tuple[float, float, float, float]


def _polygon_sets(geometry: dict) -> list[list[list[list[float]]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    return []


def _bbox(polygons: list[list[list[list[float]]]]) -> tuple[float, float, float, float]:
    points = [p for polygon in polygons for ring in polygon for p in ring]
    return (
        min(p[0] for p in points), max(p[0] for p in points),
        min(p[1] for p in points), max(p[1] for p in points),
    )


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    x1 = ((ring[-1][0] - lon + 180.0) % 360.0) - 180.0
    y1 = ring[-1][1] - lat
    for point in ring:
        x2 = ((point[0] - lon + 180.0) % 360.0) - 180.0
        y2 = point[1] - lat
        # Keep dateline-crossing segments locally continuous around the point.
        if x2 - x1 > 180.0:
            x2 -= 360.0
        elif x2 - x1 < -180.0:
            x2 += 360.0
        if (y1 > 0) != (y2 > 0):
            crossing_x = x1 + (x2 - x1) * (-y1) / (y2 - y1)
            if crossing_x > 0:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _point_in_feature(lon: float, lat: float, feature: MapFeature) -> bool:
    xmin, xmax, ymin, ymax = feature.bbox
    if not (ymin <= lat <= ymax):
        return False
    # The cheap longitude bbox is skipped for dateline-spanning features.
    if xmax - xmin < 180 and not (xmin <= lon <= xmax):
        return False
    for polygon in feature.polygons:
        if polygon and _point_in_ring(lon, lat, polygon[0]):
            if not any(_point_in_ring(lon, lat, hole) for hole in polygon[1:]):
                return True
    return False


class CountryBoundaries:
    def __init__(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.features: list[MapFeature] = []
        self.by_iso: dict[str, list[MapFeature]] = collections.defaultdict(list)
        self.names: dict[str, str] = {}
        self._class_cache: dict[tuple[float, float, str], str] = {}
        self._boundary_indexes: dict[str, BoundaryIndex] = {}

        for item in raw["features"]:
            props = item.get("properties", {})
            iso = props.get("ISO_A2_EH")
            if not _valid_iso(iso):
                iso = props.get("ISO_A2")
            if not _valid_iso(iso):
                continue
            polygons = _polygon_sets(item["geometry"])
            if not polygons:
                continue
            feature = MapFeature(
                iso=iso,
                name=props.get("NAME_EN") or props.get("NAME") or iso,
                sovereign=props.get("SOVEREIGNT") or props.get("ADMIN") or "",
                polygons=polygons,
                bbox=_bbox(polygons),
            )
            self.features.append(feature)
            self.by_iso[iso].append(feature)
            self.names.setdefault(iso, feature.name)

        for alias, members in GEOMETRY_ALIASES.items():
            self.by_iso[alias] = [feature for member in members for feature in self.by_iso.get(member, [])]

        # Link a sovereign's ISO to all of its map units so raw `FR` positions,
        # for example, can be refined to Réunion (`RE`) by point containment.
        sovereign_iso: dict[str, str] = {}
        for feature in self.features:
            if feature.name == feature.sovereign:
                sovereign_iso[feature.sovereign] = feature.iso
        # Only index *subunits whose ISO differs from their sovereign*. The raw
        # reverse-geocoder code is already a country code, so point-testing its
        # often very detailed mainland polygon millions of times adds no
        # information. Refinement is needed only for cases such as raw FR at a
        # Réunion coordinate, or raw IL inside the PS map units.
        self.by_raw_host: dict[str, list[MapFeature]] = collections.defaultdict(list)
        for feature in self.features:
            sov_iso = sovereign_iso.get(feature.sovereign)
            if sov_iso and feature.iso != sov_iso:
                self.by_raw_host[sov_iso].append(feature)
        for iso, features in self.by_raw_host.items():
            # Small/discrete map units should win before their enclosing polity.
            self.by_raw_host[iso] = sorted(
                {id(f): f for f in features}.values(),
                key=lambda f: (f.bbox[1] - f.bbox[0]) * (f.bbox[3] - f.bbox[2]),
            )

    def has_geometry(self, iso: str) -> bool:
        return bool(self.by_iso.get(iso))

    def contains(self, iso: str, lon: float, lat: float) -> bool:
        return any(_point_in_feature(lon, lat, feature) for feature in self.by_iso.get(iso, []))

    def classify_host(self, lon: float, lat: float, raw: str) -> str:
        raw = HOST_CODE_ALIASES.get(raw, raw)
        if not _valid_iso(raw):
            return ""
        key = (lon, lat, raw)
        if key in self._class_cache:
            return self._class_cache[key]
        candidates = self.by_raw_host.get(raw, [])
        result = raw
        for feature in candidates:
            if _point_in_feature(lon, lat, feature):
                result = feature.iso
                break
        self._class_cache[key] = result
        return result

    def distance_to_boundary_km(self, iso: str, lon: float, lat: float, search_km: float) -> float:
        if iso not in self._boundary_indexes:
            self._boundary_indexes[iso] = BoundaryIndex(self.by_iso.get(iso, []))
        return self._boundary_indexes[iso].distance_km(lon, lat, search_km)


class BoundaryIndex:
    """One-degree spatial index over polygon boundary segments."""

    def __init__(self, features: Iterable[MapFeature]):
        self.tiles: dict[tuple[int, int], list[tuple[float, float, float, float]]] = collections.defaultdict(list)
        for feature in features:
            for polygon in feature.polygons:
                for ring in polygon:
                    for p1, p2 in zip(ring, ring[1:]):
                        lon1, lat1 = float(p1[0]), float(p1[1])
                        lon2, lat2 = float(p2[0]), float(p2[1])
                        lon2_local = lon2
                        if lon2_local - lon1 > 180:
                            lon2_local -= 360
                        elif lon2_local - lon1 < -180:
                            lon2_local += 360
                        for tx in range(math.floor(min(lon1, lon2_local)), math.floor(max(lon1, lon2_local)) + 1):
                            ntx = ((tx + 180) % 360) - 180
                            for ty in range(math.floor(min(lat1, lat2)), math.floor(max(lat1, lat2)) + 1):
                                self.tiles[(ntx, ty)].append((lon1, lat1, lon2, lat2))

    def distance_km(self, lon: float, lat: float, search_km: float) -> float:
        lat_radius = max(1, math.ceil(search_km / 110.0) + 1)
        cos_lat = max(abs(math.cos(math.radians(lat))), 0.05)
        lon_radius = max(1, math.ceil(search_km / (111.0 * cos_lat)) + 1)
        segments: list[tuple[float, float, float, float]] = []
        tx0, ty0 = math.floor(lon), math.floor(lat)
        for dx in range(-lon_radius, lon_radius + 1):
            tx = ((tx0 + dx + 180) % 360) - 180
            for dy in range(-lat_radius, lat_radius + 1):
                segments.extend(self.tiles.get((tx, ty0 + dy), ()))
        if not segments:
            return math.inf
        arr = np.asarray(segments, dtype=float)
        x1 = ((arr[:, 0] - lon + 180.0) % 360.0 - 180.0) * 111.320 * cos_lat
        y1 = (arr[:, 1] - lat) * 110.574
        x2 = ((arr[:, 2] - lon + 180.0) % 360.0 - 180.0) * 111.320 * cos_lat
        y2 = (arr[:, 3] - lat) * 110.574
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        t = np.divide(-(x1 * dx + y1 * dy), denom, out=np.zeros_like(denom), where=denom > 0)
        t = np.clip(t, 0.0, 1.0)
        distance = np.hypot(x1 + t * dx, y1 + t * dy)
        return float(distance.min())


@dataclass
class Audit:
    candidate_cells: int = 0
    inside_home: int = 0
    border_buffered: int = 0
    missing_home_geometry: int = 0
    unresolved_host: int = 0
    retained: int = 0


def collect_pairs(
    cache: Path,
    boundaries: CountryBoundaries,
    border_buffer_km: float,
) -> tuple[dict[tuple[str, str], int], Audit]:
    pairs: dict[tuple[str, str], int] = collections.defaultdict(int)
    audit = Audit()
    distance_cache: dict[tuple[str, float, float], float] = {}

    with gzip.open(cache, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cells = int(row["cells"])
            audit.candidate_cells += cells
            home = row["home_iso"]
            lat, lon = float(row["lat"]), float(row["lon"])
            if not boundaries.has_geometry(home):
                audit.missing_home_geometry += cells
                continue
            host = boundaries.classify_host(lon, lat, row["host_raw"])
            if not host:
                audit.unresolved_host += cells
                continue
            if host == home:
                audit.inside_home += cells
                continue
            dkey = (home, lon, lat)
            if dkey not in distance_cache:
                distance_cache[dkey] = boundaries.distance_to_boundary_km(home, lon, lat, border_buffer_km)
            if distance_cache[dkey] <= border_buffer_km:
                audit.border_buffered += cells
                continue
            pairs[(host, home)] += cells
            audit.retained += cells
    return dict(pairs), audit


def short_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def display_name(iso: str, names: dict[str, str]) -> str:
    name = NAME_OVERRIDES.get(iso, names.get(iso, iso))
    return f"{name} ({iso})"


def write_pairs(path: Path, pairs: dict[tuple[str, str], int], buffer_km: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["located_iso", "mcc_country_iso", "cells", "border_buffer_km"])
        for (host, home), cells in sorted(pairs.items(), key=lambda item: item[1], reverse=True):
            writer.writerow([host, home, cells, f"{buffer_km:g}"])


def make_plot(
    pairs: dict[tuple[str, str], int],
    names: dict[str, str],
    output: Path,
    border_buffer_km: float,
    top_hosts: int,
    top_homes: int,
) -> None:
    host_totals: collections.Counter[str] = collections.Counter()
    home_totals: collections.Counter[str] = collections.Counter()
    for (host, home), cells in pairs.items():
        host_totals[host] += cells
        home_totals[home] += cells
    hosts = [iso for iso, _ in host_totals.most_common(top_hosts)]
    homes = [iso for iso, _ in home_totals.most_common(top_homes)]
    if not hosts or not homes:
        raise RuntimeError("No country pairs remain after filtering")

    matrix = np.zeros((len(hosts), len(homes) + 1), dtype=np.int64)
    host_idx = {iso: i for i, iso in enumerate(hosts)}
    home_idx = {iso: i for i, iso in enumerate(homes)}
    for (host, home), cells in pairs.items():
        if host not in host_idx:
            continue
        j = home_idx.get(home, len(homes))
        matrix[host_idx[host], j] += cells

    columns = homes + ["OTHER"]
    positive = matrix[matrix > 0]
    vmax = int(positive.max())

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans 3", "Source Sans Pro", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # A single visual channel is easier to scan than the former combination of
    # bubble area, colour, and two marginal bar charts. Totals live directly in
    # the labels; the matrix is a log-colour table with selective annotations.
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    fig.subplots_adjust(left=0.145, right=0.885, bottom=0.055, top=0.76)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cells", ["#eef5f3", "#b8d8d2", "#4b948c", "#145f64", "#163b4a"]
    )
    cmap.set_bad("#f6f7f6")
    norm = mpl.colors.LogNorm(vmin=1, vmax=vmax)
    masked = np.ma.masked_where(matrix == 0, matrix)
    image = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="none", aspect="auto")

    x = np.arange(len(columns))
    y = np.arange(len(hosts))

    def column_label(iso: str) -> str:
        if iso == "OTHER":
            return "Other"
        name = NAME_OVERRIDES.get(iso, names.get(iso, iso))
        return name.replace("United States", "United\nStates").replace(
            "United Kingdom", "United\nKingdom"
        ).replace("French Guiana", "French\nGuiana")

    ax.set_xticks(
        x,
        [column_label(iso) for iso in columns],
        fontsize=5.8,
    )
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=5, length=0)
    ax.set_yticks(
        y,
        [NAME_OVERRIDES.get(iso, names.get(iso, iso)) for iso in hosts],
        fontsize=6.8,
    )
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel("MCC Country", fontsize=7.3, fontweight="semibold",
                  color="#26383c", labelpad=5)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Located Country", fontsize=7.3, fontweight="semibold",
                  color="#26383c", labelpad=8)

    # White gutters create a light table structure without a prominent grid.
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(hosts), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Labels are reserved for material cells so the long tail remains visible
    # without turning the figure into a table of tiny numbers.
    for yi, xi in zip(*np.nonzero(matrix), strict=True):
        value = int(matrix[yi, xi])
        if value < 100:
            continue
        color = "white" if norm(value) >= 0.58 else "#17343a"
        ax.text(xi, yi, short_count(value), ha="center", va="center",
                fontsize=6.2, fontweight="semibold", color=color)

    # Keep the scale outside the matrix so the lower edge can sit close to the
    # paper caption without sacrificing the full plotting width.
    cax = fig.add_axes([0.91, 0.22, 0.014, 0.49])
    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.set_ticks([1, 100, 10_000])
    cbar.set_ticklabels(["1", "100", "10k"])
    cbar.ax.tick_params(labelsize=5.8, length=2, pad=2)
    cbar.outline.set_visible(False)
    cbar.set_label("Distinct cells (log scale)", fontsize=6.2, color="#394b4f",
                   labelpad=3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    preview = output.with_suffix(".png")
    fig.savefig(preview, dpi=360, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {output}")
    print(f"[preview] {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--border-buffer-km", type=float, default=25.0,
        help="exclude cells no farther than this distance outside their MCC territory (default: 25)",
    )
    parser.add_argument("--top-hosts", type=int, default=12, help="number of located countries (rows)")
    parser.add_argument("--top-mcc-countries", type=int, default=12, help="number of MCC countries (columns)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pair-output", type=Path,
        help="CSV output for filtered country-pair counts (default: data/out_of_country_mcc_pairs_<buffer>km.csv)",
    )
    parser.add_argument("--refresh-cache", action="store_true", help="repeat the remote read-only extraction")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.border_buffer_km < 0:
        raise SystemExit("--border-buffer-km must be non-negative")
    if not MCC_TABLE.exists() or not BOUNDARIES.exists():
        raise SystemExit(f"missing reference data; see {REFERENCE / 'README.md'}")

    ref = load_mcc_reference(MCC_TABLE)
    print(f"[mapping] {len(ref.plmn_iso):,} exact PLMNs; {len(ref.mcc_iso):,} unambiguous MCC fallbacks")
    print(f"[mapping] ambiguous MCCs: {ref.ambiguous_mccs}")
    if ref.ambiguous_plmns:
        print(f"[mapping] excluding {ref.ambiguous_plmns} ambiguous PLMN key(s)")

    cache = cache_path(ref)
    if args.refresh_cache or not cache.exists():
        print(f"[query] read-only extraction from {HOST} -> {cache}")
        query_to_cache(extraction_sql(ref), cache)
    else:
        print(f"[cache] {cache}")

    boundaries = CountryBoundaries(BOUNDARIES)
    pairs, audit = collect_pairs(cache, boundaries, args.border_buffer_km)
    print(
        "[audit] "
        f"candidate={audit.candidate_cells:,}; inside-home={audit.inside_home:,}; "
        f"within-{args.border_buffer_km:g}km={audit.border_buffered:,}; "
        f"missing-home-geometry={audit.missing_home_geometry:,}; "
        f"unresolved-host={audit.unresolved_host:,}; retained={audit.retained:,}"
    )
    print(f"[pairs] {len(pairs):,} located-country x MCC-country pairs")
    for (host, home), cells in sorted(pairs.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"  {host:<5} <- {home:<5} {cells:>12,}")

    pair_output = args.pair_output
    if pair_output is None:
        token = f"{args.border_buffer_km:g}".replace(".", "p")
        pair_output = ROOT / "data" / f"out_of_country_mcc_pairs_{token}km.csv"
    write_pairs(pair_output, pairs, args.border_buffer_km)
    print(f"[data] {pair_output}")

    names = {**ref.names, **boundaries.names, **NAME_OVERRIDES}
    make_plot(
        pairs, names, args.output, args.border_buffer_km,
        top_hosts=args.top_hosts, top_homes=args.top_mcc_countries,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
