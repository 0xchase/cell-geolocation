#!/usr/bin/env python3
"""Expand and summarize supported local-motion rogue-BTS leads.

This operates entirely on the local moving-identity extract.  It retrieves all
position bins for identities selected by ``detect_rogue_bts_families.py`` and
computes campaign and path-shape diagnostics.  Aggregated bin timestamps do not
preserve repeated visits, so these metrics rank raw-history follow-up rather
than establish physical motion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_rogue_bts_families import OUT, ROOT, haversine  # noqa: E402


IDENTITIES = OUT / "local_mobile_identity_scores.csv"
ALL_POSITIONS = ROOT / "data" / "moving-mccs" / "positions.csv.zst"
POSITIONS = OUT / "local_mobile_candidate_positions.csv"


KEYS = ["mcc", "mnc", "lac", "cid", "cell_type"]


def extract_positions(identities: pd.DataFrame) -> pd.DataFrame:
    key_frame = identities[KEYS].drop_duplicates()
    if POSITIONS.exists():
        cached = pd.read_csv(POSITIONS)
        cached_keys = cached[KEYS].drop_duplicates()
        if len(cached_keys.merge(key_frame, on=KEYS)) == len(key_frame):
            return cached
    pieces = []
    for chunk in pd.read_csv(ALL_POSITIONS, compression="zstd", chunksize=500_000):
        match = chunk.merge(key_frame, on=KEYS, how="inner")
        if not match.empty:
            pieces.append(match)
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    frame.to_csv(POSITIONS, index=False)
    return frame


def path_metrics(identities: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in positions.groupby(KEYS, sort=False):
        ordered = group.sort_values(["first_seen", "last_seen", "lat", "lon"])
        coords = [(r.lat, r.lon) for r in ordered.itertuples()]
        steps = [haversine(*a, *b) for a, b in zip(coords, coords[1:])]
        direct = haversine(*coords[0], *coords[-1]) if len(coords) > 1 else 0
        route = sum(steps)
        rows.append({
            **dict(zip(KEYS, key)),
            "extracted_position_rows": len(group),
            "first_appearance_path_km": route,
            "first_to_last_appearance_km": direct,
            "first_appearance_tortuosity": route / direct if direct > 0 else np.nan,
            "largest_first_appearance_step_km": max(steps, default=0),
            "latitude_min": min(c[0] for c in coords),
            "latitude_max": max(c[0] for c in coords),
            "longitude_min": min(c[1] for c in coords),
            "longitude_max": max(c[1] for c in coords),
        })
    metrics = pd.DataFrame(rows)
    return identities.merge(metrics, on=KEYS, how="left")


def campaign_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    groups = ["endpoint_a_country_iso", "mcc", "mnc", "lac", "cell_type"]
    rows = []
    for key, group in metrics.groupby(groups):
        rows.append({
            **dict(zip(groups, key)),
            "assigned_country_iso": ";".join(sorted(set(group.assigned_country_iso))),
            "identities": len(group),
            "distinct_cids": group.cid.nunique(),
            "total_observations": int(group.total_observations.sum()),
            "median_span_km": group.max_span_km.median(),
            "maximum_span_km": group.max_span_km.max(),
            "median_position_rows": group.position_rows.median(),
            "median_tortuosity": group.first_appearance_tortuosity.median(),
            "latitude_min": group[["endpoint_a_lat", "endpoint_b_lat"]].min().min(),
            "latitude_max": group[["endpoint_a_lat", "endpoint_b_lat"]].max().max(),
            "longitude_min": group[["endpoint_a_lon", "endpoint_b_lon"]].min().min(),
            "longitude_max": group[["endpoint_a_lon", "endpoint_b_lon"]].max().max(),
            "first_seen": group.first_seen.min(),
            "last_seen": group.last_seen.max(),
        })
    return pd.DataFrame(rows).sort_values(
        ["identities", "total_observations"], ascending=False
    )


def main() -> None:
    identities = pd.read_csv(IDENTITIES)
    positions = extract_positions(identities)
    metrics = path_metrics(identities, positions)
    campaigns = campaign_summary(metrics)
    metrics.to_csv(OUT / "local_mobile_path_scores.csv", index=False)
    campaigns.to_csv(OUT / "local_mobile_campaigns.csv", index=False)
    print(campaigns.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
