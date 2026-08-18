#!/usr/bin/env python3
"""Screen the frozen cell-location detector against independently reported GNSS events.

The event catalogue is fixed in ``data/spoofing/news_validation`` before this
script is run.  Queries are read-only and use the precomputed, news-blind
``spoof.onsets_f`` table.  A match is not declared from raw volume: the script
retains the detector's family-wise threshold (nine simultaneous cell-identity
onsets in a 0.1-degree source square on one day) and applies the same mixture
geometry classification used by ``scripts/spoof2/s10_classify.py``.

Jamming-only reports are deliberately included as negative-capability tests.
The database stores inferred cell coordinates, not receiver signal quality, so
an absence for those events is expected and is not evidence against the report.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "data/spoofing/news_validation/known_gnss_events.csv"
DEFAULT_OUT = ROOT / "data/spoofing/news_validation"

MIN_KM = 25.0
SIGNIFICANT_N = 9
CONTROL_DAYS = 28
CONTROL_GAP_DAYS = 2
DEST_RADIUS_DEG = 0.10
R_EARTH_KM = 6371.0

SETTINGS = {"optimize_aggregation_in_order": 0, "max_threads": 10}


def sql_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"non-finite SQL value: {value}")
    return f"{value:.8f}"


def event_windows(row: pd.Series) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(row.screen_start)
    end = pd.Timestamp(row.screen_end)
    gap = pd.Timedelta(days=CONTROL_GAP_DAYS)
    one = pd.Timedelta(days=1)
    return {
        "pre": (
            start - gap - pd.Timedelta(days=CONTROL_DAYS),
            start - gap - one,
        ),
        "event": (start, end),
        "post": (
            end + gap + one,
            end + gap + pd.Timedelta(days=CONTROL_DAYS),
        ),
    }


def window_name(day: pd.Timestamp, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> str:
    for name, (start, end) in windows.items():
        if start <= day <= end:
            return name
    return "gap"


def fetch_bins(row: pd.Series) -> pd.DataFrame:
    windows = event_windows(row)
    query_start = min(v[0] for v in windows.values()).date()
    query_end = max(v[1] for v in windows.values()).date()

    if pd.notna(row.known_dest_lat) and pd.notna(row.known_dest_lon):
        match_expr = f"""
            countIf(greatCircleDistance(
                toFloat64(top_plon) / 100.0,
                toFloat64(top_plat) / 100.0,
                {sql_number(float(row.known_dest_lon))},
                {sql_number(float(row.known_dest_lat))}
            ) <= {sql_number(float(row.known_dest_radius_km) * 1000.0)})
        """
    else:
        match_expr = "toUInt64(0)"

    df = ch_df(
        f"""
        SELECT
            src_lat10,
            src_lon10,
            onset_day,
            count() AS n_onsets,
            uniqExact((mcc, mnc)) AS n_operators,
            uniqExact(cell_type) AS n_tech,
            quantileExact(0.5)(med_km) AS median_km,
            max(max_km) AS max_km,
            quantileExact(0.5)(top_plat) / 100.0 AS destination_lat,
            quantileExact(0.5)(top_plon) / 100.0 AS destination_lon,
            {match_expr} AS known_destination_matches,
            toString(topK(3)((top_plat, top_plon))) AS top_destinations_grid
        FROM spoof.onsets_f
        WHERE onset_day BETWEEN '{query_start}' AND '{query_end}'
          AND src_lat10 BETWEEN {math.floor(float(row.lat_min) * 10)}
                            AND {math.ceil(float(row.lat_max) * 10)}
          AND src_lon10 BETWEEN {math.floor(float(row.lon_min) * 10)}
                            AND {math.ceil(float(row.lon_max) * 10)}
          AND med_km >= {MIN_KM}
        GROUP BY src_lat10, src_lon10, onset_day
        ORDER BY onset_day, n_onsets DESC
        """,
        settings=SETTINGS,
    )
    if df.empty:
        return df
    df["onset_day"] = pd.to_datetime(df["onset_day"])
    df["window"] = df.onset_day.map(lambda d: window_name(d, windows))
    df = df[df.window != "gap"].copy()
    df.insert(0, "event_id", row.event_id)
    df.insert(1, "event_label", row.event_label)
    df["source_lat"] = df.src_lat10 / 10.0
    df["source_lon"] = df.src_lon10 / 10.0
    df["significant"] = df.n_onsets >= SIGNIFICANT_N
    return df


def fetch_static_support(row: pd.Series) -> dict:
    """Measure whether the frozen detector has any reference-cell support here.

    This is intentionally static.  It distinguishes a geographically unsupported
    null from a supported region with no qualifying coordinate-displacement
    onset, without scanning the 63-billion-row raw table for every news event.
    """
    df = ch_df(
        f"""
        SELECT
            count() AS reference_cells,
            uniqExact(mcc) AS reference_mccs,
            uniqExact((mcc, mnc)) AS reference_operators,
            sum(obs) AS reference_observations
        FROM spoof.cellref
        WHERE rlat BETWEEN {math.floor(float(row.lat_min) * 100)}
                       AND {math.ceil(float(row.lat_max) * 100)}
          AND rlon BETWEEN {math.floor(float(row.lon_min) * 100)}
                       AND {math.ceil(float(row.lon_max) * 100)}
        """,
        settings=SETTINGS,
    )
    if df.empty:
        return {
            "reference_cells": 0,
            "reference_mccs": 0,
            "reference_operators": 0,
            "reference_observations": 0,
        }
    return {
        "reference_cells": int(df.iloc[0].reference_cells),
        "reference_mccs": int(df.iloc[0].reference_mccs),
        "reference_operators": int(df.iloc[0].reference_operators),
        "reference_observations": int(df.iloc[0].reference_observations),
    }


def fetch_destination_identities(row: pd.Series) -> pd.DataFrame:
    """Bound the raw follow-up to identities associated with a named decoy."""
    if pd.isna(row.known_dest_lat) or pd.isna(row.known_dest_lon):
        return pd.DataFrame()
    radius_m = float(row.known_dest_radius_km) * 1000.0
    return ch_df(
        f"""
        SELECT DISTINCT mcc, mnc, lac, cid, toString(cell_type) AS cell_type
        FROM
        (
            SELECT mcc, mnc, lac, cid, cell_type
            FROM spoof.onsets_f
            WHERE onset_day BETWEEN '{row.screen_start}' AND '{row.screen_end}'
              AND src_lat10 BETWEEN {math.floor(float(row.lat_min) * 10)}
                                AND {math.ceil(float(row.lat_max) * 10)}
              AND src_lon10 BETWEEN {math.floor(float(row.lon_min) * 10)}
                                AND {math.ceil(float(row.lon_max) * 10)}
              AND med_km >= {MIN_KM}
              AND greatCircleDistance(
                    toFloat64(top_plon) / 100.0,
                    toFloat64(top_plat) / 100.0,
                    {sql_number(float(row.known_dest_lon))},
                    {sql_number(float(row.known_dest_lat))}
                  ) <= {sql_number(radius_m)}
            UNION DISTINCT
            SELECT mcc, mnc, lac, cid, cell_type
            FROM spoof.away
            WHERE rlat BETWEEN {math.floor(float(row.lat_min) * 100)}
                           AND {math.ceil(float(row.lat_max) * 100)}
              AND rlon BETWEEN {math.floor(float(row.lon_min) * 100)}
                           AND {math.ceil(float(row.lon_max) * 100)}
              AND km >= {MIN_KM}
              AND t_first <= toDateTime('{row.screen_end} 23:59:59')
              AND t_last >= toDateTime('{row.screen_start} 00:00:00')
              AND greatCircleDistance(
                    toFloat64(plon) / 100.0,
                    toFloat64(plat) / 100.0,
                    {sql_number(float(row.known_dest_lon))},
                    {sql_number(float(row.known_dest_lat))}
                  ) <= {sql_number(radius_m)}
        )
        ORDER BY mcc, mnc, lac, cid, cell_type
        """,
        settings=SETTINGS,
    )


def fetch_raw_destination_daily(row: pd.Series, identities: pd.DataFrame) -> pd.DataFrame:
    """Count exact raw observations at a named destination for bounded identities."""
    if identities.empty:
        return pd.DataFrame()
    if len(identities) > 2_000:
        raise RuntimeError(
            f"{row.event_id}: {len(identities)} destination identities exceeds raw-query cap"
        )
    keys = ",".join(
        f"({int(r.mcc)},{int(r.mnc)},{int(r.lac)},{int(r.cid)},'{r.cell_type}')"
        for r in identities.itertuples(index=False)
    )
    windows = event_windows(row)
    start = min(v[0] for v in windows.values()).strftime("%Y-%m-%d")
    end_exclusive = (
        max(v[1] for v in windows.values()) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    radius_m = float(row.known_dest_radius_km) * 1000.0
    df = ch_df(
        f"""
        SELECT
            toDate(timestamp) AS day,
            count() AS identity_observations,
            countIf(greatCircleDistance(
                toFloat64(lon), toFloat64(lat),
                {sql_number(float(row.known_dest_lon))},
                {sql_number(float(row.known_dest_lat))}
            ) <= {sql_number(radius_m)}) AS destination_observations,
            uniqExactIf(
                (mcc, mnc, lac, cid, cell_type),
                greatCircleDistance(
                    toFloat64(lon), toFloat64(lat),
                    {sql_number(float(row.known_dest_lon))},
                    {sql_number(float(row.known_dest_lat))}
                ) <= {sql_number(radius_m)}
            ) AS destination_identities
        FROM cell.geos
        PREWHERE (mcc, mnc, lac, cid, toString(cell_type)) IN ({keys})
        WHERE timestamp >= toDateTime('{start}')
          AND timestamp < toDateTime('{end_exclusive}')
          AND lat BETWEEN -90 AND 90 AND lon BETWEEN -180 AND 180
        GROUP BY day
        ORDER BY day
        """,
        settings={"max_threads": 6, "optimize_aggregation_in_order": 0},
    )
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df.day)
    df["window"] = df.day.map(lambda d: window_name(d, windows))
    df = df[df.window != "gap"].copy()
    df.insert(0, "event_id", row.event_id)
    df.insert(1, "event_label", row.event_label)
    df["candidate_identities"] = len(identities)
    return df


def fetch_mechanism(src_lat10: int, src_lon10: int, dlat: float, dlon: float) -> dict:
    df = ch_df(
        f"""
        SELECT plat, plon, rlat, rlon, obs
        FROM spoof.away
        WHERE plat BETWEEN {int((dlat - DEST_RADIUS_DEG) * 100)}
                       AND {int((dlat + DEST_RADIUS_DEG) * 100)}
          AND plon BETWEEN {int((dlon - DEST_RADIUS_DEG) * 100)}
                       AND {int((dlon + DEST_RADIUS_DEG) * 100)}
          AND intDiv(rlat, 10) = {int(src_lat10)}
          AND intDiv(rlon, 10) = {int(src_lon10)}
        """,
        settings=SETTINGS,
    )
    if df.empty:
        return {
            "mechanism_cells": 0,
            "mid_mass": np.nan,
            "cross_over_along": np.nan,
            "baseline_km": np.nan,
            "tier": "R (no measurable axis)",
        }

    h_lat = df.rlat.to_numpy() / 100.0
    h_lon = df.rlon.to_numpy() / 100.0
    p_lat = df.plat.to_numpy() / 100.0
    p_lon = df.plon.to_numpy() / 100.0
    cos0 = np.cos(np.radians(h_lat))
    dx_d = np.radians(dlon - h_lon) * cos0 * R_EARTH_KM
    dy_d = np.radians(dlat - h_lat) * R_EARTH_KM
    dx_p = np.radians(p_lon - h_lon) * cos0 * R_EARTH_KM
    dy_p = np.radians(p_lat - h_lat) * R_EARTH_KM
    baseline = np.hypot(dx_d, dy_d)
    ok = baseline > 1e-6
    ux = np.where(ok, dx_d / np.where(ok, baseline, 1), 0)
    uy = np.where(ok, dy_d / np.where(ok, baseline, 1), 0)
    along = dx_p * ux + dy_p * uy
    cross = np.abs(-dx_p * uy + dy_p * ux)
    weight = np.where(ok, along / np.where(ok, baseline, 1), np.nan)

    obs = df.obs.to_numpy(dtype=int)
    wr = np.repeat(weight, obs)
    cr = np.repeat(cross, obs)
    ar = np.repeat(np.abs(along), obs)
    br = np.repeat(baseline, obs)
    finite = np.isfinite(wr)
    mid_mass = float(((wr[finite] > 0.2) & (wr[finite] < 0.8)).mean()) if finite.any() else np.nan
    median_along = float(np.median(ar)) if len(ar) else np.nan
    cross_frac = float(np.median(cr) / median_along) if median_along else np.nan

    if not np.isfinite(mid_mass):
        tier = "R (no measurable axis)"
    elif np.isfinite(cross_frac) and cross_frac > 0.15:
        tier = "R (off-axis)"
    elif mid_mass >= 0.25:
        tier = "T1 mixture confirmed"
    elif mid_mass >= 0.10:
        tier = "T2 mixture ambiguous"
    else:
        tier = "T3 coherent, not a mixture"

    return {
        "mechanism_cells": len(df),
        "mid_mass": mid_mass,
        "cross_over_along": cross_frac,
        "baseline_km": float(np.median(br)) if len(br) else np.nan,
        "tier": tier,
    }


def summarize(row: pd.Series, bins: pd.DataFrame, support: dict) -> dict:
    windows = event_windows(row)
    result: dict[str, object] = {
        "event_id": row.event_id,
        "event_label": row.event_label,
        "phenomenon": row.phenomenon,
        "dataset_testability": row.dataset_testability,
        "screen_start": row.screen_start,
        "screen_end": row.screen_end,
        **support,
    }
    for name, (start, end) in windows.items():
        sub = bins[bins.window == name] if not bins.empty else bins
        days = (end - start).days + 1
        result[f"{name}_days"] = days
        result[f"{name}_onsets"] = int(sub.n_onsets.sum()) if not sub.empty else 0
        result[f"{name}_onsets_per_day"] = (
            float(sub.n_onsets.sum()) / days if days else np.nan
        )
        result[f"{name}_bins_ge3"] = int((sub.n_onsets >= 3).sum()) if not sub.empty else 0
        result[f"{name}_significant_bins"] = int(sub.significant.sum()) if not sub.empty else 0
        result[f"{name}_max_bin"] = int(sub.n_onsets.max()) if not sub.empty else 0
        result[f"{name}_known_destination_matches"] = (
            int(sub.known_destination_matches.sum()) if not sub.empty else 0
        )

    control_rate = np.mean([
        result["pre_onsets_per_day"], result["post_onsets_per_day"]
    ])
    event_rate = float(result["event_onsets_per_day"])
    result["control_onsets_per_day"] = float(control_rate)
    result["event_control_rate_ratio"] = (
        event_rate / control_rate if control_rate > 0 else np.nan
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-mechanism", action="store_true")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    all_bins: list[pd.DataFrame] = []
    raw_destination_frames: list[pd.DataFrame] = []
    destination_summaries: list[dict] = []

    for row in events.itertuples(index=False):
        series = pd.Series(row._asdict())
        print(f"screening {series.event_id} ...", flush=True)
        bins = fetch_bins(series)
        support = fetch_static_support(series)
        summaries.append(summarize(series, bins, support))
        if not bins.empty:
            all_bins.append(bins)
        if pd.notna(series.known_dest_lat) and pd.notna(series.known_dest_lon):
            identities = fetch_destination_identities(series)
            print(
                f"  raw destination follow-up: {len(identities)} identities",
                flush=True,
            )
            raw_daily = fetch_raw_destination_daily(series, identities)
            if not raw_daily.empty:
                raw_destination_frames.append(raw_daily)
            destination_result = {
                "event_id": series.event_id,
                "event_label": series.event_label,
                "known_destination": series.known_destination,
                "candidate_identities": len(identities),
            }
            for name, (start, end) in event_windows(series).items():
                sub = raw_daily[raw_daily.window == name] if not raw_daily.empty else raw_daily
                days = (end - start).days + 1
                destination_observations = (
                    int(sub.destination_observations.sum()) if not sub.empty else 0
                )
                destination_result[f"{name}_days"] = days
                destination_result[f"{name}_observations"] = destination_observations
                destination_result[f"{name}_observations_per_day"] = (
                    destination_observations / days if days else np.nan
                )
                destination_result[f"{name}_max_identities_per_day"] = (
                    int(sub.destination_identities.max()) if not sub.empty else 0
                )
            # These are durable coordinate assignments, so the post-window is a
            # persistence check rather than a valid negative control.  Compare
            # exact raw activity only to the pre-event baseline.
            baseline = destination_result["pre_observations_per_day"]
            destination_result["pre_baseline_observations_per_day"] = baseline
            destination_result["event_pre_rate_ratio"] = (
                destination_result["event_observations_per_day"] / baseline
                if baseline > 0 else (
                    np.inf if destination_result["event_observations_per_day"] > 0 else np.nan
                )
            )
            destination_summaries.append(destination_result)

    candidate_bins = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()
    event_candidates = candidate_bins[
        (candidate_bins.window == "event") & (candidate_bins.n_onsets >= 3)
    ].copy()

    if not args.skip_mechanism and not event_candidates.empty:
        mechanism_rows = []
        significant = event_candidates[event_candidates.significant].copy()
        for candidate in significant.itertuples(index=False):
            print(
                f"mechanism {candidate.event_id} {candidate.onset_day.date()} "
                f"{candidate.src_lat10}/{candidate.src_lon10} ...",
                flush=True,
            )
            mechanism_rows.append({
                "event_id": candidate.event_id,
                "onset_day": candidate.onset_day,
                "src_lat10": candidate.src_lat10,
                "src_lon10": candidate.src_lon10,
                **fetch_mechanism(
                    int(candidate.src_lat10),
                    int(candidate.src_lon10),
                    float(candidate.destination_lat),
                    float(candidate.destination_lon),
                ),
            })
        mechanism = pd.DataFrame(mechanism_rows)
        if not mechanism.empty:
            event_candidates = event_candidates.merge(
                mechanism,
                on=["event_id", "onset_day", "src_lat10", "src_lon10"],
                how="left",
            )

    summary_df = pd.DataFrame(summaries)
    if "tier" in event_candidates.columns:
        tier_counts = (
            event_candidates[event_candidates.significant]
            .assign(
                t1=lambda d: d.tier.eq("T1 mixture confirmed"),
                t2=lambda d: d.tier.eq("T2 mixture ambiguous"),
                t3=lambda d: d.tier.eq("T3 coherent, not a mixture"),
                rejected=lambda d: d.tier.str.startswith("R (", na=False),
            )
            .groupby("event_id")[["t1", "t2", "t3", "rejected"]]
            .sum()
            .rename(columns=lambda c: f"{c}_bins")
        )
        summary_df = summary_df.merge(tier_counts, on="event_id", how="left")
        for column in ["t1_bins", "t2_bins", "t3_bins", "rejected_bins"]:
            summary_df[column] = summary_df[column].fillna(0).astype(int)
        summary_df["detector_outcome"] = np.select(
            [
                summary_df.t1_bins.gt(0),
                summary_df.t2_bins.gt(0),
                summary_df.t3_bins.gt(0),
                summary_df.event_significant_bins.gt(0),
            ],
            [
                "mixture-consistent synchronized displacement",
                "mixture-ambiguous synchronized displacement",
                "coherent endpoint displacement; not a mixture",
                "significant synchronized displacement rejected by geometry",
            ],
            default="no significant synchronized displacement",
        )
    else:
        summary_df["detector_outcome"] = "mechanism test skipped"
    summary_df.to_csv(args.out / "event_screen_summary.csv", index=False)
    candidate_bins.to_csv(args.out / "event_control_bins.csv", index=False)
    event_candidates.to_csv(args.out / "event_candidate_bins.csv", index=False)
    raw_destination_daily = (
        pd.concat(raw_destination_frames, ignore_index=True)
        if raw_destination_frames else pd.DataFrame()
    )
    raw_destination_daily.to_csv(args.out / "raw_destination_daily.csv", index=False)
    pd.DataFrame(destination_summaries).to_csv(
        args.out / "raw_destination_summary.csv", index=False
    )
    print(f"wrote {len(summary_df)} event summaries and {len(event_candidates)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
