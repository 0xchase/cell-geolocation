#!/usr/bin/env python3
"""Export and validate raw histories for the strongest detector survivors.

Selections are structural, not hand-entered identity lists: two static
repeated-CID/LAC-rotation families and two supported local-motion campaigns.
The remote query is a bounded exact-key read through ``ch_remote``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch_remote import ch_df  # noqa: E402
from detect_rogue_bts_families import OUT, haversine  # noqa: E402


KEYS = ["mcc", "mnc", "lac", "cid", "cell_type"]


def selected_keys() -> pd.DataFrame:
    repeated = pd.read_csv(OUT / "foreign_far_candidate_members.csv")
    mobile = pd.read_csv(OUT / "local_mobile_identity_scores.csv")
    groups = []
    for label, condition in [
        ("estepona_kw_419_02_gsm_cid1971", repeated.mcc.eq(419) & repeated.mnc.eq(2)
         & repeated.cell_type.eq("gsm") & repeated.cid.eq(1971)),
        ("krugersdorp_zm_645_02_gsm_cid2730", repeated.mcc.eq(645) & repeated.mnc.eq(2)
         & repeated.cell_type.eq("gsm") & repeated.cid.eq(2730)),
    ]:
        part = repeated.loc[condition, KEYS].drop_duplicates().copy()
        part["case_label"] = label
        groups.append(part)
    for label, condition in [
        ("northwest_syria_cn_460_00_lte_lac123", mobile.mcc.eq(460) & mobile.mnc.eq(0)
         & mobile.cell_type.eq("lte") & mobile.lac.eq(123)
         & mobile.endpoint_a_country_iso.eq("SY")),
        ("eastern_shan_hk_454_03_lte_lac12596", mobile.mcc.eq(454) & mobile.mnc.eq(3)
         & mobile.cell_type.eq("lte") & mobile.lac.eq(12596)
         & mobile.endpoint_a_country_iso.eq("MM")),
    ]:
        part = mobile.loc[condition, KEYS].drop_duplicates().copy()
        part["case_label"] = label
        groups.append(part)
    return pd.concat(groups, ignore_index=True)


def query_history(keys: pd.DataFrame) -> pd.DataFrame:
    tuples = ",\n".join(
        f"({int(r.mcc)},{int(r.mnc)},{int(r.lac)},{int(r.cid)},'{r.cell_type}')"
        for r in keys.itertuples(index=False)
    )
    query = f"""
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,lat,lon,timestamp
FROM cell.geos
WHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({tuples})
ORDER BY mcc,mnc,lac,cid,cell_type,timestamp
"""
    history = ch_df(query, settings={"max_threads": 8})
    return history.merge(keys, on=KEYS, how="inner")


def validate(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in history.groupby(["case_label", *KEYS], sort=False):
        group = group.sort_values("timestamp")
        transitions = []
        speeds = []
        previous = None
        for row in group.itertuples(index=False):
            current = (float(row.lat), float(row.lon), pd.Timestamp(row.timestamp))
            if previous and current[:2] != previous[:2]:
                distance = haversine(previous[0], previous[1], current[0], current[1])
                hours = (current[2] - previous[2]).total_seconds() / 3600
                transitions.append(distance)
                if hours > 0:
                    speeds.append(distance / hours)
            previous = current
        rows.append({
            **dict(zip(["case_label", *KEYS], key)),
            "raw_observations": len(group),
            "raw_position_count": group[["lat", "lon"]].drop_duplicates().shape[0],
            "first_seen": group.timestamp.min(),
            "last_seen": group.timestamp.max(),
            "largest_transition_km": max(transitions, default=0),
            "maximum_observed_transition_speed_kmh": max(speeds, default=0),
            "total_observed_transition_km": sum(transitions),
        })
    return pd.DataFrame(rows)


def main() -> None:
    keys = selected_keys()
    history = query_history(keys)
    metrics = validate(history)
    keys.to_csv(OUT / "case_history_keys.csv", index=False)
    history.to_csv(OUT / "case_raw_history.csv.gz", index=False, compression="gzip")
    metrics.to_csv(OUT / "case_raw_history_metrics.csv", index=False)
    summary = metrics.groupby("case_label").agg(
        identities=("cid", "size"),
        raw_observations=("raw_observations", "sum"),
        identities_with_multiple_positions=("raw_position_count", lambda x: int((x > 1).sum())),
        maximum_transition_km=("largest_transition_km", "max"),
        maximum_observed_speed_kmh=("maximum_observed_transition_speed_kmh", "max"),
        first_seen=("first_seen", "min"),
        last_seen=("last_seen", "max"),
    )
    summary.to_csv(OUT / "case_raw_history_summary.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
