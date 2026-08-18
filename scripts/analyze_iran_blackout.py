#!/usr/bin/env python3
"""Audit whether the June 2025 Iranian blackout is visible in collection data.

The raw table is queried read-only.  Iranian daily identity visibility is
normalized by seven regional and distant control MCCs so synchronized crawler
gaps are not mistaken for an Iran-specific communications outage.  The extract
continues through mid-September so the audit can share an axis with the full
2025 Iran event timeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "spoofing" / "iran_blackout_audit_daily.csv"
AUDIT_START = "2025-06-01"
AUDIT_END_EXCLUSIVE = "2025-09-16"
MCC_NAMES = {
    255: "Ukraine",
    418: "Iraq",
    419: "Kuwait",
    420: "Saudi Arabia",
    424: "United Arab Emirates",
    426: "Bahrain",
    427: "Qatar",
    432: "Iran",
}
CONTROLS = [255, 418, 419, 420, 424, 426, 427]


def extract() -> pd.DataFrame:
    frame = ch_df(
        """
        SELECT
            toDate(timestamp) AS day,
            mcc,
            count() AS observations,
            uniqExact(tuple(mnc,lac,cid,cell_type)) AS identities
        FROM cell.geos
        PREWHERE mcc IN (255,418,419,420,424,426,427,432)
        WHERE timestamp >= toDateTime('2025-06-01 00:00:00')
          AND timestamp <  toDateTime('2025-09-16 00:00:00')
        GROUP BY day,mcc
        ORDER BY day,mcc
        """,
        settings={"optimize_aggregation_in_order": 0},
    )
    frame["day"] = pd.to_datetime(frame["day"])
    full_index = pd.MultiIndex.from_product(
        [pd.date_range(AUDIT_START, pd.Timestamp(AUDIT_END_EXCLUSIVE) - pd.Timedelta(days=1), freq="D"), MCC_NAMES],
        names=["day", "mcc"],
    )
    frame = (
        frame.set_index(["day", "mcc"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    frame["country"] = frame["mcc"].map(MCC_NAMES)
    control = (
        frame[frame["mcc"].isin(CONTROLS)]
        .groupby("day")["identities"].sum()
    )
    iran = frame[frame["mcc"].eq(432)].set_index("day")["identities"]
    share = iran.div(control.where(control.ne(0)))
    frame["control_identities"] = frame["day"].map(control)
    frame["iran_to_control_ratio"] = frame["day"].map(share)
    frame["period"] = "outside audit window"
    frame.loc[frame["day"].between("2025-06-11", "2025-06-17"), "period"] = "pre-blackout"
    frame.loc[frame["day"].between("2025-06-18", "2025-06-25"), "period"] = "blackout"
    frame.loc[frame["day"].between("2025-06-26", "2025-07-05"), "period"] = "post-blackout"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    return frame


def summarize(frame: pd.DataFrame) -> None:
    daily = frame[frame["mcc"].eq(432)].set_index("day")
    medians = daily.groupby("period")["iran_to_control_ratio"].median()
    pre = float(medians["pre-blackout"])
    during = float(medians["blackout"])
    print(f"pre-blackout median Iran/control ratio: {pre:.6f}")
    print(f"blackout median Iran/control ratio:     {during:.6f}")
    print(f"relative change:                         {(during / pre - 1) * 100:.2f}%")
    missing = daily.index[daily["identities"].eq(0)].strftime("%Y-%m-%d").tolist()
    print(f"zero-Iran days: {', '.join(missing) if missing else 'none'}")
    if missing:
        all_zero = []
        for day in pd.to_datetime(missing):
            rows = frame[frame["day"].eq(day)]
            if rows["identities"].eq(0).all():
                all_zero.append(day.strftime("%Y-%m-%d"))
        print(f"zero days shared by every control MCC: {', '.join(all_zero) if all_zero else 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.refresh or not OUTPUT.exists():
        frame = extract()
    else:
        frame = pd.read_csv(OUTPUT, parse_dates=["day"])
    summarize(frame)
    print(OUTPUT)


if __name__ == "__main__":
    main()
