#!/usr/bin/env python3
"""Extract read-only position data for the criminal-activity anomaly figure."""

from pathlib import Path

import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rogue-bts-detector"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    query = """
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
       plat / 100.0 AS latitude, plon / 100.0 AS longitude,
       obs,first_seen,last_seen
FROM cell.cellpos
WHERE (mcc=454 AND mnc=3 AND lac=12596 AND toString(cell_type)='lte')
   OR (mcc=455 AND mnc=1 AND lac IN (12580,13880) AND toString(cell_type)='lte')
ORDER BY mcc,mnc,lac,cid,first_seen
"""
    shan = ch_df(query, settings={"max_threads": 4})
    shan.to_csv(OUT / "eastern_shan_foreign_plmn_positions.csv", index=False)

    raw = pd.read_csv(OUT / "case_raw_history.csv.gz", compression="gzip")
    raw = raw[raw.case_label.isin([
        "estepona_kw_419_02_gsm_cid1971",
        "krugersdorp_zm_645_02_gsm_cid2730",
    ])].copy()
    raw.to_csv(OUT / "static_criminal_anomaly_raw_history.csv", index=False)
    print(f"Eastern Shan position rows: {len(shan):,}")
    print(f"Static anomaly raw rows: {len(raw):,}")


if __name__ == "__main__":
    main()
