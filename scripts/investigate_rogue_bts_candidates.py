#!/usr/bin/env python3
"""Validate high-scoring repeated-CID leads against identity and peer data.

The input is the global detector output.  This script expands only clusters
that are both foreign and at least 200 km outside the PLMN's assigned area,
then retrieves three read-only follow-ups:

* the exact summary rows comprising each cluster;
* other cellular identities sharing each candidate LAC, grouped by country;
* every aggregate position for the candidate identities.

Nearby bins are joined into phenomena for review, but no phenomenon is labeled
as rogue or criminal.  The outputs are an auditable triage corpus.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch_remote import ch_df  # noqa: E402
from detect_rogue_bts_families import (  # noqa: E402
    OUT,
    assigned_countries,
    haversine,
    load_plmn_countries,
)


SCORES = OUT / "repeated_cid_family_scores.csv"


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def candidate_conditions(leads: pd.DataFrame) -> str:
    terms = []
    for row in leads.itertuples(index=False):
        terms.append(
            "(mcc={mcc} AND mnc={mnc} AND toString(cell_type)={tech} "
            "AND cid={cid} AND toString(round(glat,1))={lat} "
            "AND toString(round(glon,1))={lon})".format(
                mcc=int(row.mcc), mnc=int(row.mnc), tech=sql_string(row.cell_type),
                cid=int(row.cid), lat=sql_string(f"{float(row.dst_lat):g}"),
                lon=sql_string(f"{float(row.dst_lon):g}"),
            )
        )
    return " OR\n".join(terms)


def add_phenomena(leads: pd.DataFrame, join_km: float = 25.0) -> pd.DataFrame:
    """Join adjacent spatial bins; families at one place become one lead."""
    leads = leads.reset_index(drop=True).copy()
    parent = list(range(len(leads)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    for i in range(len(leads)):
        for j in range(i):
            if leads.loc[i, "host_country_iso"] != leads.loc[j, "host_country_iso"]:
                continue
            if haversine(
                leads.loc[i, "dst_lat"], leads.loc[i, "dst_lon"],
                leads.loc[j, "dst_lat"], leads.loc[j, "dst_lon"],
            ) <= join_km:
                union(i, j)
    roots = [find(i) for i in range(len(leads))]
    ordering = {root: n + 1 for n, root in enumerate(dict.fromkeys(roots))}
    leads["phenomenon_id"] = [f"FBS-{ordering[root]:02d}" for root in roots]
    return leads


def expand_members(leads: pd.DataFrame) -> pd.DataFrame:
    query = f"""
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
       first_seen,last_seen,obs,glat,glon,first_lat,first_lon,
       lat_min,lat_max,lon_min,lon_max,n_pos,country_iso,country,
       region,state,county,city,suburb,road
FROM cell.summary_full
WHERE {candidate_conditions(leads)}
ORDER BY mcc,mnc,lac,cid,cell_type
"""
    members = ch_df(query, settings={"max_threads": 8})
    lookup = {
        (int(r.mcc), int(r.mnc), r.cell_type, int(r.cid),
         round(float(r.dst_lat), 1), round(float(r.dst_lon), 1)): r.phenomenon_id
        for r in leads.itertuples(index=False)
    }
    members["phenomenon_id"] = [
        lookup[(int(r.mcc), int(r.mnc), r.cell_type, int(r.cid),
                round(float(r.glat), 1), round(float(r.glon), 1))]
        for r in members.itertuples(index=False)
    ]
    return members


def peer_country_counts(members: pd.DataFrame) -> pd.DataFrame:
    operator_lacs: dict[tuple[int, int, str], set[int]] = {}
    for row in members.itertuples(index=False):
        operator_lacs.setdefault((int(row.mcc), int(row.mnc), row.cell_type), set()).add(
            int(row.lac)
        )
    clauses = []
    for (mcc, mnc, technology), lacs in operator_lacs.items():
        lac_sql = ",".join(map(str, sorted(lacs)))
        clauses.append(
            f"(mcc={mcc} AND mnc={mnc} AND toString(cell_type)={sql_string(technology)} "
            f"AND lac IN ({lac_sql}))"
        )
    query = f"""
SELECT mcc,mnc,lac,toString(cell_type) AS cell_type,country_iso,
       count() AS peer_identities,sum(obs) AS peer_observations,
       uniqExact(cid) AS peer_cids,min(first_seen) AS peer_first_seen,
       max(last_seen) AS peer_last_seen
FROM cell.summary_full
WHERE {' OR '.join(clauses)}
GROUP BY mcc,mnc,lac,cell_type,country_iso
ORDER BY mcc,mnc,lac,cell_type,peer_identities DESC
"""
    return ch_df(query, settings={"max_threads": 8})


def candidate_positions(members: pd.DataFrame) -> pd.DataFrame:
    keys = sorted({
        (int(r.mcc), int(r.mnc), int(r.lac), int(r.cid), r.cell_type)
        for r in members.itertuples(index=False)
    })
    tuple_sql = ",\n".join(
        f"({mcc},{mnc},{lac},{cid},{sql_string(technology)})"
        for mcc, mnc, lac, cid, technology in keys
    )
    query = f"""
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
       plat / 100.0 AS latitude,plon / 100.0 AS longitude,
       obs,first_seen,last_seen
FROM cell.cellpos
WHERE (mcc,mnc,lac,cid,toString(cell_type)) IN ({tuple_sql})
ORDER BY mcc,mnc,lac,cid,cell_type,first_seen
"""
    return ch_df(query, settings={"max_threads": 8})


def host_suffix_matches(members: pd.DataFrame) -> pd.DataFrame:
    """Find host-country cells reusing a candidate's LAC/CID suffix."""
    by_plmn, by_mcc = load_plmn_countries()
    host_mccs: dict[str, set[int]] = {}
    for mcc, isos in by_mcc.items():
        for iso in isos:
            host_mccs.setdefault(iso, set()).add(mcc)
    member_host = members.groupby("phenomenon_id").country_iso.agg(
        lambda values: next((str(v) for v in values if str(v)), "")
    )
    clauses = []
    for phenomenon, group in members.groupby("phenomenon_id"):
        mccs = host_mccs.get(member_host[phenomenon], set())
        if not mccs:
            continue
        suffixes = sorted({(int(r.lac), int(r.cid), r.cell_type) for r in group.itertuples()})
        suffix_sql = ",".join(
            f"({lac},{cid},{sql_string(technology)})" for lac, cid, technology in suffixes
        )
        clauses.append(
            f"(mcc IN ({','.join(map(str, sorted(mccs)))}) AND "
            f"(lac,cid,toString(cell_type)) IN ({suffix_sql}))"
        )
    query = f"""
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type,
       glat,glon,country_iso,obs,first_seen,last_seen
FROM cell.summary_full
WHERE {' OR '.join(clauses)}
ORDER BY mcc,mnc,lac,cid,cell_type
"""
    return ch_df(query, settings={"max_threads": 8})


def summarize(
    leads: pd.DataFrame, members: pd.DataFrame, peers: pd.DataFrame,
    positions: pd.DataFrame, suffix_matches: pd.DataFrame,
) -> pd.DataFrame:
    by_plmn, by_mcc = load_plmn_countries()
    member_to_phenomenon = {
        (int(r.mcc), int(r.mnc), int(r.lac), int(r.cid), r.cell_type): r.phenomenon_id
        for r in members.itertuples(index=False)
    }
    positions["phenomenon_id"] = [
        member_to_phenomenon[(int(r.mcc), int(r.mnc), int(r.lac), int(r.cid), r.cell_type)]
        for r in positions.itertuples(index=False)
    ]
    home_lacs: set[tuple[int, int, int, str]] = set()
    for row in peers.itertuples(index=False):
        homes = assigned_countries(row.mcc, row.mnc, by_plmn, by_mcc)
        if row.country_iso in homes and row.peer_identities > 0:
            home_lacs.add((int(row.mcc), int(row.mnc), int(row.lac), row.cell_type))
    members["lac_has_home_country_peer"] = [
        (int(r.mcc), int(r.mnc), int(r.lac), r.cell_type) in home_lacs
        for r in members.itertuples(index=False)
    ]
    host_suffixes = {
        (int(r.lac), int(r.cid), r.cell_type)
        for r in suffix_matches.itertuples(index=False)
    }
    members["suffix_exists_under_host_mcc"] = [
        (int(r.lac), int(r.cid), r.cell_type) in host_suffixes
        for r in members.itertuples(index=False)
    ]

    rows = []
    for phenomenon, group in members.groupby("phenomenon_id"):
        lead_group = leads[leads.phenomenon_id.eq(phenomenon)]
        pos = positions[positions.phenomenon_id.eq(phenomenon)]
        family_density = []
        for _, family in group.groupby(["mcc", "mnc", "cell_type", "cid"]):
            width = int(family.lac.max()) - int(family.lac.min()) + 1
            family_density.append(family.lac.nunique() / width if width else 1.0)
        rows.append({
            "phenomenon_id": phenomenon,
            "host_country_iso": ";".join(sorted(set(lead_group.host_country_iso))),
            "center_latitude": np.average(group.glat, weights=group.obs),
            "center_longitude": np.average(group.glon, weights=group.obs),
            "plmns": ";".join(sorted({f"{r.mcc}/{r.mnc}" for r in group.itertuples()})),
            "families": lead_group[["mcc", "mnc", "cell_type", "cid"]].drop_duplicates().shape[0],
            "identities": len(group),
            "distinct_lacs": group[["mcc", "mnc", "cell_type", "lac"]].drop_duplicates().shape[0],
            "crawler_observations": int(group.obs.sum()),
            "multi_position_identities": int(group.n_pos.gt(1).sum()),
            "position_rows": len(pos),
            "distinct_position_bins": pos[["latitude", "longitude"]].drop_duplicates().shape[0],
            "first_seen": group.first_seen.min(),
            "last_seen": group.last_seen.max(),
            "median_identity_lifespan_days": pd.to_timedelta(
                pd.to_datetime(group.last_seen) - pd.to_datetime(group.first_seen)
            ).dt.total_seconds().median() / 86400,
            "lacs_with_home_country_peers": int(group.lac_has_home_country_peer.sum()),
            "fraction_lacs_with_home_country_peers": float(group.lac_has_home_country_peer.mean()),
            "host_mcc_suffix_matches": int(group.suffix_exists_under_host_mcc.sum()),
            "fraction_host_mcc_suffix_matches": float(group.suffix_exists_under_host_mcc.mean()),
            "fraction_lacs_294xx": float(group.lac.between(29400, 29499).mean()),
            "maximum_family_lac_interval_density": max(family_density),
            "max_detector_score": float(lead_group.detector_score.max()),
        })
    return pd.DataFrame(rows).sort_values(
        ["max_detector_score", "identities"], ascending=False
    )


def als_samples(members: pd.DataFrame, phenomena: pd.DataFrame) -> pd.DataFrame:
    """Select three candidates and three nearby ordinary controls per survivor."""
    survivors = phenomena[
        phenomena.fraction_lacs_294xx.lt(0.8) & phenomena.identities.ge(50)
    ].phenomenon_id.tolist()
    by_plmn, by_mcc = load_plmn_countries()
    host_mccs: dict[str, set[int]] = {}
    for mcc, isos in by_mcc.items():
        for iso in isos:
            host_mccs.setdefault(iso, set()).add(mcc)
    selected = []
    for phenomenon in survivors:
        group = members[members.phenomenon_id.eq(phenomenon)].sort_values(
            ["mcc", "mnc", "cid", "lac"]
        )
        indices = sorted(set(np.linspace(0, len(group) - 1, 3).round().astype(int)))
        for row in group.iloc[indices].itertuples(index=False):
            selected.append({
                "sample_kind": "candidate", "phenomenon_id": phenomenon,
                "mcc": int(row.mcc), "mnc": int(row.mnc), "lac": int(row.lac),
                "cid": int(row.cid), "cell_type": row.cell_type,
            })
        summary = phenomena[phenomena.phenomenon_id.eq(phenomenon)].iloc[0]
        host_iso = summary.host_country_iso
        mccs = set(host_mccs.get(host_iso, set()))
        if 36.7 < summary.center_latitude < 37.1 and 38.1 < summary.center_longitude < 38.6:
            # Kobani is in Syria despite the border polygon/reverse-geocode label.
            mccs |= {417}
        if not mccs:
            continue
        lat, lon = summary.center_latitude, summary.center_longitude
        query = f"""
SELECT mcc,mnc,lac,cid,toString(cell_type) AS cell_type
FROM cell.summary_full
WHERE mcc IN ({','.join(map(str, sorted(mccs)))})
  AND glat BETWEEN {lat - 0.15} AND {lat + 0.15}
  AND glon BETWEEN {lon - 0.15} AND {lon + 0.15}
  AND cid >= 0 AND obs >= 10
ORDER BY greatCircleDistance(glon,glat,{lon},{lat}),obs DESC
LIMIT 3
"""
        controls = ch_df(query, settings={"max_threads": 4})
        for row in controls.itertuples(index=False):
            selected.append({
                "sample_kind": "nearby_host_control", "phenomenon_id": phenomenon,
                "mcc": int(row.mcc), "mnc": int(row.mnc), "lac": int(row.lac),
                "cid": int(row.cid), "cell_type": row.cell_type,
            })
    return pd.DataFrame(selected)


def main() -> None:
    leads = pd.read_csv(SCORES)
    leads = add_phenomena(leads[leads.foreign & leads.far_outside].copy())
    members = expand_members(leads)
    peers = peer_country_counts(members)
    positions = candidate_positions(members)
    suffix_matches = host_suffix_matches(members)
    phenomena = summarize(leads, members, peers, positions, suffix_matches)
    samples = als_samples(members, phenomena)
    leads.to_csv(OUT / "foreign_far_cluster_leads.csv", index=False)
    members.to_csv(OUT / "foreign_far_candidate_members.csv", index=False)
    peers.to_csv(OUT / "foreign_far_lac_peer_countries.csv", index=False)
    positions.to_csv(OUT / "foreign_far_candidate_positions.csv", index=False)
    suffix_matches.to_csv(OUT / "foreign_far_host_suffix_matches.csv", index=False)
    samples.to_csv(OUT / "als_query_sample.csv", index=False)
    phenomena.to_csv(OUT / "foreign_far_phenomena.csv", index=False)
    print(phenomena.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
