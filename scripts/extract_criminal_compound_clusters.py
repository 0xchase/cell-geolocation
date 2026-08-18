#!/usr/bin/env python3
"""Extract the Cambodia 404/01 compound-cluster evidence read-only."""

from pathlib import Path

import numpy as np
import pandas as pd

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity"


NEW_CONFIRMED = [
    ("BA01", 11.0793, 106.1689), ("BA08", 11.0734, 106.1718),
    ("BM02", 13.5506, 102.5692), ("BM03", 13.7471, 102.7406),
    ("BM04", 13.7649, 102.7021), ("BM05", 13.6786, 102.6198),
    ("CT08", 10.9396, 105.0260), ("KA03", 10.6665, 104.0162),
    ("KA04", 10.6275, 104.0486), ("MD01", 12.4338, 107.1882),
    ("MD02", 12.3891, 107.3124), ("PO17", 13.6495, 102.5977),
    ("PO21", 13.6674, 102.5755), ("PO22", 13.6721, 102.5730),
    ("PO23", 13.6676, 102.5646), ("PO24", 13.6787, 102.5685),
    ("PO25", 13.6604, 102.5545), ("PO26", 13.6706, 102.5772),
    ("PP19", 11.5702, 104.8602), ("PSP02", 12.9332, 102.4934),
    ("PV01", 11.6129, 105.7854), ("PV02", 10.9664, 105.4189),
    ("SI03", 10.5733, 103.5578), ("SI29", 10.5905, 103.5398),
    ("SI52", 10.6323, 103.5076), ("SI54", 10.6411, 103.5362),
    ("SV04", 10.9991, 106.1904), ("TBK02", 11.6427, 105.8266),
    ("TBK03", 11.6733, 105.9745), ("TBK04", 11.6901, 105.9706),
    ("TBK05", 11.7138, 105.9638), ("TBK08", 11.6686, 105.9789),
    ("TD01H", 12.1211, 102.7455),
]

SUSPECTED = [
    ("BA04", 11.0742, 106.1640), ("BA07", 11.0680, 106.1698),
    ("BA10", 11.0676, 106.1614), ("BA19", 11.0788, 106.1638),
    ("BA21", 11.0607, 106.1791), ("BA22", 11.0623, 106.1776),
    ("BA23", 11.0752, 106.1512), ("BT01", 13.1974, 102.4016),
    ("CH02", 14.3405, 104.0587), ("CT06", 10.9594, 105.0607),
    ("CT07", 10.9555, 105.0594), ("CT09", 10.9564, 105.0599),
    ("CT10", 10.9536, 105.0593), ("CT11", 10.9420, 105.0543),
    ("CT12", 10.9784, 105.0417), ("KCC01", 11.8855, 104.7094),
    ("PO02", 13.6521, 102.5582), ("PO11", 13.6635, 102.5562),
    ("PP02", 11.4530, 104.8561), ("PP11", 11.4107, 104.6514),
    ("PP14", 11.5985, 104.8683), ("PP21", 11.6048, 104.8691),
    ("PSP06", 12.9227, 102.4968), ("SI05", 10.5964, 103.6315),
    ("SI07", 10.6099, 103.5349), ("SI10", 10.6162, 103.5652),
    ("SI14", 10.6323, 103.5068), ("SI17", 10.6246, 103.5009),
    ("SI18", 10.6250, 103.4977), ("SI23", 10.6150, 103.5271),
    ("SI31", 10.6058, 103.5287), ("SI38", 10.6214, 103.5126),
    ("SI41", 10.6212, 103.5019), ("SV01", 10.9829, 106.1954),
    ("SV02", 10.9292, 106.1441), ("TBK01", 11.6419, 105.8159),
]


def nearest(points: pd.DataFrame, sites: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lat = np.radians(points.glat.to_numpy())[:, None]
    lon = np.radians(points.glon.to_numpy())[:, None]
    slat = np.radians(sites.latitude.to_numpy())[None, :]
    slon = np.radians(sites.longitude.to_numpy())[None, :]
    q = np.sin((slat - lat) / 2) ** 2
    q += np.cos(lat) * np.cos(slat) * np.sin((slon - lon) / 2) ** 2
    distance = 6371.0088 * 2 * np.arcsin(np.sqrt(q)) * 1000
    index = distance.argmin(axis=1)
    return sites.site_id.to_numpy()[index], distance.min(axis=1)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(DATA / "cambodia_verified_scam_compounds.csv")
    old = old[["site_id", "latitude", "longitude"]]
    confirmed = pd.concat([
        old,
        pd.DataFrame(NEW_CONFIRMED, columns=["site_id", "latitude", "longitude"]),
    ]).drop_duplicates("site_id", keep="last")
    suspected = pd.DataFrame(SUSPECTED, columns=["site_id", "latitude", "longitude"])
    confirmed.to_csv(DATA / "cambodia_compounds_86.csv", index=False)

    identities = ch_df("""
        SELECT cid,glat,glon,obs,n_pos,first_seen,last_seen
        FROM cell.summary_full
        WHERE mcc=404 AND mnc=1 AND cell_type='lte' AND lac=0
        ORDER BY glat,glon,cid
    """)
    identities["confirmed_site"], identities["confirmed_m"] = nearest(identities, confirmed)
    identities["suspected_site"], identities["suspected_m"] = nearest(identities, suspected)
    identities["cluster"] = np.where(
        identities.confirmed_m <= identities.suspected_m,
        identities.confirmed_site,
        identities.suspected_site,
    )
    identities["cluster_status"] = np.where(
        identities.confirmed_m <= identities.suspected_m, "confirmed", "suspected"
    )
    identities["cluster_m"] = np.minimum(identities.confirmed_m, identities.suspected_m)
    identities.to_csv(DATA / "cambodia_40401_lac0_identities.csv", index=False)

    raw = ch_df("""
        SELECT cid,toDate(timestamp) AS day,count() AS observations
        FROM cell.geos
        WHERE mcc=404 AND mnc=1 AND lac=0 AND cell_type='lte'
        GROUP BY cid,day
        ORDER BY day,cid
    """)
    raw = raw.merge(identities[["cid", "cluster", "cluster_status"]], on="cid", how="left")
    raw["month"] = raw.day.str[:7]
    monthly = raw.groupby(["cluster_status", "cluster", "month"], as_index=False).agg(
        active_identities=("cid", "nunique"),
        observations=("observations", "sum"),
    )
    monthly.to_csv(DATA / "cambodia_40401_cluster_monthly.csv", index=False)

    enrichment = pd.DataFrame([
        (250, 31, 9.587070215974684, 5.8158782715889215e-12),
        (500, 34, 15.800328517047095, 3.6914579856145955e-08),
        (750, 36, 21.350421220782177, 1.470994131817587e-05),
    ], columns=["radius_m", "observed_identities", "matched_expected", "one_sided_p"])
    enrichment.to_csv(DATA / "cambodia_40401_compound_enrichment.csv", index=False)
    print(f"Wrote {len(identities)} identities and {len(raw)} daily identity-days")


if __name__ == "__main__":
    main()
