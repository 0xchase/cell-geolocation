#!/usr/bin/env python3
"""Match private/testing MCC observations to an Overpass prison extract.

Proximity is a screening signal only.  Managed-access networks can be lawful,
and an urban cell within two kilometres of a prison is not evidence that the
network serves the facility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "criminal-activity"
CACHE = ROOT / ".cache" / "criminal-activity" / "special_mcc_nearby_prisons.json"
EARTH_KM = 6371.0088


def main() -> None:
    frames = []
    for family in ("private", "testing"):
        frame = pd.read_csv(ROOT / "data" / "test-mccs" / f"{family}.csv")
        frame["source_family"] = family
        frames.append(frame)
    cells = pd.concat(frames, ignore_index=True)

    prisons = []
    for element in json.loads(CACHE.read_text())["elements"]:
        center = element.get("center", {})
        prisons.append(
            {
                "osm_id": element["id"],
                "prison_name": element.get("tags", {}).get("name", ""),
                "prison_lat": element.get("lat", center.get("lat")),
                "prison_lon": element.get("lon", center.get("lon")),
                "osm_url": f"https://www.openstreetmap.org/{element['type']}/{element['id']}",
            }
        )
    prisons = pd.DataFrame(prisons)

    lat = np.radians(cells.glat.to_numpy())[:, None]
    lon = np.radians(cells.glon.to_numpy())[:, None]
    p_lat = np.radians(prisons.prison_lat.to_numpy())[None, :]
    p_lon = np.radians(prisons.prison_lon.to_numpy())[None, :]
    q = np.sin((p_lat - lat) / 2) ** 2
    q += np.cos(lat) * np.cos(p_lat) * np.sin((p_lon - lon) / 2) ** 2
    distance = EARTH_KM * 2 * np.arcsin(np.sqrt(q))

    rows = []
    for cell_index, prison_index in zip(*np.where(distance <= 2.0)):
        rows.append(
            {
                **cells.iloc[cell_index].to_dict(),
                **prisons.iloc[prison_index].to_dict(),
                "distance_km": distance[cell_index, prison_index],
                "interpretation": "proximity lead; lawful managed access or chance proximity unresolved",
            }
        )
    pd.DataFrame(rows).sort_values(["distance_km", "prison_name"]).to_csv(
        DATA / "special_mcc_prison_proximity_candidates.csv", index=False
    )


if __name__ == "__main__":
    main()
