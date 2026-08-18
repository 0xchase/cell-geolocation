#!/usr/bin/env python3
"""Conservatively recheck candidate cell identities in Apple ALS.

The wire format follows the open-source CellGuard implementation.  Apple's
endpoint is undocumented, so this utility is intentionally bounded, serial,
delayed, and cache-oriented.  It supports GSM/UMTS, LTE, and NR input rows and
never sends more than ``--max-queries`` requests in one run.

Input CSV columns: mcc,mnc,lac,cid,cell_type
Output adds query time, status, ALS coordinates/metadata, and any error.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ENDPOINT = "https://gs-loc.apple.com/clls/wloc"
HEADERS = {
    "User-Agent": "locationd/2964.0.8 CFNetwork/3826.600.41 Darwin/24.6.0",
    "Accept": "*/*",
    "Accept-Language": "en-us",
}
REQUEST_FIELD = {"gsm": 1, "umts": 1, "lte": 25, "nr": 29}
RESPONSE_FIELD = {"gsm": 1, "umts": 1, "lte": 22, "nr": 24}


def varint(value: int) -> bytes:
    value %= 1 << 64
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def key(field: int, wire: int) -> bytes:
    return varint((field << 3) | wire)


def integer(field: int, value: int) -> bytes:
    return key(field, 0) + varint(value)


def blob(field: int, value: bytes) -> bytes:
    return key(field, 2) + varint(len(value)) + value


def string(field: int, value: str) -> bytes:
    return blob(field, value.encode())


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7


def fields(data: bytes):
    offset = 0
    while offset < len(data):
        field_key, offset = read_varint(data, offset)
        number, wire = field_key >> 3, field_key & 7
        if wire == 0:
            value, offset = read_varint(data, offset)
        elif wire == 1:
            value, offset = data[offset:offset + 8], offset + 8
        elif wire == 2:
            size, offset = read_varint(data, offset)
            value, offset = data[offset:offset + size], offset + size
        elif wire == 5:
            value, offset = data[offset:offset + 4], offset + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def signed(value: int, bits: int = 64) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def decode_cell(data: bytes, technology: str) -> dict:
    result: dict = {"als_cell_type": technology}
    names = {1: "als_mcc", 2: "als_mnc", 3: "als_cid", 4: "als_lac"}
    radio = ({6: "als_arfcn", 7: "als_physical_cell"}
             if technology in {"lte", "nr"}
             else {11: "als_arfcn", 12: "als_physical_cell"})
    for number, wire, value in fields(data):
        if wire == 0 and number in names:
            result[names[number]] = signed(value)
        elif wire == 0 and number in radio:
            result[radio[number]] = signed(value)
        elif number == 5 and wire == 2:
            location_names = {
                1: "als_latitude_e8", 2: "als_longitude_e8", 3: "als_accuracy",
                7: "als_confidence", 11: "als_score", 12: "als_reach",
            }
            for loc_number, loc_wire, loc_value in fields(value):
                if loc_wire == 0 and loc_number in location_names:
                    result[location_names[loc_number]] = signed(loc_value)
    if "als_latitude_e8" in result:
        result["als_latitude"] = result.pop("als_latitude_e8") / 1e8
        result["als_longitude"] = result.pop("als_longitude_e8") / 1e8
    result["als_found"] = bool(
        result.get("als_accuracy", -1) > 0
        and result.get("als_latitude", -180) != -180
        and result.get("als_longitude", -180) != -180
    )
    return result


def request_body(row: dict) -> bytes:
    technology = row["cell_type"].lower()
    if technology not in REQUEST_FIELD:
        raise ValueError(f"unsupported technology {technology!r}")
    cell = b"".join([
        integer(1, int(row["mcc"])), integer(2, int(row["mnc"])),
        integer(3, int(row["cid"])), integer(4, int(row["lac"])),
    ])
    meta = string(1, "iPhone OS18.6.2/22G100") + string(2, "iPhone17,3")
    proto = b"".join([
        blob(REQUEST_FIELD[technology], cell),
        integer(3, 0), integer(4, 1), blob(31, varint(1)), blob(33, meta),
    ])

    def packed_string(value: str) -> bytes:
        encoded = value.encode()
        return struct.pack(">h", len(encoded)) + encoded

    header = b"\x00\x01"
    header += packed_string("en_US")
    header += packed_string("com.apple.locationd")
    header += packed_string("18.6.2.22G100")
    header += b"\x00\x00\x00\x01\x00\x00"
    return header + struct.pack(">h", len(proto)) + proto


def query(row: dict, timeout: float) -> dict:
    technology = row["cell_type"].lower()
    request = urllib.request.Request(
        ENDPOINT, data=request_body(row), method="POST", headers=HEADERS
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        status = response.status
    decoded = []
    for number, wire, value in fields(raw[10:]):
        if number == RESPONSE_FIELD[technology] and wire == 2:
            decoded.append(decode_cell(value, technology))
    requested = (int(row["mcc"]), int(row["mnc"]), int(row["lac"]), int(row["cid"]))
    exact = [
        cell for cell in decoded
        if (cell.get("als_mcc"), cell.get("als_mnc"), cell.get("als_lac"), cell.get("als_cid"))
        == requested
    ]
    result = exact[0] if exact else (decoded[0] if decoded else {"als_found": False})
    result["http_status"] = status
    result["response_cells"] = len(decoded)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_queries < 1 or args.max_queries > 500:
        raise ValueError("--max-queries must be between 1 and 500")
    rows = list(csv.DictReader(args.input.open()))
    if len(rows) > args.max_queries:
        raise ValueError(f"input has {len(rows)} rows; limit is {args.max_queries}")
    output = []
    for index, row in enumerate(rows):
        result = dict(row)
        result["queried_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            result.update(query(row, args.timeout))
            result["error"] = ""
        except Exception as error:  # Preserve a complete, auditable batch.
            result["als_found"] = False
            result["error"] = repr(error)
        output.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if index + 1 < len(rows):
            time.sleep(args.delay)
    columns = list(dict.fromkeys(key for row in output for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
