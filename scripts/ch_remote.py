"""Remote ClickHouse access for the regenerated (post-dedup-fix) dataset.

The corrected `cell.geos` on nominatim.cybre.io is 63.3B rows / 463 GiB, so it
cannot be exported to a laptop the way the old 282M-row snapshot was. All
queries therefore run server-side over SSH.

SQL is passed on stdin rather than via --query, which sidesteps shell quoting
entirely (the analysis queries are full of nested single quotes).
"""

from __future__ import annotations

import os
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

HOST = os.environ.get("CELL_DB_HOST", "ckanipe@nominatim.cybre.io")
CH_PASSWORD = os.environ.get("CELL_DB_PASSWORD", "password")

# Reuse one SSH connection across the many queries a figure needs; without this
# each query pays a fresh TCP + auth handshake.
# The socket path must stay under the 104-byte sun_path limit, so it lives in
# ~/.ssh rather than a deep scratch directory.
_CTL_PATH = Path(os.environ.get("CELL_DB_CTL_PATH", str(Path.home() / ".ssh" / "cm-cellgeo.sock")))
_CTL_DIR = _CTL_PATH.parent

_SSH_OPTS = [
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={_CTL_PATH}",
    "-o", "ControlPersist=10m",
    "-o", "ServerAliveInterval=30",
]

# Server has 16 cores and also serves Nominatim; leave headroom.
DEFAULT_SETTINGS = {
    # Analysis access is intentionally read-only; this also protects future
    # scripts that reuse this helper from accidentally mutating the server.
    "readonly": 2,
    "max_threads": 12,
    "max_execution_time": 7200,
    # The heavy per-cell aggregations group by the table's exact ORDER BY key,
    # so ClickHouse can stream them instead of building a giant hash table.
    "optimize_aggregation_in_order": 1,
}


def ch_query(sql: str, settings: dict | None = None, fmt: str = "CSVWithNames") -> str:
    """Run SQL on the remote server and return raw output."""
    _CTL_DIR.mkdir(parents=True, exist_ok=True)
    merged = {**DEFAULT_SETTINGS, **(settings or {})}
    remote_cmd = ["clickhouse-client", "--password", CH_PASSWORD]
    for key, value in merged.items():
        remote_cmd += [f"--{key}", str(value)]

    sql = sql.strip().rstrip(";")
    if fmt and "FORMAT" not in sql.upper().split("\n")[-1]:
        sql = f"{sql}\nFORMAT {fmt}"

    proc = subprocess.run(
        ["ssh", *_SSH_OPTS, HOST, " ".join(remote_cmd)],
        input=sql,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ClickHouse query failed:\n{proc.stderr.strip()}\n\nSQL:\n{sql}")
    return proc.stdout


def ch_df(query: str, settings: dict | None = None) -> pd.DataFrame:
    """Run SQL on the remote server and return a DataFrame."""
    out = ch_query(query, settings=settings, fmt="CSVWithNames")
    if not out.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(out))


def ch_scalar(query: str, settings: dict | None = None):
    """Run SQL expected to return exactly one value."""
    df = ch_df(query, settings=settings)
    return df.iloc[0, 0]
