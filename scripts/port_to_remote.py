#!/usr/bin/env python3
"""Repoint the figure scripts from the local ClickHouse export to the server.

Every script defines its own `ch_df` (or `ch_csv`) wrapper around
`clickhouse local --path db-export/chdata`. That export is the deduplicated
282M-row snapshot; the corrected 63.3B-row table only exists on the server. This
rewrites each wrapper to delegate to ch_remote, leaving the SQL untouched.

The per-script `--max_bytes_before_external_group_by 0` tuning is dropped on
purpose: it existed to stop a laptop-side spill to disk, and the server has its
own memory settings.

Idempotent — a script already delegating to ch_remote is skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# Matches the whole wrapper body: `def ch_df(...)` through its `return pd.read_csv(...)`
# line (including the optional `if res.stdout.strip() else pd.DataFrame()` tail).
WRAPPER = re.compile(
    r"^def (ch_df|ch_csv)\(query: str\) -> pd\.DataFrame:\n"
    r"(?:[ \t]+.*\n|\n)*?"
    r"[ \t]+return pd\.read_csv\(StringIO\(res\.stdout\)\).*\n",
    re.MULTILINE,
)

IMPORT_LINE = "from ch_remote import ch_df as _remote_ch_df\n"


def patch(path: Path) -> str:
    src = path.read_text()
    if "_remote_ch_df" in src:
        return "already ported"

    match = WRAPPER.search(src)
    if not match:
        return "NO MATCH - needs manual port"

    name = match.group(1)
    replacement = (
        f"def {name}(query: str) -> pd.DataFrame:\n"
        f"    # Ported: queries now run against the corrected 63.3B-row table on the\n"
        f"    # server. See ch_remote.py.\n"
        f"    return _remote_ch_df(query)\n"
    )
    src = src[: match.start()] + replacement + src[match.end() :]

    # Insert the import after the last existing `from`/`import` line.
    lines = src.split("\n")
    last_import = max(
        i for i, l in enumerate(lines)
        if l.startswith("import ") or l.startswith("from ")
    )
    lines.insert(last_import + 1, IMPORT_LINE.rstrip("\n"))
    path.write_text("\n".join(lines))
    return f"ported ({name})"


def main() -> int:
    targets = sorted(
        p for p in SCRIPTS.glob("*.py")
        if p.name not in {"plot_helpers.py", "ch_remote.py", "port_to_remote.py"}
    )
    failures = 0
    for path in targets:
        result = patch(path)
        if "NO MATCH" in result:
            failures += 1
        print(f"{result:<32} {path.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
