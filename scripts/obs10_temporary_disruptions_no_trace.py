#!/usr/bin/env python3
"""Generate a four-panel figure for temporary-disruption null results."""

from __future__ import annotations

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ch_remote import ch_df as _remote_ch_df


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
CLICKHOUSE = DATA_ROOT / "clickhouse"
CH_PATH = DATA_ROOT / "db-export" / "chdata"
PLOTS = ROOT / "plots"
STALE_CUTOFF = "2025-06-01"

COLORS = {
    "Null": "#2f6f9f",
    "Baseline": "#8a8f98",
    "Ambiguous": "#c9743a",
    "Positive": "#b23a48",
}


def ch_df(query: str) -> pd.DataFrame:
    # Ported: queries now run against the corrected 63.3B-row table on the
    # server. See ch_remote.py.
    return _remote_ch_df(query)


def load_data() -> dict[str, pd.DataFrame | float]:
    cases = ch_df(
        f"""
        SELECT 'Global baseline' AS region, count() AS cells,
               countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count() AS stale_frac
        FROM cell.summary WHERE cid > 0
        UNION ALL
        SELECT 'Cuba blackouts', count(),
               countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count()
        FROM cell.summary WHERE cid > 0 AND mcc = 368
        UNION ALL
        SELECT 'Syria regime change', count(),
               countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count()
        FROM cell.summary WHERE cid > 0 AND mcc = 417
        UNION ALL
        SELECT 'Taiwan Hualien EQ bbox', count(),
               countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count()
        FROM cell.summary
        WHERE cid > 0 AND mcc = 466 AND glat BETWEEN 23.7 AND 24.4 AND glon BETWEEN 121.3 AND 122.0
        UNION ALL
        SELECT 'Hajj Mina/Arafat bbox', count(),
               countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count()
        FROM cell.summary
        WHERE cid > 0 AND mcc = 420 AND glat BETWEEN 21.25 AND 21.45 AND glon BETWEEN 39.85 AND 40.05
        """
    )
    cases["stale_pct"] = cases["stale_frac"] * 100
    cases["verdict"] = cases["region"].map(
        {
            "Global baseline": "Baseline",
            "Cuba blackouts": "Null",
            "Syria regime change": "Baseline",
            "Taiwan Hualien EQ bbox": "Ambiguous",
            "Hajj Mina/Arafat bbox": "Ambiguous",
        }
    )
    baseline = float(cases.loc[cases["region"] == "Global baseline", "stale_frac"].iloc[0])

    porto = ch_df(
        """
        SELECT toDate(toStartOfQuarter(last_seen)) AS quarter, count() AS cells
        FROM cell.summary
        WHERE cid > 0 AND mcc = 724
              AND glat BETWEEN -30.2 AND -29.8 AND glon BETWEEN -51.4 AND -50.9
        GROUP BY quarter
        ORDER BY quarter
        """
    )
    porto["quarter"] = pd.to_datetime(porto["quarter"])
    porto["quarter_label"] = porto["quarter"].map(lambda d: f"{d.year} Q{((d.month - 1) // 3) + 1}")

    timeline = ch_df(
        """
        SELECT toDate(toStartOfQuarter(last_seen)) AS quarter,
               multiIf(mcc = 368, 'Cuba', 'Syria') AS country,
               count() AS cells
        FROM cell.summary
        WHERE cid > 0 AND mcc IN (368, 417)
        GROUP BY quarter, country
        ORDER BY quarter, country
        """
    )
    timeline["quarter"] = pd.to_datetime(timeline["quarter"])
    timeline["quarter_label"] = timeline["quarter"].map(lambda d: f"{d.year} Q{((d.month - 1) // 3) + 1}")

    positive = ch_df(
        f"""
        SELECT 'Gaza Palestinian operators' AS case_name, count() AS cells,
               countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count() AS stale_frac
        FROM cell.summary
        WHERE cid > 0 AND mcc = 425 AND mnc IN (5, 6)
              AND glat BETWEEN 31.2 AND 31.6 AND glon BETWEEN 34.2 AND 34.6
        UNION ALL
        SELECT 'Australia GSM shutdown', count(),
               countIf(last_seen < toDateTime('{STALE_CUTOFF}')) / count()
        FROM cell.summary
        WHERE cid > 0 AND mcc = 505 AND cell_type = 'gsm'
        """
    )
    matrix = pd.concat(
        [
            cases[cases["region"].isin(["Cuba blackouts", "Syria regime change", "Taiwan Hualien EQ bbox", "Hajj Mina/Arafat bbox"])][["region", "cells", "stale_frac"]].rename(columns={"region": "case_name"}),
            positive,
        ],
        ignore_index=True,
    )
    matrix["stale_pct"] = matrix["stale_frac"] * 100
    matrix["event_class"] = matrix["case_name"].map(
        {
            "Cuba blackouts": "temporary",
            "Syria regime change": "non-radio governance change",
            "Taiwan Hualien EQ bbox": "ambiguous geography",
            "Hajj Mina/Arafat bbox": "seasonal/ambiguous",
            "Gaza Palestinian operators": "persistent infrastructure loss",
            "Australia GSM shutdown": "permanent tech sunset",
        }
    )

    return {"cases": cases, "baseline": baseline, "porto": porto, "timeline": timeline, "matrix": matrix}


def make_figure(data: dict[str, pd.DataFrame | float], output: Path, preview: Path | None) -> None:
    cases = data["cases"].copy()
    baseline = float(data["baseline"])
    porto = data["porto"].copy()
    timeline = data["timeline"].copy()
    matrix = data["matrix"].copy()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.7), constrained_layout=True)
    fig.suptitle(
        "Temporary disruptions leave little durable trace in last-seen cell data",
        fontsize=14,
        fontweight="bold",
    )

    # A. Event sweep.
    ax = axes[0, 0]
    order = ["Cuba blackouts", "Syria regime change", "Global baseline", "Hajj Mina/Arafat bbox", "Taiwan Hualien EQ bbox"]
    cases["region"] = pd.Categorical(cases["region"], categories=order, ordered=True)
    cases = cases.sort_values("region")
    ax.barh(cases["region"], cases["stale_pct"], color=[COLORS[v] for v in cases["verdict"]])
    ax.axvline(baseline * 100, color="#444444", linestyle="--", linewidth=1)
    for patch, stale, cells in zip(ax.patches, cases["stale_pct"], cases["cells"], strict=False):
        ax.text(stale + 1.0, patch.get_y() + patch.get_height() / 2, f"{stale:.1f}%  n={cells:,}", va="center", fontsize=8)
    ax.set_title("A. Temporary/null cases do not consistently exceed baseline")
    ax.set_xlabel(f"Cells last seen before {STALE_CUTOFF} (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 62)

    # B. Porto Alegre flood timing.
    ax = axes[0, 1]
    ax.bar(porto["quarter_label"], porto["cells"], color="#2f6f9f")
    flood_idx = list(porto["quarter_label"]).index("2024 Q2")
    ax.axvspan(flood_idx - 0.5, flood_idx + 0.5, color="#b23a48", alpha=0.14)
    ax.text(flood_idx, porto["cells"].max() * 0.72, "May 2024\nflood", ha="center", fontsize=8, color="#5a2a2f")
    ax.set_title("B. Porto Alegre: no flood-quarter disappearance")
    ax.set_xlabel("Quarter of last observation")
    ax.set_ylabel("Cells in Porto Alegre bbox")
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x):,}")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    # C. Cuba/Syria last-seen timing.
    ax = axes[1, 0]
    sns.barplot(data=timeline, x="quarter_label", y="cells", hue="country", palette={"Cuba": "#2f6f9f", "Syria": "#8a8f98"}, ax=ax)
    ax.set_title("C. Cuba/Syria towers are re-observed after events")
    ax.set_xlabel("Quarter of last observation")
    ax.set_ylabel("Cells")
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x):,}")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="", loc="upper left", frameon=True)

    # D. Mechanism check: durable removal vs transient event.
    ax = axes[1, 1]
    class_order = [
        "temporary",
        "non-radio governance change",
        "seasonal/ambiguous",
        "ambiguous geography",
        "persistent infrastructure loss",
        "permanent tech sunset",
    ]
    matrix["event_class"] = pd.Categorical(matrix["event_class"], categories=class_order, ordered=True)
    matrix = matrix.sort_values("event_class")
    colors = matrix["case_name"].map(
        lambda x: COLORS["Positive"] if x in {"Gaza Palestinian operators", "Australia GSM shutdown"} else COLORS["Ambiguous"] if "bbox" in x else COLORS["Null"]
    )
    ax.scatter(matrix["event_class"].astype(str), matrix["stale_pct"], s=(matrix["cells"] ** 0.5) * 2.5, color=colors, alpha=0.78, edgecolor="white", linewidth=0.6)
    ax.axhline(baseline * 100, color="#444444", linestyle="--", linewidth=1)
    for row in matrix.itertuples():
        ax.text(str(row.event_class), row.stale_pct + 2.0, row.case_name.split(" bbox")[0], fontsize=7, ha="center")
    ax.set_title("D. Staleness detects durable removal, not brief outage")
    ax.set_xlabel("")
    ax.set_ylabel(f"Cells last seen before {STALE_CUTOFF} (%)")
    ax.set_ylim(0, 108)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if preview is not None:
        fig.savefig(preview, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS / "obs10_temporary_disruptions_no_trace.pdf",
        help="PDF output path.",
    )
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview path.")
    args = parser.parse_args()

    data = load_data()
    make_figure(data, args.output, args.preview)
    print(args.output)


if __name__ == "__main__":
    main()
