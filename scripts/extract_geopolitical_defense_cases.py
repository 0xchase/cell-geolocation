#!/usr/bin/env python3
"""Export the audited geopolitical and institutional defense-site cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from ch_remote import ch_df


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "out-of-country" / "geopolitical_defense_cases.csv"


QUERY = r"""
SELECT
    'Cross-country' AS category,
    mcc,
    mnc,
    lac,
    cid,
    cell_type,
    obs,
    glat,
    glon
FROM cell.mil_cells
WHERE cid > 0 AND (
       (country_iso = 'AZ' AND base = 'Azərbaycan Silahlı Qüvvələri' AND mcc = 283)
    OR (country_iso = 'IQ' AND base = 'آمرية افواج محافظة المثنى' AND mcc = 460)
    OR (country_iso = 'IR' AND base = 'فرماندهی کل سپاه' AND mcc = 400)
    OR (country_iso = 'MM' AND base = '418旅' AND mcc = 460)
    OR (country_iso = 'PS' AND base = 'Yellow Line' AND mcc IN (280, 284, 286, 416, 420, 424, 426, 428, 602, 606))
    OR (country_iso = 'RU' AND base = 'Аэродром Украинка' AND mcc = 460)
    OR (country_iso = 'RU' AND base = 'в/ч 51424' AND mcc = 255)
    OR (country_iso = 'CH' AND base = 'Militärflugplatz Dübendorf' AND mcc = 294)
    OR (country_iso = 'UA' AND base = 'Аеропорт "Бердянськ"' AND mcc = 250)
    OR (country_iso = 'US' AND base = 'Muscatatuck Urban Training Center' AND mcc IN (230, 283, 426, 525))
    OR (country_iso = 'US' AND base = 'Fort Bragg' AND mcc = 553)
    OR (country_iso = 'VE' AND base = 'Fuerte Tiuna' AND mcc = 363)
)
ORDER BY country_iso, mcc, mnc, lac, cid, cell_type
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = ch_df(QUERY)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"{args.output}: {len(data):,} identities, {int(data['obs'].sum()):,} observations")


if __name__ == "__main__":
    main()
