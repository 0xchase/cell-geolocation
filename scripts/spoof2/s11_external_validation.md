# S11: held-out external validation

## The rule

External evidence is consulted **only after** detection and classification are
frozen. Nothing in S0–S10 used any outside source: no news reports, no NOTAMs,
no ADS-B interference maps, no prior findings from the earlier pipeline. The
detector was built from the observation model and the data alone.

This matters because the previous draft's Queen Alia section reads as
externally validated, but the detector that found it had already been tuned
with that case in view. Validation after tuning measures agreement, not
prediction.

## Status of each candidate

**This stage is NOT complete.** It requires checking sources this session did not
access. What follows is the frozen prediction set — the thing external evidence
should be checked against — plus the one comparison that can be made from
material already in the repository.

### Frozen predictions (T1, mixture-confirmed)

| source | destination | dates | cells | mid-mass |
|---|---|---|---|---|
| Moscow 55.6–55.7 / 37.4 | 55.92–55.97 / 37.42 (Sheremetyevo) | 2024-11-20, 11-21, 11-23; 2025-05-08, 05-09, 05-10, 05-13, 05-16 | 73, 27, 23, 18, 13, 12, 11, 10, 9 | 0.40–0.47 |
| Levant 31.7–32.0 / 35.5–35.7 | 31.72–31.76 / 35.94–35.97 (Queen Alia) | 2024-09-05 | 73, 55, 13 | 0.27–0.46 |
| Cherepovets 59.1 / 38.0 | 59.36 / 38.21 | 2024-09-05 | 11 | 0.709 |
| Samara 53.3 / 50.2 | 53.15 / 50.01 | 2025-02-14 | 10 | 0.420 |
| Shenzhen/HK 22.3 / 113.8 | 22.34 / 113.59 | 2024-12-21 | 9 | 0.323 |

The Sheremetyevo recurrence is the strongest prediction to check: eight separate
days in two clusters seven months apart, same decoy each time. If external
reporting shows GNSS disruption around Moscow on those dates and not on
comparable control dates, that is genuine corroboration. If it shows nothing, the
platform-artifact reading (C10) gains weight.

### The one comparison available in-repo

The earlier pipeline (`paper-spoofing/DRAFT.md` §5.1) independently reports a
Queen Alia campaign beginning **2024-09-05** at **31.717 N, 35.999 E**. This
pipeline reaches 2024-09-05 at 31.72–31.76 / 35.94–35.97 from the opposite
direction — anchored on the source rather than the destination, with a different
reference estimator, a different detection unit, and a different statistic.
Agreement on the date to the day and the coordinate to ~0.04 degrees is
meaningful precisely because the two paths share no machinery.

That is method-vs-method agreement, not external validation. It says the
detection is reproducible; it does not say the phenomenon is GNSS spoofing.

### What must be checked, and what would falsify

| claim | corroborates | falsifies |
|---|---|---|
| Sheremetyevo campaign | reported GNSS interference near Moscow on those eight dates | no reports; or reports on control dates equally |
| Queen Alia 2024-09-05 | contemporaneous Levant accounts of devices placed in Jordan | accounts confined to a different window |
| Iran → Tehran (T3) | documented platform behaviour assigning cells to capitals | independent evidence of real long-range spoofing there |
| Western Europe 27 bins | nothing — this is the control | any of them corroborated as spoofing would invalidate S9 |

The last row is the important one. S9 found the detector fires as often in
Western Europe as in Russia. If external checking corroborated a substantial
share of those European bins as genuine spoofing, the negative control fails and
so does the interpretation built on it. If it corroborates none, the T3/rejected
tiers are doing their job.

## Why this stage cannot be skipped

S8 measures whether the detector recovers events it was given. It cannot measure
whether the events it finds unaided are the phenomenon of interest. Only external
evidence closes that gap, and only if it is consulted after the fact.
