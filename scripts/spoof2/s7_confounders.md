# S7: Confounder battery

Every alternative explanation for "many cells in one region were first seen
displaced at the same time", and the **measured quantity** that retires it. A
confounder with no measurement against it is not controlled, and is listed here
as uncontrolled rather than omitted.

Two of these were not hypothetical. They were found by looking at S3's first
output and asking whether its top-ranked results were physically possible.

---

## C1. Global crawl volume changes — **CONFIRMED PRESENT, controlled**

**The confound.** Crawl volume steps by large factors between months (796M in
2024-03 to 2.43B in 2024-04). When the crawler suddenly samples more, more cells
are *first observed* displaced, everywhere at once. This is not a property of
the world.

**Evidence it is real here.** S3's initial ranking, normalised only against each
region's own mean rate, was topped by:

| day | onsets | regions involved |
|---|---|---|
| 2024-04-03 | 207,522 | 68,086 |
| 2023-11-04 | 166,080 | 78,147 |
| 2026-05-01 | 67,163 | 45,525 |

No terrestrial transmitter reaches 68,086 regions across five continents in one
day. 2024-04-03 sits exactly on the March→April 2024 crawl step.

**Control.** `spoof.exposure_region_day` — observations of candidate cells per
(source region, day). Onset counts are rates against it, never raw counts.
Monthly exposure (`exposure_sq`) is insufficient because the confound operates
within a month.

**Residual risk.** Exposure measures how hard we looked, not how many *distinct*
cells we looked at (the cardinality column was dropped for memory reasons, see
`s0b_exposure_daily.sql`). A day that re-samples few cells many times looks like
high exposure. This inflates the denominator and is therefore *conservative* —
it suppresses events, it cannot manufacture them.

---

## C2. First contact with a region — **CONFIRMED PRESENT, controlled**

**The confound.** A cell cannot be seen displaced before it is seen at all. When
the crawler reaches a region for the first time, its cells acquire a reference
and their first away-observation follows shortly after — an apparent synchronised
onset that is purely an artifact of first contact.

**Evidence it is real here.** 2023-11-04, the first day of the dataset, is the
second-largest onset day globally, spanning 78,147 regions.

**Control.** Lead time: a cell contributes an onset only if it was already known
for ≥30 days before that onset (`cellref.m_first` vs `cell_away.t_first_away`).
Regions whose events survive only without this filter are reported as
crawl-onset indeterminate, not as events.

---

## C3. Reference capture by a long campaign — **controlled by construction**

**The confound.** If displacement persists long enough, it becomes the cell's
apparent normal position, the estimator adopts the decoy as the reference, and
the sign inverts: the cell's true location is then reported as the displacement.

**Control.** S1's two-level estimator weights each *month* equally rather than
each observation, giving a breakdown point of 50% of observed months per
coordinate. `cellref_stability.stab_frac` measures per cell how much of its
history actually sits at the reference.

**Measured impact.** Against the old plurality estimator this relocates 117,255
cells by >10 km and 8,104 by >100 km. 1,558,362 candidate cells (16.5%) have
`stab_frac < 0.5` and are flagged as having an untrustworthy reference rather
than silently used.

---

## C4. Genuine cell relocation

**The confound.** Operators move base stations. A relocated cell steps to a new
position and stays there — superficially a displacement episode.

**Control.** A relocation has no return: the cell's monthly modes are at the old
position before the step and the new one after, with `stab_frac` near 0.5 and a
single change point. A spoofing episode returns, so the reference is recovered.
Relocations are also unsynchronised — one operator's maintenance schedule does
not move five operators' cells on one day, which C10 tests directly.

---

## C5. Dense-area positional scatter

**The confound.** In dense urban deployments the platform's estimate for a cell
wanders by 10–20 km in **every** direction. Such a cell generates away
observations continuously and can coincide with others by chance.

**Control.** Source anisotropy — do the affected cells' displacements share a
direction, or do they surround the site — plus S5's cross-track statistic, which
is near zero for a genuine mixture and comparable to along-track for isotropic
scatter.

---

## C6. Border spillover and genuine roaming coverage

**The confound.** A foreign operator's cells legitimately appear near a frontier;
coverage genuinely crosses borders.

**Control.** A displacement floor well beyond RF reach, and the observation that
spillover is *continuous* rather than having a datable onset. This is why onset
(C10) does more work here than the foreign-MCC test, which the old pipeline
leaned on and which fails in occupied territory (C8).

---

## C7. Equipment-test leakage

**The confound.** Test equipment radiates real cell identities from a lab. The
identity is false for that location; the position is correct. This is the
*inverse* failure to spoofing and must not be counted as it.

**Control.** Source spread: contributing cells' references are scattered
worldwide rather than forming one coherent region. Also C11's technology and
operator mix, which for test gear departs sharply from any local baseline.

---

## C8. Operators legitimately present in contested territory — **flagged, not dropped**

**The confound.** Russian-operator cells inside occupied Ukrainian territory
satisfy any cross-border test without being spoofing: those networks do operate
there.

**Control.** None available from the data. These are flagged explicitly and
excluded from any cross-border headline count, because the honest statement is
that the test cannot distinguish them, not that they were removed.

---

## C9. Traveller upload batching

**The confound.** Handsets that measured cells abroad may upload on landing,
associating foreign cells with an airport.

**Control.** Source geography — travellers depart from airports and cities, not
from a rural border strip — plus C10 synchrony across multiple operators and
countries within hours, which an uncoordinated travel pattern cannot produce.

---

## C10. The platform's own estimator changing behaviour — **partially controlled**

**The confound.** We observe the platform's *belief*, not handset reports. A
change in its aggregation, a re-index, or a fallback rule could move many cells
at once and would look exactly like an area effect.

**Control.** Geographic bounding: a platform-side change is global or
account-wide, not confined to one 11 km region. Events whose onset day is also a
global spike (C1) are rejected on that basis.

**Residual risk — this is the weakest point in the battery.** A platform change
applied to one *region* would be indistinguishable from spoofing by any test
here. Nothing in the dataset can separate them, and no claim should be made that
it can. This is a limitation to state, not a confounder to declare controlled.

---

## C11. Operator- or technology-specific artifacts

**The confound.** A misconfigured DAS, one operator's bad location backhaul, or
a single vendor's firmware moves many cells at once, but only that operator's or
that technology's.

**Control.** S6's G-test of affected technology and MCC/MNC mix against the local
baseline. Spoofing corrupts the handset receiver and is therefore indiscriminate;
these artifacts are not. Note the inverted reading: a **high** p-value supports
the spoofing interpretation.

---

## C12. Identity replay

**The confound.** A transmitter re-broadcasting another region's cell identities
produces "region A's cells at point B" — the same coarse signature as spoofing,
with the opposite mechanism (position true, identity false).

**Control.** S5 mixture geometry. Spoofing is a weighted mean and *must* produce
graded intermediate positions along the home→decoy axis; replay yields two
populations of genuine fixes and is bimodal. This replaces the old pipeline's
1,000 km distance cut, which encoded an assumption about attacker intent rather
than a measurement.

---

## Summary

| # | Confounder | Status |
|---|---|---|
| C1 | Global crawl volume | confirmed present, controlled by daily exposure |
| C2 | First contact | confirmed present, controlled by lead time |
| C3 | Reference capture | controlled by construction, impact measured |
| C4 | Genuine relocation | controlled (no return + unsynchronised) |
| C5 | Dense-area scatter | controlled (anisotropy, cross-track) |
| C6 | Border spillover | controlled (displacement floor + onset) |
| C7 | Equipment-test leakage | controlled (source spread, mix) |
| C8 | Contested-territory operators | **uncontrollable — flagged explicitly** |
| C9 | Traveller batching | controlled (source geography + synchrony) |
| C10 | Platform-side change | partially — **regional change is indistinguishable** |
| C11 | Operator/technology artifact | controlled (S6 G-test) |
| C12 | Identity replay | controlled (S5 mixture geometry) |

Two are not fully controlled (C8, C10). Both are stated as limitations. The
sensitivity and false-positive rates that bound everything else come from S8,
which is the only stage that measures the pipeline end to end.
