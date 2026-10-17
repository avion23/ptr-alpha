# Decay-variant panel: does changing the position-evidence branch help?

Written 2026-10-17. Applies the pre-registered promotion rule
(`docs/strategy-charter.md`) to the question left open by
`first-principles-taylor-moskowitz-20261017.md`: keep the zero-decay
position branch (V0), decay positions like events (V1), or cap them at
180d (V2). Panel: 8 decision dates × 3 variants, production kernel, only
`setups.py:217-222` differs. Artifacts: `/tmp/opencode/bt/` (cohort CSVs,
`RESULTS.md`, `progress.log`); estimator pinned by exact 8/8 replication of
the published `rank-order-20261016.md` table on the Oct-10 CSVs before any
variant ran.

## Result: FAIL for all three variants — the semantics question cannot be
## settled by returns at current power.

| variant | eligible n | ranked n | top−bot pp | clustered t | 95% CI | top vs SPY | eligible vs SPY |
|---|---:|---:|---:|---:|---|---:|---:|
| V0 zero-decay (incumbent) | 2,864 | 1,391 | −0.45 | −0.52 | [−2.21, +1.30] | −3.14 pp | −2.57 pp |
| V1 event decay | 2,864 | 414 | UNMEASURABLE¹ | — | — | −1.09 pp | −2.57 pp |
| V2 180d cap | 2,864 | 419 | UNMEASURABLE¹ | — | — | −1.20 pp | −2.57 pp |
| V1−V0 paired | — | — | UNMEASURABLE¹ | — | — | — | — |
| V2−V0 paired | — | — | UNMEASURABLE¹ | — | — | — | — |

¹ Published estimator degenerates: with decay/cap, 85% of eligible rows tie
at score 0.0 (below the 1.00 ranking floor), the bottom tercile is empty in
**all 8 dates**, and ranked coverage collapses 1,391 → ~415 (−70%). The
published rule cannot produce a contrast. Secondary sensitivity (labeled
non-primary, deterministic tie-break, equal-count terciles): V0 +1.03
(t=0.72), V1 +1.29 (t=1.13), V2 +1.25 (t=1.09); paired V1−V0 **+0.27pp
(t=0.18)**, V2−V0 **+0.22pp (t=0.15)** — nothing approaches the |t|≥2 bar.
Returns are gross of spread/fees (close-to-close, published basis; cost hits
both tercile legs identically, so the contrast is cost-invariant).

Promotion rule (charter): item 1 FAIL (V0 −0.45pp not >0; V1/V2
unmeasurable), item 2 FAIL (|t|=0.52 / 1.13 / 1.09), item 3 fill-existence
attrition only (1,099 tickers dropped; true limit-fill rate UNKNOWN, R8
PARTIAL), item 4 not reached — and no frozen walk-forward block exists in
the repo (gap, see below). **No variant clears; per the charter, sizing
stays flat and the branch is not promoted on returns evidence.**

## Adjudication

- Killing or capping the position branch buys **no measurable signal** and
  costs ~70% of ranked coverage (1,391 → ~415) — including RPM, which ranks
  only under V0. The entry-filter claim (the only supported claim) is a
  level property shared by all variants (every top bucket still trails SPY).
- **Decision (evidence-backed): keep V0** as product semantics — the broad
  insider-holdings screen — and document it in the charter as the
  adjudicated resolution of the "economics-free" flag
  (`docs/reviews/first-principles-20261016.md:45-54`), with this panel as
  the recorded reason. Re-label the holdings reason string from
  `buy @<basis>` to `basis @<basis>` (display correctness; the quantity is
  a blended FIFO basis, `positions.py:542, 844-848`, not a trade price).
- RPM recommendation remains **NO** (unchanged): rank 128 only via V0's
  unadjudicated-but-now-documented branch, +2.4% over Taylor's real 10/9
  entry, ex-div 2026-10-20.

## New finding: the DB cannot replay history point-in-time

Under the current tree's `canonical_as_of` scope, 7 of 8 historical decision
dates see **zero** events: every ingestion watermark in `data/congress.duckdb`
is Jun–Oct 2026 (the rescrapes re-stamped promotions;
`replace_archive` re-promotes whole archives), so promotion-time filtering
empties historical frames. The panel above — like the published
`rank-order-20261016.md` panel — is therefore a **disclosure-time** replay
(rows selected by disclosure ≤ decision date, today's complete DB), not an
ingestion-time replay. That is hindsight completeness, not price look-ahead
(filings are public at disclosure), but the charter's protocol should say
which frame is canonical, and the backtest harness should state it in its
outputs. Also a gap: no frozen walk-forward block exists for this harness
(promotion-rule item 4 is currently unexecutable). Both are recorded as
UNKNOWN-to-protocol, not silently patched.
