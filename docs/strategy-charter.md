# Strategy charter: skilled-insider delayed filings, fair-price entry

Written 2026-10-16 from the first-principles review. This is the R6 the
review found missing: the single objective, what is NOT claimed, and the
pre-registered rule for real money. Parameters below are the current
operating values, not tuned optima — tuning without an objective was the
trap this charter closes.

## Objective (R6)

Maximize 90-calendar-day SPY-relative forward excess per served decision,
net of spread and fees, at a modeled limit-fill rate for the entry rule.
Unit of account: one served ranked set (per-decision receipt). Secondary:
hit rate (fraction of entries with excess > 0), worst-decision loss.

- Baselines every result must beat: SPY buy-and-hold over the same window
  (by construction, excess is SPY-relative) and the unskilled filter (all
  eligible buys, unranked).
- Non-goals: sizing (flat $3k always), timing within the window, dividends
  capture (avoided, currently operator-enforced — see R9).

## What the score is (and is not)

Score(T) = 1.5^C(T) · Σ w_a · d_a over accepted actors (see
`src/analyzer/setups.py:score`). Production weights take exactly five
values ({0.52, 0.26, 0.60, 0.30, 0}; `actors.py:compute_actor_weights`).
The score is a ranker over entry-quality-weighted recent activity — an
actor-count vote with adjustments — NOT a predicted return and NOT a
validated skill estimate. Rank differences above the floor mostly order
actor counts. Never size by score magnitude.

## Requirements table

| # | Requirement | Check / threshold | Status 2026-10-16 |
|---|-------------|-------------------|-------------------|
| R1 | Served set recorded per decision | receipt verifies `agreement: True` | PASS |
| R2 | Point-in-time (no future data) | as-of scoping + consumer-inventory test | PASS (code; ingestion history begins 2026-06-21 — earlier dates are disclosure-frame only, see Replay frames) |
| R3 | Determinism | `test_decision_determinism` | PASS |
| R4 | Entry ≤ 1.05× insider ref | gate in `score()`, unit-tested | PASS |
| R5 | Quote ≤ 5 calendar days old | stale-quote block, unit-tested | PASS |
| R6 | Objective + promotion rule stated | this charter | PASS (as of here) |
| R7 | Skill predicts forward excess | backtest clustered \|t\|≥2 (8 clusters) | FAIL (t=0.81 / 0.55) |
| R8 | Costs modeled (spread, fees, fill-rate) | harness accounts spread/fees/fills | PARTIAL — fill existence proven (eligible ⟺ day-1 fill, limit arithmetic on closes); spread as sensitivity only (0–50bps moves nothing: eligible −2.56→−3.06pp, tercile contrast unchanged at −0.35pp) |
| R9 | Dividend avoidance | ex-div check in scoring path | PARTIAL — gate live in `score()` (7d ticker block, `ex_dividends` map, missing = unknown/off) with `dividends` table + refresh population; live rows pending provider quota (Yahoo throttled 2026-10-16) |

## Promotion rule (pre-registered)

No position size above flat $3k unless ALL hold on a frozen harness:
1. Among-eligible top-vs-bottom score-tercile forward excess > 0,
   net of spread + fees at modeled fill rate — TESTED 2026-10-16: FAIL
   (top−bot −0.35pp, t=−0.44; see `docs/reviews/rank-order-20261016.md`);
2. clustered (by month) |t| ≥ 2;
3. limit-fill rate for the ≤1.05× rule reported (fills, not assumptions);
4. result replicated on the frozen walk-forward block (below) — ingestion-time frame, defined 2026-10-17, unrun.

Current evidence meets none of 1–4: sizing stays flat, paper is defensible.

## Replay frames (declared 2026-10-17)

Two frames exist. They answer different questions and are never pooled.

- **Disclosure-time frame** (research lane): today's complete database,
  rows filtered by `disclosure_date <= decision date`. Backs every
  historical signal-research claim (rank-order-20261016.md,
  decay-variants-20261017.md, and any future historical panel). Known
  bias: hindsight completeness — it sees filings the live pipeline had
  not yet ingested. This is completeness bias, not price look-ahead:
  all added filings were public at disclosure, and entry/exit prices are
  strictly decision-forward. Effect on the top−bot contrast is
  second-order (both legs share the frame); the level vs SPY is the
  complete-population quantity, not the live quantity.
- **Ingestion-time frame** (validation lane): rows promoted, accepted,
  and created by end of the decision date (`canonical_as_of`), plus the
  per-decision decision receipts, which freeze candidates, weights, and
  quotes outside the database. Backs every product-validation and
  promotion-rule claim. The DB's first ingestion watermark is
  2026-06-21, so no earlier date is replayable in this frame; that
  history is closed, not reconstructed.

Every backtest artifact must record which frame produced it. A pooled
estimate that mixes frames is invalid regardless of its t-statistic.

### Frozen walk-forward block (promotion-rule item 4)

Defined 2026-10-17, before any of its data exist. Block = decision dates
2027-01-15, 2027-07-15, 2028-01-15, 2028-07-15. Frame: ingestion-time
(canonical_as_of), the live frame — the DB will contain those ingestions
naturally. Estimator: unchanged published estimator (per-decision score
terciles via rank(pct=True), bottom ≤ 1/3, top ≥ 2/3; pooled equal-weighted
mean of per-date top−bot diffs; SE = std(ddof=1)/√4; clustered t with
cluster = decision date; bar |t| ≥ 2). Sample-size expectation from the
published panels: ~330–440 eligible rows and ~150–215 ranked rows per date
(~1,400–1,700 eligible, ~600–880 ranked over the block); a date yielding
< 50 eligible events or < 20 candidate tickers is DEGENERATE and the
harness refuses it.

Per decision date, the live unscoped rank MUST run with --receipt (the
receipt is the frame anchor and the degeneracy cross-check), and the
harness cohort run (--frame ingestion) MUST be written to
data/backtest_assumptions/walkforward/ at D+1, write-once. No interim
analysis of block cohorts is permitted; the artifacts are sealed.

Unseal: no earlier than 2028-11-01 (last entry 2028-07-15 + 90d
maturity + session adjustment). The operator (single-operator repo; integrity via
the receipts hash chain + git) runs the harness on the block, records in
one docs/reviews/walkforward-unseal-<date>.md: DB sha256, git sha,
verify_decision output (fail-closed) for each block receipt, per-date
cohort counts + guard results, and the full estimator table. At unseal,
promotion-rule items 1–4 are evaluated ON THE BLOCK cohorts. A block FAIL
is final for the tested variant; a block PASS advances the variant to a
full charter re-test. The block may be extended only by a pre-registered
charter amendment committed before the first unsealed date.

## Known non-circular result

Eligible (pass-gate) −2.56 vs ineligible −0.52 (clustered t=−4.65) validates
the ENTRY FILTER (don't chase >5% above the print on stale quotes), not
insider skill. It justifies keeping the gates; it does not justify the score.
