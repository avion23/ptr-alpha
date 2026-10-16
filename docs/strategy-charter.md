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
| R2 | Point-in-time (no future data) | as-of scoping + consumer-inventory test | PASS (code; pre-versioning era UNKNOWN) |
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
4. result replicated on a walk-forward block never used for selection.

Current evidence meets none of 1–4: sizing stays flat, paper is defensible.

## Known non-circular result

Eligible (pass-gate) −2.56 vs ineligible −0.52 (clustered t=−4.65) validates
the ENTRY FILTER (don't chase >5% above the print on stale quotes), not
insider skill. It justifies keeping the gates; it does not justify the score.
