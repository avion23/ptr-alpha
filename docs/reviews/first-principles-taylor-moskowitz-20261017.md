# First-principles review: the Taylor/Moskowitz (RPM) treatment

Written 2026-10-17. Question: *is the way the 2026-10-16 decision treated
congress:DAVID J TAYLOR and congress:JARED MOSKOWITZ on RPM correct?*
Method: requirements → first principles → logic → mathematics → CS → data
design → empirical. Evidence receipts: three parallel subagent runs
(sessions `ses_eb58c308affe…`, `ses_eb58c3087ffe…`, `ses_eb58c3083ffe…`)
plus direct reproduction in this session.

## 0. Requirements

| # | Requirement | Check | Threshold | Status |
|---|---|---|---|---|
| R0.1 | Decisions are point-in-time: "rowsets promoted AND accepted by the end of this day" (`src/analyzer/database.py:319-333`) | differential: per-member vs batch holdings for as_of=2026-10-16 | identical | **FAIL → FIXED** (was 4 differing tickers for Taylor; receipt `734927d4` carried the unscoped values) |
| R0.2 | One notion of "entry" per row, labeled truthfully | reason string vs producing path (`setups.py:225-228`, `positions.py:844-848`) | label matches quantity | **FAIL** (blended basis printed as "buy @"; no single buy at 102.48/103.72 happened) |
| R0.3 | Equal information at equal age weighs equally (implied by the decay design; the flag split is documented at `setups.py:214-216` but never economically justified) | same purchase through event vs holdings framing | same weight | **FAIL** (event: 0.5^(t/90); holdings: 1.0 ∀t — Moskowitz's same trade: 0.0094 vs 0.52) |
| R0.4 | Rank reflects evidence strength (charter R7) | backtest clustered \|t\|≥2 | t≥2 | **FAIL, pre-existing** (`docs/strategy-charter.md:42`, t=0.81/0.55) — weights are risk controls, not alpha |
| R0.5 | Replay reproduces recorded decisions | rebuild 2026-10-16 from current DB + HEAD | exact match | **PASS with caveat** — 339/339 ranked match only via the unscoped path (`holdings_candidates_batch.__wrapped__`); the as-of-scoped tree gives 338 and RPM lag 210. The recorded receipt is not point-in-time reproducible |

## 1. First principles

The score decides *which tickers to show*, never size (charter). Two distinct
quantities are conflated in one sum:

- **Event**: "an insider transacted recently; timing information decays."
- **Position**: "an insider holds shares with basis B; a *state*, current
  until a sale is disclosed."

The production branch makes the weight a function of the *pipeline that
produced the row*, not of the information's age: `position_evidence=True →
decay=1.0` at any age (`setups.py:217-222`), while `13F` corroboration of the
same kind of state is hard-capped at 180 days with an explicit economic
rationale (`setups.py:139-143`). The project's own review flagged the branch
as "economics-free" (`docs/reviews/first-principles-20261016.md:45-54`) and it
was never adjudicated. Verdict: **FAIL as a staleness model** — the flag is a
data-availability bit. Whether it helps or hurts returns is **UNKNOWN at
current power** (R0.4).

## 2. Logic

- **Contradictory notions of time (REAL, fixed this session).**
  `holdings_candidates` (`positions.py:612`) is `@canonical_decision("as_of")`;
  `holdings_candidates_batch` (`positions.py:674`) was not, and its only
  production call site (`cli.py:1768`) did not own the scope either, so the
  exemption reason at `tests/test_asof_consumer_inventory.py:84` was false.
  Measured: a doc promoted 2026-10-17 13:07 entered the 2026-10-16 decision —
  Taylor RPM entry_ref 102.4755/lag 1 (unscoped) vs 103.7155/lag 210 (scoped);
  the recorded receipt contains the unscoped values (manifest `members[0]`).
  **Fix**: decorator added, stale exemption removed, differential regression
  test with real promotion watermarks added
  (`tests/test_positions.py:1450-1472`); suite 1274 passed / 11 skipped.
- **Duplicated truth**: the same purchase exists as an event row (decaying)
  and a holdings row (persistent) with different entry anchors; the max-decay
  row selection (`setups.py:260`) silently prefers the holdings framing.
- **Leak blast radius**: RPM's rank is unchanged by the fix (score stays 1.04 —
  both members' position rows survive scoping); what changes is
  entry_ref (102.48→103.72), min_disclosure_lag_days (1→210), and 4 of
  Taylor's 22 tickers (RPM/FITB/T/PG anchors). Ranked set 339→338.

## 3. Mathematics (closed form)

With weights 0.52/0.52, floor 0.05, half-life 90d, ranked floor 1.00:

- Two fresh actors: 1.04. Fresh + decaying clears 1.00 iff
  `0.52·0.5^(t/90) ≥ 0.48 ⟺ t ≤ 90·log2(0.52/0.48) = 10.39 d`.
- Floor alone: `t ≤ 90·log2(0.52/0.05) = 304.1 d`.
- Position branch: `d(t) = 1` ∀t — the only age-blind term in the model.
  Limit behavior: event weight →0, position weight constant; measured accepted
  position lags min/median/max = **1 / 526 / 4,292 days** (≈11.75 y at full
  weight).
- Consequence for RPM: under event decay Moskowitz contributes
  `0.52·0.5^(525/90) = 0.0094 < 0.05` → floored; score 0.516 < 1.00, single
  actor → not ranked. Same under a 180d cap ("no actor weight"). **RPM ranks
  only via the zero-decay branch.** Same for the top of the book: 334/339
  ranked tickers have an accepted position row; **72.2% of total ranked score
  mass comes from position rows older than 180d (60.4% older than 365d);
  97.25% from position rows at all**. 128/339 ranked tickers have no member
  row younger than 180d.
- Alternatives (production kernel, only the decay block at `setups.py:217-222`
  changed; unmodified copy verified frame-exact vs production):
  positions decayed like events → **101 ranked** (238 lose rank, 0 gain);
  positions 180d-capped → **124 ranked** (215 lose, 0 gain). Top-20 loses
  CMCSA, DIS, BABA, ACN, BIIB under both; NFLX 13→3/4, HD 8→2, BA 4→6/31.

## 4. CS / algorithms

The batch loader was an O(1)-round-trip optimization whose scope contract was
never carried over — a correctness defect in the fast path only (the slow path
was right). Fix cost: one TEMP VIEW rebuild per call (`database.py:318-333`),
negligible vs the 3.7-min unscoped run. Negative control: stripping the
decorator makes the new test fail ("0 INSM, 1 LATE"). **PASS post-fix.**

## 5. Data design

- **Two entry definitions in one gate.** Winning rows are 93.3% holdings
  blended basis (`Σ lot_cost/Σ lot_shares`, FIFO-relieved, spanning months —
  `positions.py:542, 844-848`) vs 6.7% event trade-date close
  (`pipeline.py:383-393`). Median |blended − last event close| = 0 (540
  single-lot rows), max 958.97 (NVR). Re-anchoring holdings rows to the same
  actor's most recent event close flips the 1.05× gate for 134 rows across
  104/338 ranked tickers (47 rows pass only via blended; 87 pass only via
  event) and moves 13 tickers across the ranked boundary (8 out, 5 in).
- **Recency and anchor come from different lots**: the position row's
  `disclosure_date` is the newest lot's disclosure (Taylor: lag 1) while its
  `entry_ref` averages 11 months of lots incl. a FIFO-relieved sale — the
  reported "recency" is not the age of the price being used.
- **Reason string mislabels the quantity**: "buy @102.48 (est.)" — no single
  buy at 102.48 occurred (his scoped lots: 105.84/109.76/98.63/100.48/97.73;
  102.48 is a basis). Display-only, but this is the operator-facing surface.

## 6. Empirical

- Receipt reproduction: 339/339 ranked tickers, 0 score/actor mismatches
  (via `__wrapped__` = HEAD semantics). Scoped tree: 338 ranked, RPM ranked at
  1.04 with lag 210 — matching the older receipts `1b19c306`/`fe980e69`.
- Charter state: R7 FAIL, rank-order FAIL (top−bot −0.35pp, t=−0.44), eligible
  avg −2.56pp — **no measured alpha to protect**; the entry filter (price ≤
  1.05× basis) remains the only supported claim, and the position branch is
  precisely what maximizes that screen's coverage.

## Adjudication

1. **Taylor treatment — was WRONG, now fixed.** Leak: post-decision promotion
   entered the decision (fixed, `positions.py:674`; test
   `tests/test_positions.py:1450-1472`). The receipt `734927d4` stands as
   recorded (append-only) but is marked not-point-in-time-reproducible.
2. **Moskowitz treatment — consistent with the holdings-screen the code
   actually implements, inconsistent with the charter's "delayed filings"
   premise, and never adjudicated.** The 525-day-old position at full weight
   is the *reason RPM ranks*. Keeping it is a legitimate product choice
   ("buy what insiders hold below basis — 334 tickers of coverage"); removing
   it (decay or 180d cap) shrinks the screen to 101–124 tickers and drops RPM.
   This is a strategy-semantics decision, not a bug fix → **user decision**,
   with the numbers above.
3. **Entry anchor duality — REAL, undecided.** Two definitions of "entry"
   flip 13 ranked outcomes; blended basis is defensible for the position
   framing but must be labeled as basis, not "buy @".

## Decision menu (user)

- **A. Keep holdings-screen semantics** (status quo post-leak-fix): document
  the no-decay position policy in the charter with the measured age
  distribution; fix the reason label to "basis @X".
- **B. One staleness policy**: cap/decay position evidence at the 13F's 180d
  (`setups.py:143`) — ranked 339→124; RPM drops out; screen becomes
  "recent-disclosure, below-basis" only.
- **C. Backtest first**: run the rank-order harness under A vs B before
  choosing (both variants are one-line changes at `setups.py:217-222`).

## UNKNOWN

- Whether A or B has better forward returns: R7 is FAIL at current power, so
  the choice cannot be settled by the existing backtest — needs C.
- 15 accepted holdings rows have no same-actor event close (cannot be
  re-anchored); 79 excluded docs await the Gemini OCR pass; committee claim
  for Taylor remains UNVERIFIED.
