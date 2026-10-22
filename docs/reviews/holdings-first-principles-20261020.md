# Holdings check and first-principles review — 2026-10-20

Scope: ABT, ONDS, SMPL, SOXL, SPCX, VST, followed by a bounded project review.
Code reviewed at `24aca363`. Three parallel Sol audits; canonical DB opened
read-only. No estimator changes, canonical repairs, trading, or production
receipt publication. Purchase dates, fills, quantities, tax circumstances and
the user's agreed exit policy are UNKNOWN. This is not an execution ticket.

## Corrections to the preceding conversation

- Withdraw “entry filter proven” and “filter cuts losses.” Eligible outcomes
  were **worse**, not better, than ineligible outcomes in the saved comparison.
  Neither comparison identifies a causal filter effect.
- Withdraw “all findings resolved.” This review reproduces outstanding identity,
  extraction, replay and receipt defects below.
- A 90-day evaluation horizon is not an established instruction to liquidate
  every actual user holding after 90 days. No such user exit agreement is on file.
- A failed new-buy screen is not a sell signal. Missing ticker coverage is not
  evidence of a sale or proof that no real insider purchases occurred.
- The prior suite history was 1,602 passed / 11 skipped / one crontab failure,
  followed by a three-test doctor rerun passing after installation. That is not
  a separately observed full-suite run of 1,605 tests. No full suite was run here.

## 0. Requirements, verification and objective

| Requirement | Check and threshold | Measured status |
|---|---|---|
| Same issuer/instrument in filings and prices | Original source + canonical identity + dated listing; zero wrong issuer joins | FAIL: 10/12 SPCX institutional rows name other issuers; congressional SpaceX identity independently checked |
| Critical extraction survives replacement | Same-artifact transaction-critical identity comparison; zero lost critical fields | FAIL: Pelosi VST purchase retained with ticker NULL |
| Cutoff causality | Future-only append cannot change historical generation state | PASS legacy hash vector; FAIL schema-present vector; schema branch absent in current canonical DB |
| Live/replay eligibility parity | Identical candidates, weights, quotes and dividend state give identical ranking | FAIL: replay omits dividend input |
| Self-contained receipt recipe | Changing environment alone leaves replay unchanged | FAIL: 13F flag changes score 0.78 to 0.52 |
| Price freshness/executability | Timestamp every quote; scorer quote age <=5 days; do not equate a close with an executable order | PASS timestamp reporting; current bid/ask/fill UNKNOWN |
| Distinguish entry from exit | No inferred user P&L, maturity or liquidation quantities without fills and exit contract | PASS in this report; user-specific exit decision UNKNOWN |
| Evidence before predictive claims | Positive net chronological held-out 90-day SPY-relative performance with grouped uncertainty; charter promotion bar met | FAIL on saved evidence; corrected live-recipe performance UNKNOWN |
| Audit must preserve production data | Read-only connections, no production receipts, inspect git changes | PASS for audit commands; only this report added to repository |

Primary economic objective: net 90-calendar-day SPY-relative forward excess on
fixed candidate support, versus an equivalent fixed-size SPY investment. Parser
coverage, score, discounted historic basis and passing tests are not objectives
that substitute for economic performance. Correctness constraints win over
apparent returns. Before a new experiment, register the economic minimum effect,
clustered confidence rule, power/sample target, costs/fills and missing-outcome
policy; no new powered experiment or parameter selection is claimed here.

## Holdings assessment

### Market observations, not order prices

External regular-session closes **October 19, 2026, 16:00 US Eastern**, fetched
from the linked Stock Analysis pages. They were not ingested into canonical
prices and do not constitute audited adjusted-history generations.

| Ticker | External close USD | Local cached close USD, October 16 | Provisional action and reason |
|---|---:|---:|---|
| [ABT](https://stockanalysis.com/stocks/abt/) | 104.06 | 104.89 | Pause additions; existing-holding decision remains conditional. Ex-dividend entry block is not bearish evidence. Earnings October 21: explicitly decide whether to carry that event risk. |
| [ONDS](https://stockanalysis.com/stocks/onds/) | 6.06 | absent | No addition on this insider screen. Retain only under a separate, risk-sized speculative thesis; otherwise consider reducing. Local coverage cannot establish an insider edge. |
| [SMPL](https://stockanalysis.com/stocks/smpl/) | 11.00 | 11.19 | Strongest ordinary-stock reduction/exit candidate absent a deliberate turnaround thesis: weak old single-member support plus deteriorating operating results. Not a mechanical model sell. |
| [SOXL](https://stockanalysis.com/etf/soxl/) | 98.65 | absent | Priority leverage review. Reduce/exit if bought as a passive multi-month insider-following investment; retain only as an intentionally monitored leveraged trade with an explicit loss/risk budget. |
| [SPCX](https://stockanalysis.com/stocks/spcx/) | 147.96 | 146.03 | Strongest screen support among these six, not proven alpha. Conditional hold/review, not size-up; earnings October 20 makes event exposure an immediate separate decision. |
| [VST](https://stockanalysis.com/stocks/vst/) | 149.70 | 152.02 | Do not mechanically sell because its rank vanished: Pelosi's ticker was lost in replacement extraction. Pause additions until identity/coverage is repaired and rechecked. |

These are temporary review postures, not indefinite hold endorsements. Actual
size, diversification, liquidity needs and previously agreed exits can override
them. No current user returns can be computed from an old July blotter.

### Source and signal evidence

Reproduction from repository root:

```sh
python3 -m analyzer.cli setups --ticker ABT --ticker ONDS --ticker SMPL --ticker SOXL --ticker SPCX --ticker VST --as-of 2026-10-20 --no-receipt
```

Observed output: only **SPCX** ranked, score **2.34**, **7** congressional actors,
freshest supporting disclosure **41 days** old. ABT blocked for ex-dividend
2026-10-15; ONDS Citadel corroboration only; SMPL Tim Moore singleton below floor;
SOXL absent; VST Letlow singleton below floor and Burke lacking an entry reference.
Output retained at `/tmp/opencode/setups-1020.txt`.

In-memory quote sensitivity check: intercept the live `setups.score` call,
reuse its candidates/weights/watchlist/dividends, and supply the six external
October-19 closes to that same kernel. **SPCX remains the sole ranked ticker,
score 2.34**. This is not a new published recommendation receipt or a full
price-history reconciliation. SPCX's lowest selected reference is Salazar's
estimated **146.15**; external close is **1.24% above**, within the 5% tolerance.

ABT's block is a seven-day ex-dividend entry safeguard
(`src/analyzer/setups.py:208–229`), not an exit policy. External sources confirm
the October-15 ex-date. Attributing the entire valuation difference or stock
movement to a $0.63 dividend is not justified.

ONDS has unresolved institutional evidence, not a locally established initiating
buy: canonical row **354158**, `ticker=NULL`, `raw_asset_description='ONDAS INC'`,
Citadel, CUSIP **68236H204**, **3,244,935 shares**, quarter-end June 30 and
disclosure September 2. A quarterly inferred increase is not an exact trade.
Only 684 Form 4 rows exist locally; no claim of exhaustive corporate-insider
coverage is warranted.

SMPL's only congressional actor is Tim Moore: purchases February 3, February 11
and March 13, latest disclosure March 24. Multiple purchases by one actor do not
satisfy the independent-actor floor. Company [Q3 release filed with SEC](https://www.sec.gov/Archives/edgar/data/1702744/000170274426000021/a991_pressreleasexq3x2026.htm)
reports sales **−6.3%**, adjusted EBITDA **−22.5%**, and FY sales guidance
**−7% to −6%**. [Company events](https://www.thesimplygoodfoodscompany.com/events-and-presentations)
list October 27 earnings. Turnaround and earnings exposure, not an insider score,
would need to justify retention.

SOXL is a daily **3x** semiconductor ETF. [Issuer objective and risk disclosure](https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs)
explicitly distinguish daily leverage from multi-day cumulative returns and
require understanding and active management. Omission by an ordinary-equity
screen cannot itself determine expected returns; it does expose mandate mismatch
if this was purchased as an ordinary long-horizon insider-following position.

SPCX congressional instrument identity is confirmed by the original [House PDF](https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/20034963.pdf):
Space Exploration Technologies Corp. Class A Common Stock (SPCX) [ST], June 12.
Ticker reuse is date-gated in `src/analyzer/ticker_resolver.py:62–67,155–180`.
Read-only SQL finds **9 purchase rows / 7 distinct members**. This is neither
seven corporate insiders nor proof of seven independent investment theses.
Do not use the contaminated institutional population as extra conviction.

VST canonical row **370262**: Nancy Pelosi, **Purchase**, January 16, disclosed
January 23, document **20033725**, description **Vistra Corp. Common Stock**,
but **ticker=NULL**, origin `not_reported`. Prior version **358346** had ticker
VST and was retired October 18. Same artifact SHA-256
`8ddf562e617976594595f2e921d63f9aab8dfd898524673d4560dfe495cfd28f`;
validated replacement has 18 rows versus previous 19. Missing ticker removes
both event and reconstructed position support (`pipeline.py:367–375`,
`positions.py:655–660`). It does not establish that Pelosi sold. Even the older
two-actor score was only **0.52**, with no fresh evidence, not established alpha.
An independent in-memory download/text extraction of the [original PDF](https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/20033725.pdf),
page 3, confirms the literal **“Vistra Corp. Common Stock (VST) [ST]”** and
Purchase entry. Its explanatory text says **50 calls exercised, 5,000 shares,
$50 strike**, from options purchased January 14, 2025. Thus the economic event
was an option exercise, not a new discretionary open-market stock purchase.
`positions.py:7–11` deliberately counts exercises as share accumulation; that
is different from establishing fresh informed demand. Do not restore the ticker
and then relabel this exercise a fresh corporate-insider buy. The missing original
ticker is source-verifiable, but the exercise nuance also limits the old thesis.
[Vistra Q2 release](https://investor.vistracorp.com/2026-08-07-Vistra-Reports-Second-Quarter-2026-Results)
reports adjusted EBITDA +30% and reaffirmed $6.8–7.6bn guidance; that is a separate
fundamental case, not a repair to the missing filing identity.

## 1. First principles

Keep four quantities distinct: information available at disclosure, inferred
continuing insider ownership, proximity to an estimated insider reference, and
expected executable net excess return. The setup score measures weighted
eligible actor evidence; it is not a calibrated probability, fair intrinsic
value, or measured expected return. Inferring a bargain from a lower price alone
is unjustified. Congressional estimated basis is not necessarily the actual fill.

There are two serving products: recent-buyer consensus (`README.md:61–85`,
`member_ranking/buyer_scoring.py:15–17`) and actor-weighted fair-entry setups
(`cli.py:1804–1830,1996–2028`). Global documentation describing only the former
does not describe the latter. Distinct products are not themselves a defect;
which is the owner's primary execution contract needs explicit documentation,
not an unrequested model substitution.

## 2. Logic: verified defects and boundaries

| Priority | Verdict | Finding, evidence and minimal proposed action |
|---|---|---|
| Before corroboration deployment | REAL | 10 of 12 canonical SPCX 13F rows name other issuers, including Howmet, Rackspace and Extra Space Storage. `form13f.py:462–471` allows substring watchlist assignment. Use exact issuer/security identity or leave unresolved; audit the affected population before repairing. Historical keyword configuration UNKNOWN. |
| Before interpreting rank loss | REAL | Pelosi VST ticker-loss regression above. Compare replacement critical-field multisets to original pixels/provenance; do not fabricate a sale or silently restore a guessed ticker. |
| Before replay performance claims | REAL | Live passes dividends (`cli.py:1996–2005`); replay omits them (`replay.py:243–248`). One numerical vector ranks in replay at 0.52 but is blocked live. Pass identical dividend-state inputs; no gate tuning. |
| Before environment-independent replay claims | REAL | 13F enablement is runtime environment state (`setups.py:73–76`), not frozen in recipe parameters (`decision_snapshot.py:130–147,392–401`). Freeze resolved flag and explicitly replay it; preserve legacy uncertainty. |
| Before generation schema rollout | REAL, latent locally | Schema-present generation selection ignores as_of (`price_source.py:348–365`); future-only append changes historical generation ID. Current real DB lacks these tables. Define cutoff/knowledge-time contract and version state; do not claim current canonical scores changed from this latent branch. |
| Before historical skill claims | REAL mismatch | `scripts/backtest_assumptions.py:457–468` builds different weights and omits dividend state versus live. Re-evaluate the declared recipe on fixed support; do not tune from old contrasts. |
| Before setups execution claims | REAL contract gap, not generic simulator defect | Generic simulators buy next-session prices plus slippage (`portfolio_sim.py:203–244`, `portfolio/simulation.py:250–289`), without a setup-specific entry-reference limit. Their output does not validate execution of a fair-entry limit strategy. Establish which product is being simulated and its order contract before attributing results. |
| Before population-wide performance claims | UNKNOWN missingness effect | Historical `clustered_t` and `compare_groups` drop missing outcomes (`scripts/backtest_assumptions.py:735–765`). This is a complete-case estimate, not evidence for the entire candidate population. Report total/mature/missing counts by decision and group, exclusions and sensitivity; no measured bias direction or magnitude is claimed. |
| Reporting boundary | ALREADY-HANDLED | Old receipt verifier labels `pre-recipe anchor`; integrity is not historical scoring replay (`decision_snapshot.py:510–555`). Require `replay.status == 'pass'` for an actual recipe replay claim. |
| Strategy ownership | UNKNOWN | Which of two production-serving scorers is intended as the trading product, and what executable user exit rule applies. |

Default 13F corroboration is OFF. The wrong institutional rows are a data defect,
but their effect on the observed SPCX score is **not established**: seven
congressional actors produce 2.34 without corroboration.

## 3. Mathematics and numerical checks

For eligible actor a, contribution is `w_a * d_a`, with persistent position
`d_a=1`, otherwise `d_a=2^(-disclosure_age/90)`. Score is the contribution sum,
multiplied by 1.5 only when corroborated (`setups.py:277–286,327–354`). Eligibility
requires quote <=1.05 times reference and contribution >=0.05. Ranking is score
>=1, or score>0.25 with at least two actors or corroboration (`:32–45`). A lone
0.52 and two actors totaling 0.52 therefore legitimately have different ranking
membership. This discontinuity is not itself a reason to tune a floor.

Independent numerical vectors, using real production APIs and isolated
in-memory databases, are at `/tmp/opencode/narrow_setups_audit.py`.
Run with the project's system interpreter:

```sh
python3 /tmp/opencode/narrow_setups_audit.py
```

Observed assertions pass:

- Dividend parity: two actor weights 0.26 each; replay returns
  `(0.52, False, 2, None)`; live gives **0 ranked / 2 ex-dividend blocked**.
- Legacy cutoff hash unchanged after a future-only bar append.
- Schema-present cutoff hash changes from `yahoo:bd0ef7a2556b4bb033442505` to
  `yahoo:456f024ebd53f179a9b8ed7e` for unchanged October-16 query after appending
  October-19 price. No numerical score-change claim follows from identity alone.
- Receipt environmental sensitivity: flag on yields **0.78**, integrity true,
  replay pass; flag off yields **0.52**, same integrity true, replay FAIL on
  ranked/eligibility. Magnitude **0.26**, exactly 50% corroboration uplift.
- Real receipt inventory: **49 setups without recipe / 5 with recipe / 1
  consensus without recipe**. Sample old receipts pass integrity with explicit
  **pre-recipe anchor**, not recipe replay.

Aggregate portfolio effects of these defects remain UNKNOWN. Numerical vectors
prove violations, not their frequency or realized P&L.

## 4. Computer science / algorithms

This review does not establish a compute bottleneck or justify another model,
framework, optimizer or scanner. Reuse the shared scorer with explicit,
immutable decision inputs and identical live/replay loaders. Keep candidate
support fixed; report eligibility, missing references, missing prices and
unavailable outcomes rather than letting exclusions improve performance.
No throughput benchmark was run here; current latency/memory bounds UNKNOWN.
The shared-cash ledger and explicit all-or-skip execution in the generic
portfolio simulator are useful implemented constraints
(`portfolio/simulation.py:149–156,266–276`); do not call them broken simply
because a different setup product would require a limit-order contract.

## 5. Data-oriented design

The immediate requirement is stable issuer/instrument identity and preserved
critical fields across revisions, not more per-row heuristics. Existing raw
descriptions, artifact hashes, row versions and immutable receipts permit
targeted reconciliation. Keep original and replacement evidence; audit before
promotion. Do not patch canonical rows during a review or silently map all text
containing “space” to one issuer. Store all decision-affecting configuration,
including environment-derived flags, at the receipt boundary. A current mapping
or current generation state cannot silently stand for a historical one.

## 6. Empirical data and deployment conclusion

Measured evidence, not a new model fit:

| Saved measurement | Result | Interpretation |
|---|---|---|
| Skilled versus not, `data/backtest_assumptions/summary.csv:2` | t=0.808 | No demonstrated skill edge under this test |
| Fresh+skilled versus rest, `:3` | t=0.551 | No demonstrated freshness+skill edge |
| Eligible versus ineligible, `:5` | −2.560pp versus −0.522pp; difference −2.038pp | Not a loss-cutting result; selection contrast is not causal |
| Among-eligible score top minus bottom, `docs/reviews/rank-order-20261016.md:16–37` | −0.35pp; SE0.79; t=−0.44; reported CI [−1.94,+1.24]; 8 date clusters | No demonstrated rank-order edge; rows are not independent sample units |

These historical harness outcomes are not validated performance of a corrected
current recipe. Neither prior null findings nor eight clusters prove absence
of every possible edge. No promoted variant or new holdout result exists here.

Keep estimator and size ceiling frozen; paper/no-new-model-driven-risk is the
defensible model-deployment posture. Existing-holding reductions may instead be
justified by leverage, concentration, business deterioration or an agreed exit,
but need an explicit rationale separate from a blocked buy screen.

## Independent web review

The initial evidence-packet-only ChatGPT review completed under label
`ptr-fp-holdings-20261020`, job `j-8ac7d77e7e23`. It confirmed the central
decision-boundary and evidence concerns, but is not a whole-codebase review.
Its citation combining `summary.csv:2–5,16–37` is imprecise: the tercile results
are in `docs/reviews/rank-order-20261016.md:16–37`, not the five-line CSV.

**Access correction:** the coordinator incorrectly told the web reviewer it
had no repository access. DevSpace is configured to allow `/home/ubuntu` and
the `instance2-devspace` connector can access this checkout. A read-only probe
in the same thread successfully opened `/home/ubuntu/insider-trading` as
`ws_94133e8c35` at 02:45:42 UTC; server logs confirm successful shell execution
and `README.md` read at 02:46:04 UTC. No access-configuration changes were needed.

A new repository-grounded review was delivered as job `j-088e02868b76` in
[the same web thread](https://chatgpt.com/c/6ad6d0f7-046c-83ed-938d-319491bc2d51).
The observed transcript now contains direct code reads and read-only canonical
SQL queries. Completion requires a coverage matrix and actual verification;
it is **in progress**, not a completed whole-program audit. Follow-up job
`j-43f2b3cd460f` informs the reviewer of concurrent staging-only OCR work.

## Parallel subscription OCR continuation — October 20

Owner explicitly requested Z.AI subscription `glm-5.3-flash` parsing of remaining
documents. Verified existing endpoint configuration uses
`https://api.z.ai/api/coding/paas/v4` and that exact model; credentials were not
printed. Before restart, staging inventory was **990** records: **814 success /
176 error / 0 promoted**, across 2016 and 2017. No existing drain process was
found.

Launched existing `bash cron/backlog_ocr.sh` at **03:00:32 UTC**, using its
single-instance lock, authoritative unresolved-work-manifest export, two-worker
cap, six-hour whole-run budget and quota-stop behavior. The first observed log
confirms configured engine credentials; manifest export was still in progress
at the initial check. Log: `data/logs/ocr-backlog-cron/20261020/run.log`.

This is a staging-only drain, not canonical promotion. No audit approval or
budget gate is bypassed; no fallback to a metered API or different model is
authorized. Final extraction counts and provider availability for this run are
pending, not assumed successful from process launch.

## Verified continuation checkpoints

### Subscription drain

The first drain produced **424** 2020 staging records (**363 success / 1
no-transactions / 60 error**) before stopping without a clean completion line.
No cause of termination was established; no quota-exhaustion marker was found.
Its logs were preserved in
`/tmp/opencode/ocr-20261020-before-resume-1307.tar.gz` before resuming.

At 13:06:17 UTC, the existing wrapper was restarted under the transient user
service `ptr-ocr-backlog-20261020-resume.service`, with an explicit repository
`PYTHONPATH` and a 6h05m service ceiling. An initial launch without that path
failed to import `scripts` before parsing; the corrected launch was verified
active. This changes no parser source or persistent service configuration.

The frozen manifest's 3,676 unresolved items are **not** a remaining-work
denominator: the selector also skips already staged successes and previously
parsed documents. The 13:05 selection found **1,536 eligible unstaged/retry
documents** and **160 missing PDFs**. These are observations of that manifest
and selector, not a complete independently reconciled source inventory.

At 14:33 UTC, the Python staging worker was active, a successful result had
been written approximately five minutes earlier, and the processing log had
advanced at 14:29. Thus progress was verified beyond merely finding a live
wrapper. Subsequent observed 2020 status: **436 staged / 406 success / 1
no-transactions / 29 error**. Some improved successes replaced archived failed
attempts, so successful-record growth differs from total-file growth.
Subscription remaining balance is UNKNOWN; no metered fallback is authorized.

### Repository-grounded web review

The reviewer delivered substantial repository checks and reported **232 focused
tests passed**, not a fresh full-suite pass. The 13:37:44 continuation completed
its declared bounded checks, not an exhaustive corpus certification. Findings
below are attributed to that review unless independently checked here.

- **Same-artifact replacement comparison:** 4,552 shared source/document/hash
  groups; 49,955 earlier versus 49,300 current rows; 43,866 exact economic-row
  matches, 1,339 further conservative unique matches, and 4,750 earlier / 4,095
  current rows unpaired. Unpaired rows are unresolved differences, not proven
  extraction errors. All 1,832 strictly matched OCR non-null-to-null ticker
  changes were previously name-resolved; automatic restoration is not justified.
- **Official source distinctions:** Gill filing `20033476` distinguishes BTC
  `[CT]` from IBIT `[ST]`; filing `20030889` explicitly labels IBIT `[ET]`.
  Pelosi `20033725` explicitly reports VST `[ST]` as an option exercise. These
  are different economic mechanisms and instruments.
- **Fixed diagnostic scope:** on 27 tickers, the reviewer reconstructed 255
  candidates, 59 frozen actor weights and 20 dated quotes through live functions.
  Current rankings were SPCX 2.34, BTC 0.78, ETHE 0.78, IBIT 0.78 and BITC 0.52.
  A diagnostic eligibility exclusion of four non-corporate instruments removed
  the four latter rankings; SPCX remained 2.34. This is a correctness-boundary
  contrast, not a return improvement, general ticker-ban proposal or data repair.
- **Scope sensitivity:** ETHE scored 0.78 with three actors in the 27-ticker
  scope, versus 0.52 with two actors in a five-ticker scope. The reviewer is
  continuing verification of the candidate-assembly caller mechanism rather
  than treating unequal evidence sets as a mathematical scoring difference.
- **VST in-memory restoration:** restoring Pelosi's ticker alone changed VST
  from not ranked to ranked at 0.52. That demonstrates ranking sensitivity but
  is not a valid repair of option-exercise quantity/cost semantics or a sell rule.
- **Replay-clock vector:** reviewer reports a constructed 23.75pp difference
  between `replay.py:278–332` and next-session/entry-based 90-day labels. That
  is an isolated non-equivalence vector, not measured historical damage. All
  5,688 retained assumption-panel rows independently matched their expected
  next-session entry closes; no realized delayed-fill mismatch was found there.

Independent local read-only verification of two additional findings:

```sql
SELECT count(*), count(DISTINCT doc_id)
FROM canonical_transactions WHERE ticker='STATUS';
-- 624 rows, 354 documents
SELECT count(*) FROM prices WHERE ticker='STATUS';
-- 0
```

Canonical document `20011793` contains the Agilent purchase twice: once with
ticker NULL and description `agilent Technologies, Inc. (a) [ST]`, and once
with ticker `STATUS` and added filing-status text. `STATUS` is parser metadata,
not a verified security; 624 malformed canonical rows are not 624 proven
incorrect recommendations. Its absence from prices and rejection by ticker
eligibility remain meaningful existing protections.

An independent, no-write scalar reproduction used the actual
`parsing.cells._extract_instrument_type` and
`member_ranking.buyer_scoring._equity_transaction_fields`:

```text
'BTC [CT]' -> instrument stock; equity-eligible True
'iShares Bitcoin Trust ETF (IBIT) [ET]' -> instrument stock; equity-eligible True
'Agilent Technologies, Inc. (A) [ST]' -> instrument stock; equity-eligible True
```

Mechanism: `parsing/cells.py:559–603` defaults unrecognized classes to stock;
`buyer_scoring.py:395–425` does not explicitly reject crypto markers or the ETF
abbreviation. `download.py:1041–1105` can preserve an old non-null ticker on
matching row identity without independently checking its economic correctness.
No source or canonical repair was performed. Current instrument eligibility,
replacement continuity and replay parity remain unresolved correctness work.

The latest review continuation is job `j-c5f7723ab6b1`, requesting remaining
architectural coverage and source-backed invariants. It explicitly separates
a finite first-principles program review from exhaustive document certification
and future alpha validation; neither of the latter is claimed complete.

## Operational root-cause checks and narrow parser correction

### Partial-sale vocabulary

The read-only extraction diagnostic at 15:34 UTC found 29 live 2020 errors:
17 transaction-type failures, one date failure, two first-failure amount errors,
one page-count mismatch, four empty completions, three timeouts and one
disconnected transport. Every live type failure first rejected `Sale (partial)`
or its capitalization variant. The shared parser already accepted `partial
sale`, so this is a narrow normalization asymmetry, not demonstrated reversal
of buy/sell direction. Other critical fields still require source verification.

Coordinator correction: the existing `_normalize_tx_type` exact alias set now
also accepts `sale (partial)` as `Sale`. No wildcard qualifier stripping,
missing-date imputation, invalid-amount conversion, prompt change or scoring
change was made. Existing raw outputs are unchanged. Added regressions in
`tests/test_gemini_ocr.py` verify capitalization and continued rejection of
unknown qualifiers, missing transaction dates and invalid amount ranges.

Verification:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest tests/test_gemini_ocr.py tests/test_glm_stage.py tests/test_vision_dual.py -q -p no:cacheprovider
# 201 passed, 1 skipped in 45.71s
ruff check scripts/gemini_ocr_common.py tests/test_gemini_ocr.py
# All checks passed!
git diff --check
# no output
```

An offline, in-memory reparse of retained error outputs now validates **16
documents** without further model calls. Four outputs still fail amount-range
validation and one still fails transaction-date validation. This check did not
rewrite live records, retire attempts, promote data or approve a pixel audit.
The running Python worker had already imported the previous parser; the narrow
fix does not change its in-flight records. Subsequent fresh worker processes
will load it.

Empty-completion cause remains UNKNOWN: `vision_engines.py:361–374` discards
finish-reason/content-shape/usage diagnostics before raising the empty-content
error. Timeout budgets matched the configured `120 + 60 × pages`; no evidence
justifies silently increasing token/time budgets or treating these as quota
exhaustion. Missing critical dates remain fail-closed.

### Authoritative queue accounting

Read-only queue audit at approximately 15:34 UTC partitioned all **3,676 unique
manifest entries** exactly:

| Disposition | Count |
|---|---:|
| Staged success | 540 |
| Staged no-transactions | 1 |
| Skipped by broad canonical success check | 1,482 |
| Runnable, no staged record | 1,415 |
| Runnable retry after staged error | 78 |
| Missing at the selected PDF path | 160 |
| Total | 3,676 |

Thus that snapshot had **1,493 runnable entries**, not 3,676 untouched documents.
Global staging totals include 803 files outside this manifest. Counts can change
while the active subscription drain is running.

Two independently diagnosed selection issues prevent claiming completion:

1. `canonical_parsed_doc_ids` skips any document with any historic success,
   ignoring artifact/generation (`glm_stage.py:653–668,777–798`). Of 1,482
   canonical-only skips, only 13 had matching current-bound successes and row
   counts; 1,469 lacked a matching terminal artifact/generation binding. This
   does not establish that they all need OCR: verified prior evidence may be
   reconcilable without another read or canonical mutation.
2. All 160 explicitly “missing PDF” artifacts are already present under their
   recorded archive years, with recorded House artifact hashes. Queue selection
   instead used filing year (`ocr_zero_rows.py:308–337`), unlike acquisition's
   archive-year paths (`download.py:196–208,444–468`). Two more such path errors
   were hidden by the broad canonical skip. Numeric ID prefixes do not establish
   Senate provenance; blind redownloads are not justified.

An isolated-worktree implementation has been assigned for archive-aware,
artifact-bound selection and generation-aware terminal checking, reusing
existing canonical predicates. The active drain and canonical data remain
untouched by that lane; its changes require focused tests and review before
application. No force-promotion, metered fallback or blanket removal of
canonical checks is authorized.

## Finite web program review completed — 15:52:41 UTC

Job `j-c5f7723ab6b1` completed the remaining declared architectural review in
the existing DevSpace checkout. This supersedes earlier **program-review in
progress** statuses, but not outstanding source adjudication, corrections,
promotion approval or investment-performance validation.

The final phase reports **298 passing tests / 1 skipped / 12 passing subtests**:
128 Senate/canonical tests, 81 selected audit/promotion tests, and 89 execution/
price tests. These are separate focused runs, not a full-suite result or unique
cumulative test count across all review turns.

Additional repository-grounded findings and safeguards:

| Area | Measured result | Interpretation |
|---|---|---|
| ETHE ticker scope | ETHE alone and ETHE+empty-control: 7 rows, 2 actors, score 0.52; ETHE+RSP: 8 rows, 3 actors, score 0.78 | Adding RSP recruits Toomey's persistent ETHE position through `cli.py:1882–1943`; seven common rows and frozen weights were unchanged. The defect is caller-level actor selection, not the pure scorer. |
| Senate history | 14,904 canonical transactions, 1,594 distinct historical reports; active report inventory 156 | Additive transactions and latest-window report inventory are different coverage claims; historical rows are not thereby deleted. |
| Senate completeness | Failed/unavailable reports block refresh; authenticated paper-only reports can complete with no transactions | A complete source inventory does not establish complete transaction extraction. |
| Audit promotion | Current report: 30 documents, zero observed critical errors, 95% upper bound about 9.50%, budget 1%, `budget_met=False` | Pure gate returns False; no ordinary promotion is authorized. The explicit unaudited override was not used. |
| Next-session arithmetic | Independent 10bps-per-leg vector: security +9.78%, SPY +4.79%, excess +4.99pp | Matches fixed-horizon backtest arithmetic; missing benchmark entry correctly leaves outcome unavailable. |
| Cash accounting | $1,000 cash / 2 slots / $100 entry / $110 exit -> 5 shares, $500 retained cash, $1,050 ending equity | Tested accounting reconciles; not evidence of profitability. |
| Fill policies | One portfolio path rejects missing next-session price; another allows delayed fills | Different declared recipes, not interchangeable performance results. |
| Adjustment basis | Network-free raw [100,90], adjusted [90,90] response persisted adjusted [90,90], factors [0.9,1.0] | Consistent total-return series is not automatically an executable raw-price or whole-share fill history. |
| Output authority | Consensus 28-day/3-buyer discovery, actor-weighted setups, member descriptions and historical simulations have distinct contracts | None is a personalized SELL engine or calibrated expected-return/intrinsic-value estimate. |
| Horizon | Generic backtest default 60 days versus charter objective 90 days | Default output cannot be cited as the declared 90-day strategy experiment. |

The reviewer traced House/Senate admission, Form 4/13F, instrument/position
assembly, actor weights/setup calls, receipts/replay, prices/returns/execution,
audit/staging/promotion and historical evaluation. It explicitly retained
unexamined individual filings, intraday revisions and sealed future outcomes as
separate evidence gaps, rather than extending an architectural review into an
unfinishable whole-corpus certification task.

Priority correctness work remains: exact economic instrument eligibility;
source-backed replacement/preservation rather than stale ticker carry-forward;
wrong-issuer institutional mappings; ticker-scope-invariant actor selection;
live/replay and receipt-configuration parity; historical price-generation state;
and clear reporting of product/horizon/coverage differences. Frozen predictive
parameters were not changed. **Deployable net alpha remains UNKNOWN**, not
established by the tested safeguards or by historical diagnostic nulls.

## Queue fix prepared, not yet deployed

Isolated branch `fix/ocr-queue-artifact-binding`, commit
`ac5b4931ab70e9196a8b06a243b83c5aa2524226`, changes four files: queue export,
stage/artifact selection, audit artifact lookup and existing staging tests.
Its author reports **515 passed / 3 skipped / 24 passing subtests** and a
15:55 UTC read-only comparison of **1,491 -> 3,118** eligible candidates,
**160 -> 0** false missing-path notes, preserving 13 current-bound successes
and two verified invalid dispositions. The candidate count is not a count of
necessary new API calls.

Independent adversarial review is pending. The commit has not been merged or
cherry-picked into main, and has not changed the running worker or canonical
data. Combined verification with the coordinator's partial-sale alias is still
required before deployment. The explicit staging audit gate remains closed.

### Independent queue review: BLOCKED, not deployed

The independent reviewer ran 66 existing staging tests and four isolated
reproductions against `ac5b4931`, finding:

1. **Stale-generation reuse:** `(doc_id, PDF hash)` reuse can silently hide work
   when a bound older-generation stage record cannot pass current promotion.
2. **Pixel/hash race:** audit lookup verifies one read but renders a mutable
   original path. Replacing it in between produced two rendered pages for a
   one-page staged record without a reported problem.
3. **Serial false success:** `workers=1` omitted the blocked-artifact count,
   printing zero and exiting zero after refusing an artifact; parallel mode
   correctly returned a blocked exit.
4. **Artifact root escape:** legacy unbound manifests could follow file or
   ancestor symlinks outside the approved data root and send unrelated bytes to
   the OCR reader.

All four are routed back to the isolated implementation branch with regression
requirements. Approval and combined tests are still pending. The first commit's
large passing suite is not treated as sufficient approval; no queue patch has
been applied to main or to the active service.

The active original queue drain reached its expected per-year timeout for
2020 (`exit=124`) and proceeded to 2019 under the same whole-run budget. The
subsequent status sample found **15 staged 2019 successes / zero errors**, and
2020 at **441 records: 408 success / 1 no-transactions / 32 error**. The worker
did not stop the entire drain merely because a year's time allocation expired.
Promotion remains zero across staging years.

### Queue correction commit awaiting independent re-review

Commit `d945f469bfa4b0a15d94bb12cc1af8b02d0d9da7` on the same isolated branch
addresses all four blockers: archive/hash/current-generation-compatible reuse,
single verified audit snapshot, serial blocked counts, and root-anchored file
descriptor reads. It conservatively rejects interior symlinks while supporting
an explicitly approved data-root symlink. Stale or unbound staged evidence is
reported blocked rather than silently reused or rebound.

The implementation lane reports **80 focused tests passed** and **529 broader
tests passed / 3 skipped / 24 passing subtests**, with no new Ruff findings.
These are not yet independent approval. Re-review is running against both
queue commits, and neither has been deployed to main.

A native `systemctl --user start --wait` waiter was attached while the owned
transient drain service was active. It waits for that existing bounded run to
finish without a sleep/poll loop, allowing the coordinator to resume deployment
checks only after the active writer exits. Approval, combined tests and explicit
staged-evidence reconciliation remain prerequisites; no canonical promotion is
authorized by these changes.

### Second independent queue review: still BLOCKED

Re-review of `d945f469` confirmed that all original four reproductions now
pass and independently ran 80 passing tests. It nevertheless found three
remaining boundary violations:

- Re-resolving the approved root in nested helpers follows a replacement
  symlink after selection. An unbound legacy manifest still caused an external
  PDF to reach the reader after retargeting the physical data-root path.
- Audit bundling copies a staging record, then independently rereads its live
  JSON for PDF verification. Concurrent restaging can therefore bundle the old
  attempt but render the new attempt without a reported problem.
- Promotion resolves the artifact safely, then reopens its path using ordinary
  `pdf_sha256`. A race caused an external PDF to be read before its mismatching
  hash was rejected. No incorrect canonical insertion was demonstrated.

The isolated implementation is being corrected around three shared invariants:
pin the approved root identity once per operation, snapshot staged-record bytes
once, and render/hash/consume the exact verified PDF snapshot without reopening
mutable paths. The patch remains undeployed; tests passing against the earlier
reproductions do not waive these new blockers. Live OCR and the promotion gate
remain unchanged.

### Third queue candidate and observed parser benefit

Commit `455563bafb3c17f9c5218c7419178dc92f3d2938` on the isolated queue branch
retains a root FD per operation, snapshots staging-record bytes once, and shares
verified PDF bytes/digests across stage/audit/promotion without reopening the
original path. The author reports 535 passing tests, 3 skipped and 24 passing
subtests. A third independent review is running; this is not yet approval or
deployment, and main remains untouched by the queue branch.

A subsequent read-only observation of the ongoing 2019 drain found **101
successes / 3 errors**. Fourteen successful records retain the literal
`Sale (partial)` in their raw output, demonstrating the narrow shared-parser
alias working in newly started worker processes. Representative documents:
`20010918` (82 rows), `20010971` (39), `20010975` (109), `20011033` (11), and
`20011063` (30). This verifies schema acceptance without erasing raw evidence;
it does not certify pixel accuracy or permit canonical promotion.

### Independent approval received; combined verification pending

The third reviewer **APPROVED `455563ba` for the reviewed queue/snapshot
scope**, independently running 86 tests and all seven prior reproductions.
Private-read spies found zero private reads during the tested symlink, root
replacement and promotion races; actual rendering used matching immutable
record/PDF snapshots. Root-descriptor lifetime, overlapping workers, cleanup,
legacy entries, approved root aliases and explicit external DB configuration
were checked. Existing stale/unbound successes block explicitly without mutation.

Approval does not certify every possible race or authorize promotion. No live
canonical promotion, provider call or full broad-suite rerun was performed by
the reviewer. Combined tests with the main partial-sale alias are now assigned
to the isolated worktree, leaving the approved queue commit unchanged and main
untouched. The existing bounded drain was still active at the deployment check,
so no queue merge or replacement of its running code was attempted.

## Queue fix deployed after drain completion — October 20

Owner reported the run finished at 19:01 UTC. Before deployment the coordinator
independently verified the owned unit inactive/dead, MainPID zero, no remaining
drain/staging processes, and a clean `doctor exit=0` completion in the run log.
Pre-deployment source edits, manifest, audit state/history and run log were
backed up to `/tmp/opencode/queue-fix-predeploy-20261020.tar.gz`.

The approved queue branch was fast-forwarded into main **under the shared drain
lock**. Main HEAD is now **`455563ba`**. The existing partial-sale edits and
review document were preserved. No queue patch was installed during active OCR.

Actual post-deployment verification on main:

```text
Relevant staging/OCR/vision/parsing/audit/database suite:
542 passed, 2 skipped, 24 subtests passed in 127.00s
doctor: ALL PASS
ruff (partial-sale source and tests): All checks passed!
git diff --check: no output
```

This is a focused suite, not a fresh full-suite claim. Test skip counts differ
from the isolated integration environment; the actual deployment result above
is the measured main result. No canonical promotion was performed.

### GLM staging versus pixel audit

Post-run live staging inventory is **1,853 documents**:

| Stage year | Success | No transactions | Error | Total | Promoted |
|---|---:|---:|---:|---:|---:|
| 2016 | 292 | 0 | 114 | 406 | 0 |
| 2017 | 522 | 0 | 62 | 584 | 0 |
| 2018 | 70 | 0 | 0 | 70 | 0 |
| 2019 | 334 | 1 | 17 | 352 | 0 |
| 2020 | 408 | 1 | 32 | 441 | 0 |
| **Total** | **1,626** | **2** | **225** | **1,853** | **0** |

These are extraction/schema outcomes, **not pixel-audit approvals**. Relative
to the original 990-record inventory, there are 863 additional live records;
archived retries and corrected failed attempts mean this is not a count of API
requests or all successfully processed new documents.

Audit history has **44 entries / 38 distinct document IDs**. Latest recorded
verdicts by document: **34 pass / 3 cosmetic fail / 1 critical fail**. Five of
the pass stamps are not content-bound to the current staged record; only **29
current pass stamps** satisfy `verdict_binding`. The five needing re-audit are
`20006470`, `20006657`, `20006682`, `20006694`, and `20006832`.

Of the 1,628 successful/no-transactions current records, **1,593 have no recorded
audit verdict**; 35 have audit history. Historical unique-document counts are
therefore not equivalent to current successful-record audit coverage. Doctor's
history-based inventory reports 1,815 unaudited staged documents, including
extraction-error records; this is a different denominator.

The existing frozen audit report is dated October 18, not a new audit of the
expanded corpus. It contains a 30-document lot with zero observed critical
failures, a **95% upper bound of 9.5034%**, and a configured **1% maximum-rate
threshold**. `budget_met=False`; the actual ordinary promotion gate returns
**False**. One critical failure remains visible in population audit history.
The report also says **LLM-assisted/silver-standard, not a substitute for human
pixel verification**, and `omission_check_status='not_run'`. No stronger
human-verification or completeness claim is warranted.

### Fresh post-fix queue check

A fresh official manifest was exported to the non-production diagnostic path
`/tmp/opencode/glm-work-manifest-after-queue-fix-20261020.json`; it contains
3,676 unresolved entries. Read-only selection under the deployed code found:

- **2,713 runnable entries**, not necessarily new model calls.
- **Zero missing-PDF path notes**.
- **938 explicit staged-identity reconciliation blocks**, instead of silently
  accepting stale/unbound staged evidence.

No manifest or record was silently rebound to bypass those blocks. The live
manifest and staged records were not rewritten by this diagnostic export. The
canonical database remained **776,482,816 bytes**, modified October 19 at
02:54:57 UTC, with **55 production decision JSONs**. No new OCR run or canonical
promotion was started as part of these deployment checks.

## Work resumed after owner correction

The coordinator had paused too broadly after deployment reporting. The promotion
gate does **not** prohibit further staging, source reconciliation or independent
pixel review. No new owner decision or confirmed quota stop required that pause.

At **22:33:32 UTC**, the existing subscription drain was restarted as transient
unit `ptr-ocr-backlog-postfix.service`, using the same Z.AI coding endpoint,
`glm-5.3-flash`, two-worker cap and shared lock. It exported a fresh live
manifest, found no 2026 work and proceeded to 2025. A native completion waiter
is attached. No additional model process was started outside the two-stream
budget, and no canonical promotion was invoked.

In parallel:

- A read-only operator reconciliation inventory is classifying the 938
  incompatible staged identities before any proposed versioned re-staging.
  Existing evidence will not be silently rebound or old verdicts inherited.
- The existing sampler froze **10 documents, two per staged year**, in
  `audit-sample-20261020.json`, with immutable review bundles. Eight are
  asset-review flagged, so this pilot is deliberately risk-enriched; its
  partially resolved failure percentages are not corpus-wide error estimates.
- Independent Sol auditors actually viewed all pages for six assigned
  documents (22 pages, 163 rows). The coordinator viewed three more pages for
  the two 2018 documents. The web session was assigned the remaining 2017
  checks and a second opinion on an `[OT]` classification; its job is queued
  after a connector thread-read error, not presumed delivered.
- The coordinator is the sole audit-history writer; audit delegates return
  proposals only, avoiding concurrent append/stamp races.

Five unambiguous critical identity-context omissions have been recorded against
the frozen attempts: private-equity capital-call context in `20005634` and
`20006205`, a bond-call description in `20008705`, private-security class markers
in `20012750`, and debt maturity/coupon identity in `20016765`. Numeric fields,
reported directions and row multiplicities matched the viewed pixels. These
are not cosmetic spelling failures or evidence that every extracted date/type
is wrong; they show loss of economic-instrument meaning in the five-column
extraction contract.

Proposed failures based only on omitted redemption or exchange notes are under
separate severity adjudication rather than automatically becoming sticky
critical failures. The CLOW `[OT]` case is also unresolved pending classification
review. The ordinary report on the new frozen lot returned **exit 2 / incomplete**:
six documents currently resolved (one has a matching pre-existing pass), five
critical failures and four unresolved; `budget_met` cannot be certified. These
are LLM-assisted/silver-standard pixel checks, not human verification. Staging
continues while promotion remains closed.

### Reconciliation proposals and audit severity adjudication

The read-only reconciliation inventory verified that **all 938 blocks** were
missing archive/generation metadata, not different PDF bytes: each retained
hash matched an anchored local PDF and current recorded House acquisition.
Original model, parser and prompt identities matched; raw responses passed
strict page/schema checks. This is not evidence of pixel accuracy or a need
for 938 fresh provider calls.

The preparation lane produced **939 dry-run, fresh unaudited proposals** under
`/tmp/opencode/ocr-raw-replay-proposals-20261020/`: 924 unsampled metadata-binding
replays and 15 exact partial-sale parser recoveries. Fourteen sampled records
remain protected. Existing live attempts were not changed. Each proposal retains
raw/model/prompt provenance, original record and raw-response hashes, original
reading time, current parser hash and an explicit replay marker; new parsing
time is not represented as a new model reading.

The current strict recovery count is **15**, superseding the earlier 16-record
observation; subsequent retries changed the live evidence population. Two
otherwise relevant partial-sale attempts still fail amount validation, and
other invalid amount/full-sale outputs remain rejected. No critical fields
were invented to increase the recovery count.

Plan and inventory:

- `/tmp/opencode/ocr-reconciliation-operator-plan-20261020.md`
- `/tmp/opencode/ocr-reconciliation-inventory-20261020.json`
- `/tmp/opencode/ocr-raw-replay-proposals-20261020/proposal-manifest.json`

An independent review is checking the proposed allowlist and procedure before
any publication. Application requires writer inactivity, exclusive drain lock,
current hash/generation/source revalidation, protected-sample exclusion and
preservation of old attempts without pruning. No proposal currently authorizes
canonical writes or inherited audit approval.

For the new pixel lot, a separate severity review adjudicated `20012318` as
**pass**: the source explicitly reports S and the staged structured-note
identities, indices, maturity dates and `[OT]` markers remain present. Omission
of the supplementary word “Redemption” is not an established critical error
under the declared normalized P/S/E contract. The parent recorded the pass
without ever recording the unsupported proposed failure.

Exchange document `20017632` remains **unresolved**, not automatically failed:
outgoing asset identity and Exchange fields match, but received-security names
in source notes are absent. Whether both exchange legs are mandatory exceeds
the currently explicit identity contract; that ambiguity must not silently
become either a sticky critical failure or approval. It does not block other
parsing, reconciliation or audit work. A context-preserving shared-prompt pilot
is being prepared in isolation, with no provider calls or live prompt changes
while the active two-stream drain runs.

### Replay publication blocked; evidence remains usable

The independent reviewer verified every one of the 939 proposal payloads and
all 958 inventory records, including hashes, source acquisition, exact parser
results, page attribution, unaudited state and replay provenance. Three rebuilt
canaries matched apart from timestamps; no proposed ID intersected current
sampled or historically audited IDs.

Publication was nevertheless blocked on two concrete procedure defects:

1. Moving the old live record to its archive before writing the replacement
   creates a crash window where the selector sees the document as runnable and
   may send it back to the provider. Keeping recoverable archive bytes alone
   is not sufficient. The corrected procedure must preserve old live bytes
   while copying/verifying the archive, then atomically replace live, with
   explicit pending/completed/HOLD restart states under the same drain lock.
2. Refresh eligibility could admit explicit stale/wrong binding metadata or
   row/member disagreement, beyond the reviewed missing-metadata category.
   The corrected procedure must pin literal approved IDs, proposal/source hashes
   and exact allowed differences; refresh can shrink that set, never broaden it.

Only temporary plans/proposals are being corrected. No replay proposal has
been published to production staging or used to inherit audit approval.

### Context-preserving extraction candidate, not deployed

Isolated candidate `176a85d7` on `candidate/extraction-context-prompt` adds two
paragraphs to the single shared prompt. It requests verbatim printed ticker/
class markers and economically identifying notes inside the existing asset
field, canonical P/S/E directions, no guessed unknown critical fields, no extra
exchange rows, and no administrative filing-status noise. Validators, schema,
thresholds and page coverage are unchanged. The author reports 281 tests passed
and one skipped; this does not demonstrate actual model extraction quality.

Prompt recipe identity changes from
`1255f8c399afab203e7639c099861628c9855ecf54b37f2a227d7ccd2d9499aa`
to `00833ce6f15123d3a33544737ab38c3b2aac8dc99a4c7a59adc56f191c88884b`.
No old cache, record or verdict is rewritten as if it used the new recipe.
Independent review is checking cache/queue impact and a five-document isolated
pilot before any provider call or deployment. The draft procedure names a venv
interpreter contrary to project policy; it must be corrected to system Python
before execution. The active two-stream subscription drain is not interrupted
or over-subscribed for this pilot. Actual quality improvement is UNKNOWN.

### Context-pilot review: blocked before provider execution

Independent review of `176a85d7` ran 75 parser tests (one skipped), confirmed all
five source-PDF hashes/page counts, and reproduced three pilot/design gaps:

- The ordinary client discards successful response envelopes and retry history,
  so the proposed pilot could not retain its promised finish-reason, usage,
  response/model identity and per-attempt diagnostics. A pilot-only capture
  around the existing client is required before calls are approved.
- Literal source `|` characters conflict with the four ASCII field separators.
  An explicit deterministic asset-text serialization rule is required, without
  loosening the five-field parser. No pipe occurrence was claimed in these PDFs.
- Future cache/work-population consequences were unspecified. A prompt change
  invalidates Gemini cache reuse/rebuild identity, while compatible GLM staged
  records are skipped independently of prompt hash. Deployment therefore needs
  a recipe/disposition inventory, preserved old evidence, explicit bounded
  re-extraction policy and fresh audits—not automatic mixed-recipe acceptance.

Corrections, including the system-Python command fix, are assigned to the
isolated candidate. Main's prompt, the 939 replay proposals' pinned old recipe,
live records and active provider streams remain unchanged. No pilot provider
request has been sent.

### Corrected pilot wrapper: additional independent blockers

Candidate correction `171e157d` added pilot-only attempt capture, explicit
asset-pipe serialization and recipe-migration accounting; its author reports
289 system-Python tests passed and one skipped. Re-review independently ran
245 relevant tests (one skipped), confirmed the source hashes/page counts and
candidate prompt hash `78fca698…6077becd`, and verified the original gaps were
addressed for ordinary responses.

It still blocked pilot execution on two reproduced issues:

- Automatic HTTP redirects bypassed the wrapper counter: 15 counted attempts
  generated 30 adapter sends, including image POSTs to another host. The
  wrapper must disable redirects, capture the redirect response and reject it.
- Parsed-envelope redaction did not sanitize the separately saved raw JSON
  text. Escaped known credentials and unknown sensitive-key values remained
  recoverable there. JSON response text must be serialized from the sanitized
  decoded envelope, with the transformation explicitly labelled.

These are being corrected only in the isolated candidate. Bulk parsing is
uninterrupted and still uses the original shared prompt.

The coordinator independently bound the five proposed pilot originals to the
frozen source bundles and attributed actual pixel reviews (16 pages, 104 rows)
in `/tmp/opencode/context-pilot-pixel-binding-20261020.json`. The file includes
original PDF hashes, frozen record hashes and current page-image fingerprints;
it explicitly states `pilot_authorized=False`. Those reviews found omissions
in the old extraction, not correctness of outputs from the unrun candidate.

### Corrected operator publisher and pilot transport: re-review underway

The replay lane prepared a default-dry-run operator publisher at
`/tmp/opencode/ocr-reconciliation-apply-20261020.py`, SHA-256
`81707406f9962def22d3b193560b3f6ae93748910d7d5efba0e61222f116f00e`.
It reports 16 passing temporary-state checks, including process-crash/resume
and strict eligibility negatives, with all 939 frozen proposals still pending.
No publication has occurred. Independent re-review is checking the live-preserving
archive copy, atomic replacement, restart states, pinned allowlist/source hashes,
protected audit IDs and exact drain-lock ownership before canaries can run.

Pilot candidate correction `a068559e` reports 298 offline tests passed and one
skipped. It forces redirects off, checks approved request settings before
transport, and serializes JSON evidence only after decoded-envelope redaction.
Adapter-level fake tests report at most 15 sends and zero cross-host sends;
those claims are now undergoing independent re-review. No live prompt was
changed and no pilot provider call occurred. Pilot approval remains limited
to an isolated experiment in an exclusively reserved provider window; it can
never imply extraction-quality or production-deployment approval.

### 23:10 UTC checkpoint: approvals received, drain still active

Independent review approved the frozen publisher for **operator canaries only**
(`20005855`, `20012719`), after 16 self-checks and additional temporary-lock,
active-unit, new-audit-lot and full-batch-preflight rejection checks. The
coordinator verified publisher SHA `81707406…116f00e` and reran its read-only
two-ID preview: both remain pending. No publication occurred: the owned bulk
unit is still active with the sole `--workers 2` parser in year 2025. Audit
writes will pause before any exclusive publication window.

Candidate `a068559e` independently passed finite isolated-pilot review: 254
relevant tests passed, one skipped; redirect and credential reproducers no
longer failed. This approves the **experiment only**, not extraction quality,
deployment, migration or inherited audit verdicts. The reviewed runner is
`/tmp/opencode/run-approved-context-pilot-a068559e.py`, SHA
`57fad9850f02167a7bd23bcd18950e050f9a0623d3653dba7320c7f1d077f819`.
It has not run. Its Python body does not acquire the drain lock: the coordinator
must wrap it with exclusive ownership after natural bulk completion and reserve
provider capacity for its one-worker, five-PDF, 15-request/30-minute experiment.

The undelivered web audit job `j-5ef2b23ceec0` was cancelled after four connector
thread-read failures. The coordinator directly viewed both 2017 one-page sources
and the CLOW source. The two ETF records match all required fields and received
content-bound passes after checking live/frozen stable hashes. The current lot
is **8/10 resolved: 3 passes, 5 critical failures, 2 unresolved**. `20010268`
remains unresolved because its source `[OT]` classification is absent from the
bare staged `CLOW`; no corporate-equity identity is inferred. `20017632` remains
unresolved on exchange-leg identity scope. The report exits 2/incomplete and
does not certify the budget. LLM-assisted pixel checks are not human verification.

Read-only live inventory at **23:09:56 UTC**:
**1,878 records = 1,650 success + 2 no-transactions + 226 errors**. Compared with
the stopped-run inventory, year 2025 has added **24 successes and one error**;
the latter is an explicit 26-versus-27-page mismatch, not silently accepted.
Inventory file: `/tmp/opencode/glm-live-status-20261020-2310.json`. These counts
are a non-atomic observation during active parsing. No canonical promotion or
live prompt change occurred. The existing completion waiter remains attached.

## October 21 continuation: canaries published, isolated pilot started

At 14:47 UTC the owned drain unit was confirmed inactive with no remaining OCR
processes. Its run reached the shared six-hour deadline, saved the staging
backup and finished with `doctor exit=0 / ALL PASS`. Before reconciliation,
doctor reported 1,706 success, 11 no-transactions and 249 errors. This is an
operational health check, not a certification of extracted transaction accuracy.

The coordinator paused audit/sample/verdict writes and applied only the two
independently approved literal canaries using the unchanged pinned publisher:
`20005855` and `20012719`. Both published successfully under its exclusive drain
lock. Subsequent checks verified exact old archival bytes, exact approved new
live bytes, fresh-audit-required provenance, `audit=None`, completed restart
states and empty queue selections/notes for their current artifact identities.
The latter recovered 69 rows from retained partial-sale output, with no provider
call or invented critical field. No canonical promotion occurred.

Verification evidence:
`/tmp/opencode/ocr-reconciliation-canary-result-20261021.json`.
An independent read-only reviewer is verifying those production canaries and
preflighting an explicit bounded 100-record binding-only batch. It may not apply
records, expand the frozen allowlist or acquire a competing lock.

The coordinator then launched the approved five-document context-prompt pilot
from isolated commit `a068559e`, under exclusive ownership of the same existing
drain lock. Original PDF, immutable record and page-image fingerprints were
rechecked against the pixel-binding attestation. Capacity is reserved for one
subscription worker; the experiment retains its 15-request/30-minute bounds,
attempt capture and no-deployment/no-promotion conditions. The launcher is a
notified background command, not an unobserved detached process. Its log is
`/tmp/opencode/context-pilot-approved-a068559e-20261021.log` with a separate
operator authorization file. Results remain UNKNOWN until it finishes and new
outputs are compared independently against the source pixels. Main's prompt
and source pins are unchanged.

At 14:49:34 UTC the live staging inventory was **1,966 documents: 1,707 success,
11 no-transactions and 248 errors**, with zero unreadable live JSON files. A
new timestamped `glm-backlog-20261021-postcanary-*.tar.gz` snapshot was saved
under `data/backups/glm-backlog/` after publication. Pilot outputs are isolated
under `/tmp/opencode` and do not enter these production staging counts.

### Bounded batch approved; pilot comparison begun in parallel

Independent production-canary verification passed and approved the exact next
100 binding-only pending records (all grouped in 2017), excluding all 47
protected IDs and all remaining error recoveries. Reviewed literal list and
individual hashes: `/tmp/opencode/ocr-reconciliation-batch100-20261021.json`,
SHA `86342b86f7ac337dcea9f04c8ef440849753773a1e0218b81cb8259da22a78bc`.
The reviewer established unchanged audit/sample bytes during that review, not
an independently supplied pre-canary baseline. Application still requires pilot
completion, inactive bulk unit, exclusive lock, paused audit writes and fresh
whole-batch preflight; any drift holds the entire batch. No batch publication
has yet occurred.

The actual pilot evidence root is `/tmp/opencode/context-pilot-696kinse`.
Two complete pilot records (`20005634`, `20008705`) were already saved while
the remaining documents continued. An independent read-only pixel auditor was
assigned those two finished outputs against all five source pages (39 rows),
including the private-fund and bond-call context previously lost. Delegates may
return proposals only; no production audit history writes or implied quality
approval are authorized by a successful parse.

### Pilot finished; first 100-record batch published and verified

The isolated pilot finished with **exit 0, five requests and five saved records**
(no retries). Its exclusive provider window ended. The coordinator then applied
the exact independently approved 100-ID batch using the unchanged publisher,
after verifying the reviewed plan/publisher hashes, inactive unit and unchanged
audit baseline. Whole-batch preflight ran under the publisher's exclusive lock.
All 100 published, exact new live and old archival bytes verified, rerun states
were completed, audit/sample hashes remained unchanged, and the combined queue
selector returned no candidates or refusal notes. Evidence:
`/tmp/opencode/ocr-reconciliation-batch100-result-20261021.json` and the matching
apply log. **102 replay records total** are now published, all unaudited; there
were no replay provider calls or canonical promotions.

Four pilot documents have now passed independently attributed actual pixel
comparison: 10 pages, 61 rows. Bound proposals are saved inside the pilot root
as `pixel-review-first-two.json` and `pixel-review-next-two.json`. The second
reviewer initially proposed failures solely for omitted administrative filing
status/account notes; those proposals were withdrawn after checking the actual
contract. Filing-status text is explicitly excluded, and missing account labels
alone do not establish economic-security identity loss. No unsupported failure
was entered in production audit history. Both `[PS]` securities/reinvestment
notes and the Crown Castle maturity/coupon now match their source pixels.

The final 43-row exchange document is under independent all-six-page pixel
review. Five selected known-failure documents cannot establish corpus accuracy,
even if all five pass. The main prompt and original production records of these
five documents remain unchanged; there is no inherited audit approval.

A read-only reviewer is partitioning the remaining frozen reconciliation set
for explicit sequential chunks of at most 100, rather than a broad 939-record
apply. No further records may publish without that review, current whole-chunk
preflight, the same exclusive lock and paused audit writes.

### All five selected pilot cases passed; complete evidence backed up

Independent actual-image comparison of the final six-page/43-row exchange
document passed. ETFC's exchange for Morgan Stanley and AMTD's exchange for
Charles Schwab are retained with the source merger descriptions, without
invented extra transactions. `[GS]`, ADR/share-class details and all 16 printed
partial-sale mechanisms match. All source PDF/page fingerprints were checked.

The isolated pilot now has **5/5 proposed passes, 16 pages and 104 rows compared**.
Its five requests all returned HTTP 200 and `finish_reason=stop`; attempt files
retain request bindings, timestamps and full sanitized envelopes with usage.
No retry or provider error occurred in this finite experiment. This is a
selected known-failure repair test, **not a probability sample or corpus-wide
error-rate certification**, and the reviews remain LLM-assisted/silver-standard.

Bound per-document proposals and aggregate result:
`/tmp/opencode/context-pilot-696kinse/pilot-result.json`.
The complete source/response/recipe/authorization/review bundle was saved to
`data/backups/glm-backlog/context-pilot-a068559e-20261021-complete.tar.gz`.
A separate timestamped `glm-backlog-20261021-after102-*.tar.gz` snapshots the
production staging state after the first 102 replay publications.

There is no live prompt deployment or pilot-record promotion. The candidate
author is preparing the required read-only recipe/cache population inventory
and a bounded new-work continuation plan while the independent reviewer checks
the remaining frozen replay chunks. Main's old prompt/source pins remain fixed
until that reconciliation window is completed or explicitly held.

### Remaining replay chunks approved and executing

The independent reviewer verified **102 completed, 837 pending, zero HOLD**
against the original frozen payloads. Pending records comprise 823 binding-only
replays and 14 literal strict-parser partial-sale recoveries. All 47 protected
IDs remain excluded; source pins and audit/sample hashes match the baseline.

The reviewed plan is
`/tmp/opencode/ocr-reconciliation-remaining-batches-20261021.json`, SHA
`e57b2d61627a6fb341b653fe2fe0768a39bf4c5be283549d0f86b2ea26eb90c0`.
The coordinator launched exactly its ten sequential chunks: eight of 100, one
of 23 binding-only records, then 14 error recoveries. Each uses the unchanged
publisher's exclusive-lock/whole-chunk preflight; subsequent exact live/archive,
source/audit hash and queue-selector checks must pass before the next starts.
No skips, expanded IDs, pin updates, model calls or canonical promotion are
permitted. A failure stops later chunks and preserves partial-state evidence.

Application logs and per-chunk verification live at
`/tmp/opencode/ocr-reconciliation-remaining-applied-20261021/` with a durable
verified-chunk journal. This is a notified background command. Its launch is
not a claim that all chunks have completed. Production audit writes and prompt
deployment remain paused throughout this old-recipe reconciliation window.

### All 939 replay publications completed; post-run health caveat

All ten remaining chunks completed and passed post-publication verification.
Together with the first 102, this finishes **939 fresh unaudited publications**:
924 binding-only replays plus 15 strict parser recoveries. Exact archived old
bytes, approved new bytes, source pins, audit/sample hashes and compatible
selector states were checked. No replay model calls or canonical promotions
occurred; all 47 protected document IDs were excluded.

Staging and operator evidence were backed up under `data/backups/glm-backlog/`:
timestamped `glm-backlog-20261021-after939-*.tar.gz` and
`reconciliation-939-evidence-20261021.tar.gz`. A fresh authoritative diagnostic
manifest was exported to `/tmp/opencode/glm-work-manifest-after939-20261021.json`
at 15:05:59 UTC; its 3,676 entries remain canonically unresolved, not all runnable
provider work. The staged inventory is **1,966 = 1,721 success + 11 no-transactions
+ 234 errors**. Reconciliation did not certify economic content accuracy.

The fresh doctor invocation is **NOT ALL PASS**: all checks except
`cron.ocr.engagement` passed. The failed check labels the October 21 cron entry
truncated because it has no doctor-exit line. Actual log inspection shows that
03:30 invocation explicitly skipped on the shared lock held by the previous
day's still-running drain; the winning October 20 invocation later completed
with doctor exit 0. No log lines were edited or success fabricated. An isolated
candidate is investigating fail-closed recognition of evidenced overlapping
lock skips, including stale/failed/no-winner cases and appended invocation logs.
The false diagnosis is not a promotion bypass or a basis to discard errors.

### Fresh retry selection and isolated continuation

The predeployment inventory arrived after its scan had been overtaken by the
reconciliation chunks. Its 1,707-success/248-error and 815-binding counts are
historical observations, not current totals. The coordinator freshly validated
the exact ten proposed retries against the 15:05:59 authoritative export,
current acquisition generations, anchored PDF bytes and real page counts.
All ten remain failed/unreusable, no matching cache exists, all retained
same-PDF raw attempts fail strict parsing, and all 47 protected IDs are excluded.
There are 27 source pages. Updated selection:
`/tmp/opencode/context-retry10-selection-20261021.json`, SHA
`b47832d83f69465dedf06af7387f3f6167d744e4d8189132a76826e2f50e6605`.
Its current snapshot records 1,721 success/11 no-transactions/234 errors,
1,052 complete bindings and 999 archived attempts. Main still uses the old prompt.

The coordinator created and byte-verified a durable backup of all 535 production
cache envelopes and five audit JSON files, alongside the full staging snapshot:
`data/backups/glm-backlog/old-recipe-cache-audit-before-retry10-20261021.tar.gz`,
SHA `5b5789567a82156b3f90f742314110740c82004ea746ed044dc0ec1cd1989b21`.
The cache namespace is not being overwritten or mass-invalidated.

An independently reviewed runner now reuses the unchanged `a068559e` candidate
and existing capture/client/parser, with ten literal retries, one worker, a
15-send aggregate ceiling and external 1,800-second timeout. The review caught
and corrected a stale process-FD provenance path: old records are read beneath
the current approved root, while their exact frozen hashes remain required.
Approved runner SHA is
`5ccb1098fa414061f7a5b69aa2c2cbd3964bb8606e4e5bc42bca540b74c92810`.

Before launch, the coordinator held the exact exclusive drain lock, confirmed
the owned unit inactive, and revalidated all acquisition/manifest/old-record/
protected-ID/source/cache-backup checks under that lock. The new notified
experiment writes only to `/tmp/opencode/context-retry10-25wo9sbb`; no main
prompt deployment is needed. It stops at the first unresolved error, preserves
partials, and cannot automatically fill reserves or rerun. Source-image bundles
for all 27 pages are prepared with per-page hashes for subsequent pixel review.
No output accuracy or completion is claimed merely from this launch.

### Lock-skip doctor diagnosis deployed, checks green

The focused doctor fix passed independent review and was cherry-picked as
main `f07e266e`. It uses the latest appended invocation and requires concrete
fresh overlapping winner evidence before accepting a lock-skipped cron attempt;
skip-only, failed, stale or ambiguous evidence does not become healthy by fiat.
The exact uncommitted OCR alias/test and latest report bytes were preserved.

Actual production-checkout verification: **63 doctor tests passed**,
`git diff --check` clean, and **doctor ALL PASS**. The engagement line now
correctly identifies the October 20 overlapping winner's completed doctor exit 0.
No cron log, scheduler configuration, frozen strategy parameter, audit threshold
or canonical record was rewritten to obtain this result. The earlier isolated-
worktree crontab-path integration failure is not being reported as a passing
worktree full suite; the 63-pass result is the subsequent main-checkout run.

### Ten-retry run stopped on transport; explicit nine-record shrink launched

The ten-retry experiment exited 1 exactly as its stop policy required. Its first
document, `8221120`, exhausted three attempts, each a **240-second read timeout**
between 15:16:27 and 15:28:35 UTC. No HTTP status, response body, finish reason or
usage was returned. A hash-bound error record and all three attempt diagnostics
were saved; the other nine documents were not attempted. This demonstrates
neither quota exhaustion nor a new extraction-field error. Full partial evidence
and `stopped-result.json` were backed up in
`data/backups/glm-backlog/context-retry10-20261021-stopped.tar.gz`.

Rather than rerun the held document or fill reserves, the coordinator explicitly
reviewed a shrink to the original nine unattempted IDs. Independent delta review
confirmed all per-document identities/hashes unchanged and approved that finite
subset conditionally. Selection SHA:
`9bee9b6414d93793b2f5b8bf2cc0240e9954d4b62703be03bfa7e1a4efbd4f16`;
runner SHA:
`fed07f8a834523685e942c0c61d7680f3844104e5c024d50a57b10b29d527473`.

The new notified run rechecked all acquisition/artifact/generation/old-record/
protected/source/cache conditions under the exact exclusive drain lock, retains
one worker, 15 total HTTP attempts and a 30-minute hard deadline, and stops at
its first unresolved error. Its log is
`/tmp/opencode/context-retry9-approved-20261021.log`. It does not retry `8221120`,
admit new IDs, deploy main's prompt, overwrite old records/caches or promote
anything. Outcome is pending; no completion or accuracy claim is inferred.

### Second stopped retry; bounded service-versus-input diagnosis

The nine-record run also exited 1 at its first document, `8220834`: three
**300-second read timeouts** from 15:32:28 through 15:47:36 UTC, without HTTP
status/body, finish reason or usage. The other eight documents were not
attempted. All captured attempts, the error record and `stopped-result.json`
were saved in `/tmp/opencode/context-retry9-osywj5si/` and backed up as
`data/backups/glm-backlog/context-retry9-20261021-stopped.tar.gz`.

No further shrinking/filling/backlog calls follow automatically. Requirements,
checks, non-goals and a discriminating single-control protocol were written
before further inference in `/tmp/opencode/ocr-timeout-diagnosis-20261021.md`.
Three correlated attempts per document cost at least 720 and 900 seconds plus
overhead; they are not six independent service-health samples. Local read
timeouts do not establish model failure, quota exhaustion or server cancellation.

Unauthenticated HEAD checks at 15:49:57 returned host-root 301 and completions-
route 401 in about 0.17 seconds. The latter is expected without credentials;
it does not reject the configured credentials or prove inference availability.
Network evidence is `/tmp/opencode/ocr-timeout-connectivity-20261021.json`.
An independent read-only comparison of retained successful/failed request
settings and image geometry is running without provider calls.

After the owner's “continue”, the coordinator launched exactly **one** declared
known-fast control using original `20008705`, unchanged candidate recipe and
the existing direct client (no retry loop) under the exclusive drain lock.
It has a diagnostic-only 120-second read ceiling and 180-second external wall
ceiling; no production timeout or source parameter changes. Full capture and
source/recipe/authorization hashes are saved to a fresh isolated root. A
completion can establish contemporaneous control responsiveness, while another
timeout leaves service-versus-input causality UNKNOWN. No result automatically
authorizes bulk retries, longer timeouts, prompt deployment or promotion.

### Control succeeded; unsupported source amounts identified before more calls

The one declared known-fast control returned **HTTP 200 / finish stop in
78.023 seconds**, with 12 schema-valid rows and one HTTP attempt. It used 4,105
completion tokens including 3,708 reasoning tokens. Its complete captured bundle
is `/tmp/opencode/context-health-control-5yd44hvt/`, backed up as
`data/backups/glm-backlog/context-health-control-20261021-complete.tar.gz`.
This establishes that the same subscription inference path can complete for
that input now; it does not prove universal service health or explain the
scanned filings' stalls. No promotion or production timeout change occurred.

Independent retained-request comparison reproduced **all 21 image hashes**
and seven PDF bindings. The same recipe, model, endpoint, temperature and token
ceiling were used. Failed images were smaller than multiple successful images;
the largest successful document had six pages. Accidental PDF/image/prompt
substitution and simple payload-size explanations are not supported. One failed
document is sideways, the other includes an upright checkbox page: orientation
alone does not explain both. Missing captured status also cannot establish that
no headers/body bytes ever reached the client; capture runs after non-streaming
`requests.post` returns. Quota, failed token use and server cancellation remain
UNKNOWN. Reports: `/tmp/opencode/ocr-timeout-request-comparison-20261021.{md,json}`.

The coordinator actually viewed all six pages of three unattempted amount-error
sources. `20015068` prints $43.47/$28.98/$57.88; `20016523` includes
$348.94/$570.26/$342.19; `20016764` includes
$120.89/$80.59/$282.10/$644.37/$418.83. These are literally printed in the
Amount column, not missing model range letters. Some other rows have normal
$1,001–$15,000 bands. The current A–J contract cannot represent the sub-$1,001
amounts, so strict rejection is correct; no amounts are recast as unit prices,
invented into A, or silently dropped. No more same-recipe calls are warranted
merely to expect absent source ranges to appear.

Five explicit operator holds (two repeated transport stalls, three source-
amount incompatibilities) are bound in `/tmp/opencode/ocr-known-holds-20261021.json`.
They remain canonically unresolved; no live queue terminal, audit verdict or
record was rewritten. Diagnostic reports/holds were backed up in
`data/backups/glm-backlog/ocr-timeout-diagnosis-20261021.tar.gz`.

Next selection is limited to two genuinely new, short digital filings from the
fresh authoritative manifest, excluding protected/reusable/prior-retry inputs.
Their source pixels and required ranges are being checked before any calls.
Any subsequent two-process experiment needs separate explicit limits and shared-
lock ownership; neither control success nor five selected pilot passes certify
the backlog or open the canonical promotion gate.

### Two genuinely fresh one-page sources selected; scan preference explicit

Source selection inspected at most 20 authoritative candidate PDFs. It found
no qualifying digital table in that bounded search, so it did not mislabel scans
as digital or continue unbounded hunting. The selected fresh sources are
`2023/8219415` (David Kustoff: two purchases, YOU band A and TJX band B, dated
03/02/2023, notification 03/06/2023) and `2023/8219436` (Nicole Malliotakis:
one purchase of New York Community Bancorp, Inc. Stock, band A, transaction/
notification 03/17/2023, no printed ticker). Source examples are not transactions.

Both are sideways handwritten scans, one page each, with only two and one real
rows. The coordinator explicitly accepts the digital-preference deviation,
not a relaxation of critical-field correctness, and personally viewed both
upright source images in addition to the selector's all-page image checks.
Both have explicit supported ranges and no retained staged/archive/cache read.
All prior retry/pilot/control inputs and protected IDs remain excluded.
Selection `/tmp/opencode/context-fresh2-selection-20261021.json`, SHA
`cc20c041c369be0aa866a839f76cd0556d41d8caf488d977c66bdca151bbfacd`.

A minimal temporary worker is under delta review: each document runs in its
own local process with the unchanged capture/client/build helper. The existing
`max_retries=0` parameter is process-local, not a production edit; an offline
intentional transient check confirmed exactly one attempt and no retry sleep.
The proposed supervisor holds one exact shared lock across both processes,
permits at most two HTTP requests total, and has a separate 240-second external
deadline per child. No callbacks/fills/queued extra documents or shared mutable
capture context are introduced. Main prompt, caches, audits and canonical data
remain untouched. Selection is not an actual new GLM reading or quality result.

### Fresh two-process experiment launched, no retries

Independent delta review approved worker SHA
`2484759eb688a34a0664e7f89959b039409e45a71fee8aeb44aab546cf6dc788`
for exactly `8219415` and `8219436`. The supervisor was launched with under-lock
current acquisition/canonical-candidate, no-prior-evidence, protected-ID,
manifest/source/pixel hash checks and explicit acceptance that these are scans.
It requires `cfg.timeout is None` (one-page default read budget 180 seconds),
starts a separate interpreter per literal ID and retains the exact exclusive
drain lock until both children terminate and are waited. Each child uses a
240-second external deadline plus 10-second kill grace.

Existing process-local `max_retries=0` bounds this experiment to at most two
HTTP calls total, despite the unchanged capture helper's own 15-call ceiling.
Every capture/context is process-private; no thread-shared counter/binding was
introduced. Results and source/recipe/authorization identities remain isolated
under a fresh `context-fresh2-run-*` root. No replacement job, held-input retry,
live prompt deployment, cache overwrite or promotion follows automatically.

While the experiment ran, the coordinator executed the actual existing OCR/
staging/audit/doctor regression modules: **306 passed, 1 skipped** in 73.71s.
This is a focused five-module result, not a claim of a full repository suite.
An initial invocation referenced a nonexistent audit test filename and ran no
tests; the reported passing result is the subsequent corrected command using
`test_audit_report.py` and `test_audit_generation.py`.

### Parallel result: one reading passed, one timed out; retired-cache correction

Both children started at approximately 16:06:34 UTC. `8219415` made one request
and hit its 180-second read timeout; its captured error remains isolated and
held without rerun. `8219436` returned HTTP 200 / finish stop in **107.196s**,
one row, 4,155 completion tokens including 4,108 reasoning tokens. The supervisor
waited both, verified at most two attempts total and unchanged audit evidence.
Evidence: `/tmp/opencode/context-fresh2-run-8_eyc0lr/`.

Independent actual-image review passed the Malliotakis row: Purchase, transaction
and notification 03/17/2023, A, New York Community Bancorp, Inc. Stock, no inferred
ticker, source example excluded. The model's “NY” abbreviation is a tracked,
nonblocking cosmetic issuer-text difference. Its original full record SHA is
`7c6a13db5a7438a96e73441cd27833641d5a13b60ea43cd126b200bbcd67152b`.

That review also exposed a selection-scope defect. The selection helper checked
active staging/archive and the default cache, not retained hidden rebuild and
retired cache roots. **The global “never attempted/no retained reading” claim
is withdrawn for both inputs.** Broader filename discovery found two retained
Gemini readings per input, all with matching PDF hashes but contradicted by the
source pixels: Malliotakis Sale versus actual Purchase and NO_TRANSACTIONS versus
one row; Kustoff's other reading omits TJX, gives YOU B instead of A and notification
03/02 instead of 03/06, while its NO_TRANSACTIONS reading omits both rows.
These are not usable correct readings merely because their grammar parses.

All four original byte sequences were preserved read-only in the new bundle and
explicitly reconciled in `historical-reconciliation.json`, SHA
`d9baa2220092c7044b0eabd73d0ca25af85516055994d210feb99aca52de40db`.
Neither old cache was overwritten, relabelled or rebound. The operator explicitly
revised the publication condition to no active/current compatible terminal or
reusable correct reading, with known wrong historical attempts preserved. An
independent reviewer verified that reconciliation and approved **one** new
unaudited staging publication, not canonical promotion or more provider calls.

Under the exact shared lock, the coordinator rechecked current artifact/
generation, protected IDs, active-target absence, historical-cache hashes and
audit evidence, then atomically published the **unchanged new GLM record** to
`data/staging/glm-backlog/2023/8219436.json`. Exact bytes and queue compatibility
verified; original audit files remained unchanged during publication. A separate
subsequent fresh content-bound LLM-assisted pass was recorded after publication,
with historical audit entries and frozen lots preserved. Stable record hash:
`58b51017ba5bf0022adcabdf7d3f88b74487d313a69933ea8eb3ec1b8226aee6`.
It is silver-standard, not human verification or a population-budget certificate.

Current doctor reports **1,722 success, 11 no-transactions, 234 errors**
(1,967 live records), 37 pass / 6 critical-fail / 3 cosmetic-fail / 1,921 unaudited,
and **ALL PASS**. Main's prompt remains old; exactly the new live reading carries
the candidate prompt, explicitly preserved rather than mislabelled as old recipe.
Canonical promotion remains closed. Current staging and the complete parallel/
source/historical-reconciliation/audit bundle were backed up in the timestamped
`glm-backlog-20261021-after-fresh1-*.tar.gz` and
`context-fresh2-20261021-complete.tar.gz` archives.

A read-only named-cache evidence index is now including hidden rebuild/retired
roots as well as active records, to prevent another default-root-only freshness
claim. Its coverage and conflicts will be reported explicitly; grammar validity
will not be called pixel correctness or permission to reuse a wrong reading.

### Retained-evidence index and verified byte backup completed

The named-cache/declaration scan completed without errors or observed source
metadata changes: **8,970 records** across 19 roots, comprising 535 default
Gemini caches, 999 hidden rebuild caches, 4,470 retired-archive caches, 1,967
live GLM records and 999 archived GLM records. The latter includes all 939
reconciliation archives, not only numbered `attempt-*` files. It covers 4,896
document IDs and explicitly flags 492 potentially conflicting PDF groups.
3,055 unknown raw schemas remain unknown, not empty/correct/reusable.

Index directory: `/tmp/opencode/retained-ocr-evidence-index-20261021/`.
`all-entries.jsonl` SHA:
`5637d115753fcc48fe1593dbc4759fa7ff1237a29c97f771d401fce87296a100`.
Protected union is now **48** after the fresh 8219436 audit. Both known historical
readings and the new reading appear for that document; 8219415's isolated timeout
is explicitly outside this data-only index. Symlinks, 16 compressed containers,
arbitrary unmarked roots and outside-data attempts are not expanded. Hashes and
uncoerced row differences are not source correctness certification.

The coordinator subsequently preserved the actual bytes of **all 8,970 indexed
source records plus seven index files**, rechecking each original file hash and
then every archive member hash. Source JSON total: 24,771,141 bytes. Verified
local archive:
`data/backups/glm-backlog/retained-ocr-evidence-8970-20261021.tar.gz`, SHA
`f0d7ca663cb4a0152ad98cddd62bb67dc72aa7be9e55ec424fb058d75824f3a8`.
Verification receipt: `/tmp/opencode/retained-ocr-evidence-backup-20261021.json`.
This is a byte backup on local storage, not off-machine/WORM protection.

After the owner's “continue”, two additional offline lanes were started with
no model calls: the existing selector is refreshing its actual artifact/
generation refusal counts, and five genuinely versioned recovery proposals are
being prepared from the already pixel-checked new-prompt pilot responses. Those
five old live records are protected sampled evidence; this preparation alone
does **not** permit overwriting them, rebinding old verdicts or using the old lot
to certify new attempts. Any later publication needs separate targeted review,
exact old-byte archival copies, fresh audit bindings and an unchanged closed
population gate. No such five-record publication has yet occurred.

### Retained-output triage, one verified replay, and five fresh review passes

Actual selector refresh at 16:26:45 found **2,608 eligible / 14 explicitly blocked**
entries in the unchanged 3,676-item export. Eligibility is not authority to make
provider calls. Derived retained-evidence triage conserves all 2,608: **2,283**
strict-parser-valid/all-known-same-PDF-agree candidates for pixel review only;
**140** potential conflicts; **79** mixed valid/invalid/unknown; **84** with no
indexed same-PDF reading; **6** explicit protected/held; **16** only missing/
invalid/unknown raw. Many “agreement” cases have just one reading, not independent
corroboration; 150 say NO_TRANSACTIONS without source verification. Unknown cache
locations/containers remain outside coverage. Six long agreement documents include
K, which the existing parser accepts; triage is not an A–J correctness ruling.
Reports: `/tmp/opencode/ocr-queue-retained-triage-20261021/`.

The coordinator and an independent reviewer actually inspected the complete
single page of the first retained candidate, `20008948`: Hon. Pete Sessions,
Bank of America Corporation (BAC), Purchase, transaction/notification 02/08/2018,
band B, exactly one row and no omitted economic note. The retained cache matches
these pixels. A deterministic retained-output replay was prepared and separately
approved for **one new unaudited staging record**, preserving the true declared
`gemini/gemini-3.1-flash-lite` model, original prompt and original parser in
provenance. The local staging parser/parse timestamp are distinguished from the
old extraction; original model-reading time/generation are unknown, not invented.
No Gemini or Z.AI inference request, paid fallback or GLM relabelling occurred.

Under the shared lock the exact source-cache/PDF/current-generation/eligibility/
absence/protected/hash checks passed, and unchanged proposal bytes were saved as
`data/staging/glm-backlog/2018/20008948.json`, full SHA
`629888e3218a8202712a602b451b3d960516223afeb1755f0f34db35d115bc5c`.
Original cache and audit histories/lots remained byte-for-byte unchanged;
selector compatibility verified. **No audit stamp was added**, preserving the
five protected proposals' pinned audit history. Current live staging is therefore
**1,968 = 1,723 success + 11 no-transactions + 234 errors**; there is no canonical
promotion. The complete source/replay review and updated staging are backed up
in `retained-cache-triage-20261021.tar.gz` and a timestamped
`glm-backlog-20261021-after-retained1-*.tar.gz`.

The five protected proposals received genuinely fresh independent pixel review:
**5 PASS, 16 pages, 104 rows**, tied to their NEW full/stable proposal digests.
Attestation `/tmp/opencode/five-context-new-attempt-pixel-review-20261021.json`,
SHA `f2ff6833d7a429360f2cee9e7577044d1c8c90f3ee2e926065398769b07df668`.
That review corrected an earlier narrative count: exchange document `20017632`
contains **16 partial sales plus Webster's plain sale**, not 17 partial sales.
The actual row data was correct. Old immutable pilot reports were not rewritten;
an external bound correction is saved as
`/tmp/opencode/five-context-partial-sale-count-correction-20261021.json`.

Attempt-scoped audit-tool candidate `7876380a` remains **BLOCKED and undeployed**.
Independent review reproduced that page-count/PNG-magic-only validation could
accept a replaced valid PNG or eight-byte PNG signature, then stamp a bound pass.
The author is adding source/record/ordered-image checksum seals and mutation
regressions. Neither that candidate nor any of the five protected proposal
records has been deployed/published. No stale shared-bundle check is bypassed.

### Completed five-attempt repair and four further retained replays

The preceding HOLD was subsequently resolved, not bypassed. Candidate
`87d54e60` seals frozen record bytes, current/stable identity and ordered PNG
hashes/sizes; Pillow verifies and decodes rendered images. Missing or changed
seals/pixels fail without repairing an existing bundle. Independent re-review
observed **214 tests and 23 adversarial probes** passing, including both prior
pixel-replacement vectors and an actual 10-page rendering roundtrip.

The separate five-record publisher (`90d8efab0de947df6b54eda5d08618b6c88ccdf31262004662f9ec701d7814cd`)
passed **42 isolated checks and seven independent probes**, then received
five-only approval. Its historical read-only NOTES prescribed the wrong Python
environment; the operative correction is
`/tmp/opencode/five-context-publisher-system-python-20261021.md` and execution
used **system `/usr/bin/python3 -B`**, not the virtualenv. The earlier description
“no network monkeypatch” was inaccurate: this publisher installs process-local
network-denial hooks, which block rather than initiate requests.

Under the existing exclusive drain lock and sole-writer maintenance conditions,
all five targets reached **FRESH_AUDIT_BOUND_COMPLETE**: `20005634`, `20008705`,
`20012750`, `20016765`, `20017632`. Exact old-live records were copied to their
recognized versioned archives without pruning or a live-file gap. Old shared
bundles, sample lots, previous archives and history entries remained intact.
New sealed attempt-scoped views matched the freshly reviewed proposal/pixel
bytes before any new verdict. Fresh passes were appended, not inherited.
Journal and authorization are `/tmp/opencode/five-context-publication-journal.jsonl`
and `/tmp/opencode/five-context-apply-authorization-20261021.json`.
An immediate read-only post-operation publisher run verified all five complete.
That frozen preflight is historical after the deliberate later audit-code
deployment; its producer-source pins must not be refreshed or represented as a
post-deployment check.

Four further retained-cache cases were independently checked against all four
actual source pages and published **unaudited**, with original Gemini model,
prompt and parser provenance: `20008949` (Putnam fund/share class A), `20009070`
(PG purchase), `20009231` (CVS sale), `20009323` (VOD depositary shares representing
ten ordinary shares). No new provider requests occurred. Initial publication
stopped after the first file's byte check because the shared JSON writer omitted
the approved proposal's final newline. Its fields were identical and audit null;
those intermediate bytes were preserved separately. Atomic exact-approved-byte
copies completed all four without changing fields or timestamps. Original caches,
history and lots remained unchanged. Receipt:
`/tmp/opencode/retained-four-publication-result-20261021.json`.

The reviewed audit-tool commits are now deployed as **`da186536`, `6915d3e0`**.
Main OCR prompt remains OLD; the partial-sale alias/tests and this report retain
their previous dirty/untracked status. Main-checkout verification observed
**341 passed, 1 skipped** across the six focused OCR/staging/audit/doctor modules;
doctor ALL PASS and diff check clean. This is not a full-suite claim.
Current live staging: **1,972 = 1,727 success + 11 no-transactions + 234 errors**;
doctor audit split **38 pass / 6 critical-fail / 3 cosmetic-fail / 1,925 unaudited**.
The five retained Gemini replays have not yet received production audit stamps.

The original October-20 lot still has **8/10 resolved, five critical failures,
four repaired-after-fail and two unresolved**. Sticky failures are preserved;
selected repaired cases do not establish a corpus error budget. No canonical
promotion occurred and the gate remains closed.

Complete post-operation staging byte backup (all **3,158 members** independently
hash-checked against both archive and unchanged sources):
`data/backups/glm-backlog/glm-backlog-20261021T175448Z-after-five-and-retained5.tar.gz`,
SHA `a727001b134240cce8ff3ab910bcafbd23ae5b4e22c22c8fddbbbbe02d9d2361`.
Completed-operation receipts, original intermediate bytes and deployed audit code
are preserved in `five-and-retained5-completed-evidence-20261021.tar.gz`, SHA
`ff849925accf92da646d8edc875f7e4e785799aaa1dee3abfb3339760fee20cd`.
These are local backups, not off-machine/WORM guarantees.

### Orientation experiment: fast schema-valid response, critical pixel failure

The separately predeclared single experiment for held `8219415` used a lossless
90-degree counterclockwise transpose of the exact original 150-DPI request
pixels, with pixel equivalence verified, unchanged prompt/model/endpoint/token
settings, one request, no retries and the unchanged 180-second read ceiling.
It returned HTTP 200 and two schema-valid rows in **18.208 seconds**. Actual
image review nevertheless found **three critical discrepancies**: both
notification dates were emitted as 03/02/23 instead of printed 03/06/23, and
TJX was misread as TSX. The untouched reading was rejected, not manually fixed
or published. Source/member/direction/amount matches do not waive those errors.

Evidence: `/tmp/opencode/context-upright-single-f15o2hg0/`, including original and
transformed pixels, pre-call protocol, sanitized capture and bound failure
review; backed up in `context-upright-single-20261021-rejected.tar.gz`.
The original request had timed out; this single comparison is not causal latency
proof or general rotation approval. The input remains held, production renderer
unchanged, and no further calls were automatically authorized.

A new explicitly bounded offline selection of **eight** small retained responses
is being prepared for actual source review, with no reserve refill for failures
and no model calls. Existing live/protected/held cases are excluded. Preparation
alone does not authorize publication or population-quality conclusions.

Post-deployment read-only validation separately confirmed all five exact old-live
archives, original bundles/prior archives, new seals and current source
generations. Audit history contains the **exact original 53-entry prefix plus
five fresh bound entries**. Receipt:
`/tmp/opencode/five-context-postdeployment-verification-20261021.json`.

The bounded eight-document preparation completed with **5 PASS, 1 HOLD, 2 FAIL**;
all eight actual source pages were viewed. Proposed passes: `20009337`,
`20009447`, `20009606`, `20009778`, `20009784`. `20009788` is held for missing
dividend-reinvestment context. `20009807` and `20009809` fail: printed exact
**$616.20** was invented into amount band A, with INFO ticker also omitted.
Parser-valid historical output is therefore demonstrably not safe automatic
reuse. No amounts, raw readings or missing context were manually repaired, no
reserve cases selected, and none of the eight has yet been published.

Packet: `/tmp/opencode/retained-eight-replay-proposals-20261021-v2cp9dby/`,
`result.json` SHA `6288f171c42a5c33532f8346d58e6ca934b972b636ddd08dc1154f51f37400f5`.
Independent actual-pixel re-review is underway. Complete preparation backup,
including rejected evidence and all **107 members** hash-verified:
`data/backups/glm-backlog/retained-eight-replays-20261021-prepublication.tar.gz`,
SHA `e64bf22fa0c1f89f2f5e105c76858b6cb7a32808e6689b793422ccbd0d432f74`.
No provider calls occurred in this bounded eight-document task.

### Eight-document independent review and bounded publication completed

Independent reviewer `ses_ea0643401ffeovr8L8UxoYV74B` actually viewed all eight
full-page images and confirmed **5 PASS / 1 HOLD / 2 FAIL**, including both
literal $616.20 Amount-column findings and missing reinvestment context. It
verified 106 manifest-listed file hashes, re-rendered all eight current PDFs
in memory, checked current root-bound PDF/archive/generation identities, and
approved **unaudited staging only** for the exact five PASS proposal digests.

The coordinator revalidated approved packet/output hashes, raw/cache/model/
prompt equality, all known targeted reading hashes, actual source generations,
current canonical-aware selection and protected/attempt/destination absence
under the exact existing exclusive drain lock. Exact proposal bytes were
published via existing reviewed root-FD exclusive-copy helpers and same-directory
atomic replacements with fsync. All **3,158 pre-existing staging members** and
all eight original caches stayed byte-for-byte unchanged; the three rejected
documents remain absent from live staging. No audit stamps, source edits,
reserve substitutions, provider calls or canonical mutations occurred.
Receipt: `/tmp/opencode/retained-eight-publication-result-20261021.json`.

Additional source-bound operator holds are preserved in
`/tmp/opencode/ocr-known-holds-after-eight-20261021.json`. Consult this overlay
together with earlier holds and the frozen index before future selection: the
existing selector/cron does **not** automatically consume it. No selector or cron
change/restart was made, and this completed batch does not authorize another
batch or provider call. The rejected historical readings must not be resurrected
because their original grammar was valid.

Current staging: **1,977 = 1,732 success + 11 no-transactions + 234 errors**.
Doctor **ALL PASS**, audit split **38 pass / 6 critical-fail / 3 cosmetic-fail /
1,930 unaudited**; diff check clean. The ten newly recovered retained Gemini
records are still audit null; independent source review does not silently stamp
them or certify a population budget. The prior **341 passed, 1 skipped** focused
test result stands; no code changed in this bounded eight-document task.

Final complete staging snapshot, all **3,163 members** hash-verified against
archive and unchanged sources:
`data/backups/glm-backlog/glm-backlog-20261021T180209Z-after-retained-eight-selection.tar.gz`,
SHA `e8d8286882b36bec02893d5bca9d2f246eaec14322f77e6cd8745ffcde79f4e9`.
Verification receipt:
`/tmp/opencode/retained-eight-final-backup-verification-20261021.json`.
Main remains `6915d3e0`, six commits ahead of last confirmed pushed main; original
OCR alias/tests and this report remain uncommitted/untracked. No provider or
publication worker remains active, and canonical promotion remains closed.

### Requested Gemini/Z.AI error retry: preflight and retained recovery candidate

The owner subsequently permitted trying Gemini on error files and requested
“retry with zai too”. Current environment has a Gemini API key but no Gemini
CLI executable; API billing/quota is unknown. Confirmation was requested before
relaxing the earlier no-paid-fallback restriction. No new Gemini call was made.
Z.AI Coding subscription configuration was checked without a request: model
`glm-5.3-flash`, unchanged endpoint, no timeout override, one-page 180-second /
two-page 240-second limits and 16,384-token budget. Quota balance remains unknown.

A predeclared up-to-three-file Z.AI retry allows one request per selected file,
no automatic retries or reserve refill, and excludes protected/held cases.
The first read-only selection returned **zero**: its fresh full manifest export
hit a 30-second local limit. That does not establish a provider failure or quota
exhaustion. Its three source previews were scaled exploratory images, not approved
model inputs; one provisional ID, `20006575`, was protected and never cleared.
The original stopped selection and its limitations remain intact at
`/tmp/opencode/zai-three-error-retry-selection-20261021-sol-01/selection.json`.

Further current-record inventory found **210 errors lacking generation identity
and 24 generation-bound errors**. Only two bound errors have one/two-page counts:
`8220119` and already-held `8221120`. Before any provider call, scope was fixed to
at most **one request for 8220119**, subject to actual live eligibility/source/
retained-evidence checks. Other legacy/larger/held records were not cleared or
silently rebound. This is a pre-call selection limit, not a shrink-and-rerun of
a failed provider experiment.

Targeted live canonical-aware checks passed for `8220119` with the existing
selector: exactly that candidate, no notes, current archive/generation and PDF
identity matching. The full original-orientation, unscaled 150-DPI source page
was actually inspected: Nicole Malliotakis, one US Treasury bill purchase,
01/24/24 transaction, 01/26/24 notification, band D. Printed Mega Corp is the
example row, not a second transaction. **Identical retained Gemini raw already
matches those pixels** in default and retired caches; the current GLM record is
an earlier timeout with missing raw. Thus this second selection also returned
zero provider-call candidates rather than spending another inference request.
Current targeted selection:
`/tmp/opencode/zai-one-bound-error-selection-20261021-sol-01/selection.json`, SHA
`cf3f2b1b83cc4aca9fd59601df8360fdfb907e27bd4860cd5fe69a9d50a705d7`.

A separate one-record retained-output recovery proposal was prepared in
`/tmp/opencode/8220119-retained-error-recovery-ogkjopg9/`, full proposal SHA
`cdff6d51505bb5c01e5da0beeabe8da59e8bebe544ae981cd7bc450be3ae5c43`.
Its original cache-declared prompt is **ad85aea5…87393ec**, not today's OLD
1255f8c3…9499aa or the isolated NEW prompt. The proposal explicitly preserves
that original declaration and the Gemini model/source parser; local builder
default prompt metadata is identified separately. No prompt was sent, no raw
or critical field edited, and original wire/rendering/read-time/generation
metadata remain unknown. New generation binding comes from the exact current
PDF snapshot; the old timeout record has not been changed. Independent fresh
pixel/provenance review is underway before any versioned publication decision.

Preflight, source snapshots and proposed recovery are backed up in
`data/backups/glm-backlog/zai-error-retry-preflight-and-8220119-proposal-20261021.tar.gz`,
all **23 members** verified, SHA
`39ab3ecec6c2d9af62f1cb01ddbf056fd381923628a6c3c9f830bdaccfdff044`.
**No provider calls or live/audit/canonical writes have occurred in this retry
request so far.** No alternative IDs were substituted automatically.

### 8220119 retained recovery completed; no fresh provider retry was needed

Independent reviewer `ses_e9f89c441ffeVj4V3hmkb2SvDu` actually viewed the complete
source page, independently re-rendered the current root-bound PDF, and verified
all critical fields, original cache declarations, raw equality, proposal full/
stable hashes and live canonical-aware eligibility. It specifically confirmed
that the different source/current prompt hashes are not a defect in this
explicitly documented local semantic replay: no new extraction prompt was sent.

The separately approved unaudited publication completed under the held existing
drain lock with inactive owner unit. The exact old GLM timeout bytes were copied
to `data/staging/glm-backlog/2024/attempts/8220119.attempt-0.json`, full SHA
`bbf715de423e6a5b00e434b67be66e895fb7ea35616327acfd50d60505c4a244`.
The exact reviewed new proposal replaced only the live record, with audit null;
no pruning, live gap or old-cache edits occurred. The other **3,162 existing
staging members**, including history and lots, stayed byte-for-byte unchanged.
Current source/generation, selector compatibility and both cache copies were
rechecked. Receipt:
`/tmp/opencode/8220119-retained-error-recovery-ogkjopg9/publication-result.json`.

Current live counts: **1,977 = 1,733 success + 11 no-transactions + 233 errors**.
Doctor **ALL PASS**; audit split remains **38 / 6 / 3 / 1,930** because this was
an unaudited timeout-to-success replacement, not a new audit or extra live ID.
Diff check clean. No repository code changed; no new test-suite run is claimed.
Both retry selection reports genuinely selected zero new provider calls; this
recovery must not be reported as a successful fresh Z.AI or Gemini API retry.
Gemini billing confirmation remains outstanding, no substitute IDs were called,
and canonical promotion remains closed.

Final complete local staging backup: **3,164 members**, each verified against
archive and unchanged source bytes,
`data/backups/glm-backlog/glm-backlog-20261021-after-8220119.tar.gz`, SHA
`edbaade26533b6dcbfa33598c291d506857e4191cf367208cb869cc4b17442a5`.
Receipt: `/tmp/opencode/8220119-final-backup-verification-20261021.json`.
These remain local backups, not off-machine or WORM guarantees.
