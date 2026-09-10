# Trading prediction architecture

## Status

This document describes the current PTR Alpha decision and evaluation architecture. Historical audit transcripts under `docs/reviews/` and old ADRs are evidence about past states; they are not current product authority.

PTR Alpha has one production decision rule: count distinct canonical congressional buyers of the same eligible public equity inside a public-disclosure lookback window. Member statistics are descriptive research and do not authorize recommendations.

## 1. Information boundary

A congressional transaction becomes usable only when the disclosure is public.

```text
private transaction date ---- filing delay ----> public disclosure date
                                             decision information starts here
```

The private transaction date describes when the member traded. It cannot make a later filing visible earlier. Delayed filings therefore remain actionable when they first become public.

Every live or historical decision must satisfy:

```text
disclosure_date <= decision_date
```

Rows with impossible chronology, such as a transaction date after its disclosure date, are excluded from normal decision paths.

## 2. Execution graph

```text
CLI
 |
 +-- fetch / refresh House
 |     -> House metadata + PDFs
 |     -> parser cascade
 |     -> staged generation
 |     -> canonical activation when complete
 |
 +-- fetch Senate eFD
 |     -> official report inventory
 |     -> normalized Senate transactions
 |
 +-- analyze --mode tickers / --ticker
 |     -> canonical transactions, read-only
 |     -> public disclosure window
 |     -> eligible equity identity
 |     -> distinct canonical buyers
 |     -> consensus score
 |
 +-- analyze --mode ranks / signals / member
 |     -> canonical transactions, read-only
 |     -> market prices
 |     -> executable historical labels
 |     -> descriptive reports
 |
 +-- backtest / portfolio / validate
       -> canonical transactions, read-only
       -> same consensus candidate rule
       -> exact NYSE execution endpoints
       -> retrospective outcomes / statistics
```

The main code boundaries are:

| Area | Authority |
| --- | --- |
| House ingestion | `src/analyzer/download.py` |
| Senate ingestion | `src/analyzer/senate_efd.py` |
| Parser cascade | `src/analyzer/parser_cascade.py`, `src/analyzer/parsing/` |
| Persistence | `src/analyzer/database.py` and repository modules |
| Production equity/buyer rule | `src/analyzer/member_ranking/buyer_scoring.py` |
| Replay recommendation generation | `src/analyzer/backtest/recommend.py` |
| Historical label construction | `src/analyzer/signals/`, `src/analyzer/backtest/evaluate.py` |
| Descriptive member ranking | `src/analyzer/member_ranking/ranking.py` |
| Portfolio execution | `src/analyzer/portfolio/`, `src/analyzer/portfolio_sim.py` |
| Statistical validation | `src/analyzer/validation.py`, `src/analyzer/snooping.py` |
| CLI | `src/analyzer/cli.py` |

There is no second production optimization engine, member-scoring engine, adaptive OU holding-period engine, forecast-model framework, or exact-machine certification gate.

## 3. Canonical data model

Official House and Senate records share the canonical database model. Source and chamber provenance preserve their independent refresh boundaries.

`canonical_transactions` is the decision-facing transaction view. It excludes reconciliation-only Capitol Trades rows. Capitol Trades artifacts may be used to compare source evidence but are not an official canonical transaction source.

House ingestion uses generation state so incomplete or failed refreshes do not replace a prior complete generation. Read-only database opens can shadow stale persisted canonical-view definitions without mutating the live database.

Important transaction provenance includes source/chamber identity, source record/row identity, ingestion generation, artifact identity, raw asset evidence, transaction type, instrument type, ticker provenance, private transaction date, and public disclosure date.

## 4. Production stock evaluation

Production scorer provenance:

```text
identity_free_distinct_buyer_count_v2
```

Defaults:

```text
lookback_days = 28
min_buyers    = 3
signal_score  = distinct recent buyer count
```

For decision date `t`:

1. Read disclosures in `[t - lookback_days, t]`.
2. Keep purchases only.
3. Reject rows with invalid public chronology.
4. Reject explicit options, funds, bonds, cash-like/non-equity instruments, private assets, and quarantined ticker artifacts.
5. Use official canonical provenance or an explicit resolver mapping for ticker identity.
6. Canonicalize ticker aliases/class shares/renames to one economic security.
7. Resolve the symbol tradable on the decision date. A delayed pre-rename transaction disclosed after a rename enters the post-rename tradable symbol.
8. Canonicalize member identity and count distinct buyers.
9. Require `min_buyers`.
10. Set `signal_score` to the distinct-buyer count.
11. Sort score descending, then ticker ascending.

No member skill, price history, trade size, owner field, filing-delay penalty, recency coefficient, confidence factor, crash model, Bayesian prior, or blended quality score enters production scoring.

An empty eligible window is a successful no-signal result. It is distinct from a stale/missing-data warning.

## 5. Ticker identity

Ticker text is not sufficient by itself when symbols are aliases, renamed, reused, acquired, or parser artifacts.

Current rules include:

- class-share aliases collapse to the same economic security;
- verified parser pseudo-tickers map only where filing evidence establishes the intended public equity;
- quarantined ambiguous tokens are rejected;
- renamed symbols form one economic family, with the tradable symbol selected at public decision time;
- reused symbols are date-gated so a filing cannot borrow price history from an earlier security that used the same ticker;
- acquired securities do not silently substitute the acquirer's equity.

Price acquisition may need both sides of a rename because historical decisions can occur on either side of the effective date. Entry selection still chooses only the symbol tradable at that disclosure-time decision.

## 6. Descriptive member analysis

Member ranking is not part of recommendation generation.

At a chosen horizon, the maintained member statistic uses completed purchase outcomes only. One observation is one member/ticker/disclosure-date/horizon purchase episode. Duplicate rows for the same public event collapse without trade-size weighting.

Reported fields are based on exact executable endpoints:

- `purchase_episodes`;
- mean stock return;
- mean SPY return on identical support;
- mean SPY alpha;
- observed positive-return rate;
- observed positive-alpha rate;
- empirical normal-normal partially pooled alpha mean;
- posterior standard deviation;
- shrinkage.

The empirical hierarchy has no user-configurable prior strength, recency weight, conviction score, member-specific decay coefficient, trade-size weight, owner multiplier, or pseudo-count probability. The ranking orders `shrunk_alpha_pct` descending, then member name for deterministic ties.

These estimates remain descriptive associations. They do not control sufficiently for sector exposure, market regime, committee relationships, ticker concentration, or disclosure-selection effects and must not be described as causal skill.

Historical signal reports may still expose path-oriented diagnostics such as peak potential or the existing decay-weighted return calculation. Those fields do not enter production stock selection or member ranking.

## 7. Return and execution math

Historical executable outcomes use the same public-time boundary as live decisions.

```text
entry = first expected NYSE session after disclosure
intended_exit_target = entry + horizon calendar days
exit  = expected NYSE session on or before intended_exit_target

stock_return = P_exit / P_entry - 1
spy_return   = SPY_exit / SPY_entry - 1
alpha        = stock_return - spy_return
```

Required endpoint prices must exist on the expected sessions and be positive and finite. Missing entry, stock exit, or benchmark endpoint means the outcome is unavailable. The evaluator does not shift to a convenient later quote, use stale prior pricing as an invented endpoint, or shorten the declared horizon.

The fixed-horizon CLI backtest uses the requested horizon. Old adaptive holding-period research has been removed.

A funded basket is fail-closed: if required constituents cannot be evaluated, the replay does not reallocate their capital ex post to surviving names. Supported scheduled dates with no trade remain explicit cash observations with strategy return 0 and the same benchmark support.

## 8. Portfolio semantics

Portfolio code uses shared capital and explicit position accounting. Overlapping positions cannot reuse the same bankroll while still open.

Portfolio assumptions such as initial capital, maximum positions, rebalance cadence, holding period, and declared slippage are research/policy settings. They do not change the production candidate score.

Open positions and unresolved exits remain explicit in portfolio accounting rather than being silently marked with an invented terminal value.

## 9. Validation

`src/analyzer/validation.py` is the single production-strategy evidence engine.

Validation evaluates the actual consensus family. Result-changing family dimensions are limited to declared decision/evaluation parameters such as horizon, scheduled frequency, lookback window, buyer threshold, and top-N where supported by the experiment specification.

The validation path preserves these rules:

- public-time recommendation replay;
- next-session entry and exact fixed-horizon exit semantics;
- purge/embargo between training and evaluation support;
- one scheduled per-date strategy series with explicit cash dates;
- identical benchmark support;
- Newey-West/HAC statistics where declared;
- moving-block bootstrap with support-aware dependence handling;
- family-wise correction across the declared strategy family;
- fail-closed minimum resampling/sample-support requirements;
- identity-invariance diagnostics for the consensus scorer;
- retrospective wording for previously explored history;
- rejection of evaluation windows that enter the reserved final holdout.

Hashes and manifests may identify evidence. They do not authorize execution, make a result correct, or consume a one-shot right to evaluate. Exact-machine fingerprints, filesystem locks, consumption ledgers, and frozen database-hash gates are not correctness mechanisms.

## 10. Parsing and ingestion

House parsing starts with cheap deterministic text/table extraction and escalates only when additional evidence is needed.

Current deterministic/text engines include pdfplumber, Camelot lattice/stream, and `pdftotext`. Returned tables/pages are aggregated rather than accepting the first non-empty table. OCR paths include Docling/Tesseract and optional Gemini recovery for unresolved filings.

When text engines disagree, reconciliation is evidence-based and fails closed if disagreement cannot be resolved. Parser telemetry records attempted engines, row counts, and failure reasons.

Official row identity, raw fields, instrument evidence, ticker provenance, amendment/generation identity, and artifact identity are preserved for persistence/reconciliation. Transaction replacement and generation activation must not expose a partial new filing set as canonical success.

See `docs/house-data-parsing.md` for the current House parser flow and `docs/house-ingestion-error-catalog.md` for known ingestion risks.

## 11. Failure and logging semantics

Operational code should prefer one bounded summary over one warning/error per row or ticker. Detailed provenance belongs at debug level unless an operator can act on it.

Important user-visible states are distinct:

- success with positive candidates;
- success with no signal;
- stale/missing source warning;
- incomplete/unavailable historical outcome;
- parser/ingestion failure;
- validation with no deployable configuration.

Broad exception handling must not silently change implementations or convert partial output into success when completeness is required.

## 12. Interface scope

The current product has no graphical frontend. User-facing surfaces are Typer CLI output and generated text/CSV/JSON artifacts.

CLI/report style should stay compact and utilitarian:

- precise field names;
- deterministic ordering;
- minimal headings;
- no decorative badges or prose;
- explicit warning/error/no-signal distinctions;
- no production-sounding score names for descriptive research metrics.

Impeccable is installed and initialized for durable product context, but graphical audit/polish workflows are not applicable until a real graphical surface exists. Do not create a frontend merely to satisfy design tooling.

## 13. Data freshness limitation

A correct scorer can still be operating on stale local data. Live candidate output is current only to the latest successfully ingested official House and Senate disclosures in the local canonical database.

Live analysis warns when a chamber has no disclosure inside the candidate window. A no-signal result should not be interpreted as current if source freshness is stale or missing.

## 14. Non-authorities

The following are intentionally not current product authorities:

- `docs/reviews/` adversarial-review transcripts;
- historical ADR text describing removed modules;
- Capitol Trades reconciliation artifacts;
- deleted `optimize_profit/` and `member_profitability/` research stacks;
- deleted prediction/decision-adapter research frameworks;
- deleted adaptive OU holding-period research;
- deleted exact-revision/database-hash baseline certification tooling.

Current behavior is defined by maintained source, tests, README, this architecture document, and the current parsing/data-quality docs.
