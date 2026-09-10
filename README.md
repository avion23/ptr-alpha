# PTR Alpha

PTR Alpha analyzes **public congressional financial disclosures**. It ingests official House and Senate records, normalizes disclosed transactions, joins market prices for retrospective analysis, and evaluates a simple public-time trading rule without using information that was private at the decision time.

The production recommendation rule is deliberately small: **recent distinct congressional buyers of the same public equity**. Member-performance models are descriptive research tools, not deployment gates.

## What the project is for

At public time `t`, a congressional trade may already be days or weeks old:

```text
private transaction date ---- filing delay ----> public disclosure date t
                                             decision information starts here
```

PTR Alpha asks two separate questions:

1. **What is publicly actionable now?** Find recent public-equity purchases disclosed by multiple distinct members.
2. **What happened historically?** Measure executable next-session returns, SPY-relative outcomes, member-level descriptive statistics, portfolio behavior, and statistical validation.

The system must never backdate knowledge to the private transaction date.

## Project parts

| Part | Purpose |
| --- | --- |
| `src/analyzer/download.py` | House metadata/PDF acquisition and parse orchestration |
| `src/analyzer/senate_efd.py` | Official Senate eFD ingestion |
| `src/analyzer/parser_cascade.py`, `src/analyzer/parsing/` | Deterministic PDF/table/OCR extraction |
| `src/analyzer/database.py`, `*_repository.py` | DuckDB persistence, canonical views, generations, provenance, prices, parse reports |
| `src/analyzer/member_ranking/buyer_scoring.py` | Shared public-equity candidate universe and distinct-buyer scorer for live analysis and replay |
| `src/analyzer/member_ranking/` | Descriptive member statistics and historical diagnostic scorers |
| `src/analyzer/signals/` | Historical forward-return label construction and signal reports |
| `src/analyzer/backtest/` | Point-in-time recommendation replay and fixed-horizon evaluation |
| `src/analyzer/portfolio/`, `portfolio_sim.py` | Shared-cash equal-slot portfolio simulation and Kelly research helpers |
| `src/analyzer/validation.py`, `snooping.py` | Purged retrospective validation and multiple-testing controls |
| `src/analyzer/capitol_trades.py` | Capitol Trades reconciliation input; not an official canonical source |
| `member_profitability/` | Separate descriptive member-profitability research workflow |
| `optimize_profit/` | Older optimization/research workflow; not the production authority |
| `scripts/` | Audits, reparsing, OCR, staging, reconciliation, refresh, and operational tools |
| `tests/` | Unit, integration, statistical-invariant, parser, database, replay, and CLI checks |
| `docs/` | Current architecture/parsing docs plus explicitly historical audit/review evidence |

## Data model

The canonical DuckDB contains several kinds of data:

| Data | Examples |
| --- | --- |
| Filing metadata | document ID, archive year, member name, filing date/type |
| Raw-source identity | chamber, source, source record/row IDs, ingestion generation, artifact SHA-256 |
| Transactions | member, ticker, private transaction date, public disclosure/availability dates, purchase/sale, owner, disclosed amount interval/midpoint, instrument type, option fields, raw asset text |
| House generation state | metadata generations, PDF artifacts, quarantine records, parse completion state |
| Parser telemetry | parser version, attempted engines, raw extracted row count, persisted row count, errors |
| Senate source reports | parsed/paper-only/unavailable/failed reports and row accounting |
| Market data | daily adjusted close by ticker/date |
| Derived research data | forward returns, SPY alpha, member rankings, backtests, portfolio results, validation artifacts |
| OCR cache | optional validated Gemini OCR responses for unresolved House PDFs |

`canonical_transactions` exposes the active complete House generation plus canonical non-House sources. Normal reads exclude rows whose transaction date is after their disclosure date because those are usually OCR/date-order errors.

Capitol Trades is useful for **reconciliation**, but `fetch-capitol` does not write canonical transactions. Official House and Senate sources remain authoritative and are stored together in the canonical DuckDB; `source` and `chamber` keep their refresh boundaries separate.

## How a stock is evaluated now

The live scorer provenance is `identity_free_distinct_buyer_count_v2`.

For an `as_of` date:

1. Read disclosures in `[as_of - days_back, as_of]`.
2. Keep purchases only.
3. Keep normalized stocks; reject explicit options/funds/bonds/private assets and quarantined ticker artifacts.
4. Trust ordinary ticker identity only when it comes from canonical official-source provenance or an explicit resolver mapping. Reused symbols are date-gated so a filing cannot borrow another security's historical prices; for listing gates, the public disclosure date controls eligibility.
5. Canonicalize member names and count **distinct buyers** per ticker.
6. Require at least `min_buyers`.
7. Set

```text
signal_score = number_of_distinct_recent_buyers
```

8. Sort by score descending, then ticker ascending for deterministic ties.

There is no hidden member-skill multiplier, trade-size multiplier, owner multiplier, crash-hazard model, price-history requirement, or second exponential-recency coefficient in the production score.

### Delayed filings

A delayed filing is still actionable when it becomes public. The score uses **disclosure time**, not the private trade date. A lucrative old trade disclosed today therefore enters today's candidate window; filing delay does not separately penalize or reject it.

The hard limitation is freshness of the local disclosure database. If the most recent stored disclosure is old, the program warns that the data must be refreshed.

## How a member is evaluated

Member ranking is **descriptive research**, not the live trading rule.

For completed purchase episodes at a chosen horizon, PTR Alpha reports:

- number of purchase episodes;
- endpoint return and SPY-relative endpoint alpha;
- hit rates;
- a Beta-binomial descriptive positive-alpha probability;
- an empirical normal-normal partially pooled alpha estimate;
- posterior standard deviation and shrinkage diagnostics;
- additional display diagnostics such as trade-count/size conviction.

The ranking is sorted by `shrunk_alpha`.

This member model is not causal. Committee membership, information access, sector exposure, market regime, ticker concentration, and disclosure-selection effects are not controlled sufficiently to interpret the member effect as skill.

## Return math and first-principles boundary

Historical executable labels use public disclosure time:

```text
entry session = first expected NYSE session after disclosure date
intended end  = entry session + horizon calendar days
exit session  = expected NYSE session on or before intended end

stock_return = P_exit / P_entry - 1
spy_return   = SPY_exit / SPY_entry - 1
spy_alpha    = stock_return - spy_return
```

A label is complete only when the exact expected entry/exit sessions exist for both the stock and SPY. Immature or missing endpoints remain missing; they are not converted to zero or shortened horizons.

The fixed-horizon CLI backtest now evaluates exactly `--horizon`. It does not replace that horizon with an Ornstein-Uhlenbeck-derived holding period.

### What is mathematically solid

- availability begins at public disclosure time;
- next-session entry avoids same-day hindsight execution;
- stock and SPY use the same executable endpoints;
- incomplete forward windows remain censored/missing;
- consensus scoring is identity-invariant and uses no future outcomes;
- backtest and live candidate selection share the same equity/buyer rule.

### What remains heuristic research

- `DECAY_LAMBDA` for the historical decay-weighted return diagnostic;
- the empirical member hierarchy and its distributional assumptions;
- the 14-day member episode collapse used by member ranking;
- portfolio rebalance cadence, holding period, maximum position count, and any explicitly declared slippage assumption;
- validation family choices such as horizon, buyer threshold, and top-N.

These must not be described as laws of the data-generating process. The current production candidate score avoids them.

## Parsing: cheap vs expensive

### Cheap parsing

Text-layer extraction is relatively cheap and deterministic:

1. pdfplumber
2. Camelot lattice
3. Camelot stream
4. `pdftotext`

All text engines are compared. `pdftotext` and Docling aggregate transactions from **all** returned tables instead of stopping at the first non-empty table. Parser debug logs include per-engine row count, quality, and elapsed time.

When pdfplumber and pdftotext agree by multiset containment, the more complete trusted result can return without OCR.

### Expensive parsing

OCR is expensive because it rasterizes pages and/or runs large external processes:

5. Docling OCR, unless explicitly disabled for a bounded bulk pass
6. Tesseract OCR
7. Optional Gemini OCR recovery for unresolved documents

When text engines disagree, deterministic parsing requires complete OCR corroboration or fails the document closed. Per-document watchdogs and subprocess timeouts prevent one malformed PDF from stalling a year-long run.

See [`docs/house-data-parsing.md`](docs/house-data-parsing.md) for the exact current flow and [`docs/house-ingestion-error-catalog.md`](docs/house-ingestion-error-catalog.md) for residual risks.

## CLI

```bash
pip install .
# development
pip install ".[dev]"
```

Common commands:

| Command | Purpose |
| --- | --- |
| `ptr-alpha fetch --year 2026` | Acquire/reconcile official House archive PDFs |
| `ptr-alpha parse --year 2026` | Parse cached House PDFs into a generation |
| `ptr-alpha parse --year 2026 --gemini-ocr` | Add optional Gemini recovery for unresolved PDFs |
| `ptr-alpha refresh --year 2026` | Official House fetch + parse refresh |
| `ptr-alpha fetch-senate-efd ...` | Fetch official Senate eFD records |
| `ptr-alpha fetch-capitol --all --output capitol.json --generation run-id` | Write Capitol Trades reconciliation artifact only |
| `ptr-alpha analyze --year 2026 --mode tickers` | Current multi-buyer candidates |
| `ptr-alpha analyze --year 2026 --ticker SPCX` | One ticker using the same `--days-back`/`--min-buyers` live rule |
| `ptr-alpha analyze --year 2025 --mode ranks` | Descriptive member rankings |
| `ptr-alpha analyze --year 2025 --mode signals` | Historical top purchase outcomes |
| `ptr-alpha analyze --year 2025 --mode sales` | Historical sale/loss-avoidance ranking |
| `ptr-alpha backtest --start 2024-01-01 --end 2025-12-31` | Fixed-horizon public-time replay |
| `ptr-alpha portfolio --start 2024-01-01 --end 2025-12-31` | Shared-cash equal-slot portfolio simulation; optional slippage is declared in basis points |
| `ptr-alpha snapshot` | Explicitly write a reproducible price snapshot |
| `ptr-alpha validate ...` | Purged retrospective research validation |

`analyze`, `backtest`, `portfolio`, and `snapshot` open the canonical database read-only. Fetch/parse/refresh are the mutating paths.

## Validation status

A positive live score means only that multiple distinct members disclosed recent purchases of the same equity. It is not statistical proof of abnormal future return.

The repository's retrospective validation machinery evaluates only the production consensus rule. It uses scheduled no-trade support, exact holding-period purging, SPY-relative net outcomes, and family-wise controls. Historical member-skill modes remain separate descriptive research. The older `optimize_profit/` workflow remains a separate research artifact and is not a production authority.

No existing retrospective result should be relabeled as fresh out-of-sample evidence after changing the scorer or implementation. A scorer change requires a new predeclared evaluation.

## Documentation

- [`docs/trading-prediction-architecture.md`](docs/trading-prediction-architecture.md): first-principles prediction/evaluation architecture and remaining research roadmap.
- [`docs/house-data-parsing.md`](docs/house-data-parsing.md): current House parser and persistence behavior.
- [`docs/house-ingestion-error-catalog.md`](docs/house-ingestion-error-catalog.md): current ingestion risk register.
- [`docs/HOUSE_PARSER_AUDIT.md`](docs/HOUSE_PARSER_AUDIT.md): dated historical corpus audit; evidence, not current architecture.
- [`docs/adr/001-refactoring.md`](docs/adr/001-refactoring.md): historical refactoring decision record.
- `docs/reviews/`: archived adversarial-review evidence. It is not current product documentation; current behavior is defined by code, tests, README, and the three current docs above.

## Tests

```bash
PYTHONPATH=$PWD/src python3 -m pytest -q
python3 -m ruff check .
```
