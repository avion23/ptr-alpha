# PTR Alpha

Analyze congressional Periodic Transaction Report (PTR) disclosures to identify trading patterns and performance signals.

## Prerequisites

Python 3.11+ and system libraries for PDF parsing:

```bash
# macOS
brew install tesseract poppler ghostscript

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils ghostscript
```

## Install

```bash
pip install .

# With dev dependencies
pip install ".[dev]"
```

## Approach

1. **Data Collection**: Downloads official House PTR PDFs and can ingest Capitol Trades API records as a backup source
2. **Parsing**: Extracts transaction rows through a deterministic PDF parser cascade, with optional Gemini OCR for zero-row PDFs
3. **Signal Generation**: Calculates trading signal potential across configurable time horizons using exponential decay weighting
4. **Performance Analysis**: Ranks members by hit rate, SPY alpha, Bayesian win probability, and Sharpe ratio
5. **Ticker Scoring**: Identifies tickers with multiple congressional buyers weighted by historical member performance, position size, and ownership type

## Architecture

```
src/analyzer/
├── backtest/                  # Recommendations, evaluation, prices, filters, curves, summaries, OU parameters
├── member_ranking/            # Bayesian scoring, decay weighting, ranking, factors, buyer scoring, sales, lookups
├── parsing/                   # pdfplumber/pdftotext/docling/OCR parsers plus row, cell, column, and metadata helpers
├── portfolio/                 # Kelly sizing, portfolio simulation, and portfolio metrics
├── signals/                   # Core signal generation, assembly, filters, prices, constants, and top-signal helpers
├── analysis.py                # Analysis output assembly for ranks, signals, members, sales, and tickers
├── capitol_trades.py          # Capitol Trades API ingestion
├── cli.py                     # Typer CLI; all formatting and display logic
├── database.py                # DuckDB facade delegating to repository modules
├── datasources.py             # Backward-compatible re-exports (parser_cascade, download, price_source)
├── download.py                # House PTR PDF download and caching
├── exceptions.py              # Exception hierarchy, StepResult, DataResult types
├── interfaces.py              # Source protocol interfaces
├── matched_control.py         # Matched-control return comparisons
├── member_skill.py            # Member skill and profitability helpers
├── member_names.py            # Member name normalization helpers
├── metadata_repository.py     # Metadata CRUD operations
├── models.py                  # Data models and enums
├── options.py                 # Option-contract parsing helpers
├── parse_run_repository.py    # Parse run tracking
├── parser_cascade.py          # Deterministic PDF parser cascade
├── pipeline.py                # Fetch, parse, analysis, and backtest orchestration (returns DataResult)
├── portfolio_sim.py           # Portfolio simulator used by the CLI
├── price_repository.py        # Price data CRUD operations
├── price_snapshot.py          # Price snapshot manifests for reproducible backtests
├── price_source.py            # Price data sourcing with yfinance and cache
├── return_process.py          # Return process statistics
├── sector_data.py             # Sector data loading and analysis
├── settings.py                # Pydantic settings for data paths and parser behavior
├── signal_features.py         # Feature engineering for signals
├── snooping.py                # Multiple-comparison corrections and HAC statistics
├── ticker_resolver.py         # Ticker cleaning and symbol resolution
├── transaction_repository.py  # Transaction CRUD operations
└── validation.py              # Honest time-split calibration and evaluation
```

```
scripts/
├── backfill_tickers.py      # Backfill missing ticker symbols
├── cleanup_tickers.py       # Clean and normalize ticker symbols
├── download_missing_pdfs.py # Download missing House PTR PDFs
├── fetch_capitol_trades.py  # Fetch congressional trades from Capitol Trades API
├── gemini_ocr_common.py     # Shared Gemini OCR cache and validation helpers
├── ocr_parallel.py          # Parallel Gemini OCR runner
├── ocr_zero_rows.py         # Gemini OCR for PDFs with no parsed rows
├── purge_phantom_rows.py    # Remove historical duplicate transaction rows
├── reparse_all.py           # Reparse cached PDFs
└── run_kelly_backtest.py    # Kelly-sizing backtest helper

sweep.py                     # Parameter sweep using analyzer.validation
```

Modules follow a layered design: `cli.py` handles presentation and formatting, `pipeline.py` contains pure computation returning `DataResult`, repository modules (`transaction_repository`, `price_repository`, `metadata_repository`, `parse_run_repository`) encapsulate database access behind the `database.py` facade, and `exceptions.py` defines the `StepResult` and `DataResult[T]` types used throughout for error propagation.

## CLI

| Command | Description |
| --- | --- |
| `ptr-alpha fetch --year 2026` | Download House Clerk PTR PDFs for a year. |
| `ptr-alpha parse --year 2026` | Parse cached PDFs into DuckDB. |
| `ptr-alpha parse --year 2026 --gemini-ocr` | After deterministic parsing, run Gemini OCR for zero-row PDFs; results are validated and cached. |
| `ptr-alpha analyze --year 2026 --mode ranks` | Rank members by trading performance. |
| `ptr-alpha analyze --year 2026 --mode signals` | Show top individual trade signals. |
| `ptr-alpha analyze --year 2026 --mode sales` | Rank members by sale/loss-avoidance performance. |
| `ptr-alpha analyze --year 2026 --mode tickers` | Score recent multi-buyer ticker setups. |
| `ptr-alpha analyze --year 2026 --mode member --member "Nancy Pelosi"` | Show signals for one member. |
| `ptr-alpha analyze --year 2026 --ticker NVDA` | Deep-dive one ticker. |
| `ptr-alpha backtest --start 2024-01-01 --end 2024-12-31` | Run a rolling, no-lookahead recommendation backtest. |
| `ptr-alpha portfolio --start 2024-01-01 --end 2024-12-31` | Simulate portfolio-level execution with overlapping positions and constraints. |
| `ptr-alpha snapshot` | Write a reproducible price snapshot manifest. |
| `ptr-alpha refresh --year 2026` | Fetch House PDFs, parse cached PDFs, fetch Capitol Trades, and optionally run Gemini OCR. |
| `ptr-alpha refresh --year 2026 --gemini-ocr` | Include Gemini OCR in the full refresh pipeline. |
| `ptr-alpha fetch-capitol --all` | Fetch recent Capitol Trades API records. |
| `ptr-alpha validate --train-start 2022-01-01 --train-end 2023-12-31 --test-start 2024-01-01 --test-end 2025-06-30` | Select parameters on the train window, then evaluate once on the test window. |
| `ptr-alpha validate --full-grid` | Run the larger validation parameter grid. |

Analysis modes accept `--horizons`, `--threshold`, `--top-n`, and `--output csv` where supported. Ticker scoring also accepts `--days-back` and `--min-buyers`.

## Data & caveats

- The House Clerk publishes only a filing index; the PTR PDFs must be downloaded and parsed.
- Parsing uses a deterministic parser cascade with an optional Gemini OCR fallback. Gemini OCR output is validated and cached under `data/gemini_cache/`.
- Transactions carry a `source` provenance column (`house_pdf`, `capitol_trades`, or `gemini_ocr`); old pre-migration rows may have `NULL` source.
- `ptr-alpha validate` does train-only parameter selection with Benjamini-Hochberg snooping correction and Newey-West statistics. As of 2026-07, no configuration shows statistically significant alpha.

## Tests

```bash
PYTHONPATH=$PWD/src python3 -m pytest -q
```
