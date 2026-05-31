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

1. **Data Collection**: Downloads and parses official House PTR disclosures from government sources
2. **Signal Generation**: Calculates trading signal potential across multiple time horizons (30, 60, 90, 180 days) using exponential decay weighting
3. **Performance Analysis**: Ranks members by hit rate, Spy alpha, Bayesian win probability, and Sharpe ratio
4. **Ticker Scoring**: Identifies tickers with multiple congressional buyers weighted by historical member performance, position size, and ownership type

## Architecture

```
src/analyzer/
├── datasources.py      # Data acquisition (House PTRs, yfinance prices)
├── parsing.py          # PDF extraction, OCR fallback, ticker identification
├── analysis.py         # Signal calculation, member ranking, ticker scoring
├── pipeline.py         # End-to-end processing orchestration
├── database.py         # DuckDB database layer (ASOF joins for entry prices)
├── models.py           # Data models and enums
├── interfaces.py       # Abstract sources (TransactionSource, PriceSource)
├── exceptions.py       # Error hierarchy
├── settings.py         # Configuration
└── cli.py              # Unified command-line interface
```

## CLI

```bash
ptr-alpha fetch --year 2026                          # Download House PTR PDFs
ptr-alpha parse --year 2026                          # Extract transactions to DuckDB
ptr-alpha analyze --year 2026 --mode ranks           # Rank members by Spy alpha
ptr-alpha analyze --year 2026 --mode signals         # Top individual trade signals
ptr-alpha analyze --year 2026 --mode sales           # Rank members by loss avoidance
ptr-alpha analyze --year 2026 --mode tickers         # Multi-buyer ticker scores
ptr-alpha analyze --year 2026 --mode member --member "Nancy Pelosi"  # Member-specific signals
ptr-alpha analyze --year 2026 --ticker NVDA          # Deep-dive single ticker
```

All analysis modes accept `--horizons`, `--threshold`, `--top-n`, and `--output csv` flags.

## Tests

```bash
pytest
```