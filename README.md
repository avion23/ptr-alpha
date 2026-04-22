# Congressional Insider Trading Analysis

Analyze congressional financial disclosures to identify trading patterns and performance signals.

## Goal

Track which congressional members consistently outperform the market and create following strategies based on their disclosed trades.

## Approach

1. **Data Collection**: Downloads and parses official House PTR disclosures from government sources
2. **Signal Generation**: Calculates trading signal potential across multiple time horizons (30, 60, 90, 180 days)
3. **Performance Analysis**: Ranks members by hit rate, average returns, and risk-adjusted metrics
4. **Strategy Development**: Identifies optimal timing and thresholds for following disclosed trades

## Architecture

```
src/analyzer/
├── datasources.py      # Data acquisition (House PTRs, yfinance prices)
├── parsing.py          # PDF extraction and ticker identification
├── analysis.py         # Signal calculation and member ranking
├── pipeline.py         # End-to-end processing orchestration
├── database.py         # DuckDB database layer
├── models.py           # Data models and enums
└── cli.py              # Command-line interface

data/
├── congress.duckdb     # SQLite-backed DuckDB database
├── {year}/
│   └── pdfs/           # Raw disclosure PDFs
└── *.csv               # Analysis output files
```

### Core Pipeline

1. **Fetch** → Downloads House metadata and PDFs from official sources
2. **Parse** → Extracts transactions using PDF parsing and ticker extraction
3. **Analyze** → Merges with stock prices to calculate signal potential
4. **Rank** → Generates member performance metrics and top signals

### CLI Usage

```bash
insider-trading fetch --year 2024        # Download PDFs
insider-trading parse --year 2024        # Extract transactions
insider-trading show-signals --top-n 20  # View best signals
insider-trading rank-members --source house  # Member performance
```