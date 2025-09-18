# Congressional Insider Trading Analysis

Track and analyze congressional financial disclosures to identify potential insider trading patterns and successful trading strategies.

## Overview

Congressional members must disclose stock trades within 45 days via Periodic Transaction Reports (PTRs). This project analyzes these disclosures to:

1. Identify statistically significant trading performance vs market
2. Track which members consistently outperform
3. Create following strategies based on disclosed trades

## Data Sources

### Congressional Trading Data

**Free Options (Recommended):**
- **This Tool**: Built-in House PTR scraping from official government sources (2020-2025)
- **Quiver Quantitative**: Python package `pip install quiverquant` ($10/month, covers both chambers since 2016)
- **Financial Modeling Prep**: Free tier available, comprehensive APIs for House/Senate

**Government Sources (Implemented):**
- **House PTRs**: This tool automatically downloads and parses official disclosure PDFs
  - Metadata: `https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{YEAR}FD.ZIP`
  - PDFs: `https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{YEAR}/{DOC_ID}.pdf`
- **Senate PTRs**: `https://efdsearch.senate.gov/search/home/` (PDFs requiring OCR, not implemented)

**Third-party Tracking Sites:**
- **Capitol Trades**: https://www.capitoltrades.com/ (free website)
- **House Stock Watcher**: https://housestockwatcher.com/ (daily updates)
- **InsiderFinance**: https://www.insiderfinance.io/congress-trades

### Stock Price Data
- **Free**: Yahoo Finance (`yfinance`), Alpha Vantage, Tiingo, Finnhub
- **Paid**: CRSP/WRDS (gold standard), Sharadar/Quandl, Polygon.io
- **Recommendation**: Tiingo for prototyping, CRSP for publication-grade analysis

## Key Challenges

### Disclosure Timing
- **Execution Date**: When trade actually occurred
- **Disclosure Date**: When public learned (0-45 days later)
- **Analysis**: Run both execution-based (true profitability) and disclosure-based (public signal) studies

### Data Quality
- Dollar ranges instead of exact amounts ($15k-$50k)
- Missing execution dates
- Variable disclosure delays (average 35 days, up to 45 days legal limit)
- $200 fine for violations (minimal deterrent)

## Analysis Framework

### Performance Metrics
- **Cumulative Abnormal Returns (CAR)**: Event study around disclosure dates
- **Alpha**: Risk-adjusted excess returns vs Fama-French factors
- **Win Rate**: Percentage of profitable trades
- **Information Ratio**: Risk-adjusted outperformance vs benchmark

### Statistical Tests
- t-tests for mean CAR vs zero
- Cross-sectional regression with member fixed effects
- Bootstrap analysis for robustness
- Factor model adjustment (Mkt-RF, SMB, HML, Momentum)

## Implementation Options

### Option 1: Use Third-Party APIs (Recommended)

**Quiver Quantitative (Best Overall):**
```python
pip install quiverquant
import quiver

# Get all congress trades
df_congress = quiver.congress_trading()

# Get trades for specific stock
df_tesla = quiver.congress_trading("TSLA")

# Get trades by specific politician
df_pelosi = quiver.congress_trading("Nancy Pelosi", politician=True)
```

**Financial Modeling Prep:**
```python
import requests

# Senate trades
url = "https://financialmodelingprep.com/api/v4/senate-trading?symbol=AAPL"
response = requests.get(url)
```

### Option 2: Government Sources (Complex)

**For House XML (candidate disclosures only):**
```python
import requests, zipfile, lxml

url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
# Download, extract, parse XML with lxml
```

**For House PTRs (requires JavaScript scraping):**
- Modern website uses JavaScript/React interface
- Requires Selenium or Playwright for dynamic content
- No bulk download available

### Core Libraries
```python
# Third-party APIs (recommended)
import quiverquant  # or requests for FMP
import pandas
import numpy

# Stock data
import yfinance  # or tiingo

# Statistical analysis
import statsmodels
import linearmodels  # Panel regressions
import scipy

# Alternative: Government scraping
import selenium  # For JavaScript sites
import lxml      # For XML parsing
```

## Research Questions

1. Which members consistently outperform market?
2. Do certain committees show better performance (Banking, Tech)?
3. Is there alpha in following disclosures after they're public?
4. How does performance vary by trade size/frequency?
5. What's the optimal following strategy timing?

## Legal/Ethical Notes

- All data is public record
- Follow robots.txt guidelines
- Implement respectful rate limiting
- This is for research/educational purposes
- Not financial advice

## Next Steps

1. Implement PDF scraping and OCR pipeline
2. Build trade extraction and cleaning system
3. Create stock price alignment module
4. Develop event study framework
5. Build performance tracking dashboard
6. Backtest following strategies

## File Structure
```
├── data/
│   ├── disclosures/     # Raw PDF files
│   ├── extracted/       # Structured trade data
│   └── prices/          # Stock price data
├── src/
│   ├── scraping/        # Disclosure download
│   ├── extraction/      # OCR and parsing
│   ├── analysis/        # Statistical methods
│   └── visualization/   # Charts and dashboards
└── tests/
```

## Quick Start

### Option 1: Using This Tool (Recommended)
```bash
# Install dependencies
pip install pandas yfinance pdfplumber requests defusedxml

# Run analysis with House data (2024)
python cli.py --source house --output console

# Run analysis with specific year
python cli.py --source house --year 2024 --output console

# Show top trading signals instead of member rankings
python cli.py --source house --signals --output console

# Export results to CSV
python cli.py --source house --output csv

# Filter for specific member
python cli.py --source house --member "Richard W. Allen" --output console
```

### Option 2: Quiver Quantitative (Easiest)
```bash
pip install quiverquant pandas yfinance statsmodels

# Get API key from api.quiverquant.com ($10/month)
python -c "
import quiver
df = quiver.congress_trading()
logger.info(f'Downloaded {len(df)} congressional trades')
df.to_csv('congress_trades.csv', index=False)
"
```

### Option 3: Direct Python Usage
```python
from data_acquisition import load_data, fetch_prices
from signal_evaluation import calculate_signal_potential, rank_members

# Load congressional trading data
trades = load_data(2024)  # or any year

# Fetch stock prices
prices = fetch_prices(trades['ticker'].unique(), start_date, end_date)

# Calculate trading signals
signals = calculate_signal_potential(trades, prices)

# Rank members by performance
rankings = rank_members(signals, horizon=90, threshold=5.0)
logger.info(rankings.to_string())
```

## Features

### Current Implementation
- **House PTR Parsing**: Automatically downloads and extracts transaction data from official House disclosure PDFs
- **Ticker Extraction**: Intelligent parsing of asset names to extract stock tickers (e.g., "Albemarle Corporation (ALB)" → "ALB")
- **Signal Analysis**: Calculate trading signal potential with configurable time horizons (30, 60, 90, 180 days)
- **Member Rankings**: Rank congressional members by trading performance metrics
- **CLI Interface**: Command-line tool with multiple output formats and filtering options
- **Year Support**: Download data from any available year (2020-2025)

### Available Analysis
- **Peak Potential Analysis**: Calculate maximum profit potential from disclosed trades
- **Hit Rate Calculation**: Percentage of trades exceeding specified threshold
- **Member Performance**: Average and median returns by congressional member
- **Time Horizon Analysis**: Performance across different holding periods
- **Export Options**: Console display or CSV export

## CLI Usage

```bash
# Show help
python cli.py --help

# Basic analysis with House data
python cli.py --source house

# Specify year
python cli.py --source house --year 2024

# Show top trading signals
python cli.py --source house --signals

# Filter by member
python cli.py --source house --member "Richard W. Allen"

# Custom parameters
python cli.py --source house --horizons 30 60 90 --threshold 10.0 --top-n 15
```

## Recommendation

**For immediate use**: This tool provides free access to House trading data
- Real-time download from official government sources
- No API keys required
- Covers recent years (2020-2025)
- Built-in analysis and visualization

**For comprehensive analysis**: Combine with Quiver Quantitative API ($10/month)
- Adds Senate data and historical coverage back to 2016
- Clean, structured data format
- More extensive coverage of members and trades