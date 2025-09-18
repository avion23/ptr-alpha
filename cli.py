#!/usr/bin/env python3

import argparse
import sys
import pandas as pd
import logging
from datetime import timedelta
from data_acquisition import fetch_quiver_data, fetch_and_cache_pdfs, parse_cached_pdfs, load_cached_data, fetch_prices, load_data
from signal_evaluation import calculate_signal_potential, rank_members, get_top_signals, get_member_signals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_fetch(year):
    fetch_and_cache_pdfs(year)
    return 0

def run_parse(year):
    parse_cached_pdfs(year)
    return 0

def run_analysis(source, horizons, threshold, output, member_filter, top_n, show_signals, year=2024):
    trades = load_data(source, year)
    if trades.empty:
        print("No data found", file=sys.stderr)
        return 1

    start_date = trades['disclosure_date'].min() - timedelta(days=30)
    end_date = trades['disclosure_date'].max() + timedelta(days=max(horizons) + 10)

    prices = fetch_prices(trades['ticker'].unique(), start_date, end_date)
    if prices is None or prices.empty:
        print("No price data", file=sys.stderr)
        return 1

    signals = calculate_signal_potential(trades, prices, horizons=horizons)
    if signals.empty:
        print("No signals generated", file=sys.stderr)
        return 1

    if member_filter:
        table = get_member_signals(signals, member_filter, horizon=horizons[0], top_n=top_n)
        filename = f"{member_filter.replace(' ', '_').lower()}_signals.csv"
    elif show_signals:
        table = get_top_signals(signals, horizon=horizons[0], top_n=top_n)
        filename = "top_signals.csv"
    else:
        table = rank_members(signals, horizon=horizons[0], threshold=threshold)
        filename = "member_rankings.csv"

    if output == 'csv':
        table.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    else:
        print(table.to_string(index=False))

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description="Congressional insider trading analyzer")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    fetch_parser = subparsers.add_parser('fetch', help='Download PDFs for a year')
    fetch_parser.add_argument('--year', type=int, default=2024, help='Year to fetch (default: 2024)')

    parse_parser = subparsers.add_parser('parse', help='Parse cached PDFs to CSV')
    parse_parser.add_argument('--year', type=int, default=2024, help='Year to parse (default: 2024)')

    analyze_parser = subparsers.add_parser('analyze', help='Run analysis on cached data')
    analyze_parser.add_argument('--source', choices=['quiver', 'house'], default='quiver',
                               help='Data source (default: quiver)')
    analyze_parser.add_argument('--horizons', nargs='+', type=int, default=[90],
                               help='Time horizons in days (default: 90)')
    analyze_parser.add_argument('--threshold', type=float, default=5.0,
                               help='Hit rate threshold percentage (default: 5.0)')
    analyze_parser.add_argument('--output', choices=['console', 'csv'], default='console',
                               help='Output format (default: console)')
    analyze_parser.add_argument('--member', help='Filter for specific member name')
    analyze_parser.add_argument('--top-n', type=int, default=10,
                               help='Number of top results (default: 10)')
    analyze_parser.add_argument('--signals', action='store_true',
                               help='Show top signals instead of member rankings')
    analyze_parser.add_argument('--year', type=int, default=2024,
                               help='Year to analyze (default: 2024)')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == 'fetch':
        return run_fetch(args.year)
    elif args.command == 'parse':
        return run_parse(args.year)
    elif args.command == 'analyze':
        return run_analysis(
            source=args.source,
            horizons=args.horizons,
            threshold=args.threshold,
            output=args.output,
            member_filter=args.member,
            top_n=args.top_n,
            show_signals=args.signals,
            year=args.year
        )
    else:
        print("Use 'fetch', 'parse', or 'analyze' commands", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())