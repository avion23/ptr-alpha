#!/usr/bin/env python3

import argparse
import sys
import logging
from analyzer.sources import Config
from analyzer.pipeline import run_fetch_pipeline, run_parse_pipeline, run_analysis_pipeline, run_ticker_analysis, run_recent_ticker_scoring
from analyzer.exceptions import DataSourceError, AnalysisError, ParsingError, ConfigurationError

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_config(args):
    return Config(
        data_dir=args.data_dir,
        cache_enabled=not args.no_cache,
        parallel_workers=args.workers
    )

def handle_command(args):
    config = create_config(args)

    if args.command == 'fetch':
        return 0 if run_fetch_pipeline(args.year, config) else 1
    elif args.command == 'parse':
        return 0 if run_parse_pipeline(args.year, config) else 1
    elif args.command == 'rank-members':
        return 0 if run_analysis_pipeline(
            args.source, args.year, args.horizons, args.threshold,
            None, args.top_n, False, args.output, config
        ) else 1
    elif args.command == 'show-signals':
        return 0 if run_analysis_pipeline(
            args.source, args.year, args.horizons, args.threshold,
            None, args.top_n, True, args.output, config
        ) else 1
    elif args.command == 'show-member-signals':
        return 0 if run_analysis_pipeline(
            args.source, args.year, args.horizons, args.threshold,
            args.member, args.top_n, False, args.output, config
        ) else 1
    elif args.command == 'analyze-ticker':
        return 0 if run_ticker_analysis(
            args.ticker, args.source, args.year, args.horizon, args.threshold, config
        ) else 1
    elif args.command == 'score-recent-tickers':
        return 0 if run_recent_ticker_scoring(
            args.source, args.year, args.horizons, args.threshold,
            args.days_back, args.min_buyers, args.top_n, config
        ) else 1
    else:
        print("Use 'fetch', 'parse', 'rank-members', 'show-signals', 'show-member-signals', 'analyze-ticker', or 'score-recent-tickers'", file=sys.stderr)
        return 1

def parse_args():
    parser = argparse.ArgumentParser(
        description="Congressional insider trading analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  %(prog)s fetch --year 2024\n"
               "  %(prog)s parse --year 2024\n"
               "  %(prog)s rank-members --source house\n"
               "  %(prog)s show-signals --top-n 20\n"
               "  %(prog)s show-member-signals --member 'Nancy Pelosi'\n"
               "  %(prog)s analyze-ticker NVDA\n"
               "  %(prog)s score-recent-tickers --days-back 28 --min-buyers 2"
    )

    parser.add_argument('--year', type=int, default=2025, help='Year to process (default: 2025)')
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    fetch_parser = subparsers.add_parser('fetch', help='Download House PDFs for a year')

    parse_parser = subparsers.add_parser('parse', help='Parse cached PDFs to CSV')

    rank_parser = subparsers.add_parser('rank-members', help='Rank congressional members by trading performance')
    rank_parser.add_argument('--source', choices=['house'], default='house', help='Data source')
    rank_parser.add_argument('--horizons', nargs='+', type=int, default=[90], help='Time horizons in days')
    rank_parser.add_argument('--threshold', type=float, default=5.0, help='Hit rate threshold percentage')
    rank_parser.add_argument('--output', choices=['console', 'csv'], default='console', help='Output format')
    rank_parser.add_argument('--top-n', type=int, default=20, help='Number of top members to show')

    signals_parser = subparsers.add_parser('show-signals', help='Show top trading signals')
    signals_parser.add_argument('--source', choices=['house'], default='house', help='Data source')
    signals_parser.add_argument('--horizons', nargs='+', type=int, default=[90], help='Time horizons in days')
    signals_parser.add_argument('--threshold', type=float, default=5.0, help='Hit rate threshold percentage')
    signals_parser.add_argument('--output', choices=['console', 'csv'], default='console', help='Output format')
    signals_parser.add_argument('--top-n', type=int, default=15, help='Number of top signals to show')

    member_parser = subparsers.add_parser('show-member-signals', help='Show signals for a specific member')
    member_parser.add_argument('--member', required=True, help='Member name to analyze')
    member_parser.add_argument('--source', choices=['house'], default='house', help='Data source')
    member_parser.add_argument('--horizons', nargs='+', type=int, default=[90], help='Time horizons in days')
    member_parser.add_argument('--threshold', type=float, default=5.0, help='Hit rate threshold percentage')
    member_parser.add_argument('--output', choices=['console', 'csv'], default='console', help='Output format')
    member_parser.add_argument('--top-n', type=int, default=10, help='Number of top signals to show')

    ticker_parser = subparsers.add_parser('analyze-ticker', help='Show all buyers of a ticker with rankings and signal score')
    ticker_parser.add_argument('ticker', help='Ticker symbol to analyze')
    ticker_parser.add_argument('--source', choices=['house'], default='house', help='Data source')
    ticker_parser.add_argument('--horizon', type=int, default=90, help='Time horizon in days')
    ticker_parser.add_argument('--threshold', type=float, default=5.0, help='Hit rate threshold percentage')

    recent_parser = subparsers.add_parser('score-recent-tickers', help='Score multi-buyer tickers from recent period')
    recent_parser.add_argument('--source', choices=['house'], default='house', help='Data source')
    recent_parser.add_argument('--horizons', nargs='+', type=int, default=[90], help='Time horizons in days')
    recent_parser.add_argument('--threshold', type=float, default=5.0, help='Hit rate threshold percentage')
    recent_parser.add_argument('--days-back', type=int, default=28, help='How many days back to analyze')
    recent_parser.add_argument('--min-buyers', type=int, default=2, help='Minimum number of buyers required')
    recent_parser.add_argument('--top-n', type=int, default=15, help='Number of top signals to show')

    return parser.parse_args()

def main():
    try:
        args = parse_args()
        setup_logging(args.verbose)

        if not args.command:
            print("Error: No command specified", file=sys.stderr)
            return 1

        return handle_command(args)

    except (DataSourceError, AnalysisError, ParsingError, ConfigurationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())