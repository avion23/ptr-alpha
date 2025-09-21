#!/usr/bin/env python3

import argparse
import sys
import logging
from analyzer.sources import Config
from analyzer.pipeline import run_fetch_pipeline, run_parse_pipeline, run_analysis_pipeline
from analyzer.exceptions import DataSourceError, AnalysisError, ParsingError, ConfigurationError

def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_config(args):
    try:
        return Config(
            data_dir=args.data_dir,
            cache_enabled=not args.no_cache,
            parallel_workers=args.workers
        )
    except Exception as e:
        raise ConfigurationError(f"Failed to create configuration: {e}")

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
    else:
        print("Use 'fetch', 'parse', 'rank-members', 'show-signals', or 'show-member-signals'", file=sys.stderr)
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
               "  %(prog)s show-member-signals --member 'Nancy Pelosi'"
    )

    parser.add_argument('--year', type=int, default=2024, help='Year to process (default: 2024)')
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

    return parser.parse_args()

def main():
    try:
        args = parse_args()
        setup_logging(args.verbose)

        if not args.command:
            print("Error: No command specified", file=sys.stderr)
            return 1

        return handle_command(args)

    except (DataSourceError, AnalysisError, ParsingError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ConfigurationError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())