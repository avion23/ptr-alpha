"""Run Kelly portfolio backtest against the DB and report metrics."""

import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="cryptography")

# Ensure repo src is importable regardless of working directory
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from datetime import date, timedelta

import pandas as pd

from analyzer.settings import Settings
from analyzer.datasources import HouseTransactionSource
from analyzer.portfolio import (
    KellyConfig,
    build_portfolios_from_backtest,
    compute_portfolio_metrics,
    simulate_portfolio_returns,
)


def main():
    settings = Settings()
    tx_source = HouseTransactionSource(settings, read_only=True)

    # Best config from parameter sweeps
    horizon = 60
    frequency = 30
    min_buyers = 2
    top_n = 5
    training_lookback_days = 365

    start_date = date(2022, 1, 1)
    end_date = date(2025, 12, 31)

    # Load data
    tx_start = start_date - timedelta(days=training_lookback_days + horizon + 30)
    tx_end = end_date

    print("Loading transactions...")
    all_transactions = tx_source.db.get_transactions_by_date_range(tx_start, tx_end)
    if "ticker" in all_transactions.columns:
        all_transactions["ticker"] = all_transactions["ticker"].astype(str)
    print(f"  {len(all_transactions)} transactions")

    price_start = tx_start
    price_end = end_date + timedelta(days=horizon + 10)
    all_tickers = [str(t) for t in all_transactions["ticker"].unique() if pd.notna(t)]
    all_tickers = sorted(set(all_tickers) | {"SPY"})

    print("Loading prices (cached only)...")
    prices = tx_source.db.get_prices(all_tickers, price_start, price_end)
    available = [t for t in all_tickers if t in prices.columns]
    prices = prices[available].dropna(axis=1, how="all")
    print(f"  {prices.shape[1]} tickers, {len(prices)} days")

    print("Computing entry prices...")
    entry_prices = tx_source.db.get_entry_prices(all_tickers, price_start, price_end)
    if not entry_prices.empty and "ticker" in entry_prices.columns:
        entry_prices["ticker"] = entry_prices["ticker"].astype(str)
    print(f"  {len(entry_prices)} entry prices")

    print("Computing signals...")
    from analyzer import analysis

    signals = analysis.calculate_signal_potential(entry_prices, prices, [horizon])
    print(f"  {len(signals)} signals")

    # Build portfolios across all as_of dates
    as_of_dates = pd.date_range(start_date, end_date, freq=f"{frequency}D")

    kelly_config = KellyConfig(
        capital=100_000,
        max_ticker_pct=0.20,
        max_member_pct=0.05,
        total_exposure_pct=1.00,
        use_half_kelly=True,
        crash_guard=True,
    )

    print(f"\nBuilding Kelly portfolios across {len(as_of_dates)} rebalance dates...")
    portfolio_df = build_portfolios_from_backtest(
        signals,
        all_transactions,
        prices,
        as_of_dates,
        horizon=horizon,
        lookback_days=30,
        min_buyers=min_buyers,
        top_n=top_n,
        threshold=5.0,
        training_lookback_days=training_lookback_days,
        config=kelly_config,
    )

    if portfolio_df.empty:
        print("No portfolios generated!")
        tx_source.close()
        return

    print(
        f"  {len(portfolio_df)} portfolio entries across {portfolio_df['as_of_date'].nunique()} dates"
    )

    # Position sizing distribution
    print("\n=== Position Sizing Distribution ===")
    print(f"  Mean weight:    {portfolio_df['weight'].mean():.3f}")
    print(f"  Median weight:  {portfolio_df['weight'].median():.3f}")
    print(f"  Max weight:     {portfolio_df['weight'].max():.3f}")
    print(f"  Min weight:     {portfolio_df['weight'].min():.3f}")
    print(f"  Mean kelly:     {portfolio_df['kelly_fraction'].mean():.4f}")
    print(f"  Mean position:  ${portfolio_df['position_value'].mean():,.0f}")
    print("  Total invested per period:")
    totals = portfolio_df.groupby("as_of_date")["position_value"].sum()
    print(f"    Mean:   ${totals.mean():,.0f}")
    print(f"    Median: ${totals.median():,.0f}")
    print(f"    Max:    ${totals.max():,.0f}")
    print(f"    Min:    ${totals.min():,.0f}")

    # Simulate returns
    print("\nSimulating portfolio returns...")
    returns_df = simulate_portfolio_returns(portfolio_df, prices, horizon=horizon)

    if returns_df.empty:
        print("No returns computed!")
        tx_source.close()
        return

    # Compute metrics
    metrics = compute_portfolio_metrics(returns_df)

    print(f"\n{'=' * 60}")
    print("=== Kelly Portfolio Performance ===")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s} {v:>10.2f}")
        else:
            print(f"  {k:35s} {str(v):>10s}")

    # Compare to equal-weight
    print(f"\n{'=' * 60}")
    print("=== Per-Period Returns ===")
    print(returns_df.to_string(index=False))

    # Win/loss analysis
    wins = returns_df[returns_df["portfolio_return"] > 0]
    losses = returns_df[returns_df["portfolio_return"] <= 0]
    print(
        f"\nWinning periods: {len(wins)}/{len(returns_df)} ({len(wins) / len(returns_df) * 100:.1f}%)"
    )
    if not wins.empty:
        print(f"  Avg win:  {wins['portfolio_return'].mean():.2f}%")
    if not losses.empty:
        print(f"  Avg loss: {losses['portfolio_return'].mean():.2f}%")

    tx_source.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
