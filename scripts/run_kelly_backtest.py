"""Run the shared-capital Kelly backtest against explicit sizing estimates."""

import os
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="cryptography")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from analyzer.download import HouseTransactionSource
from analyzer.member_ranking.buyer_scoring import (
    CONSENSUS_LOOKBACK_DAYS,
    CONSENSUS_MIN_BUYERS,
    _get_consensus_price_tickers,
)
from analyzer.portfolio import (
    KellyConfig,
    build_portfolios_from_backtest,
    compute_portfolio_metrics,
    simulate_portfolio_returns,
)
from analyzer.price_repository import next_nyse_session
from analyzer.settings import Settings
from analyzer.validation import _phase_end


def main() -> int:
    sizing_path = Path(
        os.environ.get("KELLY_SIZING_INPUTS", "data/kelly_sizing_inputs.csv")
    )
    if not sizing_path.exists():
        print(
            "Error: Kelly sizing abstained. Provide dated historical estimates at "
            f"{sizing_path} with as_of_date,ticker,member,win_rate,"
            "avg_win_pct,avg_loss_pct.",
            file=sys.stderr,
        )
        return 2
    sizing_inputs = pd.read_csv(sizing_path)

    settings = Settings()
    tx_source = HouseTransactionSource(settings, read_only=True)
    try:
        return _run(tx_source, sizing_inputs)
    finally:
        tx_source.close()


def _run(tx_source, sizing_inputs: pd.DataFrame) -> int:
    horizon = 60
    frequency = 30
    min_buyers = CONSENSUS_MIN_BUYERS
    top_n = 5
    start_date = date(2022, 1, 1)
    outcome_boundary = date(2025, 12, 31)
    last_as_of = _phase_end(outcome_boundary, horizon)
    capital = 100_000.0

    tx_start = start_date - timedelta(days=CONSENSUS_LOOKBACK_DAYS)
    print("Loading transactions...")
    transactions = tx_source.db.get_transactions_by_date_range(tx_start, last_as_of)
    print(f"  {len(transactions)} transactions")

    tickers = sorted(set(_get_consensus_price_tickers(transactions)) | {"SPY"})
    price_start = next_nyse_session(start_date)
    prices = tx_source.db.get_prices(tickers, price_start, outcome_boundary)
    prices = prices.dropna(axis=1, how="all")
    print(f"  {prices.shape[1]} price series, {len(prices)} dates")

    as_of_dates = pd.date_range(start_date, last_as_of, freq=f"{frequency}D")
    config = KellyConfig(
        capital=capital,
        max_ticker_pct=0.20,
        max_member_pct=0.05,
        total_exposure_pct=1.00,
        use_half_kelly=True,
        crash_guard=False,
    )
    targets = build_portfolios_from_backtest(
        pd.DataFrame(),
        transactions,
        prices,
        as_of_dates,
        horizon=horizon,
        lookback_days=CONSENSUS_LOOKBACK_DAYS,
        min_buyers=min_buyers,
        top_n=top_n,
        config=config,
        sizing_inputs_df=sizing_inputs,
    )
    if targets.empty:
        print(
            "Error: no targets had complete, dated member/outcome sizing inputs; "
            "Kelly abstained.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Built {len(targets)} targets across {targets['as_of_date'].nunique()} dates; "
        f"mean requested exposure={targets.groupby('as_of_date')['weight'].sum().mean():.3f}"
    )
    equity = simulate_portfolio_returns(
        targets, prices, horizon=horizon, initial_capital=capital
    )
    if equity.empty:
        print("Error: no executable equity curve; Kelly abstained.", file=sys.stderr)
        return 2

    metrics = compute_portfolio_metrics(equity)
    print("\n=== Shared-Capital Kelly Performance ===")
    for key, value in metrics.items():
        print(f"  {key:35s} {value}")
    final = equity.iloc[-1]
    if final["valuation_complete"]:
        print(
            f"\nFinal gross value ${final['gross_value']:,.2f}; "
            f"liquidation value ${final['liquidation_value']:,.2f}; "
            f"open exposure ${final['open_exposure']:,.2f}; "
            f"estimated liquidation cost ${final['estimated_liquidation_cost']:,.2f}."
        )
    else:
        print(
            "\nFinal portfolio value unavailable: "
            f"{int(final['unavailable_open_positions'])} open positions lack a "
            "fresh, valid mark."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
