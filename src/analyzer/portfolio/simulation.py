"""Portfolio simulation: build portfolios from backtests and simulate returns.

`build_portfolios_from_backtest` runs the backtest at every as_of_date and
builds a Kelly-sized portfolio per date.

`simulate_portfolio_returns` then simulates holding-period returns for
each portfolio using the price index (O(log N) per ticker).
"""

from __future__ import annotations

import pandas as pd

from analyzer.backtest import _price_at_or_before_arrays, _price_on_or_before_arrays
from analyzer.signals import _price_arrays

from analyzer.portfolio.kelly import KellyConfig, build_kelly_portfolio

NS_PER_DAY = 86_400_000_000_000


def build_portfolios_from_backtest(
    signals_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_dates: pd.DatetimeIndex | list,
    horizon: int = 60,
    lookback_days: int = 30,
    min_buyers: int = 2,
    top_n: int = 5,
    threshold: float = 5.0,
    training_lookback_days: int | None = None,
    config: KellyConfig | None = None,
) -> pd.DataFrame:
    """For each as_of_date, build a Kelly-sized portfolio from top-N signals.

    Returns a DataFrame with columns: as_of_date, ticker, member, weight,
        kelly_fraction, signal_score, crash_prob, position_value.
    """
    from analyzer.backtest import backtest_recommendations

    all_portfolios: list[pd.DataFrame] = []
    for as_of in as_of_dates:
        recs = backtest_recommendations(
            signals_df,
            transactions_df,
            pd.Timestamp(as_of),
            horizon=horizon,
            lookback_days=lookback_days,
            min_buyers=min_buyers,
            top_n=top_n,
            threshold=threshold,
            prices_df=prices_df,
            training_lookback_days=training_lookback_days,
        )
        portfolio = _build_one_portfolio(recs, as_of, config)
        if portfolio is not None:
            all_portfolios.append(portfolio)

    if not all_portfolios:
        return pd.DataFrame()
    return pd.concat(all_portfolios, ignore_index=True)


def _build_one_portfolio(recs: pd.DataFrame, as_of, config: KellyConfig | None):
    if recs.empty:
        return None
    portfolio = build_kelly_portfolio(recs, config)
    if portfolio.empty:
        return None
    portfolio.insert(0, "as_of_date", pd.Timestamp(as_of).date())
    return portfolio


def simulate_portfolio_returns(
    portfolio_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon: int = 60,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
) -> pd.DataFrame:
    """Simulate portfolio returns over the holding period.

    For each portfolio (grouped by as_of_date), compute weighted return
    vs SPY benchmark.

    Returns DataFrame with columns: as_of_date, portfolio_return, spy_return,
        portfolio_alpha, num_positions.
    """
    if portfolio_df.empty or prices_df.empty:
        return pd.DataFrame()

    entry_mult = 1.0 + entry_slippage_bps / 10000
    exit_mult = 1.0 - exit_slippage_bps / 10000
    spy_arrs = _price_arrays(prices_df, "SPY")

    results: list[dict] = []
    for as_of_date, group in portfolio_df.groupby("as_of_date"):
        result = _simulate_one_period(
            group, prices_df, spy_arrs, as_of_date, horizon, entry_mult, exit_mult
        )
        results.append(result)

    return pd.DataFrame(results)


def _simulate_one_period(
    group: pd.DataFrame,
    prices_df: pd.DataFrame,
    spy_arrs,
    as_of_date,
    horizon: int,
    entry_mult: float,
    exit_mult: float,
) -> dict:
    """Simulate one portfolio over its holding period."""
    as_of_ts = pd.Timestamp(as_of_date)
    weighted_return = 0.0
    num_positions = 0
    exit_ns = as_of_ts.value + horizon * NS_PER_DAY

    for _, pos in group.iterrows():
        per_position = _simulate_one_position(
            pos,
            prices_df,
            as_of_ts,
            exit_ns,
            entry_mult,
            exit_mult,
        )
        if per_position is None:
            continue
        weighted_return += per_position["weight"] * per_position["return_pct"]
        num_positions += 1

    spy_return = _compute_spy_return(spy_arrs, as_of_ts, exit_ns, entry_mult, exit_mult)

    return {
        "as_of_date": as_of_date,
        "portfolio_return": round(weighted_return, 2),
        "spy_return": round(spy_return, 2),
        "portfolio_alpha": round(weighted_return - spy_return, 2),
        "num_positions": num_positions,
    }


def _simulate_one_position(
    pos: pd.Series,
    prices_df: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    exit_ns: int,
    entry_mult: float,
    exit_mult: float,
) -> dict | None:
    """Return (weight, return_pct) for one position, or None if no price data."""
    ticker = pos["ticker"]
    weight = pos["weight"]

    tkr_arrs = _price_arrays(prices_df, ticker)
    if tkr_arrs is None or tkr_arrs[0] is None:
        return None
    idx_ns, vals = tkr_arrs

    entry = _price_at_or_before_arrays(idx_ns, vals, as_of_ts, max_staleness_days=30)
    if entry is None:
        return None
    exit_price = _price_on_or_before_arrays(
        idx_ns,
        vals,
        pd.Timestamp(exit_ns),
        max_staleness_days=30,
    )
    if exit_price is None:
        return None

    entry_adj = entry * entry_mult
    exit_adj = exit_price * exit_mult
    if entry_adj <= 0 or exit_adj < 0:
        return None
    return {
        "ticker": ticker,
        "weight": weight,
        "return_pct": (exit_adj / entry_adj - 1.0) * 100,
    }


def _compute_spy_return(
    spy_arrs,
    as_of_ts: pd.Timestamp,
    exit_ns: int,
    entry_mult: float,
    exit_mult: float,
) -> float:
    """SPY benchmark return over the same holding window, or 0 if no data."""
    if spy_arrs is None or spy_arrs[0] is None:
        return 0.0
    spy_ns, spy_vals = spy_arrs
    spy_entry = _price_at_or_before_arrays(
        spy_ns, spy_vals, as_of_ts, max_staleness_days=30
    )
    spy_exit = _price_on_or_before_arrays(
        spy_ns,
        spy_vals,
        pd.Timestamp(exit_ns),
        max_staleness_days=30,
    )
    if spy_entry is None or spy_exit is None:
        return 0.0
    entry_adj = spy_entry * entry_mult
    exit_adj = spy_exit * exit_mult
    if entry_adj <= 0 or exit_adj < 0:
        return 0.0
    return (exit_adj / entry_adj - 1.0) * 100
