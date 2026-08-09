"""Kelly portfolio construction and shared-capital execution simulation.

Every rebalance uses the same cash account.  Positions remain open until their
holding horizon can be executed, so overlapping signals cannot reuse bankroll.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analyzer.portfolio.kelly import (
    REQUIRED_SIZING_COLUMNS,
    KellyConfig,
    build_kelly_portfolio,
)


@dataclass(slots=True)
class _Position:
    ticker: str
    shares: float
    entry_date: pd.Timestamp
    exit_target: pd.Timestamp
    raw_entry_price: float
    execution_cost: float


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
    sizing_inputs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-date Kelly targets from explicit historical sizing inputs.

    ``sizing_inputs_df`` must supply ``as_of_date``, ``ticker``, ``member``,
    ``win_rate``, ``avg_win_pct``, and ``avg_loss_pct``.  The estimates must be
    available as of that row's date.  If they are absent, the builder abstains;
    it never substitutes hard-coded outcome assumptions.
    """
    from analyzer.backtest import backtest_recommendations

    config = config or KellyConfig()
    sizing = _prepare_sizing_inputs(sizing_inputs_df)
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
        enriched = _attach_sizing_inputs(recs, sizing, as_of)
        portfolio = build_kelly_portfolio(enriched, config)
        if portfolio.empty:
            continue
        portfolio.insert(0, "as_of_date", pd.Timestamp(as_of).normalize())
        all_portfolios.append(portfolio)

    if not all_portfolios:
        return pd.DataFrame()
    return pd.concat(all_portfolios, ignore_index=True)


def _prepare_sizing_inputs(sizing_inputs_df: pd.DataFrame | None) -> pd.DataFrame:
    if sizing_inputs_df is None or sizing_inputs_df.empty:
        return pd.DataFrame()
    required = (REQUIRED_SIZING_COLUMNS - {"signal_score"}) | {"as_of_date"}
    missing = required - set(sizing_inputs_df.columns)
    if missing:
        raise ValueError(f"sizing_inputs_df missing required columns: {sorted(missing)}")

    sizing = sizing_inputs_df.copy()
    sizing["as_of_date"] = pd.to_datetime(sizing["as_of_date"], errors="coerce").dt.normalize()
    sizing["ticker"] = sizing["ticker"].astype("string").str.strip()
    sizing["member"] = sizing["member"].astype("string").str.strip()
    numeric = ["win_rate", "avg_win_pct", "avg_loss_pct"]
    if "crash_prob" in sizing.columns:
        numeric.append("crash_prob")
    for column in numeric:
        sizing[column] = pd.to_numeric(sizing[column], errors="coerce")
    finite = np.isfinite(sizing[numeric]).all(axis=1)
    valid = (
        sizing["as_of_date"].notna()
        & sizing["ticker"].notna()
        & sizing["ticker"].ne("")
        & sizing["member"].notna()
        & sizing["member"].ne("")
        & ~sizing["member"].str.lower().isin({"unknown", "none", "nan"})
        & finite
        & sizing["win_rate"].between(0.0, 1.0, inclusive="neither")
        & sizing["avg_win_pct"].gt(0)
        & sizing["avg_loss_pct"].gt(0)
    )
    if "crash_prob" in sizing.columns:
        valid &= sizing["crash_prob"].between(0.0, 1.0, inclusive="both")
    if not bool(valid.all()):
        bad_rows = sizing.index[~valid].tolist()
        raise ValueError(f"sizing_inputs_df has malformed rows: {bad_rows}")
    duplicate = sizing.duplicated(["as_of_date", "ticker"], keep=False)
    if bool(duplicate.any()):
        keys = sizing.loc[duplicate, ["as_of_date", "ticker"]].to_dict("records")
        raise ValueError(f"sizing_inputs_df has duplicate dated ticker estimates: {keys}")
    return sizing.sort_values(["ticker", "as_of_date"], kind="stable").reset_index(drop=True)


def _attach_sizing_inputs(
    recs: pd.DataFrame, sizing: pd.DataFrame, as_of
) -> pd.DataFrame:
    if recs.empty or sizing.empty:
        return pd.DataFrame()
    if recs["ticker"].duplicated().any():
        raise ValueError("recommendations contain duplicate tickers")
    rebalance = pd.Timestamp(as_of).normalize()
    eligible = sizing[sizing["as_of_date"] <= rebalance]
    if eligible.empty:
        return pd.DataFrame()
    latest = eligible.groupby("ticker", sort=False, as_index=False).tail(1)
    columns = [
        "ticker", "member", "win_rate", "avg_win_pct", "avg_loss_pct"
    ]
    if "crash_prob" in latest.columns and "crash_prob" not in recs.columns:
        columns.append("crash_prob")
    return recs.merge(latest[columns], on="ticker", how="inner", validate="one_to_one")


def simulate_portfolio_returns(
    portfolio_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon: int = 60,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
    initial_capital: float | None = None,
    max_execution_delay_days: int = 30,
    max_mark_staleness_days: int | None = None,
) -> pd.DataFrame:
    """Execute dated targets against one shared cash and position ledger.

    The fill policy is all-or-skip: a target is executed only when the whole
    requested notional is available in cash.  Open positions are marked from the
    latest finite, non-negative price within ``max_mark_staleness_days``.  Zero is
    a valid total-loss mark.  If any open position has no current mark, portfolio
    equity is unavailable and no new position is sized from fictional value.
    """
    if portfolio_df.empty or prices_df.empty:
        return pd.DataFrame()
    required = {"as_of_date", "ticker", "weight"}
    missing = required - set(portfolio_df.columns)
    if missing:
        raise ValueError(f"portfolio_df missing required columns: {sorted(missing)}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_execution_delay_days < 0:
        raise ValueError("max_execution_delay_days must be non-negative")
    mark_staleness = (
        max_execution_delay_days
        if max_mark_staleness_days is None
        else max_mark_staleness_days
    )
    if mark_staleness < 0:
        raise ValueError("max_mark_staleness_days must be non-negative")
    if (
        entry_slippage_bps < 0
        or exit_slippage_bps < 0
        or entry_slippage_bps >= 10_000
        or exit_slippage_bps >= 10_000
    ):
        raise ValueError("slippage must be in [0, 10000) basis points")

    capital = _resolve_initial_capital(portfolio_df, initial_capital)
    prices = prices_df.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()

    requested = len(portfolio_df)
    signals = portfolio_df.copy()
    signals["as_of_date"] = pd.to_datetime(
        signals["as_of_date"], errors="coerce"
    ).dt.normalize()
    signals["weight"] = pd.to_numeric(signals["weight"], errors="coerce")
    signals["ticker"] = signals["ticker"].astype("string").str.strip()
    valid_target = (
        signals["as_of_date"].notna()
        & signals["ticker"].notna()
        & signals["ticker"].ne("")
        & np.isfinite(signals["weight"])
        & signals["weight"].gt(0)
        & signals["weight"].le(1)
    )
    invalid_target_skips = int((~valid_target).sum())
    signals = signals.loc[valid_target].copy()
    start = (
        signals["as_of_date"].min()
        if not signals.empty
        else prices.index.min()
    )
    last_signal_date = (
        signals["as_of_date"].max()
        if not signals.empty
        else start
    )
    entry_schedule, price_coverage_skips = _entry_schedule(
        signals, prices, max_execution_delay_days
    )
    final_target = last_signal_date + pd.Timedelta(
        days=horizon + max_execution_delay_days
    )
    event_dates = prices.index[(prices.index >= start) & (prices.index <= final_target)]
    if event_dates.empty:
        return pd.DataFrame()

    cash = capital
    positions: dict[str, _Position] = {}
    executed = 0
    closed = 0
    duplicate_open_skips = 0
    cash_skips = 0
    valuation_skips = 0
    realized_costs = 0.0
    gross_traded = 0.0
    requested_entry_notional = 0.0
    filled_entry_notional = 0.0
    entry_slip = entry_slippage_bps / 10_000.0
    exit_slip = exit_slippage_bps / 10_000.0
    rows: list[dict] = []
    last_entry_date = max(entry_schedule) if entry_schedule else event_dates[0]

    for current in event_dates:
        for ticker, position in list(positions.items()):
            if current < position.exit_target:
                continue
            raw_exit = _exact_exit_price(prices, ticker, current)
            if raw_exit is None:
                continue
            if (current - position.exit_target).days > max_execution_delay_days:
                continue
            proceeds = position.shares * raw_exit * (1.0 - exit_slip)
            cash += proceeds
            realized_costs += position.shares * raw_exit * exit_slip
            gross_traded += position.shares * raw_exit
            closed += 1
            del positions[ticker]

        for signal in entry_schedule.get(current, []):
            ticker = str(signal["ticker"])
            if ticker in positions:
                duplicate_open_skips += 1
                continue
            raw_entry = _exact_positive_price(prices, ticker, current)
            if raw_entry is None:
                price_coverage_skips += 1
                continue
            liquidation = _liquidation_equity(
                cash, positions, prices, current, exit_slip, mark_staleness
            )
            liquidation_equity = liquidation[0]
            if liquidation_equity is None:
                valuation_skips += 1
                continue
            target = liquidation_equity * float(signal["weight"])
            requested_entry_notional += target
            if target <= 0 or target > cash + 1e-9:
                cash_skips += 1
                continue
            execution_price = raw_entry * (1.0 + entry_slip)
            shares = target / execution_price
            cash -= target
            realized_costs += shares * raw_entry * entry_slip
            gross_traded += shares * raw_entry
            filled_entry_notional += target
            positions[ticker] = _Position(
                ticker=ticker,
                shares=shares,
                entry_date=current,
                exit_target=current + pd.Timedelta(days=horizon),
                raw_entry_price=raw_entry,
                execution_cost=target,
            )
            executed += 1

        liquidation = _liquidation_equity(
            cash, positions, prices, current, exit_slip, mark_staleness
        )
        liquidation_equity, market_value, estimated_exit_cost = liquidation[:3]
        unavailable_positions, stale_positions = liquidation[3:]
        valuation_complete = unavailable_positions == 0
        gross_equity = (
            cash + market_value if market_value is not None else np.nan
        )
        liquidation_value = (
            liquidation_equity if liquidation_equity is not None else np.nan
        )
        exposure = market_value if market_value is not None else np.nan
        exit_cost = (
            estimated_exit_cost
            if estimated_exit_cost is not None
            else np.nan
        )
        skipped = (
            invalid_target_skips
            + price_coverage_skips
            + duplicate_open_skips
            + cash_skips
            + valuation_skips
        )
        rows.append(
            {
                "date": current,
                "simulation_start": start,
                "initial_capital": capital,
                "cash": cash,
                "gross_value": gross_equity,
                "liquidation_value": liquidation_value,
                "valuation_complete": valuation_complete,
                "valuation_coverage_pct": (
                    (len(positions) - unavailable_positions) / len(positions) * 100
                    if positions else 100.0
                ),
                "unavailable_open_positions": unavailable_positions,
                "stale_open_positions": stale_positions,
                "open_exposure": exposure,
                "open_exposure_pct": (
                    market_value / liquidation_equity * 100
                    if market_value is not None
                    and liquidation_equity is not None
                    and liquidation_equity > 0
                    else np.nan if not valuation_complete else 0.0
                ),
                "estimated_liquidation_cost": exit_cost,
                "realized_transaction_costs": realized_costs,
                "gross_traded_notional": gross_traded,
                "requested_entry_notional": requested_entry_notional,
                "filled_entry_notional": filled_entry_notional,
                "notional_fill_pct": (
                    filled_entry_notional / requested_entry_notional * 100
                    if requested_entry_notional > 0 else 0.0
                ),
                "execution_policy": "all_or_skip",
                "partial_fills": 0,
                "requested_signals": requested,
                "executed_positions": executed,
                "skipped_signals": skipped,
                "invalid_target_skips": invalid_target_skips,
                "price_coverage_skips": price_coverage_skips,
                "duplicate_open_skips": duplicate_open_skips,
                "cash_skips": cash_skips,
                "valuation_skips": valuation_skips,
                "closed_positions": closed,
                "open_positions": len(positions),
                "signal_coverage_pct": executed / requested * 100 if requested else 0.0,
                "close_coverage_pct": closed / executed * 100 if executed else 0.0,
            }
        )
        if current >= last_entry_date and not positions:
            break

    return pd.DataFrame(rows)

def _resolve_initial_capital(
    portfolio_df: pd.DataFrame, initial_capital: float | None
) -> float:
    if initial_capital is not None:
        capital = float(initial_capital)
    elif "position_value" in portfolio_df.columns:
        weights = pd.to_numeric(portfolio_df["weight"], errors="coerce")
        values = pd.to_numeric(portfolio_df["position_value"], errors="coerce")
        ratios = values[weights > 0] / weights[weights > 0]
        ratios = ratios[np.isfinite(ratios) & ratios.gt(0)]
        if ratios.empty or not np.allclose(ratios, ratios.iloc[0], rtol=1e-6):
            raise ValueError("position_value does not identify one initial bankroll")
        capital = float(ratios.iloc[0])
    else:
        raise ValueError("initial_capital is required when position_value is absent")
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("initial_capital must be positive and finite")
    return capital


def _entry_schedule(
    signals: pd.DataFrame, prices: pd.DataFrame, max_delay: int
) -> tuple[dict[pd.Timestamp, list[dict]], int]:
    schedule: dict[pd.Timestamp, list[dict]] = {}
    skipped = 0
    for row in signals.sort_values(["as_of_date", "weight"], ascending=[True, False]).to_dict("records"):
        ticker = str(row["ticker"])
        if ticker not in prices.columns:
            skipped += 1
            continue
        series = prices[ticker].dropna()
        series = series[np.isfinite(series) & series.gt(0)]
        eligible = series[series.index >= row["as_of_date"]]
        if eligible.empty:
            skipped += 1
            continue
        execution_date = eligible.index[0]
        if (execution_date - row["as_of_date"]).days > max_delay:
            skipped += 1
            continue
        schedule.setdefault(execution_date, []).append(row)
    return schedule, skipped


def _exact_positive_price(
    prices: pd.DataFrame, ticker: str, current: pd.Timestamp
) -> float | None:
    if ticker not in prices.columns or current not in prices.index:
        return None
    value = prices.at[current, ticker]
    if pd.isna(value) or not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def _exact_exit_price(
    prices: pd.DataFrame, ticker: str, current: pd.Timestamp
) -> float | None:
    if ticker not in prices.columns or current not in prices.index:
        return None
    value = prices.at[current, ticker]
    if pd.isna(value) or not np.isfinite(value) or value < 0:
        return None
    return float(value)


def _mark_price(
    prices: pd.DataFrame,
    ticker: str,
    current: pd.Timestamp,
    max_staleness_days: int,
) -> tuple[float | None, bool]:
    if ticker not in prices.columns:
        return None, False
    series = prices.loc[:current, ticker].dropna()
    if series.empty:
        return None, False
    mark_date = series.index[-1]
    value = series.iloc[-1]
    stale = (current - mark_date).days > max_staleness_days
    if stale:
        return None, True
    if not np.isfinite(value) or value < 0:
        return None, False
    return float(value), False


def _liquidation_equity(
    cash: float,
    positions: dict[str, _Position],
    prices: pd.DataFrame,
    current: pd.Timestamp,
    exit_slip: float,
    max_staleness_days: int,
) -> tuple[float | None, float | None, float | None, int, int]:
    market_value = 0.0
    unavailable = 0
    stale = 0
    for ticker, position in positions.items():
        price, is_stale = _mark_price(
            prices, ticker, current, max_staleness_days
        )
        if price is None:
            unavailable += 1
            stale += int(is_stale)
            continue
        market_value += position.shares * price
    if unavailable:
        return None, None, None, unavailable, stale
    estimated_exit_cost = market_value * exit_slip
    return (
        cash + market_value - estimated_exit_cost,
        market_value,
        estimated_exit_cost,
        0,
        0,
    )
