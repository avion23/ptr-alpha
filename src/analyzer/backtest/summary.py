"""Funded rebalance-date backtest summaries and a true SPY buy/hold benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_backtest(
    results: pd.DataFrame,
    spy_prices: pd.Series | None = None,
    *,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
) -> pd.DataFrame:
    """Summarize equal-funded portfolios once per rebalance date.

    Recommendation rows on the same date share one unit of capital equally. This
    prevents opportunity-dense dates from receiving more weight merely because
    they produced more recommendations. A SPY row is emitted only when an actual
    buy/hold return can be computed from a supplied price series or a verified
    scalar attached by the caller; repeated per-recommendation SPY windows are
    never relabeled as buy-and-hold.
    """
    n_no_price = results.attrs.get("n_no_price", 0)
    n_delisted = results.attrs.get("n_delisted", 0)
    valid = results.dropna(subset=["bt_return_pct"]).copy()
    if valid.empty:
        return _with_coverage(pd.DataFrame(), n_no_price, n_delisted)

    periods = _funded_period_returns(valid)
    holding_policy, avg_holding_days = _holding_policy(valid)
    rows = [
        {
            "rank": "PORTFOLIO",
            "count": len(periods),
            "recommendation_count": len(valid),
            "win_rate_pct": round((periods["return_pct"] > 0).mean() * 100, 1),
            "avg_return_pct": round(float(periods["return_pct"].mean()), 2),
            "avg_alpha_pct": round(float(periods["alpha_pct"].mean()), 2),
            "avg_raw_return_pct": round(float(periods["raw_return_pct"].mean()), 2),
            "avg_leverage": round(
                float(valid.get("bt_leverage", pd.Series([1.0])).mean()), 2
            ),
            "holding_policy": holding_policy,
            "avg_holding_days": avg_holding_days,
        }
    ]

    spy_return = _spy_buy_hold_return(
        valid,
        spy_prices,
        entry_slippage_bps=entry_slippage_bps,
        exit_slippage_bps=exit_slippage_bps,
    )
    if spy_return is not None:
        rows.append(
            {
                "rank": "SPY_BUY_HOLD",
                "count": 1,
                "recommendation_count": 0,
                "win_rate_pct": round(float(spy_return > 0) * 100, 1),
                "avg_return_pct": round(spy_return, 2),
                "avg_alpha_pct": 0.0,
                "avg_raw_return_pct": round(spy_return, 2),
                "avg_leverage": 1.0,
                "holding_policy": "buy_and_hold",
                "avg_holding_days": _benchmark_days(valid),
            }
        )

    return _with_coverage(pd.DataFrame(rows), n_no_price, n_delisted)


def _funded_period_returns(valid: pd.DataFrame) -> pd.DataFrame:
    period_key = "as_of_date" if "as_of_date" in valid.columns else None
    frame = valid.copy()
    if period_key is None:
        frame["_period"] = np.arange(len(frame))
        period_key = "_period"

    def numeric(name: str, fallback: float = 0.0) -> pd.Series:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(fallback)
        return pd.Series(fallback, index=frame.index, dtype=float)

    frame["_return"] = numeric("bt_return_pct")
    frame["_alpha"] = numeric("bt_alpha_pct")
    frame["_raw_return"] = numeric("bt_raw_return_pct", np.nan).fillna(frame["_return"])
    return (
        frame.groupby(period_key, sort=True)
        .agg(
            return_pct=("_return", "mean"),
            alpha_pct=("_alpha", "mean"),
            raw_return_pct=("_raw_return", "mean"),
        )
        .reset_index(drop=True)
    )


def _holding_policy(valid: pd.DataFrame) -> tuple[str, float | None]:
    for col in ("bt_holding_days", "bt_horizon_days", "optimal_horizon"):
        if col not in valid.columns:
            continue
        horizons = pd.to_numeric(valid[col], errors="coerce").dropna()
        if horizons.empty:
            continue
        avg = round(float(horizons.mean()), 1)
        unique = sorted(set(int(v) for v in horizons))
        if len(unique) == 1:
            return f"fixed_{unique[0]}d", avg
        return f"adaptive_{unique[0]}-{unique[-1]}d", avg
    return "unknown", None


def _spy_buy_hold_return(
    valid: pd.DataFrame,
    spy_prices: pd.Series | None,
    *,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
) -> float | None:
    attached = valid.attrs.get("spy_buy_hold_return_pct")
    if attached is not None and np.isfinite(attached):
        return float(attached)
    if spy_prices is None:
        return None

    series = spy_prices.dropna().sort_index()
    series = series[np.isfinite(series) & (series > 0)]
    if series.empty:
        return None
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    start, end = _benchmark_bounds(valid)
    entries = series[series.index >= start]
    exits = series[series.index <= end]
    if entries.empty or exits.empty:
        return None
    entry_date = entries.index[0]
    exit_date = exits.index[-1]
    if exit_date < entry_date:
        return None

    entry = float(entries.iloc[0]) * (1 + entry_slippage_bps / 10000)
    exit_ = float(exits.iloc[-1]) * (1 - exit_slippage_bps / 10000)
    return (exit_ / entry - 1) * 100


def _benchmark_bounds(valid: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    if "bt_entry_date" in valid.columns:
        start = pd.to_datetime(valid["bt_entry_date"]).min()
    elif "as_of_date" in valid.columns:
        start = pd.to_datetime(valid["as_of_date"]).min()
    else:
        start = pd.Timestamp.min

    if "bt_exit_date" in valid.columns:
        end = pd.to_datetime(valid["bt_exit_date"]).max()
    elif "as_of_date" in valid.columns and "bt_horizon_days" in valid.columns:
        as_of = pd.to_datetime(valid["as_of_date"])
        horizon = pd.to_timedelta(
            pd.to_numeric(valid["bt_horizon_days"], errors="coerce").fillna(0),
            unit="D",
        )
        end = (as_of + horizon).max()
    elif "as_of_date" in valid.columns:
        end = pd.to_datetime(valid["as_of_date"]).max()
    else:
        end = pd.Timestamp.max
    return pd.Timestamp(start), pd.Timestamp(end)


def _benchmark_days(valid: pd.DataFrame) -> int | None:
    start, end = _benchmark_bounds(valid)
    if start in (pd.Timestamp.min, pd.Timestamp.max) or end in (
        pd.Timestamp.min,
        pd.Timestamp.max,
    ):
        return None
    return int((end - start).days)


def _with_coverage(
    summary: pd.DataFrame, n_no_price: int, n_delisted: int
) -> pd.DataFrame:
    summary.attrs["n_no_price"] = n_no_price
    summary.attrs["n_delisted"] = n_delisted
    return summary
