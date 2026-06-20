"""Portfolio construction with Kelly criterion position sizing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KellyConfig:
    """Configuration for Kelly portfolio construction."""
    capital: float = 100_000.0
    max_ticker_pct: float = 0.20       # max 20% per ticker
    max_member_pct: float = 0.05       # max 5% per member
    total_exposure_pct: float = 1.00    # max 100% invested
    use_half_kelly: bool = True         # safer half-Kelly by default
    crash_guard: bool = True            # reduce by crash_prob
    min_kelly_floor: float = 0.01       # minimum fraction to include
    default_win_rate: float = 0.575     # from sweep results
    default_avg_win: float = 0.015      # avg winning trade return (1.5%)
    default_avg_loss: float = 0.012     # avg losing trade return (1.2%)


def kelly_fraction(p: float, b: float) -> float:
    """Compute full Kelly fraction: f* = (p*b - q) / b.

    Args:
        p: win probability (0, 1)
        b: payout ratio (avg_win / avg_loss), must be > 0

    Returns:
        Kelly fraction (may be negative = don't bet)
    """
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    q = 1.0 - p
    f = (p * b - q) / b
    return max(f, 0.0)


def half_kelly(p: float, b: float) -> float:
    """Half-Kelly fraction: f*/2 (safer, less volatile)."""
    return kelly_fraction(p, b) / 2.0


def compute_payout_ratio(avg_win: float, avg_loss: float) -> float:
    """Payout ratio b = avg_win / avg_loss. Must have avg_loss > 0."""
    if avg_loss <= 0:
        return 0.0
    return avg_win / avg_loss


def build_kelly_portfolio(
    recommendations: pd.DataFrame,
    config: KellyConfig | None = None,
) -> pd.DataFrame:
    """Build portfolio from backtest recommendations using Kelly sizing.

    Args:
        recommendations: DataFrame from backtest_recommendations() with columns
            ticker, signal_score, crash_prob, member (optional).
        config: KellyConfig with sizing parameters.

    Returns:
        DataFrame with columns: ticker, member, weight, kelly_fraction,
            signal_score, crash_prob, position_value.
    """
    if config is None:
        config = KellyConfig()

    if recommendations.empty:
        return _empty_portfolio()

    df = recommendations.copy()

    # Extract member info (may come from 'member' column or 'num_buyers')
    if "member" not in df.columns:
        df["member"] = "unknown"

    # Use crash_prob if available, else 0
    if "crash_prob" not in df.columns:
        df["crash_prob"] = 0.0

    # Use win_rate if available, else default
    if "win_rate" in df.columns:
        df["_win_rate"] = df["win_rate"].clip(0.01, 0.99)
    else:
        df["_win_rate"] = config.default_win_rate

    # Use avg_return or bt_return_pct for payout ratio
    if "avg_return_pct" in df.columns:
        avg_win = df.loc[df["avg_return_pct"] > 0, "avg_return_pct"].mean()
        avg_loss = abs(df.loc[df["avg_return_pct"] < 0, "avg_return_pct"].mean())
    elif "bt_return_pct" in df.columns:
        avg_win = df.loc[df["bt_return_pct"] > 0, "bt_return_pct"].mean()
        avg_loss = abs(df.loc[df["bt_return_pct"] < 0, "bt_return_pct"].mean())
    else:
        avg_win = config.default_avg_win * 100  # convert to pct
        avg_loss = config.default_avg_loss * 100

    if not avg_win or avg_win <= 0:
        avg_win = config.default_avg_win * 100
    if not avg_loss or avg_loss <= 0:
        avg_loss = config.default_avg_loss * 100

    payout_ratio = compute_payout_ratio(avg_win, avg_loss)

    # Compute Kelly fraction per ticker
    def _kelly(row):
        p = row["_win_rate"]
        base = half_kelly(p, payout_ratio) if config.use_half_kelly else kelly_fraction(p, payout_ratio)
        return base

    df["kelly_fraction"] = df.apply(_kelly, axis=1)

    # Apply crash guard: reduce position by crash_prob
    if config.crash_guard:
        df["kelly_fraction"] = df["kelly_fraction"] * (1.0 - df["crash_prob"].clip(0, 0.95))

    # Filter out zero/negative Kelly
    df = df[df["kelly_fraction"] > config.min_kelly_floor].copy()
    if df.empty:
        return _empty_portfolio()

    # Score-weighted allocation: weight by signal_score * kelly_fraction
    raw_weight = df["signal_score"] * df["kelly_fraction"]
    total_raw = raw_weight.sum()
    if total_raw <= 0:
        return _empty_portfolio()

    df["weight"] = raw_weight / total_raw

    # Apply risk constraints
    df = _apply_risk_constraints(df, config)

    # Scale to total exposure
    total_weight = df["weight"].sum()
    max_exposure = config.total_exposure_pct
    if total_weight > max_exposure:
        df["weight"] = df["weight"] * (max_exposure / total_weight)

    # Convert to dollar amounts
    df["position_value"] = df["weight"] * config.capital

    # Select output columns
    out_cols = ["ticker", "member", "weight", "kelly_fraction", "signal_score",
                 "crash_prob", "position_value"]
    available = [c for c in out_cols if c in df.columns]

    result = df[available].sort_values("position_value", ascending=False).reset_index(drop=True)
    return result


def _apply_risk_constraints(df: pd.DataFrame, config: KellyConfig) -> pd.DataFrame:
    """Enforce per-ticker and per-member position limits.

    Two-phase approach:
    1. Clip + normalize until stable (distributes freed weight).
    2. Final hard-clip pass (no normalization) to guarantee caps are met.
       Total weight may end up < 1.0 (cash buffer).
    """
    # Phase 1: iterative clip + normalize to distribute freed weight
    for _ in range(20):
        prev = df["weight"].copy()

        # Normalize
        total = df["weight"].sum()
        if total > 0:
            df["weight"] = df["weight"] / total

        # Per-ticker cap
        df["weight"] = df["weight"].clip(upper=config.max_ticker_pct)

        # Per-member cap
        if "member" in df.columns:
            for _member, grp in df.groupby("member"):
                total_member = grp["weight"].sum()
                if total_member > config.max_member_pct:
                    scale = config.max_member_pct / total_member
                    df.loc[grp.index, "weight"] = df.loc[grp.index, "weight"] * scale

        # Check convergence (weights unchanged after clip + normalize)
        if np.allclose(df["weight"].values, prev.values, atol=1e-8):
            break

    # Phase 2: final hard-clip to guarantee caps without re-normalizing
    df["weight"] = df["weight"].clip(upper=config.max_ticker_pct)
    if "member" in df.columns:
        for _member, grp in df.groupby("member"):
            total_member = grp["weight"].sum()
            if total_member > config.max_member_pct:
                scale = config.max_member_pct / total_member
                df.loc[grp.index, "weight"] = df.loc[grp.index, "weight"] * scale

    return df


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

    all_portfolios = []
    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)
        recs = backtest_recommendations(
            signals_df, transactions_df, as_of_ts,
            horizon=horizon,
            lookback_days=lookback_days,
            min_buyers=min_buyers,
            top_n=top_n,
            threshold=threshold,
            prices_df=prices_df,
            training_lookback_days=training_lookback_days,
        )
        if recs.empty:
            continue

        portfolio = build_kelly_portfolio(recs, config)
        if portfolio.empty:
            continue

        portfolio.insert(0, "as_of_date", as_of_ts.date())
        all_portfolios.append(portfolio)

    if not all_portfolios:
        return pd.DataFrame()
    return pd.concat(all_portfolios, ignore_index=True)


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
        portfolio_alpha, individual_returns (list).
    """
    if portfolio_df.empty or prices_df.empty:
        return pd.DataFrame()

    entry_mult = 1.0 + entry_slippage_bps / 10000
    exit_mult = 1.0 - exit_slippage_bps / 10000
    NS_PER_DAY = 86_400_000_000_000

    # Pre-extract SPY arrays
    from analyzer.signals import _price_arrays
    from analyzer.backtest import _price_at_or_before_arrays, _price_on_or_before_arrays
    spy_arrs = _price_arrays(prices_df, "SPY")

    results = []
    for as_of_date, group in portfolio_df.groupby("as_of_date"):
        as_of_ts = pd.Timestamp(as_of_date)

        weighted_return = 0.0
        spy_return = 0.0
        individual = []
        total_weight = 0.0

        for _, pos in group.iterrows():
            ticker = pos["ticker"]
            weight = pos["weight"]
            as_of_ns = as_of_ts.value
            exit_ns = as_of_ns + horizon * NS_PER_DAY

            tkr_arrs = _price_arrays(prices_df, ticker)
            if tkr_arrs is None or tkr_arrs[0] is None:
                continue
            idx_ns, vals = tkr_arrs

            # Entry price
            entry = _price_at_or_before_arrays(idx_ns, vals, as_of_ts, max_staleness_days=30)
            if not entry:
                continue

            # Exit price
            exit_price = _price_on_or_before_arrays(idx_ns, vals, pd.Timestamp(exit_ns), max_staleness_days=30)
            if not exit_price:
                continue

            entry_adj = entry * entry_mult
            exit_adj = exit_price * exit_mult
            ret = (exit_adj / entry_adj - 1.0) * 100

            weighted_return += weight * ret
            total_weight += weight
            individual.append({"ticker": ticker, "weight": weight, "return_pct": ret})

        # SPY benchmark
        if spy_arrs and spy_arrs[0] is not None:
            spy_ns, spy_vals = spy_arrs
            spy_entry = _price_at_or_before_arrays(spy_ns, spy_vals, as_of_ts, max_staleness_days=30)
            spy_exit = _price_on_or_before_arrays(spy_ns, spy_vals, pd.Timestamp(as_of_ns + horizon * NS_PER_DAY), max_staleness_days=30)
            if spy_entry and spy_exit:
                spy_return = (spy_exit * exit_mult / (spy_entry * entry_mult) - 1.0) * 100

        if total_weight > 0:
            weighted_return = weighted_return  # already weighted

        results.append({
            "as_of_date": as_of_date,
            "portfolio_return": round(weighted_return, 2),
            "spy_return": round(spy_return, 2),
            "portfolio_alpha": round(weighted_return - spy_return, 2),
            "num_positions": len(individual),
        })

    return pd.DataFrame(results)


def compute_portfolio_metrics(portfolio_returns: pd.DataFrame) -> dict:
    """Compute aggregate portfolio performance metrics from return series."""
    if portfolio_returns.empty:
        return {}

    rets = portfolio_returns["portfolio_return"].values
    spy_rets = portfolio_returns["spy_return"].values

    # Total cumulative return
    cumulative = np.prod(1 + rets / 100) - 1
    spy_cumulative = np.prod(1 + spy_rets / 100) - 1

    # Annualize (assuming ~30-day rebalance frequency)
    n_periods = len(rets)
    avg_days = 30  # frequency
    years = n_periods * avg_days / 365
    ann_return = ((1 + cumulative) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0.0

    # Sharpe ratio (annualized)
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(12))  # monthly-ish
    else:
        sharpe = 0.0

    # Max drawdown on cumulative equity curve
    equity = np.cumprod(1 + rets / 100)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = float(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0

    # Win rate
    win_rate = float(np.mean(rets > 0) * 100) if len(rets) > 0 else 0.0

    # Alpha
    avg_alpha = float(np.mean(portfolio_returns["portfolio_alpha"].values))

    # Number of periods with data
    total_positions = int(portfolio_returns["num_positions"].sum()) if "num_positions" in portfolio_returns.columns else 0

    return {
        "total_return_pct": round(cumulative * 100, 2),
        "annualized_return_pct": round(ann_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_alpha_per_period_pct": round(avg_alpha, 2),
        "spy_total_return_pct": round(spy_cumulative * 100, 2),
        "n_periods": n_periods,
        "total_positions": total_positions,
    }


def _empty_portfolio() -> pd.DataFrame:
    """Return empty DataFrame with expected columns."""
    return pd.DataFrame(columns=[
        "ticker", "member", "weight", "kelly_fraction",
        "signal_score", "crash_prob", "position_value",
    ])
