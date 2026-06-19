from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class SignalFeatures:
    ticker: str
    disclosure_date: date
    lag_days: int                     # disclosure_date - transaction_date
    pre_disclosure_return: float      # return from transaction_date to disclosure_date
    pre_disclosure_alpha: float       # alpha vs SPY over same period
    max_drawdown_to_entry: float      # max drawdown from disclosure to entry
    volatility_20d: float             # 20-day realized vol at entry
    drawdown_from_ath: float          # drawdown from all-time high
    days_since_ipo: int | None        # approximate IPO age
    n_buyers_30d: int                 # buyers in last 30 days


@dataclass
class CrashHazard:
    crash_prob: float     # probability of >20% drawdown in next 120d
    expected_return: float
    var_95: float         # 5th percentile return
    cvar_95: float        # conditional VaR at 5%


def compute_disclosure_lag_weight(
    lag_days: int,
    half_life: float = 60.0,
) -> float:
    """Exponential decay weight for disclosure lag.
    weight = exp(-lag_days * ln(2) / half_life)
    """
    return math.exp(-lag_days * math.log(2) / half_life)


def _realized_vol(prices: pd.Series, window: int = 20) -> float:
    """Compute annualized realized volatility from a price series."""
    if len(prices) < window + 1:
        return 0.0
    returns = prices.pct_change().dropna().tail(window)
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * math.sqrt(252))


def _max_drawdown(prices: pd.Series) -> float:
    """Compute max drawdown from peak as a positive fraction."""
    if prices.empty:
        return 0.0
    cummax = prices.cummax()
    drawdowns = (prices - cummax) / cummax
    return float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0


def _drawdown_from_ath(prices: pd.Series) -> float:
    """Compute current drawdown from all-time high as a positive fraction."""
    if prices.empty:
        return 0.0
    ath = prices.max()
    if ath <= 0:
        return 0.0
    return float((ath - prices.iloc[-1]) / ath)


def compute_signal_features(
    ticker: str,
    disclosure_date: date,
    transaction_date: date | None,
    prices_df: pd.DataFrame,
    all_tx: pd.DataFrame,
    as_of_date: date,
) -> SignalFeatures:
    """Compute point-in-time features for a signal."""
    disc_ts = pd.Timestamp(disclosure_date)
    as_of_ts = pd.Timestamp(as_of_date)

    # Lag days
    if transaction_date is not None:
        lag_days = (disc_ts - pd.Timestamp(transaction_date)).days
    else:
        lag_days = 0

    # Get price series for this ticker up to as_of_date
    if ticker in prices_df.columns:
        ticker_prices = prices_df[ticker].dropna()
        prices_before_asof = ticker_prices[ticker_prices.index <= as_of_ts]
    else:
        prices_before_asof = pd.Series(dtype=float)

    # Pre-disclosure return
    pre_disclosure_return = 0.0
    pre_disclosure_alpha = 0.0
    if transaction_date is not None and not prices_before_asof.empty:
        tx_ts = pd.Timestamp(transaction_date)
        entry_price = prices_before_asof[prices_before_asof.index >= tx_ts]
        disclosure_price = prices_before_asof[prices_before_asof.index >= disc_ts]
        if len(entry_price) > 0 and len(disclosure_price) > 0:
            p_tx = float(entry_price.iloc[0])
            p_disc = float(disclosure_price.iloc[0])
            if p_tx > 0:
                pre_disclosure_return = (p_disc / p_tx) - 1.0

                # SPY alpha
                if "SPY" in prices_df.columns:
                    spy_prices = prices_df["SPY"].dropna()
                    spy_before = spy_prices[spy_prices.index <= as_of_ts]
                    spy_tx = spy_before[spy_before.index >= tx_ts]
                    spy_disc = spy_before[spy_before.index >= disc_ts]
                    if len(spy_tx) > 0 and len(spy_disc) > 0:
                        spy_p_tx = float(spy_tx.iloc[0])
                        spy_p_disc = float(spy_disc.iloc[0])
                        if spy_p_tx > 0:
                            spy_return = (spy_p_disc / spy_p_tx) - 1.0
                            pre_disclosure_alpha = pre_disclosure_return - spy_return

    # Max drawdown from disclosure to entry (using as_of as entry proxy)
    max_dd_to_entry = 0.0
    if not prices_before_asof.empty:
        post_disc = prices_before_asof[prices_before_asof.index >= disc_ts]
        if len(post_disc) >= 2:
            max_dd_to_entry = _max_drawdown(post_disc)

    # 20-day realized vol at entry (as of date)
    volatility_20d = _realized_vol(prices_before_asof, window=20)

    # Drawdown from ATH
    drawdown_ath = _drawdown_from_ath(prices_before_asof)

    # Days since IPO (approximate: first available price date)
    days_since_ipo = None
    if not prices_before_asof.empty:
        first_date = prices_before_asof.index.min()
        days_since_ipo = (as_of_ts - first_date).days

    # Number of buyers in last 30 days
    n_buyers_30d = 0
    if not all_tx.empty and "ticker" in all_tx.columns and "member" in all_tx.columns:
        lookback = as_of_ts - pd.Timedelta(days=30)
        recent_tx = all_tx[
            (all_tx["ticker"] == ticker)
            & (all_tx["disclosure_date"] >= lookback)
            & (all_tx["disclosure_date"] <= as_of_ts)
            & (all_tx["transaction_type"] == "Purchase")
        ]
        if not recent_tx.empty:
            n_buyers_30d = int(recent_tx["member"].nunique())

    return SignalFeatures(
        ticker=ticker,
        disclosure_date=disclosure_date,
        lag_days=lag_days,
        pre_disclosure_return=pre_disclosure_return,
        pre_disclosure_alpha=pre_disclosure_alpha,
        max_drawdown_to_entry=max_dd_to_entry,
        volatility_20d=volatility_20d,
        drawdown_from_ath=drawdown_ath,
        days_since_ipo=days_since_ipo,
        n_buyers_30d=n_buyers_30d,
    )


def estimate_crash_hazard(
    features: SignalFeatures,
    historical_crash_rate: float = 0.10,
) -> CrashHazard:
    """Estimate left-tail risk for a ticker at entry.

    Uses logistic model:
    log(p/(1-p)) = a + b1*vol + b2*drawdown + b3*lag + b4*newness

    Calibrated coefficients (can be tuned):
    - High vol -> higher crash prob
    - Large drawdown from ATH -> higher crash prob (falling knife)
    - Long disclosure lag -> higher crash prob (stale information)
    - Short IPO age -> higher crash prob
    """
    # Log-odds of base crash rate
    base_logit = math.log(historical_crash_rate / (1 - historical_crash_rate))

    # Coefficients
    b_vol = 1.5          # high vol increases crash risk
    b_drawdown = 1.0     # falling knife effect
    b_lag = 0.01         # each day of lag adds risk
    b_newness = -0.005   # fewer days since IPO -> higher risk (negative coef, fewer days = more risk)

    # Normalize features
    vol_feature = features.volatility_20d
    dd_feature = features.drawdown_from_ath
    lag_feature = float(features.lag_days)

    # IPO age: use a proxy if None (assume very new = 30 days)
    ipo_days = features.days_since_ipo if features.days_since_ipo is not None else 30

    logit_p = base_logit + b_vol * vol_feature + b_drawdown * dd_feature + b_lag * lag_feature + b_newness * ipo_days

    # Clamp to avoid overflow
    logit_p = max(min(logit_p, 10.0), -10.0)
    crash_prob = 1.0 / (1.0 + math.exp(-logit_p))

    # Expected return (simple model: negative when crash prob is high)
    expected_return = -0.10 * crash_prob

    # VaR and CVaR (parametric normal approximation)
    # Use vol to scale the tail
    vol = max(features.volatility_20d, 0.05)  # floor at 5%
    var_95 = expected_return - 1.645 * (vol / math.sqrt(252)) * math.sqrt(120)  # 120d horizon, sqrt(T) scaling
    cvar_95 = var_95 * 1.25  # CVaR is worse than VaR

    return CrashHazard(
        crash_prob=round(crash_prob, 4),
        expected_return=round(expected_return, 4),
        var_95=round(var_95, 4),
        cvar_95=round(cvar_95, 4),
    )
