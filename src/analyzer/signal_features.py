from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from analyzer._memo import df_memoize
from analyzer.models import TransactionType


def compute_optimal_entry(
    prices_df: pd.DataFrame,
    ticker: str,
    disclosure_date: date,
    pullback_pct: float = 0.05,
    max_wait_days: int = 10,
) -> tuple[float, int, bool]:
    """Compute optimal entry price after disclosure.

    Strategy: wait for `pullback_pct` dip from disclosure price within `max_wait_days`.
    If dip occurs, enter at the dip price. Otherwise enter at disclosure price.

    Returns (entry_price, entry_delay_days, is_dip_entry).
    """
    disc_ts = pd.Timestamp(disclosure_date)

    if ticker not in prices_df.columns:
        return 0.0, 0, False

    ticker_prices = prices_df[ticker].dropna()

    # Get disclosure price (first price on or after disclosure)
    post_disc = ticker_prices[ticker_prices.index >= disc_ts]
    if post_disc.empty:
        return 0.0, 0, False

    disc_price = float(post_disc.iloc[0])
    if disc_price <= 0:
        return 0.0, 0, False

    # Look for pullback within max_wait_days
    window_end = disc_ts + pd.Timedelta(days=max_wait_days)
    window = ticker_prices[(ticker_prices.index >= disc_ts) & (ticker_prices.index <= window_end)]

    target_price = disc_price * (1 - pullback_pct)

    window_vals = window.values
    hits = np.where(window_vals <= target_price)[0]
    if len(hits) > 0:
        return float(window_vals[hits[0]]), int(hits[0]), True

    # No pullback found — enter at disclosure price
    return disc_price, 0, False


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


@df_memoize
def compute_signal_features(
    ticker: str,
    disclosure_date: date,
    transaction_date: date | None,
    prices_df: pd.DataFrame,
    all_tx: pd.DataFrame,
    as_of_date: date,
) -> SignalFeatures:
    """Compute point-in-time features for a signal.

    Uses numpy arrays via ``_price_arrays`` for O(log N) price lookups
    instead of pandas Series boolean masking.
    """
    from analyzer.signals import _price_arrays

    disc_ts = pd.Timestamp(disclosure_date)
    as_of_ts = pd.Timestamp(as_of_date)
    as_of_ns = as_of_ts.value

    # Lag days
    if transaction_date is not None:
        lag_days = (disc_ts - pd.Timestamp(transaction_date)).days
    else:
        lag_days = 0

    # Get price arrays (numpy) for this ticker — O(1) lookup via cache
    arrs = _price_arrays(prices_df, ticker)
    idx_ns = None
    vals = None
    if arrs is not None:
        idx_ns, vals = arrs

    # Pre-disclosure return
    pre_disclosure_return = 0.0
    pre_disclosure_alpha = 0.0
    if transaction_date is not None and idx_ns is not None and len(idx_ns) > 0:
        tx_ns = pd.Timestamp(transaction_date).value
        disc_ns = disc_ts.value

        # Price at transaction date (first price >= tx_ts AND <= as_of_ts)
        pos_tx = int(np.searchsorted(idx_ns, tx_ns, side="left"))
        if pos_tx < len(idx_ns) and idx_ns[pos_tx] <= as_of_ns:
            p_tx = float(vals[pos_tx])
            # Price at disclosure date (first price >= disc_ts AND <= as_of_ts)
            pos_disc = int(np.searchsorted(idx_ns, disc_ns, side="left"))
            if pos_disc < len(idx_ns) and idx_ns[pos_disc] <= as_of_ns:
                p_disc = float(vals[pos_disc])
                if p_tx > 0:
                    pre_disclosure_return = (p_disc / p_tx) - 1.0

                    # SPY alpha
                    spy_arrs = _price_arrays(prices_df, "SPY")
                    if spy_arrs is not None and spy_arrs[0] is not None:
                        spy_ns, spy_vals = spy_arrs
                        # SPY at transaction date
                        sp = int(np.searchsorted(spy_ns, tx_ns, side="left"))
                        if sp < len(spy_ns) and spy_ns[sp] <= as_of_ns:
                            spy_p_tx = float(spy_vals[sp])
                            # SPY at disclosure date
                            sd = int(np.searchsorted(spy_ns, disc_ns, side="left"))
                            if sd < len(spy_ns) and spy_ns[sd] <= as_of_ns:
                                spy_p_disc = float(spy_vals[sd])
                                if spy_p_tx > 0:
                                    spy_return = (spy_p_disc / spy_p_tx) - 1.0
                                    pre_disclosure_alpha = pre_disclosure_return - spy_return

    # Max drawdown from disclosure to entry (using as_of as entry proxy)
    max_dd_to_entry = 0.0
    if idx_ns is not None and len(idx_ns) > 0:
        disc_ns = disc_ts.value
        lo = int(np.searchsorted(idx_ns, disc_ns, side="left"))
        hi = int(np.searchsorted(idx_ns, as_of_ns, side="right"))
        post_disc_vals = vals[lo:hi]
        if len(post_disc_vals) >= 2:
            cummax = np.maximum.accumulate(post_disc_vals)
            drawdowns = (post_disc_vals - cummax) / cummax
            max_dd_to_entry = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0

    # 20-day realized vol at entry (as of date)
    volatility_20d = 0.0
    if idx_ns is not None and len(idx_ns) > 1:
        # Get last ~20 trading days up to as_of_ns
        pos = int(np.searchsorted(idx_ns, as_of_ns, side="right"))
        start = max(0, pos - 21)
        window_vals = vals[start:pos]
        if len(window_vals) >= 3:
            returns = np.diff(window_vals) / window_vals[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) >= 2:
                volatility_20d = float(np.std(returns, ddof=1) * np.sqrt(252))

    # Drawdown from ATH
    drawdown_ath = 0.0
    if idx_ns is not None and len(idx_ns) > 0:
        pos = int(np.searchsorted(idx_ns, as_of_ns, side="right"))
        hist_vals = vals[:pos]
        if len(hist_vals) > 0:
            ath = float(hist_vals.max())
            if ath > 0:
                drawdown_ath = float((ath - hist_vals[-1]) / ath)

    # Days since IPO (approximate: first available price date)
    days_since_ipo = None
    if idx_ns is not None and len(idx_ns) > 0:
        first_date = pd.Timestamp(idx_ns[0], unit="ns")
        days_since_ipo = (as_of_ts - first_date).days

    # Number of buyers in last 30 days
    n_buyers_30d = 0
    if not all_tx.empty and "ticker" in all_tx.columns and "member" in all_tx.columns:
        lookback = as_of_ts - pd.Timedelta(days=30)
        recent_tx = all_tx[
            (all_tx["ticker"] == ticker)
            & (all_tx["disclosure_date"] >= lookback)
            & (all_tx["disclosure_date"] <= as_of_ts)
            & (all_tx["transaction_type"] == TransactionType.PURCHASE.value)
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
