"""Per-member decay-lambda estimation.

Members who exit positions quickly (short holding periods) need a higher
decay lambda for backtesting — their positions decay faster. We estimate
lambda from the ratio of decayed_return to total_return: ratio=1 means the
member holds long (default lambda), ratio→0 means short holds (higher
lambda).
"""

from __future__ import annotations

import pandas as pd

from analyzer._memo import df_memoize
from analyzer.models import TransactionType


def estimate_member_decay_lambda(
    member: str,
    signals_df: pd.DataFrame,
    horizon: int = 90,
    default_lambda: float = 0.005,
    min_trades: int = 3,
) -> float:
    """Estimate per-member decay lambda from historical holding periods.

    Members who exit quickly (short holding periods) get higher lambda.
    Members who hold long get lower lambda.

    Uses the ratio of decayed_return to total_return as a proxy for
    optimal holding period, then adjusts lambda accordingly.
    """
    member_signals = signals_df[
        (signals_df["member"] == member)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]

    if len(member_signals) < min_trades:
        return default_lambda

    has_decayed = "decayed_return_pct" in member_signals.columns
    has_total = "total_return_pct" in member_signals.columns

    if has_decayed and has_total:
        decayed = member_signals["decayed_return_pct"].dropna()
        total = member_signals["total_return_pct"].dropna()
        if len(decayed) > 0 and len(total) > 0:
            ratio = abs(decayed.mean()) / max(abs(total.mean()), 1e-6)
            # ratio=1 → lambda = default (long hold), ratio→0 → higher lambda (short hold)
            member_lambda = default_lambda * (2.0 - max(0.1, min(2.0, ratio)))
            return float(member_lambda)

    return default_lambda


@df_memoize(copy=False)
def get_member_decay_map(
    signals_df: pd.DataFrame,
    horizon: int = 90,
    default_lambda: float = 0.005,
    min_trades: int = 3,
) -> dict[str, float]:
    """Get decay lambda for all members with sufficient data.

    Returns {member: lambda} for members with >= min_trades trades.
    Members not in the map use default_lambda.
    """
    members = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]["member"].unique()

    result: dict[str, float] = {}
    for member in members:
        lam = estimate_member_decay_lambda(
            member, signals_df, horizon, default_lambda, min_trades,
        )
        if abs(lam - default_lambda) > 1e-6:
            result[member] = lam

    return result
