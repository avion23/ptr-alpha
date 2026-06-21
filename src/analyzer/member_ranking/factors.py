"""Score factor helpers used by ticker scoring.

Three small multipliers (size, owner-code, conviction) that adjust the raw
Bayesian-shrunk alpha before producing a final ticker signal score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _size_score_factor(trades: pd.DataFrame) -> float:
    if "amount_midpoint" not in trades.columns:
        return 1.0
    amount = trades["amount_midpoint"].dropna()
    if amount.empty:
        return 1.0
    average_amount = max(float(amount.mean()), 1.0)
    adjustment = np.log10(average_amount / 10000.0) * 0.025
    adjustment = float(np.clip(adjustment, -0.15, 0.15))
    return 1.0 + adjustment


def _owner_score_factor(trades: pd.DataFrame) -> float:
    if "owner_code" not in trades.columns:
        return 1.0
    owner_codes = trades["owner_code"].fillna("").astype(str).str.upper()
    if owner_codes.empty:
        return 1.0
    dependent_child_ratio = (owner_codes == "DC").mean()
    return 1.0 - dependent_child_ratio * 0.15


def _conviction_score(trades: pd.DataFrame) -> float:
    trade_count = len(trades)
    if trade_count == 0:
        return 0.0
    count_score = min(trade_count / 10.0, 1.0)
    has_amounts = "amount_midpoint" in trades.columns and trades["amount_midpoint"].notna().any()
    size_score = 1.0
    if has_amounts:
        avg_amount = trades["amount_midpoint"].dropna().mean()
        size_score = min(avg_amount / 50000.0, 1.0)
    return count_score * 0.6 + size_score * 0.4
