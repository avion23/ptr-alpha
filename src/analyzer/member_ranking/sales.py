"""Sale-signal ranking: rank members by loss-avoidance.

`rank_sales` collapses sale signals into per-member loss-avoidance stats.
The path differs from `rank_members` (purchase side) in that sale returns
are inverted (good member = small loss or gain), and the implementation
uses a per-group row-by-row `_compute_member_stats` loop rather than the
vectorized path. Both stats paths share the same scalar fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.signals import (
    _collapse_to_episodes,
    _get_horizon_data,
)

from analyzer.member_ranking.bayes import bayes_factor_against_market, bayesian_win_probability


def _compute_member_stats(
    member: str,
    grp: pd.DataFrame,
    market_prior: float,
    threshold: float | None = None,
    invert_returns: bool = False,
) -> dict | None:
    rets = grp["decayed_return_pct"].dropna().values
    if len(rets) == 0:
        return None
    if invert_returns:
        rets = -rets

    median_ret = float(np.median(rets))
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0

    wins = int((rets > 0).sum())
    losses = int(len(rets) - wins)
    p_up = wins / len(rets)
    bayes_win_prob = bayesian_win_probability(wins, losses, market_prior)
    posterior_lift = bayes_win_prob / market_prior
    bayes_factor = bayes_factor_against_market(wins, losses, market_prior)

    avg_spy_alpha, avg_total_spy_alpha = _alpha_stats(grp, invert_returns)

    stats = {
        "member": member,
        "median_return_pct": round(median_ret, 2),
        "mean_return_pct": round(mean_ret, 2),
        "trades": len(rets),
        "sharpe_ratio": round(sharpe, 3),
        "prob_up": round(p_up, 3),
        "bayes_win_prob": round(bayes_win_prob, 3),
        "bayes_factor": round(bayes_factor, 3),
        "posterior_lift": round(posterior_lift, 3),
        "avg_spy_alpha_pct": round(avg_spy_alpha, 2),
        "avg_total_spy_alpha_pct": round(avg_total_spy_alpha, 2),
    }
    if threshold is not None:
        # Bug #3: (NaN > threshold) evaluates to False, making NaN rows count
        # as misses in both numerator and denominator.  Exclude NaN rows first.
        valid_peak = grp["peak_potential_pct"].dropna()
        stats["peak_hit_rate_pct"] = (
            round((valid_peak > threshold).mean() * 100, 2) if len(valid_peak) > 0 else float("nan")
        )
        if "total_return_pct" in grp.columns:
            # Bug #3: same NaN-as-miss pattern for realized returns.
            valid_ret = grp["total_return_pct"].dropna()
            stats["realized_hit_rate_pct"] = (
                round((valid_ret > 0).mean() * 100, 2) if len(valid_ret) > 0 else float("nan")
            )
    return stats


def _alpha_stats(grp: pd.DataFrame, invert_returns: bool) -> tuple[float, float]:
    """Average per-member SPY alpha (with optional sign flip for sales)."""
    spy_alpha_vals = grp["spy_alpha_pct"].dropna().values
    if invert_returns:
        spy_alpha_vals = -spy_alpha_vals
    avg_spy_alpha = float(np.mean(spy_alpha_vals)) if len(spy_alpha_vals) > 0 else 0.0

    total_spy_alpha_vals = (
        grp["total_spy_alpha_pct"].dropna().values
        if "total_spy_alpha_pct" in grp.columns
        else np.array([])
    )
    if invert_returns:
        total_spy_alpha_vals = -total_spy_alpha_vals
    avg_total_spy_alpha = (
        float(np.mean(total_spy_alpha_vals))
        if len(total_spy_alpha_vals) > 0
        else avg_spy_alpha
    )
    return avg_spy_alpha, avg_total_spy_alpha


def rank_sales(signal_df: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    sales = _get_horizon_data(signal_df, horizon, TransactionType.SALE.value)
    if sales.empty:
        raise AnalysisError(f"No sale signals found for horizon {horizon}")

    sales = _collapse_to_episodes(sales)
    valid_sale_returns = sales["decayed_return_pct"].dropna()
    sale_prior = (
        float(np.clip((valid_sale_returns < 0).mean(), 0.10, 0.90))
        if len(valid_sale_returns) > 0
        else 0.50
    )

    member_stats = []
    for member, sale_grp in sales.groupby("member"):
        row = _compute_member_stats(member, sale_grp, sale_prior, invert_returns=True)
        if row is not None:
            member_stats.append(row)

    result = pd.DataFrame(member_stats)
    if result.empty:
        return result

    return result.rename(columns={
        "mean_return_pct": "avg_loss_avoided_pct",
        "median_return_pct": "median_loss_avoided_pct",
        "trades": "sale_trades",
        "prob_up": "prob_up_given_sell",
    }).sort_values("avg_spy_alpha_pct", ascending=False)
