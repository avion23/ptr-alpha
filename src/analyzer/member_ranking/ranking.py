"""Member ranking: vectorized aggregation pipeline.

`rank_members` builds a per-member ranking DataFrame from a purchase
signals DataFrame. The expensive `_prepare_member_data` step is memoized
separately from the prior-strength-dependent `_rank_members_impl` so that
parameter sweeps hit cache across most prior-strength values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer import signals as _signals
from analyzer._memo import df_memoize
from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.signals import (
    _apply_quality_filter,
    _collapse_to_episodes,
    _get_horizon_data,
)


def rank_members(
    signal_df: pd.DataFrame,
    horizon: int = 90,
    threshold: float = 5.0,
    _bayes_prior_strength: float | None = None,
) -> pd.DataFrame:
    """Rank members by historical purchase performance."""
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")

    bayes_prior = (
        _bayes_prior_strength
        if _bayes_prior_strength is not None
        else _signals.BAYES_PRIOR_STRENGTH
    )

    return _rank_members_impl(signal_df, horizon, threshold, bayes_prior)


@df_memoize(copy=False)
def _prepare_member_data(
    signal_df: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    """Prepare collapsed purchases (prior-strength-independent).

    This is the expensive part of _rank_members_impl that doesn't depend on
    bayes_prior_strength. Extracting it allows memoization to hit cache for
    2/3 of combos (all but the bayes_prior dimension change).
    """
    purchases = _get_horizon_data(signal_df, horizon, TransactionType.PURCHASE.value)
    if purchases.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    purchases = _apply_quality_filter(purchases)
    if purchases.empty:
        raise AnalysisError(
            f"No signals survived quality filter (min price ${_signals.MIN_ENTRY_PRICE})"
        )

    return _collapse_to_episodes(purchases)


@df_memoize(copy=False)
def _rank_members_impl(
    signal_df: pd.DataFrame,
    horizon: int,
    threshold: float,
    _bayes_prior_strength: float,
) -> pd.DataFrame:
    purchases = _prepare_member_data(signal_df, horizon, threshold)

    alpha_col = (
        "total_spy_alpha_pct"
        if "total_spy_alpha_pct" in purchases.columns
        else "spy_alpha_pct"
    )
    prior_alpha_mean = float(purchases[alpha_col].mean())
    if pd.isna(prior_alpha_mean):
        prior_alpha_mean = 0.0

    grp = purchases.groupby("member")
    ret_agg = _aggregate_returns(grp)
    if ret_agg.empty:
        return pd.DataFrame()

    idx = ret_agg.index
    n = ret_agg["ret_nonnan"].astype(int)
    wins = _wins_by_member(purchases, idx)

    total_n = int(n.sum())
    total_wins = int(wins.sum())
    global_prior = float(np.clip(total_wins / total_n, 0.10, 0.90))
    peer_n = total_n - n
    peer_wins = total_wins - wins
    loo_priors = pd.Series(global_prior, index=idx, dtype=float)
    has_peer_observations = peer_n > 0
    loo_priors.loc[has_peer_observations] = np.clip(
        peer_wins.loc[has_peer_observations] / peer_n.loc[has_peer_observations],
        0.10,
        0.90,
    )

    stats = _compute_bayes_stats(n, wins, loo_priors, _bayes_prior_strength, ret_agg)
    avg_spy, avg_total_spy = _spy_alpha_by_member(purchases, grp, idx)
    hit_rates = _hit_rates_by_member(purchases, idx, threshold)
    conviction = _conviction_scores(grp, idx, purchases)
    shrunk_alpha = _shrunk_alpha_by_member(
        grp, alpha_col, idx, prior_alpha_mean, _bayes_prior_strength
    )

    avg_realized = (
        grp["total_return_pct"].mean().reindex(idx).fillna(0.0)
        if "total_return_pct" in purchases.columns
        else pd.Series(0.0, index=idx)
    )

    result = _build_ranking_result(
        idx,
        ret_agg,
        stats,
        avg_spy,
        avg_total_spy,
        hit_rates,
        conviction,
        shrunk_alpha,
        avg_realized,
    )
    return _finalize_ranking(result)


def _aggregate_returns(grp) -> pd.DataFrame:
    ret_agg = grp["decayed_return_pct"].agg(
        ret_nonnan="count",
        median_ret="median",
        mean_ret="mean",
        std_ret="std",
    )
    ret_agg = ret_agg[ret_agg["ret_nonnan"] > 0]
    if not ret_agg.empty:
        ret_agg["std_ret"] = ret_agg["std_ret"].fillna(0.0)
    return ret_agg


def _wins_by_member(purchases: pd.DataFrame, idx) -> pd.Series:
    return (
        (purchases["decayed_return_pct"] > 0)
        .groupby(purchases["member"])
        .sum()
        .reindex(idx, fill_value=0)
        .astype(int)
    )


def _compute_bayes_stats(
    n, wins, loo_priors, prior_strength: float, ret_agg: pd.DataFrame
):
    prior_values = loo_priors.to_numpy(dtype=float)
    bayes_alpha = prior_values * prior_strength
    bayes_beta = (1 - prior_values) * prior_strength
    n_vals = n.values.astype(float)
    wins_f = wins.values.astype(float)
    bayes_win_prob = (bayes_alpha + wins_f) / (bayes_alpha + bayes_beta + n_vals)
    posterior_lift = bayes_win_prob / prior_values
    sharpe = np.where(
        ret_agg["std_ret"] > 0, ret_agg["mean_ret"] / ret_agg["std_ret"], 0.0
    )
    return {
        "sharpe": sharpe,
        "prob_up": wins_f / n_vals,
        "bayes_win_prob": bayes_win_prob,
        "posterior_lift": posterior_lift,
    }


def _spy_alpha_by_member(purchases: pd.DataFrame, grp, idx):
    avg_spy = grp["spy_alpha_pct"].mean().reindex(idx).fillna(0.0)
    if "total_spy_alpha_pct" in purchases.columns:
        avg_total_spy = grp["total_spy_alpha_pct"].mean().reindex(idx)
        avg_total_spy = avg_total_spy.fillna(avg_spy)
    else:
        avg_total_spy = avg_spy.copy()
    return avg_spy, avg_total_spy


def _hit_rates_by_member(purchases: pd.DataFrame, idx, threshold: float | None):
    """Returns (peak_hits_series, realized_hits_series_or_None), or None when
    threshold is None. realized_hits is None when total_return_pct isn't
    available, so callers can distinguish the two."""
    if threshold is None:
        return None
    # Bug #3: (NaN > threshold) evaluates to False in pandas, so NaN rows
    # were counted as misses in both numerator and denominator.  Restrict to
    # non-NaN rows before computing per-member means.
    valid_peak = purchases[purchases["peak_potential_pct"].notna()]
    peak_hits = (valid_peak["peak_potential_pct"] > threshold).groupby(
        valid_peak["member"]
    ).mean().reindex(idx) * 100
    realized_hits = None
    if "total_return_pct" in purchases.columns:
        # Bug #3: same NaN-as-miss problem for realized returns.
        valid_ret = purchases[purchases["total_return_pct"].notna()]
        realized_hits = (valid_ret["total_return_pct"] > 0).groupby(
            valid_ret["member"]
        ).mean().reindex(idx) * 100
    return peak_hits, realized_hits


def _conviction_scores(grp, idx, purchases: pd.DataFrame) -> np.ndarray:
    group_sizes = grp.size().reindex(idx)
    count_scores = np.minimum(group_sizes.values / 10.0, 1.0)
    if "amount_midpoint" not in purchases.columns:
        size_scores = np.ones(len(idx))
    else:
        avg_amounts = grp["amount_midpoint"].mean().reindex(idx)
        amount_has_data = (grp["amount_midpoint"].count().reindex(idx) > 0).values
        size_scores = np.where(
            amount_has_data,
            np.minimum(avg_amounts.fillna(0.0).values / 50000.0, 1.0),
            1.0,
        )
    return count_scores * 0.6 + size_scores * 0.4


def _shrunk_alpha_by_member(
    grp, alpha_col: str, idx, prior_alpha_mean: float, prior_strength: float
):
    alpha_sums = grp[alpha_col].sum().reindex(idx).fillna(0.0)
    alpha_counts = grp[alpha_col].count().reindex(idx).fillna(0).astype(int)
    return (prior_alpha_mean * prior_strength + alpha_sums) / (
        prior_strength + alpha_counts
    )


def _build_ranking_result(
    idx,
    ret_agg,
    stats,
    avg_spy,
    avg_total_spy,
    hit_rates,
    conviction,
    shrunk_alpha,
    avg_realized,
):
    result = pd.DataFrame(
        {
            "member": idx,
            "median_return_pct": np.round(ret_agg["median_ret"].values, 2),
            "mean_return_pct": np.round(ret_agg["mean_ret"].values, 2),
            "trades": ret_agg["ret_nonnan"].astype(int).values,
            "sharpe_ratio": np.round(stats["sharpe"], 3),
            "prob_up": np.round(stats["prob_up"], 3),
            "bayes_win_prob": np.round(stats["bayes_win_prob"], 3),
            "posterior_lift": np.round(stats["posterior_lift"], 3),
            "avg_return_pct": np.round(avg_realized.values, 2),
            "avg_spy_alpha_pct": np.round(avg_spy.values, 2),
            "avg_total_spy_alpha_pct": np.round(avg_total_spy.values, 2),
        }
    )
    if hit_rates is not None:
        peak_hits, realized_hits = hit_rates
        result["peak_hit_rate_pct"] = np.round(peak_hits.values, 2)
        if realized_hits is not None:
            result["realized_hit_rate_pct"] = np.round(realized_hits.values, 2)
    result["conviction_score"] = np.round(conviction, 3)
    result["shrunk_alpha"] = shrunk_alpha.values
    return result


def _finalize_ranking(result: pd.DataFrame) -> pd.DataFrame:
    return result.rename(
        columns={
            "mean_return_pct": "avg_decay_return_pct",
            "median_return_pct": "median_decay_return_pct",
            "trades": "purchase_trades",
            "prob_up": "prob_up_given_buy",
        }
    ).sort_values("shrunk_alpha", ascending=False)
