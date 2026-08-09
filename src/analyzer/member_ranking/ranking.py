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

    outcome_col = "total_spy_alpha_pct"
    if outcome_col not in purchases.columns:
        raise AnalysisError(
            "Member ranking requires endpoint SPY alpha; total_spy_alpha_pct is missing"
        )
    purchases = purchases[purchases[outcome_col].notna()].copy()
    if purchases.empty:
        raise AnalysisError("No complete endpoint SPY-alpha outcomes found")

    grp = purchases.groupby("member")
    ret_agg = _aggregate_returns(grp, outcome_col)
    if ret_agg.empty:
        return pd.DataFrame()

    idx = ret_agg.index
    n = ret_agg["ret_nonnan"].astype(int)
    wins = _wins_by_member(purchases, idx, outcome_col)

    # One common empirical prior is estimated from the training frame supplied
    # by the caller. Every member is compared against the same reference
    # population; complementary leave-one-member-out priors can reverse perfect
    # and zero-win records and are not posterior probabilities on one scale.
    total_n = int(n.sum())
    total_wins = int(wins.sum())
    common_prior = float(np.clip(total_wins / total_n, 0.10, 0.90))

    stats = _compute_bayes_stats(n, wins, common_prior, _bayes_prior_strength, ret_agg)
    avg_spy, avg_total_spy = _spy_alpha_by_member(purchases, grp, idx)
    hit_rates = _hit_rates_by_member(purchases, idx, threshold)
    conviction = _conviction_scores(grp, idx, purchases)
    shrunk_alpha, shrunk_alpha_std, alpha_shrinkage = _shrunk_alpha_by_member(
        grp, outcome_col, idx
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
        shrunk_alpha_std,
        alpha_shrinkage,
        avg_realized,
    )
    return _finalize_ranking(result)


def _aggregate_returns(grp, outcome_col: str) -> pd.DataFrame:
    diagnostic_col = (
        "decayed_return_pct" if "decayed_return_pct" in grp.obj.columns else outcome_col
    )
    ret_agg = grp[diagnostic_col].agg(
        ret_nonnan="count",
        median_ret="median",
        mean_ret="mean",
        std_ret="std",
    )
    ret_agg["ret_nonnan"] = grp[outcome_col].count().reindex(ret_agg.index)
    ret_agg = ret_agg[ret_agg["ret_nonnan"] > 0]
    if not ret_agg.empty:
        ret_agg["std_ret"] = ret_agg["std_ret"].fillna(0.0)
    return ret_agg


def _wins_by_member(purchases: pd.DataFrame, idx, outcome_col: str) -> pd.Series:
    """Count profitable endpoint excess-alpha episodes per member."""
    return (
        (purchases[outcome_col] > 0)
        .groupby(purchases["member"])
        .sum()
        .reindex(idx, fill_value=0)
        .astype(int)
    )


def _compute_bayes_stats(
    n, wins, common_prior: float, prior_strength: float, ret_agg: pd.DataFrame
):
    bayes_alpha = common_prior * prior_strength
    bayes_beta = (1 - common_prior) * prior_strength
    n_vals = n.values.astype(float)
    wins_f = wins.values.astype(float)
    bayes_win_prob = (bayes_alpha + wins_f) / (bayes_alpha + bayes_beta + n_vals)
    sharpe = np.where(
        ret_agg["std_ret"] > 0, ret_agg["mean_ret"] / ret_agg["std_ret"], 0.0
    )
    return {
        "sharpe": sharpe,
        "prob_up": wins_f / n_vals,
        "bayes_win_prob": bayes_win_prob,
        "prior_win_prob": np.full(len(n_vals), common_prior),
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


def _shrunk_alpha_by_member(grp, alpha_col: str, idx):
    """Return a descriptive empirical-Bayes normal-normal member estimate.

    The outcome is endpoint SPY alpha. A common Normal(global_mean, tau_sq)
    prior and Normal(member_mean, sigma_sq / n) likelihood use the same
    precisions for posterior mean and variance. Member effects are predictive
    associations, not causal effects; ticker, sector, and regime confounding
    remain.
    """
    values = grp.obj[alpha_col].dropna().astype(float)
    member_means = grp[alpha_col].mean().reindex(idx).astype(float)
    counts = grp[alpha_col].count().reindex(idx).astype(float)
    global_mean = float(member_means.mean())

    member_sse = (
        grp[alpha_col]
        .apply(
            lambda x: float(
                (
                    (x.dropna().astype(float) - x.dropna().astype(float).mean()) ** 2
                ).sum()
            )
        )
        .reindex(idx)
        .fillna(0.0)
    )
    within_dof = max(int(counts.sum() - len(counts)), 0)
    if within_dof > 0:
        sigma_sq = float(member_sse.sum() / within_dof)
    elif len(values) > 1:
        sigma_sq = float(values.var(ddof=1))
    else:
        sigma_sq = 0.0

    variance_scale = max(float(values.var(ddof=0)) if len(values) else 0.0, 1.0)
    variance_floor = variance_scale * 1e-8
    sigma_sq = max(sigma_sq, variance_floor)

    observed_between = float(member_means.var(ddof=1)) if len(member_means) > 1 else 0.0
    mean_sampling_var = float((sigma_sq / counts.clip(lower=1.0)).mean())
    tau_sq = max(observed_between - mean_sampling_var, variance_floor)

    data_precision = counts / sigma_sq
    prior_precision = 1.0 / tau_sq
    posterior_precision = data_precision + prior_precision
    posterior_mean = (
        data_precision * member_means + prior_precision * global_mean
    ) / posterior_precision
    posterior_std = np.sqrt(1.0 / posterior_precision)
    shrinkage = prior_precision / posterior_precision
    return posterior_mean, posterior_std, shrinkage


def _build_ranking_result(
    idx,
    ret_agg,
    stats,
    avg_spy,
    avg_total_spy,
    hit_rates,
    conviction,
    shrunk_alpha,
    shrunk_alpha_std,
    alpha_shrinkage,
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
            "prior_win_prob": np.round(stats["prior_win_prob"], 3),
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
    result["shrunk_alpha_std"] = shrunk_alpha_std.values
    result["alpha_shrinkage"] = alpha_shrinkage.values
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
