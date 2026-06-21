"""Member ranking and scoring."""

from __future__ import annotations

from math import exp, lgamma, log
from scipy.special import gammaln as _gammaln

import numpy as np
import pandas as pd

from analyzer import signals as _signals
from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.signals import (
    _apply_quality_filter,
    _collapse_to_episodes,
    _compute_dynamic_prior,
    _get_horizon_data,
)
from analyzer._memo import df_memoize


def bayesian_win_probability(wins: int, losses: int, market_prior: float = 0.55, prior_strength: float | None = None) -> float:
    ps = prior_strength if prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH
    alpha = market_prior * ps
    beta = (1 - market_prior) * ps
    return (alpha + wins) / (alpha + beta + wins + losses)


def bayes_factor_against_market(wins: int, losses: int, market_prior: float = 0.55, prior_strength: float | None = None) -> float:
    observations = wins + losses
    if observations == 0:
        return 1.0
    ps = prior_strength if prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH
    market_prior = float(np.clip(market_prior, 1e-6, 1 - 1e-6))
    alpha = market_prior * ps
    beta = (1 - market_prior) * ps
    log_marginal = (
        lgamma(alpha + wins)
        + lgamma(beta + losses)
        - lgamma(alpha + beta + observations)
        - lgamma(alpha)
        - lgamma(beta)
        + lgamma(alpha + beta)
    )
    log_market = wins * log(market_prior) + losses * log(1 - market_prior)
    return exp(float(np.clip(log_marginal - log_market, -50, 50)))


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


@df_memoize(copy=False)
def _compute_ticker_member_performance(
    signals_df: pd.DataFrame, ticker: str, horizon: int,
    _bayes_prior_strength: float | None = None,
) -> dict[str, tuple[float, int]]:
    """Per-member Bayesian-shrunk win rate on a specific ticker from historical signals.

    Returns {member: (shrunk_win_rate, trade_count)} for members with >= 1 trade.
    Result is memoized via @df_memoize using signals_df identity + ticker + horizon.
    The _bayes_prior_strength kwarg distinguishes cache keys when the module global
    BAYES_PRIOR_STRENGTH varies across sweep combos.
    """
    prior_strength = _bayes_prior_strength if _bayes_prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH
    if signals_df.empty or "ticker" not in signals_df.columns:
        return {}

    purchases = signals_df[
        (signals_df["ticker"] == ticker)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]
    if purchases.empty:
        return {}

    # Global prior from all purchase signals
    all_purchases = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]
    all_returns = all_purchases["decayed_return_pct"].dropna()
    global_win_rate = float((all_returns > 0).mean()) if len(all_returns) > 0 else 0.5

    result: dict[str, tuple[float, int]] = {}
    for member, grp in purchases.groupby("member"):
        returns = grp["decayed_return_pct"].dropna()
        if len(returns) == 0:
            continue
        wins = int((returns > 0).sum())
        n = len(returns)
        # Bayesian shrinkage: pull toward global win rate
        shrunk_wr = (global_win_rate * prior_strength + wins) / (prior_strength + n)
        result[member] = (shrunk_wr, n)
    return result


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

    spy_alpha_vals = grp["spy_alpha_pct"].dropna().values
    if invert_returns:
        spy_alpha_vals = -spy_alpha_vals
    avg_spy_alpha = float(np.mean(spy_alpha_vals)) if len(spy_alpha_vals) > 0 else 0.0

    total_spy_alpha_vals = grp["total_spy_alpha_pct"].dropna().values if "total_spy_alpha_pct" in grp.columns else np.array([])
    if invert_returns:
        total_spy_alpha_vals = -total_spy_alpha_vals
    avg_total_spy_alpha = float(np.mean(total_spy_alpha_vals)) if len(total_spy_alpha_vals) > 0 else avg_spy_alpha

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
        stats["peak_hit_rate_pct"] = round((grp["peak_potential_pct"] > threshold).mean() * 100, 2)
        if "total_return_pct" in grp.columns:
            stats["realized_hit_rate_pct"] = round((grp["total_return_pct"] > 0).mean() * 100, 2)
    return stats


def rank_members(signal_df: pd.DataFrame, horizon: int = 90, threshold: float = 5.0,
                  _bayes_prior_strength: float | None = None) -> pd.DataFrame:
    """Rank members by historical purchase performance.

    bayesian_win_probability / bayes_factor_against_market read the module
    global BAYES_PRIOR_STRENGTH. Temporarily set it so the computation
    reflects the correct prior for this call, then restore.
    """
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")

    bayes_prior = _bayes_prior_strength if _bayes_prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH

    return _rank_members_impl(signal_df, horizon, threshold, bayes_prior)


@df_memoize(copy=False)
def _prepare_member_data(
    signal_df: pd.DataFrame, horizon: int, threshold: float,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Prepare collapsed purchases and market prior (bayes-independent).

    This is the expensive part of _rank_members_impl that doesn't depend on
    bayes_prior_strength. Extracting it allows memoization to hit cache for
    2/3 of combos (all but the bayes_prior dimension change).
    """
    purchases = _get_horizon_data(signal_df, horizon, TransactionType.PURCHASE.value)
    if purchases.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    purchases = _apply_quality_filter(purchases)
    if purchases.empty:
        raise AnalysisError(f"No signals survived quality filter (min price ${_signals.MIN_ENTRY_PRICE})")

    purchases = _collapse_to_episodes(purchases)
    market_prior = _compute_dynamic_prior(signal_df, horizon)
    return purchases, market_prior


@df_memoize(copy=False)
def _rank_members_impl(signal_df: pd.DataFrame, horizon: int, threshold: float,
                       _bayes_prior_strength: float) -> pd.DataFrame:
    purchases, market_prior = _prepare_member_data(signal_df, horizon, threshold)

    alpha_col = "total_spy_alpha_pct" if "total_spy_alpha_pct" in purchases.columns else "spy_alpha_pct"
    prior_alpha_mean = float(purchases[alpha_col].mean())
    if pd.isna(prior_alpha_mean):
        prior_alpha_mean = 0.0
    prior_strength = _bayes_prior_strength

    grp = purchases.groupby("member")

    # --- Vectorized aggregations (single pass) ---
    ret_agg = grp["decayed_return_pct"].agg(
        ret_nonnan="count",
        median_ret="median",
        mean_ret="mean",
        std_ret="std",
    )

    # Filter to members with >= 1 non-NaN return
    ret_agg = ret_agg[ret_agg["ret_nonnan"] > 0]
    if ret_agg.empty:
        return pd.DataFrame()

    idx = ret_agg.index
    n = ret_agg["ret_nonnan"].astype(int)

    # Wins: positive returns (NaN comparisons → NaN, skipped by groupby sum)
    wins = (purchases["decayed_return_pct"] > 0).groupby(purchases["member"]).sum().reindex(idx, fill_value=0).astype(int)
    losses = n - wins

    # Fix std: NaN for single-element groups → 0.0
    ret_agg["std_ret"] = ret_agg["std_ret"].fillna(0.0)

    # --- Vectorized derived stats ---
    sharpe = np.where(ret_agg["std_ret"] > 0, ret_agg["mean_ret"] / ret_agg["std_ret"], 0.0)
    prob_up = wins.values / n.values

    # Bayesian win probability
    bayes_alpha = market_prior * prior_strength
    bayes_beta = (1 - market_prior) * prior_strength
    bayes_win_prob = (bayes_alpha + wins.values) / (bayes_alpha + bayes_beta + wins.values + losses.values)
    posterior_lift = bayes_win_prob / market_prior

    # Bayes factor against market (vectorized log-space)
    n_vals = n.values.astype(float)
    wins_f = wins.values.astype(float)
    losses_f = losses.values.astype(float)
    log_marginal = (
        _gammaln(bayes_alpha + wins_f)
        + _gammaln(bayes_beta + losses_f)
        - _gammaln(bayes_alpha + bayes_beta + n_vals)
        - _gammaln(bayes_alpha)
        - _gammaln(bayes_beta)
        + _gammaln(bayes_alpha + bayes_beta)
    )
    mp_clipped = float(np.clip(market_prior, 1e-6, 1 - 1e-6))
    log_market = wins_f * np.log(mp_clipped) + losses_f * np.log(1 - mp_clipped)
    bayes_factor = np.exp(np.clip(log_marginal - log_market, -50, 50))

    # Spy alpha
    avg_spy = grp["spy_alpha_pct"].mean().reindex(idx).fillna(0.0)
    if "total_spy_alpha_pct" in purchases.columns:
        avg_total_spy = grp["total_spy_alpha_pct"].mean().reindex(idx)
        avg_total_spy = avg_total_spy.fillna(avg_spy)
    else:
        avg_total_spy = avg_spy.copy()

    # Hit rates (if threshold provided)
    threshold_has_total = False
    if threshold is not None:
        peak_hits = (purchases["peak_potential_pct"] > threshold).groupby(purchases["member"]).mean().reindex(idx) * 100
        threshold_has_total = "total_return_pct" in purchases.columns
        if threshold_has_total:
            realized_hits = (purchases["total_return_pct"] > 0).groupby(purchases["member"]).mean().reindex(idx) * 100

    # --- Conviction score (vectorized) ---
    group_sizes = grp.size().reindex(idx)
    count_scores = np.minimum(group_sizes.values / 10.0, 1.0)
    has_amounts_col = "amount_midpoint" in purchases.columns
    if has_amounts_col:
        avg_amounts = grp["amount_midpoint"].mean().reindex(idx)
        amount_has_data = (grp["amount_midpoint"].count().reindex(idx) > 0).values
        size_scores = np.where(
            amount_has_data,
            np.minimum(avg_amounts.fillna(0.0).values / 50000.0, 1.0),
            1.0,
        )
    else:
        size_scores = np.ones(len(idx))
    conviction = count_scores * 0.6 + size_scores * 0.4

    # --- Shrunk alpha (vectorized shrinkage) ---
    alpha_sums = grp[alpha_col].sum().reindex(idx).fillna(0.0)
    alpha_counts = grp[alpha_col].count().reindex(idx).fillna(0).astype(int)
    shrunk_alpha = (prior_alpha_mean * prior_strength + alpha_sums) / (prior_strength + alpha_counts)

    # --- Build result DataFrame ---
    result = pd.DataFrame({
        "member": idx,
        "median_return_pct": np.round(ret_agg["median_ret"].values, 2),
        "mean_return_pct": np.round(ret_agg["mean_ret"].values, 2),
        "trades": n.values,
        "sharpe_ratio": np.round(sharpe, 3),
        "prob_up": np.round(prob_up, 3),
        "bayes_win_prob": np.round(bayes_win_prob, 3),
        "bayes_factor": np.round(bayes_factor, 3),
        "posterior_lift": np.round(posterior_lift, 3),
        "avg_spy_alpha_pct": np.round(avg_spy.values, 2),
        "avg_total_spy_alpha_pct": np.round(avg_total_spy.values, 2),
    })

    if threshold is not None:
        result["peak_hit_rate_pct"] = np.round(peak_hits.values, 2)
        if threshold_has_total:
            result["realized_hit_rate_pct"] = np.round(realized_hits.values, 2)

    result["conviction_score"] = np.round(conviction, 3)
    result["shrunk_alpha"] = shrunk_alpha.values

    return result.rename(columns={
        "mean_return_pct": "avg_decay_return_pct",
        "median_return_pct": "median_decay_return_pct",
        "trades": "purchase_trades",
        "prob_up": "prob_up_given_buy",
    }).sort_values("shrunk_alpha", ascending=False)


def rank_sales(signal_df: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    sales = _get_horizon_data(signal_df, horizon, TransactionType.SALE.value)
    if sales.empty:
        raise AnalysisError(f"No sale signals found for horizon {horizon}")

    sales = _collapse_to_episodes(sales)

    market_prior = _compute_dynamic_prior(signal_df, horizon)
    member_stats = []
    for member, sale_grp in sales.groupby("member"):
        row = _compute_member_stats(member, sale_grp, market_prior, invert_returns=True)
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


@df_memoize(copy=False)
def _get_ticker_purchases(
    ticker: str,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    return transactions_df[
        (transactions_df["ticker"] == ticker)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    ]


def _lookup_buyer_bayes_win_prob(
    member: str, member_rankings: pd.DataFrame | None
) -> float | None:
    """Fetch a member's Bayesian win probability from member_rankings.

    Returns None when rankings are missing, the column is absent, the member
    is unrated, or the value is NaN.
    """
    if member_rankings is None or member_rankings.empty:
        return None
    if "bayes_win_prob" not in member_rankings.columns:
        return None
    row = member_rankings.loc[member_rankings["member"] == member]
    if row.empty:
        return None
    val = row["bayes_win_prob"].iloc[0]
    return float(val) if pd.notna(val) else None


def _build_ranking_dicts(
    member_rankings: pd.DataFrame | None,
    scoring_mode: str = "shrunk_alpha",
) -> dict[str, dict[str, float]]:
    """Pre-build O(1) lookup dicts from member_rankings DataFrame.

    Returns {"alpha": {member: float}, "trades": {member: int}, "prob": {member: float}}.
    Avoids repeated DataFrame linear scans in the per-ticker scoring loop.

    scoring_mode controls how member scores are computed:
      - "shrunk_alpha": Bayesian-shrunk historical SPY alpha (default)
      - "consistency": prob_up * log(1 + trades) — continuous, differentiable
      - "bayesian_quality": bayes_win_prob * shrunk_alpha
      - "trade_frequency": log(1 + trades)
    """
    if member_rankings is None or member_rankings.empty:
        return {"alpha": {}, "trades": {}, "prob": {}, "has_shrunk": False}

    has_shrunk = "shrunk_alpha" in member_rankings.columns
    alpha_col = "shrunk_alpha" if has_shrunk else "avg_spy_alpha_pct"

    # Vectorized dict construction (avoids iterrows — O(N) Series allocs)
    cols = ["member", alpha_col, "purchase_trades"]
    if "bayes_win_prob" in member_rankings.columns:
        cols.append("bayes_win_prob")
    if "prob_up_given_buy" in member_rankings.columns:
        cols.append("prob_up_given_buy")
    valid = member_rankings[cols].dropna(subset=["member"])

    # Compute alpha scores based on scoring_mode
    if scoring_mode == "consistency":
        prob_up = valid["prob_up_given_buy"].fillna(0.5).values if "prob_up_given_buy" in valid.columns else np.full(len(valid), 0.5)
        trades = valid["purchase_trades"].fillna(0).values.astype(float)
        alpha_values = prob_up * np.log1p(trades)
        alpha = dict(zip(valid["member"], alpha_values))
    elif scoring_mode == "bayesian_quality":
        bayes = valid["bayes_win_prob"].fillna(0.5).values if "bayes_win_prob" in valid.columns else np.full(len(valid), 0.5)
        raw_alpha = valid[alpha_col].fillna(0.0).values.astype(float)
        alpha = dict(zip(valid["member"], bayes * raw_alpha))
    elif scoring_mode == "trade_frequency":
        trades = valid["purchase_trades"].fillna(0).values.astype(float)
        alpha = dict(zip(valid["member"], np.log1p(trades)))
    else:  # "shrunk_alpha" (default)
        alpha = dict(zip(valid["member"], valid[alpha_col].astype(float)))

    trades_dict = dict(zip(valid["member"], valid["purchase_trades"].fillna(0).astype(int)))
    prob = dict(zip(valid["member"], valid["bayes_win_prob"].fillna(0.5).astype(float))) if "bayes_win_prob" in valid.columns else {}

    return {"alpha": alpha, "trades": trades_dict, "prob": prob, "has_shrunk": has_shrunk}


@df_memoize(copy=False)
def score_ticker_by_buyers(
    ticker: str,
    transactions_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    horizon: int = 90,
    threshold: float = 5.0,
    member_rankings: pd.DataFrame | None = None,
    min_buyers: int = 2,
    ticker_perf_signals: pd.DataFrame | None = None,
    member_skills: dict | None = None,
    uncertainty_penalty_lambda: float = 0.5,
    solo_buyer_skill_threshold: float = 0.60,
    solo_buyer_penalty: float = 0.8,
    _bayes_prior_strength: float | None = None,
    _ranking_dicts: dict | None = None,
) -> pd.DataFrame:
    """Score a ticker by its buyer composition. Memoized via @df_memoize.

    When ``_ranking_dicts`` is provided (pre-built by the caller), dict
    lookups replace DataFrame linear scans for buyer stats.
    """
    if signals_df.empty:
        raise AnalysisError("Empty signal dataframe")
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")

    bayes_prior = _bayes_prior_strength if _bayes_prior_strength is not None else _signals.BAYES_PRIOR_STRENGTH

    if member_rankings is None:
        member_rankings = rank_members(signals_df, horizon, threshold, _bayes_prior_strength=bayes_prior)

    ticker_trades = _get_ticker_purchases(ticker, transactions_df)

    if ticker_trades.empty:
        return pd.DataFrame({
            "ticker": [ticker],
            "num_buyers": [0],
            "signal_score": [0.0]
        })

    min_trades = ticker_trades["member"].nunique()
    if min_trades < min_buyers:
        return pd.DataFrame({
            "ticker": [ticker],
            "num_buyers": [min_trades],
            "signal_score": [0.0],
            "note": [f"Below minimum buyer threshold ({min_buyers})"]
        })

    # Solo-buyer skill gate
    apply_solo_penalty = False
    if (
        min_buyers == 1
        and min_trades == 1
        and solo_buyer_skill_threshold > 0
    ):
        sole_buyer = str(ticker_trades["member"].iloc[0])
        bayes_prob = _lookup_buyer_bayes_win_prob(sole_buyer, member_rankings)
        if bayes_prob is None or bayes_prob < solo_buyer_skill_threshold:
            return pd.DataFrame({
                "ticker": [ticker],
                "num_buyers": [min_trades],
                "signal_score": [0.0],
                "note": [
                    f"Solo buyer '{sole_buyer}' below skill threshold "
                    f"({solo_buyer_skill_threshold})"
                ],
            })
        apply_solo_penalty = True

    buyers = ticker_trades["member"].unique()
    use_skills = member_rankings is not None and member_skills is not None and len(member_skills) > 0

    # Build or use pre-built ranking dicts for O(1) member lookups
    rd = _ranking_dicts if _ranking_dicts is not None else _build_ranking_dicts(member_rankings)
    alpha_dict = rd["alpha"]
    trades_dict = rd["trades"]

    if use_skills:
        from analyzer.member_skill import score_members_for_ticker

        skill_score, skill_uncertainty = score_members_for_ticker(
            ticker, list(buyers), member_skills
        )
        skill_buyers = [m for m in buyers if m in member_skills]

        # O(1) lookups via dict instead of DataFrame filter
        rated_buyers_list = [m for m in buyers if m in alpha_dict]
        if rated_buyers_list:
            best_rank = max(alpha_dict[m] for m in rated_buyers_list)
            total_trades = sum(trades_dict.get(m, 0) for m in rated_buyers_list)
            rated_buyers = len(rated_buyers_list)
        else:
            best_rank = skill_score
            total_trades = len(skill_buyers)
            rated_buyers = len(skill_buyers)

        if skill_buyers:
            skill_posteriors = [member_skills[m] for m in skill_buyers]
            inv_stds = np.array([1.0 / max(s.alpha_std, 1e-6) for s in skill_posteriors])
            inv_std_sum = inv_stds.sum()
            if inv_std_sum > 0:
                weights = inv_stds / inv_std_sum
                quality_adjusted_avg = float(np.dot(
                    weights,
                    np.array([s.alpha_mean for s in skill_posteriors]),
                ))
            else:
                quality_adjusted_avg = skill_score
        else:
            quality_adjusted_avg = skill_score

        base_signal_score = quality_adjusted_avg - uncertainty_penalty_lambda * skill_uncertainty
        buyer_stats_empty = not rated_buyers_list
    else:
        # O(1) dict lookups instead of DataFrame isin filter
        rated_buyers_list = [m for m in buyers if m in alpha_dict]

        if not rated_buyers_list:
            # Fallback: ticker history score
            fallback_score = 0.0
            fallback_source = "none"
            perf_signals = ticker_perf_signals if ticker_perf_signals is not None else signals_df
            if not perf_signals.empty and "ticker" in perf_signals.columns:
                ticker_hist = perf_signals[
                    (perf_signals["ticker"] == ticker)
                    & (perf_signals["signal_type"] == TransactionType.PURCHASE.value)
                    & (perf_signals["total_spy_alpha_pct"].notna())
                ]
                if len(ticker_hist) >= 2:
                    fallback_score = float(ticker_hist["total_spy_alpha_pct"].mean())
                    fallback_source = f"ticker_hist({len(ticker_hist)})"

            return pd.DataFrame({
                "ticker": [ticker],
                "num_buyers": [len(buyers)],
                "buyers": [", ".join(buyers[:3])],
                "signal_score": [round(fallback_score, 2)],
                "fallback_source": [fallback_source],
            })

        best_rank = max(alpha_dict[m] for m in rated_buyers_list)
        total_trades = sum(trades_dict.get(m, 0) for m in rated_buyers_list)
        rated_buyers = len(rated_buyers_list)

        # Recency weighting via dict lookups
        n_rated = len(rated_buyers_list)
        confidence_weights = np.ones(n_rated, dtype=float)
        if "disclosure_date" in ticker_trades.columns:
            rated_ticker_trades = ticker_trades[ticker_trades["member"].isin(rated_buyers_list)]
            if not rated_ticker_trades.empty:
                latest_disclosure = rated_ticker_trades["disclosure_date"].max()
                member_disclosures = rated_ticker_trades.groupby("member")["disclosure_date"].max()
                days_since = (latest_disclosure - member_disclosures.reindex(rated_buyers_list)).dt.days.fillna(0).clip(lower=0)
                confidence_weights = np.exp(-_signals.BUYER_RECENCY_DECAY * days_since.values)

        alpha_values = np.array([alpha_dict[m] for m in rated_buyers_list])
        confidence_weight_sum = confidence_weights.sum()
        quality_adjusted_avg = (
            (alpha_values * confidence_weights).sum() / confidence_weight_sum
            if confidence_weight_sum > 0
            else 0
        )

        base_signal_score = quality_adjusted_avg
        buyer_stats_empty = False

    size_factor = _size_score_factor(ticker_trades)
    owner_factor = _owner_score_factor(ticker_trades)
    ticker_perf_factor = 1.0

    signal_score = base_signal_score * size_factor * owner_factor
    if apply_solo_penalty:
        signal_score *= solo_buyer_penalty

    if not buyer_stats_empty:
        # Sort rated buyers by alpha for top-3 display
        top_buyers = sorted(rated_buyers_list, key=lambda m: alpha_dict.get(m, 0), reverse=True)[:3]
    else:
        skill_members = [m for m in buyers if m in (member_skills or {})]
        top_buyers = skill_members[:3] if skill_members else list(buyers[:3])
    buyer_label = f"Top {len(top_buyers)} of {len(buyers)}" if len(buyers) > 3 else f"{len(buyers)}"

    return pd.DataFrame({
        "ticker": [ticker],
        "num_buyers": [len(buyers)],
        "rated_buyers": [rated_buyers],
        "buyer_label": [buyer_label],
        "buyers": [", ".join(top_buyers)],
        "avg_buyer_performance": [round(quality_adjusted_avg, 2)],
        "best_buyer_performance": [round(best_rank, 2)],
        "total_buyer_trades": [int(total_trades)],
        "convergence_factor": [1.0],
        "ticker_perf_factor": [round(ticker_perf_factor, 3)],
        "base_signal_score": [round(base_signal_score, 2)],
        "size_factor": [round(size_factor, 3)],
        "owner_factor": [round(owner_factor, 3)],
        "signal_score": [round(signal_score, 2)],
        "fallback_source": ["member_ranked"],
        "uncertainty_lambda": [uncertainty_penalty_lambda if use_skills else 0.0],
        "solo_buyer": [apply_solo_penalty],
    })


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
        & (signals_df["signal_type"] == "Purchase")
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
        & (signals_df["signal_type"] == "Purchase")
    ]["member"].unique()

    result = {}
    for member in members:
        lam = estimate_member_decay_lambda(
            member, signals_df, horizon, default_lambda, min_trades,
        )
        if abs(lam - default_lambda) > 1e-6:
            result[member] = lam

    return result


def get_ticker_buyers_with_rankings(
    ticker: str,
    transactions_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    horizon: int = 90,
    threshold: float = 5.0,
) -> pd.DataFrame:
    if signals_df.empty:
        raise AnalysisError("Empty signal dataframe")
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")

    member_rankings = rank_members(signals_df, horizon, threshold)

    ticker_trades = _get_ticker_purchases(ticker, transactions_df)

    if ticker_trades.empty:
        raise AnalysisError(f"No purchases found for {ticker}")

    buyers_with_dates = ticker_trades.groupby("member").agg({
        "transaction_date": list,
        "disclosure_date": list
    }).reset_index()

    result = pd.merge(
        buyers_with_dates,
        member_rankings[["member", "avg_spy_alpha_pct", "peak_hit_rate_pct", "purchase_trades"]],
        on="member",
        how="left"
    )

    result = result.sort_values("avg_spy_alpha_pct", ascending=False, na_position="last")
    result["num_purchases"] = result["transaction_date"].apply(len)

    return result[["member", "num_purchases", "transaction_date", "disclosure_date",
                   "avg_spy_alpha_pct", "peak_hit_rate_pct", "purchase_trades"]]
