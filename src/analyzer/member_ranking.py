"""Member ranking and scoring."""

from __future__ import annotations

from math import exp, lgamma, log

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


def bayesian_win_probability(wins: int, losses: int, market_prior: float = 0.55) -> float:
    alpha = market_prior * _signals.BAYES_PRIOR_STRENGTH
    beta = (1 - market_prior) * _signals.BAYES_PRIOR_STRENGTH
    return (alpha + wins) / (alpha + beta + wins + losses)


def bayes_factor_against_market(wins: int, losses: int, market_prior: float = 0.55) -> float:
    observations = wins + losses
    if observations == 0:
        return 1.0
    market_prior = float(np.clip(market_prior, 1e-6, 1 - 1e-6))
    alpha = market_prior * _signals.BAYES_PRIOR_STRENGTH
    beta = (1 - market_prior) * _signals.BAYES_PRIOR_STRENGTH
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


def _compute_ticker_member_performance(
    signals_df: pd.DataFrame, ticker: str, horizon: int
) -> dict[str, tuple[float, int]]:
    """Per-member Bayesian-shrunk win rate on a specific ticker from historical signals.

    Returns {member: (shrunk_win_rate, trade_count)} for members with >= 1 trade.
    Uses Bayesian shrinkage toward global prior instead of raw win-rate penalty.
    """
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
    prior_strength = _signals.BAYES_PRIOR_STRENGTH  # 20 pseudo-observations

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


def rank_members(signal_df: pd.DataFrame, horizon: int = 90, threshold: float = 5.0) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    purchases = _get_horizon_data(signal_df, horizon, TransactionType.PURCHASE.value)
    if purchases.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    purchases = _apply_quality_filter(purchases)
    if purchases.empty:
        raise AnalysisError(f"No signals survived quality filter (min price ${_signals.MIN_ENTRY_PRICE})")

    purchases = _collapse_to_episodes(purchases)

    market_prior = _compute_dynamic_prior(signal_df, horizon)
    member_stats = []
    for member, purchase_grp in purchases.groupby("member"):
        row = _compute_member_stats(member, purchase_grp, market_prior, threshold)
        if row is not None:
            conviction = _conviction_score(purchase_grp)
            row["conviction_score"] = round(conviction, 3)
            member_stats.append(row)

    result = pd.DataFrame(member_stats)
    if result.empty:
        return result

    return result.rename(columns={
        "mean_return_pct": "avg_decay_return_pct",
        "median_return_pct": "median_decay_return_pct",
        "trades": "purchase_trades",
        "prob_up": "prob_up_given_buy",
    }).sort_values("avg_total_spy_alpha_pct", ascending=False)


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


def _get_ticker_purchases(
    ticker: str,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    return transactions_df[
        (transactions_df["ticker"] == ticker)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    ]


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
    cache=None,
    as_of_date=None,
    cache_parent_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if signals_df.empty:
        raise AnalysisError("Empty signal dataframe")
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")

    if member_rankings is None:
        member_rankings = rank_members(signals_df, horizon, threshold)

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

    buyers = ticker_trades["member"].unique()

    # If Bayesian skill posteriors are provided, use them instead of raw rankings
    use_skills = member_skills is not None and len(member_skills) > 0

    if use_skills:
        from analyzer.member_skill import score_members_for_ticker

        skill_score, skill_uncertainty = score_members_for_ticker(
            ticker, list(buyers), member_skills
        )
        # Build buyer stats from skills for downstream display fields
        skill_buyers = [m for m in buyers if m in member_skills]
        buyer_stats = member_rankings[
            member_rankings["member"].isin(buyers)
        ].sort_values("avg_spy_alpha_pct", ascending=False) if member_rankings is not None else pd.DataFrame()

        if not buyer_stats.empty:
            best_rank = buyer_stats["avg_spy_alpha_pct"].max()
            total_trades = buyer_stats["purchase_trades"].sum()
            rated_buyers = len(buyer_stats)
        else:
            best_rank = skill_score
            total_trades = len(skill_buyers)
            rated_buyers = len(skill_buyers)

        # Quality-adjusted average uses posterior means weighted by inverse uncertainty
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

        # Uncertainty penalty
        base_signal_score = quality_adjusted_avg - uncertainty_penalty_lambda * skill_uncertainty
    else:
        buyer_stats = member_rankings[member_rankings["member"].isin(buyers)].sort_values(
            "avg_spy_alpha_pct", ascending=False
        )

        if buyer_stats.empty:
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

        best_rank = buyer_stats["avg_spy_alpha_pct"].max()
        total_trades = buyer_stats["purchase_trades"].sum()
        rated_buyers = len(buyer_stats)
        confidence_weights = np.sqrt(buyer_stats["purchase_trades"].clip(lower=1).astype(float).values)
        if "bayes_win_prob" in buyer_stats.columns:
            confidence_weights *= buyer_stats["bayes_win_prob"].fillna(0.55).astype(float).values
        if "disclosure_date" in ticker_trades.columns:
            rated_ticker_trades = ticker_trades[ticker_trades["member"].isin(buyer_stats["member"])]
            latest_disclosure = rated_ticker_trades["disclosure_date"].max()
            member_disclosures = rated_ticker_trades.groupby("member")["disclosure_date"].max()
            days_since = (latest_disclosure - member_disclosures.reindex(buyer_stats["member"])).dt.days.fillna(0).clip(lower=0)
            confidence_weights *= np.exp(-_signals.BUYER_RECENCY_DECAY * days_since.values)
        confidence_weight_sum = confidence_weights.sum()
        quality_adjusted_avg = (
            (buyer_stats["avg_spy_alpha_pct"].values * confidence_weights).sum() / confidence_weight_sum
            if confidence_weight_sum > 0
            else 0
        )

        base_signal_score = quality_adjusted_avg

    size_factor = _size_score_factor(ticker_trades)
    owner_factor = _owner_score_factor(ticker_trades)

    perf_source = ticker_perf_signals if ticker_perf_signals is not None else signals_df
    # For the ticker_member_perf cache key, use the PARENT signals df (stable
    # identity across calls in a sweep) rather than the freshly-filtered
    # perf_source slice (whose id() differs every call). The result of
    # _compute_ticker_member_performance depends only on the parent's content
    # restricted to (ticker, horizon, as_of), so the parent id is content-stable.
    cache_key_df = cache_parent_signals if cache_parent_signals is not None else perf_source
    if cache is not None and as_of_date is not None:
        from analyzer import member_ranking as mr_mod
        bayes_prior = mr_mod.BAYES_PRIOR_STRENGTH
        hit, ticker_perf = cache.get_ticker_member_perf(
            cache_key_df, ticker, horizon, bayes_prior, as_of_date,
        )
        if not hit:
            ticker_perf = _compute_ticker_member_performance(
                perf_source, ticker, horizon,
            )
            cache.set_ticker_member_perf(
                cache_key_df, ticker, horizon, bayes_prior, as_of_date, ticker_perf,
            )
    else:
        ticker_perf = _compute_ticker_member_performance(
            perf_source, ticker, horizon,
        )
    if ticker_perf and not buyer_stats.empty:
        member_trade_counts = ticker_trades.groupby("member").size()
        weighted_sum = 0.0
        weight_total = 0.0
        for member in buyer_stats["member"]:
            if member in ticker_perf:
                shrunk_wr, n = ticker_perf[member]
                # Weight by trade count (more data = more weight on shrunk rate)
                weighted_sum += shrunk_wr * n
                weight_total += n
        ticker_perf_factor = weighted_sum / weight_total if weight_total > 0 else 1.0
    else:
        ticker_perf_factor = 1.0

    signal_score = base_signal_score * size_factor * owner_factor * ticker_perf_factor

    if not buyer_stats.empty:
        top_buyers = buyer_stats["member"].head(3).tolist()
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
    })


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
