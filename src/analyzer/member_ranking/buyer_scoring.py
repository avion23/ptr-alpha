"""Score a ticker by its buyer composition.

`score_ticker_by_buyers` combines member rankings with a ticker's actual
buyers to produce a single signal score. Supports four scoring modes
(shrunk_alpha, consistency, bayesian_quality, trade_frequency), solo-buyer
skill gates, member-skill overrides, recency weighting, and a fallback to
ticker-history alpha when no rated buyers are found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer import signals as _signals
from analyzer._memo import df_memoize
from analyzer.exceptions import AnalysisError
from analyzer.member_names import canonical_member_key
from analyzer.models import TransactionType
from analyzer.signals import TICKER_PERF_MIN_TRADES

from analyzer.member_ranking.factors import _owner_score_factor, _size_score_factor
from analyzer.member_ranking.ranking import rank_members
from analyzer.member_ranking.lookups import (
    _build_ranking_dicts,
    _get_ticker_purchases,
    _lookup_buyer_posterior_lift,
)


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
    solo_buyer_skill_threshold: float = 1.0,
    solo_buyer_penalty: float = 0.8,
    _bayes_prior_strength: float | None = None,
    _ranking_dicts: dict | None = None,
) -> pd.DataFrame:
    """Score a ticker by its buyer composition. Memoized via @df_memoize.

    When ``_ranking_dicts`` is provided (pre-built by the caller), dict
    lookups replace DataFrame linear scans for buyer stats.
    """
    _validate_inputs(signals_df, transactions_df)

    bayes_prior = (
        _bayes_prior_strength
        if _bayes_prior_strength is not None
        else _signals.BAYES_PRIOR_STRENGTH
    )

    if member_rankings is None:
        member_rankings = rank_members(
            signals_df, horizon, threshold, _bayes_prior_strength=bayes_prior
        )

    ticker_trades = _get_ticker_purchases(ticker, transactions_df).copy()
    if ticker_trades.empty:
        return _empty_ticker_result(ticker)

    ticker_trades["_member_canonical"] = ticker_trades["member"].map(
        canonical_member_key
    )
    min_trades = ticker_trades["_member_canonical"].nunique()
    if min_trades < min_buyers:
        return _below_threshold_result(ticker, min_trades, min_buyers)

    solo_gate = _solo_buyer_gate(
        ticker,
        ticker_trades,
        member_rankings,
        min_buyers,
        min_trades,
        solo_buyer_skill_threshold,
    )
    if isinstance(solo_gate, pd.DataFrame):
        return solo_gate
    apply_solo_penalty = solo_gate

    buyers = ticker_trades["_member_canonical"].unique()
    if member_skills:
        member_skills = member_skills | {
            canonical_member_key(member): skill
            for member, skill in member_skills.items()
            if canonical_member_key(member) not in member_skills
        }
    use_skills = (
        member_rankings is not None
        and member_skills is not None
        and len(member_skills) > 0
    )
    rd = (
        _ranking_dicts
        if _ranking_dicts is not None
        else _build_ranking_dicts(member_rankings)
    )
    alpha_dict = rd["alpha"]
    trades_dict = rd["trades"]

    if use_skills:
        inputs = _skill_inputs(
            ticker,
            buyers,
            member_skills,
            alpha_dict,
            trades_dict,
            uncertainty_penalty_lambda,
        )
    else:
        fallback = _member_only_inputs(
            ticker,
            buyers,
            alpha_dict,
            trades_dict,
            ticker_trades,
            signals_df,
            ticker_perf_signals,
        )
        if isinstance(fallback, pd.DataFrame):
            return fallback
        inputs = fallback

    return _final_result(
        ticker,
        buyers,
        ticker_trades,
        inputs,
        apply_solo_penalty,
        solo_buyer_penalty,
        use_skills,
        uncertainty_penalty_lambda,
        alpha_dict,
        member_skills,
    )


def _validate_inputs(signals_df: pd.DataFrame, transactions_df: pd.DataFrame) -> None:
    if signals_df.empty:
        raise AnalysisError("Empty signal dataframe")
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")


def _empty_ticker_result(ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [0],
            "signal_score": [0.0],
            "signal_score_raw": [0.0],
        }
    )


def _below_threshold_result(
    ticker: str, min_trades: int, min_buyers: int
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [min_trades],
            "signal_score": [0.0],
            "signal_score_raw": [0.0],
            "note": [f"Below minimum buyer threshold ({min_buyers})"],
        }
    )


def _solo_buyer_gate(
    ticker,
    ticker_trades,
    member_rankings,
    min_buyers,
    min_trades,
    solo_buyer_skill_threshold,
):
    """Returns True to apply penalty, False to skip, or a DataFrame to short-circuit."""
    if not (min_buyers == 1 and min_trades == 1 and solo_buyer_skill_threshold > 0):
        return False
    sole_buyer = str(ticker_trades["member"].iloc[0])
    # The gate compares posterior_lift (posterior / leave-one-out peer prior),
    # not the absolute bayes_win_prob. Each member's prior is estimated
    # leave-one-out, so bayes_win_prob is no longer comparable across members
    # against a fixed threshold; posterior_lift is the prior-invariant skill
    # statistic (a lift above 1.0 means the buyer beats their peer prior).
    lift = _lookup_buyer_posterior_lift(sole_buyer, member_rankings)
    if lift is None or lift < solo_buyer_skill_threshold:
        result = pd.DataFrame(
            {
                "ticker": [ticker],
                "num_buyers": [min_trades],
                "signal_score": [0.0],
                "signal_score_raw": [0.0],
                "note": [
                    f"Solo buyer '{sole_buyer}' below skill threshold "
                    f"({solo_buyer_skill_threshold})"
                ],
            }
        )
        return result
    return True


def _skill_inputs(
    ticker, buyers, member_skills, alpha_dict, trades_dict, uncertainty_penalty_lambda
):
    from analyzer.member_skill import score_members_for_ticker

    skill_score, skill_uncertainty = score_members_for_ticker(
        ticker, list(buyers), member_skills
    )
    skill_buyers = [m for m in buyers if m in member_skills]

    rated_buyers_list = [m for m in buyers if m in alpha_dict]
    if rated_buyers_list:
        best_rank = max(alpha_dict[m] for m in rated_buyers_list)
        total_trades = sum(trades_dict.get(m, 0) for m in rated_buyers_list)
        rated_buyers = len(rated_buyers_list)
    else:
        best_rank = skill_score
        total_trades = len(skill_buyers)
        rated_buyers = len(skill_buyers)

    quality_adjusted_avg = _skill_weighted_alpha(
        skill_buyers, member_skills, skill_score
    )
    base_signal_score = (
        quality_adjusted_avg - uncertainty_penalty_lambda * skill_uncertainty
    )

    return {
        "base_signal_score": base_signal_score,
        "rated_buyers_list": rated_buyers_list,
        "best_rank": best_rank,
        "total_trades": total_trades,
        "rated_buyers": rated_buyers,
        "quality_adjusted_avg": quality_adjusted_avg,
    }


def _skill_weighted_alpha(skill_buyers, member_skills, skill_score: float) -> float:
    if not skill_buyers:
        return skill_score
    skill_posteriors = [member_skills[m] for m in skill_buyers]
    inv_stds = np.array([1.0 / max(s.alpha_std, 1e-6) for s in skill_posteriors])
    inv_std_sum = inv_stds.sum()
    if inv_std_sum <= 0:
        return skill_score
    weights = inv_stds / inv_std_sum
    return float(np.dot(weights, np.array([s.alpha_mean for s in skill_posteriors])))


def _member_only_inputs(
    ticker,
    buyers,
    alpha_dict,
    trades_dict,
    ticker_trades,
    signals_df,
    ticker_perf_signals,
):
    rated_buyers_list = [m for m in buyers if m in alpha_dict]
    if not rated_buyers_list:
        return _ticker_history_fallback(ticker, buyers, signals_df, ticker_perf_signals)

    best_rank = max(alpha_dict[m] for m in rated_buyers_list)
    total_trades = sum(trades_dict.get(m, 0) for m in rated_buyers_list)
    rated_buyers = len(rated_buyers_list)

    confidence_weights = _recency_weights(ticker_trades, rated_buyers_list)
    alpha_values = np.array([alpha_dict[m] for m in rated_buyers_list])
    confidence_weight_sum = confidence_weights.sum()
    quality_adjusted_avg = (
        (alpha_values * confidence_weights).sum() / confidence_weight_sum
        if confidence_weight_sum > 0
        else 0
    )

    return {
        "base_signal_score": quality_adjusted_avg,
        "rated_buyers_list": rated_buyers_list,
        "best_rank": best_rank,
        "total_trades": total_trades,
        "rated_buyers": rated_buyers,
        "quality_adjusted_avg": quality_adjusted_avg,
    }


def _ticker_history_fallback(ticker, buyers, signals_df, ticker_perf_signals):
    fallback_score = 0.0
    fallback_source = "none"
    perf_signals = (
        ticker_perf_signals if ticker_perf_signals is not None else signals_df
    )
    if not perf_signals.empty and "ticker" in perf_signals.columns:
        ticker_hist = perf_signals[
            (perf_signals["ticker"] == ticker)
            & (perf_signals["signal_type"] == TransactionType.PURCHASE.value)
            & (perf_signals["total_spy_alpha_pct"].notna())
        ]
        if "window_complete" in ticker_hist.columns:
            ticker_hist = ticker_hist[
                ticker_hist["window_complete"].fillna(False).astype(bool)
            ]
        if len(ticker_hist) >= TICKER_PERF_MIN_TRADES:
            fallback_score = float(ticker_hist["total_spy_alpha_pct"].mean())
            fallback_source = f"ticker_hist({len(ticker_hist)})"

    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [len(buyers)],
            "buyers": [", ".join(buyers[:3])],
            "signal_score": [round(fallback_score, 2)],
            "signal_score_raw": [fallback_score],
            "fallback_source": [fallback_source],
        }
    )


def _recency_weights(ticker_trades: pd.DataFrame, rated_buyers_list) -> np.ndarray:
    n_rated = len(rated_buyers_list)
    if n_rated == 0 or "disclosure_date" not in ticker_trades.columns:
        return np.ones(n_rated, dtype=float)
    member_col = (
        "_member_canonical"
        if "_member_canonical" in ticker_trades.columns
        else "member"
    )
    rated_ticker_trades = ticker_trades[
        ticker_trades[member_col].isin(rated_buyers_list)
    ]
    if rated_ticker_trades.empty:
        return np.ones(n_rated, dtype=float)
    latest_disclosure = rated_ticker_trades["disclosure_date"].max()
    member_disclosures = rated_ticker_trades.groupby(member_col)[
        "disclosure_date"
    ].max()
    days_since = (
        (latest_disclosure - member_disclosures.reindex(rated_buyers_list))
        .dt.days.fillna(0)
        .clip(lower=0)
    )
    return np.exp(-_signals.BUYER_RECENCY_DECAY * days_since.values)


def _final_result(
    ticker,
    buyers,
    ticker_trades,
    inputs,
    apply_solo_penalty,
    solo_buyer_penalty,
    use_skills,
    uncertainty_penalty_lambda,
    alpha_dict,
    member_skills,
) -> pd.DataFrame:
    base_signal_score = inputs["base_signal_score"]
    rated_buyers_list = inputs["rated_buyers_list"]
    best_rank = inputs["best_rank"]
    total_trades = inputs["total_trades"]
    rated_buyers = inputs["rated_buyers"]
    quality_adjusted_avg = inputs["quality_adjusted_avg"]

    size_factor = _size_score_factor(ticker_trades)
    owner_factor = _owner_score_factor(ticker_trades)

    signal_score_raw = base_signal_score
    if apply_solo_penalty:
        signal_score_raw *= solo_buyer_penalty
    signal_score = round(signal_score_raw, 2)

    top_buyers = _top_buyers_for_label(
        buyers, rated_buyers_list, alpha_dict, member_skills
    )
    buyer_label = _buyer_label(len(top_buyers), len(buyers))

    return pd.DataFrame(
        {
            "ticker": [ticker],
            "num_buyers": [len(buyers)],
            "rated_buyers": [rated_buyers],
            "buyer_label": [buyer_label],
            "buyers": [", ".join(top_buyers)],
            "avg_buyer_performance": [round(quality_adjusted_avg, 2)],
            "best_buyer_performance": [round(best_rank, 2)],
            "total_buyer_trades": [int(total_trades)],
            "convergence_factor": [1.0],
            "ticker_perf_factor": [round(1.0, 3)],
            "base_signal_score": [round(base_signal_score, 2)],
            "size_factor": [round(size_factor, 3)],
            "owner_factor": [round(owner_factor, 3)],
            "signal_score": [signal_score],
            "signal_score_raw": [signal_score_raw],
            "fallback_source": ["member_ranked"],
            "uncertainty_lambda": [uncertainty_penalty_lambda if use_skills else 0.0],
            "solo_buyer": [apply_solo_penalty],
        }
    )


def _top_buyers_for_label(buyers, rated_buyers_list, alpha_dict, member_skills) -> list:
    if rated_buyers_list:
        return sorted(
            rated_buyers_list, key=lambda m: alpha_dict.get(m, 0), reverse=True
        )[:3]
    skill_members = [m for m in buyers if m in (member_skills or {})]
    return skill_members[:3] if skill_members else list(buyers[:3])


def _buyer_label(num_top: int, num_buyers: int) -> str:
    return f"Top {num_top} of {num_buyers}" if num_buyers > 3 else f"{num_buyers}"
