"""Score a ticker by its buyer composition.

`score_ticker_by_buyers` combines a ticker's disclosed buyers into a signal.
The safe default is an identity-free consensus score based only on distinct
recent buyers and recency. Historical member effects are descriptive,
noncausal opt-ins; Bayesian probability-times-alpha and solo posterior gates
are not tradable scores.
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
    _validate_scoring_mode,
)

CONSENSUS_SCORER_PROVENANCE = "identity_free_distinct_buyer_recency_v1"


@df_memoize(copy=False)
def score_ticker_by_buyers(
    ticker: str,
    transactions_df: pd.DataFrame,
    signals_df: pd.DataFrame | None = None,
    horizon: int = 90,
    threshold: float = 5.0,
    member_rankings: pd.DataFrame | None = None,
    min_buyers: int = 2,
    ticker_perf_signals: pd.DataFrame | None = None,
    _bayes_prior_strength: float | None = None,
    _ranking_dicts: dict | None = None,
    scoring_mode: str = "consensus",
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score a ticker by its buyer composition. Memoized via @df_memoize.

    When ``_ranking_dicts`` is provided (pre-built by the caller), dict
    lookups replace DataFrame linear scans for buyer stats.
    """
    _validate_scoring_mode(scoring_mode)
    _validate_inputs(signals_df, transactions_df, scoring_mode)

    if scoring_mode == "consensus" and as_of_date is None:
        raise AnalysisError("consensus scoring requires an explicit as_of_date")
    if scoring_mode == "consensus" and pd.isna(pd.Timestamp(as_of_date)):
        raise AnalysisError("consensus as_of_date must be a valid timestamp")
    if scoring_mode != "consensus" and member_rankings is None:
        bayes_prior = (
            _bayes_prior_strength
            if _bayes_prior_strength is not None
            else _signals.BAYES_PRIOR_STRENGTH
        )
        member_rankings = rank_members(
            signals_df, horizon, threshold, _bayes_prior_strength=bayes_prior
        )

    ticker_trades = _get_ticker_purchases(ticker, transactions_df).copy()
    if scoring_mode == "consensus":
        disclosure_dates = pd.to_datetime(
            ticker_trades["disclosure_date"], errors="coerce"
        )
        ticker_trades = ticker_trades[
            disclosure_dates.notna() & (disclosure_dates <= pd.Timestamp(as_of_date))
        ].copy()
    if ticker_trades.empty:
        return _empty_ticker_result(ticker)

    ticker_trades["_member_canonical"] = ticker_trades["member"].map(
        canonical_member_key
    )
    min_trades = ticker_trades["_member_canonical"].nunique()
    if min_trades < min_buyers:
        return _below_threshold_result(ticker, min_trades, min_buyers)

    buyers = ticker_trades["_member_canonical"].unique()
    if scoring_mode == "consensus":
        alpha_dict = {}
        inputs = _consensus_inputs(
            buyers, ticker_trades, as_of_date=pd.Timestamp(as_of_date)
        )
    else:
        rd = (
            _ranking_dicts
            if _ranking_dicts is not None
            else _build_ranking_dicts(member_rankings, scoring_mode=scoring_mode)
        )
        dict_mode = rd.get("mode")
        if dict_mode != scoring_mode:
            raise AnalysisError(
                "_ranking_dicts must declare the same validated scoring_mode"
            )
        alpha_dict = rd["alpha"]
        trades_dict = rd["trades"]
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
    inputs["scoring_mode"] = scoring_mode
    return _final_result(ticker, buyers, ticker_trades, inputs, alpha_dict)


def _validate_inputs(
    signals_df: pd.DataFrame | None,
    transactions_df: pd.DataFrame,
    scoring_mode: str,
) -> None:
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")
    if scoring_mode != "consensus" and (signals_df is None or signals_df.empty):
        raise AnalysisError("Historical scoring requires a non-empty signal dataframe")


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


def _consensus_inputs(
    buyers, ticker_trades: pd.DataFrame, *, as_of_date: pd.Timestamp
) -> dict:
    """Build an identity-free distinct-buyer recency score.

    Each canonical buyer contributes its most recent disclosure weight. Names,
    historical returns, trade counts, and member posteriors do not enter the
    score, so permuting member identities leaves it unchanged.
    """
    member_col = "_member_canonical"
    disclosures = (
        ticker_trades.assign(
            _disclosure=pd.to_datetime(
                ticker_trades["disclosure_date"], errors="coerce"
            )
        )
        .groupby(member_col)["_disclosure"]
        .max()
        .reindex(buyers)
    )
    days_since = (as_of_date - disclosures).dt.days
    if days_since.isna().any() or (days_since < 0).any():
        raise AnalysisError(
            "Consensus disclosures must be known on or before as_of_date"
        )
    weights = np.exp(-_signals.BUYER_RECENCY_DECAY * days_since.to_numpy(dtype=float))
    score = float(weights.sum())
    return {
        "base_signal_score": score,
        "rated_buyers_list": list(buyers),
        "best_rank": 1.0,
        "total_trades": len(buyers),
        "rated_buyers": len(buyers),
        "quality_adjusted_avg": score,
    }


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
    alpha_dict,
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
    signal_score = round(signal_score_raw, 2)

    top_buyers = _top_buyers_for_label(buyers, rated_buyers_list, alpha_dict)
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
            "fallback_source": [inputs.get("scoring_mode", "member_ranked")],
            "scoring_mode": [inputs.get("scoring_mode", "custom")],
            "scorer_provenance": [
                CONSENSUS_SCORER_PROVENANCE
                if inputs.get("scoring_mode") == "consensus"
                else "descriptive_member_skill_v1"
            ],
        }
    )


def _top_buyers_for_label(buyers, rated_buyers_list, alpha_dict) -> list:
    if not rated_buyers_list:
        return list(buyers[:3])
    if not alpha_dict:
        return list(rated_buyers_list[:3])
    return sorted(rated_buyers_list, key=lambda m: alpha_dict.get(m, 0), reverse=True)[
        :3
    ]


def _buyer_label(num_top: int, num_buyers: int) -> str:
    return f"Top {num_top} of {num_buyers}" if num_buyers > 3 else f"{num_buyers}"
