"""Non-overlapping walk-forward evaluation with auditable coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.backtest import evaluate_backtest
from analyzer.member_names import canonical_member_key
from analyzer.member_ranking import score_ticker_by_buyers

from optimize_profit.metrics import summarize_walk_forward


def _score_candidates(
    candidate_tickers,
    recent_trades,
    training,
    horizon,
    threshold,
    member_rankings,
    min_buyers,
    ticker_perf_signals,
    custom_ranking_dicts,
):
    """Score candidates and return both accepted rows and explicit rejections."""
    scores: list[dict] = []
    rejections: list[dict] = []
    for ticker in candidate_tickers:
        score_df = score_ticker_by_buyers(
            ticker,
            recent_trades,
            training,
            horizon,
            threshold,
            member_rankings,
            min_buyers,
            ticker_perf_signals=ticker_perf_signals,
            solo_buyer_skill_threshold=0.0,
            _ranking_dicts=custom_ranking_dicts,
        )
        if score_df.empty:
            rejections.append(
                {"stage": "candidate", "ticker": ticker, "reason": "empty_score"}
            )
            continue
        score = float(score_df["signal_score"].iloc[0])
        if not np.isfinite(score) or score <= 0:
            reason = score_df.get("note", pd.Series(["non_positive_score"])).iloc[0]
            rejections.append(
                {
                    "stage": "candidate",
                    "ticker": ticker,
                    "reason": str(reason),
                    "signal_score": score,
                }
            )
            continue
        scores.append({column: score_df[column].iloc[0] for column in score_df.columns})
    return scores, rejections


def run_walk_forward(
    signals_df,
    transactions_df,
    prices_df,
    precomputed,
    scoring_fn,
    top_n,
    min_buyers,
    allocation,
    max_dd_pct=None,
):
    """Evaluate one frozen configuration on non-overlapping periods.

    A period is one fully invested portfolio held for ``horizon`` calendar days.
    Requested dates must be at least one horizon apart; this prevents reuse of the
    same bankroll across overlapping vintages. Returned details contain every
    executed period and every normal rejection. Unexpected failures propagate.
    """
    del signals_df, transactions_df  # inputs retained for API compatibility
    _validate_non_overlapping_periods(precomputed)

    all_returns: list[dict] = []
    period_results: list[dict] = []
    rejection_ledger: list[dict] = []
    cumulative_wealth = 1.0
    peak_wealth = 1.0
    stopped = False

    for as_of_iso, data in precomputed.items():
        as_of_date = data["as_of_ts"].date()
        if stopped:
            rejection_ledger.append(
                {
                    "as_of_date": as_of_date,
                    "stage": "period",
                    "reason": "drawdown_stop_active",
                }
            )
            continue
        if data.get("status") != "ready":
            rejection_ledger.append(
                {
                    "as_of_date": as_of_date,
                    "stage": "precompute",
                    "reason": data.get("reason", "period_not_ready"),
                    "detail": data.get("detail"),
                }
            )
            continue
        if not data["candidate_tickers"].get(min_buyers):
            rejection_ledger.append(
                {
                    "as_of_date": as_of_date,
                    "stage": "period",
                    "reason": "no_candidates_for_min_buyers",
                }
            )
            continue

        period_return, period_detail, period_rejections = _run_one_period(
            data, min_buyers, scoring_fn, prices_df, top_n, allocation
        )
        for rejection in period_rejections:
            rejection_ledger.append({"as_of_date": as_of_date, **rejection})
        if period_return is None:
            rejection_ledger.append(
                {
                    "as_of_date": as_of_date,
                    "stage": "period",
                    "reason": period_detail["reason"],
                }
            )
            continue

        all_returns.append(period_return)
        cumulative_wealth, peak_wealth, stopped = _track_drawdown(
            cumulative_wealth,
            peak_wealth,
            period_return["portfolio_return_pct"] / 100,
            max_dd_pct,
        )
        period_results.append(
            {
                **period_detail,
                **period_return,
                "ending_wealth": cumulative_wealth,
                "drawdown_stop_triggered": stopped,
            }
        )

    metrics = summarize_walk_forward(
        all_returns,
        stopped,
        periods_per_year=365.0 / _horizon_from(precomputed),
    )
    return {
        **metrics,
        "period_results": period_results,
        "rejection_ledger": rejection_ledger,
        "requested_periods": len(precomputed),
        "coverage_pct": round(100 * len(all_returns) / len(precomputed), 1)
        if precomputed
        else 0.0,
    }


def _run_one_period(data, min_buyers, scoring_fn, prices_df, top_n, allocation):
    custom_ranking_dicts = _build_custom_ranking_dicts(
        data["member_rankings"], scoring_fn
    )
    candidates = data["candidate_tickers"].get(min_buyers, [])
    scored, rejections = _score_candidates(
        candidate_tickers=candidates,
        recent_trades=data["recent_trades"],
        training=data["training"],
        horizon=data["horizon"],
        threshold=5.0,
        member_rankings=data["member_rankings"],
        min_buyers=min_buyers,
        ticker_perf_signals=data["ticker_perf_signals"],
        custom_ranking_dicts=custom_ranking_dicts,
    )
    if not scored:
        return None, {"reason": "no_positive_scores"}, rejections

    result = (
        pd.DataFrame(scored)
        .sort_values("signal_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    evaluated = evaluate_backtest(result, prices_df, data["as_of_ts"], data["horizon"])
    valid = evaluated.dropna(subset=["bt_return_pct"]).copy()
    for _, row in evaluated[evaluated["bt_return_pct"].isna()].iterrows():
        rejections.append(
            {
                "stage": "evaluation",
                "ticker": row.get("ticker"),
                "reason": "missing_evaluation_return",
            }
        )
    if valid.empty:
        return None, {"reason": "no_evaluable_positions"}, rejections

    weights = _portfolio_weights(valid["signal_score"].to_numpy(), allocation)
    port_ret = _portfolio_return(valid["bt_return_pct"].to_numpy(), weights)
    spy_ret = _portfolio_return(valid["bt_spy_return_pct"].to_numpy(), weights)
    detail = {
        "candidate_count": len(candidates),
        "positive_score_count": len(scored),
        "n_positions": len(valid),
        "selected_tickers": ",".join(valid["ticker"].astype(str)),
        "n_no_price": int(evaluated.attrs.get("n_no_price", 0)),
        "n_delisted": int(evaluated.attrs.get("n_delisted", 0)),
    }
    return (
        {
            "as_of_date": data["as_of_ts"].date(),
            "portfolio_return_pct": port_ret * 100,
            "spy_return_pct": spy_ret * 100,
            "n_positions": len(valid),
        },
        detail,
        rejections,
    )


def _build_custom_ranking_dicts(member_rankings, scoring_fn) -> dict:
    alpha = scoring_fn(member_rankings)
    trades = dict(
        zip(
            member_rankings["member"],
            member_rankings["purchase_trades"].fillna(0).astype(int),
        )
    )
    probability = (
        dict(
            zip(
                member_rankings["member"],
                member_rankings["bayes_win_prob"].fillna(0.5).astype(float),
            )
        )
        if "bayes_win_prob" in member_rankings.columns
        else {}
    )
    for lookup in (alpha, trades, probability):
        aliases = {
            canonical_member_key(str(member)): value
            for member, value in lookup.items()
            if canonical_member_key(str(member)) not in lookup
        }
        lookup.update(aliases)
    return {"alpha": alpha, "trades": trades, "prob": probability, "has_shrunk": True}


def _portfolio_weights(signal_scores, allocation: str) -> np.ndarray:
    n = len(signal_scores)
    if allocation == "signal":
        signals = np.maximum(signal_scores.astype(float), 0)
        total = signals.sum()
        if total > 0:
            return signals / total
    if allocation != "equal":
        raise ValueError(f"Unknown allocation: {allocation}")
    return np.ones(n) / n


def _portfolio_return(return_pcts: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * return_pcts.astype(float) / 100))


def _track_drawdown(cumulative_wealth, peak_wealth, port_ret, max_dd_pct):
    cumulative_wealth *= 1 + port_ret
    peak_wealth = max(peak_wealth, cumulative_wealth)
    current_dd = (cumulative_wealth - peak_wealth) / peak_wealth * 100
    stopped = max_dd_pct is not None and current_dd <= -abs(max_dd_pct)
    return cumulative_wealth, peak_wealth, stopped


def _validate_non_overlapping_periods(precomputed: dict) -> None:
    if len(precomputed) < 2:
        return
    dates = sorted(pd.Timestamp(data["as_of_ts"]) for data in precomputed.values())
    horizon = _horizon_from(precomputed)
    gaps = np.diff(np.array([date.value for date in dates], dtype=np.int64))
    min_gap_days = float(gaps.min() / 86_400_000_000_000)
    if min_gap_days < horizon:
        raise ValueError(
            f"Overlapping bankroll periods are forbidden: minimum gap "
            f"{min_gap_days:.0f}d < horizon {horizon}d"
        )


def _horizon_from(precomputed: dict) -> int:
    if not precomputed:
        return 90
    horizons = {int(data["horizon"]) for data in precomputed.values()}
    if len(horizons) != 1:
        raise ValueError(f"Mixed horizons are unsupported: {sorted(horizons)}")
    return horizons.pop()
