"""Non-overlapping walk-forward evaluation with identical scheduled support."""

from __future__ import annotations

import hashlib

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
    """Evaluate every scheduled period on identical non-overlapping support.

    A normal rejection is represented as cash (zero strategy return) and retains
    same-date SPY opportunity return. Unexpected failures propagate. Endpoint-only
    observations never trigger a drawdown stop because intraperiod NAV is absent.
    """
    del signals_df, transactions_df
    if max_dd_pct is not None:
        raise ValueError(
            "Endpoint-only drawdown stops were removed; daily NAV is required"
        )
    _validate_non_overlapping_periods(precomputed)

    all_returns: list[dict] = []
    period_results: list[dict] = []
    rejection_ledger: list[dict] = []

    for data in precomputed.values():
        as_of_date = data["as_of_ts"].date()
        reason = _period_rejection_reason(data, min_buyers)
        if reason is not None:
            period = _cash_period(data, prices_df, reason)
            all_returns.append(period)
            period_results.append({**period, "status": "cash", "reason": reason})
            rejection_ledger.append(
                {"as_of_date": as_of_date, "stage": "period", "reason": reason}
            )
            continue

        period_return, detail, rejections = _run_one_period(
            data, min_buyers, scoring_fn, prices_df, top_n, allocation
        )
        rejection_ledger.extend(
            {"as_of_date": as_of_date, **rejection} for rejection in rejections
        )
        if period_return is None:
            reason = detail["reason"]
            period = _cash_period(data, prices_df, reason)
            all_returns.append(period)
            period_results.append({**period, "status": "cash", "reason": reason})
            rejection_ledger.append(
                {"as_of_date": as_of_date, "stage": "period", "reason": reason}
            )
            continue

        all_returns.append(period_return)
        period_results.append(
            {**detail, **period_return, "status": "invested", "reason": None}
        )

    support = [row["as_of_date"].isoformat() for row in all_returns]
    metrics = summarize_walk_forward(
        all_returns, periods_per_year=365.0 / _horizon_from(precomputed)
    )
    return {
        **metrics,
        "period_results": period_results,
        "rejection_ledger": rejection_ledger,
        "requested_periods": len(precomputed),
        "coverage_pct": 100.0 if len(all_returns) == len(precomputed) else 0.0,
        "support_dates": support,
        "support_sha256": hashlib.sha256("|".join(support).encode()).hexdigest(),
    }


def _period_rejection_reason(data: dict, min_buyers: int) -> str | None:
    if data.get("status") != "ready":
        return str(data.get("reason", "period_not_ready"))
    if not data["candidate_tickers"].get(min_buyers):
        return "no_candidates_for_min_buyers"
    return None


def _cash_period(data: dict, prices_df: pd.DataFrame, reason: str) -> dict:
    del reason
    spy_return = _spy_period_return(prices_df, data["as_of_ts"], data["horizon"])
    return {
        "as_of_date": data["as_of_ts"].date(),
        "portfolio_return_pct": 0.0,
        "spy_return_pct": spy_return,
        "n_positions": 0,
    }


def _spy_period_return(prices_df, as_of_ts, horizon) -> float:
    evaluated = evaluate_backtest(
        pd.DataFrame({"ticker": ["SPY"], "signal_score": [1.0]}),
        prices_df,
        as_of_ts,
        horizon,
    ).dropna(subset=["bt_return_pct"])
    if evaluated.empty:
        raise RuntimeError(f"SPY support unavailable for scheduled date {as_of_ts}")
    return float(evaluated["bt_return_pct"].iloc[0])


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
    return {
        "alpha": alpha,
        "trades": trades,
        "prob": probability,
        "has_shrunk": True,
    }


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
