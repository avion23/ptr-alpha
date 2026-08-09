"""Chronological candidate selection with a final untouched holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from member_profitability.config import (
    BUYER_LOOKBACK_DAYS,
    HORIZON,
    MIN_BUYERS_VALUES,
    MIN_HOLDOUT_DECISION_DATES,
    ROBUST_P_VALUE,
    TARGET_RETURN_COLUMN,
    TOP_N_VALUES,
)
from member_profitability.walk_forward import _rank_train, _slice_window


def position_sizing_grid_search(sigs: pd.DataFrame, windows: list[dict]) -> dict:
    """Tune candidate-count parameters before evaluating one final holdout.

    The final chronological window never participates in parameter selection.
    Returned recommendations include their decision dates and use only buyers
    disclosed on or before each date.
    """
    if len(windows) < 2:
        return _empty_research_result("insufficient_windows")

    selection_windows = windows[:-1]
    holdout_window = windows[-1]
    selection_grid: list[dict] = []
    for top_n in TOP_N_VALUES:
        for min_buyers in MIN_BUYERS_VALUES:
            recommendations = _recommendations_for_windows(
                sigs, selection_windows, top_n, min_buyers
            )
            summary = _summarize_recommendations(recommendations)
            summary.update({"top_n": top_n, "min_buyers": min_buyers})
            selection_grid.append(summary)

    eligible = [row for row in selection_grid if row["n_decision_dates"] > 0]
    if not eligible:
        result = _empty_research_result("no_selection_candidates")
        result["selection_grid"] = selection_grid
        result["selection_window_count"] = len(selection_windows)
        return result

    selected = max(
        eligible,
        key=lambda row: (row["mean_excess_return_pct"], row["n_decision_dates"]),
    )
    holdout_recommendations = _recommendations_for_windows(
        sigs,
        [holdout_window],
        int(selected["top_n"]),
        int(selected["min_buyers"]),
    )
    holdout = _summarize_recommendations(holdout_recommendations)
    status = _holdout_status(holdout)
    return {
        "selection_window_count": len(selection_windows),
        "holdout_window": _serialize_window(holdout_window),
        "selection_grid": selection_grid,
        "selected_candidate": selected,
        "holdout": holdout,
        "holdout_recommendations": _serialize_recommendations(holdout_recommendations),
        "status": status,
    }


def _recommendations_for_windows(
    sigs: pd.DataFrame,
    windows: list[dict],
    top_n: int,
    min_buyers: int,
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    for window_index, window in enumerate(windows):
        train_sigs, test_sigs = _slice_window(sigs, window)
        rankings = _rank_train(train_sigs)
        if rankings.empty:
            continue
        recommendations = _timestamped_recommendations(
            test_sigs,
            rankings,
            top_n,
            min_buyers,
        )
        if recommendations.empty:
            continue
        recommendations["window_index"] = window_index
        results.append(recommendations)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def _timestamped_recommendations(
    test_sigs: pd.DataFrame,
    train_rankings: pd.DataFrame,
    top_n: int,
    min_buyers: int,
) -> pd.DataFrame:
    purchases = test_sigs[
        (test_sigs["signal_type"] == "Purchase")
        & test_sigs[TARGET_RETURN_COLUMN].notna()
    ].copy()
    if purchases.empty:
        return pd.DataFrame()
    purchases["disclosure_date"] = pd.to_datetime(purchases["disclosure_date"]).dt.normalize()
    score_by_member = train_rankings.set_index("member")[
        "shrunk_excess_return_pct"
    ].to_dict()

    rows: list[dict] = []
    last_decision: pd.Timestamp | None = None
    for decision_date in sorted(purchases["disclosure_date"].unique()):
        decision_date = pd.Timestamp(decision_date)
        if last_decision is not None and (
            decision_date - last_decision
        ).days < HORIZON:
            continue
        day = purchases[purchases["disclosure_date"] == decision_date]
        lookback_start = decision_date - pd.Timedelta(days=BUYER_LOOKBACK_DAYS)
        known = purchases[
            (purchases["disclosure_date"] >= lookback_start)
            & (purchases["disclosure_date"] <= decision_date)
        ]
        candidates: list[dict] = []
        for ticker in sorted(day["ticker"].dropna().unique()):
            ticker_known = known[known["ticker"] == ticker]
            buyers = sorted(set(ticker_known["member"]))
            buyer_scores = [score_by_member[b] for b in buyers if b in score_by_member]
            if len(buyer_scores) < min_buyers:
                continue
            event_returns = day.loc[
                day["ticker"] == ticker, TARGET_RETURN_COLUMN
            ].dropna()
            if event_returns.empty:
                continue
            score = float(np.mean(buyer_scores) * np.log1p(len(buyer_scores)))
            if score <= 0:
                continue
            candidates.append(
                {
                    "decision_date": decision_date,
                    "ticker": ticker,
                    "rated_buyers": len(buyer_scores),
                    "score": score,
                    "realized_excess_return_pct": float(event_returns.mean()),
                }
            )
        if not candidates:
            continue
        selected = (
            pd.DataFrame(candidates)
            .sort_values(["score", "ticker"], ascending=[False, True])
            .head(top_n)
        )
        rows.extend(selected.to_dict("records"))
        # Outcomes of consecutive decisions must not overlap.
        last_decision = decision_date
    return pd.DataFrame(rows)


def _summarize_recommendations(recommendations: pd.DataFrame) -> dict:
    if recommendations.empty:
        return {
            "n_recommendations": 0,
            "n_decision_dates": 0,
            "mean_excess_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "one_sided_p_value": 1.0,
        }
    per_date = recommendations.groupby("decision_date")[
        "realized_excess_return_pct"
    ].mean()
    mean_return = float(per_date.mean())
    p_value = 1.0
    if len(per_date) >= 2 and float(per_date.std(ddof=1)) > 0:
        test = stats.ttest_1samp(per_date, popmean=0.0, alternative="greater")
        p_value = float(test.pvalue)
    return {
        "n_recommendations": int(len(recommendations)),
        "n_decision_dates": int(len(per_date)),
        "mean_excess_return_pct": round(mean_return, 4),
        "win_rate_pct": round(float((per_date > 0).mean() * 100), 1),
        "one_sided_p_value": round(p_value, 6),
    }


def _holdout_status(holdout: dict) -> str:
    if holdout["n_decision_dates"] == 0:
        return "no_holdout_recommendations"
    if holdout["mean_excess_return_pct"] <= 0:
        return "nonpositive_holdout"
    if (
        holdout["n_decision_dates"] < MIN_HOLDOUT_DECISION_DATES
        or holdout["one_sided_p_value"] >= ROBUST_P_VALUE
    ):
        return "positive_holdout_not_robust"
    return "positive_holdout_prespecified_test_passed"


def _serialize_recommendations(recommendations: pd.DataFrame) -> list[dict]:
    if recommendations.empty:
        return []
    result = recommendations.copy()
    result["decision_date"] = pd.to_datetime(result["decision_date"]).dt.strftime("%Y-%m-%d")
    return result.to_dict("records")


def _serialize_window(window: dict) -> dict:
    return {key: str(pd.Timestamp(value).date()) for key, value in window.items()}


def _empty_research_result(status: str) -> dict:
    return {
        "selection_window_count": 0,
        "holdout_window": None,
        "selection_grid": [],
        "selected_candidate": None,
        "holdout": _summarize_recommendations(pd.DataFrame()),
        "holdout_recommendations": [],
        "status": status,
    }
