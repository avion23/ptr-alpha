"""Chronological candidate selection with retrospective validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from member_profitability.config import (
    BUYER_LOOKBACK_DAYS,
    HORIZON,
    MIN_BUYERS_VALUES,
    TARGET_RETURN_COLUMN,
    TOP_N_VALUES,
)
from member_profitability.walk_forward import _rank_train, _slice_window


def position_sizing_grid_search(sigs: pd.DataFrame, windows: list[dict]) -> dict:
    """Tune parameters, then report a selection-isolated retrospective window.

    The last chronological window does not participate in parameter selection
    in this run. The history has been used by earlier repository research, so
    this window is retrospective validation, not prospective evidence.
    Returned recommendations use only buyers disclosed by each decision date.
    """
    if len(windows) < 2:
        return _empty_research_result("insufficient_windows")

    selection_windows = windows[:-1]
    retrospective_validation_window = windows[-1]
    selection_grid: list[dict] = []
    for top_n in TOP_N_VALUES:
        for min_buyers in MIN_BUYERS_VALUES:
            recommendations = _recommendations_for_windows(
                sigs, selection_windows, top_n, min_buyers
            )
            summary = _summarize_recommendations(recommendations)
            summary.update({"top_n": top_n, "min_buyers": min_buyers})
            selection_grid.append(summary)

    eligible = [
        row for row in selection_grid if row["n_evaluable_decision_dates"] > 0
    ]
    if not eligible:
        result = _empty_research_result("no_selection_candidates")
        result["selection_grid"] = selection_grid
        result["selection_window_count"] = len(selection_windows)
        return result

    selected = max(
        eligible,
        key=lambda row: (
            row["mean_excess_return_pct"],
            row["n_evaluable_decision_dates"],
        ),
    )
    selected_recommendations = _recommendations_for_windows(
        sigs,
        selection_windows,
        int(selected["top_n"]),
        int(selected["min_buyers"]),
    )
    retrospective_validation_recommendations = _recommendations_for_windows(
        sigs,
        [retrospective_validation_window],
        int(selected["top_n"]),
        int(selected["min_buyers"]),
    )
    retrospective_validation = _summarize_recommendations(
        retrospective_validation_recommendations
    )
    validation_status = _retrospective_validation_status(retrospective_validation)
    return {
        "selection_window_count": len(selection_windows),
        "retrospective_validation_window": _serialize_window(
            retrospective_validation_window
        ),
        "selection_grid": selection_grid,
        "selected_candidate": selected,
        "selection_recommendations": _serialize_recommendations(
            selected_recommendations
        ),
        "retrospective_validation": retrospective_validation,
        "retrospective_validation_recommendations": _serialize_recommendations(
            retrospective_validation_recommendations
        ),
        "retrospective_validation_status": validation_status,
    }


def _recommendations_for_windows(
    sigs: pd.DataFrame,
    windows: list[dict],
    top_n: int,
    min_buyers: int,
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    for window_index, window in enumerate(windows):
        train_sigs, _ = _slice_window(sigs, window)
        rankings = _rank_train(train_sigs)
        if rankings.empty:
            continue
        test_events = _disclosed_test_events(sigs, window)
        recommendations = _timestamped_recommendations(
            test_events,
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
    combined = pd.concat(results, ignore_index=True)
    return _apply_global_decision_spacing(combined)


def _apply_global_decision_spacing(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Apply one execution clock across all chronological research windows."""
    if recommendations.empty:
        return recommendations.copy()
    result = recommendations.copy()
    result["decision_date"] = pd.to_datetime(result["decision_date"]).dt.normalize()
    accepted_dates: list[pd.Timestamp] = []
    last_accepted: pd.Timestamp | None = None
    for decision_date in sorted(result["decision_date"].unique()):
        decision_date = pd.Timestamp(decision_date)
        if last_accepted is not None and (
            decision_date - last_accepted
        ).days < HORIZON:
            continue
        accepted_dates.append(decision_date)
        last_accepted = decision_date
    # Keep every top-N row on an accepted date; spacing rejects whole dates.
    return (
        result[result["decision_date"].isin(accepted_dates)]
        .sort_values(["decision_date", "score", "ticker"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def _disclosed_test_events(sigs: pd.DataFrame, window: dict) -> pd.DataFrame:
    """Return all disclosed test events and mask labels immature at test_end."""
    disclosure = pd.to_datetime(sigs["disclosure_date"])
    events = sigs[
        (disclosure >= window["test_start"])
        & (disclosure < window["test_end"])
    ].copy()
    if events.empty:
        return events
    event_disclosure = pd.to_datetime(events["disclosure_date"])
    mature = event_disclosure + pd.Timedelta(days=HORIZON) <= window["test_end"]
    if "window_complete" in events.columns:
        mature &= events["window_complete"].fillna(False).astype(bool)
    events.loc[~mature, TARGET_RETURN_COLUMN] = np.nan
    return events


def _timestamped_recommendations(
    test_sigs: pd.DataFrame,
    train_rankings: pd.DataFrame,
    top_n: int,
    min_buyers: int,
) -> pd.DataFrame:
    """Select from disclosed events, then attach any available outcome labels."""
    purchases = test_sigs[test_sigs["signal_type"] == "Purchase"].copy()
    if purchases.empty:
        return pd.DataFrame()
    purchases["disclosure_date"] = pd.to_datetime(
        purchases["disclosure_date"]
    ).dt.normalize()
    score_by_member = train_rankings.set_index("member")[
        "shrunk_excess_return_pct"
    ].to_dict()

    rows: list[dict] = []
    for decision_date in sorted(purchases["disclosure_date"].unique()):
        decision_date = pd.Timestamp(decision_date)
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
            score = float(np.mean(buyer_scores) * np.log1p(len(buyer_scores)))
            if score <= 0:
                continue
            candidates.append(
                {
                    "decision_date": decision_date,
                    "ticker": ticker,
                    "rated_buyers": len(buyer_scores),
                    "score": score,
                }
            )
        if not candidates:
            continue
        selected = (
            pd.DataFrame(candidates)
            .sort_values(["score", "ticker"], ascending=[False, True])
            .head(top_n)
        )
        # Selection is complete before labels are read. Missing outcomes affect
        # evaluation coverage only, never eligibility, buyer counts, or rank.
        outcomes = day.groupby("ticker", dropna=False)[TARGET_RETURN_COLUMN].mean()
        selected["realized_excess_return_pct"] = selected["ticker"].map(outcomes)
        rows.extend(selected.to_dict("records"))
    return pd.DataFrame(rows)


def _summarize_recommendations(recommendations: pd.DataFrame) -> dict:
    if recommendations.empty:
        return _empty_summary()

    eligible_recommendations = int(len(recommendations))
    eligible_dates = int(recommendations["decision_date"].nunique())
    evaluable = recommendations[
        recommendations["realized_excess_return_pct"].notna()
    ]
    evaluable_recommendations = int(len(evaluable))
    evaluable_dates = int(evaluable["decision_date"].nunique())
    missing = recommendations[
        recommendations["realized_excess_return_pct"].isna()
    ]
    missing_dates = int(missing["decision_date"].nunique())
    base = {
        "n_eligible_recommendations": eligible_recommendations,
        "n_evaluable_recommendations": evaluable_recommendations,
        "n_missing_outcome_recommendations": (
            eligible_recommendations - evaluable_recommendations
        ),
        "n_eligible_decision_dates": eligible_dates,
        "n_evaluable_decision_dates": evaluable_dates,
        "n_missing_outcome_decision_dates": missing_dates,
        "n_unevaluable_decision_dates": eligible_dates - evaluable_dates,
        "evaluation_coverage_pct": round(
            evaluable_recommendations / eligible_recommendations * 100,
            1,
        ),
        # Backward-readable aliases. Their denominators are explicit above.
        "n_recommendations": eligible_recommendations,
        "n_decision_dates": evaluable_dates,
        "mean_excess_return_pct": 0.0,
        "win_rate_pct": 0.0,
        "one_sided_p_value": 1.0,
    }
    if evaluable.empty:
        return base

    per_date = evaluable.groupby("decision_date")[
        "realized_excess_return_pct"
    ].mean()
    mean_return = float(per_date.mean())
    p_value = 1.0
    if len(per_date) >= 2 and float(per_date.std(ddof=1)) > 0:
        test = stats.ttest_1samp(per_date, popmean=0.0, alternative="greater")
        p_value = float(test.pvalue)
    base.update(
        {
            "mean_excess_return_pct": round(mean_return, 4),
            "win_rate_pct": round(float((per_date > 0).mean() * 100), 1),
            "one_sided_p_value": round(p_value, 6),
        }
    )
    return base


def _empty_summary() -> dict:
    return {
        "n_eligible_recommendations": 0,
        "n_evaluable_recommendations": 0,
        "n_missing_outcome_recommendations": 0,
        "n_eligible_decision_dates": 0,
        "n_evaluable_decision_dates": 0,
        "n_missing_outcome_decision_dates": 0,
        "n_unevaluable_decision_dates": 0,
        "evaluation_coverage_pct": 0.0,
        "n_recommendations": 0,
        "n_decision_dates": 0,
        "mean_excess_return_pct": 0.0,
        "win_rate_pct": 0.0,
        "one_sided_p_value": 1.0,
    }


def _retrospective_validation_status(validation: dict) -> str:
    if validation["n_eligible_recommendations"] == 0:
        return "no_retrospective_validation_recommendations"
    if validation["n_evaluable_recommendations"] == 0:
        return "retrospective_validation_outcomes_unavailable"
    incomplete = validation["n_missing_outcome_recommendations"] > 0
    if validation["mean_excess_return_pct"] <= 0:
        return (
            "nonpositive_retrospective_validation_incomplete_coverage"
            if incomplete
            else "nonpositive_retrospective_validation"
        )
    if incomplete:
        return "positive_retrospective_validation_incomplete_coverage_not_robust"
    # This history was previously explored. Even a positive p-value here is
    # retrospective evidence and cannot establish robustness or profitability.
    return "positive_retrospective_validation_not_robust"


def _serialize_recommendations(recommendations: pd.DataFrame) -> list[dict]:
    if recommendations.empty:
        return []
    result = recommendations.copy()
    result["decision_date"] = pd.to_datetime(result["decision_date"]).dt.strftime("%Y-%m-%d")
    result = result.astype(object).where(pd.notna(result), None)
    return result.to_dict("records")


def _serialize_window(window: dict) -> dict:
    return {key: str(pd.Timestamp(value).date()) for key, value in window.items()}


def _empty_research_result(status: str) -> dict:
    return {
        "selection_window_count": 0,
        "retrospective_validation_window": None,
        "selection_grid": [],
        "selected_candidate": None,
        "selection_recommendations": [],
        "retrospective_validation": _summarize_recommendations(pd.DataFrame()),
        "retrospective_validation_recommendations": [],
        "retrospective_validation_status": status,
    }
