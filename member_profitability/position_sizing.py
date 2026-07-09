"""Position-sizing grid search: pick top_n tickers by composite buyer score.

For each (top_n, min_buyers) combo, walks through every window, ranks test-
period tickers by their buyer-score (mean shrunk_alpha across rated buyers),
picks the top_n, and accumulates realized alpha. Reports per-combo stats
plus the best by sharpe_proxy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from member_profitability.config import (
    MIN_BUYERS_VALUES,
    TOP_N_VALUES,
)
from member_profitability.walk_forward import _slice_window, _rank_train


def position_sizing_grid_search(sigs: pd.DataFrame, windows: list[dict]) -> list[dict]:
    """Run a grid search over (top_n, min_buyers) for the position-sizing recommendation.

    Returns a list of per-combo result dicts (one per top_n × min_buyers combo).
    """
    position_results: list[dict] = []
    for top_n in TOP_N_VALUES:
        for min_buyers in MIN_BUYERS_VALUES:
            window_returns, window_wins, window_total = _evaluate_grid(
                sigs, windows, top_n, min_buyers,
            )
            if window_returns:
                position_results.append(_summarize_grid_result(
                    window_returns, window_wins, window_total, top_n, min_buyers,
                ))
    return position_results


def _evaluate_grid(
    sigs: pd.DataFrame,
    windows: list[dict],
    top_n: int,
    min_buyers: int,
) -> tuple[list[float], int, int]:
    window_returns: list[float] = []
    window_wins = 0
    window_total = 0
    for w in windows:
        train_sigs, test_sigs = _slice_window(sigs, w)
        if train_sigs.empty or test_sigs.empty:
            continue

        train_rankings = _rank_train(train_sigs)
        if train_rankings.empty:
            continue

        ticker_scores = _score_test_tickers(test_sigs, train_rankings, min_buyers)
        if not ticker_scores:
            continue

        wins, total = _accumulate_picks(ticker_scores, top_n, window_returns)
        window_wins += wins
        window_total += total
    return window_returns, window_wins, window_total


def _score_test_tickers(
    test_sigs: pd.DataFrame,
    train_rankings: pd.DataFrame,
    min_buyers: int,
) -> list[dict]:
    """For each ticker in the test period, score by mean shrunk_alpha across
    its buyers (filtered to those present in the train rankings)."""
    test_purchases = test_sigs[test_sigs["signal_type"] == "Purchase"].copy()
    if test_purchases.empty:
        return []

    ticker_scores: list[dict] = []
    for ticker, t_grp in test_purchases.groupby("ticker"):
        buyers = t_grp["member"].unique()
        if len(buyers) < min_buyers:
            continue

        buyer_scores = _buyer_alphas(buyers, train_rankings)
        if not buyer_scores:
            continue

        avg_score = float(np.mean(buyer_scores))
        best_score = max(buyer_scores)
        count_bonus = float(np.log1p(len(buyers)))
        composite = avg_score * count_bonus

        ticker_scores.append({
            "ticker": ticker,
            "n_buyers": len(buyers),
            "avg_buyer_alpha": avg_score,
            "best_buyer_alpha": best_score,
            "composite_score": composite,
            "test_returns": t_grp["spy_alpha_pct"].dropna().tolist(),
        })
    return ticker_scores


def _buyer_alphas(buyers: np.ndarray, train_rankings: pd.DataFrame) -> list[float]:
    """Look up each buyer's shrunk_alpha from the train rankings."""
    buyer_scores: list[float] = []
    for b in buyers:
        match = train_rankings[train_rankings["member"] == b]
        if not match.empty and "shrunk_alpha" in match.columns:
            buyer_scores.append(float(match["shrunk_alpha"].iloc[0]))
    return buyer_scores


def _accumulate_picks(
    ticker_scores: list[dict],
    top_n: int,
    window_returns: list[float],
) -> tuple[int, int]:
    wins = 0
    total = 0
    score_df = pd.DataFrame(ticker_scores)
    score_df = score_df.sort_values("composite_score", ascending=False).head(top_n)
    for _, row in score_df.iterrows():
        if not row["test_returns"]:
            continue
        avg_ret = float(np.mean(row["test_returns"]))
        window_returns.append(avg_ret)
        total += 1
        if avg_ret > 0:
            wins += 1
    return wins, total


def _summarize_grid_result(
    window_returns: list[float],
    window_wins: int,
    window_total: int,
    top_n: int,
    min_buyers: int,
) -> dict:
    avg_ret = float(np.mean(window_returns)) if window_returns else 0.0
    std_ret = float(np.std(window_returns)) if len(window_returns) > 1 else 0.0
    sharpe = avg_ret / std_ret if std_ret > 0 else 0.0
    win_rate = window_wins / window_total * 100 if window_total > 0 else 0
    max_dd = float(np.min(window_returns)) if window_returns else 0.0

    return {
        "top_n": top_n,
        "min_buyers": min_buyers,
        "total_picks": window_total,
        "avg_spy_alpha_pct": round(avg_ret, 4),
        "std_spy_alpha_pct": round(std_ret, 4),
        "sharpe_proxy": round(float(sharpe), 4),
        "win_rate_pct": round(win_rate, 1),
        "worst_pick_alpha": round(max_dd, 4),
    }
