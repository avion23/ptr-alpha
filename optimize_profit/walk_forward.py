"""Walk-forward backtest engine: per-as_of_date scoring + portfolio return.

For each as_of_date:
  1. Score candidate tickers using the provided scoring function
  2. Evaluate entry/exit via backtest
  3. Apply portfolio allocation (equal-weight or signal-weighted)
  4. Track drawdown; stop early if it exceeds the threshold

The expensive precomputation (filters, member rankings) is done once in
`precompute.py` and shared across all scoring-function combinations.
"""

import numpy as np
import pandas as pd

from analyzer.backtest import evaluate_backtest
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
    """Score candidate tickers using custom ranking dicts. Returns list of dicts."""
    scores = []
    for ticker in candidate_tickers:
        try:
            score_df = score_ticker_by_buyers(
                ticker,
                recent_trades,
                training,
                horizon,
                threshold,
                member_rankings,
                min_buyers,
                ticker_perf_signals=ticker_perf_signals,
                _ranking_dicts=custom_ranking_dicts,
            )
        except Exception:
            continue

        if score_df.empty or score_df["signal_score"].iloc[0] <= 0:
            continue

        row = {c: score_df[c].iloc[0] for c in score_df.columns}
        scores.append(row)

    return scores


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
    """Run walk-forward backtest with custom scoring and allocation.

    Parameters
    ----------
    precomputed : dict
        Per-as_of_date precomputed data from ``precompute_walk_forward_data``.
    """
    all_returns = []
    cumulative_wealth = 1.0
    peak_wealth = 1.0
    stopped = False

    for as_of_iso, data in precomputed.items():
        if stopped:
            break

        if not _period_has_data(data, min_buyers):
            continue

        period_return, custom_ranking_dicts, period_stopped = _run_one_period(
            data, min_buyers, scoring_fn, prices_df, top_n, allocation,
        )
        if period_return is None:
            continue
        if period_stopped:
            stopped = True
            break

        all_returns.append(period_return)
        cumulative_wealth, peak_wealth, _ = _track_drawdown(
            cumulative_wealth, peak_wealth, period_return["portfolio_return_pct"] / 100, max_dd_pct,
        )

    return summarize_walk_forward(all_returns, stopped)


def _run_one_period(
    data, min_buyers, scoring_fn, prices_df, top_n, allocation,
) -> tuple[dict | None, dict | None, bool]:
    """Run one as_of_date iteration. Returns (period_return_or_None,
    custom_ranking_dicts_or_None, stopped_flag)."""
    custom_ranking_dicts = _build_custom_ranking_dicts(data["member_rankings"], scoring_fn)

    scored = _score_candidates(
        candidate_tickers=data["candidate_tickers"].get(min_buyers, []),
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
        return None, custom_ranking_dicts, False

    result = pd.DataFrame(scored)
    result = (
        result.sort_values("signal_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", range(1, len(result) + 1))

    try:
        evaluated = evaluate_backtest(result, prices_df, data["as_of_ts"], data["horizon"])
        evaluated = evaluated.dropna(subset=["bt_return_pct"])
    except Exception:
        return None, custom_ranking_dicts, False

    if evaluated.empty:
        return None, custom_ranking_dicts, False

    weights = _portfolio_weights(evaluated["signal_score"].values, allocation)
    port_ret = _portfolio_return(evaluated["bt_return_pct"].values, weights)

    return (
        {
            "as_of_date": data["as_of_ts"].date(),
            "portfolio_return_pct": port_ret * 100,
            "n_positions": len(evaluated),
        },
        custom_ranking_dicts,
        False,
    )


def _period_has_data(data: dict, min_buyers: int) -> bool:
    """Returns False if rankings empty or no candidates for the buyer threshold."""
    if data["member_rankings"] is None or data["member_rankings"].empty:
        return False
    if not data["candidate_tickers"].get(min_buyers):
        return False
    return True


def _build_custom_ranking_dicts(member_rankings, scoring_fn) -> dict:
    """Build the dict-of-dicts structure that `score_ticker_by_buyers`
    accepts as ``_ranking_dicts`` — replacing the default alpha lookup
    with the caller-supplied scoring function."""
    return {
        "alpha": scoring_fn(member_rankings),
        "trades": dict(
            zip(
                member_rankings["member"],
                member_rankings["purchase_trades"].fillna(0).astype(int),
            )
        ),
        "prob": (
            dict(
                zip(
                    member_rankings["member"],
                    member_rankings["bayes_win_prob"].fillna(0.5).astype(float),
                )
            )
            if "bayes_win_prob" in member_rankings.columns
            else {}
        ),
        "has_shrunk": True,
    }


def _portfolio_weights(signal_scores, allocation: str) -> np.ndarray:
    """Equal-weight or signal-weighted (proportional) portfolio weights."""
    n = len(signal_scores)
    if allocation == "signal":
        sigs = np.maximum(signal_scores.astype(float), 0)
        total = sigs.sum()
        if total > 0:
            return sigs / total
    return np.ones(n) / n


def _portfolio_return(bt_return_pcts: np.ndarray, weights: np.ndarray) -> float:
    """Total portfolio return for one period."""
    ticker_rets = bt_return_pcts.astype(float) / 100
    return float(np.sum(weights * ticker_rets))


def _track_drawdown(
    cumulative_wealth: float, peak_wealth: float,
    port_ret: float, max_dd_pct: float | None,
) -> tuple[float, float, bool]:
    """Update wealth/peak, decide whether to stop early on max drawdown."""
    cumulative_wealth *= 1 + port_ret
    peak_wealth = max(peak_wealth, cumulative_wealth)
    current_dd = (cumulative_wealth - peak_wealth) / peak_wealth * 100
    stopped = max_dd_pct is not None and current_dd < -max_dd_pct
    return cumulative_wealth, peak_wealth, stopped
