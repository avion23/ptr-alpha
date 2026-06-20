"""Optimize profit by finding best scoring function and portfolio allocation.

Walk-forward validation: train on expanding window, test on next period.
Avoids look-ahead bias. Scoring functions are continuous and differentiable.
"""

import sys

sys.argv = ["ptr-alpha"]  # prevent typer from parsing sweep args

import itertools
import time
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from scipy.special import expit, softplus as _softplus

from analyzer.database import Database
from analyzer import analysis
from analyzer import signals as signals_mod
from analyzer.member_ranking import (
    rank_members,
    score_ticker_by_buyers,
    _build_ranking_dicts,
)
from analyzer.backtest import (
    _filter_training,
    _filter_recent_trades,
    _filter_ticker_perf,
    _compute_ticker_ou_params,
    evaluate_backtest,
)


# ── Scoring functions (continuous, differentiable) ─────────────────────
# Each takes member_rankings DataFrame and returns {member: score}.
# Higher score = better (for ranking purposes).


def score_shrunk_alpha(member_rankings):
    """Baseline: shrunk_alpha (INVERTED — picks worst performers)."""
    return dict(zip(member_rankings["member"], member_rankings["shrunk_alpha"]))


def score_inverted_alpha(member_rankings):
    """Inverted shrunk_alpha — fixes the negative correlation."""
    return dict(zip(member_rankings["member"], -member_rankings["shrunk_alpha"]))


def score_trade_frequency(member_rankings):
    """log(1 + trade_count) — more active members score higher."""
    return dict(
        zip(member_rankings["member"], np.log1p(member_rankings["purchase_trades"]))
    )


def score_consistency(member_rankings):
    """prob_up * log(1 + trades) — consistent winners with volume."""
    prob = member_rankings["prob_up_given_buy"].values
    trades = np.log1p(member_rankings["purchase_trades"].values)
    return dict(zip(member_rankings["member"], prob * trades))


def score_bayesian_quality(member_rankings):
    """bayes_win_prob * shrunk_alpha — combined signal."""
    bayes = member_rankings["bayes_win_prob"].values
    alpha = member_rankings["shrunk_alpha"].values
    return dict(zip(member_rankings["member"], bayes * alpha))


def score_neg_bayesian_quality(member_rankings):
    """-bayes_win_prob * shrunk_alpha — inverted combined signal."""
    bayes = member_rankings["bayes_win_prob"].values
    alpha = member_rankings["shrunk_alpha"].values
    return dict(zip(member_rankings["member"], -bayes * alpha))


def score_smooth_trade_threshold(member_rankings):
    """Sigmoid-thresholded trade count weighted by prob_up.

    Uses expit for smooth thresholding at 5 trades (differentiable).
    """
    trades = member_rankings["purchase_trades"].values.astype(float)
    prob = member_rankings["prob_up_given_buy"].values
    trade_weight = expit((trades - 5) / 2.0)  # sigmoid centered at 5
    return dict(zip(member_rankings["member"], prob * trade_weight))


def score_softplus_quality(member_rankings):
    """Softplus-thresholded trades weighted by bayes_win_prob.

    softplus(x) = log(1 + exp(x)) — smooth approximation to ReLU.
    Threshold at 3 trades, normalized.
    """
    trades = member_rankings["purchase_trades"].values.astype(float)
    bayes = member_rankings["bayes_win_prob"].values
    # softplus(trades - 3) smoothly activates above 3 trades
    trade_signal = _softplus(trades - 3)
    norm = float(_softplus(np.array([10.0])).item())  # normalize by softplus(10)
    trade_weight = trade_signal / norm
    return dict(zip(member_rankings["member"], bayes * trade_weight))


def score_sharpe(member_rankings):
    """Sharpe ratio — risk-adjusted return."""
    return dict(zip(member_rankings["member"], member_rankings["sharpe_ratio"]))


SCORING_FUNCTIONS = {
    "shrunk_alpha": score_shrunk_alpha,
    "inverted_alpha": score_inverted_alpha,
    "trade_frequency": score_trade_frequency,
    "consistency": score_consistency,
    "bayesian_quality": score_bayesian_quality,
    "neg_bayesian_quality": score_neg_bayesian_quality,
    "smooth_trade_thresh": score_smooth_trade_threshold,
    "softplus_quality": score_softplus_quality,
    "sharpe": score_sharpe,
}


# ── Walk-forward backtest engine ───────────────────────────────────────


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

        training = data["training"]
        member_rankings = data["member_rankings"]
        recent_trades = data["recent_trades"]
        candidate_tickers_by_mb = data["candidate_tickers"]
        ticker_perf_signals = data["ticker_perf_signals"]
        as_of_ts = data["as_of_ts"]
        horizon = data["horizon"]

        if member_rankings is None or member_rankings.empty:
            continue

        candidates = candidate_tickers_by_mb.get(min_buyers, [])
        if not candidates:
            continue

        # Build custom ranking dicts with our scoring function
        custom_alpha_dict = scoring_fn(member_rankings)
        custom_ranking_dicts = {
            "alpha": custom_alpha_dict,
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

        # Score candidates
        scored = _score_candidates(
            candidates,
            recent_trades,
            training,
            horizon,
            5.0,
            member_rankings,
            min_buyers,
            ticker_perf_signals,
            custom_ranking_dicts,
        )

        if not scored:
            continue

        result = pd.DataFrame(scored)
        result = (
            result.sort_values("signal_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        result.insert(0, "rank", range(1, len(result) + 1))

        # Evaluate backtest
        try:
            evaluated = evaluate_backtest(result, prices_df, as_of_ts, horizon)
            evaluated = evaluated.dropna(subset=["bt_return_pct"])
        except Exception:
            continue

        if evaluated.empty:
            continue

        # Compute portfolio return based on allocation
        n = len(evaluated)
        if allocation == "equal":
            weights = np.ones(n) / n
        elif allocation == "signal":
            sigs = evaluated["signal_score"].values.astype(float)
            sigs = np.maximum(sigs, 0)
            total = sigs.sum()
            weights = sigs / total if total > 0 else np.ones(n) / n
        else:
            weights = np.ones(n) / n

        ticker_rets = evaluated["bt_return_pct"].values.astype(float) / 100
        port_ret = float(np.sum(weights * ticker_rets))

        all_returns.append(
            {
                "as_of_date": as_of_ts.date(),
                "portfolio_return_pct": port_ret * 100,
                "n_positions": n,
            }
        )

        # Track drawdown
        cumulative_wealth *= 1 + port_ret
        peak_wealth = max(peak_wealth, cumulative_wealth)
        current_dd = (cumulative_wealth - peak_wealth) / peak_wealth * 100

        if max_dd_pct is not None and current_dd < -max_dd_pct:
            stopped = True

    if not all_returns:
        return {"total_return_pct": 0, "sharpe": 0, "max_drawdown_pct": 0,
                "win_rate_pct": 0, "n_periods": 0, "avg_positions": 0,
                "stopped_early": False}

    rets_df = pd.DataFrame(all_returns)
    period_rets = rets_df["portfolio_return_pct"].values / 100
    cumulative = np.cumprod(1 + period_rets)
    total_return = float(cumulative[-1] - 1) * 100

    # Sharpe (annualized, monthly rebalance)
    if len(period_rets) > 1 and np.std(period_rets) > 0:
        sharpe = float(
            np.mean(period_rets) / np.std(period_rets) * np.sqrt(12)
        )
    else:
        sharpe = 0.0

    # Max drawdown
    rolling_max = np.maximum.accumulate(cumulative)
    dd = (cumulative - rolling_max) / rolling_max
    max_dd = float(dd.min() * 100)

    win_rate = float((period_rets > 0).mean() * 100)

    return {
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "win_rate_pct": round(win_rate, 1),
        "n_periods": len(rets_df),
        "avg_positions": round(float(rets_df["n_periods"].mean()), 1)
        if "n_periods" in rets_df.columns
        else round(float(rets_df["n_positions"].mean()), 1),
        "stopped_early": stopped,
    }


# ── Precomputation (shared across all scoring combos) ──────────────────


def precompute_walk_forward_data(
    signals_df,
    transactions_df,
    prices_df,
    as_of_dates,
    horizon,
    lookback_days,
    training_lookback_days,
    min_buyers_list,
):
    """Precompute per-as_of_date data that's shared across all scoring combos.

    Returns {as_of_iso: {training, member_rankings, recent_trades, ...}}.
    """
    precomputed = {}

    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)
        as_of_iso = as_of_ts.isoformat()
        training_lookback_iso = (
            (as_of_ts - pd.Timedelta(days=training_lookback_days)).isoformat()
        )

        # Filter training signals
        training = _filter_training(
            signals_df, horizon, as_of_iso, training_lookback_iso
        )
        if training.empty:
            continue

        # Rank members (expensive, but memoized)
        try:
            member_rankings = rank_members(training, horizon, 5.0)
        except Exception:
            continue

        if member_rankings is None or member_rankings.empty:
            continue

        # Filter recent trades
        recent_trades = _filter_recent_trades(
            transactions_df, lookback_days, as_of_iso
        )
        if recent_trades.empty:
            continue

        # Candidate tickers per min_buyers threshold
        buyer_counts = recent_trades.groupby("ticker")["member"].nunique()
        candidate_tickers_by_mb = {}
        for mb in min_buyers_list:
            candidate_tickers_by_mb[mb] = buyer_counts[
                buyer_counts >= mb
            ].index.tolist()

        # Ticker perf signals (for fallback scoring)
        ticker_perf_signals = _filter_ticker_perf(
            signals_df, horizon, as_of_iso
        )

        precomputed[as_of_iso] = {
            "training": training,
            "member_rankings": member_rankings,
            "recent_trades": recent_trades,
            "candidate_tickers": candidate_tickers_by_mb,
            "ticker_perf_signals": ticker_perf_signals,
            "as_of_ts": as_of_ts,
            "horizon": horizon,
        }

    return precomputed


# ── Main ───────────────────────────────────────────────────────────────


def main():
    t0 = time.time()

    db = Database(Path("data") / "congress.duckdb", read_only=True)

    # Load data
    tx_start = pd.Timestamp("2021-10-07")
    tx_end = pd.Timestamp("2025-06-30")
    all_tx = db.get_transactions_by_date_range(tx_start, tx_end)

    price_start = tx_start
    price_end = pd.Timestamp("2025-06-30") + pd.Timedelta(days=130)
    all_tickers = sorted(
        set(t for t in all_tx["ticker"].dropna().unique() if isinstance(t, str))
        | {"SPY"}
    )

    prices = db.get_prices(all_tickers, price_start, price_end)
    entry_prices = db.get_entry_prices(all_tickers, price_start, price_end)

    print(f"Data loaded: {len(all_tx)} transactions, {prices.shape[1]} tickers")

    # Precompute signals (default horizon=90, decay=0.005)
    horizon = 90
    signals = analysis.calculate_signal_potential(entry_prices, prices, [horizon])
    print(f"Signals computed: {len(signals)}")

    # Walk-forward dates
    start_date = date(2022, 1, 1)
    end_date = date(2025, 6, 30)
    as_of_dates = pd.date_range(start_date, end_date, freq="30D")
    print(f"Walk-forward: {len(as_of_dates)} periods from {start_date} to {end_date}")

    # Precompute shared data (expensive, done once)
    min_buyers_list = [1, 2, 3]
    print("Precomputing walk-forward data...")
    precomputed = precompute_walk_forward_data(
        signals, all_tx, prices,
        as_of_dates, horizon,
        lookback_days=60,
        training_lookback_days=365,
        min_buyers_list=min_buyers_list,
    )
    print(f"  {len(precomputed)}/{len(as_of_dates)} periods have data")

    if not precomputed:
        print("ERROR: No walk-forward periods with data. Check date ranges.")
        return

    # Parameter grid
    param_grid = {
        "scoring_fn": list(SCORING_FUNCTIONS.keys()),
        "top_n": [3, 5, 10],
        "min_buyers": [2, 3],
        "allocation": ["equal", "signal"],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)
    print(f"Grid: {total} combinations ({len(SCORING_FUNCTIONS)} scorings x "
          f"{len(param_grid['top_n'])} top_n x {len(param_grid['min_buyers'])} min_buyers x "
          f"{len(param_grid['allocation'])} allocations)")

    # Run sweep
    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        metrics = run_walk_forward(
            signals, all_tx, prices,
            precomputed,
            scoring_fn=SCORING_FUNCTIONS[params["scoring_fn"]],
            top_n=params["top_n"],
            min_buyers=params["min_buyers"],
            allocation=params["allocation"],
            max_dd_pct=50,  # stop if DD exceeds 50%
        )

        result = {**params, **metrics}
        results.append(result)

        if (i + 1) % 18 == 0 or i == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1:3d}/{total}] "
                f"{params['scoring_fn']:22s} "
                f"top={params['top_n']} "
                f"mb={params['min_buyers']} "
                f"{params['allocation']:6s} "
                f"→ ret={metrics['total_return_pct']:+7.1f}% "
                f"sharpe={metrics['sharpe']:+5.2f} "
                f"DD={metrics['max_drawdown_pct']:6.1f}% "
                f"wr={metrics['win_rate_pct']:4.0f}% "
                f"({rate:.1f}/s ETA {eta:.0f}s)"
            )

    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.1f}s ({total} combos)")

    # Build results table
    results_df = pd.DataFrame(results)

    # ── Baseline comparison ────────────────────────────────────────────
    baseline = results_df[
        (results_df["scoring_fn"] == "shrunk_alpha")
        & (results_df["top_n"] == 5)
        & (results_df["min_buyers"] == 2)
        & (results_df["allocation"] == "equal")
    ]
    if not baseline.empty:
        b = baseline.iloc[0]
        print(f"\n{'=' * 70}")
        print("BASELINE (shrunk_alpha, top=5, mb=2, equal-weight)")
        print(f"{'=' * 70}")
        print(f"  Return:  {b['total_return_pct']:+.1f}%")
        print(f"  Sharpe:  {b['sharpe']:+.2f}")
        print(f"  DD:      {b['max_drawdown_pct']:.1f}%")
        print(f"  Win%:    {b['win_rate_pct']:.0f}%")

    # ── Best by Sharpe (DD constraint) ────────────────────────────────
    valid = results_df[results_df["max_drawdown_pct"] > -30].copy()
    if not valid.empty:
        best_sharpe = valid.nlargest(1, "sharpe").iloc[0]
        print(f"\n{'=' * 70}")
        print("BEST BY SHARPE (max DD < 30%)")
        print(f"{'=' * 70}")
        print(f"  Scoring:  {best_sharpe['scoring_fn']}")
        print(f"  Top N:    {best_sharpe['top_n']}")
        print(f"  Min Buy:  {best_sharpe['min_buyers']}")
        print(f"  Alloc:    {best_sharpe['allocation']}")
        print(f"  Return:   {best_sharpe['total_return_pct']:+.1f}%")
        print(f"  Sharpe:   {best_sharpe['sharpe']:+.2f}")
        print(f"  DD:       {best_sharpe['max_drawdown_pct']:.1f}%")
        print(f"  Win%:     {best_sharpe['win_rate_pct']:.0f}%")

        # Improvement over baseline
        if not baseline.empty:
            b = baseline.iloc[0]
            ret_imp = best_sharpe["total_return_pct"] - b["total_return_pct"]
            sharpe_imp = best_sharpe["sharpe"] - b["sharpe"]
            dd_imp = best_sharpe["max_drawdown_pct"] - b["max_drawdown_pct"]
            print(f"\n  vs baseline: return {ret_imp:+.1f}pp, "
                  f"sharpe {sharpe_imp:+.2f}, DD {dd_imp:+.1f}pp")

    # ── Best by total return (DD constraint) ──────────────────────────
    if not valid.empty:
        best_ret = valid.nlargest(1, "total_return_pct").iloc[0]
        print(f"\n{'=' * 70}")
        print("BEST BY TOTAL RETURN (max DD < 30%)")
        print(f"{'=' * 70}")
        print(f"  Scoring:  {best_ret['scoring_fn']}")
        print(f"  Top N:    {best_ret['top_n']}")
        print(f"  Min Buy:  {best_ret['min_buyers']}")
        print(f"  Alloc:    {best_ret['allocation']}")
        print(f"  Return:   {best_ret['total_return_pct']:+.1f}%")
        print(f"  Sharpe:   {best_ret['sharpe']:+.2f}")
        print(f"  DD:       {best_ret['max_drawdown_pct']:.1f}%")
        print(f"  Win%:     {best_ret['win_rate_pct']:.0f}%")

    # ── Best by return-to-DD ratio ────────────────────────────────────
    if not valid.empty:
        valid = valid.copy()
        valid["ret_dd_ratio"] = valid["total_return_pct"] / np.abs(valid["max_drawdown_pct"].values).clip(min=1)
        best_ratio = valid.nlargest(1, "ret_dd_ratio").iloc[0]
        print(f"\n{'=' * 70}")
        print("BEST RETURN/DRAWDOWN RATIO (max DD < 30%)")
        print(f"{'=' * 70}")
        print(f"  Scoring:  {best_ratio['scoring_fn']}")
        print(f"  Top N:    {best_ratio['top_n']}")
        print(f"  Min Buy:  {best_ratio['min_buyers']}")
        print(f"  Alloc:    {best_ratio['allocation']}")
        print(f"  Return:   {best_ratio['total_return_pct']:+.1f}%")
        print(f"  Sharpe:   {best_ratio['sharpe']:+.2f}")
        print(f"  DD:       {best_ratio['max_drawdown_pct']:.1f}%")
        print(f"  Win%:     {best_ratio['win_rate_pct']:.0f}%")
        print(f"  Ret/DD:   {best_ratio['ret_dd_ratio']:.2f}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("ALL RESULTS (sorted by Sharpe)")
    print(f"{'=' * 90}")
    display_cols = [
        "scoring_fn", "top_n", "min_buyers", "allocation",
        "total_return_pct", "sharpe", "max_drawdown_pct", "win_rate_pct", "n_periods",
    ]
    available = [c for c in display_cols if c in results_df.columns]
    print(
        results_df.nlargest(len(results_df), "sharpe")[available].to_string(
            index=False
        )
    )

    # ── Scoring function summary ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SCORING FUNCTION SUMMARY (best config per scoring)")
    print(f"{'=' * 70}")
    summary_rows = []
    for scoring_fn_name in SCORING_FUNCTIONS:
        subset = results_df[results_df["scoring_fn"] == scoring_fn_name]
        if subset.empty:
            continue
        best = subset.nlargest(1, "sharpe").iloc[0]
        summary_rows.append({
            "scoring": scoring_fn_name,
            "best_return": best["total_return_pct"],
            "best_sharpe": best["sharpe"],
            "best_dd": best["max_drawdown_pct"],
            "best_top_n": best["top_n"],
            "best_mb": best["min_buyers"],
            "best_alloc": best["allocation"],
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("best_sharpe", ascending=False)
    print(summary_df.to_string(index=False))

    # Save results
    out_path = Path("data/optimize_profit_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
