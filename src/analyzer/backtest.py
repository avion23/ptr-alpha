"""Backtest evaluation and recommendation scoring."""

from __future__ import annotations

import pandas as pd

from analyzer._memo import df_memoize
from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.signals import _price_at_or_before, _price_on_or_before, _price_arrays
from analyzer.member_ranking import rank_members, score_ticker_by_buyers


def _find_dip_entry(
    prices_df: pd.DataFrame,
    ticker: str,
    as_of_date: pd.Timestamp,
    pullback_pct: float = 0.05,
    max_wait_days: int = 10,
) -> tuple[float, int]:
    """Find dip entry price after as_of_date (which represents disclosure date in backtest).

    Returns (entry_price, delay_days). If no dip, returns (price_at_as_of, 0).
    """
    if ticker not in prices_df.columns:
        return 0.0, 0

    ticker_prices = prices_df[ticker].dropna()
    post_disc = ticker_prices[ticker_prices.index >= as_of_date]

    if post_disc.empty:
        return 0.0, 0

    disc_price = float(post_disc.iloc[0])
    if disc_price <= 0:
        return 0.0, 0

    window_end = as_of_date + pd.Timedelta(days=max_wait_days)
    window = ticker_prices[(ticker_prices.index >= as_of_date) & (ticker_prices.index <= window_end)]

    target_price = disc_price * (1 - pullback_pct)

    for i, (date_idx, price) in enumerate(window.items()):
        if price <= target_price:
            return float(price), i

    return disc_price, 0


@df_memoize(copy=False)
def _filter_training(
    signals_df: pd.DataFrame,
    horizon: int,
    as_of_iso: str,
    training_lookback_iso: str | None,
) -> pd.DataFrame:
    """Filter signals to the training window. Result is shared (id-stable)."""
    as_of_date = pd.Timestamp(as_of_iso)
    cutoff = as_of_date - pd.Timedelta(days=horizon)
    training = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= cutoff)
    ].copy()
    if training_lookback_iso is not None:
        training_start = pd.Timestamp(training_lookback_iso)
        training = training[training["disclosure_date"] >= training_start]
    return training


@df_memoize(copy=False)
def _filter_recent_trades(
    transactions_df: pd.DataFrame,
    lookback_days: int,
    as_of_iso: str,
) -> pd.DataFrame:
    """Filter transactions to the recent-trade window (Purchase only)."""
    as_of_date = pd.Timestamp(as_of_iso)
    lookback_start = as_of_date - pd.Timedelta(days=lookback_days)
    mask = (
        (transactions_df["disclosure_date"] >= lookback_start)
        & (transactions_df["disclosure_date"] <= as_of_date)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    )
    return transactions_df[mask].copy()


@df_memoize(copy=False)
def _filter_ticker_perf(
    signals_df: pd.DataFrame,
    horizon: int,
    as_of_iso: str,
) -> pd.DataFrame:
    """Filter signals to the ticker-performance window. Result is id-stable."""
    as_of_date = pd.Timestamp(as_of_iso)
    cutoff = as_of_date - pd.Timedelta(days=horizon)
    return signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= cutoff)
    ].copy()


@df_memoize
def _build_ticker_curves(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
) -> list:
    """Build historical return curves for a specific ticker's prior purchases.

    Returns a list of 1-D numpy arrays. Each array is the cumulative return
    curve r(t) = P(t)/P(entry) - 1 over [disclosure, disclosure+horizon].
    """
    if ticker not in prices_df.columns:
        return []

    cutoff = as_of_date - pd.Timedelta(days=horizon)
    eligible = signals_df[
        (signals_df["ticker"] == ticker)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= cutoff)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]
    if eligible.empty:
        return []

    return _build_curves_for_rows(eligible, prices_df, horizon)


@df_memoize
def _build_global_curves(
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
) -> list:
    """Build historical return curves across ALL tickers' prior purchases.

    Used as a global prior when a ticker has no own disclosure history.
    """
    cutoff = as_of_date - pd.Timedelta(days=horizon)
    all_purchases = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= cutoff)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]
    if all_purchases.empty:
        return []

    return _build_curves_for_rows(all_purchases, prices_df, horizon)


def _build_curves_for_rows(
    rows: pd.DataFrame, prices_df: pd.DataFrame, horizon: int
) -> list:
    """Vectorized-ish curve builder.

    Precomputes per-ticker (date_index_ns, price_values) once via the shared
    ``_price_arrays`` cache and uses searchsorted to slice windows instead of
    re-filtering pandas for every row.
    """
    import numpy as np

    price_cols = set(prices_df.columns)
    # Cache (idx_ns, values) per ticker encountered. _price_arrays already
    # normalizes to nanoseconds regardless of source-index resolution.
    per_ticker: dict[str, tuple | None] = {}

    curves: list = []
    disclosures = rows["disclosure_date"].values
    entry_prices = rows["entry_price"].values
    tickers = rows["ticker"].values

    horizon_ns = pd.Timedelta(days=horizon).value  # ns int

    for i in range(len(rows)):
        entry_price = entry_prices[i]
        if not entry_price or entry_price <= 0:
            continue
        tkr = tickers[i]
        if tkr not in price_cols:
            continue

        if tkr not in per_ticker:
            per_ticker[tkr] = _price_arrays(prices_df, tkr)

        cached = per_ticker[tkr]
        if cached is None:
            continue
        idx_ns, vals = cached
        if idx_ns is None:
            continue

        disc_ns = pd.Timestamp(disclosures[i]).value
        end_ns = disc_ns + horizon_ns

        # Left bound: first index >= disc_ns
        lo = int(np.searchsorted(idx_ns, disc_ns, side="left"))
        # Right bound: last index <= end_ns
        hi = int(np.searchsorted(idx_ns, end_ns, side="right"))
        window = vals[lo:hi]
        if len(window) < 3:
            continue
        curves.append(window / entry_price - 1.0)

    return curves


@df_memoize
def _compute_ticker_entry_value(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    rho: float = 0.000137,
) -> float | None:
    """Compute OU entry value V(0) for a ticker from historical return curves.

    Falls back to global prior (average across all tickers) if ticker has
    no prior disclosure history. Result depends only on
    (ticker, signals_df, prices_df, as_of_date, horizon, rho), so it is
    memoized via :func:`df_memoize` and shared across sweep combos.
    """
    from analyzer.return_process import compute_entry_value

    ticker_col = ticker in prices_df.columns if hasattr(prices_df, "columns") else False

    curves: list = []
    if ticker_col:
        curves = _build_ticker_curves(
            ticker, signals_df, prices_df, as_of_date, horizon
        )

    if curves:
        v0, _mu, _theta = compute_entry_value(curves, rho=rho)
        return v0

    if not ticker_col:
        return None

    global_curves = _build_global_curves(
        signals_df, prices_df, as_of_date, horizon
    )
    if not global_curves:
        return None

    v0, _mu, _theta = compute_entry_value(global_curves, rho=rho)
    return v0


@df_memoize
def _compute_ticker_optimal_horizon(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    min_horizon: int = 20,
    max_horizon: int = 120,
) -> int:
    """Compute optimal holding period for a ticker from historical curves."""
    from analyzer.return_process import compute_optimal_horizon

    ticker_col = ticker in prices_df.columns
    if not ticker_col:
        return horizon

    curves = _build_ticker_curves(ticker, signals_df, prices_df, as_of_date, horizon)
    if not curves:
        return horizon

    return compute_optimal_horizon(curves, min_horizon=min_horizon, max_horizon=max_horizon)


def backtest_recommendations(
    signals_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int = 90,
    lookback_days: int = 60,
    min_buyers: int = 2,
    top_n: int = 10,
    threshold: float = 5.0,
    prices_df: pd.DataFrame | None = None,
    training_lookback_days: int | None = None,
    solo_buyer_skill_threshold: float = 0.60,
) -> pd.DataFrame:
    # Read the bayes prior strength once at entry. The sweep mutates the
    # module global per combo; we capture the current value and plumb it
    # explicitly through the memoized leaf functions so their cache keys
    # distinguish combos that differ only in bayes prior.
    from analyzer.signals import BAYES_PRIOR_STRENGTH
    bayes = BAYES_PRIOR_STRENGTH

    as_of_iso = as_of_date.isoformat()
    training_lookback_iso = (
        (as_of_date - pd.Timedelta(days=training_lookback_days)).isoformat()
        if training_lookback_days is not None
        else None
    )

    training = _filter_training(signals_df, horizon, as_of_iso, training_lookback_iso)

    member_rankings: pd.DataFrame | None = None
    if not training.empty:
        try:
            member_rankings = rank_members(
                training, horizon, threshold,
                _bayes_prior_strength=bayes,
            )
        except AnalysisError:
            member_rankings = None

    if member_rankings is None or member_rankings.empty:
        return pd.DataFrame()

    recent_trades = _filter_recent_trades(transactions_df, lookback_days, as_of_iso)
    if recent_trades.empty:
        return pd.DataFrame()

    buyer_counts = recent_trades.groupby("ticker")["member"].nunique()
    candidate_tickers = buyer_counts[buyer_counts >= min_buyers].index.tolist()

    if not candidate_tickers:
        return pd.DataFrame()

    ticker_perf_signals = _filter_ticker_perf(signals_df, horizon, as_of_iso)

    from analyzer.signal_features import (
        compute_signal_features,
        compute_disclosure_lag_weight,
        estimate_crash_hazard,
    )

    as_of_for_features = (
        as_of_date.date() if hasattr(as_of_date, "date") else as_of_date
    )

    scores = []
    for ticker in candidate_tickers:
        try:
            # score_ticker_by_buyers, _compute_ticker_entry_value and
            # compute_signal_features are all `@df_memoize`'d. Their cache
            # keys are content-stable because their DataFrame inputs
            # (training, recent_trades, ticker_perf_signals, signals_df,
            # prices_df) all have stable identities across sweep calls.
            score = score_ticker_by_buyers(
                ticker, recent_trades, training, horizon, threshold,
                member_rankings, min_buyers,
                ticker_perf_signals=ticker_perf_signals,
                _bayes_prior_strength=bayes,
                solo_buyer_skill_threshold=solo_buyer_skill_threshold,
            )

            # Compute OU entry value V(0) if prices available
            if prices_df is not None:
                v0 = _compute_ticker_entry_value(
                    ticker, signals_df, prices_df, as_of_date, horizon,
                )
                score["ou_entry_value"] = round(v0, 4) if v0 is not None else None

                # Compute optimal holding period from OU half-life
                optimal_h = _compute_ticker_optimal_horizon(
                    ticker, signals_df, prices_df, as_of_date, horizon,
                )
                score["optimal_horizon"] = optimal_h

            # Compute signal features and crash hazard for this ticker
            # Use the most recent transaction date for this ticker
            ticker_recent = recent_trades[recent_trades["ticker"] == ticker]
            if not ticker_recent.empty and prices_df is not None:
                try:
                    latest_tx = ticker_recent.sort_values("disclosure_date").iloc[-1]
                    tx_date = latest_tx.get("transaction_date")
                    if tx_date is not None:
                        tx_date = pd.Timestamp(tx_date).date()
                    disc_date = pd.Timestamp(latest_tx["disclosure_date"]).date()

                    features = compute_signal_features(
                        ticker=ticker,
                        disclosure_date=disc_date,
                        transaction_date=tx_date,
                        prices_df=prices_df,
                        all_tx=recent_trades,
                        as_of_date=as_of_for_features,
                    )
                    crash = estimate_crash_hazard(features)

                    # Apply lag weight
                    lag_weight = compute_disclosure_lag_weight(features.lag_days)
                    base_score = float(score["signal_score"].iloc[0])
                    adjusted_score = base_score * lag_weight

                    # Apply crash penalty (only if crash_prob is reasonable)
                    if 0.0 <= crash.crash_prob <= 1.0:
                        adjusted_score *= (1 - crash.crash_prob)

                    score["signal_score"] = round(adjusted_score, 2)
                    score["lag_days"] = features.lag_days
                    score["lag_weight"] = round(lag_weight, 4)
                    score["crash_prob"] = crash.crash_prob
                    score["crash_var_95"] = crash.var_95
                    score["volatility_20d"] = round(features.volatility_20d, 4)
                    score["drawdown_from_ath"] = round(features.drawdown_from_ath, 4)
                except Exception:
                    # Crash hazard estimation can fail on edge cases — keep base score
                    pass

            scores.append(score)
        except AnalysisError:
            continue

    if not scores:
        return pd.DataFrame()

    result = pd.concat(scores, ignore_index=True)
    # Drop rejected rows. score_ticker_by_buyers emits a zero signal_score
    # (with a `note`) for tickers that fail the min-buyers / solo-buyer skill
    # gate; those should not surface as recommendations.
    if "signal_score" in result.columns:
        result = result[result["signal_score"].fillna(0) > 0]
    if result.empty:
        return pd.DataFrame()
    result = result.sort_values("signal_score", ascending=False).head(top_n).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))

    # Propagate instrument_type and amount_midpoint from recent trades so
    # evaluate_backtest can apply options leverage.
    if "instrument_type" in recent_trades.columns:
        inst_map = (
            recent_trades.drop_duplicates("ticker")
            .set_index("ticker")["instrument_type"]
            .to_dict()
        )
        result["instrument_type"] = result["ticker"].map(inst_map).fillna("stock")
    if "amount_midpoint" in recent_trades.columns:
        amt_map = (
            recent_trades.drop_duplicates("ticker")
            .set_index("ticker")["amount_midpoint"]
            .to_dict()
        )
        result["amount_midpoint"] = result["ticker"].map(amt_map)

    return result


def evaluate_backtest(
    recommendations: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    max_staleness_days: int | None = 30,
    entry_slippage_bps: float = 10.0,
    exit_slippage_bps: float = 10.0,
    use_dip_entry: bool = True,
    pullback_pct: float = 0.05,
    max_wait_days: int = 10,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    from analyzer.options import estimate_options_leverage

    rows = []
    for _, rec in recommendations.iterrows():
        ticker = rec["ticker"]
        # Use per-ticker optimal horizon if available
        ticker_horizon = int(rec.get("optimal_horizon", horizon)) if "optimal_horizon" in rec.index else horizon

        # Entry with dip timing
        entry_delay = 0
        if use_dip_entry:
            entry, entry_delay = _find_dip_entry(
                prices_df, ticker, as_of_date, pullback_pct, max_wait_days,
            )
            if entry <= 0:
                entry = _price_at_or_before(prices_df, ticker, as_of_date, max_staleness_days)
                entry_delay = 0
        else:
            entry = _price_at_or_before(prices_df, ticker, as_of_date, max_staleness_days)
            entry_delay = 0

        if not entry:
            continue

        # Adjust exit date based on entry delay (hold for horizon from entry, not disclosure)
        actual_entry_date = as_of_date + pd.Timedelta(days=entry_delay)
        exit_date = actual_entry_date + pd.Timedelta(days=ticker_horizon)

        exit_price = _price_on_or_before(prices_df, ticker, exit_date, max_staleness_days=30)
        if not exit_price:
            continue

        # Apply slippage
        entry *= (1 + entry_slippage_bps / 10000)
        exit_price *= (1 - exit_slippage_bps / 10000)
        return_pct = (exit_price / entry - 1) * 100

        # SPY benchmark: use original disclosure date for consistency
        spy_exit_date = as_of_date + pd.Timedelta(days=ticker_horizon)
        spy_start = _price_at_or_before(prices_df, "SPY", as_of_date, max_staleness_days)
        spy_end = _price_on_or_before(prices_df, "SPY", spy_exit_date, max_staleness_days=30)
        if not spy_start or not spy_end:
            continue
        spy_start_adj = spy_start * (1 + entry_slippage_bps / 10000)
        spy_end_adj = spy_end * (1 - exit_slippage_bps / 10000)
        spy_return_pct = (spy_end_adj / spy_start_adj - 1) * 100

        # Apply options leverage
        inst_type = "stock"
        if "instrument_type" in rec.index:
            val = rec["instrument_type"]
            if pd.notna(val):
                inst_type = str(val)
        amount = rec.get("amount_midpoint") if "amount_midpoint" in rec.index else None
        leverage = estimate_options_leverage(inst_type, amount)
        leveraged_return_pct = return_pct * leverage
        alpha_pct = leveraged_return_pct - spy_return_pct

        rows.append({
            "ticker": ticker,
            "bt_entry_price": round(entry, 2),
            "bt_exit_price": round(exit_price, 2),
            "bt_raw_return_pct": round(return_pct, 2),
            "bt_return_pct": round(leveraged_return_pct, 2),
            "bt_leverage": round(leverage, 2),
            "bt_spy_return_pct": round(spy_return_pct, 2),
            "bt_alpha_pct": round(alpha_pct, 2),
            "bt_horizon_days": ticker_horizon,
            "bt_entry_delay": entry_delay,
        })

    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        recommendations = recommendations.copy()
        for col in ["bt_entry_price", "bt_exit_price", "bt_raw_return_pct", "bt_return_pct", "bt_leverage", "bt_spy_return_pct", "bt_alpha_pct", "bt_entry_delay"]:
            recommendations[col] = None
        return recommendations
    return recommendations.merge(eval_df, on="ticker", how="left")


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    valid = results.dropna(subset=["bt_return_pct"])
    if valid.empty:
        return pd.DataFrame()

    by_rank = []
    for rank, grp in valid.groupby("rank"):
        entry = {
            "rank": rank,
            "count": len(grp),
            "win_rate_pct": round((grp["bt_return_pct"] > 0).mean() * 100, 1),
            "avg_return_pct": round(grp["bt_return_pct"].mean(), 2),
            "avg_alpha_pct": round(grp["bt_alpha_pct"].mean(), 2) if "bt_alpha_pct" in grp.columns else None,
        }
        if "bt_raw_return_pct" in grp.columns:
            entry["avg_raw_return_pct"] = round(grp["bt_raw_return_pct"].mean(), 2)
        if "bt_leverage" in grp.columns:
            entry["avg_leverage"] = round(grp["bt_leverage"].mean(), 2)
        by_rank.append(entry)

    summary = pd.DataFrame(by_rank)

    overall = {
        "rank": "ALL",
        "count": len(valid),
        "win_rate_pct": round((valid["bt_return_pct"] > 0).mean() * 100, 1),
        "avg_return_pct": round(valid["bt_return_pct"].mean(), 2),
        "avg_alpha_pct": round(valid["bt_alpha_pct"].mean(), 2) if "bt_alpha_pct" in valid.columns else None,
    }
    if "bt_raw_return_pct" in valid.columns:
        overall["avg_raw_return_pct"] = round(valid["bt_raw_return_pct"].mean(), 2)
    if "bt_leverage" in valid.columns:
        overall["avg_leverage"] = round(valid["bt_leverage"].mean(), 2)
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return summary
