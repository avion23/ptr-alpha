"""Backtest evaluation and recommendation scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer._memo import df_memoize
from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.signals import _price_arrays
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
    arrs = _price_arrays(prices_df, ticker)
    if arrs is None:
        return 0.0, 0
    idx_ns, vals = arrs
    if idx_ns is None:
        return 0.0, 0
    return _find_dip_entry_arrays(idx_ns, vals, as_of_date, pullback_pct, max_wait_days)


def _find_dip_entry_arrays(
    idx_ns, vals, as_of_date, pullback_pct=0.05, max_wait_days=10,
):
    """Find dip entry using pre-extracted price arrays (avoids repeated _price_arrays lookup)."""
    target_ns = pd.Timestamp(as_of_date).value
    window_end_ns = target_ns + max_wait_days * 86_400_000_000_000

    # First price >= as_of_date
    lo = int(np.searchsorted(idx_ns, target_ns, side="left"))
    if lo >= len(idx_ns):
        return 0.0, 0
    disc_price = float(vals[lo])
    if disc_price <= 0:
        return 0.0, 0

    # Window of prices within [as_of_date, as_of_date + max_wait_days]
    hi = int(np.searchsorted(idx_ns, window_end_ns, side="right"))
    window_vals = vals[lo:hi]
    if len(window_vals) == 0:
        return 0.0, 0

    target_price = disc_price * (1 - pullback_pct)
    hits = np.where(window_vals <= target_price)[0]
    if len(hits) > 0:
        return float(window_vals[hits[0]]), int(hits[0])

    return disc_price, 0


def _price_at_or_before_arrays(idx_ns, vals, target_date, max_staleness_days=None):
    """Price lookup using pre-extracted arrays."""
    target = pd.Timestamp(target_date).value
    pos = int(np.searchsorted(idx_ns, target, side="right")) - 1
    if pos < 0:
        return None
    if max_staleness_days is not None:
        staleness_ns = target - int(idx_ns[pos])
        if staleness_ns > max_staleness_days * 86_400_000_000_000:
            return None
    return float(vals[pos])


def _price_on_or_before_arrays(idx_ns, vals, target_date, max_staleness_days=5):
    """Price lookup using pre-extracted arrays."""
    return _price_at_or_before_arrays(idx_ns, vals, target_date, max_staleness_days=max_staleness_days)


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


@df_memoize(copy=False)
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


@df_memoize(copy=False)
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
    """Vectorized curve builder.

    Precomputes per-ticker (date_index_ns, price_values) once via the shared
    ``_price_arrays`` cache and uses searchsorted to slice windows instead of
    re-filtering pandas for every row.
    """
    import numpy as np

    price_cols = set(prices_df.columns)
    per_ticker: dict[str, tuple | None] = {}

    curves: list = []
    disclosures = rows["disclosure_date"].values
    entry_prices_arr = rows["entry_price"].values
    tickers = rows["ticker"].values

    horizon_ns = pd.Timedelta(days=horizon).value  # ns int

    # Pre-compute all disclosure timestamps as int64 ns to avoid
    # per-row pd.Timestamp() creation in the loop.
    disc_ns_all = np.empty(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        disc_ns_all[i] = pd.Timestamp(disclosures[i]).value

    for i in range(len(rows)):
        entry_price = entry_prices_arr[i]
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

        disc_ns = disc_ns_all[i]
        end_ns = disc_ns + horizon_ns

        lo = int(np.searchsorted(idx_ns, disc_ns, side="left"))
        hi = int(np.searchsorted(idx_ns, end_ns, side="right"))
        window = vals[lo:hi]
        if len(window) < 3:
            continue
        curves.append(window / entry_price - 1.0)

    return curves


@df_memoize(copy=False)
def _compute_ticker_ou_params(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    rho: float = 0.000137,
) -> tuple[float | None, int]:
    """Compute both OU entry value V(0) and optimal horizon for a ticker.

    Builds curves once and fits OU once, returning both V0 and optimal
    holding period.  Falls back to global prior (average across all
    tickers) when the ticker has no own disclosure history.

    Returns (v0, optimal_horizon).
    """
    from analyzer.return_process import compute_entry_value_and_horizon

    ticker_col = ticker in prices_df.columns if hasattr(prices_df, "columns") else False

    curves: list = []
    if ticker_col:
        curves = _build_ticker_curves(
            ticker, signals_df, prices_df, as_of_date, horizon
        )

    if not curves and ticker_col:
        curves = _build_global_curves(
            signals_df, prices_df, as_of_date, horizon
        )

    if not curves:
        return (None, horizon) if ticker_col else (None, horizon)

    v0, _mu, _theta, optimal_h = compute_entry_value_and_horizon(curves, rho=rho)
    return v0, optimal_h


@df_memoize(copy=False)
def _compute_ticker_entry_value(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    rho: float = 0.000137,
) -> float | None:
    """Compute OU entry value V(0) for a ticker from historical return curves."""
    v0, _ = _compute_ticker_ou_params(ticker, signals_df, prices_df, as_of_date, horizon, rho)
    return v0


@df_memoize(copy=False)
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
    _v0, optimal_h = _compute_ticker_ou_params(ticker, signals_df, prices_df, as_of_date, horizon)
    return max(min_horizon, min(max_horizon, optimal_h))


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

    # Pre-group recent_trades by ticker to avoid repeated boolean masking
    recent_by_ticker = {t: grp for t, grp in recent_trades.groupby("ticker")}

    ticker_perf_signals = _filter_ticker_perf(signals_df, horizon, as_of_iso)

    from analyzer.signal_features import (
        compute_signal_features,
        compute_disclosure_lag_weight,
        estimate_crash_hazard,
    )

    as_of_for_features = (
        as_of_date.date() if hasattr(as_of_date, "date") else as_of_date
    )

    # Pre-build O(1) lookup dicts from member_rankings
    from analyzer.member_ranking import _build_ranking_dicts
    _ranking_dicts = _build_ranking_dicts(member_rankings)

    # Pre-check which tickers have prices (avoids per-ticker hasattr)
    has_prices = prices_df is not None
    price_cols = set(prices_df.columns) if has_prices else set()

    # Pre-compute instrument_type and amount_midpoint maps outside ticker loop
    inst_map: dict = {}
    amt_map: dict = {}
    if has_prices and "instrument_type" in recent_trades.columns:
        inst_map = (
            recent_trades.drop_duplicates("ticker")
            .set_index("ticker")["instrument_type"]
            .to_dict()
        )
    if has_prices and "amount_midpoint" in recent_trades.columns:
        amt_map = (
            recent_trades.drop_duplicates("ticker")
            .set_index("ticker")["amount_midpoint"]
            .to_dict()
        )

    scores = []
    for ticker in candidate_tickers:
        score_df = score_ticker_by_buyers(
            ticker, recent_trades, training, horizon, threshold,
            member_rankings, min_buyers,
            ticker_perf_signals=ticker_perf_signals,
            _bayes_prior_strength=bayes,
            solo_buyer_skill_threshold=solo_buyer_skill_threshold,
            _ranking_dicts=_ranking_dicts,
        )

        if score_df.empty:
            continue

        # Extract score scalars into a dict to avoid DataFrame mutations
        # (avoids N x __setitem__ overhead on 1-row DataFrames).
        row = {c: score_df[c].iloc[0] for c in score_df.columns}

        # Compute OU entry value V(0) + optimal horizon in one pass
        if has_prices and ticker in price_cols:
            v0, optimal_h = _compute_ticker_ou_params(
                ticker, signals_df, prices_df, as_of_date, horizon,
            )
            row["ou_entry_value"] = round(v0, 4) if v0 is not None else None
            row["optimal_horizon"] = optimal_h
        else:
            row["ou_entry_value"] = None
            row["optimal_horizon"] = horizon

        # Compute signal features and crash hazard for this ticker
        ticker_recent = recent_by_ticker.get(ticker)
        if ticker_recent is not None and has_prices:
            latest_idx = ticker_recent["disclosure_date"].idxmax()
            latest_tx = ticker_recent.loc[latest_idx]
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
            base_score = float(row.get("signal_score", 0))
            adjusted_score = base_score * lag_weight

            # Apply crash penalty (only if crash_prob is reasonable)
            if 0.0 <= crash.crash_prob <= 1.0:
                adjusted_score *= (1 - crash.crash_prob)

            row["signal_score"] = round(adjusted_score, 2)
            row["lag_days"] = features.lag_days
            row["lag_weight"] = round(lag_weight, 4)
            row["crash_prob"] = crash.crash_prob
            row["crash_var_95"] = crash.var_95
            row["volatility_20d"] = round(features.volatility_20d, 4)
            row["drawdown_from_ath"] = round(features.drawdown_from_ath, 4)

        scores.append(row)

    if not scores:
        return pd.DataFrame()

    result = pd.DataFrame(scores)
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
    # evaluate_backtest can apply options leverage.  Maps were precomputed
    # above, outside the ticker loop.
    if inst_map:
        result["instrument_type"] = result["ticker"].map(inst_map).fillna("stock")
    if amt_map:
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

    # Precompute slippage multipliers and ns-per-day constant
    entry_mult = 1.0 + entry_slippage_bps / 10000
    exit_mult = 1.0 - exit_slippage_bps / 10000
    NS_PER_DAY = 86_400_000_000_000

    # ── Precompute SPY benchmark once per as_of_date (hoisted out of loop) ──
    horizons = recommendations["optimal_horizon"].values if "optimal_horizon" in recommendations.columns else None

    # Pre-extract SPY price arrays once (avoids repeated _price_arrays lookups)
    spy_arrs = _price_arrays(prices_df, "SPY")
    spy_ns, spy_vals = (spy_arrs if spy_arrs and spy_arrs[0] is not None else (None, None))

    spy_start = None
    spy_entry_adj = 0.0
    if spy_ns is not None:
        spy_start = _price_at_or_before_arrays(spy_ns, spy_vals, as_of_date, max_staleness_days)
        if spy_start:
            spy_entry_adj = spy_start * entry_mult

    # Precompute spy_end for each unique horizon using arrays
    spy_ends: dict[int, float | None] = {}
    spy_returns: dict[int, float] = {}
    if spy_start:
        for h in set(horizons) if horizons is not None else [horizon]:
            spy_exit_ns = as_of_date.value + int(h) * NS_PER_DAY
            se = _price_on_or_before_arrays(spy_ns, spy_vals, pd.Timestamp(spy_exit_ns), max_staleness_days=30)
            spy_ends[h] = se
            if se:
                spy_exit_adj = se * exit_mult
                spy_returns[h] = round((spy_exit_adj / spy_entry_adj - 1) * 100, 2)
            else:
                spy_returns[h] = 0.0

    # ── Batch price lookups for all tickers at once ──
    tickers = recommendations["ticker"].tolist()
    ticker_horizons = (
        [int(h) for h in horizons]
        if horizons is not None
        else [horizon] * len(tickers)
    )

    # Pre-extract price arrays for all tickers (one searchsorted call each)
    price_cache: dict[str, tuple | None] = {}
    for t in tickers:
        price_cache[t] = _price_arrays(prices_df, t)

    # Pre-extract column arrays to avoid per-row Series creation
    ticker_arr = recommendations["ticker"].values
    has_inst_type = "instrument_type" in recommendations.columns
    has_amount = "amount_midpoint" in recommendations.columns
    inst_type_arr = recommendations["instrument_type"].values if has_inst_type else None
    amount_arr = recommendations["amount_midpoint"].values if has_amount else None

    rows = []
    for i in range(len(ticker_arr)):
        ticker = ticker_arr[i]
        t_horizon = ticker_horizons[i]

        cached = price_cache.get(ticker)
        if cached is None:
            continue
        idx_ns, vals = cached
        if idx_ns is None:
            continue

        # Entry with dip timing
        entry_delay = 0
        if use_dip_entry:
            entry, entry_delay = _find_dip_entry_arrays(
                idx_ns, vals, as_of_date, pullback_pct, max_wait_days,
            )
            if entry <= 0:
                entry = _price_at_or_before_arrays(idx_ns, vals, as_of_date, max_staleness_days)
                entry_delay = 0
        else:
            entry = _price_at_or_before_arrays(idx_ns, vals, as_of_date, max_staleness_days)
            entry_delay = 0

        if not entry:
            continue

        # Compute exit date as ns timestamp directly (avoids Timedelta creation)
        as_of_ns = as_of_date.value
        exit_ns = as_of_ns + (entry_delay + t_horizon) * NS_PER_DAY
        exit_price = _price_on_or_before_arrays(idx_ns, vals, pd.Timestamp(exit_ns), max_staleness_days=30)
        if not exit_price:
            continue

        entry_adj = entry * entry_mult
        exit_adj = exit_price * exit_mult
        return_pct = (exit_adj / entry_adj - 1) * 100

        # SPY benchmark (precomputed)
        spy_ret = spy_returns.get(t_horizon, 0.0)
        if not spy_start or (spy_ret == 0.0 and not spy_ends.get(t_horizon)):
            # Spy lookup failed for this horizon — skip rec
            continue

        # Options leverage
        inst_type = "stock"
        if inst_type_arr is not None:
            val = inst_type_arr[i]
            if pd.notna(val):
                inst_type = str(val)
        amount = amount_arr[i] if has_amount else None
        leverage = estimate_options_leverage(inst_type, amount)
        leveraged_return_pct = return_pct * leverage
        alpha_pct = leveraged_return_pct - spy_ret

        rows.append({
            "ticker": ticker,
            "bt_entry_price": round(entry, 2),
            "bt_exit_price": round(exit_price, 2),
            "bt_raw_return_pct": round(return_pct, 2),
            "bt_return_pct": round(leveraged_return_pct, 2),
            "bt_leverage": round(leverage, 2),
            "bt_spy_return_pct": spy_ret,
            "bt_alpha_pct": round(alpha_pct, 2),
            "bt_horizon_days": t_horizon,
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
