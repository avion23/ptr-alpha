"""Honest time-split validation for PTR Alpha strategies.

Implements a proper train/test split:
  1. Parameter sweep restricted to the TRAINING window only.
  2. Benjamini-Hochberg / Bonferroni snooping corrections on all tested configs.
  3. Newey-West HAC t-stats to account for overlapping return windows.
  4. Frozen config evaluated EXACTLY ONCE on the TEST window.

Public API
----------
SweepResult          – backtest summary dataclass (moved from repo-root sweep.py)
run_single_backtest  – run one config, return SweepResult (moved from sweep.py)
sweep_configs        – iterate a grid on a date window, return DataFrame
newey_west_tstat     – Bartlett-kernel HAC t-statistic
select_config        – BH-corrected config selection
run_validation       – full train→select→test pipeline
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyzer import analysis
from analyzer import signals as signals_mod
from analyzer.pipeline import BacktestParams
from analyzer.snooping import benjamini_hochberg, bonferroni_correction


# ---------------------------------------------------------------------------
# SweepResult (moved verbatim from repo-root sweep.py)
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    horizon: int
    frequency_days: int
    training_lookback_days: int
    min_buyers: int
    top_n: int
    decay_lambda: float
    bayes_prior_strength: float
    scoring_mode: str = "shrunk_alpha"
    total_recs: int = 0
    dates_evaluated: int = 0
    overall_alpha: float = 0.0
    overall_return: float = 0.0
    rank1_alpha: float = 0.0
    rank5_alpha: float = 0.0
    alpha_slope: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _backtest_core(
    all_transactions: pd.DataFrame,
    prices: pd.DataFrame,
    params: BacktestParams,
    signals: pd.DataFrame,
    bayes_prior_strength: float,
    decay_lambda: float,
    scoring_mode: str = "shrunk_alpha",
) -> tuple[SweepResult, pd.Series]:
    """Core backtest loop returning (SweepResult, per_date_mean_alpha_series).

    The per-date series is needed for Newey-West t-stat computation.
    run_single_backtest is a thin public wrapper that discards the series.
    """
    old_bayes = signals_mod.BAYES_PRIOR_STRENGTH
    old_decay = signals_mod.DECAY_LAMBDA
    signals_mod.BAYES_PRIOR_STRENGTH = bayes_prior_strength
    signals_mod.DECAY_LAMBDA = decay_lambda

    empty = SweepResult(
        horizon=params.horizon,
        frequency_days=params.frequency_days,
        training_lookback_days=params.training_lookback_days,
        min_buyers=params.min_buyers,
        top_n=params.top_n,
        decay_lambda=decay_lambda,
        bayes_prior_strength=bayes_prior_strength,
        scoring_mode=scoring_mode,
    )

    try:
        as_of_dates = pd.date_range(
            params.start_date, params.end_date, freq=f"{params.frequency_days}D"
        )

        all_results = []
        for as_of in as_of_dates:
            as_of_ts = pd.Timestamp(as_of)
            recs = analysis.backtest_recommendations(
                signals, all_transactions, as_of_ts,
                horizon=params.horizon,
                lookback_days=params.lookback_days,
                min_buyers=params.min_buyers,
                top_n=params.top_n,
                threshold=params.threshold,
                prices_df=prices,
                training_lookback_days=params.training_lookback_days,
                scoring_mode=scoring_mode,
            )
            if recs.empty:
                continue
            try:
                evaluated = analysis.evaluate_backtest(
                    recs, prices, as_of_ts, params.horizon
                )
                evaluated = evaluated.dropna(subset=["bt_return_pct"])
                evaluated.insert(0, "as_of_date", as_of_ts.date())
                all_results.append(evaluated)
            except Exception:
                continue

        if not all_results:
            return empty, pd.Series(dtype=float)

        combined = pd.concat(all_results, ignore_index=True)
        valid = combined.dropna(subset=["bt_alpha_pct"])

        # Per-date mean alpha (NW t-stat needs this series ordered by date)
        per_date = (
            valid.groupby("as_of_date")["bt_alpha_pct"]
            .mean()
            .sort_index()
        )

        result = SweepResult(
            horizon=params.horizon,
            frequency_days=params.frequency_days,
            training_lookback_days=params.training_lookback_days,
            min_buyers=params.min_buyers,
            top_n=params.top_n,
            decay_lambda=decay_lambda,
            bayes_prior_strength=bayes_prior_strength,
            scoring_mode=scoring_mode,
            total_recs=len(combined),
            dates_evaluated=int(valid["as_of_date"].nunique()),
            overall_alpha=round(float(valid["bt_alpha_pct"].mean()), 2),
            overall_return=round(float(valid["bt_return_pct"].mean()), 2),
        )

        rank_alpha = valid.groupby("rank")["bt_alpha_pct"].mean()
        if 1 in rank_alpha.index:
            result.rank1_alpha = round(float(rank_alpha.loc[1]), 2)
        if 5 in rank_alpha.index:
            result.rank5_alpha = round(float(rank_alpha.loc[5]), 2)
        # Convention: rank 1 = highest-scored ticker (best model prediction).
        # A well-calibrated ranker has rank-1 picks outperforming rank-5 picks,
        # so alpha_slope > 0 means the ranker is working.
        result.alpha_slope = round(result.rank1_alpha - result.rank5_alpha, 2)
        result.win_rate = round(float((valid["bt_alpha_pct"] > 0).mean()) * 100, 1)

        if len(valid) > 1:
            mean_a = valid["bt_alpha_pct"].mean()
            std_a = valid["bt_alpha_pct"].std()
            if std_a > 0:
                periods_per_year = 365 / params.frequency_days
                result.sharpe = round(
                    float(mean_a / std_a * (periods_per_year ** 0.5)), 2
                )

        cumulative = (1 + valid["bt_alpha_pct"] / 100).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        result.max_drawdown = round(float(drawdown.min()) * 100, 2)

        return result, per_date

    finally:
        signals_mod.BAYES_PRIOR_STRENGTH = old_bayes
        signals_mod.DECAY_LAMBDA = old_decay


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def run_single_backtest(
    all_transactions: pd.DataFrame,
    prices: pd.DataFrame,
    params: BacktestParams,
    signals: pd.DataFrame,
    bayes_prior_strength: float,
    decay_lambda: float,
    scoring_mode: str = "shrunk_alpha",
) -> SweepResult:
    """Run one backtest with given params and return metrics.

    Moved from repo-root sweep.py.  alpha_slope = rank1_alpha - rank5_alpha:
    positive means rank-1 picks outperform rank-5 (the ranker is working).
    """
    result, _ = _backtest_core(
        all_transactions, prices, params, signals,
        bayes_prior_strength, decay_lambda, scoring_mode,
    )
    return result


def newey_west_tstat(alpha_series: pd.Series, lag: int) -> float:
    """t-statistic of the mean with Bartlett-kernel HAC standard error.

    Uses 1/n normalization for autocovariances (standard Newey-West).
    For lag=0 this is equivalent to mean / (biased_std / sqrt(n)).

    Args:
        alpha_series: Per-period alpha values (NaN are dropped).
        lag: Maximum autocorrelation lag for HAC correction.
             Recommended: max(0, ceil(horizon / frequency_days) - 1).

    Returns:
        HAC-robust t-statistic.
    """
    x = np.asarray(alpha_series.dropna(), dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    # Autocovariances beyond lag n-1 are undefined; cap the Bartlett window accordingly.
    lag = max(0, min(lag, n - 1))

    mu = float(x.mean())
    demeaned = x - mu

    # Autocovariances: γ_k = (1/n) Σ_{t=k}^{n-1} (x_t − μ)(x_{t-k} − μ)
    gamma = np.array([
        np.dot(demeaned[k:], demeaned[: n - k]) / n
        for k in range(lag + 1)
    ])

    if lag == 0:
        long_run_var = float(gamma[0])
    else:
        # Bartlett kernel: w_k = 1 − k/(lag+1)
        weights = 1.0 - np.arange(1, lag + 1) / (lag + 1)
        long_run_var = float(gamma[0] + 2.0 * np.dot(weights, gamma[1:]))

    long_run_var = max(long_run_var, 0.0)
    se = math.sqrt(long_run_var / n)

    if se < 1e-14:
        if mu > 0:
            return math.inf
        return -math.inf if mu < 0 else 0.0

    return float(mu / se)


def sweep_configs(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    entry_prices: pd.DataFrame,
    grid: dict,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Sweep a parameter grid restricted to the given date window.

    Precomputes signals once per (horizon, decay_lambda) pair, then runs
    each config's backtest over as_of dates in [start, end]. Appends
    nw_tstat and p_value columns to the returned DataFrame.

    Args:
        all_tx: Transactions DataFrame.
        prices: Wide-format prices (dates × tickers).
        entry_prices: Entry-price DataFrame from db.get_entry_prices.
        grid: Dict mapping param-name → list of values.  Must include the
              keys used by BacktestParams plus decay_lambda,
              bayes_prior_strength, scoring_mode.
        start: Backtest window start date.
        end: Backtest window end date.

    Returns:
        DataFrame with one row per config; all SweepResult fields plus
        nw_tstat and p_value columns.
    """
    unique_horizons: set[int] = set(grid.get("horizon", [60]))
    unique_decays: set[float] = set(grid.get("decay_lambda", [0.005]))

    signal_cache: dict[tuple[int, float], pd.DataFrame] = {}
    for h in unique_horizons:
        for d in unique_decays:
            signal_cache[(h, d)] = analysis.calculate_signal_potential(
                entry_prices, prices, [h], decay_lambda=d,
            )

    keys = list(grid.keys())
    rows: list[dict] = []

    for combo in itertools.product(*grid.values()):
        params_dict = dict(zip(keys, combo))

        horizon = int(params_dict["horizon"])
        freq = int(params_dict.get("frequency_days", 30))
        lag = max(0, math.ceil(horizon / freq) - 1)

        params = BacktestParams(
            start_date=start,
            end_date=end,
            horizon=horizon,
            lookback_days=60,
            training_lookback_days=int(params_dict.get("training_lookback_days", 365)),
            min_buyers=int(params_dict["min_buyers"]),
            top_n=int(params_dict["top_n"]),
            threshold=float(params_dict.get("threshold", 5.0)),
            frequency_days=freq,
        )
        sigs = signal_cache[(horizon, float(params_dict["decay_lambda"]))]
        result, per_date = _backtest_core(
            all_tx, prices, params, sigs,
            bayes_prior_strength=float(params_dict["bayes_prior_strength"]),
            decay_lambda=float(params_dict["decay_lambda"]),
            scoring_mode=str(params_dict.get("scoring_mode", "shrunk_alpha")),
        )

        t_stat = newey_west_tstat(per_date, lag=lag)
        p_val = (
            float(2.0 * stats.norm.sf(abs(t_stat)))
            if math.isfinite(t_stat)
            # Only positive-alpha configs are selection candidates; -inf must not survive BH.
            else 0.0 if t_stat > 0 else 1.0
        )

        row = asdict(result)
        row["nw_tstat"] = round(t_stat, 4) if math.isfinite(t_stat) else t_stat
        row["p_value"] = p_val
        rows.append(row)

    return pd.DataFrame(rows)


def select_config(sweep_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Select best config with Benjamini-Hochberg snooping correction.

    Among BH survivors, picks max alpha_slope (tie-break: overall_alpha).
    If no configs survive BH, returns the nominal best flagged
    survives_correction=False.

    Args:
        sweep_df: Output of sweep_configs.  Must have columns p_value,
                  alpha_slope, overall_alpha.
        alpha: FDR level (default 0.05).

    Returns:
        Dict of the selected row's fields, plus:
          survives_correction (bool)
          n_trials (int)
          n_survivors (int)
          bonferroni_threshold (float)
    """
    n_trials = len(sweep_df)
    bonf_thresh = bonferroni_correction(n_trials, alpha)
    bh_mask = benjamini_hochberg(sweep_df["p_value"].values, alpha)

    survivors = sweep_df[bh_mask]
    if survivors.empty:
        best_row = sweep_df.sort_values(
            ["alpha_slope", "overall_alpha"], ascending=False
        ).iloc[0]
        survives = False
        n_survivors = 0
    else:
        best_row = survivors.sort_values(
            ["alpha_slope", "overall_alpha"], ascending=False
        ).iloc[0]
        survives = True
        n_survivors = int(bh_mask.sum())

    result = best_row.to_dict()
    result["survives_correction"] = survives
    result["n_trials"] = n_trials
    result["n_survivors"] = n_survivors
    result["bonferroni_threshold"] = bonf_thresh
    return result


def run_validation(
    db_path: str | Path,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    grid: dict,
) -> dict:
    """Full honest time-split validation pipeline.

    1. Load all data once (read_only=True).
    2. Sweep *grid* on the TRAIN window; apply snooping corrections to select
       the best config.
    3. Evaluate the frozen config EXACTLY ONCE on the TEST window.
    4. Compute Newey-West t-stats for both windows.
    5. Write results to data/validation_results.json and print a summary.

    Args:
        db_path: Path to congress.duckdb.
        train_start / train_end: In-sample calibration window.
        test_start / test_end: Genuine out-of-sample evaluation window.
        grid: Parameter grid dict (same format as sweep_configs).

    Returns:
        JSON-serializable dict containing train/test metrics, selected config,
        degradation ratio, and a plain-language verdict.
    """
    from analyzer.database import Database

    db = Database(Path(db_path), read_only=True)
    try:
        return _run_validation_with_db(
            db, train_start, train_end, test_start, test_end, grid
        )
    finally:
        db.conn.close()


# ---------------------------------------------------------------------------
# Private implementation helpers
# ---------------------------------------------------------------------------

def _run_validation_with_db(
    db,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    grid: dict,
) -> dict:
    """Inner implementation that accepts an open Database connection."""
    tx_start = pd.Timestamp("2021-10-07")
    tx_end = pd.Timestamp(test_end)
    price_end = pd.Timestamp(test_end) + pd.Timedelta(days=130)

    all_tx = db.get_transactions_by_date_range(tx_start, tx_end)
    all_tickers = sorted(
        set(all_tx["ticker"].dropna().astype(str).unique().tolist()) | {"SPY"}
    )
    prices = db.get_prices(all_tickers, tx_start, price_end)
    entry_prices = db.get_entry_prices(all_tickers, tx_start, price_end)

    n_tx = len(all_tx)
    n_tickers = prices.shape[1] if not prices.empty else 0
    print(f"Data loaded: {n_tx} transactions, {n_tickers} tickers")

    # Phase 1: sweep on TRAIN window only
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(
        f"Sweeping {n_combos} configs on TRAIN window "
        f"[{train_start} → {train_end}] ..."
    )

    train_df = sweep_configs(all_tx, prices, entry_prices, grid, train_start, train_end)
    selected = select_config(train_df)

    print(
        f"Selected: horizon={int(selected['horizon'])}, "
        f"min_buyers={int(selected['min_buyers'])}, "
        f"top_n={int(selected['top_n'])}, "
        f"scoring_mode={selected['scoring_mode']}"
    )
    print(
        f"  Snooping: {selected['n_survivors']}/{selected['n_trials']} configs "
        f"survive BH | Bonferroni threshold: "
        f"{selected['bonferroni_threshold']:.5f} | "
        f"survives={selected['survives_correction']}"
    )

    # Phase 2: evaluate frozen config ONCE on TEST window
    horizon = int(selected["horizon"])
    freq = int(selected["frequency_days"])
    lag = max(0, math.ceil(horizon / freq) - 1)
    decay_lambda = float(selected["decay_lambda"])
    bayes_prior = float(selected["bayes_prior_strength"])
    scoring_mode = str(selected.get("scoring_mode", "shrunk_alpha"))
    training_lookback = int(selected["training_lookback_days"])
    min_buyers = int(selected["min_buyers"])
    top_n = int(selected["top_n"])
    threshold = float(selected.get("threshold", 5.0))

    # Pre-compute signals for the frozen (horizon, decay_lambda) once
    frozen_signals = analysis.calculate_signal_potential(
        entry_prices, prices, [horizon], decay_lambda=decay_lambda,
    )

    train_params = BacktestParams(
        start_date=train_start,
        end_date=train_end,
        horizon=horizon,
        lookback_days=60,
        training_lookback_days=training_lookback,
        min_buyers=min_buyers,
        top_n=top_n,
        threshold=threshold,
        frequency_days=freq,
    )
    test_params = BacktestParams(
        start_date=test_start,
        end_date=test_end,
        horizon=horizon,
        lookback_days=60,
        training_lookback_days=training_lookback,
        min_buyers=min_buyers,
        top_n=top_n,
        threshold=threshold,
        frequency_days=freq,
    )

    # Re-run TRAIN with the selected config to get per-date series for NW stat
    train_result, train_per_date = _backtest_core(
        all_tx, prices, train_params, frozen_signals,
        bayes_prior_strength=bayes_prior,
        decay_lambda=decay_lambda,
        scoring_mode=scoring_mode,
    )

    # One evaluation on TEST — no peeking before this point
    test_result, test_per_date = _backtest_core(
        all_tx, prices, test_params, frozen_signals,
        bayes_prior_strength=bayes_prior,
        decay_lambda=decay_lambda,
        scoring_mode=scoring_mode,
    )

    train_t = newey_west_tstat(train_per_date, lag=lag)
    train_p = float(2.0 * stats.norm.sf(abs(train_t))) if math.isfinite(train_t) else 1.0
    test_t = newey_west_tstat(test_per_date, lag=lag)
    test_p = float(2.0 * stats.norm.sf(abs(test_t))) if math.isfinite(test_t) else 1.0

    spy_train = _spy_mean_return(prices, train_start, train_end, horizon)
    spy_test = _spy_mean_return(prices, test_start, test_end, horizon)

    train_alpha = train_result.overall_alpha
    test_alpha = test_result.overall_alpha
    deg_ratio = round(test_alpha / train_alpha, 3) if train_alpha != 0 else None

    verdict = _verdict(test_result, test_t, test_p, deg_ratio, selected["survives_correction"])

    config_keys = [
        "horizon", "frequency_days", "training_lookback_days",
        "min_buyers", "top_n", "decay_lambda", "bayes_prior_strength", "scoring_mode",
    ]
    frozen_config = {k: selected[k] for k in config_keys if k in selected}

    output = {
        "selected_config": frozen_config,
        "snooping": {
            "survives_bh": bool(selected["survives_correction"]),
            "n_trials": int(selected["n_trials"]),
            "n_survivors": int(selected["n_survivors"]),
            "bonferroni_threshold": float(selected["bonferroni_threshold"]),
        },
        "train": _window_metrics(train_result, train_t, train_p, spy_train),
        "test": _window_metrics(test_result, test_t, test_p, spy_test),
        "degradation_ratio": deg_ratio,
        "verdict": verdict,
        "meta": {
            "train_window": [str(train_start), str(train_end)],
            "test_window": [str(test_start), str(test_end)],
            "nw_lag": lag,
            "n_transactions": n_tx,
            "n_tickers": n_tickers,
        },
    }

    _print_summary(output)

    out_path = Path("data") / "validation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    return output


def _window_metrics(
    result: SweepResult,
    t_stat: float,
    p_val: float,
    spy_return: float | None,
) -> dict:
    return {
        "N": int(result.total_recs),
        "dates_evaluated": int(result.dates_evaluated),
        "mean_alpha": float(result.overall_alpha),
        "win_rate": float(result.win_rate),
        "rank1_alpha": float(result.rank1_alpha),
        "rank5_alpha": float(result.rank5_alpha),
        "alpha_slope": float(result.alpha_slope),
        "nw_tstat": round(t_stat, 4) if math.isfinite(t_stat) else None,
        "nw_pval": round(p_val, 6),
        "spy_mean_return": spy_return,
    }


def _spy_mean_return(
    prices: pd.DataFrame, start: date, end: date, horizon: int
) -> float | None:
    """Mean SPY forward return over the evaluation window, or None."""
    if prices.empty or "SPY" not in prices.columns:
        return None
    try:
        spy = prices["SPY"].dropna()
        dates = pd.date_range(start, end, freq="30D")
        returns = []
        for d in dates:
            ts = pd.Timestamp(d)
            end_ts = ts + pd.Timedelta(days=horizon)
            entry = spy.asof(ts)
            exit_ = spy.asof(end_ts)
            if pd.notna(entry) and pd.notna(exit_) and entry > 0:
                returns.append((exit_ - entry) / entry * 100)
        return round(float(np.mean(returns)), 2) if returns else None
    except Exception:
        return None


def _verdict(
    test_result: SweepResult,
    test_t: float,
    test_p: float,
    deg_ratio: float | None,
    survives_correction: bool,
) -> str:
    """Plain-language verdict: robust / partially robust / not robust."""
    positive_alpha = test_result.overall_alpha > 0
    # Use 10% significance for OOS (lower power due to smaller test window)
    significant = math.isfinite(test_t) and test_p < 0.10
    healthy_decay = deg_ratio is not None and deg_ratio > 0.3

    if positive_alpha and significant and healthy_decay and survives_correction:
        return "robust"
    if positive_alpha and healthy_decay:
        return "partially robust"
    return "not robust"


def _print_summary(output: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("=== Validation Summary ===")
    print(sep)

    cfg = output["selected_config"]
    print(
        f"  Config:  horizon={cfg['horizon']}, freq={cfg['frequency_days']}, "
        f"min_buyers={cfg['min_buyers']}, top_n={cfg['top_n']}, "
        f"mode={cfg['scoring_mode']}"
    )

    snoop = output["snooping"]
    print(
        f"  Snooping: {snoop['n_survivors']}/{snoop['n_trials']} survive BH "
        f"| Bonferroni α={snoop['bonferroni_threshold']:.5f} "
        f"| survives={snoop['survives_bh']}"
    )

    for label, key in [("TRAIN (in-sample)", "train"), ("TEST  (out-of-sample)", "test")]:
        m = output[key]
        t_str = f"{m['nw_tstat']:+.2f}" if m["nw_tstat"] is not None else "N/A"
        print(f"\n  {label}:")
        print(f"    N={m['N']} recs, {m['dates_evaluated']} dates")
        print(f"    mean alpha: {m['mean_alpha']:+.2f}%")
        print(f"    win rate:   {m['win_rate']:.1f}%")
        print(f"    rank1/5:    {m['rank1_alpha']:+.2f}% / {m['rank5_alpha']:+.2f}%")
        print(f"    NW t-stat:  {t_str}  (p={m['nw_pval']:.4f})")
        if m.get("spy_mean_return") is not None:
            print(f"    SPY return: {m['spy_mean_return']:+.2f}%")

    deg = output["degradation_ratio"]
    if deg is not None:
        print(f"\n  Degradation ratio (OOS/IS alpha): {deg:.3f}")
    print(f"  Verdict: {output['verdict'].upper()}")
    print(sep)
