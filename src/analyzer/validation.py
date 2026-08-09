"""Purged nested validation for PTR Alpha strategies.

The validation contract is fail closed:
* every phase ends early enough for the maximum executable holding to mature;
* one per-date net-alpha statistic drives inference, correction, selection, and verdict;
* arbitrary-dependence Bonferroni and synchronized block max-stat gates must pass;
* fewer than 999 null permutations can never produce a deployable configuration;
* the post-2025 final phase is locked and is never loaded by this module.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import platform
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyzer import analysis
from analyzer.exceptions import AnalysisError
from analyzer.pipeline import BacktestParams
from analyzer.snooping import bonferroni_correction, max_stat_block_permutation

logger = logging.getLogger(__name__)

MIN_DATES_FOR_CANDIDACY = 8
MIN_RECS_FOR_CANDIDACY = 20
MIN_RELEASE_PERMUTATIONS = 999
LOCKED_FINAL_START = date(2026, 1, 1)
VALIDATION_ENTRY_DELAY_DAYS = 0  # evaluate_backtest(use_dip_entry=False)
PRIMARY_METRIC = "mean_per_date_net_alpha"


@dataclass(frozen=True, slots=True)
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
    scheduled_dates: int = 0
    benchmark_dates: int = 0
    no_trade_dates: int = 0
    coverage_pct: float = 0.0
    overall_alpha: float = 0.0
    overall_return: float = 0.0
    overall_spy_return: float = 0.0
    rank1_alpha: float = 0.0
    rank5_alpha: float = 0.0
    alpha_slope: float = 0.0  # descriptive only; never a selection statistic
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0


def _empty_result(
    params: BacktestParams, bayes: float, decay: float, mode: str
) -> SweepResult:
    scheduled = len(
        pd.date_range(
            params.start_date, params.end_date, freq=f"{params.frequency_days}D"
        )
    )
    return SweepResult(
        horizon=params.horizon,
        frequency_days=params.frequency_days,
        training_lookback_days=params.training_lookback_days,
        min_buyers=params.min_buyers,
        top_n=params.top_n,
        decay_lambda=decay,
        bayes_prior_strength=bayes,
        scoring_mode=mode,
        scheduled_dates=scheduled,
    )


def _benchmark_return(
    prices: pd.DataFrame, as_of: pd.Timestamp, horizon: int
) -> float | None:
    """Return the executable SPY return used by evaluate_backtest on one date."""
    recommendation = pd.DataFrame(
        [{"rank": 1, "ticker": "SPY", "signal_score": 1.0, "instrument_type": "stock"}]
    )
    try:
        evaluated = analysis.evaluate_backtest(recommendation, prices, as_of, horizon)
    except (AnalysisError, KeyError):
        return None
    if evaluated.empty or "bt_spy_return_pct" not in evaluated.columns:
        return None
    value = evaluated["bt_spy_return_pct"].iloc[0]
    return float(value) if pd.notna(value) else None


def _backtest_core(
    all_transactions: pd.DataFrame,
    prices: pd.DataFrame,
    params: BacktestParams,
    signals: pd.DataFrame,
    bayes_prior_strength: float,
    decay_lambda: float,
    scoring_mode: str = "shrunk_alpha",
) -> tuple[SweepResult, pd.Series]:
    """Run one configuration and return its summary and primary alpha series.

    The support is the scheduled rebalance calendar for which the identical SPY
    benchmark is executable. A date with no executable strategy trade earns a
    zero cash return; it is not silently dropped. Validation removes the
    data-dependent ``optimal_horizon`` column so the declared horizon is the
    actual maximum holding used for both strategy and benchmark.
    """
    empty = _empty_result(params, bayes_prior_strength, decay_lambda, scoring_mode)
    as_of_dates = pd.date_range(
        params.start_date, params.end_date, freq=f"{params.frequency_days}D"
    )
    date_rows: list[dict] = []
    evaluated_rows: list[pd.DataFrame] = []
    total_recommendations = 0
    failures: list[tuple[pd.Timestamp, Exception]] = []

    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)
        benchmark_return = _benchmark_return(prices, as_of_ts, params.horizon)
        if benchmark_return is None:
            continue
        try:
            recommendations = analysis.backtest_recommendations(
                signals,
                all_transactions,
                as_of_ts,
                horizon=params.horizon,
                lookback_days=params.lookback_days,
                min_buyers=params.min_buyers,
                top_n=params.top_n,
                threshold=params.threshold,
                prices_df=prices,
                training_lookback_days=params.training_lookback_days,
                scoring_mode=scoring_mode,
                bayes_prior_strength=bayes_prior_strength,
            )
        except (AnalysisError, KeyError) as exc:
            failures.append((as_of_ts, exc))
            recommendations = pd.DataFrame()

        strategy_return = 0.0
        traded = False
        if not recommendations.empty:
            frozen_recommendations = recommendations.drop(
                columns=["optimal_horizon"], errors="ignore"
            )
            try:
                evaluated = analysis.evaluate_backtest(
                    frozen_recommendations, prices, as_of_ts, params.horizon
                )
            except (AnalysisError, KeyError) as exc:
                failures.append((as_of_ts, exc))
                evaluated = pd.DataFrame()
            if not evaluated.empty and "bt_return_pct" in evaluated.columns:
                returns = pd.to_numeric(evaluated["bt_return_pct"], errors="coerce")
                strategy_return = float(
                    returns.fillna(0.0).sum() / len(recommendations)
                )
                total_recommendations += int(returns.notna().sum())
                valid = evaluated[returns.notna()].copy()
                if not valid.empty:
                    valid.insert(0, "as_of_date", as_of_ts.date())
                    evaluated_rows.append(valid)
                    traded = True

        date_rows.append(
            {
                "as_of_date": as_of_ts,
                "strategy_return_pct": strategy_return,
                "spy_return_pct": benchmark_return,
                "net_alpha_pct": strategy_return - benchmark_return,
                "traded": traded,
            }
        )

    if failures:
        logger.warning(
            "Skipped %d recommendation/evaluation operation(s)", len(failures)
        )
    if not date_rows:
        return empty, pd.Series(dtype=float)

    by_date = pd.DataFrame(date_rows).set_index("as_of_date").sort_index()
    per_date = by_date["net_alpha_pct"].astype(float)
    combined = (
        pd.concat(evaluated_rows, ignore_index=True)
        if evaluated_rows
        else pd.DataFrame(columns=["rank", "bt_alpha_pct"])
    )
    valid_alpha = (
        combined.dropna(subset=["bt_alpha_pct"]) if not combined.empty else combined
    )
    rank_alpha = (
        valid_alpha.groupby("rank")["bt_alpha_pct"].mean()
        if not valid_alpha.empty
        else pd.Series(dtype=float)
    )
    rank1 = float(rank_alpha.loc[1]) if 1 in rank_alpha.index else math.nan
    rank5 = float(rank_alpha.loc[5]) if 5 in rank_alpha.index else math.nan
    slope = rank1 - rank5 if math.isfinite(rank1) and math.isfinite(rank5) else math.nan

    standard_deviation = float(per_date.std())
    sharpe = 0.0
    if len(per_date) > 1 and standard_deviation > 0:
        periods_per_year = 365.0 / params.frequency_days
        sharpe = float(
            per_date.mean() / standard_deviation * math.sqrt(periods_per_year)
        )
    cumulative = (1.0 + by_date["strategy_return_pct"] / 100.0).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    scheduled = len(as_of_dates)
    supported = len(by_date)

    result = SweepResult(
        horizon=params.horizon,
        frequency_days=params.frequency_days,
        training_lookback_days=params.training_lookback_days,
        min_buyers=params.min_buyers,
        top_n=params.top_n,
        decay_lambda=decay_lambda,
        bayes_prior_strength=bayes_prior_strength,
        scoring_mode=scoring_mode,
        total_recs=total_recommendations,
        dates_evaluated=supported,
        scheduled_dates=scheduled,
        benchmark_dates=supported,
        no_trade_dates=int((~by_date["traded"]).sum()),
        coverage_pct=round(100.0 * supported / scheduled, 2) if scheduled else 0.0,
        overall_alpha=round(float(per_date.mean()), 4),
        overall_return=round(float(by_date["strategy_return_pct"].mean()), 4),
        overall_spy_return=round(float(by_date["spy_return_pct"].mean()), 4),
        rank1_alpha=round(rank1, 4) if math.isfinite(rank1) else math.nan,
        rank5_alpha=round(rank5, 4) if math.isfinite(rank5) else math.nan,
        alpha_slope=round(slope, 4) if math.isfinite(slope) else math.nan,
        win_rate=round(float((per_date > 0).mean()) * 100.0, 2),
        sharpe=round(sharpe, 4),
        max_drawdown=round(float(drawdown.min()) * 100.0, 4),
    )
    return result, per_date


def run_single_backtest(
    all_transactions: pd.DataFrame,
    prices: pd.DataFrame,
    params: BacktestParams,
    signals: pd.DataFrame,
    bayes_prior_strength: float,
    decay_lambda: float,
    scoring_mode: str = "shrunk_alpha",
) -> SweepResult:
    result, _ = _backtest_core(
        all_transactions,
        prices,
        params,
        signals,
        bayes_prior_strength,
        decay_lambda,
        scoring_mode,
    )
    return result


def newey_west_tstat(alpha_series: pd.Series, lag: int) -> float:
    """Bartlett-kernel HAC t-statistic for the per-date net-alpha mean."""
    x = np.asarray(pd.Series(alpha_series).dropna(), dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    lag = max(0, min(int(lag), n - 1))
    mean = float(x.mean())
    demeaned = x - mean
    gamma = np.array(
        [np.dot(demeaned[k:], demeaned[: n - k]) / n for k in range(lag + 1)]
    )
    if lag == 0:
        long_run_variance = float(gamma[0])
    else:
        weights = 1.0 - np.arange(1, lag + 1) / (lag + 1)
        long_run_variance = float(gamma[0] + 2.0 * np.dot(weights, gamma[1:]))
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    if standard_error < 1e-14:
        if mean > 0:
            return math.inf
        if mean < 0:
            return -math.inf
        return 0.0
    return float(mean / standard_error)


def permute_signal_member_labels(
    signals_by_horizon: dict[tuple[int, float], pd.DataFrame], *, seed: int
) -> dict[tuple[int, float], pd.DataFrame]:
    """Return a deterministic member-attribution negative-control cache.

    A single bijection is used across horizons so each member's complete
    historical outcome path is attributed to another disclosed member while
    row order, dates, tickers, outcomes, and the member-count distribution stay
    unchanged. The real transaction candidate universe is not mutated.
    """
    members = sorted(
        {
            str(member)
            for frame in signals_by_horizon.values()
            if "member" in frame.columns
            for member in frame["member"].dropna().unique()
        }
    )
    if len(members) < 2:
        raise ValueError("member-label permutation requires at least two members")
    rng = np.random.default_rng(seed)
    permuted = list(rng.permutation(members))
    if all(source == target for source, target in zip(members, permuted)):
        permuted = permuted[1:] + permuted[:1]
    mapping = dict(zip(members, permuted))
    output: dict[tuple[int, float], pd.DataFrame] = {}
    for key, frame in signals_by_horizon.items():
        changed = frame.copy()
        changed["member"] = changed["member"].map(
            lambda value: mapping.get(str(value), value) if pd.notna(value) else value
        )
        output[key] = changed
    return output


def sweep_configs(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    entry_prices: pd.DataFrame,
    grid: dict,
    start: date,
    end: date,
    *,
    signals_by_horizon: dict[tuple[int, float], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Evaluate every configuration on one already-purged phase."""
    if end < start:
        raise ValueError("purged sweep phase has no executable dates")
    horizons = {int(value) for value in grid.get("horizon", [60])}
    decays = {float(value) for value in grid.get("decay_lambda", [0.005])}
    signal_cache = dict(signals_by_horizon or {})
    for horizon in horizons:
        for decay in decays:
            if (horizon, decay) not in signal_cache:
                signal_cache[(horizon, decay)] = analysis.calculate_signal_potential(
                    entry_prices, prices, [horizon], decay_lambda=decay
                )

    keys = list(grid)
    rows: list[dict] = []
    series_by_trial: dict[int, pd.Series] = {}
    for trial_id, combo in enumerate(itertools.product(*grid.values())):
        values = dict(zip(keys, combo))
        horizon = int(values["horizon"])
        frequency = int(values.get("frequency_days", 30))
        lag = max(0, math.ceil(horizon / frequency) - 1)
        params = BacktestParams(
            start_date=start,
            end_date=end,
            horizon=horizon,
            lookback_days=60,
            training_lookback_days=int(values.get("training_lookback_days", 365)),
            min_buyers=int(values["min_buyers"]),
            top_n=int(values["top_n"]),
            threshold=float(values.get("threshold", 5.0)),
            frequency_days=frequency,
        )
        decay = float(values["decay_lambda"])
        result, per_date = _backtest_core(
            all_tx,
            prices,
            params,
            signal_cache[(horizon, decay)],
            bayes_prior_strength=float(values["bayes_prior_strength"]),
            decay_lambda=decay,
            scoring_mode=str(values.get("scoring_mode", "shrunk_alpha")),
        )
        statistic = newey_west_tstat(per_date, lag)
        p_value = (
            float(stats.norm.sf(statistic))
            if math.isfinite(statistic)
            else (0.0 if statistic > 0 else 1.0)
        )
        row = asdict(result)
        row["trial_id"] = trial_id
        row["primary_metric"] = PRIMARY_METRIC
        row["nw_lag"] = lag
        row["nw_tstat"] = statistic
        row["p_value"] = p_value
        row["min_sample_ok"] = bool(
            result.dates_evaluated >= MIN_DATES_FOR_CANDIDACY
            and result.total_recs >= MIN_RECS_FOR_CANDIDACY
        )
        rows.append(row)
        series_by_trial[trial_id] = per_date
    frame = pd.DataFrame(rows)
    frame.attrs["series_by_trial"] = series_by_trial
    return frame


def select_config(
    sweep_df: pd.DataFrame,
    alpha: float = 0.05,
    *,
    series_by_trial: dict[int, pd.Series] | None = None,
    n_permutations: int = 999,
    permutation_seed: int = 0,
    block_days: int = 90,
) -> dict:
    """Select by the corrected primary metric or return no deployable config."""
    if sweep_df.empty:
        raise ValueError("sweep_df must not be empty")
    required = {"overall_alpha", "overall_return", "p_value", "nw_tstat"}
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"sweep_df missing required columns: {sorted(missing)}")

    working = sweep_df.copy()
    n_trials = len(working)
    bonferroni_threshold = bonferroni_correction(n_trials, alpha)
    candidate = (
        working["min_sample_ok"].fillna(False).astype(bool).to_numpy()
        if "min_sample_ok" in working.columns
        else np.ones(n_trials, dtype=bool)
    )
    source_series = series_by_trial or sweep_df.attrs.get("series_by_trial")
    expected_trial_ids = {
        int(row.get("trial_id", position)) for position, row in working.iterrows()
    }
    supplied_trial_ids = {int(key) for key in source_series} if source_series else set()
    complete_null_series = expected_trial_ids == supplied_trial_ids
    max_stat_p = np.ones(n_trials, dtype=float)
    null_ready = complete_null_series and n_permutations >= MIN_RELEASE_PERMUTATIONS
    permutation_summary: dict = {
        "method": "synchronized_calendar_block_sign_max_stat",
        "n_permutations": int(n_permutations),
        "minimum_release_permutations": MIN_RELEASE_PERMUTATIONS,
        "block_days": int(block_days),
        "seed": int(permutation_seed),
        "release_ready": null_ready,
    }
    if source_series and complete_null_series:
        normalized = {int(key): value for key, value in source_series.items()}
        lags = {
            int(row.get("trial_id", position)): int(row.get("nw_lag", 0))
            for position, row in working.iterrows()
        }
        permutation = max_stat_block_permutation(
            normalized,
            lags,
            n_permutations=n_permutations,
            block_days=block_days,
            seed=permutation_seed,
        )
        trial_ids = sorted(normalized)
        adjusted_by_trial = dict(zip(trial_ids, permutation.adjusted_p_values))
        max_stat_p = np.asarray(
            [
                float(adjusted_by_trial.get(int(row.get("trial_id", position)), 1.0))
                for position, row in working.iterrows()
            ]
        )
        permutation_summary["null_max_t_quantile_95"] = float(
            np.quantile(permutation.null_max_statistics, 0.95)
        )

    working["bonferroni_p_value"] = np.minimum(
        pd.to_numeric(working["p_value"], errors="coerce").fillna(1.0) * n_trials,
        1.0,
    )
    working["max_stat_p_value"] = max_stat_p
    survivor = (
        candidate
        & null_ready
        & (working["overall_alpha"].to_numpy(dtype=float) > 0)
        & (working["overall_return"].to_numpy(dtype=float) > 0)
        & (working["p_value"].to_numpy(dtype=float) <= bonferroni_threshold)
        & (max_stat_p <= alpha)
    )
    valid_positions = np.flatnonzero(candidate)
    descriptive_position = (
        int(
            valid_positions[
                np.argmax(working.iloc[valid_positions]["overall_alpha"].to_numpy())
            ]
        )
        if len(valid_positions)
        else int(np.argmax(working["overall_alpha"].to_numpy(dtype=float)))
    )
    descriptive = working.iloc[descriptive_position].to_dict()
    descriptive["label"] = "descriptive_only_not_deployable"

    survivor_positions = np.flatnonzero(survivor)
    deployable = None
    if len(survivor_positions):
        survivor_frame = working.iloc[survivor_positions]
        order = survivor_frame.sort_values(
            ["overall_alpha", "nw_tstat"], ascending=False
        )
        deployable = order.iloc[0].to_dict()
        deployable["label"] = "deployable_train_survivor"

    if not source_series:
        reason = "missing_null_series"
    elif not complete_null_series:
        reason = "incomplete_null_series"
    elif n_permutations < MIN_RELEASE_PERMUTATIONS:
        reason = "insufficient_null_permutations"
    elif deployable is None:
        reason = "no_dependence_safe_survivor"
    else:
        reason = None
    return {
        "deployable_config": deployable,
        "descriptive_best": descriptive,
        "failure_reason": reason,
        "primary_metric": PRIMARY_METRIC,
        "n_trials": n_trials,
        "n_min_sample_candidates": int(candidate.sum()),
        "n_survivors": int(survivor.sum()),
        "bonferroni_threshold": bonferroni_threshold,
        "alpha": alpha,
        "permutation": permutation_summary,
    }


def _phase_end(
    boundary_end: date, max_holding_days: int, max_entry_delay_days: int
) -> date:
    return (
        pd.Timestamp(boundary_end)
        - pd.Timedelta(days=max_holding_days + max_entry_delay_days)
    ).date()


def run_validation(
    db_path: str | Path,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    grid: dict,
    *,
    out_path: Path | None = None,
    n_permutations: int = 999,
    permutation_seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Run purged train selection and, only after survival, one test evaluation."""
    if train_end < train_start or test_end < test_start:
        raise ValueError("validation window end must be on or after its start")
    if test_start <= train_end:
        raise ValueError("test window must start after the training window ends")
    if test_end >= LOCKED_FINAL_START:
        raise ValueError(
            f"test window enters locked final phase starting {LOCKED_FINAL_START}"
        )
    if not grid or not grid.get("horizon"):
        raise ValueError("validation grid must include at least one horizon")
    horizons = [int(value) for value in grid["horizon"]]
    if any(value < 1 for value in horizons):
        raise ValueError("validation horizons must be positive")
    max_holding = max(horizons)
    train_effective_end = _phase_end(
        train_end, max_holding, VALIDATION_ENTRY_DELAY_DAYS
    )
    test_effective_end = _phase_end(test_end, max_holding, VALIDATION_ENTRY_DELAY_DAYS)
    if train_effective_end < train_start or test_effective_end < test_start:
        raise ValueError("phase is too short after executable holding-period purge")

    from analyzer.database import Database

    db_path = Path(db_path)
    db = Database(db_path, read_only=True)
    try:
        return _run_validation_with_db(
            db,
            db_path,
            train_start,
            train_end,
            train_effective_end,
            test_start,
            test_end,
            test_effective_end,
            grid,
            max_holding=max_holding,
            n_permutations=n_permutations,
            permutation_seed=permutation_seed,
            alpha=alpha,
            out_path=out_path,
        )
    finally:
        db.conn.close()


def _run_validation_with_db(
    db,
    db_path: Path,
    train_start: date,
    train_end: date,
    train_effective_end: date,
    test_start: date,
    test_end: date,
    test_effective_end: date,
    grid: dict,
    *,
    max_holding: int,
    n_permutations: int,
    permutation_seed: int,
    alpha: float,
    out_path: Path | None,
) -> dict:
    tx_start = pd.Timestamp("2021-10-07")
    tx_end = pd.Timestamp(test_effective_end)
    price_end = pd.Timestamp(test_end)
    all_tx = db.get_transactions_by_date_range(tx_start, tx_end)
    tickers = sorted(set(all_tx["ticker"].dropna().astype(str)) | {"SPY"})
    prices = db.get_prices(tickers, tx_start, price_end)
    entry_prices = db.get_entry_prices(tickers, tx_start, price_end)

    train_df = sweep_configs(
        all_tx, prices, entry_prices, grid, train_start, train_effective_end
    )
    selection = select_config(
        train_df,
        alpha,
        n_permutations=n_permutations,
        permutation_seed=permutation_seed,
        block_days=max_holding,
    )
    manifest = _build_manifest(
        db_path,
        all_tx,
        prices,
        entry_prices,
        grid,
        train_start,
        train_end,
        train_effective_end,
        test_start,
        test_end,
        test_effective_end,
        max_holding,
        n_permutations,
        permutation_seed,
        alpha,
    )
    output = {
        "status": "no_deployable_config",
        "primary_metric": PRIMARY_METRIC,
        "selected_config": None,
        "descriptive_train_best": _json_safe(selection["descriptive_best"]),
        "correction": _json_safe(
            {
                key: value
                for key, value in selection.items()
                if key not in {"deployable_config", "descriptive_best"}
            }
        ),
        "train": _metrics_from_row(selection["descriptive_best"], "descriptive_only"),
        "test": {"status": "not_run_without_corrected_train_survivor"},
        "degradation_ratio": None,
        "verdict": "not_robust",
        "manifest": manifest,
    }

    selected = selection["deployable_config"]
    if selected is not None:
        config = _config_from_row(selected)
        signals = analysis.calculate_signal_potential(
            entry_prices,
            prices,
            [int(config["horizon"])],
            decay_lambda=float(config["decay_lambda"]),
        )
        train_result, train_series = _run_frozen(
            all_tx, prices, signals, config, train_start, train_effective_end
        )
        test_result, test_series = _run_frozen(
            all_tx, prices, signals, config, test_start, test_effective_end
        )
        lag = max(
            0,
            math.ceil(int(config["horizon"]) / int(config["frequency_days"])) - 1,
        )
        train_t, train_p = _statistic_and_p(train_series, lag)
        test_t, test_p = _statistic_and_p(test_series, lag)
        test_passes = bool(
            test_result.dates_evaluated >= MIN_DATES_FOR_CANDIDACY
            and test_result.total_recs >= MIN_RECS_FOR_CANDIDACY
            and test_result.overall_alpha > 0
            and test_result.overall_return > 0
            and test_p <= alpha
        )
        output.update(
            status="validated" if test_passes else "failed_out_of_sample",
            selected_config=_json_safe(config),
            train=_window_metrics(
                train_result, train_t, train_p, "corrected_train_survivor"
            ),
            test=_window_metrics(
                test_result, test_t, test_p, "single_frozen_out_of_sample"
            ),
            degradation_ratio=(
                round(test_result.overall_alpha / train_result.overall_alpha, 4)
                if train_result.overall_alpha
                else None
            ),
            verdict="robust" if test_passes else "not_robust",
        )

    _print_summary(output)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    return output


def _run_frozen(all_tx, prices, signals, config, start: date, end: date):
    params = BacktestParams(
        start_date=start,
        end_date=end,
        horizon=int(config["horizon"]),
        lookback_days=60,
        training_lookback_days=int(config["training_lookback_days"]),
        min_buyers=int(config["min_buyers"]),
        top_n=int(config["top_n"]),
        threshold=float(config.get("threshold", 5.0)),
        frequency_days=int(config["frequency_days"]),
    )
    return _backtest_core(
        all_tx,
        prices,
        params,
        signals,
        float(config["bayes_prior_strength"]),
        float(config["decay_lambda"]),
        str(config.get("scoring_mode", "shrunk_alpha")),
    )


def _config_from_row(row: dict) -> dict:
    keys = [
        "horizon",
        "frequency_days",
        "training_lookback_days",
        "min_buyers",
        "top_n",
        "decay_lambda",
        "bayes_prior_strength",
        "scoring_mode",
    ]
    return {key: row[key] for key in keys if key in row}


def _statistic_and_p(series: pd.Series, lag: int) -> tuple[float, float]:
    statistic = newey_west_tstat(series, lag)
    p_value = (
        float(stats.norm.sf(statistic))
        if math.isfinite(statistic)
        else (0.0 if statistic > 0 else 1.0)
    )
    return statistic, p_value


def _window_metrics(
    result: SweepResult, statistic: float, p_value: float, label: str
) -> dict:
    return {
        "status": label,
        "N": result.total_recs,
        "dates_evaluated": result.dates_evaluated,
        "scheduled_dates": result.scheduled_dates,
        "benchmark_dates": result.benchmark_dates,
        "no_trade_dates": result.no_trade_dates,
        "coverage_pct": result.coverage_pct,
        "mean_net_alpha": result.overall_alpha,
        "mean_strategy_return": result.overall_return,
        "mean_spy_return": result.overall_spy_return,
        "win_rate": result.win_rate,
        "nw_tstat": round(statistic, 6) if math.isfinite(statistic) else None,
        "nw_pval": round(p_value, 8),
        "rank1_alpha_descriptive": result.rank1_alpha,
        "rank5_alpha_descriptive": result.rank5_alpha,
        "rank_slope_descriptive": result.alpha_slope,
    }


def _metrics_from_row(row: dict, label: str) -> dict:
    return {
        "status": label,
        "N": int(row.get("total_recs", 0)),
        "dates_evaluated": int(row.get("dates_evaluated", 0)),
        "scheduled_dates": int(row.get("scheduled_dates", 0)),
        "benchmark_dates": int(row.get("benchmark_dates", 0)),
        "no_trade_dates": int(row.get("no_trade_dates", 0)),
        "coverage_pct": float(row.get("coverage_pct", 0.0)),
        "mean_net_alpha": float(row.get("overall_alpha", 0.0)),
        "mean_strategy_return": float(row.get("overall_return", 0.0)),
        "mean_spy_return": float(row.get("overall_spy_return", 0.0)),
        "nw_tstat": _finite_or_none(row.get("nw_tstat")),
        "nw_pval": float(row.get("p_value", 1.0)),
        "label": "not_selected_for_deployment",
    }


def _finite_or_none(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _spy_mean_return(
    prices: pd.DataFrame,
    start: date,
    end: date,
    horizon: int,
    frequency_days: int = 30,
) -> float | None:
    """Mean executable SPY return on exactly the requested calendar support."""
    returns = [
        _benchmark_return(prices, pd.Timestamp(as_of), horizon)
        for as_of in pd.date_range(start, end, freq=f"{frequency_days}D")
    ]
    valid = [value for value in returns if value is not None]
    return round(float(np.mean(valid)), 4) if valid else None


def _build_manifest(
    db_path: Path,
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    entry_prices: pd.DataFrame,
    grid: dict,
    train_start: date,
    train_end: date,
    train_effective_end: date,
    test_start: date,
    test_end: date,
    test_effective_end: date,
    max_holding: int,
    n_permutations: int,
    permutation_seed: int,
    alpha: float,
) -> dict:
    config_payload = {
        "grid": grid,
        "alpha": alpha,
        "n_permutations": n_permutations,
        "permutation_seed": permutation_seed,
        "primary_metric": PRIMARY_METRIC,
        "max_holding_days": max_holding,
        "max_entry_delay_days": VALIDATION_ENTRY_DELAY_DAYS,
    }
    dependencies = {
        name: _dependency_version(name)
        for name in ["numpy", "pandas", "scipy", "duckdb"]
    }
    dependencies["python"] = platform.python_version()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phases": {
            "train": {
                "boundary": [str(train_start), str(train_end)],
                "executable_as_of": [str(train_start), str(train_effective_end)],
                "outcomes_end_by": str(train_end),
            },
            "test": {
                "boundary": [str(test_start), str(test_end)],
                "executable_as_of": [str(test_start), str(test_effective_end)],
                "outcomes_end_by": str(test_end),
            },
            "locked_final": {
                "start": str(LOCKED_FINAL_START),
                "end": None,
                "status": "locked_not_loaded_not_evaluated",
                "consumed": False,
            },
        },
        "purge": {
            "max_executable_entry_delay_days": VALIDATION_ENTRY_DELAY_DAYS,
            "max_possible_holding_days": max_holding,
            "calendar_purge_days": max_holding + VALIDATION_ENTRY_DELAY_DAYS,
        },
        "trial_grid": _json_safe(grid),
        "n_trials": int(math.prod(len(values) for values in grid.values())),
        "null": {
            "method": "synchronized_calendar_block_sign_max_stat",
            "n_permutations": n_permutations,
            "minimum_release_permutations": MIN_RELEASE_PERMUTATIONS,
            "seed": permutation_seed,
        },
        "hashes": {
            "database_sha256": _sha256_file(db_path),
            "value_snapshot_sha256": _value_snapshot_hash(all_tx, prices, entry_prices),
            "code_sha256": _code_hash(),
            "config_sha256": _sha256_json(config_payload),
            "git_revision": _git_revision(),
            "dependency_sha256": _sha256_json(dependencies),
        },
        "dependencies": dependencies,
        "coverage_input": {
            "transactions": len(all_tx),
            "price_rows": len(prices),
            "price_columns": len(prices.columns),
            "entry_price_rows": len(entry_prices),
            "price_start": str(prices.index.min()) if not prices.empty else None,
            "price_end": str(prices.index.max()) if not prices.empty else None,
        },
    }


def _dependency_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_snapshot_hash(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(json.dumps([str(value) for value in frame.columns]).encode())
        digest.update(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        )
    return digest.hexdigest()


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src" / "analyzer").rglob("*.py")) + [root / "sweep.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision() -> str:
    """Read the worktree Git revision without invoking a subprocess."""
    root = Path(__file__).resolve().parents[2]
    dot_git = root / ".git"
    try:
        if dot_git.is_file():
            git_dir = Path(dot_git.read_text().split(":", 1)[1].strip())
        else:
            git_dir = dot_git
        common_dir_file = git_dir / "commondir"
        common_dir = (
            (git_dir / common_dir_file.read_text().strip()).resolve()
            if common_dir_file.exists()
            else git_dir
        )
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        reference = head.removeprefix("ref: ")
        loose = common_dir / reference
        if loose.exists():
            return loose.read_text().strip()
        packed = common_dir / "packed-refs"
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                revision, name = line.split(" ", 1)
                if name == reference:
                    return revision
    except (OSError, IndexError, ValueError):
        return "unavailable"
    return "unavailable"


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (pd.Timestamp, date, datetime)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print_summary(output: dict) -> None:
    logger.info("Validation status: %s", output["status"])
    logger.info("Primary metric: %s", output["primary_metric"])
    logger.info("Verdict: %s", output["verdict"])
    if output["selected_config"] is None:
        logger.warning(
            "No deployable configuration: %s",
            output["correction"].get("failure_reason"),
        )
