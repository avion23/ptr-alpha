"""Purged nested validation for PTR Alpha strategies.

The validation contract is fail closed:
* every phase ends early enough for the maximum executable holding to mature;
* one per-date net-alpha statistic drives inference, correction, selection, and verdict;
* arbitrary-dependence Bonferroni and moving-block max-stat gates must pass;
* consensus is identity-invariant and has no member-identity hypothesis;
* identity-dependent scoring modes are nondeployable diagnostics;
* incomplete or under-resolved statistical-family controls fail closed;
* the post-2025 final phase is locked and is never loaded by this module.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import platform
import os
import shutil
import subprocess
import tempfile
import time
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
from analyzer.member_ranking.buyer_scoring import CONSENSUS_SCORER_PROVENANCE
from analyzer.snooping import bonferroni_correction, max_stat_moving_block_bootstrap

logger = logging.getLogger(__name__)

MIN_DATES_FOR_CANDIDACY = 8
MIN_RECS_FOR_CANDIDACY = 20
MIN_RELEASE_PERMUTATIONS = 999
LOCKED_FINAL_START = date(2026, 1, 1)
VALIDATION_ENTRY_DELAY_DAYS = 0  # evaluate_backtest(use_dip_entry=False)
PRIMARY_METRIC = "mean_per_date_net_alpha"
MEMBER_EXACT_GROUP_LIMIT = 720
MEMBER_RUNTIME_BUDGET_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class MemberIdentityControlResult:
    status: str
    gating: bool
    method: str
    requested_permutations: int
    evaluated_permutations: int
    permutation_group_size: int
    exact_enumeration: bool
    sampled_without_replacement: bool
    p_value_resolution: float
    max_stat_p_value: float
    null_max_t_quantile_95: float | None
    release_ready: bool
    runtime_seconds: float
    runtime_budget_seconds: float
    family_sha256: str
    observed_trial_id: int
    observed_statistic: float


@dataclass(frozen=True, slots=True)
class SweepResult:
    horizon: int
    frequency_days: int
    training_lookback_days: int
    min_buyers: int
    top_n: int
    decay_lambda: float
    bayes_prior_strength: float
    scoring_mode: str = "consensus"
    scorer_provenance: str = ""
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
    scoring_mode: str = "consensus",
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
                as_of_date=as_of_ts,
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
            if scoring_mode == "consensus" and not recommendations.empty:
                provenance = set(
                    recommendations.get(
                        "scorer_provenance", pd.Series(dtype=str)
                    ).dropna()
                )
                if provenance != {CONSENSUS_SCORER_PROVENANCE}:
                    raise AnalysisError(
                        "consensus recommendations lack executed-scorer provenance"
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
        scorer_provenance=(
            CONSENSUS_SCORER_PROVENANCE
            if scoring_mode == "consensus" and total_recommendations > 0
            else "descriptive_member_skill_v1"
            if scoring_mode != "consensus"
            else ""
        ),
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
    scoring_mode: str = "consensus",
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
    signals_by_horizon: dict[tuple[int, float], pd.DataFrame],
    *,
    seed: int | None = None,
    permutation: tuple[str, ...] | None = None,
) -> dict[tuple[int, float], pd.DataFrame]:
    """Apply one full-group member-label bijection across every horizon."""
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
    if permutation is None:
        rng = np.random.default_rng(seed)
        permutation = tuple(str(value) for value in rng.permutation(members))
    if len(permutation) != len(members) or set(permutation) != set(members):
        raise ValueError("permutation must be a bijection over every member")
    mapping = dict(zip(members, permutation))
    output: dict[tuple[int, float], pd.DataFrame] = {}
    for key, frame in signals_by_horizon.items():
        changed = frame.copy()
        changed["member"] = changed["member"].map(
            lambda value: mapping.get(str(value), value) if pd.notna(value) else value
        )
        output[key] = changed
    return output


def _member_identity_permutations(
    members: list[str], requested: int, seed: int
) -> tuple[list[tuple[str, ...]], int, bool]:
    """Enumerate small groups; otherwise draw a uniform unique subset."""
    if requested < 1:
        raise ValueError("requested member permutations must be positive")
    group_size = math.factorial(len(members))
    if group_size <= MEMBER_EXACT_GROUP_LIMIT or requested >= group_size:
        return list(itertools.permutations(members)), group_size, True
    target = requested
    rng = np.random.default_rng(seed)
    sampled: set[tuple[str, ...]] = set()
    while len(sampled) < target:
        sampled.add(tuple(str(value) for value in rng.permutation(members)))
    return sorted(sampled), group_size, False


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
            scoring_mode=str(values.get("scoring_mode", "consensus")),
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
        row["asymptotic_p_value_descriptive"] = p_value
        row["min_sample_ok"] = bool(
            result.dates_evaluated >= MIN_DATES_FOR_CANDIDACY
            and result.total_recs >= MIN_RECS_FOR_CANDIDACY
        )
        rows.append(row)
        series_by_trial[trial_id] = per_date
    frame = pd.DataFrame(rows)
    frame.attrs["series_by_trial"] = series_by_trial
    return frame


def _member_family_sha256(
    sweep_df: pd.DataFrame, series_by_trial: dict[int, pd.Series]
) -> str:
    digest = hashlib.sha256()
    ordered = sweep_df.sort_values("trial_id").copy()
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    for trial_id in sorted(series_by_trial):
        digest.update(str(trial_id).encode())
        series = pd.Series(series_by_trial[trial_id], dtype=float).sort_index()
        digest.update(
            pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes()
        )
    return digest.hexdigest()


def select_config(
    sweep_df: pd.DataFrame,
    alpha: float = 0.05,
    *,
    series_by_trial: dict[int, pd.Series] | None = None,
    n_permutations: int = 999,
    permutation_seed: int = 0,
) -> dict:
    """Select consensus by statistical-family gates; authorization stays ledger-only."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if sweep_df.empty:
        raise ValueError("sweep_df must not be empty")
    required = {
        "trial_id",
        "overall_alpha",
        "overall_return",
        "nw_tstat",
        "nw_lag",
        "horizon",
        "frequency_days",
        "scoring_mode",
        "scorer_provenance",
    }
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"sweep_df missing required columns: {sorted(missing)}")
    if sweep_df["trial_id"].duplicated().any():
        raise ValueError("trial_id values must be unique")

    working = sweep_df.copy()
    n_trials = len(working)
    bonferroni_threshold = bonferroni_correction(n_trials, alpha)
    candidate = (
        working["min_sample_ok"].fillna(False).astype(bool).to_numpy(copy=True)
        if "min_sample_ok" in working.columns
        else np.ones(n_trials, dtype=bool)
    )
    candidate &= working["scoring_mode"].astype(str).eq("consensus").to_numpy()
    candidate &= (
        working["scorer_provenance"]
        .astype(str)
        .eq(CONSENSUS_SCORER_PROVENANCE)
        .to_numpy()
    )
    source_series = series_by_trial or sweep_df.attrs.get("series_by_trial")
    expected_trial_ids = {int(value) for value in working["trial_id"]}
    supplied_trial_ids = {int(key) for key in source_series} if source_series else set()
    complete_series = expected_trial_ids == supplied_trial_ids
    bootstrap_p = np.ones(n_trials, dtype=float)
    max_stat_p = np.ones(n_trials, dtype=float)
    bootstrap_error = None
    minimum_resolution_bootstrap = max(
        MIN_RELEASE_PERMUTATIONS, math.ceil(n_trials / alpha) - 1
    )
    bootstrap_summary: dict = {
        "method": "centered_moving_block_bootstrap_max_stat",
        "n_bootstrap": int(n_permutations),
        "minimum_release_bootstrap": MIN_RELEASE_PERMUTATIONS,
        "minimum_family_resolution_bootstrap": minimum_resolution_bootstrap,
        "seed": int(permutation_seed),
        "centered_null": True,
        "release_ready": False,
    }
    if source_series and complete_series:
        normalized = {
            int(key): pd.Series(value, dtype=float)
            for key, value in source_series.items()
        }
        lags = {
            int(row["trial_id"]): int(row["nw_lag"]) for _, row in working.iterrows()
        }
        block_lengths = {
            int(row["trial_id"]): max(
                1, math.ceil(int(row["horizon"]) / int(row["frequency_days"]))
            )
            for _, row in working.iterrows()
        }
        try:
            bootstrap = max_stat_moving_block_bootstrap(
                normalized,
                lags,
                block_lengths,
                n_bootstrap=n_permutations,
                seed=permutation_seed,
            )
            ordered = sorted(normalized)
            marginal_by_trial = dict(zip(ordered, bootstrap.marginal_p_values))
            adjusted_by_trial = dict(zip(ordered, bootstrap.adjusted_p_values))
            bootstrap_p = np.asarray(
                [float(marginal_by_trial[int(value)]) for value in working["trial_id"]]
            )
            max_stat_p = np.asarray(
                [float(adjusted_by_trial[int(value)]) for value in working["trial_id"]]
            )
            bootstrap_summary.update(
                release_ready=n_permutations >= minimum_resolution_bootstrap,
                null_max_t_quantile_95=float(
                    np.quantile(bootstrap.null_max_statistics, 0.95)
                ),
                assumptions=list(bootstrap.assumptions),
            )
        except ValueError as exc:
            bootstrap_error = str(exc)
            bootstrap_summary["error"] = bootstrap_error

    bootstrap_ready = bool(
        complete_series
        and bootstrap_error is None
        and n_permutations >= minimum_resolution_bootstrap
    )
    working["bootstrap_p_value"] = bootstrap_p
    working["bonferroni_p_value"] = np.minimum(bootstrap_p * n_trials, 1.0)
    working["max_stat_p_value"] = max_stat_p
    statistical_survivor = (
        candidate
        & bootstrap_ready
        & (working["overall_alpha"].to_numpy(dtype=float) > 0)
        & (working["overall_return"].to_numpy(dtype=float) > 0)
        & (bootstrap_p <= bonferroni_threshold)
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

    survivor_positions = np.flatnonzero(statistical_survivor)
    statistical_candidate = None
    if len(survivor_positions):
        order = working.iloc[survivor_positions].sort_values(
            ["overall_alpha", "nw_tstat"], ascending=False
        )
        statistical_candidate = order.iloc[0].to_dict()
        statistical_candidate["label"] = "statistical_family_survivor"

    deployable = None
    if statistical_candidate is not None:
        deployable = dict(statistical_candidate)
        deployable["label"] = "deployable_statistical_family_survivor"
    member_summary = {
        "status": (
            "audit_pending"
            if statistical_candidate is not None
            else "not_needed_no_statistical_candidate"
        ),
        "gating": False,
        "diagnostic_only": True,
    }

    if not source_series:
        reason = "missing_bootstrap_series"
    elif not complete_series:
        reason = "incomplete_bootstrap_series"
    elif bootstrap_error is not None:
        reason = "bootstrap_sample_too_small"
    elif n_permutations < minimum_resolution_bootstrap:
        reason = "insufficient_bootstrap_count_or_family_resolution"
    elif statistical_candidate is None:
        reason = "no_dependence_safe_survivor"
    else:
        reason = None
    return {
        "deployable_config": deployable,
        "statistical_candidate": statistical_candidate,
        "descriptive_best": descriptive,
        "failure_reason": reason,
        "primary_metric": PRIMARY_METRIC,
        "n_trials": n_trials,
        "n_min_sample_candidates": int(candidate.sum()),
        "n_statistical_survivors": int(statistical_survivor.sum()),
        "n_survivors": 1 if deployable is not None else 0,
        "bonferroni_threshold": bonferroni_threshold,
        "alpha": alpha,
        "bootstrap": bootstrap_summary,
        "member_identity_control": member_summary,
    }


def _empirical_upper_quantile(values: list[float], probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def _run_identity_invariant_control(
    sweep_df: pd.DataFrame,
    observed_trial_id: int,
    ledger_path: Path,
) -> MemberIdentityControlResult:
    """Record that consensus has no member-identity hypothesis to test."""
    series_by_trial = sweep_df.attrs.get("series_by_trial")
    expected_ids = {int(value) for value in sweep_df["trial_id"]}
    if not isinstance(series_by_trial, dict) or set(series_by_trial) != expected_ids:
        raise ValueError("identity-invariant family lacks complete trial series")
    selected = sweep_df[sweep_df["trial_id"] == observed_trial_id]
    if len(selected) != 1:
        raise ValueError("observed trial_id is not unique in consensus family")
    row = selected.iloc[0]
    if (
        str(row["scoring_mode"]) != "consensus"
        or str(row["scorer_provenance"]) != CONSENSUS_SCORER_PROVENANCE
    ):
        raise ValueError(
            "identity-invariant control requires executed consensus provenance"
        )
    result = MemberIdentityControlResult(
        status="identity_invariant",
        gating=False,
        method="identity_invariant_by_consensus_scorer_contract_v1",
        requested_permutations=0,
        evaluated_permutations=0,
        permutation_group_size=1,
        exact_enumeration=True,
        sampled_without_replacement=False,
        p_value_resolution=1.0,
        max_stat_p_value=1.0,
        null_max_t_quantile_95=None,
        release_ready=True,
        runtime_seconds=0.0,
        runtime_budget_seconds=0.0,
        family_sha256=_member_family_sha256(sweep_df, series_by_trial),
        observed_trial_id=observed_trial_id,
        observed_statistic=float(row["nw_tstat"]),
    )
    _record_member_control(ledger_path, result)
    return result


def _run_member_identity_control(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    entry_prices: pd.DataFrame,
    grid: dict,
    start: date,
    end: date,
    *,
    observed_trial_id: int,
    ledger_path: Path,
    n_permutations: int,
    seed: int,
    runtime_budget_seconds: float = MEMBER_RUNTIME_BUDGET_SECONDS,
) -> MemberIdentityControlResult:
    """Execute and fingerprint the actual family, then run its identity null."""
    started = time.perf_counter()
    signal_cache: dict[tuple[int, float], pd.DataFrame] = {}
    for horizon in {int(value) for value in grid["horizon"]}:
        for decay in {float(value) for value in grid["decay_lambda"]}:
            signal_cache[(horizon, decay)] = analysis.calculate_signal_potential(
                entry_prices, prices, [horizon], decay_lambda=decay
            )
    baseline_frame = sweep_configs(
        all_tx,
        prices,
        entry_prices,
        grid,
        start,
        end,
        signals_by_horizon=signal_cache,
    )
    baseline_series = baseline_frame.attrs.get("series_by_trial")
    expected_ids = {int(value) for value in baseline_frame["trial_id"]}
    if not isinstance(baseline_series, dict) or set(baseline_series) != expected_ids:
        raise ValueError("executed member-control family lacks complete trial series")
    selected = baseline_frame[baseline_frame["trial_id"] == observed_trial_id]
    if len(selected) != 1:
        raise ValueError("observed trial_id is not unique in executed family")
    observed_statistic = float(selected.iloc[0]["nw_tstat"])
    family_sha256 = _member_family_sha256(baseline_frame, baseline_series)
    baseline_eligible = baseline_frame[baseline_frame["min_sample_ok"]]
    baseline_max = (
        float(baseline_eligible["nw_tstat"].max())
        if not baseline_eligible.empty
        else -math.inf
    )

    members = sorted(
        {
            str(member)
            for frame in signal_cache.values()
            for member in frame["member"].dropna().unique()
        }
    )
    if len(members) < 2:
        raise ValueError("member-label permutation requires at least two members")
    permutations, group_size, exact = _member_identity_permutations(
        members, n_permutations, seed
    )
    null_max_statistics: list[float] = []
    status = "completed"
    identity = tuple(members)
    for identity_permutation in permutations:
        if time.perf_counter() - started >= runtime_budget_seconds:
            status = "infeasible_runtime_budget"
            break
        if identity_permutation == identity:
            null_max_statistics.append(baseline_max)
            continue
        permuted = permute_signal_member_labels(
            signal_cache, permutation=identity_permutation
        )
        null_frame = sweep_configs(
            all_tx,
            prices,
            entry_prices,
            grid,
            start,
            end,
            signals_by_horizon=permuted,
        )
        eligible = null_frame[null_frame["min_sample_ok"]]
        null_max_statistics.append(
            float(eligible["nw_tstat"].max()) if not eligible.empty else -math.inf
        )
    evaluated = len(null_max_statistics)
    complete_exact = exact and evaluated == group_size
    complete_sample = not exact and evaluated == len(permutations)
    if evaluated == 0:
        max_stat_p = 1.0
        quantile = None
    elif complete_exact:
        exceedances = int(np.sum(np.asarray(null_max_statistics) >= observed_statistic))
        max_stat_p = float(max(1, exceedances) / group_size)
        quantile = _empirical_upper_quantile(null_max_statistics, 0.95)
    else:
        max_stat_p = float(
            (1.0 + np.sum(np.asarray(null_max_statistics) >= observed_statistic))
            / (evaluated + 1.0)
        )
        quantile = _empirical_upper_quantile(null_max_statistics, 0.95)
    release_ready = bool(
        status == "completed"
        and (
            complete_exact
            or (complete_sample and evaluated >= MIN_RELEASE_PERMUTATIONS)
        )
    )
    payload = {
        "status": status,
        "gating": False,
        "method": "uniform_full_permutation_group_family_max_stat",
        "requested_permutations": n_permutations,
        "evaluated_permutations": evaluated,
        "permutation_group_size": group_size,
        "exact_enumeration": complete_exact,
        "sampled_without_replacement": not exact,
        "p_value_resolution": (
            1.0 / group_size if complete_exact else 1.0 / (evaluated + 1.0)
        ),
        "max_stat_p_value": max_stat_p,
        "null_max_t_quantile_95": quantile,
        "release_ready": release_ready,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "runtime_budget_seconds": runtime_budget_seconds,
        "family_sha256": family_sha256,
        "observed_trial_id": observed_trial_id,
        "observed_statistic": observed_statistic,
    }
    result = MemberIdentityControlResult(**payload)
    _record_member_control(ledger_path, result)
    return result


def _phase_end(
    boundary_end: date, max_holding_days: int, max_entry_delay_days: int
) -> date:
    return (
        pd.Timestamp(boundary_end)
        - pd.Timedelta(days=max_holding_days + max_entry_delay_days)
    ).date()


class EvaluationAlreadyConsumedError(RuntimeError):
    """Raised when a frozen evaluation overlaps a consumed interval."""


class EvaluationLedgerIntegrityError(RuntimeError):
    """Raised when the local append-only ledger hash chain is invalid."""


def _canonical_ledger_path(db_path: Path) -> Path:
    return db_path.resolve().parent / ".ptr-alpha-evaluation-ledger-v2.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _empty_ledger() -> dict:
    return {
        "schema_version": 2,
        "integrity": "append_only_sha256_hash_chain",
        "local_tamper_limitation": (
            "A local attacker who can rewrite the ledger can recompute the chain; "
            "external anchoring is not implemented."
        ),
        "events": [],
    }


def _validate_ledger(ledger: dict) -> None:
    if ledger.get("schema_version") != 2 or not isinstance(ledger.get("events"), list):
        raise EvaluationLedgerIntegrityError("unsupported or malformed ledger")
    previous = "0" * 64
    for sequence, event in enumerate(ledger["events"]):
        payload = {key: value for key, value in event.items() if key != "event_sha256"}
        if (
            payload.get("sequence") != sequence
            or payload.get("previous_sha256") != previous
        ):
            raise EvaluationLedgerIntegrityError(
                "ledger sequence or previous hash is invalid"
            )
        expected = _sha256_json(payload)
        if event.get("event_sha256") != expected:
            raise EvaluationLedgerIntegrityError("ledger event hash is invalid")
        previous = expected


def _append_ledger_event(ledger: dict, event: dict) -> None:
    previous = ledger["events"][-1]["event_sha256"] if ledger["events"] else "0" * 64
    payload = {
        **event,
        "sequence": len(ledger["events"]),
        "previous_sha256": previous,
    }
    ledger["events"].append({**payload, "event_sha256": _sha256_json(payload)})


def _refuse_legacy_ledger(ledger_path: Path) -> None:
    legacy_path = ledger_path.parent / "validation_evaluation_ledger.json"
    if not legacy_path.exists():
        return
    try:
        legacy = json.loads(legacy_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationLedgerIntegrityError(
            "legacy validation_evaluation_ledger.json exists but is unreadable; "
            "archive or migrate it explicitly before validation"
        ) from exc
    if legacy.get("evaluations"):
        raise EvaluationLedgerIntegrityError(
            "legacy validation_evaluation_ledger.json contains consumed evaluations; "
            "archive or migrate it explicitly before validation"
        )
    raise EvaluationLedgerIntegrityError(
        "legacy validation_evaluation_ledger.json exists; archive or migrate it "
        "explicitly before validation"
    )


def _record_member_control(
    ledger_path: Path, result: MemberIdentityControlResult
) -> None:
    """Append a non-gating identity diagnostic audit event."""
    import fcntl

    if result.gating:
        raise TypeError("identity diagnostics must declare gating=False")
    _refuse_legacy_ledger(ledger_path)
    lock_path = ledger_path.with_suffix(f"{ledger_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else _empty_ledger()
        )
        _validate_ledger(ledger)
        _append_ledger_event(
            ledger,
            {
                "event_type": "member_identity_control",
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "control": asdict(result),
            },
        )
        _atomic_write_json(ledger_path, ledger)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _reserve_evaluation(
    ledger_path: Path,
    manifest: dict,
    config: dict,
    grid: dict,
    test_start: date,
    test_end: date,
) -> str:
    """Atomically append a reservation before a frozen evaluation."""
    import fcntl

    _refuse_legacy_ledger(ledger_path)
    lock_path = ledger_path.with_suffix(f"{ledger_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = (
            json.loads(ledger_path.read_text())
            if ledger_path.exists()
            else _empty_ledger()
        )
        _validate_ledger(ledger)
        for event in ledger["events"]:
            if event.get("event_type") != "reservation":
                continue
            prior_start, prior_end = map(date.fromisoformat, event["window"])
            if test_start <= prior_end and prior_start <= test_end:
                raise EvaluationAlreadyConsumedError(
                    "frozen evaluation interval overlaps a consumed reservation; "
                    "repeats and alternate configs/grids/snapshots are refused"
                )
        hashes = manifest["hashes"]
        window = [str(test_start), str(test_end)]
        key_payload = {
            "database_sha256": hashes["database_sha256"],
            "value_snapshot_sha256": hashes["value_snapshot_sha256"],
            "config": config,
            "grid": grid,
            "window": window,
        }
        evaluation_key = _sha256_json(key_payload)
        _append_ledger_event(
            ledger,
            {
                "event_type": "reservation",
                "evaluation_key": evaluation_key,
                "database_sha256": hashes["database_sha256"],
                "value_snapshot_sha256": hashes["value_snapshot_sha256"],
                "config_sha256": _sha256_json(config),
                "grid_sha256": _sha256_json(grid),
                "window": window,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "reserved_consumed",
            },
        )
        _atomic_write_json(ledger_path, ledger)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return evaluation_key


def _complete_evaluation(ledger_path: Path, evaluation_key: str, status: str) -> None:
    import fcntl

    lock_path = ledger_path.with_suffix(f"{ledger_path.suffix}.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_text())
        _validate_ledger(ledger)
        reservations = {
            event["evaluation_key"]
            for event in ledger["events"]
            if event.get("event_type") == "reservation"
        }
        if evaluation_key not in reservations:
            raise EvaluationLedgerIntegrityError("completion has no reservation")
        _append_ledger_event(
            ledger,
            {
                "event_type": "completion",
                "evaluation_key": evaluation_key,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
            },
        )
        _atomic_write_json(ledger_path, ledger)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if n_permutations < 1:
        raise ValueError("bootstrap count must be positive")
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
            evaluation_ledger_path=_canonical_ledger_path(db_path),
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
    evaluation_ledger_path: Path,
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
    )
    statistical_candidate = selection["statistical_candidate"]
    if statistical_candidate is not None:
        identity_diagnostic = _run_identity_invariant_control(
            train_df,
            int(statistical_candidate["trial_id"]),
            evaluation_ledger_path,
        )
        selection["member_identity_control"] = asdict(identity_diagnostic)
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
                if key
                not in {
                    "deployable_config",
                    "statistical_candidate",
                    "descriptive_best",
                }
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
        train_result, _ = _run_frozen(
            all_tx, prices, signals, config, train_start, train_effective_end
        )
        evaluation_key = _reserve_evaluation(
            evaluation_ledger_path, manifest, config, grid, test_start, test_end
        )
        try:
            test_result, test_series = _run_frozen(
                all_tx, prices, signals, config, test_start, test_effective_end
            )
            lag = max(
                0,
                math.ceil(int(config["horizon"]) / int(config["frequency_days"])) - 1,
            )
            block_length = max(
                1, math.ceil(int(config["horizon"]) / int(config["frequency_days"]))
            )
            test_t, test_p, test_bootstrap_error = _bootstrap_statistic_and_p(
                test_series,
                lag,
                block_length,
                n_permutations,
                permutation_seed + 3 * n_permutations,
            )
            test_passes = bool(
                test_bootstrap_error is None
                and test_result.dates_evaluated >= MIN_DATES_FOR_CANDIDACY
                and test_result.total_recs >= MIN_RECS_FOR_CANDIDACY
                and test_result.overall_alpha > 0
                and test_result.overall_return > 0
                and test_p <= alpha
            )
            output.update(
                status=(
                    "retrospective_positive_result"
                    if test_passes
                    else "retrospective_failed_result"
                ),
                selected_config=_json_safe(config),
                train=_window_metrics(
                    train_result,
                    float(selected["nw_tstat"]),
                    float(selected["bootstrap_p_value"]),
                    "corrected_train_survivor",
                ),
                test=_window_metrics(
                    test_result,
                    test_t,
                    test_p,
                    "retrospective_previously_used_not_fresh_oos",
                ),
                degradation_ratio=(
                    round(test_result.overall_alpha / train_result.overall_alpha, 4)
                    if train_result.overall_alpha
                    else None
                ),
                verdict="not_fresh_oos_evidence",
                evaluation_ledger={
                    "path": str(evaluation_ledger_path),
                    "evaluation_key": evaluation_key,
                    "status": "consumed_retrospective",
                },
            )
            if test_bootstrap_error is not None:
                output["test"]["bootstrap_error"] = test_bootstrap_error
            _complete_evaluation(
                evaluation_ledger_path, evaluation_key, "completed_retrospective"
            )
        except Exception:
            _complete_evaluation(
                evaluation_ledger_path, evaluation_key, "failed_consumed"
            )
            raise

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
        str(config.get("scoring_mode", "consensus")),
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


def _bootstrap_statistic_and_p(
    series: pd.Series,
    lag: int,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, str | None]:
    try:
        result = max_stat_moving_block_bootstrap(
            {0: series},
            {0: lag},
            {0: block_length},
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    except ValueError as exc:
        return newey_west_tstat(series, lag), 1.0, str(exc)
    return (
        float(result.observed_statistics[0]),
        float(result.marginal_p_values[0]),
        None,
    )


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
        "nw_pval": float(row.get("bootstrap_p_value", 1.0)),
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
    git_state = _git_state()
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
                "evidence_class": "retrospective_previously_used_not_fresh_oos",
            },
            "locked_final": {
                "start": str(LOCKED_FINAL_START),
                "end": None,
                "status": "locked_not_queried_or_evaluated",
                "value_rows_queried": False,
                "whole_database_file_hashed_for_provenance": True,
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
            "bootstrap_method": "centered_moving_block_bootstrap_max_stat",
            "n_bootstrap": n_permutations,
            "member_identity_policy": (
                "consensus_is_identity_invariant_no_member_identity_hypothesis"
            ),
            "identity_dependent_modes": "descriptive_non_deployable",
            "member_control_integrity": (
                "audit_only_canonical_hash_chain_event_not_authorization"
            ),
            "minimum_release_count": MIN_RELEASE_PERMUTATIONS,
            "minimum_family_resolution_bootstrap": max(
                MIN_RELEASE_PERMUTATIONS,
                math.ceil(math.prod(len(values) for values in grid.values()) / alpha)
                - 1,
            ),
            "seed": permutation_seed,
            "assumptions": [
                "local stationarity within moving blocks",
                "shared block-start uniforms preserve aligned cross-config dependence",
                "Bonferroni is the arbitrary-dependence controlling gate",
            ],
        },
        "hashes": {
            "database_sha256": _sha256_file(db_path),
            "value_snapshot_sha256": _value_snapshot_hash(all_tx, prices, entry_prices),
            "code_sha256": _code_hash(),
            "config_sha256": _sha256_json(config_payload),
            "git_revision": git_state["revision"],
            "git_diff_sha256": git_state["diff_sha256"],
            "dependency_sha256": _sha256_json(dependencies),
        },
        "git": git_state,
        "evaluation_ledger": {
            "path": str(_canonical_ledger_path(db_path)),
            "integrity": "append_only_sha256_hash_chain",
            "overlap_policy": "any_overlapping_reserved_interval_is_consumed",
            "legacy_v1_policy": (
                "validation_evaluation_ledger.json must be explicitly archived or migrated"
            ),
            "local_tamper_limitation": (
                "A local attacker who can rewrite the ledger can recompute the chain; "
                "external anchoring is not implemented."
            ),
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


def _hash_untracked_path(digest: "hashlib._Hash", root: Path, path: Path) -> None:
    paths = [path]
    if path.is_dir() and not path.is_symlink():
        paths = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_symlink() or not candidate.is_dir()
        )
    for candidate in paths:
        relative = str(candidate.relative_to(root))
        digest.update(relative.encode(errors="surrogateescape"))
        if candidate.is_symlink():
            digest.update(os.readlink(candidate).encode(errors="surrogateescape"))
        elif candidate.is_file():
            digest.update(candidate.read_bytes())


def _git_state() -> dict:
    """Return revision, dirty state, and a content hash of tracked/untracked diff."""
    root = Path(__file__).resolve().parents[2]
    git = shutil.which("git")
    if git is None:
        return {"revision": "unavailable", "dirty": None, "diff_sha256": "unavailable"}
    try:
        revision = subprocess.run(  # nosec B603
            [git, "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(  # nosec B603
            [git, "status", "--porcelain", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        diff = subprocess.run(  # nosec B603
            [git, "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256(diff)
        entries = [entry for entry in status.split(b"\0") if entry]
        for entry in sorted(entries):
            if entry.startswith(b"?? "):
                relative = entry[3:].decode(errors="surrogateescape")
                path = root / relative
                _hash_untracked_path(digest, root, path)
        return {
            "revision": revision,
            "dirty": bool(entries),
            "diff_sha256": digest.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "dirty": None, "diff_sha256": "unavailable"}


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
