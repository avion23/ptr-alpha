"""Purged nested validation for PTR Alpha strategies.

The validation contract is fail closed:
* every phase ends early enough for the maximum executable holding to mature;
* one per-date net-alpha statistic drives inference, correction, selection, and verdict;
* arbitrary-dependence Bonferroni and moving-block max-stat gates must pass;
* validation accepts only the production consensus scorer;
* consensus is identity-invariant and has no member-identity hypothesis;
* incomplete or under-resolved statistical-family controls fail closed;
* the post-2025 final phase is locked and is never loaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyzer import analysis
from analyzer.experiments.family import (
    FAMILY_PROVENANCE,
    build_family,
    trial_spec_sha256,
)
from analyzer.exceptions import AnalysisError
from analyzer.pipeline import BacktestParams
from analyzer.price_repository import next_nyse_session, previous_nyse_session
from analyzer.member_ranking.buyer_scoring import (
    CONSENSUS_LOOKBACK_DAYS,
    CONSENSUS_MIN_BUYERS,
    CONSENSUS_SCORER_PROVENANCE,
    _get_consensus_price_tickers,
)
from analyzer.snooping import bonferroni_correction, max_stat_moving_block_bootstrap

logger = logging.getLogger(__name__)

MIN_DATES_FOR_CANDIDACY = 8
MIN_RECS_FOR_CANDIDACY = 20
MIN_RELEASE_PERMUTATIONS = 999
LOCKED_FINAL_START = date(2026, 1, 1)
PRIMARY_METRIC = "mean_per_date_net_alpha"
_VALIDATION_GRID_PARAMETERS = frozenset(
    {"horizon", "frequency_days", "lookback_days", "min_buyers", "top_n"}
)


def _effective_validation_grid(grid: Mapping[str, object]) -> dict[str, object]:
    """Return the complete consensus strategy family and reject inert knobs."""
    if not grid:
        raise ValueError("validation grid must not be empty")
    unknown = set(grid) - _VALIDATION_GRID_PARAMETERS
    if unknown:
        raise ValueError(
            f"validation grid has unsupported parameter(s): {sorted(unknown)}"
        )
    effective = {str(name): values for name, values in grid.items()}
    effective.setdefault("frequency_days", (30,))
    effective.setdefault("lookback_days", (CONSENSUS_LOOKBACK_DAYS,))
    effective.setdefault("min_buyers", (CONSENSUS_MIN_BUYERS,))
    effective.setdefault("top_n", (5,))
    return effective


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
    lookback_days: int
    min_buyers: int
    top_n: int
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
    status: str = "completed"
    failure_reason: str | None = None
    failure_count: int = 0
    failure_records: tuple[dict[str, str], ...] = ()


def _empty_result(params: BacktestParams) -> SweepResult:
    scheduled = len(
        pd.date_range(
            params.start_date, params.end_date, freq=f"{params.frequency_days}D"
        )
    )
    return SweepResult(
        horizon=params.horizon,
        frequency_days=params.frequency_days,
        lookback_days=params.lookback_days,
        min_buyers=params.min_buyers,
        top_n=params.top_n,
        scheduled_dates=scheduled,
    )


def _benchmark_return(
    prices: pd.DataFrame, as_of: pd.Timestamp, horizon: int
) -> float | None:
    """Return the executable SPY return used by evaluate_backtest on one date."""
    recommendation = pd.DataFrame(
        [{"rank": 1, "ticker": "SPY", "signal_score": 1.0, "instrument_type": "stock"}]
    )
    evaluated = analysis.evaluate_backtest(recommendation, prices, as_of, horizon)
    if not isinstance(evaluated, pd.DataFrame):
        raise TypeError("benchmark evaluation must return a DataFrame")
    if len(evaluated) != 1 or "bt_spy_return_pct" not in evaluated.columns:
        return None
    value = pd.to_numeric(evaluated["bt_spy_return_pct"], errors="coerce").iloc[0]
    if pd.isna(value) or not math.isfinite(float(value)):
        return None
    return float(value)


def _operation_failure(
    stage: str,
    as_of: pd.Timestamp,
    reason: str,
    error: BaseException,
) -> dict[str, str]:
    """Serialize one operation failure for a trial/family report."""
    message = str(error).strip() or type(error).__name__
    return {
        "stage": stage,
        "as_of_date": as_of.date().isoformat(),
        "reason": reason,
        "error_type": type(error).__name__,
        "message": message[:500],
    }


def _backtest_core(
    all_transactions: pd.DataFrame,
    prices: pd.DataFrame,
    params: BacktestParams,
) -> tuple[SweepResult, pd.Series]:
    """Run one configuration and return its summary and primary alpha series.

    The support is the scheduled rebalance calendar for which the identical SPY
    benchmark is executable. A date with no executable strategy trade earns a
    zero cash return; it is not silently dropped. The declared horizon is the
    actual holding used for both strategy and benchmark.
    """
    empty = _empty_result(params)
    as_of_dates = pd.date_range(
        params.start_date, params.end_date, freq=f"{params.frequency_days}D"
    )
    date_rows: list[dict] = []
    evaluated_rows: list[pd.DataFrame] = []
    total_recommendations = 0
    failures: list[dict[str, str]] = []

    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)
        try:
            benchmark_return = _benchmark_return(prices, as_of_ts, params.horizon)
        except Exception as exc:  # fail closed: benchmark work is executable work
            failures.append(
                _operation_failure("benchmark", as_of_ts, "benchmark_exception", exc)
            )
            continue
        if benchmark_return is None:
            failures.append(
                _operation_failure(
                    "benchmark",
                    as_of_ts,
                    "benchmark_return_unavailable",
                    ValueError("benchmark return is unavailable"),
                )
            )
            continue
        try:
            recommendations = analysis.backtest_recommendations(
                all_transactions,
                as_of_date=as_of_ts,
                lookback_days=params.lookback_days,
                min_buyers=params.min_buyers,
                top_n=params.top_n,
            )
            if not isinstance(recommendations, pd.DataFrame):
                raise TypeError("recommendations must be returned as a DataFrame")
            if not recommendations.empty:
                provenance = set(
                    recommendations.get(
                        "scorer_provenance", pd.Series(dtype=str)
                    ).dropna()
                )
                if provenance != {CONSENSUS_SCORER_PROVENANCE}:
                    raise AnalysisError(
                        "consensus recommendations lack executed-scorer provenance"
                    )
        except Exception as exc:  # fail closed: never convert an exception to cash
            failures.append(
                _operation_failure(
                    "recommendation", as_of_ts, "recommendation_exception", exc
                )
            )
            continue

        strategy_return = 0.0
        traded = False
        if not recommendations.empty:
            try:
                evaluated = analysis.evaluate_backtest(
                    recommendations, prices, as_of_ts, params.horizon
                )
                if not isinstance(evaluated, pd.DataFrame):
                    raise TypeError("evaluation must return a DataFrame")
                if evaluated.empty:
                    raise ValueError(
                        "evaluation returned no rows for non-empty recommendations"
                    )
                if len(evaluated) != len(recommendations):
                    raise ValueError(
                        "evaluation row count does not match recommendations"
                    )
                if "bt_return_pct" not in evaluated.columns:
                    raise KeyError("evaluation is missing bt_return_pct")
                returns = pd.to_numeric(evaluated["bt_return_pct"], errors="coerce")
                if returns.isna().any() or not np.isfinite(returns).all():
                    raise ValueError("evaluation contains unavailable returns")
                strategy_return = float(returns.mean())
                total_recommendations += len(returns)
                valid = evaluated.copy()
                valid.insert(0, "as_of_date", as_of_ts.date())
                evaluated_rows.append(valid)
                traded = True
            except Exception as exc:  # fail closed: no partial evaluation as cash
                failures.append(
                    _operation_failure(
                        "evaluation", as_of_ts, "evaluation_exception", exc
                    )
                )
                continue

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
            "Validation trial failed for %d recommendation/evaluation operation(s)",
            len(failures),
        )
    if not date_rows:
        return (
            replace(
                empty,
                status="failed" if failures else "completed",
                failure_reason=failures[0]["reason"] if failures else None,
                failure_count=len(failures),
                failure_records=tuple(failures),
            ),
            pd.Series(dtype=float),
        )

    by_date = pd.DataFrame(date_rows).set_index("as_of_date").sort_index()
    per_date = by_date["net_alpha_pct"].astype(float)
    combined = (
        pd.concat(evaluated_rows, ignore_index=True)
        if evaluated_rows
        else pd.DataFrame(columns=["rank", "bt_alpha_pct"])
    )
    if "bt_alpha_pct" not in combined.columns:
        combined["bt_alpha_pct"] = np.nan
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
        lookback_days=params.lookback_days,
        min_buyers=params.min_buyers,
        top_n=params.top_n,
        scorer_provenance=(
            CONSENSUS_SCORER_PROVENANCE if total_recommendations > 0 else ""
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
        status="failed" if failures else "completed",
        failure_reason=failures[0]["reason"] if failures else None,
        failure_count=len(failures),
        failure_records=tuple(failures),
    )
    return result, per_date


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


def sweep_configs(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    grid: dict,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Evaluate every consensus configuration on one already-purged phase."""
    if end < start:
        raise ValueError("purged sweep phase has no executable dates")
    family = build_family(_effective_validation_grid(grid))

    rows: list[dict] = []
    series_by_trial: dict[int, pd.Series] = {}
    for trial in family.trials:
        trial_id = trial.trial_id
        values = dict(trial.config)
        horizon = int(values["horizon"])
        frequency = int(values["frequency_days"])
        lag = max(0, math.ceil(horizon / frequency) - 1)
        params = BacktestParams(
            start_date=start,
            end_date=end,
            horizon=horizon,
            lookback_days=int(values["lookback_days"]),
            min_buyers=int(values["min_buyers"]),
            top_n=int(values["top_n"]),
            frequency_days=frequency,
        )
        try:
            result, per_date = _backtest_core(all_tx, prices, params)
        except Exception as exc:  # fail closed: preserve a failed trial row
            failure = _operation_failure(
                "trial", pd.Timestamp(start), "trial_exception", exc
            )
            result = replace(
                _empty_result(params),
                status="failed",
                failure_reason="trial_exception",
                failure_count=1,
                failure_records=(failure,),
            )
            per_date = pd.Series(dtype=float)
        statistic = newey_west_tstat(per_date, lag)
        p_value = (
            float(stats.norm.sf(statistic))
            if math.isfinite(statistic)
            else (0.0 if statistic > 0 else 1.0)
        )
        row = asdict(result)
        row["failure_records"] = _json_safe(result.failure_records)
        row["trial_id"] = trial_id
        row.update(_json_safe(values))
        row["trial_config"] = _json_safe(values)
        row["trial_status"] = result.status
        row["trial_failed"] = result.status == "failed"
        row["trial_spec_sha256"] = trial.trial_sha256
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
    frame.attrs["family"] = family.metadata()
    frame.attrs["family_sha256"] = family.family_sha256
    frame.attrs["family_size"] = family.family_size
    frame.attrs["family_provenance"] = family.provenance
    return frame


def _member_family_sha256(
    sweep_df: pd.DataFrame, series_by_trial: dict[int, pd.Series]
) -> str:
    digest = hashlib.sha256()
    ordered = sweep_df.sort_values("trial_id").copy()
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    for column in ordered.columns:
        ordered[column] = ordered[column].map(
            lambda value: (
                json.dumps(_json_safe(value), sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        )
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    for trial_id in sorted(series_by_trial):
        digest.update(str(trial_id).encode())
        series = pd.Series(series_by_trial[trial_id], dtype=float).sort_index()
        digest.update(
            pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes()
        )
    return digest.hexdigest()


def _family_sha256_for_sweep(
    sweep_df: pd.DataFrame, series_by_trial: dict[int, pd.Series]
) -> str:
    recorded = sweep_df.attrs.get("family")
    if isinstance(recorded, dict) and recorded.get("family_sha256"):
        return str(recorded["family_sha256"])
    return _member_family_sha256(sweep_df, series_by_trial)


def _strict_trial_id(value) -> int | None:
    """Return only a genuine integer trial id, never a lossy coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _canonical_config(value):
    """Detach nested row values into the same JSON shape as family values."""
    if isinstance(value, Mapping):
        return {str(key): _canonical_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_config(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_config(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, date, datetime)):
        return str(value)
    return value


def _config_identity(value) -> str:
    return _sha256_json(_canonical_config(value))


def _family_integrity(sweep_df: pd.DataFrame) -> tuple[bool, dict]:
    """Check that an executed frame still represents its declared family."""
    raw_ids = sweep_df["trial_id"].tolist()
    parsed_ids = [_strict_trial_id(value) for value in raw_ids]
    if any(value is None for value in parsed_ids):
        return False, {
            "status": "invalid",
            "reason": "trial_ids_are_not_integers",
            "actual_trial_ids": [str(value) for value in raw_ids],
        }
    actual_ids = tuple(parsed_ids)

    recorded = sweep_df.attrs.get("family")
    if not isinstance(recorded, dict) or not recorded.get("family_sha256"):
        expected_ids = tuple(range(len(actual_ids)))
        if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(
            expected_ids
        ):
            return False, {
                "status": "partial",
                "reason": "trial_ids_are_not_contiguous",
                "expected_trial_ids": list(expected_ids),
                "actual_trial_ids": list(actual_ids),
                "missing_trial_ids": sorted(set(expected_ids) - set(actual_ids)),
                "unexpected_trial_ids": sorted(set(actual_ids) - set(expected_ids)),
            }
        return True, {"status": "complete", "provenance": "legacy_frame"}

    try:
        raw_family_size = recorded["family_size"]
        family_size = _strict_trial_id(raw_family_size)
        raw_declared_ids = recorded["trial_ids"]
        declared_ids = tuple(_strict_trial_id(value) for value in raw_declared_ids)
        family_hash = recorded["family_sha256"]
    except (KeyError, TypeError, ValueError):
        return False, {"status": "invalid", "reason": "malformed_family_metadata"}
    if (
        family_size is None
        or any(value is None for value in declared_ids)
        or not isinstance(family_hash, str)
        or len(family_hash) != 64
        or any(character not in "0123456789abcdef" for character in family_hash)
    ):
        return False, {"status": "invalid", "reason": "malformed_family_metadata"}

    expected_ids = tuple(range(family_size))
    details = {
        "status": "complete",
        "family_sha256": family_hash,
        "family_size": family_size,
        "expected_trial_ids": list(expected_ids),
        "actual_trial_ids": list(actual_ids),
    }
    if (
        family_size < 1
        or declared_ids != expected_ids
        or len(actual_ids) != family_size
        or len(set(actual_ids)) != len(actual_ids)
        or set(actual_ids) != set(expected_ids)
    ):
        details.update(
            status="partial",
            reason="declared_family_and_executed_trials_differ",
            missing_trial_ids=sorted(set(expected_ids) - set(actual_ids)),
            unexpected_trial_ids=sorted(set(actual_ids) - set(expected_ids)),
        )
        return False, details

    ordered_grid = recorded.get("ordered_grid")
    parameter_order = recorded.get("parameter_order")
    if not isinstance(ordered_grid, list) or not isinstance(parameter_order, list):
        details.update(status="invalid", reason="family_grid_metadata_missing")
        return False, details
    try:
        rebuilt_grid = {
            str(item["parameter"]): item["values"] for item in ordered_grid
        }
        rebuilt = build_family(rebuilt_grid, parameter_order=parameter_order)
    except (KeyError, TypeError, ValueError):
        details.update(status="invalid", reason="family_grid_metadata_invalid")
        return False, details
    if rebuilt.family_sha256 != family_hash or rebuilt.family_size != family_size:
        details.update(status="invalid", reason="family_hash_metadata_mismatch")
        return False, details
    declared_specs = recorded.get("trial_spec_sha256")
    expected_specs = {trial.trial_id: trial.trial_sha256 for trial in rebuilt.trials}
    if not isinstance(declared_specs, list) or declared_specs != [
        trial.trial_sha256 for trial in rebuilt.trials
    ]:
        details.update(status="invalid", reason="trial_spec_metadata_mismatch")
        return False, details
    required_columns = {"trial_spec_sha256", "trial_config", *parameter_order}
    missing_columns = sorted(required_columns - set(sweep_df.columns))
    if missing_columns:
        details.update(
            status="invalid",
            reason="row_configuration_columns_missing",
            missing_columns=missing_columns,
        )
        return False, details
    for position, row in sweep_df.iterrows():
        trial_id = _strict_trial_id(row["trial_id"])
        expected_trial = rebuilt.trials[trial_id]
        expected_config = dict(expected_trial.config)
        actual_config = row["trial_config"]
        if isinstance(actual_config, str):
            try:
                actual_config = json.loads(actual_config)
            except json.JSONDecodeError:
                actual_config = None
        if not isinstance(actual_config, Mapping):
            details.update(
                status="invalid",
                reason="trial_config_missing_or_invalid",
                trial_id=trial_id,
            )
            return False, details
        if _config_identity(actual_config) != _config_identity(expected_config):
            details.update(
                status="invalid",
                reason="trial_config_mismatch",
                trial_id=trial_id,
            )
            return False, details
        for parameter in parameter_order:
            if _config_identity(row[parameter]) != _config_identity(
                expected_config[parameter]
            ):
                details.update(
                    status="invalid",
                    reason="row_configuration_mismatch",
                    trial_id=trial_id,
                    parameter=parameter,
                )
                return False, details
        actual_spec = row["trial_spec_sha256"]
        expected_spec = expected_specs[trial_id]
        actual_spec_from_config = trial_spec_sha256(
            family_hash,
            trial_id,
            actual_config,
            parameter_order,
        )
        if actual_spec != expected_spec or actual_spec != actual_spec_from_config:
            details.update(
                status="invalid",
                reason="trial_spec_hash_mismatch",
                trial_id=trial_id,
            )
            return False, details
    return True, details


def _family_metadata_for_sweep(sweep_df: pd.DataFrame) -> dict:
    """Return canonical family metadata, including a legacy-frame fallback."""
    recorded = sweep_df.attrs.get("family")
    if isinstance(recorded, dict) and recorded.get("family_sha256"):
        metadata = dict(recorded)
        metadata.setdefault("family_hash", metadata["family_sha256"])
        metadata.setdefault(
            "family_provenance", metadata.get("provenance", FAMILY_PROVENANCE)
        )
        metadata.setdefault("family_size", len(sweep_df))
        return metadata

    configuration_columns = [
        column
        for column in (
            "horizon",
            "frequency_days",
            "lookback_days",
            "min_buyers",
            "top_n",
        )
        if column in sweep_df.columns
    ]
    parsed_ids = [_strict_trial_id(value) for value in sweep_df["trial_id"]]
    if all(value is not None for value in parsed_ids):
        ordered = sweep_df.assign(_validated_trial_id=parsed_ids).sort_values(
            "_validated_trial_id", kind="mergesort"
        )
    else:
        ordered = sweep_df
    payload = {
        "provenance": "legacy_sweep_frame_identity_v1",
        "parameter_order": configuration_columns,
        "trials": [
            {
                "trial_id": _canonical_config(row["trial_id"]),
                "config": {
                    column: _canonical_config(row[column])
                    for column in configuration_columns
                },
            }
            for _, row in ordered.iterrows()
        ],
    }
    digest = _sha256_json(payload)
    return {
        "schema_version": 1,
        "provenance": payload["provenance"],
        "family_provenance": payload["provenance"],
        "family_sha256": digest,
        "family_hash": digest,
        "family_size": len(sweep_df),
        "parameter_order": configuration_columns,
        "trial_ids": [_canonical_config(value) for value in ordered["trial_id"]],
    }


def _trial_failure_mask(sweep_df: pd.DataFrame) -> np.ndarray:
    failed = np.zeros(len(sweep_df), dtype=bool)
    if "trial_failed" in sweep_df.columns:
        failed |= sweep_df["trial_failed"].fillna(False).astype(bool).to_numpy()
    if "trial_status" in sweep_df.columns:
        failed |= sweep_df["trial_status"].astype(str).ne("completed").to_numpy()
    elif "status" in sweep_df.columns:
        failed |= sweep_df["status"].astype(str).ne("completed").to_numpy()
    if "failure_count" in sweep_df.columns:
        failed |= (
            pd.to_numeric(sweep_df["failure_count"], errors="coerce")
            .fillna(0)
            .gt(0)
            .to_numpy()
        )
    return failed


def _family_failure_records(
    sweep_df: pd.DataFrame, failed_positions: np.ndarray
) -> list[dict]:
    records: list[dict] = []
    for position in failed_positions:
        row = sweep_df.iloc[int(position)]
        encoded = row.get("failure_records", "[]")
        if isinstance(encoded, str):
            try:
                details = json.loads(encoded)
            except json.JSONDecodeError:
                details = [{"message": encoded}]
        elif isinstance(encoded, list):
            details = encoded
        else:
            details = []
        records.append(
            {
                "trial_id": _strict_trial_id(row["trial_id"]),
                "status": str(row.get("trial_status", "failed")),
                "failure_reason": str(row.get("failure_reason", "trial_failed")),
                "failure_count": int(row.get("failure_count", len(details))),
                "failures": _json_safe(details),
            }
        )
    return records


def select_config(
    sweep_df: pd.DataFrame,
    alpha: float = 0.05,
    *,
    series_by_trial: dict[int, pd.Series] | None = None,
    n_permutations: int = 999,
    permutation_seed: int = 0,
) -> dict:
    """Select a consensus configuration by statistical-family gates."""
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
        "scorer_provenance",
    }
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"sweep_df missing required columns: {sorted(missing)}")
    working = sweep_df.copy()
    n_trials = len(working)
    family = _family_metadata_for_sweep(working)
    family_complete, family_integrity = _family_integrity(working)
    failed_trial_mask = _trial_failure_mask(working)
    failed_positions = np.flatnonzero(failed_trial_mask)
    family_failed = bool(len(failed_positions))
    family_failures = _family_failure_records(working, failed_positions)
    bonferroni_threshold = bonferroni_correction(n_trials, alpha)
    candidate = (
        working["min_sample_ok"].fillna(False).astype(bool).to_numpy(copy=True)
        if "min_sample_ok" in working.columns
        else np.ones(n_trials, dtype=bool)
    )
    candidate &= (
        working["scorer_provenance"]
        .astype(str)
        .eq(CONSENSUS_SCORER_PROVENANCE)
        .to_numpy()
    )
    candidate &= ~failed_trial_mask
    candidate &= family_complete
    source_series = series_by_trial or sweep_df.attrs.get("series_by_trial")
    expected_trial_ids = {
        value
        for value in (_strict_trial_id(item) for item in working["trial_id"])
        if value is not None
    }
    supplied_trial_ids = (
        {
            value
            for value in (_strict_trial_id(key) for key in source_series)
            if value is not None
        }
        if source_series
        else set()
    )
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
        "family_sha256": family["family_sha256"],
        "family_size": int(family["family_size"]),
        "family_provenance": family["family_provenance"],
    }
    if source_series and complete_series and family_complete and not family_failed:
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
        and family_complete
        and not family_failed
    )
    working["bootstrap_p_value"] = bootstrap_p
    working["bonferroni_p_value"] = np.minimum(bootstrap_p * n_trials, 1.0)
    working["max_stat_p_value"] = max_stat_p
    overall_alpha = pd.to_numeric(working["overall_alpha"], errors="coerce").to_numpy(
        dtype=float
    )
    overall_return = pd.to_numeric(working["overall_return"], errors="coerce").to_numpy(
        dtype=float
    )
    finite_metrics = np.isfinite(overall_alpha) & np.isfinite(overall_return)
    statistical_survivor = (
        candidate
        & bootstrap_ready
        & finite_metrics
        & (overall_alpha > 0)
        & (overall_return > 0)
        & (bootstrap_p <= bonferroni_threshold)
        & (max_stat_p <= alpha)
    )
    descriptive_positions = np.flatnonzero(~failed_trial_mask)
    if not len(descriptive_positions):
        descriptive_positions = np.arange(n_trials)
    descriptive_values = pd.to_numeric(
        working.iloc[descriptive_positions]["overall_alpha"], errors="coerce"
    ).to_numpy(dtype=float)
    descriptive_values = np.where(
        np.isfinite(descriptive_values), descriptive_values, -math.inf
    )
    if len(descriptive_positions):
        descriptive_order = pd.DataFrame(
            {
                "position": descriptive_positions,
                "alpha": descriptive_values,
                "trial_id": [
                    _strict_trial_id(value)
                    for value in working.iloc[descriptive_positions]["trial_id"]
                ],
            }
        )
        descriptive_order["trial_id"] = descriptive_order["trial_id"].fillna(
            np.iinfo(np.int64).max
        )
        descriptive_position = int(
            descriptive_order.sort_values(
                ["alpha", "trial_id"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0]["position"]
        )
    else:
        descriptive_position = 0
    descriptive = working.iloc[descriptive_position].to_dict()
    descriptive["label"] = "descriptive_only_not_deployable"

    survivor_positions = np.flatnonzero(statistical_survivor)
    statistical_candidate = None
    if len(survivor_positions):
        order = working.iloc[survivor_positions].copy()
        order["_alpha_order"] = pd.to_numeric(
            order["overall_alpha"], errors="coerce"
        )
        order["_tstat_order"] = pd.to_numeric(order["nw_tstat"], errors="coerce").fillna(
            -math.inf
        )
        order["_trial_id_order"] = [
            _strict_trial_id(value) for value in order["trial_id"]
        ]
        order["_trial_id_order"] = order["_trial_id_order"].fillna(
            np.iinfo(np.int64).max
        )
        order = order.sort_values(
            ["_alpha_order", "_tstat_order", "_trial_id_order"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        statistical_candidate = order.iloc[0].to_dict()
        for key in ("_alpha_order", "_tstat_order", "_trial_id_order"):
            statistical_candidate.pop(key, None)
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

    if not family_complete:
        reason = (
            "invalid_family"
            if family_integrity.get("status") == "invalid"
            else "partial_family"
        )
    elif family_failed:
        reason = "family_trial_failure"
    elif not source_series:
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
        "family": family,
        "family_sha256": family["family_sha256"],
        "family_size": int(family["family_size"]),
        "family_provenance": family["family_provenance"],
        "family_integrity": family_integrity,
        "family_failure": {
            "status": (
                family_integrity.get("status")
                if not family_complete
                else "failed"
                if family_failed
                else "none"
            ),
            "failed_trial_count": len(family_failures),
            "trials": family_failures,
        },
    }


def _run_identity_invariant_control(
    sweep_df: pd.DataFrame,
    observed_trial_id: int,
) -> MemberIdentityControlResult:
    """Describe the identity-invariant consensus contract for one trial."""
    family_complete, family_details = _family_integrity(sweep_df)
    if not family_complete:
        raise ValueError(
            "identity-invariant control requires a complete family: "
            f"{family_details.get('reason', 'invalid family')}"
        )
    if _trial_failure_mask(sweep_df).any():
        raise ValueError("identity-invariant control requires completed trials")
    series_by_trial = sweep_df.attrs.get("series_by_trial")
    expected_ids = {int(value) for value in sweep_df["trial_id"]}
    if not isinstance(series_by_trial, dict) or set(series_by_trial) != expected_ids:
        raise ValueError("identity-invariant family lacks complete trial series")
    selected = sweep_df[sweep_df["trial_id"] == observed_trial_id]
    if len(selected) != 1:
        raise ValueError("observed trial_id is not unique in consensus family")
    row = selected.iloc[0]
    if str(row["scorer_provenance"]) != CONSENSUS_SCORER_PROVENANCE:
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
        family_sha256=_family_sha256_for_sweep(sweep_df, series_by_trial),
        observed_trial_id=observed_trial_id,
        observed_statistic=float(row["nw_tstat"]),
    )
    return result


def _phase_end(boundary_end: date, max_holding_days: int) -> date:
    """Return the latest as-of whose exact execution window matures by boundary."""
    boundary = pd.Timestamp(boundary_end).normalize()
    candidate = boundary - pd.Timedelta(days=max_holding_days)
    while True:
        entry = next_nyse_session(candidate)
        exit_date = previous_nyse_session(
            entry + pd.Timedelta(days=max_holding_days)
        )
        if exit_date <= boundary:
            return candidate.date()
        candidate -= pd.Timedelta(days=1)


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
    train_effective_end = _phase_end(train_end, max_holding)
    test_effective_end = _phase_end(test_end, max_holding)
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
    effective_grid = _effective_validation_grid(grid)
    max_lookback = max(int(value) for value in effective_grid["lookback_days"])
    tx_start = pd.Timestamp(train_start) - pd.Timedelta(days=max_lookback)
    train_tx_end = pd.Timestamp(train_effective_end)
    train_price_end = pd.Timestamp(train_end)
    train_tx = db.get_transactions_by_date_range(tx_start, train_tx_end)
    train_tickers = sorted(set(_get_consensus_price_tickers(train_tx)) | {"SPY"})
    train_price_start = next_nyse_session(pd.Timestamp(train_start))
    train_prices = db.get_prices(
        train_tickers, train_price_start, train_price_end
    )
    train_df = sweep_configs(
        train_tx,
        train_prices,
        effective_grid,
        train_start,
        train_effective_end,
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
        )
        selection["member_identity_control"] = asdict(identity_diagnostic)
    manifest = _build_manifest(
        train_tx,
        train_prices,
        effective_grid,
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
        "family": manifest["family"],
        "family_sha256": manifest["family"]["family_sha256"],
        "family_size": manifest["family"]["family_size"],
        "family_provenance": manifest["family"]["family_provenance"],
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
        train_result, _ = _run_frozen(
            train_tx,
            train_prices,
            config,
            train_start,
            train_effective_end,
        )
        if (
            train_result.status != "completed"
            or train_result.failure_count
            or train_result.failure_records
        ):
            output["correction"]["failure_reason"] = "selected_train_trial_failure"
            output["correction"]["selected_train_trial_failure"] = _json_safe(
                {
                    "failure_reason": train_result.failure_reason,
                    "failure_count": train_result.failure_count,
                    "failure_records": train_result.failure_records,
                }
            )
            output["train"] = _window_metrics(
                train_result,
                float(selected["nw_tstat"]),
                float(selected["bootstrap_p_value"]),
                "failed_train_trial",
            )
            _print_summary(output)
            if out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(output, indent=2, sort_keys=True, default=str)
                )
            return output
        tx_end = pd.Timestamp(test_effective_end)
        test_tx_start = pd.Timestamp(test_start) - pd.Timedelta(
            days=int(config.get("lookback_days", CONSENSUS_LOOKBACK_DAYS))
        )
        price_start = next_nyse_session(pd.Timestamp(test_start))
        price_end = pd.Timestamp(test_end)
        all_tx = db.get_transactions_by_date_range(test_tx_start, tx_end)
        tickers = sorted(set(_get_consensus_price_tickers(all_tx)) | {"SPY"})
        prices = db.get_prices(tickers, price_start, price_end)
        test_result, test_series = _run_frozen(
            all_tx, prices, config, test_start, test_effective_end
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
        test_completed_without_failures = bool(
            test_result.status == "completed"
            and test_result.failure_count == 0
            and not test_result.failure_records
        )
        test_passes = bool(
            test_completed_without_failures
            and test_bootstrap_error is None
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
        )
        if test_bootstrap_error is not None:
            output["test"]["bootstrap_error"] = test_bootstrap_error

    _print_summary(output)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    return output


def _run_frozen(all_tx, prices, config, start: date, end: date):
    params = BacktestParams(
        start_date=start,
        end_date=end,
        horizon=int(config["horizon"]),
        lookback_days=int(config["lookback_days"]),
        min_buyers=int(config["min_buyers"]),
        top_n=int(config["top_n"]),
        frequency_days=int(config["frequency_days"]),
    )
    return _backtest_core(all_tx, prices, params)


def _config_from_row(row: dict) -> dict:
    keys = ["horizon", "frequency_days", "lookback_days", "min_buyers", "top_n"]
    return {key: row[key] for key in keys}


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
        "result_status": result.status,
        "failure_reason": result.failure_reason,
        "failure_count": result.failure_count,
        "failure_records": _json_safe(result.failure_records),
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


def _build_manifest(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
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
    effective_grid = _effective_validation_grid(grid)
    family = build_family(effective_grid)
    family_metadata = family.metadata()
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
            },
        },
        "purge": {
            "rule": "exact_next_nyse_entry_and_fixed_horizon_exit",
            "max_possible_holding_days": max_holding,
            "train_calendar_purge_days": (train_end - train_effective_end).days,
            "test_calendar_purge_days": (test_end - test_effective_end).days,
        },
        "trial_grid": _json_safe(effective_grid),
        "n_trials": family.family_size,
        "family": family_metadata,
        "null": {
            "bootstrap_method": "centered_moving_block_bootstrap_max_stat",
            "n_bootstrap": n_permutations,
            "member_identity_policy": (
                "consensus_is_identity_invariant_no_member_identity_hypothesis"
            ),
            "minimum_release_count": MIN_RELEASE_PERMUTATIONS,
            "minimum_family_resolution_bootstrap": max(
                MIN_RELEASE_PERMUTATIONS,
                math.ceil(family.family_size / alpha) - 1,
            ),
            "seed": permutation_seed,
            "assumptions": [
                "local stationarity within moving blocks",
                "shared block-start uniforms preserve aligned cross-config dependence",
                "Bonferroni is the arbitrary-dependence controlling gate",
            ],
        },
        "coverage_input": {
            "transactions": len(all_tx),
            "price_rows": len(prices),
            "price_columns": len(prices.columns),
            "price_start": str(prices.index.min()) if not prices.empty else None,
            "price_end": str(prices.index.max()) if not prices.empty else None,
        },
    }


def _value_snapshot_hash(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(json.dumps([str(value) for value in frame.columns]).encode())
        digest.update(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        )
    return digest.hexdigest()


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
