"""Purged validation for the fixed production consensus strategy.

The validation contract is fail closed:
* every phase ends early enough for the maximum executable holding to mature;
* one per-date net-alpha statistic drives inference and family correction;
* arbitrary-dependence Bonferroni and moving-block max-stat gates must pass;
* validation executes only the production consensus scorer and its fixed BUY
  policy;
* horizon, rebalance frequency, and top-N are evaluation sensitivities only;
* incomplete or under-resolved statistical-family controls fail closed;
* delayed disclosures, ticker identity, and asset eligibility remain part of the
  production scorer contract.
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
PRIMARY_METRIC = "mean_per_date_net_alpha"
_VALIDATION_GRID_PARAMETERS = frozenset({"horizon", "frequency_days", "top_n"})


def _effective_validation_grid(grid: Mapping[str, object]) -> dict[str, object]:
    """Return the sensitivity family and reject production-policy knobs."""
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("validation grid must not be empty")
    unknown = set(grid) - _VALIDATION_GRID_PARAMETERS
    if unknown:
        raise ValueError(
            f"validation grid has unsupported parameter(s): {sorted(unknown)}"
        )
    effective = {str(name): values for name, values in grid.items()}
    effective.setdefault("frequency_days", (30,))
    effective.setdefault("top_n", (5,))
    return effective


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
    if (
        int(params.lookback_days) != CONSENSUS_LOOKBACK_DAYS
        or int(params.min_buyers) != CONSENSUS_MIN_BUYERS
    ):
        raise ValueError("validation requires lookback_days=28 and min_buyers=3")
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
                required_policy_columns = {
                    "scorer_provenance",
                    "signal_score",
                    "num_buyers",
                    "instrument_type",
                }
                missing_policy_columns = required_policy_columns - set(
                    recommendations.columns
                )
                if missing_policy_columns:
                    raise AnalysisError(
                        "consensus recommendations lack production scorer fields: "
                        f"{sorted(missing_policy_columns)}"
                    )
                provenance = set(recommendations["scorer_provenance"].dropna())
                if provenance != {CONSENSUS_SCORER_PROVENANCE}:
                    raise AnalysisError(
                        "consensus recommendations lack executed-scorer provenance"
                    )
                scores = pd.to_numeric(
                    recommendations["signal_score"], errors="coerce"
                )
                buyers = pd.to_numeric(
                    recommendations["num_buyers"], errors="coerce"
                )
                if (
                    scores.isna().any()
                    or buyers.isna().any()
                    or not np.isfinite(scores).all()
                    or not np.isfinite(buyers).all()
                    or not np.equal(buyers, np.floor(buyers)).all()
                    or buyers.lt(CONSENSUS_MIN_BUYERS).any()
                    or not np.array_equal(scores.to_numpy(), buyers.to_numpy())
                ):
                    raise AnalysisError(
                        "consensus score must equal the distinct-buyer count"
                    )
                instruments = recommendations["instrument_type"].astype(str).str.lower()
                if instruments.ne("stock").any():
                    raise AnalysisError(
                        "consensus recommendations must contain public equities only"
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
            lookback_days=CONSENSUS_LOOKBACK_DAYS,
            min_buyers=CONSENSUS_MIN_BUYERS,
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
        return False, {
            "status": "invalid",
            "reason": "family_metadata_missing",
            "actual_trial_ids": list(actual_ids),
        }

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
    unsupported_parameters = sorted(
        set(parameter_order) - _VALIDATION_GRID_PARAMETERS
    )
    if unsupported_parameters:
        details.update(
            status="invalid",
            reason="production_policy_parameter_in_family",
            unsupported_parameters=unsupported_parameters,
        )
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
    for column, expected in (
        ("lookback_days", CONSENSUS_LOOKBACK_DAYS),
        ("min_buyers", CONSENSUS_MIN_BUYERS),
    ):
        if column in sweep_df.columns:
            values = pd.to_numeric(sweep_df[column], errors="coerce")
            if values.isna().any() or not values.eq(expected).all():
                details.update(
                    status="invalid",
                    reason="production_policy_mismatch",
                    parameter=column,
                    expected=expected,
                )
                return False, details
    if not sweep_df["scorer_provenance"].eq(CONSENSUS_SCORER_PROVENANCE).all():
        details.update(
            status="invalid",
            reason="production_policy_mismatch",
            parameter="scorer_provenance",
            expected=CONSENSUS_SCORER_PROVENANCE,
        )
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
    """Return the family metadata declared by the sweep producer."""
    recorded = sweep_df.attrs.get("family")
    if not isinstance(recorded, dict) or not recorded.get("family_sha256"):
        return {
            "schema_version": None,
            "provenance": "missing",
            "family_provenance": "missing",
            "family_sha256": None,
            "family_hash": None,
            "family_size": len(sweep_df),
            "parameter_order": [],
            "trial_ids": [_canonical_config(value) for value in sweep_df["trial_id"]],
        }

    metadata = dict(recorded)
    metadata.setdefault("family_hash", metadata["family_sha256"])
    metadata.setdefault(
        "family_provenance", metadata.get("provenance", FAMILY_PROVENANCE)
    )
    metadata.setdefault("family_size", len(sweep_df))
    return metadata


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
    """Evaluate statistical support for every declared sensitivity."""
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
    # Net alpha is the statistic; overall return is not a deployment gate.
    statistical_survivor = (
        candidate
        & bootstrap_ready
        & np.isfinite(overall_alpha)
        & (overall_alpha > 0)
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
    descriptive["label"] = "descriptive_best"

    # Preserve every corrected survivor. Validation reports sensitivity support;
    # it never ranks one sensitivity as a live configuration.
    survivor_positions = np.flatnonzero(statistical_survivor)
    statistical_survivors = []
    for position in survivor_positions:
        survivor = working.iloc[int(position)].to_dict()
        survivor["label"] = "statistically_supported_sensitivity"
        statistical_survivors.append(survivor)

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
    elif not statistical_survivors:
        reason = "no_dependence_safe_survivor"
    else:
        reason = None
    sensitivity_results = []
    for position, (_, row) in enumerate(working.iterrows()):
        result = row.to_dict()
        result["statistical_support"] = bool(statistical_survivor[position])
        result["label"] = (
            "statistically_supported_sensitivity"
            if result["statistical_support"]
            else "sensitivity_result"
        )
        sensitivity_results.append(result)
    return {
        "statistical_survivors": statistical_survivors,
        "sensitivity_results": sensitivity_results,
        "descriptive_best": descriptive,
        "failure_reason": reason,
        "primary_metric": PRIMARY_METRIC,
        "n_trials": n_trials,
        "n_min_sample_candidates": int(candidate.sum()),
        "n_statistical_survivors": int(statistical_survivor.sum()),
        "bonferroni_threshold": bonferroni_threshold,
        "alpha": alpha,
        "bootstrap": bootstrap_summary,
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
    """Run purged fixed-policy sensitivity evaluation on both phases."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if n_permutations < 1:
        raise ValueError("bootstrap count must be positive")
    if train_end < train_start or test_end < test_start:
        raise ValueError("validation window end must be on or after its start")
    if test_start <= train_end:
        raise ValueError("test window must start after the training window ends")
    effective_grid = _effective_validation_grid(grid)
    if not effective_grid.get("horizon"):
        raise ValueError("validation grid must include at least one horizon")
    try:
        horizons = [int(value) for value in effective_grid["horizon"]]
    except (TypeError, ValueError) as exc:
        raise ValueError("validation horizons must be integers") from exc
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
            effective_grid,
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
    max_lookback = CONSENSUS_LOOKBACK_DAYS

    train_tx = db.get_transactions_by_date_range(
        pd.Timestamp(train_start) - pd.Timedelta(days=max_lookback),
        pd.Timestamp(train_effective_end),
    )
    train_tickers = sorted(set(_get_consensus_price_tickers(train_tx)) | {"SPY"})
    train_prices = db.get_prices(
        train_tickers,
        next_nyse_session(pd.Timestamp(train_start)),
        pd.Timestamp(train_end),
    )
    train_df = sweep_configs(
        train_tx,
        train_prices,
        effective_grid,
        train_start,
        train_effective_end,
    )
    train_selection = select_config(
        train_df,
        alpha,
        n_permutations=n_permutations,
        permutation_seed=permutation_seed,
    )

    test_tx = db.get_transactions_by_date_range(
        pd.Timestamp(test_start) - pd.Timedelta(days=max_lookback),
        pd.Timestamp(test_effective_end),
    )
    test_tickers = sorted(set(_get_consensus_price_tickers(test_tx)) | {"SPY"})
    test_prices = db.get_prices(
        test_tickers,
        next_nyse_session(pd.Timestamp(test_start)),
        pd.Timestamp(test_end),
    )
    test_df = sweep_configs(
        test_tx, test_prices, effective_grid, test_start, test_effective_end
    )
    test_selection = select_config(
        test_df,
        alpha,
        n_permutations=n_permutations,
        permutation_seed=permutation_seed + 3 * n_permutations,
    )
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
    train_report = _phase_report(train_selection)
    test_report = _phase_report(test_selection)
    supported = bool(
        train_selection["statistical_survivors"]
        or test_selection["statistical_survivors"]
    )
    output = {
        "status": "completed",
        "primary_metric": PRIMARY_METRIC,
        "family": manifest["family"],
        "family_sha256": manifest["family"]["family_sha256"],
        "family_size": manifest["family"]["family_size"],
        "family_provenance": manifest["family"]["family_provenance"],
        "production_policy": _production_policy(),
        "train": train_report,
        "test": test_report,
        "correction": {
            "train": train_report["correction"],
            "test": test_report["correction"],
        },
        "supported_sensitivities": {
            "train": train_report["supported_sensitivities"],
            "test": test_report["supported_sensitivities"],
        },
        "support_status": "supported" if supported else "not_supported",
        "manifest": manifest,
    }

    _print_summary(output)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    return output


def _production_policy() -> dict[str, object]:
    """Return the immutable BUY policy used by every validation trial."""
    return {
        "scorer_provenance": CONSENSUS_SCORER_PROVENANCE,
        "lookback_days": CONSENSUS_LOOKBACK_DAYS,
        "min_buyers": CONSENSUS_MIN_BUYERS,
        "score": "signal_score=distinct_canonical_buyer_count",
        "time_basis": "public_disclosure_time",
        "delayed_filings_actionable": True,
        "ticker_identity": "decision_time",
        "eligible_assets": "public_equities_only",
    }


def _selection_correction(selection: dict) -> dict:
    """Return correction metadata without duplicating sensitivity rows."""
    return _json_safe(
        {
            key: value
            for key, value in selection.items()
            if key not in {"statistical_survivors", "sensitivity_results", "descriptive_best"}
        }
    )


def _phase_report(selection: dict) -> dict:
    """Serialize every sensitivity result and its corrected support subset."""
    failure_reason = selection.get("failure_reason")
    failure_reasons = {
        "invalid_family",
        "partial_family",
        "family_trial_failure",
        "missing_bootstrap_series",
        "incomplete_bootstrap_series",
        "bootstrap_sample_too_small",
        "insufficient_bootstrap_count_or_family_resolution",
    }
    if selection["statistical_survivors"]:
        status = "supported"
    elif failure_reason in failure_reasons:
        status = "failed"
    else:
        status = "not_supported"
    return {
        "status": status,
        "failure_reason": failure_reason,
        "sensitivity_results": _json_safe(selection["sensitivity_results"]),
        "supported_sensitivities": _json_safe(selection["statistical_survivors"]),
        "descriptive_best": _json_safe(selection["descriptive_best"]),
        "correction": _selection_correction(selection),
        "family_size": int(selection["family_size"]),
        "family_sha256": selection["family_sha256"],
        "family_provenance": selection["family_provenance"],
    }


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
            },
        },
        "purge": {
            "rule": "exact_next_nyse_entry_and_fixed_horizon_exit",
            "max_possible_holding_days": max_holding,
            "train_calendar_purge_days": (train_end - train_effective_end).days,
            "test_calendar_purge_days": (test_end - test_effective_end).days,
        },
        "trial_grid": _json_safe(effective_grid),
        "production_policy": _production_policy(),
        "n_trials": family.family_size,
        "family": family_metadata,
        "null": {
            "bootstrap_method": "centered_moving_block_bootstrap_max_stat",
            "n_bootstrap": n_permutations,
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
    logger.info("Support status: %s", output["support_status"])
    for phase in ("train", "test"):
        report = output.get(phase, {})
        if report.get("status") != "supported":
            logger.warning(
                "%s phase has no statistically supported sensitivity: %s",
                phase,
                report.get("failure_reason"),
            )
