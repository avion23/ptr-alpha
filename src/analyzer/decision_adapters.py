"""Adapters between legacy backtest DataFrames and decision contracts.

The adapters deliberately keep the forecast/policy boundary one-way:
recommendation ``signal_score`` values become ``Forecast.ranking_score`` and
never ``Forecast.expected_net_alpha``.  Realized ``bt_*`` columns are accepted
only at the evidence boundary and are never copied into a forecast or policy
object.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from typing import Any, cast

import numpy as np
import pandas as pd

from analyzer.backtest.evaluate import evaluate_backtest
from analyzer.backtest.recommend import backtest_recommendations
from analyzer.decision_contracts import (
    DateLike,
    Evidence,
    EvidenceReport,
    Forecast,
    MarketSnapshot,
    MetadataValue,
    PortfolioState,
    Provenance,
    TargetPortfolio,
    TargetPosition,
)

_REALIZED_PREFIX = "bt_"
_DECISION_INDEX = "_decision_index"
_DECISION_EVENT_ID = "_decision_event_id"

# These fields are model inputs or diagnostics, not realized outcomes.  In
# particular, signal_score is represented only by Forecast.ranking_score.
_FORECAST_METADATA_COLUMNS = (
    "scoring_mode",
    "num_buyers",
    "rated_buyers",
    "base_signal_score",
    "lag_days",
    "lag_weight",
    "crash_prob",
    "crash_var_95",
    "volatility_20d",
    "drawdown_from_ath",
    "ou_entry_value",
    "rank",
)


def _assert_no_realized_columns(frame: pd.DataFrame) -> None:
    """Reject evaluator output accidentally fed back as a decision frame."""
    realized = [
        str(column)
        for column in frame.columns
        if str(column).startswith(_REALIZED_PREFIX)
    ]
    if realized:
        raise ValueError(
            "realized bt_* columns cannot cross the forecast/policy boundary: "
            + ", ".join(sorted(realized))
        )


def _python_scalar(value: object) -> object:
    """Convert pandas/numpy scalar wrappers without retaining mutable data."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, pd.Timedelta):
        return value.to_pytimedelta()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    return value


def _text(value: object) -> str | None:
    value = _python_scalar(value)
    return None if value is None else str(value)


def _date_like(value: object) -> date | datetime | None:
    value = _python_scalar(value)
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value
    try:
        converted = pd.Timestamp(cast(Any, value)).to_pydatetime()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid date-like value: {value!r}") from exc
    return converted if isinstance(converted, (date, datetime)) else None


def _number(value: object) -> float | None:
    """Read an optional finite numeric cell, failing closed on bad values."""
    value = _python_scalar(value)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"expected a numeric value, got {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid numeric value: {value!r}") from exc
    if not np.isfinite(converted):
        raise ValueError(f"numeric value must be finite: {value!r}")
    return converted


def _positive_int(value: object, field_name: str) -> int | None:
    number = _number(value)
    if number is None:
        return None
    try:
        converted = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if number != converted or converted <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return converted


def _resolve_alias(
    primary: DateLike | None,
    alias: DateLike | None,
    primary_name: str,
    alias_name: str,
) -> DateLike | None:
    if primary is not None and alias is not None:
        left = _date_like(primary)
        right = _date_like(alias)
        if left != right:
            raise ValueError(f"{primary_name} and {alias_name} disagree")
    return primary if primary is not None else alias


def _resolve_horizon(
    primary: int | None,
    alias: int | None,
) -> int | None:
    resolved_primary = _positive_int(primary, "horizon")
    resolved_alias = _positive_int(alias, "horizon_days")
    if resolved_primary is not None and resolved_alias is not None:
        if resolved_primary != resolved_alias:
            raise ValueError("horizon and horizon_days disagree")
    return resolved_primary if resolved_primary is not None else resolved_alias


def _assert_common_forecast_context(forecasts: Sequence[Forecast]) -> None:
    """Require one point-in-time context for a policy/evidence batch."""
    values = tuple(forecasts)
    if not values:
        return
    first = values[0]
    if any(
        forecast.as_of != first.as_of
        or forecast.horizon_days != first.horizon_days
        for forecast in values[1:]
    ):
        raise ValueError("forecasts must share as_of and horizon_days")


def _provenance_from_row(row: Mapping[str, object]) -> Provenance:
    return Provenance(
        source=_text(row.get("source")),
        source_record_id=_text(row.get("source_record_id")),
        source_row_id=_text(row.get("source_row_id")),
        ticker_origin=_text(row.get("ticker_origin")),
        available_date=_date_like(row.get("available_date")),
        notification_date=_date_like(row.get("notification_date")),
    )


def _metadata_from_row(row: Mapping[str, object]) -> tuple[tuple[str, MetadataValue], ...]:
    metadata: list[tuple[str, MetadataValue]] = []
    for column in _FORECAST_METADATA_COLUMNS:
        value = _python_scalar(row.get(column))
        if value is not None and isinstance(
            value, (str, int, float, bool, date, datetime)
        ):
            metadata.append((column, cast(MetadataValue, value)))
    return tuple(metadata)


def _event_id_from_row(
    row: Mapping[str, object], ticker: str, as_of: DateLike, row_index: int
) -> str:
    explicit = _text(row.get("event_id"))
    if explicit:
        return explicit
    provenance = _provenance_from_row(row)
    if provenance.source_row_id:
        return provenance.source_row_id
    if provenance.source_record_id:
        return provenance.source_record_id
    timestamp = pd.Timestamp(as_of).isoformat()
    return f"{ticker}:{timestamp}:{row_index}"


def _recommendation_to_forecast(
    row: Mapping[str, object],
    row_index: int,
    *,
    as_of_date: DateLike | None,
    horizon: int | None,
    model_id: str,
    feature_snapshot_sha256: str | None,
) -> Forecast:
    ticker = _text(row.get("ticker"))
    if not ticker:
        raise ValueError("recommendations require a non-empty ticker")

    row_as_of = _date_like(row.get("as_of_date"))
    if as_of_date is not None and row_as_of is not None:
        if pd.Timestamp(as_of_date) != pd.Timestamp(row_as_of):
            raise ValueError("as_of_date disagrees with recommendation row")
    resolved_as_of = as_of_date if as_of_date is not None else row_as_of
    if resolved_as_of is None:
        raise ValueError("an as_of_date is required to build a Forecast")

    explicit_horizon = _positive_int(horizon, "horizon")
    row_horizon = _positive_int(row.get("horizon_days"), "horizon_days")
    if (
        explicit_horizon is not None
        and row_horizon is not None
        and explicit_horizon != row_horizon
    ):
        raise ValueError("horizon disagrees with recommendation row")
    effective_horizon = explicit_horizon or row_horizon
    optimal_horizon = _positive_int(row.get("optimal_horizon"), "optimal_horizon")
    if effective_horizon is None:
        effective_horizon = optimal_horizon
    if effective_horizon is None:
        raise ValueError("a horizon is required to build a Forecast")

    row_feature_hash = _text(row.get("feature_snapshot_sha256"))
    instrument_type = _text(row.get("instrument_type")) or "stock"
    event_id = _event_id_from_row(row, ticker, resolved_as_of, row_index)

    # signal_score is an ordering score, not a return estimate.  In
    # particular, do not populate expected_net_alpha or any distributional
    # field from it.
    return Forecast(
        event_id=event_id,
        ticker=ticker,
        as_of=resolved_as_of,
        horizon_days=effective_horizon,
        expected_net_alpha=None,
        net_alpha_std=None,
        probability_positive=None,
        model_id=model_id,
        feature_snapshot_sha256=row_feature_hash or feature_snapshot_sha256,
        provenance=_provenance_from_row(row),
        ranking_score=_number(row.get("signal_score")),
        optimal_horizon_days=optimal_horizon,
        instrument_type=instrument_type,
        amount_midpoint=_number(row.get("amount_midpoint")),
        metadata=_metadata_from_row(row),
    )


def recommendations_to_forecasts(
    recommendations: pd.DataFrame,
    as_of_date: DateLike | None = None,
    horizon: int | None = None,
    *,
    as_of: DateLike | None = None,
    horizon_days: int | None = None,
    model_id: str = "consensus-recommendation",
    feature_snapshot_sha256: str | None = None,
) -> tuple[Forecast, ...]:
    """Translate legacy recommendations into immutable ranking-only forecasts."""
    if not isinstance(recommendations, pd.DataFrame):
        raise TypeError("recommendations must be a pandas DataFrame")
    _assert_no_realized_columns(recommendations)
    resolved_as_of = _resolve_alias(as_of_date, as_of, "as_of_date", "as_of")
    resolved_horizon = _resolve_horizon(horizon, horizon_days)
    forecasts = tuple(
        _recommendation_to_forecast(
            row,
            row_index,
            as_of_date=resolved_as_of,
            horizon=resolved_horizon,
            model_id=model_id,
            feature_snapshot_sha256=feature_snapshot_sha256,
        )
        for row_index, row in enumerate(recommendations.to_dict(orient="records"))
    )
    _assert_common_forecast_context(forecasts)
    return forecasts


# Descriptive aliases make the adapter discoverable without changing the
# legacy recommendation function's name or return type.
adapt_recommendations = recommendations_to_forecasts
recommendation_forecasts = recommendations_to_forecasts


@dataclass(frozen=True, slots=True)
class ConsensusForecastAdapter:
    """Run the existing consensus recommender and expose Forecast values."""

    model_id: str = "consensus-recommendation"
    recommendation_fn: Callable[..., pd.DataFrame] = backtest_recommendations

    def from_recommendations(
        self,
        recommendations: pd.DataFrame,
        as_of_date: DateLike | None = None,
        horizon: int | None = None,
        *,
        as_of: DateLike | None = None,
        horizon_days: int | None = None,
        feature_snapshot_sha256: str | None = None,
    ) -> tuple[Forecast, ...]:
        return recommendations_to_forecasts(
            recommendations,
            as_of_date,
            horizon,
            as_of=as_of,
            horizon_days=horizon_days,
            model_id=self.model_id,
            feature_snapshot_sha256=feature_snapshot_sha256,
        )

    from_dataframe = from_recommendations

    def forecast(
        self,
        signals_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        as_of_date: DateLike,
        horizon: int = 90,
        **recommendation_kwargs: Any,
    ) -> tuple[Forecast, ...]:
        resolved_horizon = _positive_int(horizon, "horizon")
        if resolved_horizon is None:
            raise ValueError("horizon is required")
        recommendation_kwargs = dict(recommendation_kwargs)
        recommendation_kwargs.setdefault("scoring_mode", "consensus")
        recommendations = self.recommendation_fn(
            signals_df,
            transactions_df,
            pd.Timestamp(as_of_date),
            horizon=resolved_horizon,
            **recommendation_kwargs,
        )
        return self.from_recommendations(recommendations, as_of_date, resolved_horizon)

    predict = forecast


def consensus_forecasts(
    signals_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    as_of_date: DateLike,
    horizon: int = 90,
    **recommendation_kwargs: Any,
) -> tuple[Forecast, ...]:
    """Convenience function using the production consensus recommender."""
    return ConsensusForecastAdapter().forecast(
        signals_df,
        transactions_df,
        as_of_date,
        horizon,
        **recommendation_kwargs,
    )


@dataclass(frozen=True, slots=True)
class TopNPolicy:
    """Equal-weight top-N policy over an explicit ranking score.

    This policy is intentionally rank-only: forecasts without
    ``ranking_score`` are not selected, even when they carry a separately
    named ``expected_net_alpha`` value.  No conversion between score types
    occurs at this boundary.
    """

    top_n: int = 10
    min_score: float | None = None
    policy_id: str = "top-n-ranking"
    exclude_held: bool = True

    def __post_init__(self) -> None:
        resolved_top_n = _positive_int(self.top_n, "top_n")
        if resolved_top_n is None:
            raise ValueError("top_n must be positive")
        object.__setattr__(self, "top_n", resolved_top_n)
        if self.min_score is not None:
            min_score = _number(self.min_score)
            if min_score is None:
                raise ValueError("min_score must be finite")
            object.__setattr__(self, "min_score", min_score)
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if not isinstance(self.exclude_held, bool):
            raise TypeError("exclude_held must be a bool")

    @staticmethod
    def _score(forecast: Forecast) -> float | None:
        return forecast.ranking_score

    def allocate(
        self,
        forecasts: Sequence[Forecast],
        market: MarketSnapshot | None = None,
        state: PortfolioState | None = None,
    ) -> TargetPortfolio:
        values = tuple(forecasts)
        if not all(isinstance(forecast, Forecast) for forecast in values):
            raise TypeError("TopNPolicy accepts only Forecast values")
        if market is not None and not isinstance(market, MarketSnapshot):
            raise TypeError("market must be a MarketSnapshot")
        if state is not None and not isinstance(state, PortfolioState):
            raise TypeError("state must be a PortfolioState")
        _assert_common_forecast_context(values)
        if values and market is not None and market.as_of != values[0].as_of:
            raise ValueError("market as_of must match forecast as_of")

        held = (
            set(state.held_tickers)
            if state is not None and self.exclude_held
            else set()
        )
        ranked = [
            (self._score(forecast), forecast)
            for forecast in values
            if forecast.ticker and forecast.ticker not in held
        ]
        min_score = self.min_score
        ranked = [
            (score, forecast)
            for score, forecast in ranked
            if score is not None and (min_score is None or score >= min_score)
        ]
        ranked.sort(key=_ranking_sort_key)
        selected = tuple(forecast for _, forecast in ranked[: self.top_n])

        if selected:
            weight = 1.0 / len(selected)
            positions = tuple(
                TargetPosition(
                    event_id=forecast.event_id,
                    ticker=forecast.ticker,
                    weight=weight,
                    rank=rank,
                    as_of=forecast.as_of,
                    horizon_days=forecast.horizon_days,
                    forecast=forecast,
                    provenance=forecast.provenance,
                )
                for rank, forecast in enumerate(selected, start=1)
            )
            cash_weight = 0.0
            portfolio_as_of = market.as_of if market is not None else selected[0].as_of
        else:
            positions = ()
            cash_weight = 1.0
            if market is not None:
                portfolio_as_of = market.as_of
            elif values:
                portfolio_as_of = values[0].as_of
            else:
                raise ValueError("an empty forecast set requires a market snapshot")

        return TargetPortfolio(
            as_of=portfolio_as_of,
            positions=positions,
            cash_weight=cash_weight,
            policy_id=self.policy_id,
        )


TopNRankingPolicy = TopNPolicy
RankingPolicy = TopNPolicy


def _decision_forecasts(
    decision: TargetPortfolio | Sequence[Forecast] | pd.DataFrame,
    *,
    as_of_date: DateLike | None,
    horizon: int | None,
) -> tuple[Forecast, ...]:
    if isinstance(decision, TargetPortfolio):
        values = tuple(position.forecast for position in decision.positions)
    elif isinstance(decision, pd.DataFrame):
        values = recommendations_to_forecasts(
            decision,
            as_of_date=as_of_date,
            horizon=horizon,
        )
    else:
        values = tuple(decision)
        if not all(isinstance(forecast, Forecast) for forecast in values):
            raise TypeError("decision must contain only Forecast values")
    _assert_common_forecast_context(values)
    return values


def _recommendation_frame(forecasts: Sequence[Forecast]) -> pd.DataFrame:
    rows = [
        {
            "ticker": forecast.ticker,
            "instrument_type": forecast.instrument_type or "stock",
            "amount_midpoint": forecast.amount_midpoint,
            "optimal_horizon": forecast.optimal_horizon_days
            or forecast.horizon_days,
            _DECISION_INDEX: index,
            _DECISION_EVENT_ID: forecast.event_id,
        }
        for index, forecast in enumerate(forecasts)
    ]
    frame = pd.DataFrame(rows)
    _assert_no_realized_columns(frame)
    return frame


def _ranking_sort_key(
    item: tuple[float | None, Forecast],
) -> tuple[float, str, str]:
    score, forecast = item
    if score is None:
        raise ValueError("cannot rank a forecast without a ranking score")
    return (-score, forecast.ticker, forecast.event_id)


def _optional_row_value(row: Mapping[str, object], column: str) -> float | None:
    return _number(row.get(column))


def _decision_index(value: object) -> int:
    number = _number(value)
    if number is None:
        raise ValueError("evaluator returned a missing decision index")
    try:
        converted = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("evaluator returned an invalid decision index") from exc
    if number != converted or converted < 0:
        raise ValueError("evaluator returned an invalid decision index")
    return converted


def _counter(value: object) -> int:
    number = _number(value)
    if number is None or number < 0:
        return 0
    try:
        converted = int(number)
    except (TypeError, ValueError, OverflowError):
        return 0
    if number != converted:
        return 0
    return converted


def _bool_cell(value: object, field_name: str) -> bool:
    value = _python_scalar(value)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        try:
            number = float(cast(Any, value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{field_name} must be a boolean value") from exc
        if np.isfinite(number) and number in (0.0, 1.0):
            return bool(number)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise TypeError(f"{field_name} must be a boolean value")


def _evidence_from_row(
    row: Mapping[str, object],
    forecast: Forecast,
    *,
    as_of_date: DateLike,
) -> Evidence:
    reported_event_id = _text(row.get(_DECISION_EVENT_ID))
    if reported_event_id is not None and reported_event_id != forecast.event_id:
        raise ValueError("evaluator changed the decision event_id")
    reported_ticker = _text(row.get("ticker"))
    if reported_ticker is not None and reported_ticker != forecast.ticker:
        raise ValueError("evaluator changed the decision ticker")

    horizon = _positive_int(row.get("bt_horizon_days"), "bt_horizon_days")
    if horizon is None:
        horizon = _positive_int(row.get("optimal_horizon"), "optimal_horizon")
    if horizon is None:
        horizon = forecast.horizon_days
    return Evidence(
        event_id=forecast.event_id,
        ticker=forecast.ticker,
        as_of=as_of_date,
        horizon_days=horizon,
        realized_return_pct=_optional_row_value(row, "bt_return_pct"),
        realized_alpha_pct=_optional_row_value(row, "bt_alpha_pct"),
        benchmark_return_pct=_optional_row_value(row, "bt_spy_return_pct"),
        raw_return_pct=_optional_row_value(row, "bt_raw_return_pct"),
        entry_price=_optional_row_value(row, "bt_entry_price"),
        exit_price=_optional_row_value(row, "bt_exit_price"),
        leverage=_optional_row_value(row, "bt_leverage"),
        entry_date=_date_like(row.get("bt_entry_date")),
        exit_date=_date_like(row.get("bt_exit_date")),
        entry_delay_days=_optional_row_value(row, "bt_entry_delay"),
        coverage=_text(row.get("bt_coverage")) or "unavailable",
        unavailable_reason=_text(row.get("bt_unavailable_reason")),
        stale_exit=_bool_cell(row.get("bt_stale_exit"), "bt_stale_exit"),
        delisted=_bool_cell(row.get("bt_delisted"), "bt_delisted"),
        provenance=forecast.provenance,
    )


@dataclass(frozen=True, slots=True)
class BacktestEvidenceAdapter:
    """Run legacy ``evaluate_backtest`` and return typed Evidence values."""

    evaluator: Callable[..., pd.DataFrame] = evaluate_backtest
    evaluator_id: str = "backtest"

    def evaluate_dataframe(
        self,
        decision: TargetPortfolio | Sequence[Forecast] | pd.DataFrame,
        prices_df: pd.DataFrame,
        as_of_date: DateLike | None = None,
        horizon: int | None = None,
        **evaluation_kwargs: Any,
    ) -> pd.DataFrame:
        if not isinstance(prices_df, pd.DataFrame):
            raise TypeError("prices_df must be a pandas DataFrame")
        forecasts = _decision_forecasts(
            decision,
            as_of_date=as_of_date,
            horizon=horizon,
        )
        if not forecasts:
            return pd.DataFrame()
        resolved_as_of = as_of_date
        if resolved_as_of is None:
            resolved_as_of = (
                decision.as_of
                if isinstance(decision, TargetPortfolio)
                else forecasts[0].as_of
            )
        resolved_horizon = _positive_int(horizon, "horizon")
        if resolved_horizon is None:
            resolved_horizon = forecasts[0].horizon_days
        recommendations = _recommendation_frame(forecasts)
        evaluated = self.evaluator(
            recommendations,
            prices_df,
            pd.Timestamp(resolved_as_of),
            resolved_horizon,
            **evaluation_kwargs,
        )
        if not isinstance(evaluated, pd.DataFrame):
            raise TypeError("evaluator must return a pandas DataFrame")
        return evaluated

    def evaluate_report(
        self,
        decision: TargetPortfolio | Sequence[Forecast] | pd.DataFrame,
        prices_df: pd.DataFrame,
        as_of_date: DateLike | None = None,
        horizon: int | None = None,
        **evaluation_kwargs: Any,
    ) -> EvidenceReport:
        forecasts = _decision_forecasts(
            decision,
            as_of_date=as_of_date,
            horizon=horizon,
        )
        if not forecasts:
            if as_of_date is None and isinstance(decision, TargetPortfolio):
                as_of_date = decision.as_of
            if as_of_date is None:
                raise ValueError("an as_of_date is required for an empty evidence report")
            resolved_horizon = _positive_int(horizon, "horizon")
            if resolved_horizon is None:
                raise ValueError("a horizon is required for an empty evidence report")
            return EvidenceReport(
                as_of=as_of_date,
                horizon_days=resolved_horizon,
                evaluator_id=self.evaluator_id,
            )

        resolved_as_of = as_of_date
        if resolved_as_of is None:
            resolved_as_of = (
                decision.as_of
                if isinstance(decision, TargetPortfolio)
                else forecasts[0].as_of
            )
        resolved_horizon = _positive_int(horizon, "horizon")
        if resolved_horizon is None:
            resolved_horizon = forecasts[0].horizon_days
        evaluated = self.evaluate_dataframe(
            decision,
            prices_df,
            resolved_as_of,
            resolved_horizon,
            **evaluation_kwargs,
        )

        has_decision_index = _DECISION_INDEX in evaluated.columns
        if not has_decision_index and len(evaluated) not in (0, len(forecasts)):
            raise ValueError(
                "evaluator output without decision indices cannot be safely aligned"
            )

        by_index = dict(enumerate(forecasts))
        evidence_by_index: dict[int, Evidence] = {}
        for output_position, (_, series) in enumerate(evaluated.iterrows()):
            row = series.to_dict()
            index = (
                _decision_index(row.get(_DECISION_INDEX))
                if has_decision_index
                else output_position
            )
            if index not in by_index:
                raise ValueError(f"evaluator returned unknown decision index {index}")
            if index in evidence_by_index:
                raise ValueError(f"evaluator returned duplicate decision index {index}")
            evidence_by_index[index] = _evidence_from_row(
                row,
                by_index[index],
                as_of_date=resolved_as_of,
            )

        # evaluate_backtest can omit rows with no price or no executable fill.
        # A typed unavailable observation keeps the decision/evidence join
        # one-to-one without inventing a return.
        ordered_evidence: list[Evidence] = []
        for index, forecast in by_index.items():
            if index in evidence_by_index:
                ordered_evidence.append(evidence_by_index[index])
                continue
            ordered_evidence.append(
                Evidence(
                    event_id=forecast.event_id,
                    ticker=forecast.ticker,
                    as_of=resolved_as_of,
                    horizon_days=forecast.horizon_days,
                    coverage="unavailable",
                    unavailable_reason="no_evaluation_row",
                    provenance=forecast.provenance,
                )
            )

        return EvidenceReport(
            as_of=resolved_as_of,
            horizon_days=resolved_horizon,
            evidence=tuple(ordered_evidence),
            evaluator_id=self.evaluator_id,
            n_no_price=_counter(evaluated.attrs.get("n_no_price", 0)),
            n_delisted=_counter(evaluated.attrs.get("n_delisted", 0)),
            n_unavailable=_counter(evaluated.attrs.get("n_unavailable", 0)),
        )

    def evaluate(
        self,
        decision: TargetPortfolio | Sequence[Forecast] | pd.DataFrame,
        prices: pd.DataFrame,
        *,
        as_of: DateLike | None = None,
        horizon_days: int | None = None,
        **evaluation_kwargs: Any,
    ) -> tuple[Evidence, ...]:
        return self.evaluate_report(
            decision,
            prices,
            as_of_date=as_of,
            horizon=horizon_days,
            **evaluation_kwargs,
        ).evidence


BacktestEvidence = BacktestEvidenceAdapter


def evaluate_backtest_evidence(
    decision: TargetPortfolio | Sequence[Forecast] | pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: DateLike | None = None,
    horizon: int | None = None,
    **evaluation_kwargs: Any,
) -> tuple[Evidence, ...]:
    """Evaluate a typed decision using the production backtest evaluator."""
    return BacktestEvidenceAdapter().evaluate_report(
        decision,
        prices_df,
        as_of_date,
        horizon,
        **evaluation_kwargs,
    ).evidence


backtest_evidence = evaluate_backtest_evidence


__all__ = [
    "BacktestEvidence",
    "BacktestEvidenceAdapter",
    "ConsensusForecastAdapter",
    "RankingPolicy",
    "TopNPolicy",
    "TopNRankingPolicy",
    "adapt_recommendations",
    "backtest_evidence",
    "consensus_forecasts",
    "evaluate_backtest_evidence",
    "recommendation_forecasts",
    "recommendations_to_forecasts",
]
