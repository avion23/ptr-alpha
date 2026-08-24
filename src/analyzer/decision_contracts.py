"""Immutable contracts for the forecast, policy, and evidence boundary.

The legacy analysis layer is intentionally DataFrame based.  These contracts
are a small value-object layer beside it: adapters may translate to and from
DataFrames, but realized backtest columns do not belong to a forecast or a
policy object.

The current production scorer is a ranker.  ``Forecast.ranking_score`` is
therefore intentionally distinct from ``expected_net_alpha``; an adapter for
the ranker must leave the latter unset rather than inventing an economic
interpretation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from math import isfinite
from numbers import Integral, Real
from typing import Any, Protocol, TypeAlias, runtime_checkable

DateLike: TypeAlias = date | datetime
MetadataValue: TypeAlias = str | int | float | bool | date | datetime | None
Metadata: TypeAlias = tuple[tuple[str, MetadataValue], ...]


def _as_datetime(value: object) -> datetime:
    """Convert a date-like value to a standard-library datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise TypeError("value must be a datetime or date")


def _as_optional_datetime(value: DateLike | None) -> datetime | None:
    return None if value is None else _as_datetime(value)


def _as_date(value: DateLike | None) -> date | None:
    if value is None:
        return None
    return _as_datetime(value).date()


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field_name} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _optional_finite_number(value: object, field_name: str) -> float | None:
    return None if value is None else _finite_number(value, field_name)


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field_name} must be a positive integer") from exc
    try:
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if not isfinite(numeric) or numeric <= 0 or numeric != integer:
        raise ValueError(f"{field_name} must be a positive integer")
    return integer


def _optional_positive_integer(value: object, field_name: str) -> int | None:
    return None if value is None else _positive_integer(value, field_name)


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
        raise TypeError(f"{field_name} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field_name} must be a non-negative integer") from exc
    try:
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if not isfinite(numeric) or numeric < 0 or numeric != integer:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return integer


def _strict_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _freeze_string_tuple(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of strings")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of strings") from exc
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f"{field_name} must be a sequence of strings")
    return values


def _freeze_metadata(value: Mapping[str, MetadataValue] | Metadata) -> Metadata:
    """Make metadata a tuple and reject mutable or realized values."""
    if isinstance(value, Mapping):
        items = value.items()
    else:
        try:
            items = iter(value)
        except TypeError as exc:
            raise TypeError("metadata must be a mapping or pair sequence") from exc

    frozen: list[tuple[str, MetadataValue]] = []
    for pair in items:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("metadata must contain (key, value) pairs")
        key, item = pair
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if key.startswith("bt_"):
            raise ValueError("realized bt_* fields are not forecast metadata")
        if (
            item is not None
            and not isinstance(item, (str, int, float, bool, date, datetime))
        ):
            raise TypeError(f"metadata value for {key!r} is not an immutable scalar")
        if isinstance(item, float) and not isfinite(item):
            raise ValueError(f"metadata value for {key!r} must be finite")
        frozen.append((key, item))
    return tuple(frozen)


def _assert_provenance_available_by(
    provenance: "Provenance", as_of: datetime, owner: str
) -> None:
    """Reject decisions that use information unavailable at their as-of date."""
    if provenance.available_date is None:
        return
    try:
        available_after_as_of = provenance.available_date > as_of
    except TypeError as exc:
        raise ValueError(
            f"{owner} provenance available_date and as_of must use comparable timezones"
        ) from exc
    if available_after_as_of:
        raise ValueError(
            f"{owner} provenance available_date must be on or before as_of"
        )


@dataclass(frozen=True, slots=True)
class Provenance:
    """Observable source identity retained across decision-layer adapters."""

    source: str | None = None
    source_record_id: str | None = None
    source_row_id: str | None = None
    ticker_origin: str | None = None
    available_date: datetime | date | None = None
    notification_date: datetime | date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _optional_string(self.source, "source"))
        object.__setattr__(
            self,
            "source_record_id",
            _optional_string(self.source_record_id, "source_record_id"),
        )
        object.__setattr__(
            self,
            "source_row_id",
            _optional_string(self.source_row_id, "source_row_id"),
        )
        object.__setattr__(
            self,
            "ticker_origin",
            _optional_string(self.ticker_origin, "ticker_origin"),
        )
        object.__setattr__(
            self, "available_date", _as_optional_datetime(self.available_date)
        )
        object.__setattr__(
            self,
            "notification_date",
            _as_optional_datetime(self.notification_date),
        )


@dataclass(frozen=True, slots=True)
class Forecast:
    """One point-in-time forecast or ranking-only model output.

    ``ranking_score`` is deliberately separate from ``expected_net_alpha``.
    The current consensus recommender emits a ranking score, not an expected
    return distribution, so its adapter leaves the distribution fields as
    ``None`` rather than assigning a false economic interpretation.
    """

    event_id: str
    as_of: datetime | date
    horizon_days: int
    expected_net_alpha: float | None = None
    net_alpha_std: float | None = None
    probability_positive: float | None = None
    model_id: str = "unknown"
    feature_snapshot_sha256: str | None = None
    provenance: Provenance = field(default_factory=Provenance)
    ticker: str = ""
    ranking_score: float | None = None
    optimal_horizon_days: int | None = None
    instrument_type: str | None = "stock"
    amount_midpoint: float | None = None
    metadata: Mapping[str, MetadataValue] | Metadata = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required_string(self.event_id, "event_id"))
        object.__setattr__(self, "model_id", _required_string(self.model_id, "model_id"))
        if not isinstance(self.ticker, str):
            raise TypeError("ticker must be a string")
        object.__setattr__(
            self, "horizon_days", _positive_integer(self.horizon_days, "horizon_days")
        )
        object.__setattr__(
            self,
            "optimal_horizon_days",
            _optional_positive_integer(
                self.optimal_horizon_days, "optimal_horizon_days"
            ),
        )
        as_of = _as_datetime(self.as_of)
        object.__setattr__(self, "as_of", as_of)
        object.__setattr__(
            self,
            "expected_net_alpha",
            _optional_finite_number(self.expected_net_alpha, "expected_net_alpha"),
        )
        object.__setattr__(
            self,
            "net_alpha_std",
            _optional_finite_number(self.net_alpha_std, "net_alpha_std"),
        )
        if self.net_alpha_std is not None and self.net_alpha_std < 0:
            raise ValueError("net_alpha_std must not be negative")
        object.__setattr__(
            self,
            "ranking_score",
            _optional_finite_number(self.ranking_score, "ranking_score"),
        )
        object.__setattr__(
            self,
            "amount_midpoint",
            _optional_finite_number(self.amount_midpoint, "amount_midpoint"),
        )
        if self.probability_positive is not None:
            probability = _finite_number(
                self.probability_positive, "probability_positive"
            )
            if not 0.0 <= probability <= 1.0:
                raise ValueError("probability_positive must be between zero and one")
            object.__setattr__(self, "probability_positive", probability)
        object.__setattr__(
            self,
            "feature_snapshot_sha256",
            _optional_string(self.feature_snapshot_sha256, "feature_snapshot_sha256"),
        )
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance")
        _assert_provenance_available_by(self.provenance, as_of, "forecast")
        object.__setattr__(self, "instrument_type", _optional_string(self.instrument_type, "instrument_type"))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def model_score(self) -> float | None:
        """Alias for the non-economic score used by ranking policies."""
        return self.ranking_score


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Optional point-in-time market inputs supplied to a policy."""

    as_of: datetime | date
    prices: Mapping[str, Real] | tuple[tuple[str, float], ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _as_datetime(self.as_of))
        raw_prices = (
            self.prices.items() if isinstance(self.prices, Mapping) else self.prices
        )
        frozen_prices: list[tuple[str, float]] = []
        seen: set[str] = set()
        try:
            for ticker, price in raw_prices:
                ticker = _required_string(ticker, "price ticker")
                if ticker in seen:
                    raise ValueError(f"duplicate market price for {ticker}")
                seen.add(ticker)
                frozen_prices.append((ticker, _finite_number(price, f"price[{ticker}]")))
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise TypeError("prices must be a mapping or pair sequence") from exc
        object.__setattr__(self, "prices", tuple(frozen_prices))


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Minimal immutable state a policy may use without querying storage."""

    capital: float = 0.0
    held_tickers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        capital = _finite_number(self.capital, "capital")
        if capital < 0:
            raise ValueError("capital must not be negative")
        object.__setattr__(self, "capital", capital)
        object.__setattr__(
            self,
            "held_tickers",
            _freeze_string_tuple(self.held_tickers, "held_tickers"),
        )


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """One policy target derived only from a forecast."""

    event_id: str
    ticker: str
    weight: float
    rank: int
    as_of: datetime | date
    horizon_days: int
    forecast: Forecast
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required_string(self.event_id, "event_id"))
        object.__setattr__(self, "ticker", _required_string(self.ticker, "ticker"))
        object.__setattr__(self, "rank", _positive_integer(self.rank, "rank"))
        object.__setattr__(
            self,
            "horizon_days",
            _positive_integer(self.horizon_days, "horizon_days"),
        )
        weight = _finite_number(self.weight, "weight")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("position weight must be between zero and one")
        if not isinstance(self.forecast, Forecast):
            raise TypeError("position forecast must be a Forecast")
        as_of = _as_datetime(self.as_of)
        if self.event_id != self.forecast.event_id:
            raise ValueError("position event_id must match its forecast")
        if self.ticker != self.forecast.ticker:
            raise ValueError("position ticker must match its forecast")
        if as_of != self.forecast.as_of:
            raise ValueError("position as_of must match its forecast")
        if self.horizon_days != self.forecast.horizon_days:
            raise ValueError("position horizon_days must match its forecast")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("position provenance must be a Provenance")
        if self.provenance != self.forecast.provenance:
            raise ValueError("position provenance must match its forecast")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "as_of", as_of)


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """Immutable policy result; it contains no realized outcome fields."""

    as_of: datetime | date
    positions: tuple[TargetPosition, ...] = ()
    cash_weight: float = 1.0
    policy_id: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _required_string(self.policy_id, "policy_id"))
        positions = tuple(self.positions)
        if not all(isinstance(position, TargetPosition) for position in positions):
            raise TypeError("positions must contain only TargetPosition values")
        portfolio_as_of = _as_datetime(self.as_of)
        if positions:
            common_as_of = positions[0].forecast.as_of
            common_horizon = positions[0].forecast.horizon_days
            if any(
                position.forecast.as_of != common_as_of
                or position.forecast.horizon_days != common_horizon
                for position in positions[1:]
            ):
                raise ValueError("target positions must share as_of and horizon_days")
            if portfolio_as_of != common_as_of:
                raise ValueError("target portfolio as_of must match its positions")
            tickers = [position.ticker for position in positions]
            if len(set(tickers)) != len(tickers):
                raise ValueError("target positions must not contain duplicate tickers")
            event_ids = [position.event_id for position in positions]
            if len(set(event_ids)) != len(event_ids):
                raise ValueError("target positions must contain unique event_ids")
        cash_weight = _finite_number(self.cash_weight, "cash_weight")
        if not 0.0 <= cash_weight <= 1.0:
            raise ValueError("cash_weight must be between zero and one")
        total_weight = sum(position.weight for position in positions)
        if not isfinite(total_weight) or abs(total_weight + cash_weight - 1.0) > 1e-9:
            raise ValueError("target position and cash weights must sum to one")
        object.__setattr__(self, "as_of", portfolio_as_of)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cash_weight", cash_weight)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(position.ticker for position in self.positions)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One realized evaluation observation, kept separate from decisions."""

    event_id: str
    ticker: str
    as_of: datetime | date
    horizon_days: int
    realized_return_pct: float | None = None
    realized_alpha_pct: float | None = None
    benchmark_return_pct: float | None = None
    raw_return_pct: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    leverage: float | None = None
    entry_date: date | datetime | None = None
    exit_date: date | datetime | None = None
    entry_delay_days: float | None = None
    coverage: str = "unavailable"
    unavailable_reason: str | None = None
    stale_exit: bool = False
    delisted: bool = False
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required_string(self.event_id, "event_id"))
        object.__setattr__(self, "ticker", _required_string(self.ticker, "ticker"))
        object.__setattr__(
            self,
            "horizon_days",
            _positive_integer(self.horizon_days, "horizon_days"),
        )
        object.__setattr__(self, "coverage", _required_string(self.coverage, "coverage"))
        object.__setattr__(
            self,
            "unavailable_reason",
            _optional_string(self.unavailable_reason, "unavailable_reason"),
        )
        as_of = _as_datetime(self.as_of)
        object.__setattr__(self, "as_of", as_of)
        for field_name in (
            "realized_return_pct",
            "realized_alpha_pct",
            "benchmark_return_pct",
            "raw_return_pct",
            "entry_price",
            "exit_price",
            "leverage",
            "entry_delay_days",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_finite_number(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "entry_date", _as_date(self.entry_date))
        object.__setattr__(self, "exit_date", _as_date(self.exit_date))
        object.__setattr__(self, "stale_exit", _strict_bool(self.stale_exit, "stale_exit"))
        object.__setattr__(self, "delisted", _strict_bool(self.delisted, "delisted"))
        if not isinstance(self.provenance, Provenance):
            raise TypeError("evidence provenance must be a Provenance")
        _assert_provenance_available_by(self.provenance, as_of, "evidence")

    @property
    def realized_net_return(self) -> float | None:
        return self.realized_return_pct

    @property
    def realized_net_alpha(self) -> float | None:
        return self.realized_alpha_pct


@dataclass(frozen=True, slots=True)
class EvidenceReport:
    """Batch evidence plus non-row evaluator counters."""

    as_of: datetime | date
    horizon_days: int
    evidence: tuple[Evidence, ...] = ()
    evaluator_id: str = "backtest"
    n_no_price: int = 0
    n_delisted: int = 0
    n_unavailable: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _as_datetime(self.as_of))
        object.__setattr__(
            self,
            "horizon_days",
            _positive_integer(self.horizon_days, "horizon_days"),
        )
        object.__setattr__(self, "evaluator_id", _required_string(self.evaluator_id, "evaluator_id"))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not all(isinstance(item, Evidence) for item in self.evidence):
            raise TypeError("evidence must contain only Evidence values")
        if any(item.as_of != self.as_of for item in self.evidence):
            raise ValueError("evidence items must share the report as_of")
        if any(item.horizon_days != self.horizon_days for item in self.evidence):
            raise ValueError("evidence items must share the report horizon_days")
        event_ids = [item.event_id for item in self.evidence]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("evidence items must contain unique event_ids")
        for field_name in ("n_no_price", "n_delisted", "n_unavailable"):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(getattr(self, field_name), field_name),
            )


@runtime_checkable
class FrozenForecastModel(Protocol):
    def predict(self, snapshot: Any) -> tuple[Forecast, ...]:
        """Produce forecasts from an immutable point-in-time snapshot."""
        raise RuntimeError("FrozenForecastModel.predict is a protocol declaration")


@runtime_checkable
class ForecastModel(Protocol):
    def fit(self, dataset: Any) -> FrozenForecastModel:
        """Fit a model using the supplied training dataset."""
        raise RuntimeError("ForecastModel.fit is a protocol declaration")


@runtime_checkable
class Policy(Protocol):
    def allocate(
        self,
        forecasts: Sequence[Forecast],
        market: MarketSnapshot | None = None,
        state: PortfolioState | None = None,
    ) -> TargetPortfolio:
        """Convert forecasts and immutable state into a target portfolio."""
        raise RuntimeError("Policy.allocate is a protocol declaration")


@runtime_checkable
class EvidenceEvaluator(Protocol):
    def evaluate(
        self,
        decision: TargetPortfolio | Sequence[Forecast],
        prices: Any,
        *,
        as_of: DateLike | None = None,
        horizon_days: int | None = None,
    ) -> tuple[Evidence, ...]:
        """Evaluate a decision without mutating the decision contracts."""
        raise RuntimeError("EvidenceEvaluator.evaluate is a protocol declaration")


__all__ = [
    "DateLike",
    "Evidence",
    "EvidenceEvaluator",
    "EvidenceReport",
    "Forecast",
    "ForecastModel",
    "FrozenForecastModel",
    "MarketSnapshot",
    "Metadata",
    "MetadataValue",
    "Policy",
    "PortfolioState",
    "Provenance",
    "TargetPortfolio",
    "TargetPosition",
]
