"""Fold-local research baselines.

The classes in this module are deliberately *not* production models.  They do
not implement a decision contract, are never imported by the live scorer, and
hard-code ``deployment_authorized = False``.  Their purpose is to make a safe,
repeatable comparison for future research:

* :class:`DynamicHierarchicalBaseline` estimates slowly changing additive
  group effects with empirical-Bayes shrinkage and reports predictive
  uncertainty; and
* :class:`DeterministicRegularizedTabularBaseline` uses a dependency-free
  ridge fit over numeric, categorical, and deterministic threshold features.
  The threshold basis is a small tree/stump-like nonlinear baseline rather
  than a stochastic forest.

Both models have the same safety boundary.  A fit is local to a supplied
cutoff, labels are admitted only when their availability date is strictly
before that cutoff, duplicate events are rejected, and names that look like
realized or future outcomes cannot be used as features.  Non-local rows are
excluded and counted in provenance; ``strict_future=True`` turns that
exclusion into a hard error for callers that prefer fail-closed fold
construction.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from math import erf, log, sqrt
from typing import Any, cast

import numpy as np
import pandas as pd


class ResearchOnlyError(RuntimeError):
    """Raised when code attempts to authorize a research model for deployment."""


@dataclass(frozen=True, slots=True)
class ResearchModelProvenance:
    """Immutable, serializable description of a fold-local model fit."""

    model_name: str
    model_version: str
    fit_cutoff: pd.Timestamp
    fit_rows: int
    rejected_future_event_rows: int
    rejected_future_label_rows: int
    rejected_missing_label_rows: int
    label_column: str
    feature_columns: tuple[str, ...]
    group_columns: tuple[str, ...]
    seed: int | None
    hyperparameters: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    research_only: bool = True
    deployment_authorized: bool = False
    time_column: str = "entry_date"
    event_id_column: str | None = "event_id"
    label_available_column: str | None = None
    strict_future: bool = False
    uncertainty_semantics: str = "predictive_standard_deviation"
    uncertainty_horizon_days: int | None = None
    training_data_hash: str = "unavailable"
    model_parameter_hash: str = "unavailable"

    def __post_init__(self) -> None:
        if not self.research_only or self.deployment_authorized:
            raise ResearchOnlyError(
                "research model provenance cannot authorize deployment"
            )
        if self.fit_rows < 1:
            raise ValueError("fit_rows must be positive")
        try:
            fit_cutoff = pd.Timestamp(cast(Any, self.fit_cutoff))
        except (TypeError, ValueError) as exc:
            raise ValueError("fit_cutoff must be a valid timestamp") from exc
        if pd.isna(fit_cutoff):
            raise ValueError("fit_cutoff must not be missing")
        if fit_cutoff.tzinfo is not None:
            fit_cutoff = fit_cutoff.tz_convert("UTC").tz_localize(None)
        object.__setattr__(self, "fit_cutoff", cast(pd.Timestamp, fit_cutoff))
        if not self.uncertainty_semantics.startswith(
            "predictive_standard_deviation"
        ):
            raise ValueError(
                "research baseline uncertainty must use predictive standard deviation"
            )
        if self.uncertainty_horizon_days is not None and (
            self.uncertainty_horizon_days <= 0
        ):
            raise ValueError("uncertainty_horizon_days must be positive")

    @property
    def provenance_id(self) -> str:
        """Return a stable id that does not depend on object identity."""
        return _sha256_payload(self.as_dict(include_id=False))

    @property
    def provenance_hash(self) -> str:
        """Alias for the complete provenance digest used in model outputs."""
        return self.provenance_id

    @property
    def training_data_sha256(self) -> str:
        """Compatibility spelling for the training-data digest."""
        return self.training_data_hash

    @property
    def model_parameter_sha256(self) -> str:
        """Compatibility spelling for the fitted-parameter digest."""
        return self.model_parameter_hash

    def as_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "fit_cutoff": self.fit_cutoff.isoformat(),
            "fit_rows": self.fit_rows,
            "rejected_future_event_rows": self.rejected_future_event_rows,
            "rejected_future_label_rows": self.rejected_future_label_rows,
            "rejected_missing_label_rows": self.rejected_missing_label_rows,
            "label_column": self.label_column,
            "feature_columns": list(self.feature_columns),
            "group_columns": list(self.group_columns),
            "seed": self.seed,
            "hyperparameters": dict(self.hyperparameters),
            "assumptions": list(self.assumptions),
            "research_only": self.research_only,
            "deployment_authorized": self.deployment_authorized,
            "time_column": self.time_column,
            "event_id_column": self.event_id_column,
            "label_available_column": self.label_available_column,
            "strict_future": self.strict_future,
            "uncertainty_semantics": self.uncertainty_semantics,
            "uncertainty_horizon_days": self.uncertainty_horizon_days,
            "training_data_hash": self.training_data_hash,
            "model_parameter_hash": self.model_parameter_hash,
        }
        if include_id:
            result["provenance_id"] = self.provenance_id
            result["provenance_hash"] = self.provenance_hash
        return result


def _json_safe(value: object) -> object:
    """Convert numpy/pandas values to deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sha256_payload(value: object) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Hash the exact normalized rows and schema used to construct a fold."""
    digest = sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(column) for column in frame.columns],
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "shape": [int(frame.shape[0]), int(frame.shape[1])],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    try:
        row_hashes = pd.util.hash_pandas_object(frame, index=True)
        digest.update(row_hashes.to_numpy(dtype=np.uint64).tobytes())
    except (TypeError, ValueError):
        # Object columns containing an unusual extension value are uncommon,
        # but their string representation is still preferable to an absent
        # audit hash.
        rows = [
            [_json_safe(value) for value in row]
            for row in frame.itertuples(index=True, name=None)
        ]
        digest.update(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def _uncertainty_semantics(horizon_days: int) -> str:
    return f"predictive_standard_deviation_for_{int(horizon_days)}_day_horizon"


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return one column after duplicate-column validation."""
    return cast(pd.Series, frame[name])


def _as_timestamp(value: object, *, name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    if pd.isna(parsed):
        raise ValueError(f"{name} is missing")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return cast(pd.Timestamp, parsed)


def _datetime_series(values: pd.Series, *, name: str) -> pd.Series:
    """Parse mixed timezone values into one timezone-naive UTC series."""
    try:
        parsed: Any = pd.to_datetime(values, errors="coerce", utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains an invalid date") from exc
    if parsed.isna().any():
        raise ValueError(f"{name} contains an invalid or missing date")
    return parsed.dt.tz_localize(None)


_STRUCTURAL_COLUMNS = frozenset(
    {
        "event_id",
        "transaction_id",
        "event_key",
        "entry_date",
        "disclosure_date",
        "transaction_date",
        "as_of",
        "timestamp",
        "date",
        "label_available_date",
        "outcome_available_date",
        "outcome_date",
        "exit_date",
        "horizon_days",
        "requested_entry_date",
        "fit_cutoff",
        "__label_available_date",
    }
)

# These tokens are intentionally conservative.  A feature called ``future``
# or ``spy_return`` is not safe to infer from its spelling, even if a caller
# believes it was computed in advance.  Such a field must be renamed or
# transformed before entering a fold.
_LEAKAGE_TOKENS = (
    "future",
    "forward",
    "target",
    "label",
    "outcome",
    "realized",
    "return",
    "alpha",
    "profit",
    "pnl",
    "payoff",
    "exit",
    "sell",
    "available",
)


def _looks_like_leakage(
    column: str, *, label_column: str, allowed_columns: Iterable[str] = ()
) -> bool:
    normalized = str(column).strip().lower()
    allowed = {str(value).strip().lower() for value in allowed_columns}
    if normalized in allowed or normalized == str(label_column).strip().lower():
        return False
    if normalized in _STRUCTURAL_COLUMNS:
        return False
    return any(token in normalized for token in _LEAKAGE_TOKENS)


def _reject_leakage_columns(
    columns: Iterable[str],
    *,
    label_column: str,
    context: str,
    allowed_columns: Iterable[str] = (),
) -> None:
    offending = sorted(
        str(column)
        for column in columns
        if _looks_like_leakage(
            str(column),
            label_column=label_column,
            allowed_columns=allowed_columns,
        )
    )
    if offending:
        raise ValueError(
            f"{context} contains outcome/leakage columns that cannot enter a fold: "
            f"{offending}"
        )


_PREDICTION_OUTCOME_COLUMNS = frozenset(
    {
        "label_available_date",
        "outcome_available_date",
        "outcome_date",
        "exit_date",
        "requested_entry_date",
    }
)


def _reject_prediction_leakage_columns(
    columns: Iterable[str], *, label_column: str, context: str
) -> None:
    offending = sorted(
        str(column)
        for column in columns
        if str(column).strip().lower() == str(label_column).strip().lower()
        or str(column).strip().lower() in _PREDICTION_OUTCOME_COLUMNS
    )
    if offending:
        raise ValueError(
            f"{context} contains realized outcome columns that cannot be used "
            f"for prediction: {offending}"
        )


def _event_keys(
    frame: pd.DataFrame,
    *,
    time_column: str,
    event_id_column: str | None,
) -> pd.Series:
    if event_id_column is not None and event_id_column in frame.columns:
        ids = frame[event_id_column]
        if bool(ids.isna().any()) or bool(ids.astype(str).str.strip().eq("").any()):
            raise ValueError("event ids must be non-empty and non-null")
        return ids.astype(str).str.strip()

    identity_columns = [time_column]
    # These columns distinguish common same-day events when an explicit id was
    # not carried by a source.  If none are available, the time itself is the
    # only safe identity and collisions are rejected.
    for candidate in (
        "ticker",
        "symbol",
        "member",
        "sector",
        "transaction_type",
        "owner_code",
    ):
        if candidate in frame.columns:
            identity_columns.append(candidate)
    normalized = cast(pd.DataFrame, frame.loc[:, identity_columns].copy())
    for column in identity_columns:
        if column == time_column:
            normalized[column] = pd.to_datetime(normalized[column]).astype("int64")
        else:
            normalized[column] = _column(normalized, column).map(
                lambda value: (
                    "__UNKNOWN__" if pd.isna(value) else str(value).strip().upper()
                )
            )
    return cast(pd.Series, normalized.astype(str).agg("|".join, axis=1))


@dataclass(frozen=True, slots=True)
class _PreparedTraining:
    frame: pd.DataFrame
    cutoff: pd.Timestamp
    feature_columns: tuple[str, ...]
    group_columns: tuple[str, ...]
    rejected_future_events: int
    rejected_future_labels: int
    rejected_missing_labels: int
    training_data_hash: str


def _validate_group_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Normalize group names and reject duplicate grouping dimensions."""
    normalized = tuple(str(value).strip() for value in columns)
    if any(not value for value in normalized):
        raise ValueError("group_columns must not contain empty names")
    folded = tuple(value.casefold() for value in normalized)
    if len(set(folded)) != len(folded):
        raise ValueError("group_columns must contain unique dimensions")
    return normalized


def _resolve_label_availability(
    frame: pd.DataFrame,
    *,
    time_column: str,
    horizon_days: int,
    label_available_column: str | None,
) -> pd.Series:
    requested = label_available_column
    if requested is not None and requested not in frame.columns:
        raise ValueError(f"label availability column {requested!r} is missing")
    candidates = (
        requested,
        "label_available_date",
        "outcome_available_date",
        "outcome_date",
        "exit_date",
    )
    for candidate in candidates:
        if candidate is not None and candidate in frame.columns:
            return _datetime_series(
                _column(frame, candidate), name=f"{candidate} label availability"
            )
    return _datetime_series(
        _column(frame, time_column), name=time_column
    ) + pd.Timedelta(days=horizon_days)


def _candidate_features(
    frame: pd.DataFrame,
    *,
    label_column: str,
    time_column: str,
    event_id_column: str | None,
    group_columns: Sequence[str],
    explicit: Sequence[str] | None,
    label_available_column: str | None,
) -> tuple[str, ...]:
    reserved = set(_STRUCTURAL_COLUMNS) | {
        label_column,
        time_column,
        *group_columns,
    }
    if event_id_column is not None:
        reserved.add(event_id_column)
    if label_available_column is not None:
        reserved.add(label_available_column)
    if explicit is not None:
        columns = tuple(str(column) for column in explicit)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"feature columns are missing: {missing}")
        protected = sorted(set(columns) & reserved)
        if protected:
            raise ValueError(
                "feature columns contain structural or label columns that "
                f"cannot enter a fold: {protected}"
            )
    else:
        columns = tuple(
            str(column) for column in frame.columns if column not in reserved
        )
    _reject_leakage_columns(
        columns, label_column=label_column, context="feature schema"
    )
    if len(set(columns)) != len(columns):
        raise ValueError("feature columns must be unique")
    return columns


def _prepare_training(
    data: pd.DataFrame,
    *,
    cutoff: object | None,
    label_column: str,
    time_column: str,
    event_id_column: str | None,
    group_columns: Sequence[str],
    feature_columns: Sequence[str] | None,
    horizon_days: int,
    label_available_column: str | None,
    strict_future: bool,
    reserve_group_columns: bool = True,
) -> _PreparedTraining:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("training data must be a non-empty DataFrame")
    frame = data.copy(deep=True)
    if frame.columns.duplicated().any():
        raise ValueError("training data contains duplicate column names")
    for required in (label_column, time_column):
        if required not in frame.columns:
            raise ValueError(f"training data is missing required column {required!r}")
    if (
        event_id_column is not None
        and event_id_column not in frame.columns
        and event_id_column != "event_id"
    ):
        raise ValueError(
            f"training data is missing event id column {event_id_column!r}"
        )
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    _reject_leakage_columns(
        frame.columns,
        label_column=label_column,
        context="training data",
        allowed_columns=(label_available_column,)
        if label_available_column is not None
        else (),
    )
    frame[time_column] = _datetime_series(_column(frame, time_column), name=time_column)
    ids = _event_keys(frame, time_column=time_column, event_id_column=event_id_column)
    if ids.duplicated().any():
        duplicate_ids = cast(pd.Series, ids[ids.duplicated(keep=False)])
        duplicates = list(pd.unique(duplicate_ids.astype(str)))
        raise ValueError(
            f"duplicate events are not independent evidence: {duplicates[:5]}"
        )

    availability = _resolve_label_availability(
        frame,
        time_column=time_column,
        horizon_days=horizon_days,
        label_available_column=label_available_column,
    )
    frame["__label_available_date"] = availability
    if cutoff is None:
        # A fit without an explicit fold boundary is still finite and
        # reproducible, but callers should prefer an explicit cutoff.
        fit_cutoff = _as_timestamp(
            max(
                _column(frame, time_column).max(),
                cast(pd.Timestamp, availability.max()),
            ),
            name="derived cutoff",
        ) + pd.Timedelta(nanoseconds=1)
    else:
        fit_cutoff = _as_timestamp(cutoff, name="cutoff")

    future_events = frame[time_column] >= fit_cutoff
    # A row observed exactly at the fold boundary is not part of the fold.
    # This strict inequality is intentional: the boundary represents the
    # start of the next information set, not an inclusive observation date.
    future_labels = frame["__label_available_date"] >= fit_cutoff
    if strict_future and (future_events | future_labels).any():
        raise ValueError(
            "training data contains an event or label that is not known at the fold cutoff"
        )
    rejected_future_events = int(future_events.sum())
    rejected_future_labels = int((~future_events & future_labels).sum())

    numeric_labels = cast(
        pd.Series, pd.to_numeric(_column(frame, label_column), errors="coerce")
    )
    missing_labels = numeric_labels.isna() | ~np.isfinite(
        numeric_labels.to_numpy(dtype=float)
    )
    rejected_missing_labels = int(missing_labels.sum())
    frame[label_column] = numeric_labels
    eligible = ~(future_events | future_labels | missing_labels)
    frame = frame.loc[eligible].copy()
    if frame.empty:
        raise ValueError("no mature, pre-cutoff labeled rows remain for the fold")

    available_groups = tuple(
        str(column) for column in group_columns if column in frame.columns
    )
    feature_groups = available_groups if reserve_group_columns else ()
    selected_features = _candidate_features(
        frame,
        label_column=label_column,
        time_column=time_column,
        event_id_column=event_id_column,
        group_columns=feature_groups,
        explicit=feature_columns,
        label_available_column=label_available_column,
    )
    # Group columns are not allowed to disappear because their values are
    # normalized by each model for safe pooling of unseen values.
    for column in available_groups:
        frame[column] = frame[column].map(_clean_category)
    sort_columns = [time_column]
    if event_id_column is not None and event_id_column in frame:
        sort_columns.append(event_id_column)
    frame.sort_values(sort_columns, inplace=True)
    frame.reset_index(drop=True, inplace=True)
    fit_cutoff = cast(pd.Timestamp, fit_cutoff)
    return _PreparedTraining(
        frame=frame,
        cutoff=fit_cutoff,
        feature_columns=selected_features,
        group_columns=available_groups,
        rejected_future_events=rejected_future_events,
        rejected_future_labels=rejected_future_labels,
        rejected_missing_labels=rejected_missing_labels,
        training_data_hash=_frame_sha256(frame),
    )


def _clean_category(value: object) -> str:
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return "__UNKNOWN__"
    text = str(value).strip()
    return text if text else "__UNKNOWN__"


def _check_prediction_horizon(frame: pd.DataFrame, horizon_days: int) -> None:
    """Reject query rows whose declared label horizon differs from the fit."""
    if "horizon_days" not in frame.columns:
        return
    values = pd.to_numeric(_column(frame, "horizon_days"), errors="coerce")
    if values.isna().any() or (values <= 0).any():
        raise ValueError("prediction horizon_days must be positive integers")
    if not np.all(values.to_numpy(dtype=float) == float(horizon_days)):
        raise ValueError(
            "prediction horizon_days must match the fitted model horizon "
            f"({horizon_days})"
        )


class _ResearchOnlyMixin:
    research_only = True
    deployment_authorized = False
    can_deploy = False

    def authorize_deployment(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ResearchOnlyError(
            "research baselines are permanently research-only and cannot authorize deployment"
        )

    def to_deployment_contract(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ResearchOnlyError(
            "research baselines do not implement a deployment decision contract"
        )


@dataclass(frozen=True, slots=True)
class _EffectTable:
    means: Mapping[Any, float]
    variances: Mapping[Any, float]
    counts: Mapping[Any, float]
    between_var: float


def _effect_payload(effect: _EffectTable) -> dict[str, object]:
    """Serialize an effect table without relying on heterogeneous dict keys."""
    return {
        "means": sorted(
            ((repr(key), float(value)) for key, value in effect.means.items()),
            key=lambda item: item[0],
        ),
        "variances": sorted(
            ((repr(key), float(value)) for key, value in effect.variances.items()),
            key=lambda item: item[0],
        ),
        "counts": sorted(
            ((repr(key), float(value)) for key, value in effect.counts.items()),
            key=lambda item: item[0],
        ),
        "between_var": float(effect.between_var),
    }


def _group_keys(frame: pd.DataFrame, columns: Sequence[str]) -> list[Any]:
    if len(columns) == 1:
        return [_clean_category(value) for value in frame[columns[0]]]
    return [
        tuple(_clean_category(row[column]) for column in columns)
        for _, row in frame.loc[:, list(columns)].iterrows()
    ]


def _variance_floor(values: np.ndarray) -> float:
    magnitude = max(float(np.max(np.abs(values))) if len(values) else 0.0, 1.0)
    return max((np.finfo(float).eps * magnitude) ** 2, 1.0 / np.finfo(float).max)


def _fit_shrunk_effect(
    residual: np.ndarray,
    keys: Sequence[Any],
    weights: np.ndarray,
    *,
    prior_strength: float,
    min_effect_count: float = 0.0,
) -> _EffectTable:
    grouped: dict[Any, list[int]] = {}
    for position, key in enumerate(keys):
        grouped.setdefault(key, []).append(position)
    selected = {
        key: positions
        for key, positions in grouped.items()
        if float(weights[positions].sum()) >= min_effect_count
    }
    if not selected:
        return _EffectTable({}, {}, {}, _variance_floor(residual))

    group_means: dict[Any, float] = {}
    group_counts: dict[Any, float] = {}
    for key in sorted(selected, key=repr):
        positions = selected[key]
        weight = weights[positions]
        group_counts[key] = float(weight.sum())
        group_means[key] = float(np.dot(residual[positions], weight) / weight.sum())
    mean_values = np.asarray(list(group_means.values()), dtype=float)
    if len(residual) > len(selected):
        fitted = np.asarray([group_means[key] for key in keys if key in group_means])
        selected_positions = np.asarray(
            [position for position, key in enumerate(keys) if key in selected],
            dtype=int,
        )
        residual_values = residual[selected_positions] - fitted
        residual_weights = weights[selected_positions]
        within_dof = max(float(residual_weights.sum()) - len(selected), 1.0)
        within_var = float(np.dot(residual_values**2, residual_weights) / within_dof)
    else:
        within_var = float(np.var(residual, ddof=1)) if len(residual) > 1 else 0.0
    within_var = max(within_var, _variance_floor(residual))

    if len(mean_values) > 1:
        center = float(
            np.average(mean_values, weights=np.asarray(list(group_counts.values())))
        )
        observed_between = float(
            np.average(
                (mean_values - center) ** 2,
                weights=np.asarray(list(group_counts.values())),
            )
        )
    else:
        observed_between = 0.0
    # Even a perfectly consistent toy group has unobserved event noise.  A
    # tiny representability floor would make repeated, noiseless fixtures
    # behave like known constants and would defeat the model's promised
    # partial pooling.  One percent of the observed between-group spread is a
    # conservative process-noise floor; a genuinely homogeneous population
    # still has zero effect.
    within_var = max(within_var, observed_between * 0.01, _variance_floor(residual))
    average_sampling = float(
        np.mean([within_var / max(count, 1.0) for count in group_counts.values()])
    )
    estimated_between = observed_between - average_sampling
    # With one observation in every group, the within/between decomposition is
    # not identified.  Using the observed spread as a weak prior preserves
    # useful partial pooling instead of manufacturing exact zero effects.
    if all(count <= 1.0 + 1e-12 for count in group_counts.values()):
        estimated_between = observed_between
    between_var = max(estimated_between, _variance_floor(residual))

    means: dict[Any, float] = {}
    variances: dict[Any, float] = {}
    for key in sorted(group_means, key=repr):
        count = group_counts[key]
        denominator = count * between_var + prior_strength * within_var
        shrinkage = (prior_strength * within_var) / denominator
        means[key] = float((1.0 - shrinkage) * group_means[key])
        variances[key] = float(max(within_var * between_var / denominator, 0.0))
    return _EffectTable(means, variances, group_counts, between_var)


def _normal_probability_positive(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.maximum(np.asarray(std, dtype=float), np.finfo(float).tiny)
    values = 0.5 * (
        1.0 + np.vectorize(erf)(np.asarray(mean, dtype=float) / safe_std / sqrt(2.0))
    )
    return np.clip(values, 0.0, 1.0)


class DynamicHierarchicalBaseline(_ResearchOnlyMixin):
    """A deterministic, fold-local additive dynamic hierarchical baseline.

    The baseline fits a recency-weighted population mean followed by shrunk
    time, configured group, and group-interaction effects.  Effects are fit in
    sequence on residuals, which keeps the model small and makes its pooling
    behavior inspectable.  A group not observed in the fold receives a zero
    effect and the corresponding between-group prior variance, rather than a
    missing prediction or a fabricated identity-specific estimate.
    """

    def __init__(
        self,
        *,
        label_column: str = "factor_residual_return",
        time_column: str = "entry_date",
        group_columns: Sequence[str] = ("member", "sector"),
        event_id_column: str | None = "event_id",
        horizon_days: int = 90,
        prior_strength: float = 4.0,
        discount: float = 0.995,
        min_time_observations: float = 2.0,
        feature_columns: Sequence[str] | None = None,
        feature_regularization: float = 1.0,
        label_available_column: str | None = None,
        strict_future: bool = False,
    ) -> None:
        if prior_strength <= 0 or not np.isfinite(prior_strength):
            raise ValueError("prior_strength must be positive and finite")
        if not 0 < discount <= 1 or not np.isfinite(discount):
            raise ValueError("discount must be in (0, 1]")
        if min_time_observations < 1:
            raise ValueError("min_time_observations must be positive")
        if feature_regularization <= 0 or not np.isfinite(feature_regularization):
            raise ValueError("feature_regularization must be positive and finite")
        self.label_column = label_column
        self.time_column = time_column
        self.group_columns = _validate_group_columns(group_columns)
        self.event_id_column = event_id_column
        self.horizon_days = int(horizon_days)
        self.prior_strength = float(prior_strength)
        self.discount = float(discount)
        self.min_time_observations = float(min_time_observations)
        self.requested_feature_columns = (
            tuple(str(value) for value in feature_columns)
            if feature_columns is not None
            else None
        )
        self.feature_regularization = float(feature_regularization)
        self.label_available_column = label_available_column
        self.strict_future = bool(strict_future)
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        *,
        cutoff: object | None = None,
        fit_cutoff: object | None = None,
    ) -> DynamicHierarchicalBaseline:
        if cutoff is not None and fit_cutoff is not None:
            raise ValueError("pass only one of cutoff and fit_cutoff")
        self._fitted = False
        selected_cutoff = cutoff if cutoff is not None else fit_cutoff
        prepared = _prepare_training(
            data,
            cutoff=selected_cutoff,
            label_column=self.label_column,
            time_column=self.time_column,
            event_id_column=self.event_id_column,
            group_columns=self.group_columns,
            feature_columns=self.requested_feature_columns,
            horizon_days=self.horizon_days,
            label_available_column=self.label_available_column,
            strict_future=self.strict_future,
            reserve_group_columns=True,
        )
        self._fit_prepared(prepared)
        return self

    def fit_fold(
        self, data: pd.DataFrame, cutoff: object
    ) -> DynamicHierarchicalBaseline:
        """Fit a fold, excluding non-local rows even if ``strict_future`` is set."""
        self._fitted = False
        prepared = _prepare_training(
            data,
            cutoff=cutoff,
            label_column=self.label_column,
            time_column=self.time_column,
            event_id_column=self.event_id_column,
            group_columns=self.group_columns,
            feature_columns=self.requested_feature_columns,
            horizon_days=self.horizon_days,
            label_available_column=self.label_available_column,
            strict_future=False,
            reserve_group_columns=True,
        )
        self._fit_prepared(prepared)
        return self

    def _fit_prepared(self, prepared: _PreparedTraining) -> None:
        frame = prepared.frame
        y = frame[self.label_column].to_numpy(dtype=float)
        dates = pd.to_datetime(frame[self.time_column])
        days_old = (prepared.cutoff - dates).dt.total_seconds().to_numpy(
            dtype=float
        ) / 86400.0
        weights = np.exp(np.maximum(days_old, 0.0) * log(self.discount))
        weights = np.maximum(weights, np.finfo(float).tiny)
        global_mean = float(np.dot(y, weights) / weights.sum())
        self._global_mean = global_mean
        residual = y - global_mean

        time_keys = [
            _as_timestamp(value, name="entry_date").normalize() for value in dates
        ]
        self._time_effect = _fit_shrunk_effect(
            residual,
            time_keys,
            weights,
            prior_strength=self.prior_strength,
            min_effect_count=self.min_time_observations,
        )
        residual = residual - np.asarray(
            [self._time_effect.means.get(key, 0.0) for key in time_keys], dtype=float
        )

        self._effects: dict[str, _EffectTable] = {}
        for column in prepared.group_columns:
            keys = _group_keys(frame, (column,))
            effect = _fit_shrunk_effect(
                residual,
                keys,
                weights,
                prior_strength=self.prior_strength,
            )
            self._effects[column] = effect
            residual = residual - np.asarray(
                [effect.means.get(key, 0.0) for key in keys], dtype=float
            )

        if len(prepared.group_columns) > 1:
            interaction_keys = _group_keys(frame, prepared.group_columns)
            self._interaction_effect = _fit_shrunk_effect(
                residual,
                interaction_keys,
                weights,
                prior_strength=self.prior_strength,
            )
            residual = residual - np.asarray(
                [
                    self._interaction_effect.means.get(key, 0.0)
                    for key in interaction_keys
                ],
                dtype=float,
            )
        else:
            self._interaction_effect = _EffectTable(
                {}, {}, {}, _variance_floor(residual)
            )

        self._feature_columns = prepared.feature_columns
        self._feature_medians: dict[str, float] = {}
        self._feature_scales: dict[str, float] = {}
        self._feature_coefficients = np.empty(0, dtype=float)
        self._feature_covariance = np.empty((0, 0), dtype=float)
        if self._feature_columns:
            design = self._numeric_design(frame, fit=True)
            if design.shape[1]:
                # Fit features to the residual left by the hierarchical
                # effects.  Fitting them to ``y - global_mean`` and then
                # adding both effects and features double-counts group signal.
                regularizer = (
                    np.eye(design.shape[1], dtype=float) * self.feature_regularization
                )
                rhs = design.T @ (weights * residual)
                matrix = design.T @ (weights[:, None] * design) + regularizer
                coefficients = np.linalg.solve(matrix, rhs)
                self._feature_coefficients = coefficients
                self._feature_covariance = np.linalg.pinv(matrix)
                residual = residual - design @ coefficients

        effect_parameter_count = (
            len(self._time_effect.means)
            + sum(len(effect.means) for effect in self._effects.values())
            + len(self._interaction_effect.means)
            + self._feature_coefficients.size
        )
        residual_dof = max(float(weights.sum()) - effect_parameter_count - 1.0, 1.0)
        self._noise_variance = max(
            float(np.dot(residual**2, weights) / residual_dof),
            _variance_floor(y),
        )
        hyperparameters = (
            ("discount", repr(self.discount)),
            ("horizon_days", repr(self.horizon_days)),
            ("min_time_observations", repr(self.min_time_observations)),
            ("prior_strength", repr(self.prior_strength)),
            ("feature_regularization", repr(self.feature_regularization)),
            ("strict_future", repr(self.strict_future)),
        )
        self._model_parameter_hash = _sha256_payload(
            {
                "model_name": "dynamic_hierarchical_baseline",
                "model_version": "3",
                "hyperparameters": hyperparameters,
                "global_mean": self._global_mean,
                "time_effect": _effect_payload(self._time_effect),
                "effects": {
                    column: _effect_payload(effect)
                    for column, effect in sorted(self._effects.items())
                },
                "interaction_effect": _effect_payload(self._interaction_effect),
                "feature_columns": self._feature_columns,
                "feature_medians": self._feature_medians,
                "feature_scales": self._feature_scales,
                "feature_coefficients": self._feature_coefficients,
                "feature_covariance": self._feature_covariance,
                "noise_variance": self._noise_variance,
            }
        )
        self.provenance = ResearchModelProvenance(
            model_name="dynamic_hierarchical_baseline",
            model_version="3",
            fit_cutoff=prepared.cutoff,
            fit_rows=len(frame),
            rejected_future_event_rows=prepared.rejected_future_events,
            rejected_future_label_rows=prepared.rejected_future_labels,
            rejected_missing_label_rows=prepared.rejected_missing_labels,
            label_column=self.label_column,
            feature_columns=self._feature_columns,
            group_columns=prepared.group_columns,
            seed=None,
            hyperparameters=hyperparameters,
            assumptions=(
                "strict pre-cutoff fold labels",
                "recency-weighted additive effects",
                "empirical-Bayes partial pooling",
                "unseen groups use the population prior",
                "effect variances use an independent additive approximation",
                "prediction_std is predictive standard deviation for the fitted horizon",
                "research-only; no deployment authorization",
            ),
            time_column=self.time_column,
            event_id_column=self.event_id_column,
            label_available_column=self.label_available_column,
            strict_future=self.strict_future,
            uncertainty_semantics=_uncertainty_semantics(self.horizon_days),
            uncertainty_horizon_days=self.horizon_days,
            training_data_hash=prepared.training_data_hash,
            model_parameter_hash=self._model_parameter_hash,
        )
        self._fitted = True

    def _numeric_design(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        if not self._feature_columns:
            return np.empty((len(frame), 0), dtype=float)
        values: list[np.ndarray] = []
        for column in self._feature_columns:
            numeric_series = cast(
                pd.Series, pd.to_numeric(_column(frame, column), errors="coerce")
            )
            numeric = np.asarray(numeric_series, dtype=float)
            if fit:
                finite = numeric[np.isfinite(numeric)]
                median = float(np.median(finite)) if len(finite) else 0.0
                scale = float(np.std(finite)) if len(finite) > 1 else 1.0
                self._feature_medians[column] = median
                self._feature_scales[column] = max(scale, np.finfo(float).eps)
            median = self._feature_medians[column]
            scale = self._feature_scales[column]
            numeric = np.where(np.isfinite(numeric), numeric, median)
            values.append((numeric - median) / scale)
        return np.column_stack(values) if values else np.empty((len(frame), 0))

    def _check_prediction_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("prediction data must be a non-empty DataFrame")
        frame = data.copy(deep=True)
        if frame.columns.duplicated().any():
            raise ValueError("prediction data contains duplicate column names")
        if self.time_column not in frame.columns:
            raise ValueError(f"prediction data is missing {self.time_column!r}")
        frame[self.time_column] = _datetime_series(
            _column(frame, self.time_column), name=self.time_column
        )
        _check_prediction_horizon(frame, self.horizon_days)
        _reject_leakage_columns(
            frame.columns, label_column=self.label_column, context="prediction data"
        )
        _reject_prediction_leakage_columns(
            frame.columns, label_column=self.label_column, context="prediction data"
        )
        if (
            self.event_id_column is not None
            and self.event_id_column not in frame.columns
            and self.event_id_column != "event_id"
        ):
            raise ValueError(
                f"prediction data is missing event id column {self.event_id_column!r}"
            )
        ids = _event_keys(
            frame, time_column=self.time_column, event_id_column=self.event_id_column
        )
        if ids.duplicated().any():
            raise ValueError("prediction data contains duplicate events")
        if (frame[self.time_column] < self.provenance.fit_cutoff).any():
            raise ValueError("prediction rows must be on or after the fit cutoff")
        for column in self._feature_columns:
            if column not in frame.columns:
                raise ValueError(f"prediction data is missing feature {column!r}")
        for column in self._effects:
            if column not in frame.columns:
                frame[column] = "__UNKNOWN__"
            frame[column] = frame[column].map(_clean_category)
        return frame

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return means and predictive uncertainty for each future event.

        ``prediction_std`` is the predictive standard deviation: residual
        outcome noise plus the fold-local parameter/effect uncertainty used by
        this baseline.  The interval is a normal approximation, not a
        calibrated guarantee.
        """
        if not self._fitted:
            raise RuntimeError("fit the research baseline before predicting")
        frame = self._check_prediction_frame(data)
        dates = [
            _as_timestamp(value, name=self.time_column).normalize()
            for value in _column(frame, self.time_column)
        ]
        means = np.full(len(frame), self._global_mean, dtype=float)
        variances = np.full(len(frame), self._noise_variance, dtype=float)
        seen_group_count = np.zeros(len(frame), dtype=int)
        for position, key in enumerate(dates):
            if key in self._time_effect.means:
                means[position] += self._time_effect.means[key]
                variances[position] += self._time_effect.variances.get(key, 0.0)
                seen_group_count[position] += 1
            else:
                variances[position] += self._time_effect.between_var
        group_status: list[str] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            statuses: list[str] = []
            active_keys: list[Any] = []
            for column, effect in self._effects.items():
                key = _clean_category(row[column])
                active_keys.append(key)
                if key in effect.means:
                    means[position] += effect.means[key]
                    variances[position] += effect.variances.get(key, 0.0)
                    seen_group_count[position] += 1
                    statuses.append(f"{column}=observed")
                else:
                    variances[position] += effect.between_var
                    statuses.append(f"{column}=pooled")
            if self._effects and len(active_keys) > 1:
                interaction_key = tuple(active_keys)
                if interaction_key in self._interaction_effect.means:
                    means[position] += self._interaction_effect.means[interaction_key]
                    variances[position] += self._interaction_effect.variances.get(
                        interaction_key, 0.0
                    )
                    statuses.append("interaction=observed")
                else:
                    variances[position] += self._interaction_effect.between_var
                    statuses.append("interaction=pooled")
            group_status.append(";".join(statuses) if statuses else "global")

        if self._feature_coefficients.size:
            design = self._numeric_design(frame, fit=False)
            means += design @ self._feature_coefficients
            variances += np.maximum(
                np.einsum("ij,jk,ik->i", design, self._feature_covariance, design)
                * self._noise_variance,
                0.0,
            )

        std = np.sqrt(np.maximum(variances, np.finfo(float).tiny))
        lower = means - 1.96 * std
        upper = means + 1.96 * std
        result = pd.DataFrame(
            {
                "event_id": self._prediction_ids(frame),
                "prediction_mean": means,
                "prediction_std": std,
                "lower_95": lower,
                "upper_95": upper,
                "probability_positive": _normal_probability_positive(means, std),
                "observed_group_count": seen_group_count,
                "pooling_status": group_status,
                "uncertainty_semantics": self.provenance.uncertainty_semantics,
                "uncertainty_horizon_days": self.provenance.uncertainty_horizon_days,
                "uncertainty_target": self.label_column,
                "horizon_days": self.horizon_days,
                "model_name": self.provenance.model_name,
                "model_provenance_id": self.provenance.provenance_id,
                "model_provenance_hash": self.provenance.provenance_hash,
                "training_data_hash": self.provenance.training_data_hash,
                "model_parameter_hash": self.provenance.model_parameter_hash,
                "research_only": True,
                "deployment_authorized": False,
            },
            index=data.index,
        )
        result["prediction"] = result["prediction_mean"]
        return result

    def _prediction_ids(self, frame: pd.DataFrame) -> list[str]:
        if self.event_id_column in frame.columns:
            return frame[self.event_id_column].astype(str).str.strip().tolist()
        return [f"row-{position}" for position in range(len(frame))]

    @property
    def _global_mean(self) -> float:
        return float(self.__dict__.get("_fitted_global_mean", 0.0))

    @_global_mean.setter
    def _global_mean(self, value: float) -> None:
        self.__dict__["_fitted_global_mean"] = float(value)


@dataclass(frozen=True, slots=True)
class _NumericSpec:
    median: float
    scale: float
    thresholds: tuple[float, ...]


class _DeterministicEncoder:
    """Numeric/categorical basis with stable threshold (stump) columns."""

    def __init__(self, *, max_bins: int, min_samples_leaf: int) -> None:
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.numeric: dict[str, _NumericSpec] = {}
        self.categorical: dict[str, tuple[str, ...]] = {}
        self.feature_columns: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
        self.feature_columns = tuple(feature_columns)
        for column in self.feature_columns:
            numeric_series = cast(
                pd.Series, pd.to_numeric(_column(frame, column), errors="coerce")
            )
            numeric = np.asarray(numeric_series, dtype=float)
            finite = numeric[np.isfinite(numeric)]
            if len(finite) == len(numeric) or pd.api.types.is_numeric_dtype(
                _column(frame, column)
            ):
                median = float(np.median(finite)) if len(finite) else 0.0
                scale = float(np.std(finite)) if len(finite) > 1 else 1.0
                thresholds: list[float] = []
                if len(np.unique(finite)) > 1 and self.max_bins > 1:
                    quantiles = np.linspace(
                        1.0 / self.max_bins,
                        (self.max_bins - 1.0) / self.max_bins,
                        self.max_bins - 1,
                    )
                    for threshold in np.unique(np.quantile(finite, quantiles)):
                        left = int(np.sum(finite <= threshold))
                        right = int(np.sum(finite > threshold))
                        if (
                            left >= self.min_samples_leaf
                            and right >= self.min_samples_leaf
                        ):
                            thresholds.append(float(threshold))
                self.numeric[column] = _NumericSpec(
                    median=median,
                    scale=max(scale, np.finfo(float).eps),
                    thresholds=tuple(thresholds),
                )
            else:
                values = frame[column].map(_clean_category)
                self.categorical[column] = tuple(sorted(set(values), key=str))

    def pooling_status(self, frame: pd.DataFrame) -> list[str]:
        statuses: list[str] = []
        for _, row in frame.iterrows():
            unknown = any(
                column in self.categorical
                and _clean_category(row[column]) not in self.categorical[column]
                for column in self.feature_columns
            )
            statuses.append(
                "unknown_categories_pool_to_intercept"
                if unknown
                else "observed_feature_levels"
            )
        return statuses

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for column in self.feature_columns:
            if column in self.numeric:
                spec = self.numeric[column]
                numeric_series = cast(
                    pd.Series, pd.to_numeric(_column(frame, column), errors="coerce")
                )
                numeric = np.asarray(numeric_series, dtype=float)
                numeric = np.where(np.isfinite(numeric), numeric, spec.median)
                columns.append(((numeric - spec.median) / spec.scale).reshape(-1, 1))
                for threshold in spec.thresholds:
                    columns.append((numeric > threshold).astype(float).reshape(-1, 1))
            else:
                levels = self.categorical[column]
                values = frame[column].map(_clean_category)
                for level in levels:
                    columns.append(
                        (values == level).astype(float).to_numpy().reshape(-1, 1)
                    )
        if not columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(columns).astype(float)


class DeterministicRegularizedTabularBaseline(_ResearchOnlyMixin):
    """Dependency-free regularized tabular/tree-like research baseline.

    The design contains standardized numeric columns, deterministic threshold
    indicators at training quantiles, and sorted one-hot categorical columns.
    A ridge fit regularizes all non-intercept basis columns.  No RNG, external
    model library, graph library, or mutable global state is used.
    """

    def __init__(
        self,
        *,
        label_column: str = "factor_residual_return",
        time_column: str = "entry_date",
        event_id_column: str | None = "event_id",
        group_columns: Sequence[str] = ("member", "sector"),
        horizon_days: int = 90,
        alpha: float = 1.0,
        max_bins: int = 5,
        min_samples_leaf: int = 2,
        feature_columns: Sequence[str] | None = None,
        label_available_column: str | None = None,
        strict_future: bool = False,
        seed: int | None = 0,
    ) -> None:
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("alpha must be positive and finite")
        if max_bins < 2:
            raise ValueError("max_bins must be at least two")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")
        self.label_column = label_column
        self.time_column = time_column
        self.event_id_column = event_id_column
        self.group_columns = _validate_group_columns(group_columns)
        self.horizon_days = int(horizon_days)
        self.alpha = float(alpha)
        self.max_bins = int(max_bins)
        self.min_samples_leaf = int(min_samples_leaf)
        self.requested_feature_columns = (
            tuple(str(value) for value in feature_columns)
            if feature_columns is not None
            else None
        )
        self.label_available_column = label_available_column
        self.strict_future = bool(strict_future)
        self.seed = seed
        self._fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        *,
        cutoff: object | None = None,
        fit_cutoff: object | None = None,
    ) -> DeterministicRegularizedTabularBaseline:
        if cutoff is not None and fit_cutoff is not None:
            raise ValueError("pass only one of cutoff and fit_cutoff")
        self._fitted = False
        selected_cutoff = cutoff if cutoff is not None else fit_cutoff
        prepared = _prepare_training(
            data,
            cutoff=selected_cutoff,
            label_column=self.label_column,
            time_column=self.time_column,
            event_id_column=self.event_id_column,
            group_columns=self.group_columns,
            feature_columns=self.requested_feature_columns,
            horizon_days=self.horizon_days,
            label_available_column=self.label_available_column,
            strict_future=self.strict_future,
            reserve_group_columns=False,
        )
        self._fit_prepared(prepared)
        return self

    def fit_fold(
        self, data: pd.DataFrame, cutoff: object
    ) -> DeterministicRegularizedTabularBaseline:
        self._fitted = False
        prepared = _prepare_training(
            data,
            cutoff=cutoff,
            label_column=self.label_column,
            time_column=self.time_column,
            event_id_column=self.event_id_column,
            group_columns=self.group_columns,
            feature_columns=self.requested_feature_columns,
            horizon_days=self.horizon_days,
            label_available_column=self.label_available_column,
            strict_future=False,
            reserve_group_columns=False,
        )
        self._fit_prepared(prepared)
        return self

    def _fit_prepared(self, prepared: _PreparedTraining) -> None:
        frame = prepared.frame
        y = frame[self.label_column].to_numpy(dtype=float)
        self._encoder = _DeterministicEncoder(
            max_bins=self.max_bins, min_samples_leaf=self.min_samples_leaf
        )
        self._encoder.fit(frame, prepared.feature_columns)
        basis = self._encoder.transform(frame)
        x = np.column_stack([np.ones(len(frame), dtype=float), basis])
        regularizer = np.eye(x.shape[1], dtype=float) * self.alpha
        regularizer[0, 0] = 0.0
        matrix = x.T @ x + regularizer
        rhs = x.T @ y
        self._coefficients = np.linalg.solve(matrix, rhs)
        self._precision = np.linalg.pinv(matrix)
        residual = y - x @ self._coefficients
        degrees = max(len(y) - x.shape[1], 1)
        self._noise_variance = max(
            float(np.dot(residual, residual) / degrees), _variance_floor(y)
        )
        self._feature_columns = prepared.feature_columns
        self._model_parameter_hash = _sha256_payload(
            {
                "model_name": "deterministic_regularized_tabular_baseline",
                "model_version": "3",
                "hyperparameters": {
                    "alpha": self.alpha,
                    "horizon_days": self.horizon_days,
                    "max_bins": self.max_bins,
                    "min_samples_leaf": self.min_samples_leaf,
                    "strict_future": self.strict_future,
                    "seed": self.seed,
                },
                "feature_columns": self._feature_columns,
                "numeric_specs": {
                    column: {
                        "median": spec.median,
                        "scale": spec.scale,
                        "thresholds": spec.thresholds,
                    }
                    for column, spec in sorted(self._encoder.numeric.items())
                },
                "categorical_levels": self._encoder.categorical,
                "coefficients": self._coefficients,
                "precision": self._precision,
                "noise_variance": self._noise_variance,
            }
        )
        self.provenance = ResearchModelProvenance(
            model_name="deterministic_regularized_tabular_baseline",
            model_version="3",
            fit_cutoff=prepared.cutoff,
            fit_rows=len(frame),
            rejected_future_event_rows=prepared.rejected_future_events,
            rejected_future_label_rows=prepared.rejected_future_labels,
            rejected_missing_label_rows=prepared.rejected_missing_labels,
            label_column=self.label_column,
            feature_columns=self._feature_columns,
            group_columns=prepared.group_columns,
            seed=self.seed,
            hyperparameters=(
                ("alpha", repr(self.alpha)),
                ("horizon_days", repr(self.horizon_days)),
                ("max_bins", repr(self.max_bins)),
                ("min_samples_leaf", repr(self.min_samples_leaf)),
                ("strict_future", repr(self.strict_future)),
            ),
            assumptions=(
                "strict pre-cutoff fold labels",
                "ridge regularization",
                "deterministic quantile threshold basis",
                "unseen categories map to the intercept (pooled)",
                "prediction_std is predictive standard deviation for the fitted horizon",
                "research-only; no deployment authorization",
            ),
            time_column=self.time_column,
            event_id_column=self.event_id_column,
            label_available_column=self.label_available_column,
            strict_future=self.strict_future,
            uncertainty_semantics=_uncertainty_semantics(self.horizon_days),
            uncertainty_horizon_days=self.horizon_days,
            training_data_hash=prepared.training_data_hash,
            model_parameter_hash=self._model_parameter_hash,
        )
        self._fitted = True

    def _check_prediction_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("prediction data must be a non-empty DataFrame")
        frame = data.copy(deep=True)
        if frame.columns.duplicated().any():
            raise ValueError("prediction data contains duplicate column names")
        if self.time_column not in frame.columns:
            raise ValueError(f"prediction data is missing {self.time_column!r}")
        frame[self.time_column] = _datetime_series(
            _column(frame, self.time_column), name=self.time_column
        )
        _check_prediction_horizon(frame, self.horizon_days)
        _reject_leakage_columns(
            frame.columns, label_column=self.label_column, context="prediction data"
        )
        _reject_prediction_leakage_columns(
            frame.columns, label_column=self.label_column, context="prediction data"
        )
        if (
            self.event_id_column is not None
            and self.event_id_column not in frame.columns
            and self.event_id_column != "event_id"
        ):
            raise ValueError(
                f"prediction data is missing event id column {self.event_id_column!r}"
            )
        ids = _event_keys(
            frame, time_column=self.time_column, event_id_column=self.event_id_column
        )
        if ids.duplicated().any():
            raise ValueError("prediction data contains duplicate events")
        if (frame[self.time_column] < self.provenance.fit_cutoff).any():
            raise ValueError("prediction rows must be on or after the fit cutoff")
        for column in self._feature_columns:
            if column not in frame.columns:
                raise ValueError(f"prediction data is missing feature {column!r}")
        return frame

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict with a predictive, normal-approximation uncertainty interval."""
        if not self._fitted:
            raise RuntimeError("fit the research baseline before predicting")
        frame = self._check_prediction_frame(data)
        basis = self._encoder.transform(frame)
        x = np.column_stack([np.ones(len(frame), dtype=float), basis])
        means = x @ self._coefficients
        leverage = np.maximum(np.einsum("ij,jk,ik->i", x, self._precision, x), 0.0)
        std = np.sqrt(
            np.maximum(self._noise_variance * (1.0 + leverage), np.finfo(float).tiny)
        )
        result = pd.DataFrame(
            {
                "event_id": self._prediction_ids(frame),
                "prediction_mean": means,
                "prediction_std": std,
                "lower_95": means - 1.96 * std,
                "upper_95": means + 1.96 * std,
                "probability_positive": _normal_probability_positive(means, std),
                "pooling_status": self._encoder.pooling_status(frame),
                "uncertainty_semantics": self.provenance.uncertainty_semantics,
                "uncertainty_horizon_days": self.provenance.uncertainty_horizon_days,
                "uncertainty_target": self.label_column,
                "horizon_days": self.horizon_days,
                "model_name": self.provenance.model_name,
                "model_provenance_id": self.provenance.provenance_id,
                "model_provenance_hash": self.provenance.provenance_hash,
                "training_data_hash": self.provenance.training_data_hash,
                "model_parameter_hash": self.provenance.model_parameter_hash,
                "research_only": True,
                "deployment_authorized": False,
            },
            index=data.index,
        )
        result["prediction"] = result["prediction_mean"]
        return result

    def _prediction_ids(self, frame: pd.DataFrame) -> list[str]:
        if self.event_id_column in frame.columns:
            return frame[self.event_id_column].astype(str).str.strip().tolist()
        return [f"row-{position}" for position in range(len(frame))]


# Short aliases are useful in notebooks and keep the public name independent
# of the implementation detail that the nonlinear basis is tree-like.
DeterministicTabularBaseline = DeterministicRegularizedTabularBaseline
RegularizedTabularBaseline = DeterministicRegularizedTabularBaseline


__all__ = [
    "DeterministicRegularizedTabularBaseline",
    "DeterministicTabularBaseline",
    "DynamicHierarchicalBaseline",
    "RegularizedTabularBaseline",
    "ResearchModelProvenance",
    "ResearchOnlyError",
]
