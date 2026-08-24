"""Leakage-safe affine calibration for continuous return forecasts.

The calibrator is intentionally small and dependency-light.  It fits

``calibrated_mean = intercept + slope * raw_prediction``

on finite, mature observations that were public and fully labeled by an
optional historical cutoff.  A frozen fit reports predictive uncertainty from
the residual variance and the affine parameter leverage, then derives
``P(calibrated_return > 0)`` under a normal predictive approximation.

This is a calibration layer, not a model-selection or portfolio layer.  It
never queries prices and it never infers a target from an incomplete row.
Appending observations after a supplied cutoff therefore cannot change an
earlier fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erf, isfinite, sqrt
from numbers import Real
from typing import Any, cast

import numpy as np
import pandas as pd

_EVENT_DATE_CANDIDATES = (
    "as_of",
    "as_of_date",
    "event_date",
    "prediction_date",
    "disclosure_date",
    "entry_date",
    "timestamp",
)
_LABEL_DATE_CANDIDATES = (
    "target_available_date",
    "label_window_end",
    "label_available_date",
    "outcome_available_date",
    "outcome_date",
    "label_exit_date",
    "exit_date",
)


def _safe_float(value: object, *, name: str = "value") -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _safe_int(value: object, *, name: str = "value") -> int:
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _timestamp(value: object, *, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(cast(Any, value))
    if pd.isna(parsed):
        raise ValueError(f"{name} is missing")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return cast(pd.Timestamp, parsed).normalize()


def _datetime_series(values: object, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(cast(Any, values), errors="coerce")
    if isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.Series(parsed, index=getattr(values, "index", None))
    parsed_series = cast(pd.Series, parsed)
    if isinstance(parsed_series.dtype, pd.DatetimeTZDtype):
        parsed_series = parsed_series.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed_series.dt.normalize()


def _normal_cdf_zero(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """P(N(mean, std**2) > 0), including the deterministic std==0 case."""
    result = np.full(mean.shape, 0.5, dtype=float)
    positive_std = std > 0.0
    if positive_std.any():
        z = mean[positive_std] / std[positive_std]
        result[positive_std] = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    deterministic = ~positive_std
    result[deterministic & (mean > 0.0)] = 1.0
    result[deterministic & (mean < 0.0)] = 0.0
    return result


def _normal_probability_positive(mean: float, std: float) -> float:
    return _safe_float(_normal_cdf_zero(np.asarray([mean]), np.asarray([std]))[0])


@dataclass(frozen=True, slots=True)
class CalibratedPrediction:
    """One affine-calibrated forecast and its predictive diagnostics."""

    raw_prediction: float
    expected_value: float
    predictive_std: float
    probability_positive: float

    @property
    def mean(self) -> float:
        """Alias for the calibrated predictive mean."""
        return self.expected_value

    @property
    def expected_net_return(self) -> float:
        """Descriptive alias used by the forecast architecture."""
        return self.expected_value

    @property
    def uncertainty(self) -> float:
        """Predictive standard deviation."""
        return self.predictive_std

    @property
    def std(self) -> float:
        """Short alias for :attr:`predictive_std`."""
        return self.predictive_std


@dataclass(frozen=True, slots=True)
class AffineCalibration:
    """Frozen result of a leakage-safe affine calibration fit."""

    intercept: float
    slope: float
    residual_std: float
    predictive_std: float
    n_observations: int
    degrees_of_freedom: int
    fit_cutoff: pd.Timestamp | None
    prediction_column: str
    target_column: str
    event_date_column: str | None = None
    label_date_column: str | None = None
    rejected_future_rows: int = 0
    rejected_incomplete_rows: int = 0
    model_id: str = "leakage_safe_affine_v1"
    _xtx_pinv: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (0.0, 0.0),
    )

    def __post_init__(self) -> None:
        values = (
            self.intercept,
            self.slope,
            self.residual_std,
            self.predictive_std,
        )
        if not all(
            isfinite(_safe_float(value)) and _safe_float(value) >= 0.0
            for value in values[2:]
        ):
            raise ValueError(
                "calibration uncertainty values must be finite and non-negative"
            )
        if not isfinite(_safe_float(self.intercept)) or not isfinite(
            _safe_float(self.slope)
        ):
            raise ValueError("calibration coefficients must be finite")
        if (
            isinstance(self.n_observations, bool)
            or not isinstance(self.n_observations, (int, np.integer))
            or self.n_observations < 3
        ):
            raise ValueError("n_observations must be at least 3")
        if (
            isinstance(self.degrees_of_freedom, bool)
            or not isinstance(self.degrees_of_freedom, (int, np.integer))
            or self.degrees_of_freedom <= 0
        ):
            raise ValueError("degrees_of_freedom must be positive")

    @property
    def coefficient(self) -> float:
        """Alias for the affine slope."""
        return self.slope

    @property
    def uncertainty(self) -> float:
        """Base predictive standard deviation at a zero-leverage point."""
        return self.predictive_std

    @property
    def probability_positive_at_zero(self) -> float:
        return _normal_probability_positive(self.intercept, self.predictive_std)

    def calibrated_mean(self, raw_prediction: Any) -> Any:
        """Apply the frozen affine map to scalar or array-like predictions."""
        values = _finite_predictions(raw_prediction)
        result = self.intercept + self.slope * values
        return _restore_prediction_type(raw_prediction, result)

    def predictive_uncertainty(self, raw_prediction: Any) -> Any:
        """Return the standard deviation of the calibrated prediction.

        The fit stores ``(X'X)^+`` so parameter uncertainty grows away from the
        observed prediction support.  ``predictive_std`` includes one residual
        innovation variance and is therefore a predictive, not merely fitted,
        uncertainty.
        """
        values = _finite_predictions(raw_prediction)
        pinv = np.asarray(self._xtx_pinv, dtype=float)
        leverage = pinv[0, 0] + 2.0 * values * pinv[0, 1] + values * values * pinv[1, 1]
        leverage = np.maximum(leverage, 0.0)
        # ``predictive_std`` is residual_std * sqrt(1 + leverage) when the fit
        # contains residual information.  A perfect fit is deterministic and
        # therefore has zero predictive uncertainty.
        if self.residual_std == 0.0:
            result = np.zeros_like(values, dtype=float)
        else:
            result = self.residual_std * np.sqrt(1.0 + leverage)
        return _restore_prediction_type(raw_prediction, result)

    def probability_positive(self, raw_prediction: Any) -> Any:
        """Return the calibrated normal predictive probability of a positive target."""
        values = _finite_predictions(raw_prediction)
        mean = self.intercept + self.slope * values
        std = np.asarray(self.predictive_uncertainty(values), dtype=float)
        result = _normal_cdf_zero(mean, std)
        return _restore_prediction_type(raw_prediction, result)

    def predict_one(self, raw_prediction: object) -> CalibratedPrediction:
        """Return one typed forecast diagnostic."""
        if isinstance(raw_prediction, bool) or not isinstance(raw_prediction, Real):
            raise TypeError("raw_prediction must be a real number")
        raw = _safe_float(raw_prediction, name="raw_prediction")
        if not isfinite(raw):
            raise ValueError("raw_prediction must be finite")
        mean = _safe_float(self.calibrated_mean(raw), name="calibrated_mean")
        std = _safe_float(self.predictive_uncertainty(raw), name="predictive_std")
        return CalibratedPrediction(
            raw, mean, std, _normal_probability_positive(mean, std)
        )

    def predict(self, raw_prediction: Any) -> Any:
        """Return typed scalar output or a DataFrame of forecast diagnostics."""
        if np.isscalar(raw_prediction):
            return self.predict_one(raw_prediction)
        values = _finite_predictions(raw_prediction)
        mean = self.intercept + self.slope * values
        std = np.asarray(self.predictive_uncertainty(values), dtype=float)
        probability = _normal_cdf_zero(mean, std)
        if isinstance(raw_prediction, pd.Series):
            index = raw_prediction.index
        else:
            index = None
        return pd.DataFrame(
            {
                "raw_prediction": values,
                "expected_value": mean,
                "predictive_std": std,
                "probability_positive": probability,
            },
            index=index,
        )

    def predict_frame(
        self,
        frame: pd.DataFrame,
        *,
        prediction_column: str | None = None,
        expected_column: str = "calibrated_mean",
        uncertainty_column: str = "predictive_std",
        probability_column: str = "probability_positive",
    ) -> pd.DataFrame:
        """Append calibrated mean, uncertainty, and positive probability."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        column = prediction_column or self.prediction_column
        if column not in frame.columns:
            raise ValueError(f"prediction column {column!r} is missing")
        values = _finite_predictions(frame[column])
        means = self.intercept + self.slope * values
        std = np.asarray(self.predictive_uncertainty(values), dtype=float)
        output = frame.copy(deep=True)
        output[expected_column] = means
        output[uncertainty_column] = std
        output[probability_column] = _normal_cdf_zero(means, std)
        return output

    # A class-level convenience makes the fitted value discoverable whether a
    # caller thinks of calibration as a class or as a fitting function.
    @classmethod
    def fit(cls, data: pd.DataFrame, *args: Any, **kwargs: Any) -> AffineCalibration:
        return fit_affine_calibration(data, *args, **kwargs)


@dataclass(frozen=True, slots=True)
class AffineCalibrator:
    """Configuration object for repeated fold-local affine fits."""

    prediction_column: str = "prediction"
    target_column: str = "net_return_pct"
    event_date_column: str | None = None
    label_date_column: str | None = None
    complete_column: str | None = "target_available"
    model_id: str = "leakage_safe_affine_v1"

    def fit(
        self,
        data: pd.DataFrame,
        *,
        fit_cutoff: object | None = None,
        as_of_date: object | None = None,
        as_of: object | None = None,
        prediction_column: str | None = None,
        target_column: str | None = None,
        event_date_column: str | None = None,
        label_date_column: str | None = None,
        complete_column: str | None = None,
    ) -> AffineCalibration:
        return fit_affine_calibration(
            data,
            fit_cutoff=fit_cutoff,
            as_of_date=as_of_date,
            as_of=as_of,
            prediction_column=prediction_column or self.prediction_column,
            target_column=target_column or self.target_column,
            event_date_column=(
                event_date_column
                if event_date_column is not None
                else self.event_date_column
            ),
            label_date_column=(
                label_date_column
                if label_date_column is not None
                else self.label_date_column
            ),
            complete_column=(
                complete_column if complete_column is not None else self.complete_column
            ),
            model_id=self.model_id,
        )


def _finite_predictions(values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("predictions must be numeric") from exc
    if np.isinf(array).any():
        raise ValueError("predictions must be finite or NaN")
    if np.isnan(array).any():
        raise ValueError("predictions passed to predict must be finite")
    return array


def _restore_prediction_type(original: Any, values: np.ndarray) -> Any:
    if np.isscalar(original):
        return _safe_float(np.asarray(values))
    if isinstance(original, pd.Series):
        return pd.Series(
            np.asarray(values, dtype=float), index=original.index, name=original.name
        )
    if isinstance(original, pd.Index):
        return pd.Index(np.asarray(values, dtype=float), name=original.name)
    return np.asarray(values, dtype=float)


def _resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    *,
    role: str,
) -> str | None:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"{role} column {requested!r} is missing")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _complete_mask(frame: pd.DataFrame, complete_column: str | None) -> pd.Series:
    if complete_column is None:
        return pd.Series(True, index=frame.index, dtype=bool)
    if complete_column not in frame.columns:
        # A target column with no explicit status is accepted for compact
        # fixtures.  If a status exists, false/missing is always respected.
        return pd.Series(True, index=frame.index, dtype=bool)
    values = cast(pd.Series, frame[complete_column])
    return cast(pd.Series, values.eq(True).fillna(False))


def _fit_arrays(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, int, tuple[tuple[float, float], tuple[float, float]]]:
    if len(x) < 3:
        raise ValueError(
            "at least 3 finite mature observations are required for calibration"
        )
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("calibration observations must be equally sized vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("calibration observations must be finite")
    design = np.column_stack((np.ones(len(x), dtype=float), x))
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError(
            "calibration predictions must span at least two distinct values"
        )
    # The rank check is deliberate: ``pinv`` is used for numerical stability,
    # not as a way to manufacture coefficients for a constant prediction
    # column.  Deriving the leverage matrix from the same pseudoinverse also
    # avoids squaring the design condition number in ``X.T @ X``.
    design_pinv = np.linalg.pinv(design, rcond=None)
    coefficients = design_pinv @ y
    intercept = _safe_float(coefficients[0], name="intercept")
    slope = _safe_float(coefficients[1], name="slope")
    residuals = y - design @ coefficients
    dof = len(x) - 2
    residual_variance = _safe_float(np.dot(residuals, residuals) / dof)
    residual_std = sqrt(max(residual_variance, 0.0))
    xtx_pinv = design_pinv @ design_pinv.T
    if not np.isfinite(xtx_pinv).all() or not isfinite(residual_std):
        raise ValueError("calibration fit produced non-finite uncertainty")
    return (
        intercept,
        slope,
        residual_std,
        dof,
        (
            (
                _safe_float(xtx_pinv[0, 0]),
                _safe_float(xtx_pinv[0, 1]),
            ),
            (
                _safe_float(xtx_pinv[1, 0]),
                _safe_float(xtx_pinv[1, 1]),
            ),
        ),
    )


def fit_affine_calibration(
    data: pd.DataFrame,
    targets: Sequence[float] | pd.Series | np.ndarray | None = None,
    *,
    prediction_column: str = "prediction",
    target_column: str = "net_return_pct",
    fit_cutoff: object | None = None,
    as_of_date: object | None = None,
    as_of: object | None = None,
    event_date_column: str | None = None,
    label_date_column: str | None = None,
    complete_column: str | None = "target_available",
    model_id: str = "leakage_safe_affine_v1",
) -> AffineCalibration:
    """Fit an affine calibration using only mature, pre-cutoff observations.

    ``data`` normally contains prediction and target columns.  For a compact
    array API, ``targets`` may be supplied as a second vector and ``data`` is
    then interpreted as the prediction vector.  That form has no dates and is
    accepted only without a cutoff; callers must make it fold-local before
    fitting.

    A cutoff filters both event/public time and label availability.  Future
    rows are excluded rather than allowed to alter the fit.  Existing
    ``target_available``/``window_complete`` markers are also honored.
    """
    if (
        as_of_date is not None
        and as_of is not None
        and _timestamp(as_of_date, name="as_of_date") != _timestamp(as_of, name="as_of")
    ):
        raise ValueError("as_of_date and as_of disagree")
    cutoff_alias = as_of_date if as_of_date is not None else as_of
    if fit_cutoff is not None and cutoff_alias is not None:
        if _timestamp(fit_cutoff, name="fit_cutoff") != _timestamp(
            cutoff_alias, name="as_of"
        ):
            raise ValueError("fit_cutoff and as_of disagree")
    cutoff_value = fit_cutoff if fit_cutoff is not None else cutoff_alias
    cutoff = (
        _timestamp(cutoff_value, name="fit_cutoff")
        if cutoff_value is not None
        else None
    )

    event_column: str | None = event_date_column
    label_column: str | None = label_date_column
    if targets is not None:
        if cutoff is not None:
            raise ValueError(
                "a historical cutoff requires event and label date metadata"
            )
        prediction_values = data
        if not isinstance(prediction_values, (pd.Series, np.ndarray, list, tuple)):
            raise TypeError("array-form calibration predictions must be array-like")
        x = np.asarray(prediction_values, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError(
                "predictions and targets must be one-dimensional and equally sized"
            )
        if np.isinf(x).any() or np.isinf(y).any():
            raise ValueError("predictions and targets must be finite or NaN")
        finite = np.isfinite(x) & np.isfinite(y)
        rejected_incomplete = _safe_int(
            (~finite).sum(), name="rejected_incomplete_rows"
        )
        x_fit, y_fit = x[finite], y[finite]
        if len(x_fit) == 0:
            raise ValueError("no finite mature observations remain for calibration")
        intercept, slope, residual_std, dof, xtx_pinv = _fit_arrays(x_fit, y_fit)
        return AffineCalibration(
            intercept=intercept,
            slope=slope,
            residual_std=residual_std,
            predictive_std=residual_std,
            n_observations=len(x_fit),
            degrees_of_freedom=dof,
            fit_cutoff=cutoff,
            prediction_column=prediction_column,
            target_column=target_column,
            rejected_incomplete_rows=rejected_incomplete,
            model_id=model_id,
            _xtx_pinv=xtx_pinv,
        )

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if prediction_column not in data.columns:
        raise ValueError(f"prediction column {prediction_column!r} is missing")
    if target_column not in data.columns:
        raise ValueError(f"target column {target_column!r} is missing")
    frame = data.copy(deep=True)
    x_values = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, frame[prediction_column]), errors="coerce"),
    )
    y_values = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, frame[target_column]), errors="coerce"),
    )
    finite = (
        x_values.notna()
        & y_values.notna()
        & np.isfinite(x_values.to_numpy(dtype=np.float64))
        & np.isfinite(y_values.to_numpy(dtype=np.float64))
    )
    complete = _complete_mask(frame, complete_column)
    eligible = finite & complete
    rejected_incomplete = _safe_int(
        (~(finite & complete)).sum(), name="rejected_incomplete_rows"
    )
    rejected_future = pd.Series(False, index=frame.index, dtype=bool)

    event_column = _resolve_column(
        frame, event_date_column, _EVENT_DATE_CANDIDATES, role="event date"
    )
    label_column = _resolve_column(
        frame, label_date_column, _LABEL_DATE_CANDIDATES, role="label date"
    )
    if cutoff is not None and (event_column is None or label_column is None):
        missing = []
        if event_column is None:
            missing.append("event/public date")
        if label_column is None:
            missing.append("label-availability date")
        raise ValueError(
            "historical calibration cutoff requires " + " and ".join(missing)
        )
    if cutoff is not None and event_column is not None:
        event_dates = _datetime_series(frame[event_column], name=event_column)
        invalid_event = event_dates.isna()
        future_event = event_dates > cutoff
        rejected_future |= future_event | invalid_event
        eligible &= ~future_event & ~invalid_event
    if cutoff is not None and label_column is not None:
        label_dates = _datetime_series(frame[label_column], name=label_column)
        invalid_label = label_dates.isna()
        future_label = label_dates > cutoff
        rejected_future |= future_label | invalid_label
        eligible &= ~future_label & ~invalid_label

    x = x_values.loc[eligible].to_numpy(dtype=float)
    y = y_values.loc[eligible].to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError(
            "no finite mature, pre-cutoff observations remain for calibration"
        )
    intercept, slope, residual_std, dof, xtx_pinv = _fit_arrays(x, y)
    return AffineCalibration(
        intercept=intercept,
        slope=slope,
        residual_std=residual_std,
        predictive_std=residual_std,
        n_observations=len(x),
        degrees_of_freedom=dof,
        fit_cutoff=cutoff,
        prediction_column=prediction_column,
        target_column=target_column,
        event_date_column=event_column,
        label_date_column=label_column,
        rejected_future_rows=_safe_int(
            rejected_future.sum(), name="rejected_future_rows"
        ),
        rejected_incomplete_rows=rejected_incomplete,
        model_id=model_id,
        _xtx_pinv=xtx_pinv,
    )


def calibrate_predictions(
    data: pd.DataFrame,
    calibration: AffineCalibration,
    *,
    prediction_column: str | None = None,
    expected_column: str = "calibrated_mean",
    uncertainty_column: str = "predictive_std",
    probability_column: str = "probability_positive",
) -> pd.DataFrame:
    """Apply a frozen fit to a prediction frame without refitting it."""
    if not isinstance(calibration, AffineCalibration):
        raise TypeError("calibration must be an AffineCalibration instance")
    return calibration.predict_frame(
        data,
        prediction_column=prediction_column,
        expected_column=expected_column,
        uncertainty_column=uncertainty_column,
        probability_column=probability_column,
    )


# Descriptive aliases make the small pure API usable by either noun-oriented
# or verb-oriented callers.
fit_calibration = fit_affine_calibration
affine_calibration = fit_affine_calibration
LeakageSafeAffineCalibrator = AffineCalibrator
FrozenAffineCalibration = AffineCalibration


__all__ = [
    "AffineCalibration",
    "AffineCalibrator",
    "CalibratedPrediction",
    "FrozenAffineCalibration",
    "LeakageSafeAffineCalibrator",
    "affine_calibration",
    "calibrate_predictions",
    "fit_affine_calibration",
    "fit_calibration",
]
