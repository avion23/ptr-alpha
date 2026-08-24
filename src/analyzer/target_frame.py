"""Build executable prediction targets from the repository's gross labels.

This module is a pure boundary between the existing signal label frame and a
future prediction model.  It does not fetch prices or recompute endpoints.
Instead it consumes the mature, endpoint-aligned gross columns already
produced by :mod:`analyzer.signals` and applies :class:`ExecutionCosts` with
exact multiplicative arithmetic.

Rows remain in the returned frame by default.  Incomplete, immature, or
non-finite labels receive ``NaN`` targets rather than a zero or a shortened
return.  Keeping row identity and missingness intact makes it safe to build a
historical snapshot first and filter eligible observations in a later fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from analyzer.execution_costs import (
    DEFAULT_EXECUTION_COSTS,
    ExecutionCosts,
    executable_alpha_pct,
    executable_return_pct,
)

_RETURN_CANDIDATES = ("total_return_pct", "gross_return_pct", "total_return")
_ALPHA_CANDIDATES = (
    "total_spy_alpha_pct",
    "gross_alpha_pct",
    "total_spy_alpha",
)
_MATURITY_CANDIDATES = (
    "label_window_end",
    "outcome_available_date",
    "label_available_date",
    "outcome_date",
)


def _as_naive_timestamp(value: object, *, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(cast(Any, value))
    if pd.isna(parsed):
        raise ValueError(f"{name} is missing")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return cast(pd.Timestamp, parsed).normalize()


def _as_naive_datetime(values: object, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(cast(Any, values), errors="coerce")
    if isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.Series(parsed, index=getattr(values, "index", None))
    parsed_series = cast(pd.Series, parsed)
    if isinstance(parsed_series.dtype, pd.DatetimeTZDtype):
        parsed_series = parsed_series.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed_series.dt.normalize()


def _resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    *,
    role: str,
    required: bool,
) -> str | None:
    if requested is not None:
        if requested not in frame.columns:
            if required:
                raise ValueError(f"{role} column {requested!r} is missing")
            return None
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(f"could not find {role} column; tried {tuple(candidates)!r}")
    return None


def _bool_complete(values: pd.Series) -> pd.Series:
    """Interpret only truthy boolean/numeric completion markers as complete."""
    # ``astype(bool)`` makes the string "False" true.  Equality with True is
    # intentionally conservative and still accepts the bool/numpy-bool values
    # emitted by the signal assembler.
    return values.eq(True).fillna(False)


def mature_label_mask(
    labels: pd.DataFrame,
    *,
    as_of_date: object | None = None,
    as_of: object | None = None,
    maturity_column: str | None = None,
    complete_column: str | None = "window_complete",
    require_maturity_date: bool = False,
) -> pd.Series:
    """Return the rows whose existing labels are known and complete.

    ``window_complete=False`` and missing completion markers are never treated
    as zero-return labels.  When a maturity date exists it must be non-missing;
    with an ``as_of`` cutoff it must also be on or before that cutoff.  A cutoff
    without a maturity column is rejected rather than allowing unknown rows to
    enter a historical target set.  A frame without a maturity column can
    still be used for already-materialized fixtures when no cutoff is given.
    """
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be a pandas DataFrame")
    if (
        as_of_date is not None
        and as_of is not None
        and _as_naive_timestamp(as_of_date, name="as_of_date")
        != _as_naive_timestamp(as_of, name="as_of")
    ):
        raise ValueError("as_of_date and as_of disagree")
    cutoff_value = as_of_date if as_of_date is not None else as_of
    cutoff = (
        _as_naive_timestamp(cutoff_value, name="as_of_date")
        if cutoff_value is not None
        else None
    )

    mask = pd.Series(True, index=labels.index, dtype=bool)
    if complete_column is not None:
        if complete_column in labels.columns:
            mask &= _bool_complete(cast(pd.Series, labels[complete_column]))
        elif require_maturity_date:
            raise ValueError(f"completion column {complete_column!r} is missing")

    # A cutoff is meaningful only when the frame carries the metadata needed
    # to prove that each label was available by that date.  Do not silently
    # treat a compact frame with no maturity column as historical and mature.
    require_maturity = require_maturity_date or cutoff is not None
    maturity = _resolve_column(
        labels,
        maturity_column,
        _MATURITY_CANDIDATES,
        role="label maturity",
        required=require_maturity or maturity_column is not None,
    )
    if maturity is not None:
        dates = _as_naive_datetime(cast(pd.Series, labels[maturity]), name=maturity)
        mask &= dates.notna()
        if cutoff is not None:
            mask &= dates <= cutoff
    elif cutoff is not None:
        raise ValueError("a maturity date is required when an as_of cutoff is set")
    return mask


def _finite_numeric(values: pd.Series, *, name: str) -> tuple[pd.Series, pd.Series]:
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    finite = numeric.notna() & np.isfinite(numeric.to_numpy(dtype=np.float64))
    return numeric.astype(float), cast(pd.Series, finite)


def _resolve_costs(
    costs: ExecutionCosts | None,
    execution_costs: ExecutionCosts | None,
) -> ExecutionCosts:
    if costs is not None and execution_costs is not None and costs != execution_costs:
        raise ValueError("costs and execution_costs disagree")
    resolved = costs if costs is not None else execution_costs
    if resolved is None:
        return DEFAULT_EXECUTION_COSTS
    if not isinstance(resolved, ExecutionCosts):
        raise TypeError("costs must be an ExecutionCosts instance")
    return resolved


def _resolve_horizon(
    horizon_days: int | None,
    horizon: int | None,
) -> int | None:
    if horizon_days is not None and horizon is not None and horizon_days != horizon:
        raise ValueError("horizon_days and horizon disagree")
    resolved = horizon_days if horizon_days is not None else horizon
    if resolved is not None:
        try:
            converted = int(resolved)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("horizon_days must be a positive integer") from exc
        if isinstance(resolved, bool) or converted != resolved or converted <= 0:
            raise ValueError("horizon_days must be a positive integer")
        return converted
    return None


def build_target_frame(
    labels: pd.DataFrame,
    costs: ExecutionCosts | None = None,
    *,
    execution_costs: ExecutionCosts | None = None,
    as_of_date: object | None = None,
    as_of: object | None = None,
    horizon_days: int | None = None,
    horizon: int | None = None,
    gross_return_column: str | None = None,
    gross_alpha_column: str | None = None,
    maturity_column: str | None = None,
    complete_column: str | None = "window_complete",
    net_return_column: str = "net_return_pct",
    net_alpha_column: str = "net_alpha_pct",
    mature_column: str = "label_mature",
    drop_incomplete: bool = False,
    mature_only: bool | None = None,
) -> pd.DataFrame:
    """Return a copy of ``labels`` with executable net-return targets.

    Parameters
    ----------
    labels:
        Existing signal labels.  By default the builder consumes
        ``total_return_pct`` and ``total_spy_alpha_pct``; those are the exact
        endpoint labels produced by the signal pipeline.
    costs:
        Validated proportional endpoint costs.  Costs are applied to the
        security gross return only.  The benchmark implied by the existing
        gross alpha is left unchanged, so ``net_alpha_pct`` is
        ``net_return_pct - (gross_return_pct - gross_alpha_pct)``.
    as_of_date:
        Optional historical information cutoff.  A label is admitted only
        when its existing maturity date is on or before this date.
    drop_incomplete:
        If false (the default), keep every input row and write ``NaN`` target
        values for incomplete/immature labels.  If true, return only mature
        rows.  ``mature_only`` is a readable alias.

    No price lookup or endpoint reconstruction happens here.  In particular,
    a non-mature row with a partial gross value is not shortened or filled.
    """
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be a pandas DataFrame")
    if mature_only is not None:
        if not isinstance(mature_only, bool):
            raise TypeError("mature_only must be a bool")
        if drop_incomplete and mature_only != drop_incomplete:
            raise ValueError("drop_incomplete and mature_only disagree")
        drop_incomplete = mature_only

    resolved_costs = _resolve_costs(costs, execution_costs)
    resolved_horizon = _resolve_horizon(horizon_days, horizon)
    if (
        as_of_date is not None
        and as_of is not None
        and _as_naive_timestamp(as_of_date, name="as_of_date")
        != _as_naive_timestamp(as_of, name="as_of")
    ):
        raise ValueError("as_of_date and as_of disagree")

    frame = labels.copy(deep=True)
    return_column = _resolve_column(
        frame,
        gross_return_column,
        _RETURN_CANDIDATES,
        role="gross return",
        required=True,
    )
    alpha_column = _resolve_column(
        frame,
        gross_alpha_column,
        _ALPHA_CANDIDATES,
        role="gross alpha",
        required=False,
    )
    if return_column is None:  # pragma: no cover - required resolution above
        raise ValueError("a gross return column is required")

    numeric_return, valid_return = _finite_numeric(
        cast(pd.Series, frame[return_column]), name=return_column
    )
    if alpha_column is None:
        numeric_alpha = pd.Series(np.nan, index=frame.index, dtype=float)
        valid_alpha = pd.Series(False, index=frame.index, dtype=bool)
    else:
        numeric_alpha, valid_alpha = _finite_numeric(
            cast(pd.Series, frame[alpha_column]), name=alpha_column
        )

    mature = mature_label_mask(
        frame,
        as_of_date=as_of_date,
        as_of=as_of,
        maturity_column=maturity_column,
        complete_column=complete_column,
        # Existing signals expose label_window_end.  Any explicit cutoff also
        # requires maturity metadata; otherwise the caller cannot establish
        # that the target was known at the requested historical date.
        require_maturity_date=(
            maturity_column is not None
            or as_of_date is not None
            or as_of is not None
            or any(column in frame.columns for column in _MATURITY_CANDIDATES)
        ),
    )
    if resolved_horizon is not None and "horizon_days" in frame.columns:
        horizon_values = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, frame["horizon_days"]), errors="coerce"),
        )
        mature &= horizon_values.eq(resolved_horizon)

    target_valid = mature & valid_return
    net_return = pd.Series(np.nan, index=frame.index, dtype=float)
    if target_valid.any():
        net_return.loc[target_valid] = np.asarray(
            executable_return_pct(
                numeric_return.loc[target_valid].to_numpy(), resolved_costs
            ),
            dtype=float,
        )

    # Alpha needs both existing gross endpoint labels.  It is deliberately not
    # computed from an independently aligned benchmark series.
    target_alpha_valid = target_valid & valid_alpha
    net_alpha = pd.Series(np.nan, index=frame.index, dtype=float)
    if target_alpha_valid.any():
        net_alpha.loc[target_alpha_valid] = np.asarray(
            executable_alpha_pct(
                numeric_return.loc[target_alpha_valid].to_numpy(),
                numeric_alpha.loc[target_alpha_valid].to_numpy(),
                resolved_costs,
            ),
            dtype=float,
        )

    frame[net_return_column] = net_return
    frame[net_alpha_column] = net_alpha
    frame[mature_column] = mature.astype(bool)
    frame["target_available"] = target_valid.astype(bool)
    frame["target_alpha_available"] = target_alpha_valid.astype(bool)

    if drop_incomplete:
        frame = frame.loc[target_valid].copy()
    return frame


def build_executable_target_frame(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Descriptive alias for :func:`build_target_frame`."""
    return build_target_frame(*args, **kwargs)


def build_mature_target_frame(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Build a frame containing mature executable security targets only."""
    kwargs = dict(kwargs)
    kwargs["drop_incomplete"] = True
    return build_target_frame(*args, **kwargs)


# Alternate names used by callers that call the output a training dataset.
build_training_target_frame = build_target_frame
target_frame = build_target_frame


__all__ = [
    "build_executable_target_frame",
    "build_mature_target_frame",
    "build_target_frame",
    "build_training_target_frame",
    "mature_label_mask",
    "target_frame",
]
