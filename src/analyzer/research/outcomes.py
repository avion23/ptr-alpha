"""Point-in-time outcome construction for research experiments.

This module is intentionally not imported by the live signal or validation
paths.  It supplies a small, explicit label builder for experiments that need
an outcome in excess of SPY and other pre-declared factors.

The important boundary is the factor fit.  For an event entered on ``t`` the
asset exposure regression only sees daily observations with dates strictly
before ``t``.  A factor observation on the entry date can contain the entry
day's close-to-close return and is therefore not a pre-entry observation.
Future rows in a price or factor panel consequently cannot change an already
constructed outcome (provided the event's exit date is fixed or its first
available endpoint is unchanged).

Factor inputs are daily *simple returns* by default.  They are converted to
log returns for fitting so that a multi-day factor contribution and the asset
outcome are additive.  ``factor_returns_are_log=True`` is available for data
sets that already store log returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Iterable, Sequence, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FactorFitMetadata:
    """Audit information for one event's point-in-time factor fit."""

    fit_start_date: pd.Timestamp | None
    fit_end_date: pd.Timestamp | None
    factor_columns: tuple[str, ...]
    n_observations: int
    n_parameters: int
    factor_betas: tuple[float, ...]
    intercept: float | None
    residual_std_log: float | None
    fit_is_pre_entry: bool
    model: str = "point_in_time_ols_log_return_v2"
    factor_return_unit: str = "log"
    asset_return_unit: str = "log"


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, frame[name])


def _timestamp(value: object, *, name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains an invalid date") from exc
    if pd.isna(parsed):
        raise ValueError(f"{name} contains a missing date")
    # Comparing tz-aware and tz-naive timestamps is an easy way to accidentally
    # turn a point-in-time filter into an unbounded one.  Normalize aware values
    # to UTC and then remove the timezone consistently throughout this module.
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return cast(pd.Timestamp, parsed)


def _date_index(frame: pd.DataFrame | pd.Series, *, name: str) -> pd.DatetimeIndex:
    # ``utc=True`` also handles a mixed naive/aware object index.  Treating
    # naive timestamps as UTC is the only deterministic interpretation for a
    # panel that has no timezone metadata.
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce", utc=True))
    index = index.tz_localize(None)
    if index.isna().any():
        raise ValueError(f"{name} has a missing index date")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in ascending date order")
    if not index.is_unique:
        raise ValueError(f"{name} index contains duplicate dates")
    return index


def _resolve_column(
    frame: pd.DataFrame, requested: str | None, candidates: Sequence[str], *, role: str
) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"{role} column {requested!r} is missing")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"could not find {role} column; tried {tuple(candidates)!r}")


def _as_price_panel(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        name = str(prices.name or "SPY")
        prices = prices.to_frame(name=name)
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame or Series")
    index = _date_index(prices, name="prices")
    panel = prices.copy()
    panel.index = index
    # A MultiIndex column panel is a database-facing representation used by a
    # few existing price sources.  Research labels deliberately require the
    # unambiguous ticker-column representation rather than guessing which level
    # is a close price.
    if isinstance(panel.columns, pd.MultiIndex):
        raise ValueError("prices must have one ticker per column, not MultiIndex columns")
    panel.columns = [str(column).strip().upper() for column in panel.columns]
    if any(not column for column in panel.columns):
        raise ValueError("prices contains an empty ticker column")
    if len(set(panel.columns)) != len(panel.columns):
        raise ValueError("prices contains duplicate ticker columns")
    panel = cast(pd.DataFrame, panel.apply(pd.to_numeric, errors="coerce"))
    if np.isinf(panel.to_numpy(dtype=float)).any():
        raise ValueError("prices contains an infinite value")
    return panel


def _as_factor_returns(
    factor_returns: pd.DataFrame | None,
    *,
    prices: pd.DataFrame,
    spy_ticker: str,
    factor_returns_are_log: bool,
) -> pd.DataFrame:
    """Normalize the factor panel and guarantee that SPY is a factor."""
    if factor_returns is None:
        spy = _column(prices, spy_ticker).astype(float)
        # Do not forward-fill a missing close.  A filled close manufactures a
        # return using information from the next observed price and can alter
        # a historical label when the panel is refreshed.
        result = cast(
            pd.DataFrame, spy.pct_change(fill_method=None).to_frame(name=spy_ticker)
        )
        factor_returns_are_log = False
    else:
        if not isinstance(factor_returns, pd.DataFrame) or factor_returns.empty:
            raise ValueError("factor_returns must be a non-empty DataFrame")
        index = _date_index(factor_returns, name="factor_returns")
        result = cast(pd.DataFrame, factor_returns.copy())
        result.index = index
        if isinstance(result.columns, pd.MultiIndex):
            raise ValueError("factor_returns must have one factor per column")
        result.columns = [str(column).strip().upper() for column in result.columns]
        if any(not column for column in result.columns):
            raise ValueError("factor_returns contains an empty factor column")
        if len(set(result.columns)) != len(result.columns):
            raise ValueError("factor_returns contains duplicate factor columns")
        result = cast(pd.DataFrame, result.apply(pd.to_numeric, errors="coerce"))

        if spy_ticker not in result.columns:
            spy: pd.Series = cast(
                pd.Series,
                _column(prices, spy_ticker).astype(float).pct_change(
                    fill_method=None
                ),
            )
            if factor_returns_are_log:
                spy = cast(
                    pd.Series,
                    _column(prices, spy_ticker)
                    .where(_column(prices, spy_ticker) > 0)
                    .apply(np.log)
                    .diff(),
                )
            result = cast(
                pd.DataFrame, result.join(spy.rename(spy_ticker), how="outer")
            )

    values = result.to_numpy(dtype=float)
    if not np.isfinite(values[~np.isnan(values)]).all():
        raise ValueError("factor_returns contains an infinite value")
    if factor_returns_are_log:
        log_returns = cast(pd.DataFrame, result.astype(float))
    else:
        if bool((result <= -1.0).any().any()):
            raise ValueError("factor simple returns must be greater than -1")
        log_returns = cast(pd.DataFrame, np.log1p(result.astype(float)))
    log_returns = cast(pd.DataFrame, log_returns.sort_index())
    if not log_returns.index.is_unique:
        raise ValueError("factor_returns contains duplicate dates")
    # At least one SPY factor is mandatory.  Other factors may be entirely
    # missing on a particular date; rows are filtered per event below.
    if spy_ticker not in log_returns.columns:
        raise ValueError(f"factor_returns does not contain required {spy_ticker} factor")
    return log_returns


_FACTOR_LEAKAGE_TOKENS = (
    "future",
    "forward",
    "target",
    "label",
    "outcome",
    "realized",
    "profit",
    "pnl",
    "payoff",
    "exit",
    "sell",
)


def _reject_factor_leakage_columns(columns: Iterable[str]) -> None:
    offending = sorted(
        str(column)
        for column in columns
        if any(token in str(column).strip().lower() for token in _FACTOR_LEAKAGE_TOKENS)
    )
    if offending:
        raise ValueError(
            "factor_returns contains outcome/future columns that cannot be used "
            f"as factors: {offending}"
        )


def _event_identity(
    row: pd.Series,
    *,
    index: int,
    event_id_column: str | None,
    entry_column: str,
    ticker_column: str,
) -> str:
    if event_id_column is not None:
        value = cast(Any, row[event_id_column])
        if bool(pd.isna(value)) or not str(value).strip():
            raise ValueError(f"event row {index} has a missing event id")
        return str(value).strip()
    # Without an explicit transaction id, two events with the same execution
    # date and ticker cannot be distinguished safely.  Rejecting the collision
    # is preferable to silently double counting it.
    parts = [
        _timestamp(cast(Any, row[entry_column]), name=entry_column).isoformat(),
        str(row[ticker_column]).strip().upper(),
    ]
    for candidate in (
        "transaction_id",
        "member",
        "owner_code",
        "transaction_type",
    ):
        if candidate in row.index:
            value = row[candidate]
            parts.append(
                "__UNKNOWN__" if bool(pd.isna(value)) else str(value).strip().upper()
            )
    return "|".join(parts)


def _first_at_or_after(index: pd.DatetimeIndex, target: pd.Timestamp) -> int | None:
    values = index.to_numpy(dtype="datetime64[ns]")
    position = int(
        np.searchsorted(values, np.datetime64(target.to_datetime64()), side="left")
    )
    return position if position < len(values) else None


def _ols_factor_fit(
    asset_log_returns: pd.Series,
    factor_log_returns: pd.DataFrame,
    *,
    entry_date: pd.Timestamp,
    factor_columns: tuple[str, ...],
    min_observations: int,
) -> FactorFitMetadata | None:
    # Strictly less-than is deliberate.  The return at entry_date can only be
    # known after the entry close and is not a public pre-entry observation.
    pre_factor = factor_log_returns.loc[factor_log_returns.index < entry_date, factor_columns]
    pre_asset = asset_log_returns.loc[asset_log_returns.index < entry_date]
    joined = pd.concat([pre_asset.rename("asset"), pre_factor], axis=1, join="inner").dropna()
    if len(joined) < max(int(min_observations), len(factor_columns) + 2):
        return None

    x = joined.loc[:, list(factor_columns)].to_numpy(dtype=float)
    y = joined["asset"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(joined), dtype=float), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    dof = max(len(y) - design.shape[1], 1)
    residual_std = float(np.sqrt(np.dot(residuals, residuals) / dof))
    return FactorFitMetadata(
        fit_start_date=_timestamp(cast(Any, joined.index.min()), name="fit start"),
        fit_end_date=_timestamp(cast(Any, joined.index.max()), name="fit end"),
        factor_columns=factor_columns,
        n_observations=len(joined),
        n_parameters=design.shape[1],
        factor_betas=tuple(float(value) for value in coefficients[1:]),
        intercept=float(coefficients[0]),
        residual_std_log=residual_std,
        fit_is_pre_entry=bool((joined.index < entry_date).all()),
        factor_return_unit="log",
        asset_return_unit="log",
    )


def _empty_event_result(event_id: str, requested_entry: pd.Timestamp, status: str) -> dict:
    return {
        "event_id": event_id,
        "requested_entry_date": requested_entry,
        "entry_date": pd.NaT,
        "outcome_date": pd.NaT,
        "outcome_available_date": pd.NaT,
        "ticker": None,
        "raw_return": np.nan,
        "spy_return": np.nan,
        "spy_alpha_return": np.nan,
        "factor_expected_return": np.nan,
        "factor_residual_return": np.nan,
        "raw_return_pct": np.nan,
        "spy_return_pct": np.nan,
        "spy_alpha_pct": np.nan,
        "factor_expected_return_pct": np.nan,
        "factor_residual_return_pct": np.nan,
        "factor_residual_std_log": np.nan,
        "factor_input_return_unit": None,
        "factor_return_unit": "log",
        "asset_return_unit": "log",
        "outcome_return_unit": "simple",
        "factor_fit_start_date": pd.NaT,
        "factor_fit_end_date": pd.NaT,
        "factor_fit_n_observations": 0,
        "factor_fit_n_parameters": 0,
        "factor_columns": tuple(),
        "factor_betas": tuple(),
        "factor_intercept": np.nan,
        "factor_fit_is_pre_entry": False,
        "label_status": status,
        "model_provenance": "point_in_time_ols_log_return_v2",
    }


def compute_spy_factor_residual_outcomes(
    events: pd.DataFrame,
    prices: pd.DataFrame | pd.Series,
    *,
    factor_returns: pd.DataFrame | None = None,
    factor_columns: Iterable[str] | None = None,
    spy_ticker: str = "SPY",
    event_id_column: str | None = "event_id",
    entry_date_column: str | None = "entry_date",
    ticker_column: str | None = "ticker",
    exit_date_column: str | None = None,
    horizon_days: int = 90,
    min_factor_observations: int = 20,
    factor_returns_are_log: bool = False,
    label_as_of: object | None = None,
) -> pd.DataFrame:
    """Build point-in-time SPY/factor-residual labels for event rows.

    Parameters
    ----------
    events:
        Event table with an entry date and ticker.  ``event_id`` is preferred;
        when it is absent, the ``(entry_date, ticker)`` identity is used and
        duplicate events are rejected.
    prices:
        Positive close-price panel with one ticker per column.  It must contain
        ``spy_ticker`` and every event ticker.
    factor_returns:
        Optional daily factor-return panel.  Values are simple returns unless
        ``factor_returns_are_log`` is true.  SPY is added from ``prices`` when
        it is not already present.
    label_as_of:
        Optional information cutoff.  Outcomes whose endpoint is after this
        date are returned with ``label_status='future_label_rejected'`` and no
        target values.  This is useful when constructing a historical fold.

    The result keeps one row per input event.  ``factor_residual_return`` is a
    realized simple return after subtracting the pre-entry fitted factor
    contribution in log-return space.  It is a research diagnostic, not the
    executable return used by production validation.
    """
    if not isinstance(events, pd.DataFrame) or events.empty:
        raise ValueError("events must be a non-empty DataFrame")
    if events.columns.duplicated().any():
        raise ValueError("events contains duplicate column names")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if min_factor_observations < 1:
        raise ValueError("min_factor_observations must be positive")

    prices_panel = _as_price_panel(prices)
    spy_ticker = str(spy_ticker).strip().upper()
    if spy_ticker not in prices_panel.columns:
        raise ValueError(f"prices must contain required {spy_ticker} benchmark")
    entry_column = _resolve_column(
        events,
        entry_date_column,
        ("entry_date", "disclosure_date", "transaction_date", "as_of"),
        role="entry date",
    )
    ticker_column = _resolve_column(
        events, ticker_column, ("ticker", "symbol"), role="ticker"
    )
    if exit_date_column is not None and exit_date_column not in events.columns:
        raise ValueError(f"exit date column {exit_date_column!r} is missing")
    if exit_date_column is None:
        for candidate in ("exit_date", "outcome_date"):
            if candidate in events.columns:
                exit_date_column = candidate
                break
    if event_id_column is not None and event_id_column not in events.columns:
        # A missing default id is normal; an explicitly requested id is not.
        if event_id_column != "event_id":
            raise ValueError(f"event id column {event_id_column!r} is missing")
        event_id_column = None

    factor_panel = _as_factor_returns(
        factor_returns,
        prices=prices_panel,
        spy_ticker=spy_ticker,
        factor_returns_are_log=factor_returns_are_log,
    )
    if factor_columns is None:
        selected_factors = tuple(str(column) for column in factor_panel.columns)
    else:
        selected_factors = tuple(str(column).strip().upper() for column in factor_columns)
        if not selected_factors:
            raise ValueError("factor_columns must not be empty")
        if any(not column for column in selected_factors):
            raise ValueError("factor_columns must not contain empty names")
        if len(set(selected_factors)) != len(selected_factors):
            raise ValueError("factor_columns must be unique")
        missing = sorted(set(selected_factors) - set(factor_panel.columns))
        if missing:
            raise ValueError(f"factor columns are missing from factor_returns: {missing}")
        if spy_ticker not in selected_factors:
            selected_factors = (spy_ticker, *selected_factors)
    _reject_factor_leakage_columns(selected_factors)

    price_log_returns = cast(
        pd.DataFrame, np.log(prices_panel.where(prices_panel > 0))
    ).diff()
    price_dates = cast(pd.DatetimeIndex, prices_panel.index)
    label_cutoff = (
        _timestamp(label_as_of, name="label_as_of")
        if label_as_of is not None
        else None
    )
    factor_input_unit = (
        "simple_price_derived" if factor_returns is None
        else ("log" if factor_returns_are_log else "simple")
    )

    seen_ids: set[str] = set()
    output: list[dict] = []
    for row_number, (_, event) in enumerate(events.iterrows()):
        requested_entry = _timestamp(
            cast(Any, event[entry_column]), name=entry_column
        )
        event_id = _event_identity(
            event,
            index=row_number,
            event_id_column=event_id_column,
            entry_column=entry_column,
            ticker_column=ticker_column,
        )
        if event_id in seen_ids:
            raise ValueError(f"duplicate event id {event_id!r}")
        seen_ids.add(event_id)
        if bool(pd.isna(event[ticker_column])) or not str(
            event[ticker_column]
        ).strip():
            raise ValueError(f"event row {row_number} has a missing ticker")
        ticker = str(event[ticker_column]).strip().upper()
        entry_position = _first_at_or_after(price_dates, requested_entry)
        if entry_position is None or ticker not in prices_panel.columns:
            result = _empty_event_result(event_id, requested_entry, "entry_price_unavailable")
            result["ticker"] = ticker
            output.append(result)
            continue
        entry_date = _timestamp(
            cast(Any, price_dates[entry_position]), name="entry date"
        )

        if exit_date_column is not None and bool(
            pd.notna(event[exit_date_column])
        ):
            requested_exit = _timestamp(
                cast(Any, event[exit_date_column]), name=exit_date_column
            )
        else:
            requested_exit: pd.Timestamp
            row_horizon = horizon_days
            if "horizon_days" in events.columns and bool(
                pd.notna(event["horizon_days"])
            ):
                row_horizon = int(cast(Any, event["horizon_days"]))
                if row_horizon <= 0:
                    raise ValueError("horizon_days values must be positive")
            requested_exit = cast(
                pd.Timestamp, requested_entry + pd.Timedelta(days=row_horizon)
            )
        exit_position = _first_at_or_after(price_dates, requested_exit)
        if exit_position is None or exit_position <= entry_position:
            result = _empty_event_result(event_id, requested_entry, "exit_price_unavailable")
            result["ticker"] = ticker
            result["entry_date"] = entry_date
            output.append(result)
            continue
        outcome_date = _timestamp(
            cast(Any, price_dates[exit_position]), name="outcome date"
        )

        base = _empty_event_result(event_id, requested_entry, "ok")
        base["ticker"] = ticker
        base["entry_date"] = entry_date
        base["outcome_date"] = outcome_date
        base["outcome_available_date"] = outcome_date
        if label_cutoff is not None and outcome_date > label_cutoff:
            base["label_status"] = "future_label_rejected"
            output.append(base)
            continue

        entry_price = float(prices_panel.iloc[entry_position][ticker])
        exit_price = float(prices_panel.iloc[exit_position][ticker])
        spy_entry = float(prices_panel.iloc[entry_position][spy_ticker])
        spy_exit = float(prices_panel.iloc[exit_position][spy_ticker])
        if not all(np.isfinite([entry_price, exit_price, spy_entry, spy_exit])) or min(
            entry_price, exit_price, spy_entry, spy_exit
        ) <= 0:
            base["label_status"] = "non_positive_or_missing_price"
            output.append(base)
            continue

        fit = _ols_factor_fit(
            _column(price_log_returns, ticker),
            factor_panel,
            entry_date=entry_date,
            factor_columns=selected_factors,
            min_observations=min_factor_observations,
        )
        if fit is None:
            base["label_status"] = "insufficient_pre_entry_history"
            output.append(base)
            continue

        future_dates = pd.DatetimeIndex(
            price_dates[(price_dates > entry_date) & (price_dates <= outcome_date)]
        )
        # Sum every daily factor observation in the holding period, not just
        # sparse price endpoints.  Requiring all observed asset endpoint dates
        # to exist in the factor panel still catches a missing daily factor row
        # for the normal daily-price case without making sparse price panels
        # discard intermediate factor information.
        future_factors = factor_panel.loc[
            (factor_panel.index > entry_date) & (factor_panel.index <= outcome_date),
            list(selected_factors),
        ]
        endpoint_coverage = bool(future_dates.isin(factor_panel.index).all())
        if (
            future_factors.empty
            or not endpoint_coverage
            or not np.isfinite(future_factors.to_numpy(dtype=float)).all()
        ):
            base["label_status"] = "future_factor_data_unavailable"
            output.append(base)
            continue

        asset_log = log(exit_price / entry_price)
        spy_log = log(spy_exit / spy_entry)
        factor_sum = future_factors.to_numpy(dtype=float).sum(axis=0)
        expected_log = float(fit.intercept or 0.0) * len(future_factors) + float(
            np.dot(np.asarray(fit.factor_betas, dtype=float), factor_sum)
        )
        residual_log = asset_log - expected_log
        raw_return = exp(asset_log) - 1.0
        spy_return = exp(spy_log) - 1.0
        expected_return = exp(expected_log) - 1.0
        residual_return = exp(residual_log) - 1.0
        base.update(
            {
                "raw_return": raw_return,
                "spy_return": spy_return,
                "spy_alpha_return": raw_return - spy_return,
                "factor_expected_return": expected_return,
                "factor_residual_return": residual_return,
                "raw_return_pct": raw_return * 100.0,
                "spy_return_pct": spy_return * 100.0,
                "spy_alpha_pct": (raw_return - spy_return) * 100.0,
                "factor_expected_return_pct": expected_return * 100.0,
                "factor_residual_return_pct": residual_return * 100.0,
                "factor_residual_std_log": fit.residual_std_log,
                "factor_input_return_unit": factor_input_unit,
                "factor_fit_start_date": fit.fit_start_date,
                "factor_fit_end_date": fit.fit_end_date,
                "factor_fit_n_observations": fit.n_observations,
                "factor_fit_n_parameters": fit.n_parameters,
                "factor_columns": fit.factor_columns,
                "factor_betas": fit.factor_betas,
                "factor_intercept": fit.intercept,
                "factor_fit_is_pre_entry": fit.fit_is_pre_entry,
            }
        )
        output.append(base)

    result = pd.DataFrame(output)
    if result.empty:  # pragma: no cover - events is checked above
        return result
    date_columns = [
        "requested_entry_date",
        "entry_date",
        "outcome_date",
        "outcome_available_date",
        "factor_fit_start_date",
        "factor_fit_end_date",
    ]
    for column in date_columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    result["factor_input_return_unit"] = result[
        "factor_input_return_unit"
    ].fillna(factor_input_unit)
    result["label_as_of"] = label_cutoff
    result["factor_path_alignment"] = "daily_factor_rows_with_price_endpoint_coverage"
    return result


# Names kept deliberately descriptive.  The aliases make the helper easy to
# find without creating a second implementation or a production integration.
build_spy_factor_residual_outcomes = compute_spy_factor_residual_outcomes
point_in_time_spy_factor_residual_outcomes = compute_spy_factor_residual_outcomes


__all__ = [
    "FactorFitMetadata",
    "build_spy_factor_residual_outcomes",
    "compute_spy_factor_residual_outcomes",
    "point_in_time_spy_factor_residual_outcomes",
]
