from __future__ import annotations

import pandas as pd


def _normalize_price_index(
    prices: pd.DataFrame,
    *,
    invalid_error: type[Exception] | None = None,
    duplicate_error: type[Exception] = ValueError,
    duplicate_message: str = "Price index contains duplicate calendar dates",
) -> pd.DataFrame:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(prices.index))
    except (TypeError, ValueError) as exc:
        if invalid_error is None:
            raise
        raise invalid_error("Price index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize().as_unit("ns")
    if index.has_duplicates:
        raise duplicate_error(duplicate_message)
    normalized = prices.copy()
    normalized.index = index
    return normalized.sort_index()
