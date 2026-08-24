"""Pure execution-cost arithmetic for research labels.

The backtest and portfolio modules have their own execution shells.  This
module is deliberately independent of both of them: it gives a small,
validated cost value object and the arithmetic needed when an existing gross
return label is turned into an executable return target.

Returns are represented as decimal simple returns by the core functions (so
``0.10`` means ten percent).  ``*_pct`` helpers are provided for the existing
signal columns, which store percentages.  Costs are applied multiplicatively
to the two endpoints, never by subtracting a percentage from a return::

    (1 + gross_return) * (1 - exit_cost) / (1 + entry_cost) - 1

This distinction matters for non-zero returns and keeps the label arithmetic
identical to executable endpoint arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, TypeVar

import numpy as np
import pandas as pd


NumberLike = TypeVar("NumberLike")


def _validated_bps(value: object, field_name: str) -> float:
    """Return a finite basis-point value in the executable range."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    # A sell-side multiplier of zero or less is not an executable positive
    # endpoint.  Keep the same [0, 10000) convention as the portfolio shell.
    if converted < 0.0 or converted >= 10_000.0:
        raise ValueError(f"{field_name} must be in [0, 10000) basis points")
    return converted


@dataclass(frozen=True, slots=True, init=False)
class ExecutionCosts:
    """Validated proportional entry and exit execution costs.

    Parameters are basis points.  The canonical names mirror the existing
    executable backtest interface.  ``entry_bps``/``exit_bps`` and
    ``buy_bps``/``sell_bps`` are accepted as keyword aliases so callers do not
    need an adapter merely to name the two endpoints.

    No fixed-dollar fee is included: applying one would require a notional,
    and silently choosing a notional would make a return label ambiguous.
    """

    entry_slippage_bps: float
    exit_slippage_bps: float

    def __init__(
        self,
        entry_slippage_bps: float = 0.0,
        exit_slippage_bps: float = 0.0,
        *,
        entry_bps: float | None = None,
        exit_bps: float | None = None,
        buy_bps: float | None = None,
        sell_bps: float | None = None,
    ) -> None:
        entry = self._resolve_alias(
            entry_slippage_bps,
            ("entry_bps", entry_bps),
            ("buy_bps", buy_bps),
            field_name="entry_slippage_bps",
        )
        exit_ = self._resolve_alias(
            exit_slippage_bps,
            ("exit_bps", exit_bps),
            ("sell_bps", sell_bps),
            field_name="exit_slippage_bps",
        )
        object.__setattr__(
            self, "entry_slippage_bps", _validated_bps(entry, "entry_slippage_bps")
        )
        object.__setattr__(
            self, "exit_slippage_bps", _validated_bps(exit_, "exit_slippage_bps")
        )

    @staticmethod
    def _resolve_alias(
        canonical: float,
        *aliases: tuple[str, float | None],
        field_name: str,
    ) -> float:
        supplied = [(name, value) for name, value in aliases if value is not None]
        if not supplied:
            return canonical
        for name, value in supplied:
            if value is None:  # pragma: no cover - filtered above
                continue
            if canonical != 0.0 and float(canonical) != float(value):
                raise ValueError(f"{field_name} and {name} disagree")
        first = supplied[0][1]
        assert first is not None
        if any(float(value) != float(first) for _, value in supplied[1:]):
            names = ", ".join(name for name, _ in supplied)
            raise ValueError(f"execution-cost aliases disagree: {names}")
        return float(first)

    @property
    def entry_bps(self) -> float:
        """Entry cost in basis points."""
        return self.entry_slippage_bps

    @property
    def exit_bps(self) -> float:
        """Exit cost in basis points."""
        return self.exit_slippage_bps

    @property
    def buy_bps(self) -> float:
        """Alias for :attr:`entry_slippage_bps`."""
        return self.entry_slippage_bps

    @property
    def sell_bps(self) -> float:
        """Alias for :attr:`exit_slippage_bps`."""
        return self.exit_slippage_bps

    @property
    def entry_rate(self) -> float:
        """Entry cost as a decimal proportion."""
        return self.entry_slippage_bps / 10_000.0

    @property
    def exit_rate(self) -> float:
        """Exit cost as a decimal proportion."""
        return self.exit_slippage_bps / 10_000.0

    @property
    def entry_multiplier(self) -> float:
        """Multiplicative buy-side endpoint adjustment."""
        return 1.0 + self.entry_rate

    @property
    def exit_multiplier(self) -> float:
        """Multiplicative sell-side endpoint adjustment."""
        return 1.0 - self.exit_rate

    @classmethod
    def from_bps(
        cls, entry_bps: float = 0.0, exit_bps: float = 0.0
    ) -> "ExecutionCosts":
        """Construct costs using the short endpoint names."""
        return cls(entry_bps=entry_bps, exit_bps=exit_bps)


DEFAULT_EXECUTION_COSTS = ExecutionCosts()


def _as_numeric(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
) -> tuple[Any, np.ndarray]:
    """Convert a scalar/array-like value while retaining the public shape."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric returns") from exc
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} must be finite or NaN")
    finite = np.isfinite(array)
    if minimum is not None and np.any(array[finite] < minimum):
        raise ValueError(f"{name} cannot be less than {minimum}")
    return value, array


def _restore_type(original: Any, result: np.ndarray | float) -> Any:
    """Keep scalar, pandas, and numpy containers useful to callers."""
    result_array = np.asarray(result, dtype=float)
    if np.isscalar(original) and result_array.ndim == 0:
        return float(result_array)
    if isinstance(original, pd.Series):
        return pd.Series(result_array, index=original.index, name=original.name)
    if isinstance(original, pd.Index):
        return pd.Index(result_array, name=original.name)
    if np.isscalar(original):
        return result_array
    if isinstance(original, np.ndarray):
        return result_array
    return result_array


def executable_return(
    gross_return: Any,
    costs: ExecutionCosts = DEFAULT_EXECUTION_COSTS,
) -> Any:
    """Apply costs to a decimal gross simple return exactly.

    Missing values remain missing.  A gross ``-1`` is accepted and produces a
    total-loss executable return; values below ``-1`` are rejected because
    they cannot be generated by two positive price endpoints.
    """
    if not isinstance(costs, ExecutionCosts):
        raise TypeError("costs must be an ExecutionCosts instance")
    original, values = _as_numeric(gross_return, name="gross_return", minimum=-1.0)
    result = (1.0 + values) * costs.exit_multiplier / costs.entry_multiplier - 1.0
    return _restore_type(original, result)


def executable_return_pct(
    gross_return_pct: Any,
    costs: ExecutionCosts = DEFAULT_EXECUTION_COSTS,
) -> Any:
    """Apply costs to a percentage gross return and return a percentage."""
    original, values = _as_numeric(
        gross_return_pct, name="gross_return_pct", minimum=-100.0
    )
    # Validate the percentage in its natural unit before scaling so a value
    # below -100 is rejected rather than accidentally accepted as -1 percent.
    finite = np.isfinite(values)
    if np.any(values[finite] < -100.0):
        raise ValueError("gross_return_pct cannot be less than -100")
    result = executable_return(values / 100.0, costs)
    return _restore_type(original, np.asarray(result, dtype=float) * 100.0)


def executable_return_from_prices(
    entry_price: Any,
    exit_price: Any,
    costs: ExecutionCosts = DEFAULT_EXECUTION_COSTS,
) -> Any:
    """Compute an executable return from exact positive endpoint prices."""
    if not isinstance(costs, ExecutionCosts):
        raise TypeError("costs must be an ExecutionCosts instance")
    original, entry = _as_numeric(entry_price, name="entry_price")
    _, exit_ = _as_numeric(exit_price, name="exit_price")
    try:
        broadcast_entry, broadcast_exit = np.broadcast_arrays(entry, exit_)
    except ValueError as exc:
        raise ValueError("entry_price and exit_price have incompatible shapes") from exc
    finite = np.isfinite(broadcast_entry) & np.isfinite(broadcast_exit)
    if np.any(broadcast_entry[finite] <= 0.0) or np.any(broadcast_exit[finite] <= 0.0):
        raise ValueError("entry_price and exit_price must be positive or NaN")
    result = (
        broadcast_exit
        * costs.exit_multiplier
        / (broadcast_entry * costs.entry_multiplier)
        - 1.0
    )
    # The entry argument determines scalar-vs-array restoration.  Broadcasting
    # still permits a scalar entry against an endpoint array.
    template = entry_price if not np.isscalar(entry_price) else exit_price
    return _restore_type(template, result)


def executable_alpha_pct(
    gross_return_pct: Any,
    gross_alpha_pct: Any,
    costs: ExecutionCosts = DEFAULT_EXECUTION_COSTS,
) -> Any:
    """Return executable security alpha against the unchanged gross benchmark.

    Existing ``total_spy_alpha_pct`` labels equal security gross return minus
    the same-endpoint SPY gross return.  Therefore the benchmark can be
    recovered as ``gross_return_pct - gross_alpha_pct`` and is not charged
    costs a second time.
    """
    gross_original, gross = _as_numeric(
        gross_return_pct, name="gross_return_pct", minimum=-100.0
    )
    _, alpha = _as_numeric(gross_alpha_pct, name="gross_alpha_pct")
    gross, alpha = np.broadcast_arrays(gross, alpha)
    finite = np.isfinite(gross) & np.isfinite(alpha)
    net = np.asarray(executable_return_pct(gross, costs), dtype=float)
    benchmark = gross - alpha
    result = np.where(finite, net - benchmark, np.nan)
    return _restore_type(gross_original, result)


# Descriptive aliases keep the formula discoverable without introducing a
# second implementation.
net_executable_return = executable_return
net_executable_return_pct = executable_return_pct
apply_execution_costs = executable_return
apply_execution_costs_pct = executable_return_pct


__all__ = [
    "DEFAULT_EXECUTION_COSTS",
    "ExecutionCosts",
    "apply_execution_costs",
    "apply_execution_costs_pct",
    "executable_alpha_pct",
    "executable_return",
    "executable_return_from_prices",
    "executable_return_pct",
    "net_executable_return",
    "net_executable_return_pct",
]
