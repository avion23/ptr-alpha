"""Option return safety boundary.

Underlying-equity returns are not option-contract returns. Contract terms alone
are insufficient without an actual option price series, so strategies must
abstain rather than apply a modeled leverage multiplier.
"""


class UnsupportedOptionPricingError(ValueError):
    """Raised when realized option profit is requested without contract prices."""


def estimate_options_leverage(
    instrument_type: str, amount_midpoint: float | None = None
) -> float:
    """Return 1 for stock and reject unsupported option-return fabrication."""
    normalized = str(instrument_type or "").strip().lower()
    if normalized == "stock":
        return 1.0
    if normalized in {"call", "put", "option", "stock option"}:
        raise UnsupportedOptionPricingError(
            "Actual option-contract prices are required for realized returns"
        )
    raise ValueError(f"Unsupported instrument type: {instrument_type!r}")
