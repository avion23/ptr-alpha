"""Simple options leverage model for congressional trading analysis."""

# Average leverage factors for ATM options
# These are rough approximations — real delta depends on moneyness, time to expiry, vol
CALL_LEVERAGE = 10.0  # ~10x for 3-month ATM call
PUT_LEVERAGE = -5.0   # ~-5x for 3-month ATM put (negative = inverse)


def estimate_options_leverage(instrument_type: str, amount_midpoint: float | None = None) -> float:
    """Estimate the leverage multiplier for a trade.

    Returns 1.0 for stocks, ~10 for calls, ~-5 for puts.
    If amount is available, adjust slightly (larger amounts -> slightly lower leverage).
    """
    if instrument_type == 'call':
        base = CALL_LEVERAGE
    elif instrument_type == 'put':
        base = PUT_LEVERAGE
    else:
        return 1.0

    # Slight adjustment for amount (larger bets -> slightly less leveraged on average)
    if amount_midpoint and amount_midpoint > 0:
        import math
        amount_factor = 1.0 - 0.1 * math.log10(amount_midpoint / 10000)
        amount_factor = max(0.7, min(1.3, amount_factor))
        return base * amount_factor
    return base
