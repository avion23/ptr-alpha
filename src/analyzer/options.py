"""Simple model estimates for options leverage in backtest returns."""

# These are rough model estimates, not actual contract leverage. Real deltas
# depend on moneyness, time to expiry, and volatility.
CALL_LEVERAGE = 4.0   # ~4x for a call
PUT_LEVERAGE = -2.0   # ~-2x for a put (negative = inverse)


def estimate_options_leverage(instrument_type: str, amount_midpoint: float | None = None) -> float:
    """Estimate the leverage multiplier for a trade.

    Returns 1.0 for stocks, ~4 for calls, and ~-2 for puts. These are model
    estimates, not actual contract leverage.
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
