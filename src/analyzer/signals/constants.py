"""Module-level constants for the signals pipeline.

These are module defaults read by ``calculate_signal_potential`` and other
signal functions. Parameter sweeps pass ``DECAY_LAMBDA`` and
``BAYES_PRIOR_STRENGTH`` values explicitly instead of changing these defaults.
The remaining constants are immutable and declared ``Final``.
"""

from typing import Final

# Decay weight per day for the midpoint-weighted return
DECAY_LAMBDA: float = 0.005
# Prior strength for Bayesian shrinkage (alpha+beta pseudo-counts)
BAYES_PRIOR_STRENGTH: float = 20.0
# Minimum trades for a ticker to qualify for the ticker-history prior
TICKER_PERF_MIN_TRADES: Final[int] = 3

# Minimum entry price (USD) — signals with entry_price below this are dropped
MIN_ENTRY_PRICE: Final[float] = 3.0
_NS_PER_DAY: Final[int] = 86_400_000_000_000  # nanoseconds in a day
