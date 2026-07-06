"""Module-level constants for the signals pipeline.

These are global defaults read by ``calculate_signal_potential`` and other
signal functions.  ``DECAY_LAMBDA`` and ``BAYES_PRIOR_STRENGTH`` are **not**
``Final`` — ``validation.py`` saves/restores them at runtime during parameter
sweeps.  The remaining constants are truly immutable and declared ``Final``.
"""

from typing import Final

# Decay weight per day for the midpoint-weighted return
DECAY_LAMBDA: float = 0.005
# Reference position size in USD used by trade-size adjustment factors
POSITION_SIZE_BASELINE: Final[float] = 10000.0
# Cap on the disclosure-metadata score adjustment (size/owner/etc.)
MAX_DISCLOSURE_METADATA_ADJUSTMENT: Final[float] = 0.15
# Prior strength for Bayesian shrinkage (alpha+beta pseudo-counts)
BAYES_PRIOR_STRENGTH: float = 20.0
# Exponential decay (per day) for buyer recency weighting in score_ticker_by_buyers
BUYER_RECENCY_DECAY: Final[float] = 0.03
# Minimum trades for a ticker to qualify for the ticker-history prior
TICKER_PERF_MIN_TRADES: Final[int] = 3

# Minimum entry price (USD) — signals with entry_price below this are dropped
MIN_ENTRY_PRICE: Final[float] = 3.0
# Conviction-score weights: pure SPY alpha vs realized return contribution.
# Alpha-only avoids double-counting stock return.
CONVICTION_WEIGHT_ALPHA: Final[float] = 1.0
CONVICTION_WEIGHT_REALIZED: Final[float] = 0.0

_NS_PER_DAY: Final[int] = 86_400_000_000_000  # nanoseconds in a day
