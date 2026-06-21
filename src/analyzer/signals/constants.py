"""Module-level constants for the signals pipeline.

These are global knobs read by ``calculate_signal_potential`` and the sweep
driver. The parameter sweep mutates ``DECAY_LAMBDA`` and ``BAYES_PRIOR_STRENGTH``
per combo, so callers should resolve them at call time (default args would
freeze them at function-definition time).
"""

# Decay weight per day for the midpoint-weighted return
DECAY_LAMBDA = 0.005
# Reference position size in USD used by trade-size adjustment factors
POSITION_SIZE_BASELINE = 10000.0
# Cap on the disclosure-metadata score adjustment (size/owner/etc.)
MAX_DISCLOSURE_METADATA_ADJUSTMENT = 0.15
# Prior strength for Bayesian shrinkage (alpha+beta pseudo-counts)
BAYES_PRIOR_STRENGTH = 20.0
# Exponential decay (per day) for buyer recency weighting in score_ticker_by_buyers
BUYER_RECENCY_DECAY = 0.03
# Minimum trades for a ticker to qualify for the ticker-history prior
TICKER_PERF_MIN_TRADES = 3

# Minimum entry price (USD) — signals with entry_price below this are dropped
MIN_ENTRY_PRICE = 3.0
# Conviction-score weights: pure SPY alpha vs realized return contribution.
# Alpha-only avoids double-counting stock return.
CONVICTION_WEIGHT_ALPHA = 1.0
CONVICTION_WEIGHT_REALIZED = 0.0

_NS_PER_DAY = 86_400_000_000_000  # nanoseconds in a day
