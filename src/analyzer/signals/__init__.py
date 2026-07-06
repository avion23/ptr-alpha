"""Signal generation and calculation.

Data-oriented redesign: replaces the merge-then-filter pattern
(75M+ intermediate rows → 3M filtered) with per-ticker searchsorted
lookups via the `_price_arrays` index. Pre-computes SPY log returns
once on the full Series instead of per-signal groupby shifts.

The package is split into:
  - constants.py     module-level constants (decay, priors, weights)
  - prices.py        O(log N) price lookups + per-DataFrame cache
  - filters.py       horizon/quality/episode filters + dynamic prior
  - top_signals.py   get_top_signals / get_member_signals
  - core.py          vectorized signal kernel + main entry point

Public API is re-exported here so `from analyzer.signals import X` keeps
working after the split.
"""

# Constants
from analyzer.signals.constants import (
    BAYES_PRIOR_STRENGTH,
    BUYER_RECENCY_DECAY,
    CONVICTION_WEIGHT_ALPHA,
    CONVICTION_WEIGHT_REALIZED,
    DECAY_LAMBDA,
    MAX_DISCLOSURE_METADATA_ADJUSTMENT,
    MIN_ENTRY_PRICE,
    POSITION_SIZE_BASELINE,
    TICKER_PERF_MIN_TRADES,
    _NS_PER_DAY,
)

# Price index + lookups
from analyzer.signals.prices import (
    _clear_price_index_cache,
    _price_arrays,
    _price_at_or_before,
    _price_at_or_near,
    _price_index_for_df,
    _price_on_or_before,
)

# Filters + episode collapsing
from analyzer.signals.filters import (
    _apply_quality_filter,
    _assign_episode_ids,
    _collapse_to_episodes,
    _compute_dynamic_prior,
    _get_horizon_data,
)

# Top signals
from analyzer.signals.top_signals import (
    _get_member_signals,
    _get_top_signals,
    get_member_signals,
    get_top_signals,
)

# Core signal computation
from analyzer.signals.core import (
    _compute_ticker_signals,
    calculate_signal_potential,
    compute_signal_potential_with_member_decay,
)

__all__ = [
    "DECAY_LAMBDA",
    "POSITION_SIZE_BASELINE",
    "MAX_DISCLOSURE_METADATA_ADJUSTMENT",
    "BAYES_PRIOR_STRENGTH",
    "BUYER_RECENCY_DECAY",
    "TICKER_PERF_MIN_TRADES",
    "MIN_ENTRY_PRICE",
    "CONVICTION_WEIGHT_ALPHA",
    "CONVICTION_WEIGHT_REALIZED",
    "_NS_PER_DAY",
    "_clear_price_index_cache",
    "_price_arrays",
    "_price_at_or_before",
    "_price_at_or_near",
    "_price_index_for_df",
    "_price_on_or_before",
    "_apply_quality_filter",
    "_assign_episode_ids",
    "_collapse_to_episodes",
    "_compute_dynamic_prior",
    "_get_horizon_data",
    "_get_member_signals",
    "_get_top_signals",
    "get_member_signals",
    "get_top_signals",
    "_compute_ticker_signals",
    "calculate_signal_potential",
    "compute_signal_potential_with_member_decay",
]
