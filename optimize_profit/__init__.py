"""Honest chronological optimization for congressional-trade strategies.

The package enforces non-overlapping bankroll periods, selects configurations
before a fixed untouched holdout, records coverage/rejections, and persists
reproducible run artifacts. A passing holdout is evidence for further paper
trading, never a guaranteed-profit claim.
"""

from optimize_profit.precompute import precompute_walk_forward_data
from optimize_profit.scoring import SCORING_FUNCTIONS
from optimize_profit.walk_forward import run_walk_forward

__all__ = [
    "SCORING_FUNCTIONS",
    "run_walk_forward",
    "precompute_walk_forward_data",
]
