"""Retrospective strategy selection with a locked future final test.

The package enforces non-overlapping support, treats scheduled no-trade periods
as cash, records every rejection, and precommits a post-2025 test whose rows normal retrospective runs never analytically query. Retrospective
runs do read the database file to record its whole-file hash. Reused 2024-2025
history is validation only.
"""

from optimize_profit.precompute import precompute_walk_forward_data
from optimize_profit.scoring import SCORING_FUNCTIONS
from optimize_profit.walk_forward import run_walk_forward

__all__ = [
    "SCORING_FUNCTIONS",
    "run_walk_forward",
    "precompute_walk_forward_data",
]
