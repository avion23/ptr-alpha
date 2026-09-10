"""Numerical defaults for retrospective signal diagnostics."""

from typing import Final

# Midpoint-weighted log-return decay. This is retrospective diagnostic math;
# it does not enter the production consensus score or member ranking.
DECAY_LAMBDA: float = 0.005
_NS_PER_DAY: Final[int] = 86_400_000_000_000
