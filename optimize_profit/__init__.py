"""Optimize profit: walk-forward sweep across scoring functions and allocations.

Public API:
  - SCORING_FUNCTIONS      dict of {name: callable}
  - run_walk_forward       per-combo walk-forward backtest
  - precompute_walk_forward_data  per-as_of_date shared precomputation
  - main                   entry point (run as ``python -m optimize_profit.main``)

The package is split into:
  - scoring.py       continuous scoring functions
  - precompute.py    per-as_of_date precomputation
  - walk_forward.py  walk-forward backtest engine
  - main.py          sweep driver + reporting
"""

from optimize_profit.scoring import SCORING_FUNCTIONS
from optimize_profit.walk_forward import run_walk_forward
from optimize_profit.precompute import precompute_walk_forward_data

__all__ = [
    "SCORING_FUNCTIONS",
    "run_walk_forward",
    "precompute_walk_forward_data",
]
