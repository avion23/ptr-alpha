"""Backtest evaluation and recommendation scoring.

Public API (re-exported here so existing `from analyzer.backtest import X`
calls keep working after the subpackage split):

  Price array helpers:
    - _find_dip_entry, _find_dip_entry_arrays
    - _price_at_or_before_arrays, _price_on_or_before_arrays

  Window filters (memoized):
    - _filter_training, _filter_recent_trades, _filter_ticker_perf

  Curve building:
    - _build_ticker_curves, _build_global_curves, _build_curves_for_rows

  OU parameter estimation:
    - _compute_ticker_ou_params, _compute_ticker_entry_value,
      _compute_ticker_optimal_horizon

  Top-level orchestration:
    - backtest_recommendations, evaluate_backtest, summarize_backtest

The package is split into:
  - prices.py     price array helpers + dip entry
  - filters.py    training/recent/ticker-perf window filters
  - curves.py     vectorized curve building
  - ou_params.py  OU fit + entry value + optimal horizon
  - recommend.py  backtest_recommendations pipeline
  - evaluate.py   evaluate_backtest entry/exit simulation
  - summary.py    summarize_backtest per-rank rollup
"""

from analyzer.backtest.prices import (
    _find_dip_entry,
    _find_dip_entry_arrays,
    _price_at_or_before_arrays,
    _price_on_or_before_arrays,
)
from analyzer.backtest.filters import (
    _filter_recent_trades,
    _filter_ticker_perf,
    _filter_training,
)
from analyzer.backtest.curves import (
    _build_curves_for_rows,
    _build_global_curves,
    _build_ticker_curves,
)
from analyzer.backtest.ou_params import (
    _compute_ticker_entry_value,
    _compute_ticker_optimal_horizon,
    _compute_ticker_ou_params,
)
from analyzer.backtest.recommend import backtest_recommendations
from analyzer.backtest.evaluate import evaluate_backtest
from analyzer.backtest.summary import summarize_backtest

__all__ = [
    "_find_dip_entry",
    "_find_dip_entry_arrays",
    "_price_at_or_before_arrays",
    "_price_on_or_before_arrays",
    "_filter_training",
    "_filter_recent_trades",
    "_filter_ticker_perf",
    "_build_curves_for_rows",
    "_build_global_curves",
    "_build_ticker_curves",
    "_compute_ticker_entry_value",
    "_compute_ticker_optimal_horizon",
    "_compute_ticker_ou_params",
    "backtest_recommendations",
    "evaluate_backtest",
    "summarize_backtest",
]
