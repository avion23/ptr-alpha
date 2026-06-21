"""Member ranking and scoring.

Public API:
  Bayesian helpers:
    - bayesian_win_probability(wins, losses, market_prior=0.55)
    - bayes_factor_against_market(wins, losses, market_prior=0.55)
    - _compute_ticker_member_performance(...)
  Score factors:
    - _size_score_factor
    - _owner_score_factor
    - _conviction_score
  Member ranking:
    - rank_members(signal_df, horizon=90, threshold=5.0)
    - _prepare_member_data (memoized)
    - _rank_members_impl (memoized)
    - rank_sales(signal_df, horizon=90)
    - _compute_member_stats
    - _get_ticker_purchases
    - _lookup_buyer_bayes_win_prob
    - _build_buyer_bayes_dict
    - _build_ranking_dicts
    - get_ticker_buyers_with_rankings(...)
  Buyer composition scoring:
    - score_ticker_by_buyers(ticker, transactions_df, signals_df, ...)
  Decay lambda estimation:
    - estimate_member_decay_lambda
    - get_member_decay_map

The package is split into:
  - bayes.py          Bayesian math + ticker performance
  - factors.py        score factor helpers (size/owner/conviction)
  - ranking.py        ranking pipeline (purchase side)
  - sales.py          ranking pipeline (sale side) + per-member stats
  - lookups.py        O(1) dict helpers + ticker-buyer join
  - buyer_scoring.py  score_ticker_by_buyers
  - decay.py          decay-lambda estimation
"""

from analyzer.member_ranking.bayes import (
    bayes_factor_against_market,
    bayesian_win_probability,
    _compute_ticker_member_performance,
)
from analyzer.member_ranking.factors import (
    _conviction_score,
    _owner_score_factor,
    _size_score_factor,
)
from analyzer.member_ranking.decay import (
    estimate_member_decay_lambda,
    get_member_decay_map,
)
from analyzer.member_ranking.ranking import (
    _prepare_member_data,
    _rank_members_impl,
    rank_members,
)
from analyzer.member_ranking.sales import (
    _compute_member_stats,
    rank_sales,
)
from analyzer.member_ranking.lookups import (
    _build_buyer_bayes_dict,
    _build_ranking_dicts,
    _get_ticker_purchases,
    _lookup_buyer_bayes_win_prob,
    get_ticker_buyers_with_rankings,
)
from analyzer.member_ranking.buyer_scoring import score_ticker_by_buyers

__all__ = [
    "bayesian_win_probability",
    "bayes_factor_against_market",
    "_compute_ticker_member_performance",
    "_size_score_factor",
    "_owner_score_factor",
    "_conviction_score",
    "estimate_member_decay_lambda",
    "get_member_decay_map",
    "rank_members",
    "rank_sales",
    "_prepare_member_data",
    "_rank_members_impl",
    "_compute_member_stats",
    "_get_ticker_purchases",
    "_lookup_buyer_bayes_win_prob",
    "_build_buyer_bayes_dict",
    "_build_ranking_dicts",
    "get_ticker_buyers_with_rankings",
    "score_ticker_by_buyers",
]
