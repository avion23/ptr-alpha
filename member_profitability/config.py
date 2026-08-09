"""Configuration constants for point-in-time member research."""

HORIZON = 60
DECAY_LAMBDA = 0.005
BAYES_PRIOR_STRENGTH = 20.0
TRAIN_WINDOW_DAYS = 180
TEST_WINDOW_DAYS = 180
# Test periods must not overlap. Each realized outcome appears in one research window.
WINDOW_SLIDE_DAYS = TEST_WINDOW_DAYS
MIN_MEMBERS_FOR_CORR = 10
MIN_MEMBERS_FOR_TIER = 20
TIER_FRACTION = 0.10
MIN_TEST_TRADES = 2

TARGET_RETURN_COLUMN = "total_spy_alpha_pct"
TEST_RETURN_COLUMN = "test_excess_return_pct"
DATA_SCOPE = "mixed_unclassified"
BUYER_LOOKBACK_DAYS = 28

METRICS_TO_TEST = [
    "shrunk_excess_return_pct",
    "bayes_positive_excess_prob",
    "conviction_score",
    "purchase_trades",
    "prob_positive_excess",
    "sharpe_excess_return",
    "avg_excess_return_pct",
]

TOP_N_VALUES = [1, 2, 3, 5, 10, 15]
MIN_BUYERS_VALUES = [1, 2, 3, 4, 5]
TRADE_COUNT_THRESHOLDS = [2, 3, 5, 8, 10, 15, 20]

TX_START = "2021-10-07"
TX_END = "2025-06-30"
PRICE_END_BUFFER_DAYS = 130
