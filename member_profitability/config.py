"""Configuration constants for member profitability analysis.

These are the knobs the analysis loops over. Held in a separate module so
the heavy lifting modules can import without picking up `Database` /
`main()`-side-effects.
"""

HORIZON = 60
DECAY_LAMBDA = 0.005
TRAIN_WINDOW_DAYS = 180  # 6 months training
TEST_WINDOW_DAYS = 180   # 6 months test
MIN_MEMBERS_FOR_CORR = 10

METRICS_TO_TEST = [
    "shrunk_alpha",
    "bayes_win_prob",
    "conviction_score",
    "purchase_trades",
    "prob_up_given_buy",
    "sharpe_ratio",
    "avg_spy_alpha_pct",
]

TOP_N_VALUES = [1, 2, 3, 5, 10, 15]
MIN_BUYERS_VALUES = [1, 2, 3, 4, 5]

WINDOW_SLIDE_DAYS = 90
MIN_TEST_TRADES = 2

TRADE_COUNT_THRESHOLDS = [2, 3, 5, 8, 10, 15, 20]

MIN_MEMBERS_FOR_TIER = 20
TIER_FRACTION = 0.10

TX_START = "2021-10-07"
TX_END = "2025-06-30"
PRICE_END_BUFFER_DAYS = 130
