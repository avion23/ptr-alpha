from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from analyzer.member_ranking.buyer_scoring import (
    _ticker_history_fallback,
    score_ticker_by_buyers,
)
from analyzer.pipeline import TickerScoringParams, run_recent_ticker_scoring


@patch("analyzer.pipeline.analysis.rank_members")
@patch("analyzer.pipeline.analysis.score_ticker_by_buyers")
@patch("analyzer.pipeline.prepare_live_analysis_data")
def test_recent_ticker_scoring_retains_positive_raw_score(
    mock_prepare, mock_score, mock_rank
):
    trades = pd.DataFrame(
        {
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "AAPL"],
            "disclosure_date": pd.to_datetime(["2024-05-13", "2024-05-14"]),
            "transaction_type": ["Purchase", "Purchase"],
        }
    )
    mock_prepare.return_value = (trades, pd.DataFrame(), pd.DataFrame())
    mock_rank.return_value = pd.DataFrame()
    mock_score.return_value = pd.DataFrame(
        {"ticker": ["AAPL"], "signal_score": [0.0], "signal_score_raw": [0.004]}
    )

    result = run_recent_ticker_scoring(
        MagicMock(),
        MagicMock(),
        TickerScoringParams(
            year=2024,
            horizons=(90,),
            days_back=30,
            min_buyers=2,
            as_of_date=date(2024, 5, 15),
        ),
    )

    assert result.success
    assert result.data["result"]["ticker"].tolist() == ["AAPL"]
    assert result.data["result"]["signal_score_raw"].tolist() == [0.004]


def test_final_result_emits_unrounded_solo_adjusted_score():
    transactions = pd.DataFrame(
        {
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-05-14"]),
            "transaction_type": ["Purchase"],
        }
    )
    signals = pd.DataFrame({"member": ["Alice"]})
    rankings = pd.DataFrame(
        {
            "member": ["Alice"],
            "avg_spy_alpha_pct": [1.2345],
            "purchase_trades": [5],
            "bayes_win_prob": [0.9],
        }
    )

    result = score_ticker_by_buyers(
        "AAPL",
        transactions,
        signals,
        member_rankings=rankings,
        min_buyers=1,
        solo_buyer_skill_threshold=0.6,
        solo_buyer_penalty=0.8,
    )
    row = result.iloc[0]

    assert bool(row["solo_buyer"])
    assert row["signal_score_raw"] == 0.9876
    assert row["signal_score"] == 0.99


def test_ticker_history_fallback_requires_three_completed_rows():
    signals = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "signal_type": ["Purchase"] * 3,
            "total_spy_alpha_pct": [1.0, 2.0, 100.0],
            "window_complete": [True, True, False],
        }
    )

    result = _ticker_history_fallback("AAPL", ["Alice"], signals, None)
    row = result.iloc[0]

    assert row["signal_score_raw"] == 0.0
    assert row["signal_score"] == 0.0
    assert row["fallback_source"] == "none"


def test_ticker_history_fallback_excludes_incomplete_rows():
    signals = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 4,
            "signal_type": ["Purchase"] * 4,
            "total_spy_alpha_pct": [1.0, 2.0, 3.0, 100.0],
            "window_complete": [True, True, True, False],
        }
    )

    result = _ticker_history_fallback("AAPL", ["Alice"], signals, None)
    row = result.iloc[0]

    assert row["signal_score_raw"] == 2.0
    assert row["signal_score"] == 2.0
    assert row["fallback_source"] == "ticker_hist(3)"
