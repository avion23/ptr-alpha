import pandas as pd
import pytest

from analyzer.exceptions import AnalysisError
from analyzer.member_ranking.buyer_scoring import score_ticker_by_buyers


def _signals() -> pd.DataFrame:
    return pd.DataFrame({"member": ["diagnostic-only"]})


def test_solo_consensus_ignores_member_posterior_and_penalty_parameters():
    transactions = pd.DataFrame(
        {
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-05-14"]),
            "transaction_type": ["Purchase"],
        }
    )
    rankings = pd.DataFrame(
        {
            "member": ["Alice"],
            "avg_spy_alpha_pct": [1.2345],
            "purchase_trades": [5],
            "bayes_win_prob": [0.01],
            "posterior_lift": [0.01],
        }
    )

    result = score_ticker_by_buyers(
        "AAPL",
        transactions,
        _signals(),
        member_rankings=rankings,
        min_buyers=1,
        solo_buyer_skill_threshold=99.0,
        solo_buyer_penalty=0.01,
    )
    row = result.iloc[0]

    assert not bool(row["solo_buyer"])
    assert row["signal_score_raw"] == 1.0
    assert row["signal_score"] == 1.0
    assert row["scoring_mode"] == "consensus"


def test_consensus_score_is_invariant_to_member_identity_shuffle():
    transactions = pd.DataFrame(
        {
            "member": ["Alice", "Bob", "Carol"],
            "ticker": ["AAPL"] * 3,
            "disclosure_date": pd.to_datetime(
                ["2024-05-10", "2024-05-12", "2024-05-14"]
            ),
            "transaction_type": ["Purchase"] * 3,
        }
    )
    shuffled = transactions.copy()
    shuffled["member"] = ["Xavier Able", "Zelda Baker", "Yvonne Carter"]

    original = score_ticker_by_buyers("AAPL", transactions, _signals(), min_buyers=1)
    permuted = score_ticker_by_buyers("AAPL", shuffled, _signals(), min_buyers=1)

    assert original.iloc[0]["signal_score_raw"] == permuted.iloc[0]["signal_score_raw"]
    assert original.iloc[0]["num_buyers"] == permuted.iloc[0]["num_buyers"] == 3


def test_probability_times_alpha_scoring_is_rejected():
    transactions = pd.DataFrame(
        {
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "AAPL"],
            "disclosure_date": pd.to_datetime(["2024-05-13", "2024-05-14"]),
            "transaction_type": ["Purchase", "Purchase"],
        }
    )
    rankings = pd.DataFrame(
        {
            "member": ["Alice", "Bob"],
            "shrunk_alpha": [2.0, 1.0],
            "purchase_trades": [3, 4],
            "bayes_win_prob": [0.8, 0.7],
        }
    )

    with pytest.raises(AnalysisError, match="incompatible units"):
        score_ticker_by_buyers(
            "AAPL",
            transactions,
            _signals(),
            member_rankings=rankings,
            scoring_mode="bayesian_quality",
        )
