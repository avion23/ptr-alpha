import inspect

import numpy as np
import pandas as pd
import pytest

import analyzer.member_ranking as member_ranking_api
import analyzer.member_skill as member_skill_api
from analyzer.exceptions import AnalysisError
from analyzer.member_ranking.buyer_scoring import score_ticker_by_buyers


def _transactions(names=("Alice", "Bob", "Carol")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "member": list(names),
            "ticker": ["AAPL"] * len(names),
            "disclosure_date": pd.to_datetime(
                ["2024-05-10", "2024-05-12", "2024-05-14"][: len(names)]
            ),
            "transaction_type": ["Purchase"] * len(names),
        }
    )


def test_consensus_cold_start_needs_no_rankings_or_signal_history():
    result = score_ticker_by_buyers(
        "AAPL",
        _transactions(("Alice",)),
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=1,
    )

    expected = np.exp(-0.03 * 10)
    assert result.iloc[0]["signal_score_raw"] == pytest.approx(expected)
    assert result.iloc[0]["scoring_mode"] == "consensus"


def test_consensus_score_is_invariant_to_member_identity_shuffle():
    transactions = _transactions()
    shuffled = transactions.copy()
    shuffled["member"] = ["Xavier Able", "Zelda Baker", "Yvonne Carter"]
    as_of = pd.Timestamp("2024-05-20")

    original = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=as_of, min_buyers=1
    )
    permuted = score_ticker_by_buyers("AAPL", shuffled, as_of_date=as_of, min_buyers=1)

    assert original.iloc[0]["signal_score_raw"] == permuted.iloc[0]["signal_score_raw"]
    assert original.iloc[0]["num_buyers"] == permuted.iloc[0]["num_buyers"] == 3


def test_consensus_uses_absolute_age_from_explicit_as_of_date():
    transactions = _transactions(("Alice", "Bob"))
    early = score_ticker_by_buyers(
        "AAPL",
        transactions,
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=1,
    )
    late = score_ticker_by_buyers(
        "AAPL",
        transactions,
        as_of_date=pd.Timestamp("2024-06-19"),
        min_buyers=1,
    )

    assert late.iloc[0]["signal_score_raw"] == pytest.approx(
        early.iloc[0]["signal_score_raw"] * np.exp(-0.03 * 30)
    )


def test_scoring_mode_typo_and_probability_times_alpha_are_rejected():
    transactions = _transactions(("Alice", "Bob"))
    for invalid in ("consensuz", "bayesian_quality"):
        with pytest.raises(AnalysisError, match="Unknown scoring_mode"):
            score_ticker_by_buyers(
                "AAPL",
                transactions,
                as_of_date=pd.Timestamp("2024-05-20"),
                scoring_mode=invalid,
            )


def test_removed_pseudo_posterior_parameters_are_absent_from_public_api():
    parameters = inspect.signature(score_ticker_by_buyers).parameters
    assert "member_skills" not in parameters
    assert "uncertainty_penalty_lambda" not in parameters
    assert "solo_buyer_skill_threshold" not in parameters
    assert "solo_buyer_penalty" not in parameters


def test_stale_member_posterior_exports_are_absent():
    assert not hasattr(member_ranking_api, "_lookup_buyer_posterior_lift")
    assert not hasattr(member_skill_api, "score_members_for_ticker")
    fields = member_skill_api.MemberSkillPosterior.__dataclass_fields__
    assert "sector_skills" not in fields
    assert "ticker_skills" not in fields
    assert "effective_information" in fields
