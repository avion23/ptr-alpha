import inspect

import pandas as pd
import pytest

from analyzer.exceptions import AnalysisError
from analyzer.member_ranking.buyer_scoring import score_ticker_by_buyers


def _transactions(
    names=("Alice", "Bob", "Carol"),
    ticker="AAPL",
    disclosure_dates=None,
    transaction_dates=None,
    transaction_types=None,
) -> pd.DataFrame:
    names = list(names)
    disclosure_dates = (
        disclosure_dates
        or [
            "2024-05-10",
            "2024-05-12",
            "2024-05-14",
        ][: len(names)]
    )
    transaction_dates = transaction_dates or disclosure_dates
    transaction_types = transaction_types or ["Purchase"] * len(names)
    return pd.DataFrame(
        {
            "member": names,
            "ticker": [ticker] * len(names),
            "transaction_date": pd.to_datetime(transaction_dates),
            "disclosure_date": pd.to_datetime(disclosure_dates),
            "transaction_type": transaction_types,
        }
    )


def test_consensus_cold_start_needs_no_rankings_or_signal_history():
    result = score_ticker_by_buyers(
        "AAPL",
        _transactions(("Alice",)),
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=1,
    )

    assert result.iloc[0]["signal_score_raw"] == 1.0
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


def test_consensus_reports_long_filing_lag_without_penalizing_score():
    transactions = _transactions(
        ("Alice", "Bob"),
        disclosure_dates=["2024-05-10", "2024-05-12"],
        transaction_dates=["2024-01-01", "2024-05-11"],
    )
    result = score_ticker_by_buyers(
        "AAPL",
        transactions,
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=1,
    )

    assert result.iloc[0]["signal_score_raw"] == 2.0
    assert result.iloc[0]["max_trade_to_disclosure_days"] == 130
    assert result.iloc[0]["median_trade_to_disclosure_days"] == 65.5
    assert result.iloc[0]["oldest_transaction_date"] == pd.Timestamp("2024-01-01").date()


def test_consensus_excludes_impossible_or_missing_trade_chronology():
    transactions = _transactions(
        ("Valid", "After Disclosure", "Missing"),
        disclosure_dates=["2024-05-10", "2024-05-10", "2024-05-10"],
        transaction_dates=["2024-05-01", "2024-05-11", None],
    )
    result = score_ticker_by_buyers(
        "AAPL",
        transactions,
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=1,
    )

    assert result.iloc[0]["num_buyers"] == 1
    assert result.iloc[0]["buyers"] == "VALID"
    assert result.iloc[0]["max_trade_to_disclosure_days"] == 9


def test_consensus_has_no_hidden_age_decay_inside_candidate_window():
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

    assert early.iloc[0]["signal_score_raw"] == 2.0
    assert late.iloc[0]["signal_score_raw"] == 2.0


def test_consensus_excludes_blank_canonical_member_identities_before_counting():
    result = score_ticker_by_buyers(
        "AAPL",
        _transactions(("Alice", "  ", "\t")),
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=2,
    )

    assert result.iloc[0]["num_buyers"] == 1
    assert "minimum buyer threshold" in result.iloc[0]["note"]


@pytest.mark.parametrize("ticker", ["", "   ", "NOT_A_TICKER", "CASH", "BOND", "SP"])
def test_consensus_rejects_invalid_or_non_equity_tickers(ticker):
    with pytest.raises(AnalysisError, match="ticker|Ticker"):
        score_ticker_by_buyers(
            ticker,
            _transactions(ticker=ticker),
            as_of_date=pd.Timestamp("2024-05-20"),
            min_buyers=1,
        )


def test_consensus_excludes_sales_before_counting_buyers():
    result = score_ticker_by_buyers(
        "AAPL",
        _transactions(
            ("Alice", "Bob", "Carol"),
            transaction_types=["Purchase", "Sale", "Purchase"],
        ),
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=1,
    )

    assert result.iloc[0]["num_buyers"] == 2
    assert "total_buyer_trades" not in result.columns
    assert "BOB" not in result.iloc[0]["buyers"]


def test_consensus_excludes_non_equity_provenance_rows_before_counting():
    transactions = _transactions(("Alice", "Bob", "Carol"))
    transactions["ticker_origin"] = ["official", "non_equity", "official"]

    result = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=pd.Timestamp("2024-05-20"), min_buyers=1
    )

    assert result.iloc[0]["num_buyers"] == 2


def test_consensus_excludes_option_instrument_rows_before_counting():
    transactions = _transactions(("Alice", "Option Buyer", "Carol"))
    transactions["instrument_type"] = ["stock", "call", "put"]

    result = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=pd.Timestamp("2024-05-20"), min_buyers=1
    )

    assert result.iloc[0]["num_buyers"] == 1


def test_consensus_rejects_misclassified_stock_option_from_raw_evidence():
    transactions = _transactions(("Stock Buyer", "Class Option", "Legacy Option"))
    transactions["instrument_type"] = ["stock", "stock", "stock"]
    transactions["raw_asset_class"] = ["Stock", "Stock Option", None]
    transactions["asset_description"] = [
        "Apple Inc Common Stock",
        "Apple Inc Common Stock",
        "Apple Inc Common StockOption Type: PutStrike price:$145.00",
    ]

    result = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=pd.Timestamp("2024-05-20"), min_buyers=1
    )

    assert result.iloc[0]["num_buyers"] == 1
    assert result.iloc[0]["buyers"] == "STOCK BUYER"


def test_consensus_excludes_future_disclosures_before_counting_buyers():
    result = score_ticker_by_buyers(
        "AAPL",
        _transactions(
            ("Alice", "Bob", "Carol"),
            disclosure_dates=["2024-05-10", "2024-05-12", "2024-05-21"],
        ),
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=1,
    )

    assert result.iloc[0]["num_buyers"] == 2


@pytest.mark.parametrize(
    ("requested_ticker", "stored_ticker"),
    [("AAPL", "AAPL"), ("brk.b", "BRK.B"), ("BRK", "BRK.B"), ("FB", "FB")],
)
def test_consensus_accepts_valid_symbols_and_resolver_aliases(
    requested_ticker, stored_ticker
):
    result = score_ticker_by_buyers(
        requested_ticker,
        _transactions(ticker=stored_ticker),
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=1,
    )

    assert result.iloc[0]["num_buyers"] == 3
    assert result.iloc[0]["scoring_mode"] == "consensus"


def test_historical_modes_remain_descriptive_without_consensus_cutoff():
    transactions = _transactions(
        ("Alice", "Bob"),
        disclosure_dates=["2024-05-10", "2025-05-12"],
    )
    rankings = pd.DataFrame(
        {
            "member": ["ALICE", "BOB"],
            "shrunk_alpha": [10.0, 20.0],
            "purchase_trades": [2, 3],
        }
    )

    result = score_ticker_by_buyers(
        "AAPL",
        transactions,
        signals_df=pd.DataFrame({"member": ["training"]}),
        member_rankings=rankings,
        scoring_mode="shrunk_alpha",
        min_buyers=1,
    )

    assert result.iloc[0]["num_buyers"] == 2
    assert result.iloc[0]["scoring_mode"] == "shrunk_alpha"


def test_historical_modes_keep_exact_ticker_and_member_lookup_behavior():
    transactions = _transactions(("Alice", " "), ticker="aapl")
    rankings = pd.DataFrame(
        {
            "member": ["ALICE"],
            "shrunk_alpha": [10.0],
            "purchase_trades": [2],
        }
    )

    result = score_ticker_by_buyers(
        "aapl",
        transactions,
        signals_df=pd.DataFrame({"member": ["training"]}),
        member_rankings=rankings,
        scoring_mode="shrunk_alpha",
        min_buyers=1,
    )

    assert result.iloc[0]["ticker"] == "aapl"
    assert result.iloc[0]["num_buyers"] == 2


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
