import inspect

import pandas as pd
import pytest

from analyzer.exceptions import AnalysisError
from analyzer.member_ranking.buyer_scoring import (
    _UNSUPPORTED_ASSET_RE,
    _equity_transaction_row,
    PTR_PURCHASE_BELOW_FLOOR_REASON,
    get_consensus_candidate_tickers,
    prepare_consensus_purchases,
    purchase_below_ptr_floor_reason,
    score_ticker_by_buyers,
)


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

    assert result.iloc[0]["signal_score"] == 1.0


@pytest.mark.parametrize(
    "description",
    [
        "bitcoin",
        "BTC",
        "Ethereum",
        "Litecoin",
        "Solana",
        "Ripple XRP",
        "USDC",
        "Ski Mask Dog",
        "The Graph",
        "Sui",
        "TZA ETF",
    ],
)
def test_equity_guard_rejects_crypto_names_and_etf_token(description):
    row = pd.Series(
        {
            "ticker": "X",
            "source": "house_pdf",
            "asset_description": description,
            "instrument_type": "stock",
        }
    )
    assert not _equity_transaction_row(row)


def test_equity_guard_unsupported_tokens_are_case_insensitive_and_word_bounded():
    rejected = pd.Series(
        {
            "ticker": "X",
            "source": "house_pdf",
            "asset_description": "Ethereum [cT]",
            "instrument_type": "stock",
        }
    )
    accepted = pd.Series(
        {
            "ticker": "ETF",
            "source": "house_pdf",
            "asset_description": "ETFish Common Stock",
            "instrument_type": "stock",
        }
    )
    assert not _equity_transaction_row(rejected)
    assert _equity_transaction_row(accepted)


@pytest.mark.parametrize(
    "description",
    [
        "Alphabet Inc Note Coupon 4.80000% Mat 2/15/2036 [CS] Description: Corporate Note",
        "JPMorgan Chase BK N A INSTL CTF DEP ... [CS] Description: Corporate Bond",
        "Pacific Co B1 Bond [CS] ... Real estate bond",
        "FEDERAL NATIONAL MORTGAGE ASSOCIATION POOL ... [CS]",
        "Lord Abbett Short Duration High Yield I (LSYIX) bond fund [CS]",
    ],
)
def test_equity_guard_rejects_corporate_securities_marker(description):
    row = pd.Series(
        {
            "ticker": "X",
            "source": "house_pdf",
            "asset_description": description,
            "instrument_type": "stock",  # stale canonical classification
        }
    )
    assert not _equity_transaction_row(row)


@pytest.mark.parametrize(
    "ticker,description,eligible",
    [
        ("IBIT", "iShares Bitcoin Trust ETF (IBIT) [ST]", False),
        ("ETHE", "Grayscale Ethereum Trust ETF (ETHE) [ST]", False),
        ("FBTC", "Fidelity Wise Origin Bitcoin Fund (FBTC) [ST]", False),
        ("SPCX", "SpaceX Class A Common Stock [ST]", True),
        ("IBIT", "Apple Inc. Class A Common Stock [ST]", True),
        ("BTCM", "Bitcoin Mining Co Class A Common Stock [ST]", True),
    ],
)
def test_crypto_vehicle_is_the_narrow_exception_to_authoritative_stock_marker(
    ticker, description, eligible
):
    row = pd.Series(
        {
            "ticker": ticker,
            "source": "house_pdf",
            "asset_description": description,
            "instrument_type": "stock",
        }
    )
    assert _equity_transaction_row(row) is eligible


def test_non_equity_marker_regex_does_not_match_stock_or_option_markers():
    assert _UNSUPPORTED_ASSET_RE.search("Common Stock [ST]") is None
    assert _UNSUPPORTED_ASSET_RE.search("Call Option [OP]") is None


def test_consensus_score_is_invariant_to_member_identity_shuffle():
    transactions = _transactions()
    shuffled = transactions.copy()
    shuffled["member"] = ["Xavier Able", "Zelda Baker", "Yvonne Carter"]
    as_of = pd.Timestamp("2024-05-20")

    original = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=as_of, min_buyers=1
    )
    permuted = score_ticker_by_buyers("AAPL", shuffled, as_of_date=as_of, min_buyers=1)

    assert original.iloc[0]["signal_score"] == permuted.iloc[0]["signal_score"]
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

    assert result.iloc[0]["signal_score"] == 2.0
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

    assert early.iloc[0]["signal_score"] == 2.0
    assert late.iloc[0]["signal_score"] == 2.0


def test_consensus_excludes_blank_canonical_member_identities_before_counting():
    result = score_ticker_by_buyers(
        "AAPL",
        _transactions(("Alice", "  ", "\t")),
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=2,
    )

    assert result.iloc[0]["num_buyers"] == 1
    assert "minimum buyer threshold" in result.iloc[0]["note"]


def test_consensus_aliases_do_not_create_a_false_three_buyer_quorum():
    transactions = _transactions(
        ("Chuck Fleischmann", "Charles J. Fleischmann", "Nancy Pelosi"),
        disclosure_dates=["2024-05-10", "2024-05-12", "2024-05-14"],
    )
    transactions["source"] = ["house_pdf", "senate_efd", "house_pdf"]

    prepared = prepare_consensus_purchases(transactions)
    result = score_ticker_by_buyers(
        "AAPL",
        transactions,
        as_of_date=pd.Timestamp("2024-05-20"),
        min_buyers=3,
    )

    assert len(prepared) == 3
    assert prepared["member"].tolist() == transactions["member"].tolist()
    assert result.iloc[0]["num_buyers"] == 2
    assert result.iloc[0]["signal_score"] == 0.0
    assert "minimum buyer threshold" in result.iloc[0]["note"]
    assert get_consensus_candidate_tickers(
        transactions, 3, as_of_date=pd.Timestamp("2024-05-20")
    ) == []

    from analyzer.pipeline import _consensus_buyers_table

    buyers = _consensus_buyers_table("AAPL", transactions)
    assert len(buyers) == 2
    assert sorted(buyers["num_purchases"].tolist()) == [1, 2]


def test_consensus_alias_grouping_is_order_independent_and_preserves_quorum():
    transactions = _transactions(
        ("Chuck Fleischmann", "Charles J. Fleischmann", "Nancy Pelosi", "Alice Smith"),
        disclosure_dates=["2024-05-10", "2024-05-12", "2024-05-14", "2024-05-16"],
    )
    transactions["source"] = ["house_pdf", "senate_efd", "house_pdf", "senate_efd"]
    as_of = pd.Timestamp("2024-05-20")

    original = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=as_of, min_buyers=3
    ).iloc[0]
    reversed_rows = score_ticker_by_buyers(
        "AAPL", transactions.iloc[::-1].reset_index(drop=True),
        as_of_date=as_of,
        min_buyers=3,
    ).iloc[0]

    assert original["num_buyers"] == reversed_rows["num_buyers"] == 3
    assert original["signal_score"] == reversed_rows["signal_score"] == 3.0
    assert original["buyers"] == reversed_rows["buyers"]


def test_ambiguous_alias_cannot_bridge_distinct_full_names():
    transactions = _transactions(
        (
            "Charles J. Fleischmann",
            "Chuck Fleischmann",
            "Charles K. Fleischmann",
        ),
        disclosure_dates=["2024-05-10", "2024-05-11", "2024-05-12"],
    )
    transactions["source"] = ["house_pdf", "senate_efd", "house_pdf"]
    prepared = prepare_consensus_purchases(transactions)

    result = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=pd.Timestamp("2024-05-20"), min_buyers=3
    )

    assert len(prepared) == len(transactions) == 3
    assert prepared["member"].tolist() == [
        "Charles J. Fleischmann",
        "Chuck Fleischmann",
        "Charles K. Fleischmann",
    ]
    assert prepared.loc[
        prepared["member"] == "Chuck Fleischmann", "_member_canonical"
    ].item() == ""
    assert result.iloc[0]["num_buyers"] == 2
    assert result.iloc[0]["signal_score"] == 0.0
    assert "minimum buyer threshold" in result.iloc[0]["note"]
    assert get_consensus_candidate_tickers(
        transactions, 3, as_of_date=pd.Timestamp("2024-05-20")
    ) == []


def test_known_missing_member_marker_is_not_a_consensus_buyer():
    transactions = _transactions(
        ("Alice Smith", "Bob Jones", "N/A"),
        disclosure_dates=["2024-05-10", "2024-05-12", "2024-05-14"],
    )
    transactions["source"] = ["house_pdf", "senate_efd", "house_pdf"]

    prepared = prepare_consensus_purchases(transactions)
    result = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=pd.Timestamp("2024-05-20"), min_buyers=3
    ).iloc[0]

    assert transactions["member"].tolist() == ["Alice Smith", "Bob Jones", "N/A"]
    assert prepared["member"].tolist() == ["Alice Smith", "Bob Jones"]
    assert result["num_buyers"] == 2
    assert result["signal_score"] == 0.0
    assert (
        get_consensus_candidate_tickers(
            transactions, 3, as_of_date=pd.Timestamp("2024-05-20")
        )
        == []
    )


def test_full_middle_names_remain_distinct_and_ambiguous_nickname_is_excluded():
    transactions = _transactions(
        (
            "Charles James Fleischmann",
            "Charles Joseph Fleischmann",
            "Chuck Fleischmann",
            "Alice Smith",
        ),
        disclosure_dates=["2024-05-10", "2024-05-11", "2024-05-12", "2024-05-13"],
    )
    transactions["source"] = ["house_pdf", "senate_efd", "house_pdf", "senate_efd"]
    as_of = pd.Timestamp("2024-05-20")

    prepared = prepare_consensus_purchases(transactions)
    member_keys = prepared.set_index("member")["_member_canonical"]
    original = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=as_of, min_buyers=3
    ).iloc[0]
    reversed_rows = score_ticker_by_buyers(
        "AAPL",
        transactions.iloc[::-1].reset_index(drop=True),
        as_of_date=as_of,
        min_buyers=3,
    ).iloc[0]

    assert member_keys["Charles James Fleischmann"] == "CHARLES JAMES FLEISCHMANN"
    assert member_keys["Charles Joseph Fleischmann"] == "CHARLES JOSEPH FLEISCHMANN"
    assert member_keys["Chuck Fleischmann"] == ""
    assert original["num_buyers"] == reversed_rows["num_buyers"] == 3
    assert original["signal_score"] == reversed_rows["signal_score"] == 3.0
    assert original["buyers"] == reversed_rows["buyers"]


def test_all_full_middle_names_remain_distinct_in_order_independent_groups():
    transactions = _transactions(
        (
            "Charles James Allen Smith",
            "Charles James Arthur Smith",
            "Chuck Smith",
            "Alice Jones",
        ),
        disclosure_dates=["2024-05-10", "2024-05-11", "2024-05-12", "2024-05-13"],
    )
    as_of = pd.Timestamp("2024-05-20")

    prepared = prepare_consensus_purchases(transactions)
    member_keys = prepared.set_index("member")["_member_canonical"]
    original = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=as_of, min_buyers=3
    ).iloc[0]
    reversed_rows = score_ticker_by_buyers(
        "AAPL",
        transactions.iloc[::-1].reset_index(drop=True),
        as_of_date=as_of,
        min_buyers=3,
    ).iloc[0]

    assert member_keys["Charles James Allen Smith"] == "CHARLES JAMES ALLEN SMITH"
    assert member_keys["Charles James Arthur Smith"] == "CHARLES JAMES ARTHUR SMITH"
    assert member_keys["Chuck Smith"] == ""
    assert original["num_buyers"] == reversed_rows["num_buyers"] == 3
    assert original["buyers"] == reversed_rows["buyers"]


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


def test_ptr_purchase_floor_only_blocks_positive_subfloor_congressional_purchases():
    transactions = _transactions(
        ("Exact Dollar", "Band A", "Unknown Amount", "Boundary", "Small Sale"),
        disclosure_dates=[
            "2024-05-10",
            "2024-05-11",
            "2024-05-12",
            "2024-05-13",
            "2024-05-14",
        ],
    )
    transactions["source"] = [
        "house_pdf",
        "gemini_ocr",
        "senate_efd",
        "house_pdf",
        "senate_efd",
    ]
    transactions["amount_midpoint"] = [898.16, 8000.50, None, 1001.0, 898.16]
    transactions.loc[4, "transaction_type"] = "Sale"

    prepared = prepare_consensus_purchases(transactions)
    prepared_with_blocked = prepare_consensus_purchases(
        transactions, include_below_floor=True
    )
    result = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=pd.Timestamp("2024-05-20"), min_buyers=3
    )

    assert prepared["member"].tolist() == ["Band A", "Unknown Amount", "Boundary"]
    assert (
        prepared_with_blocked.set_index("member").loc[
            "Exact Dollar", "_ptr_floor_blocked_reason"
        ]
        == PTR_PURCHASE_BELOW_FLOOR_REASON
    )
    assert pd.isna(
        prepared_with_blocked.set_index("member").loc[
            "Band A", "_ptr_floor_blocked_reason"
        ]
    )
    assert purchase_below_ptr_floor_reason("senate_efd", "Sale", 898.16) is None
    assert purchase_below_ptr_floor_reason("form4", "Purchase", 898.16) is None
    assert result.iloc[0]["num_buyers"] == 3
    assert get_consensus_candidate_tickers(
        transactions, 3, as_of_date=pd.Timestamp("2024-05-20")
    ) == ["AAPL"]


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


def test_consensus_private_safe_and_existing_fund_boundary():
    from unittest.mock import patch

    from analyzer.persistence.database import Database
    from analyzer.pipeline import eligible_events

    transactions = _transactions(
        ("Stock", "Private", "Raw SAFE", "Fund"),
        disclosure_dates=["2024-05-10"] * 4,
    )
    transactions["instrument_type"] = ["stock", "private", "stock", "fund"]
    transactions["asset_description"] = [
        "Apple (AAPL) [ST]", "Private security", "LLC Description: convertible SAFE", "Fund",
    ]
    result = score_ticker_by_buyers(
        "AAPL", transactions, as_of_date=pd.Timestamp("2024-05-20"), min_buyers=1
    )
    assert result.iloc[0]["num_buyers"] == 1
    transactions["source"] = "house_pdf"
    with Database(":memory:") as db:
        with patch.object(
            db, "get_transactions_by_date_range", return_value=transactions
        ):
            events = eligible_events(db, pd.Timestamp("2024-05-20"), days_back=None)
    assert events["actor_id"].tolist() == ["congress:STOCK"]


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


def test_removed_historical_scoring_parameters_are_absent_from_public_api():
    parameters = inspect.signature(score_ticker_by_buyers).parameters
    for removed in (
        "signals_df",
        "horizon",
        "threshold",
        "member_rankings",
        "ticker_perf_signals",
        "_ranking_dicts",
        "scoring_mode",
        "member_skills",
        "uncertainty_penalty_lambda",
        "solo_buyer_skill_threshold",
        "solo_buyer_penalty",
    ):
        assert removed not in parameters
