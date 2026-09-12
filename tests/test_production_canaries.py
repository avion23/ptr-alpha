from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from analyzer.member_ranking.buyer_scoring import (
    CONSENSUS_SCORER_PROVENANCE,
    _filter_equity_rows,
    score_ticker_by_buyers,
)
from analyzer.pipeline import (
    TickerAnalysisParams,
    TickerScoringParams,
    run_recent_ticker_scoring,
    run_ticker_analysis,
)

from .test_buyer_scoring import _transactions


def _official_rows(frame: pd.DataFrame, *, source: str, chamber: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["source"] = source
    frame["chamber"] = chamber
    frame["ticker_origin"] = "official"
    return frame


def test_house_spcx_canary():
    as_of = date(2026, 8, 3)
    transactions = _transactions(
        names=[
            "Jared Moskowitz",
            "John James",
            "John McGuire",
            "William R. Timmons",
        ],
        ticker="SPCX",
        transaction_dates=["2026-06-15", "2026-07-01", "2026-07-15", "2026-07-20"],
        disclosure_dates=["2026-07-06", "2026-07-20", "2026-07-27", "2026-08-03"],
    )
    transactions = _official_rows(transactions, source="house_pdf", chamber="house")
    transactions["instrument_type"] = "stock"
    transactions["raw_asset_class"] = "Stock"
    transactions["asset_description"] = "SPCX Common Stock"

    source = MagicMock()
    source.db.get_transactions_by_date_range.return_value = transactions
    result = run_recent_ticker_scoring(
        source,
        TickerScoringParams(
            year=2026,
            days_back=28,
            min_buyers=3,
            as_of_date=as_of,
        ),
    )

    assert result.success
    assert result.data is not None
    source.db.get_transactions_by_date_range.assert_called_once_with(
        pd.Timestamp("2026-07-06"), pd.Timestamp("2026-08-03")
    )
    recommendations = result.data["result"]
    assert recommendations["ticker"].tolist() == ["SPCX"]
    score = recommendations.iloc[0]
    assert score["num_buyers"] == 4
    assert score["signal_score"] == 4.0
    assert score["signal_score"] == float(score["num_buyers"])
    assert score["scorer_provenance"] == CONSENSUS_SCORER_PROVENANCE

    outside_window = transactions.iloc[:3].copy()
    outside_window.loc[outside_window.index[-1], "transaction_date"] = pd.Timestamp(
        "2026-07-01"
    )
    outside_window.loc[outside_window.index[-1], "disclosure_date"] = pd.Timestamp(
        "2026-07-05"
    )
    source = MagicMock()
    source.db.get_transactions.return_value = outside_window
    no_buy = run_ticker_analysis(
        TickerAnalysisParams(
            ticker="SPCX",
            year=2026,
            days_back=28,
            min_buyers=3,
            as_of_date=as_of,
        ),
        source,
    )

    assert no_buy.success
    assert no_buy.data is not None
    no_buy_score = no_buy.data["score"].iloc[0]
    assert no_buy_score["num_buyers"] == 2
    assert no_buy_score["signal_score"] == 0.0
    assert "minimum buyer threshold" in no_buy_score["note"]


def test_senate_cvx_canary():
    as_of = date(2026, 8, 24)
    transactions = _transactions(
        names=["John Boozman", "Thomas H. Tuberville", "Thomas H. Tuberville"],
        ticker="CVX",
        transaction_dates=["2026-08-01", "2026-08-02", "2026-08-03"],
        disclosure_dates=["2026-08-10", "2026-08-10", "2026-08-11"],
    )
    transactions = _official_rows(
        transactions, source="senate_efd", chamber="senate"
    )
    transactions["instrument_type"] = ["stock", "option", "put"]
    transactions["raw_asset_class"] = ["Stock", "Stock Option", "Put"]
    transactions["asset_description"] = [
        "Chevron Corporation Common Stock",
        "Chevron Corporation Common Stock Option Type: Put",
        "Chevron Corporation Put",
    ]
    transactions["raw_asset_description"] = transactions["asset_description"]

    assert transactions["source"].eq("senate_efd").all()
    assert transactions["chamber"].eq("senate").all()
    eligible = _filter_equity_rows(transactions)
    assert eligible["member"].tolist() == ["John Boozman"]
    assert not eligible["member"].eq("Thomas H. Tuberville").any()

    score = score_ticker_by_buyers(
        "CVX",
        transactions,
        min_buyers=2,
        as_of_date=pd.Timestamp(as_of),
    ).iloc[0]
    assert score["num_buyers"] == 1
    assert score["signal_score"] == 0.0
    assert score["scorer_provenance"] == CONSENSUS_SCORER_PROVENANCE

    source = MagicMock()
    source.db.get_transactions_by_date_range.return_value = transactions
    no_buy = run_recent_ticker_scoring(
        source,
        TickerScoringParams(
            year=2026,
            days_back=28,
            min_buyers=2,
            as_of_date=as_of,
        ),
    )
    assert no_buy.success
    assert no_buy.data is not None
    source.db.get_transactions_by_date_range.assert_called_once_with(
        pd.Timestamp("2026-07-27"), pd.Timestamp("2026-08-24")
    )
    assert no_buy.data["result"].empty
