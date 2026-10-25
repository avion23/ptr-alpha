"""Bracket option markers must not enter equity event scoring."""

import pandas as pd
import pytest

from analyzer.member_ranking.buyer_scoring import _equity_transaction_row


def _row(description, instrument=None, ticker="VST"):
    return pd.Series(
        {
            "transaction_type": "Purchase",
            "transaction_date": "2025-01-14",
            "disclosure_date": "2025-01-17",
            "ticker": ticker,
            "member": "Nancy Pelosi",
            "source": "house_pdf",
            "asset_description": description,
            "instrument_type": instrument,
        }
    )


def test_bracket_op_description_is_not_equity():
    assert not _equity_transaction_row(
        _row("Vistra Corp. Common Stock (VST) [OP]")
    )


def test_bracket_st_description_is_equity():
    assert _equity_transaction_row(
        _row("Vistra Corp. Common Stock (VST) [ST]", instrument="stock")
    )


def test_plain_common_stock_is_equity():
    assert _equity_transaction_row(_row("Vistra Corp. Common Stock"))


@pytest.mark.parametrize(
    ("ticker", "description"),
    [
        ("BTC", "BTC [CT]"),
        ("BILL", "US Treas Bill [GS]"),
        ("UNIT", "UNITED STATES TREAS ... [GS]"),
        ("ALGN", "Elementary School District GO Bonds [GS]"),
        ("TZA", "TZA ETF [OT]"),
        ("IBIT", "BTC [CT]"),
        ("SPCX", "SpaceX pre-IPO purchase [OT]"),
    ],
)
def test_house_non_equity_markers_reject_stock_ticker_bindings(ticker, description):
    assert not _equity_transaction_row(
        _row(description, instrument="stock", ticker=ticker)
    )


def test_stock_marker_remains_equity_with_crypto_trailing_note():
    assert _equity_transaction_row(
        _row(
            "Common Stock (BRK.B) [ST] F... New Bitcoin [CT]",
            instrument="stock",
            ticker="BRK.B",
        )
    )
