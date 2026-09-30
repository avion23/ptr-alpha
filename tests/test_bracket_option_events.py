"""Bracket option markers must not enter equity event scoring."""

import pandas as pd

from analyzer.member_ranking.buyer_scoring import _equity_transaction_row


def _row(description, instrument=None):
    return pd.Series(
        {
            "transaction_type": "Purchase",
            "transaction_date": "2025-01-14",
            "disclosure_date": "2025-01-17",
            "ticker": "VST",
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
