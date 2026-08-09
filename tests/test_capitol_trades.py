import json
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from analyzer.capitol_trades import CapitolTradesError, CapitolTradesSource


def _trade(**overrides):
    record = {
        "politician_name": "Nancy Pelosi",
        "chamber": "house",
        "state": "CA",
        "party": "D",
        "ticker": "AAPL",
        "asset_name": "Apple Inc. - Common Stock",
        "asset_type": "Stock",
        "transaction_type": "sale",
        "transaction_date": "2025-10-22",
        "disclosure_date": "2025-10-23",
        "amount_text": "$250,001 - $500,000",
        "amount_min": 250001.0,
        "amount_max": 500000.0,
        "filing_url": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2025/20033337.pdf",
        "doc_id": "20033337",
    }
    record.update(overrides)
    return record


def _response(trades, *, page=1, per_page=50, total=None, pages=None):
    total = len(trades) if total is None else total
    pages = max(1, (total + per_page - 1) // per_page) if pages is None else pages
    response = MagicMock()
    response.json.return_value = {
        "trades": trades,
        "page": page,
        "per_page": per_page,
        "pages": pages,
        "total": total,
    }
    response.raise_for_status.return_value = None
    return response


class TestCapitolTradesSource(unittest.TestCase):
    def setUp(self):
        self.source = CapitolTradesSource(data_dir="unused", read_only=True)

    def tearDown(self):
        self.source.close()

    def test_fetch_trades_preserves_reconciliation_provenance(self):
        record = _trade(id=9182)
        self.source.session.get = MagicMock(return_value=_response([record]))

        df = self.source.fetch_trades("Nancy Pelosi")

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["doc_id"], "20033337")
        self.assertEqual(row["member"], "Nancy Pelosi")
        self.assertEqual(row["transaction_type"], "Sale")
        self.assertEqual(row["raw_transaction_subtype"], "sale")
        self.assertEqual(row["amount_midpoint"], 375000.5)
        self.assertEqual(row["chamber"], "house")
        self.assertEqual(row["source_record_id"], "9182")
        self.assertEqual(row["state"], "CA")
        self.assertEqual(row["party"], "D")
        self.assertEqual(row["filing_url"], record["filing_url"])
        self.assertEqual(row["raw_asset_description"], record["asset_name"])
        self.assertEqual(row["raw_asset_class"], "Stock")
        self.assertTrue(pd.isna(row["official_filing_date"]))
        self.assertEqual(row["ticker_origin"], "capitol_trades_api")
        self.assertEqual(len(row["artifact_sha256"]), 64)
        self.assertIn("/politicians/Nancy%20Pelosi/trades", row["source_endpoint"])
        self.assertEqual(json.loads(row["source_params"]), {"page": 1, "per_page": 50})
        self.assertEqual(row["source_page"], 1)
        self.assertEqual(row["source_position"], 1)

    @patch("analyzer.capitol_trades.time.sleep")
    def test_fetch_all_trades_validates_complete_pagination(self, mock_sleep):
        records = [
            _trade(id=1, ticker="AAPL"),
            _trade(id=2, ticker="NVDA", doc_id="20033338"),
            _trade(
                id=3,
                politician_name="Katie Britt",
                chamber="senate",
                state="AL",
                party="R",
                ticker="JPM",
                doc_id="Britt_Katie_01_29_2026",
            ),
        ]
        self.source.session.get = MagicMock(
            side_effect=[
                _response(records[:2], page=1, per_page=2, total=3, pages=2),
                _response(records[2:], page=2, per_page=2, total=3, pages=2),
            ]
        )

        df = self.source.fetch_all_trades()

        self.assertEqual(len(df), 3)
        self.assertEqual(df["source_record_id"].tolist(), ["1", "2", "3"])
        self.assertEqual(df["source_page"].tolist(), [1, 1, 2])
        self.assertEqual(self.source.session.get.call_count, 2)
        mock_sleep.assert_called_once()

    def test_incomplete_response_total_fails_closed(self):
        response = _response([_trade()], page=1, per_page=2, total=3, pages=2)
        self.source.session.get = MagicMock(return_value=response)

        with self.assertRaisesRegex(CapitolTradesError, "Incomplete page 1"):
            self.source.fetch_all_trades()

    def test_wrong_page_number_fails_closed(self):
        response = _response([_trade()], page=2, per_page=50, total=1, pages=1)
        self.source.session.get = MagicMock(return_value=response)

        with self.assertRaisesRegex(CapitolTradesError, "requested 1, received 2"):
            self.source.fetch_all_trades()

    @patch("analyzer.capitol_trades.time.sleep")
    def test_pagination_metadata_change_fails_closed(self, _mock_sleep):
        records = [_trade(id=i, doc_id=str(20000000 + i)) for i in range(1, 5)]
        self.source.session.get = MagicMock(
            side_effect=[
                _response(records[:2], page=1, per_page=2, total=4, pages=2),
                _response(records[2:], page=2, per_page=2, total=5, pages=3),
            ]
        )

        with self.assertRaisesRegex(CapitolTradesError, "metadata changed"):
            self.source.fetch_all_trades()

    @patch("analyzer.capitol_trades.time.sleep")
    def test_repeated_page_content_fails_closed(self, _mock_sleep):
        records = [_trade(ticker="AAPL"), _trade(ticker="NVDA", doc_id="20033338")]
        self.source.session.get = MagicMock(
            side_effect=[
                _response(records, page=1, per_page=2, total=4, pages=2),
                _response(records, page=2, per_page=2, total=4, pages=2),
            ]
        )

        with self.assertRaisesRegex(CapitolTradesError, "repeated page content"):
            self.source.fetch_all_trades()

    @patch("analyzer.capitol_trades.time.sleep")
    def test_repeated_source_record_id_fails_closed(self, _mock_sleep):
        first = [_trade(id=7), _trade(id=8, ticker="NVDA", doc_id="20033338")]
        second = [
            _trade(id=7, ticker="MSFT", doc_id="20033339"),
            _trade(id=9, ticker="JPM", doc_id="20033340"),
        ]
        self.source.session.get = MagicMock(
            side_effect=[
                _response(first, page=1, per_page=2, total=4, pages=2),
                _response(second, page=2, per_page=2, total=4, pages=2),
            ]
        )

        with self.assertRaisesRegex(
            CapitolTradesError, "repeated source record ID '7'"
        ):
            self.source.fetch_all_trades()

    def test_missing_response_metadata_fails_schema(self):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"trades": [_trade()], "total": 1}
        self.source.session.get = MagicMock(return_value=response)

        with self.assertRaisesRegex(CapitolTradesError, "missing response fields"):
            self.source.fetch_all_trades()

    def test_missing_trade_field_fails_schema(self):
        record = _trade()
        del record["chamber"]
        self.source.session.get = MagicMock(return_value=_response([record]))

        with self.assertRaisesRegex(CapitolTradesError, "missing fields .*chamber"):
            self.source.fetch_all_trades()

    def test_invalid_date_fails_schema(self):
        self.source.session.get = MagicMock(
            return_value=_response([_trade(transaction_date="not-a-date")])
        )

        with self.assertRaisesRegex(CapitolTradesError, "invalid transaction_date"):
            self.source.fetch_all_trades()

    def test_invalid_numeric_field_fails_schema(self):
        self.source.session.get = MagicMock(
            return_value=_response([_trade(amount_min="250001")])
        )

        with self.assertRaisesRegex(CapitolTradesError, "amount_min must be numeric"):
            self.source.fetch_all_trades()

    def test_pages_must_match_total_and_per_page(self):
        response = _response([_trade()], page=1, per_page=50, total=1, pages=2)
        self.source.session.get = MagicMock(return_value=response)

        with self.assertRaisesRegex(CapitolTradesError, "Pagination metadata mismatch"):
            self.source.fetch_all_trades()

    def test_chamber_filter_has_real_house_and_senate_scenarios(self):
        records = [
            _trade(id=1, chamber="house"),
            _trade(
                id=2,
                chamber="senate",
                politician_name="Katie Britt",
                doc_id="Britt_Katie_01_29_2026",
            ),
        ]
        self.source.session.get = MagicMock(return_value=_response(records))
        senate = self.source.fetch_all_trades(chamber="senate")
        self.assertEqual(senate["member"].tolist(), ["Katie Britt"])
        self.assertEqual(senate["chamber"].tolist(), ["senate"])

        self.source.session.get = MagicMock(return_value=_response(records))
        house = self.source.fetch_all_trades(chamber="HOUSE")
        self.assertEqual(house["member"].tolist(), ["Nancy Pelosi"])
        self.assertEqual(house["chamber"].tolist(), ["house"])

        with self.assertRaisesRegex(CapitolTradesError, "Invalid chamber"):
            self.source.fetch_all_trades(chamber="joint")

    def test_date_filter_uses_aggregate_disclosure_date(self):
        records = [
            _trade(id=1, disclosure_date="2025-01-02"),
            _trade(id=2, disclosure_date="2025-07-02", doc_id="20033338"),
        ]
        self.source.session.get = MagicMock(return_value=_response(records))

        df = self.source.fetch_all_trades(
            start_date=date(2025, 6, 1), end_date=date(2025, 12, 31)
        )
        self.assertEqual(df["source_record_id"].tolist(), ["2"])

    def test_invalid_date_range_fails(self):
        self.source.session.get = MagicMock(return_value=_response([_trade()]))
        with self.assertRaisesRegex(CapitolTradesError, "end_date must be"):
            self.source.fetch_all_trades(
                start_date=date(2025, 2, 1), end_date=date(2025, 1, 1)
            )

    def test_identical_genuine_lots_get_distinct_stable_synthetic_ids(self):
        lot = _trade(doc_id=None, filing_url=None)

        first = self.source._normalize([lot, dict(lot)])
        second = self.source._normalize([lot, dict(lot)])

        self.assertEqual(first["doc_id"].nunique(), 2)
        self.assertEqual(first["doc_id"].tolist(), second["doc_id"].tolist())
        self.assertTrue(all(value.startswith("ct-house-") for value in first["doc_id"]))

    def test_raw_source_id_keeps_synthetic_id_stable_across_correction(self):
        original = _trade(id=42, doc_id=None, filing_url=None, ticker="APPL")
        corrected = _trade(id=42, doc_id=None, filing_url=None, ticker="AAPL")

        original_id = self.source._normalize([original]).iloc[0]["doc_id"]
        corrected_id = self.source._normalize([corrected]).iloc[0]["doc_id"]

        self.assertEqual(original_id, corrected_id)
        self.assertTrue(original_id.startswith("ct-house-"))

    def test_text_amount_fallback_is_preserved(self):
        df = self.source._normalize(
            [
                _trade(
                    amount_min=None,
                    amount_max=None,
                    amount_text="$15,001 - $50,000",
                )
            ]
        )
        self.assertEqual(df.iloc[0]["amount_midpoint"], 32500.5)

    def test_empty_valid_response_has_full_reconciliation_schema(self):
        self.source.session.get = MagicMock(
            return_value=_response([], page=1, per_page=50, total=0, pages=1)
        )
        df = self.source.fetch_all_trades()
        self.assertTrue(df.empty)
        self.assertIn("chamber", df.columns)
        self.assertIn("source_record_id", df.columns)
        self.assertIn("artifact_sha256", df.columns)

    def test_http_503_is_external_failure_not_empty_success(self):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError(
            "503 no available server"
        )
        self.source.session.get = MagicMock(return_value=response)

        with self.assertRaisesRegex(CapitolTradesError, "503 no available server"):
            self.source.fetch_all_trades()

    def test_canonical_database_write_is_forbidden(self):
        with self.assertRaisesRegex(CapitolTradesError, "reconciliation-only"):
            self.source.save_to_db(pd.DataFrame([{"ticker": "AAPL"}]))

    def test_get_transactions_cannot_masquerade_as_canonical_source(self):
        with self.assertRaisesRegex(CapitolTradesError, "reconciliation-only"):
            self.source.get_transactions(2025)


if __name__ == "__main__":
    unittest.main()
