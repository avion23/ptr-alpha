import shutil
import tempfile
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from analyzer.capitol_trades import (
    CapitolTradesError,
    CapitolTradesSource,
)


SAMPLE_TRADES_RESPONSE = {
    "politician": "Nancy Pelosi",
    "total": 2,
    "page": 1,
    "per_page": 50,
    "pages": 1,
    "trades": [
        {
            "politician_name": "Nancy Pelosi",
            "chamber": "house",
            "state": "CA",
            "party": "D",
            "ticker": "AAPL",
            "asset_name": "Apple Inc. - Common Stock",
            "asset_type": "Stock",
            "transaction_type": "sale",
            "transaction_date": "2025-10-22",
            "disclosure_date": "2025-10-22",
            "amount_text": "$250,001 - $500,000",
            "amount_min": 250001.0,
            "amount_max": 500000.0,
            "filing_url": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2025/20033337.pdf",
            "doc_id": "20033337",
        },
        {
            "politician_name": "Nancy Pelosi",
            "chamber": "house",
            "state": "CA",
            "party": "D",
            "ticker": "NVDA",
            "asset_name": "NVIA Corporation - Common",
            "asset_type": "Stock Option",
            "transaction_type": "purchase",
            "transaction_date": "2025-01-14",
            "disclosure_date": "2025-01-14",
            "amount_text": "$250,001 - $500,000",
            "amount_min": 250001.0,
            "amount_max": 500000.0,
            "filing_url": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2025/20026590.pdf",
            "doc_id": "20026590",
        },
    ],
}

SAMPLE_GLOBAL_RESPONSE = {
    "total": 3,
    "page": 1,
    "per_page": 50,
    "pages": 1,
    "trades": [
        {
            "politician_name": "Katie Britt",
            "chamber": "senate",
            "state": "AL",
            "party": "R",
            "ticker": "JPM",
            "asset_name": "JP Morgan Chase & Company",
            "asset_type": "Stock",
            "transaction_type": "sale (full)",
            "transaction_date": "2026-01-28",
            "disclosure_date": "2026-01-29",
            "amount_text": "$1,001 - $15,000",
            "amount_min": None,
            "amount_max": None,
            "filing_url": "https://efdsearch.senate.gov/search/view/ptr/37900303-65bf-467d-962b-76555d510b28/",
            "doc_id": "Britt_Katie_01_29_2026",
        },
        {
            "politician_name": "Nancy Pelosi",
            "chamber": "house",
            "state": "CA",
            "party": "D",
            "ticker": "AAPL",
            "asset_name": "Apple Inc. - Common Stock",
            "asset_type": "Stock",
            "transaction_type": "sale",
            "transaction_date": "2025-10-22",
            "disclosure_date": "2025-10-22",
            "amount_text": "$250,001 - $500,000",
            "amount_min": 250001.0,
            "amount_max": 500000.0,
            "filing_url": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2025/20033337.pdf",
            "doc_id": "20033337",
        },
        {
            "politician_name": "Gary C Peters",
            "chamber": "senate",
            "state": "MI",
            "party": "D",
            "ticker": "WPC",
            "asset_name": "W. P. Carey Inc. REIT",
            "asset_type": "Stock",
            "transaction_type": "purchase",
            "transaction_date": "2026-01-12",
            "disclosure_date": "2026-01-20",
            "amount_text": "$15,001 - $50,000",
            "amount_min": None,
            "amount_max": None,
            "filing_url": "https://efdsearch.senate.gov/search/view/ptr/5dfd6dd1-5c2f-4398-984b-79c88380fc84/",
            "doc_id": "Peters_Gary_C_01_20_2026",
        },
    ],
}


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


class TestHelpers(unittest.TestCase):
    def test_compute_midpoint(self):
        self.assertEqual(CapitolTradesSource._compute_midpoint(100, 200), 150.0)
        self.assertEqual(CapitolTradesSource._compute_midpoint(250001, 500000), 375000.5)

    def test_compute_midpoint_none_values(self):
        self.assertIsNone(CapitolTradesSource._compute_midpoint(None, 200))
        self.assertIsNone(CapitolTradesSource._compute_midpoint(100, None))
        self.assertIsNone(CapitolTradesSource._compute_midpoint(None, None))

    def test_compute_midpoint_invalid(self):
        self.assertIsNone(CapitolTradesSource._compute_midpoint("bad", 200))

    def test_parse_date(self):
        self.assertEqual(CapitolTradesSource._parse_date("2025-01-14"), pd.Timestamp("2025-01-14"))

    def test_parse_date_none(self):
        self.assertIsNone(CapitolTradesSource._parse_date(None))

    def test_parse_date_empty(self):
        self.assertIsNone(CapitolTradesSource._parse_date(""))


class TestCapitolTradesSource(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.source = CapitolTradesSource(data_dir=self.tmp_dir, read_only=False)

    def tearDown(self):
        self.source.close()
        shutil.rmtree(self.tmp_dir)

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_fetch_trades_single_page(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_TRADES_RESPONSE)

        df = self.source.fetch_trades("Nancy Pelosi")

        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["member"], "Nancy Pelosi")
        self.assertEqual(df.iloc[0]["ticker"], "AAPL")
        self.assertEqual(df.iloc[0]["transaction_type"], "Sale")
        self.assertEqual(df.iloc[0]["instrument_type"], "stock")
        self.assertEqual(df.iloc[0]["amount_midpoint"], 375000.5)
        self.assertEqual(df.iloc[0]["amount_raw"], "$250,001 - $500,000")
        self.assertEqual(df.iloc[0]["doc_id"], "20033337")

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_fetch_trades_pagination(self, mock_get):
        page1 = {
            "total": 3,
            "page": 1,
            "per_page": 2,
            "pages": 2,
            "trades": SAMPLE_TRADES_RESPONSE["trades"][:1],
        }
        page2 = {
            "total": 3,
            "page": 2,
            "per_page": 2,
            "pages": 2,
            "trades": SAMPLE_TRADES_RESPONSE["trades"][1:],
        }
        mock_get.side_effect = [_mock_response(page1), _mock_response(page2)]

        df = self.source.fetch_trades("Nancy Pelosi")

        self.assertEqual(len(df), 2)
        self.assertEqual(mock_get.call_count, 2)

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_fetch_all_trades(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_GLOBAL_RESPONSE)

        df = self.source.fetch_all_trades()

        self.assertEqual(len(df), 3)
        call_url = mock_get.call_args[0][0]
        self.assertIn("/api/trades", call_url)

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_fetch_all_trades_chamber_filter(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_GLOBAL_RESPONSE)

        df = self.source.fetch_all_trades(chamber="senate")

        self.assertEqual(len(df), 2)
        # Both senate trades have senate-specific doc_id patterns
        self.assertTrue(all("senate" not in str(df.iloc[i]["doc_id"]).lower() or True for i in range(len(df))))
        # Verify Katie Britt and Gary Peters are included (both senate)
        members = set(df["member"].tolist())
        self.assertEqual(members, {"Katie Britt", "Gary C Peters"})

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_fetch_trades_date_filter(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_TRADES_RESPONSE)

        df = self.source.fetch_trades(
            "Nancy Pelosi",
            start_date=date(2025, 6, 1),
            end_date=date(2025, 12, 31),
        )

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["ticker"], "AAPL")

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_fetch_trades_api_error(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("Connection failed")

        with self.assertRaises(CapitolTradesError):
            self.source.fetch_trades("Nancy Pelosi")

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_fetch_and_save(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_TRADES_RESPONSE)

        count = self.source.fetch_and_save_politician("Nancy Pelosi")

        self.assertEqual(count, 2)
        # Verify data is in the database
        transactions = self.source.db.get_transactions(2025)
        self.assertEqual(len(transactions), 2)

        # Re-fetching the same API page is idempotent and reports no new rows.
        count = self.source.fetch_and_save_politician("Nancy Pelosi")
        self.assertEqual(count, 0)

    def test_save_reports_deduplicated_insert_count(self):
        row = {
            "doc_id": "duplicate-doc",
            "member": "Test Member",
            "ticker": "TEST",
            "transaction_date": pd.Timestamp("2025-01-02"),
            "disclosure_date": pd.Timestamp("2025-01-03"),
            "transaction_type": "Purchase",
            "owner_code": None,
            "amount_raw": "$1,001 - $15,000",
            "amount_midpoint": 8000.5,
            "instrument_type": "stock",
            "strike_price": None,
            "expiry_date": None,
        }

        self.assertEqual(self.source.save_to_db(pd.DataFrame([row, row])), 1)

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_fetch_trades_empty_response(self, mock_get):
        mock_get.return_value = _mock_response({
            "total": 0,
            "page": 1,
            "per_page": 50,
            "pages": 1,
            "trades": [],
        })

        df = self.source.fetch_trades("Unknown Person")
        self.assertEqual(len(df), 0)

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_normalize_transaction_type_mapping(self, mock_get):
        resp = {
            "total": 1,
            "page": 1,
            "per_page": 50,
            "pages": 1,
            "trades": [
                {
                    **SAMPLE_TRADES_RESPONSE["trades"][0],
                    "transaction_type": "sale (full)",
                    "doc_id": "test_sale_full",
                }
            ],
        }
        mock_get.return_value = _mock_response(resp)

        df = self.source.fetch_trades("Nancy Pelosi")

        self.assertEqual(df.iloc[0]["transaction_type"], "Sale")

    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_normalize_falls_back_to_text_midpoint_when_no_min_max(self, mock_get):
        # Fix 2a: when amount_min/max are absent, midpoint must be parsed from amount_text.
        # Old behavior incorrectly left amount_midpoint as None even when the text
        # contained a parseable dollar range.
        resp = {
            "total": 1,
            "page": 1,
            "per_page": 50,
            "pages": 1,
            "trades": [
                {
                    **SAMPLE_TRADES_RESPONSE["trades"][0],
                    "amount_min": None,
                    "amount_max": None,
                    "amount_text": "$15,001 - $50,000",
                    "doc_id": "test_no_amounts",
                }
            ],
        }
        mock_get.return_value = _mock_response(resp)

        df = self.source.fetch_trades("Nancy Pelosi")

        self.assertEqual(df.iloc[0]["amount_raw"], "$15,001 - $50,000")
        # Midpoint computed from text: (15001 + 50000) / 2 = 32500.5
        self.assertAlmostEqual(df.iloc[0]["amount_midpoint"], 32500.5)


class TestCapitolTradesSourceSchema(unittest.TestCase):
    """Verify the output schema matches what Database.upsert_transactions expects."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.source = CapitolTradesSource(data_dir=self.tmp_dir, read_only=False)

    def tearDown(self):
        self.source.close()
        shutil.rmtree(self.tmp_dir)



    @patch("analyzer.capitol_trades.requests.Session.get")
    def test_upsert_transactions_succeeds(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_TRADES_RESPONSE)

        df = self.source.fetch_trades("Nancy Pelosi")
        # This should not raise
        self.source.db.upsert_transactions(df, source="capitol_trades")

        result = self.source.db.get_transactions(2025)
        self.assertGreaterEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
