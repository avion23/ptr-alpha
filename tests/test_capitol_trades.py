import hashlib
import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
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


def _response(trades, *, page=1, per_page=50, total=None, pages=None, content=None):
    total = len(trades) if total is None else total
    pages = max(1, (total + per_page - 1) // per_page) if pages is None else pages
    data = {
        "trades": trades,
        "page": page,
        "per_page": per_page,
        "pages": pages,
        "total": total,
    }
    response = MagicMock()
    response.content = (
        content or json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    )
    response.json.return_value = data
    response.raise_for_status.return_value = None
    return response


class TestCapitolTradesSource(unittest.TestCase):
    def setUp(self):
        self.source = CapitolTradesSource(
            data_dir="unused", read_only=True, generation="test-generation"
        )

    def tearDown(self):
        self.source.close()

    def test_generation_is_required(self):
        for generation in ("", "   ", None):
            with (
                self.subTest(generation=generation),
                self.assertRaisesRegex(CapitolTradesError, "generation is required"),
            ):
                CapitolTradesSource(generation=generation)

    def test_fetch_preserves_reconciliation_provenance_and_raw_page_hash(self):
        response = _response([_trade(id=9182)])
        self.source.session.get = MagicMock(return_value=response)

        df = self.source.fetch_trades("Nancy Pelosi")

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        page_hash = hashlib.sha256(response.content).hexdigest()
        self.assertEqual(row["artifact_sha256"], page_hash)
        self.assertEqual(row["ingestion_generation"], "test-generation")
        self.assertEqual(row["doc_id"], "20033337")
        self.assertEqual(row["member"], "Nancy Pelosi")
        self.assertEqual(row["transaction_type"], "Sale")
        self.assertEqual(row["raw_transaction_subtype"], "sale")
        self.assertEqual(row["amount_midpoint"], 375000.5)
        self.assertEqual(row["chamber"], "house")
        self.assertEqual(row["source_record_id"], "9182")
        self.assertEqual(row["state"], "CA")
        self.assertEqual(row["party"], "D")
        self.assertEqual(row["filing_url"], _trade()["filing_url"])
        self.assertEqual(row["raw_asset_description"], _trade()["asset_name"])
        self.assertEqual(row["raw_asset_class"], "Stock")
        self.assertTrue(pd.isna(row["official_filing_date"]))
        self.assertEqual(row["ticker_origin"], "source_reported")
        self.assertIn("/politicians/Nancy%20Pelosi/trades", row["source_endpoint"])
        self.assertEqual(json.loads(row["source_params"]), {"page": 1, "per_page": 50})
        self.assertEqual(
            self.source.last_page_artifacts[0]["artifact_sha256"], page_hash
        )
        self.assertEqual(
            self.source.last_page_artifacts[0]["response_bytes"], len(response.content)
        )

    @patch("analyzer.capitol_trades.time.sleep")
    def test_fetch_all_validates_complete_pagination(self, mock_sleep):
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
        self.assertEqual(len(self.source.last_page_artifacts), 2)
        mock_sleep.assert_called_once()

    def test_incomplete_total_fails_closed(self):
        self.source.session.get = MagicMock(
            return_value=_response([_trade()], page=1, per_page=2, total=3, pages=2)
        )
        with self.assertRaisesRegex(CapitolTradesError, "Incomplete page 1"):
            self.source.fetch_all_trades()

    def test_wrong_page_number_fails_closed(self):
        self.source.session.get = MagicMock(
            return_value=_response([_trade()], page=2, total=1, pages=1)
        )
        with self.assertRaisesRegex(CapitolTradesError, "requested 1, received 2"):
            self.source.fetch_all_trades()

    @patch("analyzer.capitol_trades.time.sleep")
    def test_pagination_metadata_change_fails_closed(self, _sleep):
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
    def test_repeated_exact_raw_page_bytes_fail_closed(self, _sleep):
        records = [_trade(id=1), _trade(id=2, doc_id="20033338")]
        raw = b"exact repeated upstream page bytes"
        self.source.session.get = MagicMock(
            side_effect=[
                _response(records, page=1, per_page=2, total=4, pages=2, content=raw),
                _response(
                    [_trade(id=3), _trade(id=4, doc_id="20033339")],
                    page=2,
                    per_page=2,
                    total=4,
                    pages=2,
                    content=raw,
                ),
            ]
        )
        with self.assertRaisesRegex(CapitolTradesError, "repeated raw page bytes"):
            self.source.fetch_all_trades()

    def test_duplicate_no_id_records_on_one_page_are_ambiguous(self):
        record = _trade(doc_id=None)
        self.source.session.get = MagicMock(
            return_value=_response([record, dict(record)])
        )
        with self.assertRaisesRegex(CapitolTradesError, "without a stable source ID"):
            self.source.fetch_all_trades()

    @patch("analyzer.capitol_trades.time.sleep")
    def test_duplicate_no_id_record_across_pages_is_ambiguous(self, _sleep):
        duplicate = _trade(doc_id=None)
        first = [duplicate, _trade(doc_id=None, ticker="NVDA")]
        second = [dict(duplicate), _trade(doc_id=None, ticker="MSFT")]
        self.source.session.get = MagicMock(
            side_effect=[
                _response(first, page=1, per_page=2, total=4, pages=2),
                _response(second, page=2, per_page=2, total=4, pages=2),
            ]
        )
        with self.assertRaisesRegex(CapitolTradesError, "without a stable source ID"):
            self.source.fetch_all_trades()

    def test_id_sentinels_collide_without_nulling_non_id_text(self):
        first = _trade(doc_id=None)
        second = _trade(doc_id=" null ")
        self.source.session.get = MagicMock(return_value=_response([first, second]))

        with self.assertRaisesRegex(CapitolTradesError, "without a stable source ID"):
            self.source.fetch_all_trades()

    def test_source_id_sentinels_are_missing_before_duplicate_detection(self):
        first = _trade(id=None, doc_id="filing-1")
        second = _trade(id=" None ", doc_id="filing-1")
        self.source.session.get = MagicMock(return_value=_response([first, second]))

        with self.assertRaisesRegex(CapitolTradesError, "without a stable source ID"):
            self.source.fetch_all_trades()

    def test_logically_identical_no_id_rows_have_stable_ids_across_runs(self):
        first = _trade(
            doc_id=None,
            politician_name=" Nancy   Pelosi ",
            chamber=" HOUSE ",
            state="ca",
            party="d",
            ticker=" nan ",
            asset_name=" Apple   Inc. ",
            asset_type=" STOCK ",
            transaction_type=" Sale ",
            amount_text="$250,001  -  $500,000",
            filing_url=" HTTPS://EXAMPLE.TEST/FILING ",
        )
        self.source.session.get = MagicMock(return_value=_response([first]))
        first_id = self.source.fetch_all_trades().iloc[0]["doc_id"]

        second_source = CapitolTradesSource(generation="next-generation")
        self.addCleanup(second_source.close)
        second_source.session.get = MagicMock(
            return_value=_response(
                [
                    _trade(
                        doc_id=" none ",
                        politician_name="Nancy Pelosi",
                        chamber="house",
                        state="CA",
                        party="D",
                        ticker="NAN",
                        asset_name="apple inc.",
                        asset_type="stock",
                        transaction_type="sale",
                        filing_url="https://example.test/filing",
                    )
                ]
            )
        )
        second_id = second_source.fetch_all_trades().iloc[0]["doc_id"]

        self.assertEqual(first_id, second_id)
        self.assertTrue(first_id.startswith("ct-house-"))

    def test_nan_ticker_and_literal_null_asset_text_are_not_id_sentinels(self):
        records = [
            _trade(
                doc_id=None,
                ticker=" NAN ",
                asset_name=" None ",
                asset_type=" null ",
            ),
            _trade(
                doc_id="<NULL>",
                ticker=None,
                asset_name=None,
                asset_type=None,
            ),
        ]
        self.source.session.get = MagicMock(return_value=_response(records))

        df = self.source.fetch_all_trades()

        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["ticker"], "NAN")
        self.assertEqual(df.iloc[0]["ticker_origin"], "source_reported")
        self.assertEqual(df.iloc[0]["asset_description"], "None")
        self.assertEqual(df.iloc[0]["instrument_type"], "null")
        self.assertEqual(df.iloc[0]["raw_asset_description"], " None ")
        self.assertEqual(df.iloc[0]["raw_asset_class"], " null ")
        self.assertNotEqual(df.iloc[0]["doc_id"], df.iloc[1]["doc_id"])

    def test_raw_provenance_preserves_spaces_and_literal_sentinel_words(self):
        record = _trade(
            id=7,
            doc_id=" NaN ",
            transaction_type=" sale (full) ",
            asset_name="  None  ",
            asset_type=" NAN ",
            amount_text="  $250,001 - $500,000  ",
        )
        self.source.session.get = MagicMock(return_value=_response([record]))

        row = self.source.fetch_all_trades().iloc[0]

        self.assertEqual(row["transaction_type"], "Sale")
        self.assertEqual(row["asset_description"], "None")
        self.assertEqual(row["instrument_type"], "nan")
        self.assertEqual(row["amount_raw"], "  $250,001 - $500,000  ")
        self.assertEqual(row["raw_transaction_subtype"], " sale (full) ")
        self.assertEqual(row["raw_asset_description"], "  None  ")
        self.assertEqual(row["raw_asset_class"], " NAN ")
        self.assertTrue(pd.isna(row["source_filing_id"]))
        self.assertTrue(row["doc_id"].startswith("ct-house-"))

    @patch("analyzer.capitol_trades.time.sleep")
    def test_repeated_source_record_id_fails_closed(self, _sleep):
        first = [_trade(id=7), _trade(id=8, ticker="NVDA")]
        second = [_trade(id=7, ticker="MSFT"), _trade(id=9, ticker="JPM")]
        self.source.session.get = MagicMock(
            side_effect=[
                _response(first, page=1, per_page=2, total=4, pages=2),
                _response(second, page=2, per_page=2, total=4, pages=2),
            ]
        )
        with self.assertRaisesRegex(CapitolTradesError, "source record ID '7'"):
            self.source.fetch_all_trades()

    def test_missing_response_metadata_fails_schema(self):
        response = MagicMock()
        response.content = b"{}"
        response.json.return_value = {"trades": [_trade()], "total": 1}
        response.raise_for_status.return_value = None
        self.source.session.get = MagicMock(return_value=response)
        with self.assertRaisesRegex(CapitolTradesError, "missing response fields"):
            self.source.fetch_all_trades()

    def test_missing_trade_field_fails_schema(self):
        record = _trade()
        del record["chamber"]
        self.source.session.get = MagicMock(return_value=_response([record]))
        with self.assertRaisesRegex(CapitolTradesError, "missing fields .*chamber"):
            self.source.fetch_all_trades()

    def test_invalid_date_and_numeric_fields_fail_schema(self):
        cases = [
            (_trade(transaction_date="not-a-date"), "invalid transaction_date"),
            (_trade(amount_min="250001"), "amount_min must be finite numeric"),
        ]
        for record, message in cases:
            with self.subTest(message=message):
                self.source.session.get = MagicMock(return_value=_response([record]))
                with self.assertRaisesRegex(CapitolTradesError, message):
                    self.source.fetch_all_trades()

    def test_pages_must_match_total_and_per_page(self):
        self.source.session.get = MagicMock(
            return_value=_response([_trade()], total=1, pages=2)
        )
        with self.assertRaisesRegex(CapitolTradesError, "Pagination metadata mismatch"):
            self.source.fetch_all_trades()

    def test_chamber_filter_has_house_senate_and_invalid_scenarios(self):
        records = [
            _trade(id=1, chamber="house"),
            _trade(id=2, chamber="senate", politician_name="Katie Britt"),
        ]
        self.source.session.get = MagicMock(return_value=_response(records))
        senate = self.source.fetch_all_trades(chamber="senate")
        self.assertEqual(senate["member"].tolist(), ["Katie Britt"])
        self.assertEqual(senate["chamber"].tolist(), ["senate"])

        self.source.session.get = MagicMock(return_value=_response(records))
        house = self.source.fetch_all_trades(chamber="HOUSE")
        self.assertEqual(house["member"].tolist(), ["Nancy Pelosi"])
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

    def test_raw_source_id_keeps_synthetic_id_stable_across_correction(self):
        first = _response([_trade(id=42, doc_id=None, ticker="APPL")])
        self.source.session.get = MagicMock(return_value=first)
        original_id = self.source.fetch_all_trades().iloc[0]["doc_id"]

        second_source = CapitolTradesSource(generation="second")
        self.addCleanup(second_source.close)
        second_source.session.get = MagicMock(
            return_value=_response([_trade(id=42, doc_id=None, ticker="AAPL")])
        )
        corrected_id = second_source.fetch_all_trades().iloc[0]["doc_id"]
        self.assertEqual(original_id, corrected_id)
        self.assertTrue(original_id.startswith("ct-house-"))

    def test_filing_ids_are_stripped_and_null_sentinels_are_missing(self):
        for filing_id in (None, "", "   ", "None", " null ", "NaN", "<NULL>"):
            with self.subTest(filing_id=filing_id):
                self.source.session.get = MagicMock(
                    return_value=_response([_trade(id=1, doc_id=filing_id)])
                )
                row = self.source.fetch_all_trades().iloc[0]
                self.assertTrue(row["doc_id"].startswith("ct-house-"))
                self.assertTrue(pd.isna(row["source_filing_id"]))
        self.source.session.get = MagicMock(
            return_value=_response([_trade(id=2, doc_id="  20033337  ")])
        )
        row = self.source.fetch_all_trades().iloc[0]
        self.assertEqual(row["doc_id"], "20033337")
        self.assertEqual(row["source_filing_id"], "20033337")

    def test_ticker_origin_only_marks_nonblank_source_ticker(self):
        records = [
            _trade(id=1, ticker=" AAPL "),
            _trade(id=2, ticker="   ", doc_id="20033338"),
            _trade(id=3, ticker=None, doc_id="20033339"),
        ]
        self.source.session.get = MagicMock(return_value=_response(records))
        df = self.source.fetch_all_trades()
        self.assertEqual(df["ticker"].tolist()[:1], ["AAPL"])
        self.assertEqual(df["ticker_origin"].tolist()[0], "source_reported")
        self.assertTrue(pd.isna(df.iloc[1]["ticker_origin"]))
        self.assertTrue(pd.isna(df.iloc[2]["ticker_origin"]))

    def test_text_amount_fallback_is_preserved(self):
        self.source.session.get = MagicMock(
            return_value=_response(
                [
                    _trade(
                        amount_min=None,
                        amount_max=None,
                        amount_text="$15,001 - $50,000",
                    )
                ]
            )
        )
        df = self.source.fetch_all_trades()
        self.assertEqual(df.iloc[0]["amount_midpoint"], 32500.5)

    def test_manifest_binds_pages_generation_and_records(self):
        response = _response([_trade(id=1)])
        self.source.session.get = MagicMock(return_value=response)
        self.source.fetch_all_trades()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "capitol.json"
            self.source.write_reconciliation_artifact(output)
            manifest = json.loads(output.read_text())
            self.assertTrue(manifest["reconciliation_only"])
            self.assertEqual(manifest["ingestion_generation"], "test-generation")
            self.assertEqual(manifest["record_count"], 1)
            raw_hash = hashlib.sha256(response.content).hexdigest()
            self.assertEqual(manifest["pages"][0]["artifact_sha256"], raw_hash)
            self.assertEqual(manifest["records"][0]["artifact_sha256"], raw_hash)
            with self.assertRaisesRegex(CapitolTradesError, "Refusing to overwrite"):
                self.source.write_reconciliation_artifact(output)

    def test_manifest_rejects_subset_input_and_ignores_returned_frame_mutation(self):
        response = _response([_trade(id=1), _trade(id=2, doc_id="20033338")])
        self.source.session.get = MagicMock(return_value=response)
        returned = self.source.fetch_all_trades()
        subset = returned.iloc[:1]
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "bound.json"
            with self.assertRaises(TypeError):
                self.source.write_reconciliation_artifact(subset, output)

            returned.loc[0, "member"] = "MUTATED OUTSIDE SOURCE"
            detached_pages = self.source.last_page_artifacts
            detached_pages[0]["artifact_sha256"] = "0" * 64
            self.source.write_reconciliation_artifact(output)
            manifest = json.loads(output.read_text())

        self.assertEqual(manifest["emitted_count"], 2)
        self.assertEqual(manifest["records"][0]["member"], "Nancy Pelosi")
        self.assertNotEqual(manifest["pages"][0]["artifact_sha256"], "0" * 64)

    def test_manifest_accounts_for_selection_and_exact_normalized_result(self):
        records = [
            _trade(id=1, chamber="house", disclosure_date="2025-07-02"),
            _trade(
                id=2,
                chamber="senate",
                politician_name="Katie Britt",
                doc_id="senate-2",
                disclosure_date="2025-07-03",
            ),
            _trade(id=3, doc_id="house-3", disclosure_date="2025-01-02"),
        ]
        self.source.session.get = MagicMock(return_value=_response(records))
        df = self.source.fetch_all_trades(
            start_date=date(2025, 6, 1),
            end_date=date(2025, 12, 31),
            chamber="house",
        )
        expected_hash = hashlib.sha256(
            self.source._normalized_result_json(df).encode()
        ).hexdigest()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "selected.json"
            self.source.write_reconciliation_artifact(output)
            manifest = json.loads(output.read_text())

        self.assertEqual(
            manifest["selection"],
            {
                "politician": None,
                "chamber": "house",
                "start_date": "2025-06-01",
                "end_date": "2025-12-31",
            },
        )
        self.assertEqual(
            manifest["source_reported"], {"total": 3, "pages": 1, "per_page": 50}
        )
        self.assertEqual(manifest["fetched_raw_count"], 3)
        self.assertEqual(manifest["emitted_count"], 1)
        self.assertEqual(manifest["filtered_count"], 2)
        self.assertEqual(manifest["rejected_count"], 0)
        self.assertEqual(manifest["normalized_result_sha256"], expected_hash)
        serialized_records = json.dumps(
            manifest["records"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        self.assertEqual(
            manifest["normalized_result_sha256"],
            hashlib.sha256(serialized_records).hexdigest(),
        )
        self.assertEqual(len(manifest["records"]), 1)
        self.assertEqual(
            manifest["fetched_raw_count"],
            manifest["emitted_count"]
            + manifest["filtered_count"]
            + manifest["rejected_count"],
        )

    def test_empty_filtered_manifest_is_accounted_and_hashed(self):
        records = [_trade(id=1), _trade(id=2, doc_id="house-2")]
        self.source.session.get = MagicMock(return_value=_response(records))
        df = self.source.fetch_all_trades(chamber="senate")
        self.assertTrue(df.empty)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "empty.json"
            self.source.write_reconciliation_artifact(output)
            manifest = json.loads(output.read_text())

        self.assertEqual(manifest["fetched_raw_count"], 2)
        self.assertEqual(manifest["emitted_count"], 0)
        self.assertEqual(manifest["filtered_count"], 2)
        self.assertEqual(manifest["rejected_count"], 0)
        self.assertEqual(manifest["records"], [])
        self.assertEqual(
            manifest["normalized_result_sha256"], hashlib.sha256(b"[]").hexdigest()
        )

    def test_manifest_requires_validated_fetch(self):
        with (
            TemporaryDirectory() as tmp,
            self.assertRaisesRegex(
                CapitolTradesError, "complete validated and filtered"
            ),
        ):
            self.source.write_reconciliation_artifact(Path(tmp) / "x.json")

    def test_http_503_is_external_failure_not_empty_success(self):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError(
            "503 no available server"
        )
        self.source.session.get = MagicMock(return_value=response)
        with self.assertRaisesRegex(CapitolTradesError, "503 no available server"):
            self.source.fetch_all_trades()
        self.assertEqual(self.source.last_page_artifacts, [])

    def test_canonical_database_access_is_forbidden(self):
        with self.assertRaisesRegex(CapitolTradesError, "reconciliation-only"):
            self.source.save_to_db(pd.DataFrame([{"ticker": "AAPL"}]))
        with self.assertRaisesRegex(CapitolTradesError, "reconciliation-only"):
            self.source.get_transactions(2025)


if __name__ == "__main__":
    unittest.main()
