from datetime import date
import hashlib

import pandas as pd
import pytest

from analyzer.models import ReportOutcome, TickerOrigin
from analyzer.senate_efd import (
    SenateEFDError,
    SenateEFDSource,
    SenateRefreshSummary,
    SenateReportFetchResult,
    SenateRowValidationError,
)


KATIE_REPORT_ID = "37900303-65bf-467d-962b-76555d510b28"
KATIE_REPORT_PATH = f"/search/view/ptr/{KATIE_REPORT_ID}/"


def _source() -> SenateEFDSource:
    source = object.__new__(SenateEFDSource)
    source.ingestion_generation = "generation-2026-08-09"
    source.report_inventory = []
    source.last_refresh_summary = None
    return source


def _raw_trade(**overrides):
    trade = {
        "doc_id": KATIE_REPORT_ID,
        "source_record_id": KATIE_REPORT_ID,
        "source_report_path": KATIE_REPORT_PATH,
        "senator": "Katie Britt",
        "filed_date": pd.Timestamp("2026-01-29"),
        "official_filing_date": pd.Timestamp("2026-01-29"),
        "available_date": pd.Timestamp("2026-01-29"),
        "notification_date": "01/29/2026",
        "amends_source_record_id": None,
        "ticker": "JPM",
        "ticker_raw": "JPM",
        "ticker_origin": TickerOrigin.OFFICIAL.value,
        "transaction_date": "01/28/2026",
        "type": "Sale (Full)",
        "transaction_subtype_raw": "Sale (Full)",
        "owner": "SP",
        "owner_raw": "Spouse",
        "amount_range": "$1,001 - $15,000",
        "amount_range_raw": "$1,001 - $15,000",
        "asset_name": "JPMorgan Chase & Co. Common Stock",
        "asset_type": "Stock",
        "ingestion_generation": "generation-2026-08-09",
        "artifact_sha256": "a" * 64,
    }
    trade.update(overrides)
    return trade


def test_katie_britt_jpm_canary_preserves_provenance_and_normalizes_sale():
    result = _source()._normalize([_raw_trade()])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["chamber"] == "senate"
    assert row["source_record_id"] == KATIE_REPORT_ID
    assert row["chamber_member_key"] == "senate:KATIE BRITT"
    assert row["ticker"] == "JPM"
    assert row["ticker_origin"] == "official"
    assert row["transaction_type"] == "Sale"
    assert row["raw_transaction_subtype"] == "Sale (Full)"
    assert row["amount_raw"] == "$1,001 - $15,000"
    assert row["raw_owner"] == "Spouse"
    assert row["raw_asset_class"] == "Stock"
    assert row["raw_asset_description"] == "JPMorgan Chase & Co. Common Stock"
    assert row["official_filing_date"] == pd.Timestamp("2026-01-29")
    assert row["available_date"] == pd.Timestamp("2026-01-29")
    assert row["notification_date"] == pd.Timestamp("2026-01-29")
    assert row["disclosure_date"] == pd.Timestamp("2026-01-29")
    assert row["amends_source_record_id"] is None
    assert row["ingestion_generation"] == "generation-2026-08-09"
    assert row["artifact_sha256"] == "a" * 64


def test_rick_scott_coupon_canary_is_non_equity_not_a_ticker():
    asset = (
        "Harris County Texas Toll Road Revenue Bond "
        "Rate/Coupon:5.25% Matures:08/15/2049"
    )

    ticker, origin = SenateEFDSource._resolve_ticker("--", asset, "Municipal Bond")

    assert ticker is None
    assert origin is TickerOrigin.NON_EQUITY
    assert SenateEFDSource._normalize_instrument_type("Municipal Bond") == "bond"


def test_report_html_preserves_raw_fields_and_artifact_hash():
    html = """
    <html><head><title>eFD: Print Periodic Transaction Report</title></head><body>
    <table class="table-striped">
      <thead><tr>
        <th>Transaction Date</th><th>Notification Date</th><th>Owner</th>
        <th>Ticker</th><th>Asset Name</th><th>Asset Type</th>
        <th>Type</th><th>Amount</th>
      </tr></thead>
      <tbody><tr>
        <td>01/28/2026</td><td>01/29/2026</td><td>Spouse</td>
        <td>JPM</td><td>JPMorgan Chase &amp; Co. Common Stock</td><td>Stock</td>
        <td>Sale (Full)</td><td>$1,001 - $15,000</td>
      </tr></tbody>
    </table>
    </body></html>
    """

    class Response:
        status_code = 200
        text = html
        content = html.encode()

    source = _source()
    source._request_with_retry = lambda *args, **kwargs: Response()

    result = source._fetch_report_transactions(KATIE_REPORT_PATH)

    assert result.outcome is ReportOutcome.PARSED
    assert result.artifact_sha256 == hashlib.sha256(html.encode()).hexdigest()
    assert result.transactions == (
        {
            "ticker": "JPM",
            "ticker_raw": "JPM",
            "ticker_origin": "official",
            "asset_name": "JPMorgan Chase & Co. Common Stock",
            "asset_type": "Stock",
            "owner": "SP",
            "owner_raw": "Spouse",
            "type": "Sale (Full)",
            "transaction_subtype_raw": "Sale (Full)",
            "transaction_date": "01/28/2026",
            "notification_date": "01/29/2026",
            "amount_range": "$1,001 - $15,000",
            "amount_range_raw": "$1,001 - $15,000",
        },
    )


def test_same_and_cross_report_rows_are_never_collapsed_without_row_ids():
    first = _raw_trade()
    same_report_duplicate = _raw_trade()
    second_report_id = "22222222-2222-2222-2222-222222222222"
    cross_report_duplicate = _raw_trade(
        doc_id=second_report_id,
        source_record_id=second_report_id,
        source_report_path=f"/search/view/ptr/{second_report_id}/",
        filed_date=pd.Timestamp("2026-05-11"),
        official_filing_date=pd.Timestamp("2026-05-11"),
        available_date=pd.Timestamp("2026-05-11"),
        artifact_sha256="2" * 64,
    )

    result = _source()._normalize(
        [first, same_report_duplicate, cross_report_duplicate]
    )

    assert len(result) == 3
    assert result["source_record_id"].tolist() == [
        KATIE_REPORT_ID,
        KATIE_REPORT_ID,
        second_report_id,
    ]
    assert result["available_date"].tolist() == [
        pd.Timestamp("2026-01-29"),
        pd.Timestamp("2026-01-29"),
        pd.Timestamp("2026-05-11"),
    ]
    assert result["amends_source_record_id"].isna().all()


def test_refresh_accounts_for_parsed_paper_and_unavailable_reports(monkeypatch):
    reports = [
        {
            "senator": "Katie Britt",
            "report_path": KATIE_REPORT_PATH,
            "filed_date": pd.Timestamp("2026-01-29"),
        },
        {
            "senator": "Paper Senator",
            "report_path": "/search/view/ptr/paper/",
            "filed_date": pd.Timestamp("2026-01-30"),
        },
        {
            "senator": "Unavailable Senator",
            "report_path": "/search/view/ptr/unavailable/",
            "filed_date": pd.Timestamp("2026-01-31"),
        },
    ]
    outcomes = iter(
        [
            SenateReportFetchResult(
                outcome=ReportOutcome.PARSED,
                transactions=(
                    {
                        key: value
                        for key, value in _raw_trade().items()
                        if key
                        not in {
                            "doc_id",
                            "source_record_id",
                            "source_report_path",
                            "senator",
                            "filed_date",
                            "official_filing_date",
                            "available_date",
                            "amends_source_record_id",
                            "ingestion_generation",
                            "artifact_sha256",
                        }
                    },
                ),
                artifact_sha256="a" * 64,
            ),
            SenateReportFetchResult(outcome=ReportOutcome.PAPER_ONLY),
            SenateReportFetchResult(outcome=ReportOutcome.UNAVAILABLE),
        ]
    )
    source = _source()
    source._open_session = lambda: None
    source._search_reports = lambda start, end: reports
    source._fetch_report_transactions = lambda path: next(outcomes)
    monkeypatch.setattr("analyzer.senate_efd.time.sleep", lambda _: None)

    result = source.fetch_all_trades(date(2026, 1, 1), date(2026, 2, 1))

    assert result.attrs["refresh_summary"] == {
        "found": 3,
        "parsed": 1,
        "paper_only": 1,
        "unavailable": 1,
        "failed": 0,
    }
    assert [row["outcome"] for row in source.report_inventory] == [
        "parsed",
        "paper_only",
        "unavailable",
    ]
    assert result.iloc[0]["source_record_id"] == KATIE_REPORT_ID
    assert result.iloc[0]["artifact_sha256"] == "a" * 64
    assert source.report_inventory[0]["source_record_id"] == KATIE_REPORT_ID
    assert source.report_inventory[0]["artifact_sha256"] == "a" * 64


def test_refresh_rejects_any_failed_report(monkeypatch):
    reports = [
        {
            "senator": "Broken Senator",
            "report_path": "/search/view/ptr/broken/",
            "filed_date": pd.Timestamp("2026-01-29"),
        }
    ]
    source = _source()
    source._open_session = lambda: None
    source._search_reports = lambda start, end: reports
    source._fetch_report_transactions = lambda path: (_ for _ in ()).throw(
        SenateEFDError("layout changed")
    )
    monkeypatch.setattr("analyzer.senate_efd.time.sleep", lambda _: None)

    with pytest.raises(SenateEFDError, match="failed=1"):
        source.fetch_all_trades(date(2026, 1, 1), date(2026, 2, 1))

    assert source.last_refresh_summary == SenateRefreshSummary(
        found=1,
        parsed=0,
        paper_only=0,
        unavailable=0,
        failed=1,
    )


def test_summary_rejects_unaccounted_reports():
    with pytest.raises(ValueError, match="accounting mismatch"):
        SenateRefreshSummary(
            found=4,
            parsed=1,
            paper_only=1,
            unavailable=1,
            failed=0,
        )


def test_http_200_login_and_empty_table_are_failed_not_paper_or_parsed():
    pages = [
        "<html><head><title>eFD Login</title></head><body>Sign in</body></html>",
        """
        <html><head><title>eFD: Print Periodic Transaction Report</title></head>
        <body><table class="table-striped">
          <thead><tr><th>Transaction Date</th><th>Owner</th><th>Asset Name</th>
          <th>Type</th><th>Amount</th></tr></thead><tbody></tbody>
        </table></body></html>
        """,
    ]
    source = _source()

    for html in pages:

        class Response:
            status_code = 200
            text = html
            content = html.encode()

        source._request_with_retry = lambda *args, **kwargs: Response()
        result = source._fetch_report_transactions(KATIE_REPORT_PATH)
        assert result.outcome is ReportOutcome.FAILED
        assert result.artifact_sha256 == hashlib.sha256(html.encode()).hexdigest()


def test_paper_only_requires_explicit_official_artifact_link():
    html = """
    <html><head><title>eFD: Print Periodic Transaction Report</title></head>
    <body><a href="/media/paper-filings/report.pdf">Download paper filing</a></body>
    </html>
    """

    class Response:
        status_code = 200
        text = html
        content = html.encode()

    source = _source()
    source._request_with_retry = lambda *args, **kwargs: Response()

    result = source._fetch_report_transactions(KATIE_REPORT_PATH)

    assert result.outcome is ReportOutcome.PAPER_ONLY
    assert result.paper_artifact_url == (
        "https://efdsearch.senate.gov/media/paper-filings/report.pdf"
    )
    assert result.artifact_sha256 == hashlib.sha256(html.encode()).hexdigest()


def test_source_supplied_debt_word_tickers_are_preserved_but_inference_is_guarded():
    for supplied in ("BOND", "NOTE"):
        ticker, origin = SenateEFDSource._resolve_ticker(
            supplied,
            "Example issuer corporate debt security",
            "Corporate Bond",
        )
        assert ticker == supplied
        assert origin is TickerOrigin.OFFICIAL

    ticker, origin = SenateEFDSource._resolve_ticker(
        "--",
        "Example Treasury Note Matures:01/01/2030 (NOTE)",
        "",
    )
    assert ticker is None
    assert origin is TickerOrigin.NON_EQUITY


def test_ambiguous_description_ticker_remains_unverified_and_not_stock():
    ticker, origin = SenateEFDSource._resolve_ticker("--", "Acme Holdings (ACME)", "")

    assert ticker == "ACME"
    assert origin is TickerOrigin.UNVERIFIED
    assert SenateEFDSource._normalize_instrument_type("") == "unknown"


def test_malformed_row_fails_its_report_and_complete_refresh(monkeypatch):
    good = {
        key: value
        for key, value in _raw_trade().items()
        if key
        not in {
            "doc_id",
            "source_record_id",
            "source_report_path",
            "senator",
            "filed_date",
            "official_filing_date",
            "available_date",
            "amends_source_record_id",
            "ingestion_generation",
            "artifact_sha256",
        }
    }
    malformed = {**good, "transaction_date": "not-a-date"}
    report = {
        "senator": "Katie Britt",
        "report_path": KATIE_REPORT_PATH,
        "filed_date": pd.Timestamp("2026-01-29"),
    }
    source = _source()
    source._open_session = lambda: None
    source._search_reports = lambda start, end: [report]
    source._fetch_report_transactions = lambda path: SenateReportFetchResult(
        outcome=ReportOutcome.PARSED,
        transactions=(good, malformed),
        artifact_sha256="a" * 64,
    )
    monkeypatch.setattr("analyzer.senate_efd.time.sleep", lambda _: None)

    with pytest.raises(SenateEFDError, match="failed=1"):
        source.fetch_all_trades(date(2026, 1, 1), date(2026, 2, 1))

    assert source.last_refresh_summary == SenateRefreshSummary(
        found=1, parsed=0, paper_only=0, unavailable=0, failed=1
    )
    assert source.report_inventory[0]["artifact_sha256"] == "a" * 64
    assert source.report_inventory[0]["raw_row_count"] == 2
    assert source.report_inventory[0]["accepted_row_count"] == 1
    assert source.report_inventory[0]["rejected_row_count"] == 1
    assert (
        "raw=2, accepted=1, rejected=1" in source.report_inventory[0]["error_message"]
    )


def test_direct_normalization_rejects_any_malformed_row_atomically():
    with pytest.raises(SenateRowValidationError) as exc_info:
        _source()._normalize([_raw_trade(), _raw_trade(transaction_date="not-a-date")])

    assert exc_info.value.raw_count == 2
    assert exc_info.value.accepted_count == 1
    assert exc_info.value.rejected_count == 1


def test_save_refuses_database_that_would_drop_provenance():
    source = _source()
    source.db = object()
    source.last_refresh_summary = SenateRefreshSummary(
        found=1, parsed=1, paper_only=0, unavailable=0, failed=0
    )
    source.report_inventory = [{"source_record_id": KATIE_REPORT_ID}]

    with pytest.raises(SenateEFDError, match="persist_source_refresh"):
        source.save_to_db(source._normalize([_raw_trade()]))


def test_save_calls_atomic_refresh_persistence_with_inventory_and_rows():
    calls = []

    class Database:
        @staticmethod
        def persist_source_refresh(**kwargs):
            calls.append(kwargs)
            return len(kwargs["transactions"])

    source = _source()
    source.db = Database()
    source.last_refresh_summary = SenateRefreshSummary(
        found=1, parsed=1, paper_only=0, unavailable=0, failed=0
    )
    source.report_inventory = [
        {
            "chamber": "senate",
            "source_record_id": KATIE_REPORT_ID,
            "artifact_sha256": "a" * 64,
            "outcome": "parsed",
            "ingestion_generation": "generation-2026-08-09",
        }
    ]
    frame = source._normalize([_raw_trade()])

    inserted = source.save_to_db(frame)

    assert inserted == 1
    assert len(calls) == 1
    assert calls[0]["source"] == "senate_efd"
    assert calls[0]["chamber"] == "senate"
    assert calls[0]["ingestion_generation"] == "generation-2026-08-09"
    assert calls[0]["transactions"].equals(frame)
    assert calls[0]["reports"].to_dict("records") == source.report_inventory
