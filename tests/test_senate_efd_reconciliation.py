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
    _DERIVATION_COLUMNS,
    _NORMALIZED_TRANSACTION_COLUMNS,
)


KATIE_REPORT_ID = "37900303-65bf-467d-962b-76555d510b28"
KATIE_REPORT_PATH = f"/search/view/ptr/{KATIE_REPORT_ID}/"
KATIE_REPORT_URL = f"https://efdsearch.senate.gov{KATIE_REPORT_PATH}"


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
        "source_row_id": "official:1",
        "source_report_path": KATIE_REPORT_PATH,
        "senator": "Katie Britt",
        "filed_date": pd.Timestamp("2026-01-29"),
        "official_filing_date": pd.Timestamp("2026-01-29"),
        "available_date": pd.Timestamp("2026-01-29"),
        "notification_date": "01/29/2026",
        "amends_source_record_id": None,
        "ticker": "JPM",
        "ticker_raw": "JPM",
        "ticker_candidate": None,
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


def _report_inventory_row(**overrides):
    report = {
        "chamber": "senate",
        "source_record_id": KATIE_REPORT_ID,
        "report_path": KATIE_REPORT_PATH,
        "member": "Katie Britt",
        "official_filing_date": pd.Timestamp("2026-01-29"),
        "outcome": "parsed",
        "artifact_sha256": "a" * 64,
        "landing_sha256": "a" * 64,
        "paper_artifact_sha256": None,
        "paper_artifact_url": None,
        "error_message": None,
        "raw_row_count": 1,
        "accepted_row_count": 1,
        "rejected_row_count": 0,
        "ingestion_generation": "generation-2026-08-09",
    }
    report.update(overrides)
    return report


def _ready_persistence_source(report=None):
    class Database:
        @staticmethod
        def persist_source_refresh(**kwargs):
            return len(kwargs["transactions"])

    source = _source()
    source.db = Database()
    source.last_refresh_summary = SenateRefreshSummary(
        found=1, parsed=1, paper_only=0, unavailable=0, failed=0
    )
    source.report_inventory = [report or _report_inventory_row()]
    return source


def test_katie_britt_jpm_canary_preserves_provenance_and_normalizes_sale():
    result = _source()._normalize([_raw_trade()])

    assert len(result) == 1
    row = result.iloc[0]
    assert row["chamber"] == "senate"
    assert row["source_record_id"] == KATIE_REPORT_ID
    assert row["source_row_id"] == "official:1"
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

    ticker, candidate, origin = SenateEFDSource._resolve_ticker(
        "--", asset, "Municipal Bond"
    )

    assert ticker is None
    assert candidate is None
    assert origin is TickerOrigin.NON_EQUITY
    assert SenateEFDSource._normalize_instrument_type("Municipal Bond") == "bond"


def test_report_html_preserves_raw_fields_and_artifact_hash():
    html = """
    <html><head><title>eFD: Print Periodic Transaction Report</title></head><body>
    <table class="table-striped">
      <thead><tr>
        <th>#</th><th>Transaction Date</th><th>Notification Date</th><th>Owner</th>
        <th>Ticker</th><th>Asset Name</th><th>Asset Type</th>
        <th>Type</th><th>Amount</th>
      </tr></thead>
      <tbody><tr>
        <td>1</td><td>01/28/2026</td><td>01/29/2026</td><td>Spouse</td>
        <td>JPM</td><td>JPMorgan Chase &amp; Co. Common Stock</td><td>Stock</td>
        <td>Sale (Full)</td><td>$1,001 - $15,000</td>
      </tr></tbody>
    </table>
    </body></html>
    """

    class Response:
        url = KATIE_REPORT_URL
        status_code = 200
        text = html
        content = html.encode()

    source = _source()
    source._request_with_retry = lambda *args, **kwargs: Response()

    result = source._fetch_report_transactions(KATIE_REPORT_PATH)

    assert result.outcome is ReportOutcome.PARSED
    assert result.landing_sha256 == hashlib.sha256(html.encode()).hexdigest()
    assert result.transactions == (
        {
            "source_row_id": "official:1",
            "ticker": "JPM",
            "ticker_raw": "JPM",
            "ticker_candidate": None,
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


def test_report_without_official_row_id_uses_artifact_order_key():
    html = """
    <html><head><title>eFD: Print Periodic Transaction Report</title></head><body>
    <table class="table-striped">
      <thead><tr>
        <th>Transaction Date</th><th>Owner</th><th>Ticker</th>
        <th>Asset Name</th><th>Asset Type</th><th>Type</th><th>Amount</th>
      </tr></thead>
      <tbody><tr>
        <td>01/28/2026</td><td>Spouse</td><td>JPM</td>
        <td>JPMorgan Chase</td><td>Stock</td><td>Sale (Full)</td>
        <td>$1,001 - $15,000</td>
      </tr></tbody>
    </table></body></html>
    """

    class Response:
        url = KATIE_REPORT_URL
        status_code = 200
        text = html
        content = html.encode()

    source = _source()
    source._request_with_retry = lambda *args, **kwargs: Response()

    result = source._fetch_report_transactions(KATIE_REPORT_PATH)

    assert result.outcome is ReportOutcome.PARSED
    assert result.transactions[0]["source_row_id"] == "table:000001"


def test_same_and_cross_report_rows_are_never_collapsed_without_row_ids():
    first = _raw_trade()
    same_report_duplicate = _raw_trade(source_row_id="official:2")
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
    assert result["source_row_id"].tolist() == [
        "official:1",
        "official:2",
        "official:1",
    ]
    assert result["available_date"].tolist() == [
        pd.Timestamp("2026-01-29"),
        pd.Timestamp("2026-01-29"),
        pd.Timestamp("2026-05-11"),
    ]
    assert result["amends_source_record_id"].isna().all()


def test_refresh_accounts_for_parsed_and_verified_paper_reports(monkeypatch):
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
    ]
    parsed = {
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
    outcomes = iter(
        [
            SenateReportFetchResult(
                outcome=ReportOutcome.PARSED,
                transactions=(parsed,),
                landing_sha256="a" * 64,
            ),
            SenateReportFetchResult(
                outcome=ReportOutcome.PAPER_ONLY,
                landing_sha256="b" * 64,
                paper_artifact_sha256="c" * 64,
                paper_artifact_url=(
                    "https://efdsearch.senate.gov/media/paper-filings/report.pdf"
                ),
            ),
        ]
    )
    source = _source()
    source._open_session = lambda: None
    source._search_reports = lambda start, end: reports
    source._fetch_report_transactions = lambda path: next(outcomes)
    monkeypatch.setattr("analyzer.senate_efd.time.sleep", lambda _: None)

    result = source.fetch_all_trades(date(2026, 1, 1), date(2026, 2, 1))

    assert result.attrs["refresh_summary"] == {
        "found": 2,
        "parsed": 1,
        "paper_only": 1,
        "unavailable": 0,
        "failed": 0,
    }
    assert [row["outcome"] for row in source.report_inventory] == [
        "parsed",
        "paper_only",
    ]
    assert result.iloc[0]["artifact_sha256"] == "a" * 64
    assert source.report_inventory[1]["landing_sha256"] == "b" * 64
    assert source.report_inventory[1]["paper_artifact_sha256"] == "c" * 64
    assert (
        source.report_inventory[1]["paper_artifact_url"]
        == "https://efdsearch.senate.gov/media/paper-filings/report.pdf"
    )


def test_unavailable_report_is_inventoried_but_refresh_is_incomplete(monkeypatch):
    source = _source()
    source._open_session = lambda: None
    source._search_reports = lambda start, end: [
        {
            "senator": "Unavailable Senator",
            "report_path": "/search/view/ptr/unavailable/",
            "filed_date": pd.Timestamp("2026-01-31"),
        }
    ]
    source._fetch_report_transactions = lambda path: SenateReportFetchResult(
        outcome=ReportOutcome.UNAVAILABLE, landing_sha256="d" * 64
    )
    monkeypatch.setattr("analyzer.senate_efd.time.sleep", lambda _: None)

    with pytest.raises(SenateEFDError, match="unavailable=1"):
        source.fetch_all_trades(date(2026, 1, 1), date(2026, 2, 1))

    assert source.last_refresh_summary.complete is False
    assert source.report_inventory[0]["outcome"] == "unavailable"
    assert source.report_inventory[0]["landing_sha256"] == "d" * 64


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
            url = KATIE_REPORT_URL
            status_code = 200
            text = html
            content = html.encode()

        source._request_with_retry = lambda *args, **kwargs: Response()
        result = source._fetch_report_transactions(KATIE_REPORT_PATH)
        assert result.outcome is ReportOutcome.FAILED
        assert result.landing_sha256 == hashlib.sha256(html.encode()).hexdigest()


def test_paper_only_requires_allowlisted_fetched_pdf_and_separate_hashes():
    html = """
    <html><head><title>eFD: Print Periodic Transaction Report</title></head>
    <body><a href="/media/paper-filings/report.pdf">Download paper filing</a></body>
    </html>
    """
    pdf = b"%PDF-1.7 official paper bytes"
    paper_url = "https://efdsearch.senate.gov/media/paper-filings/report.pdf"

    class LandingResponse:
        url = KATIE_REPORT_URL
        status_code = 200
        text = html
        content = html.encode()

    class PaperResponse:
        url = paper_url
        status_code = 200
        text = ""
        content = pdf

    responses = iter([LandingResponse(), PaperResponse()])
    source = _source()
    source._request_with_retry = lambda *args, **kwargs: next(responses)

    result = source._fetch_report_transactions(KATIE_REPORT_PATH)

    assert result.outcome is ReportOutcome.PAPER_ONLY
    assert result.landing_sha256 == hashlib.sha256(html.encode()).hexdigest()
    assert result.paper_artifact_sha256 == hashlib.sha256(pdf).hexdigest()
    assert result.paper_artifact_url == paper_url


def test_source_supplied_debt_word_tickers_are_preserved_but_inference_is_guarded():
    for supplied in ("BOND", "NOTE"):
        ticker, candidate, origin = SenateEFDSource._resolve_ticker(
            supplied,
            "Example issuer corporate debt security",
            "Corporate Bond",
        )
        assert ticker == supplied
        assert candidate is None
        assert origin is TickerOrigin.OFFICIAL

    ticker, candidate, origin = SenateEFDSource._resolve_ticker(
        "--",
        "Example Treasury Note Matures:01/01/2030 (NOTE)",
        "",
    )
    assert ticker is None
    assert candidate is None
    assert origin is TickerOrigin.NON_EQUITY


def test_ambiguous_description_ticker_remains_unverified_and_not_stock():
    ticker, candidate, origin = SenateEFDSource._resolve_ticker(
        "--", "Acme Holdings (ACME)", ""
    )

    assert ticker is None
    assert candidate == "ACME"
    assert origin is TickerOrigin.UNVERIFIED
    assert SenateEFDSource._normalize_instrument_type("") == "unknown"
    normalized = _source()._normalize(
        [
            _raw_trade(
                ticker=None,
                ticker_raw="--",
                ticker_candidate=candidate,
                ticker_origin=TickerOrigin.UNVERIFIED.value,
            )
        ]
    )
    assert normalized.iloc[0]["ticker"] is None
    assert normalized.iloc[0]["ticker_candidate"] == "ACME"
    with pytest.raises(SenateRowValidationError):
        _source()._normalize(
            [
                _raw_trade(
                    ticker="ACME",
                    ticker_candidate="ACME",
                    ticker_origin=TickerOrigin.UNVERIFIED.value,
                )
            ]
        )


@pytest.mark.parametrize(
    "final_url",
    [
        "https://efdsearch.senate.gov/search/view/ptr/a-different-record/",
        f"https://evil.example{KATIE_REPORT_PATH}",
    ],
)
def test_final_response_must_stay_on_exact_official_record_path(final_url):
    html = "<html><head><title>eFD: Print Periodic Transaction Report</title></head></html>"

    class Response:
        url = final_url
        status_code = 200
        text = html
        content = html.encode()

    source = _source()
    source._request_with_retry = lambda *args, **kwargs: Response()
    result = source._fetch_report_transactions(KATIE_REPORT_PATH)
    assert result.outcome is ReportOutcome.FAILED
    assert "official path" in result.error_message


@pytest.mark.parametrize(
    "paper_href,paper_url,paper_content",
    [
        ("https://evil.example/report.pdf", None, None),
        (
            "/media/paper-filings/report.pdf",
            "https://efdsearch.senate.gov/media/paper-filings/report.pdf",
            b"not pdf",
        ),
        (
            "/media/paper-filings/report.pdf",
            "https://evil.example/report.pdf",
            b"%PDF-1.7 bytes",
        ),
    ],
)
def test_paper_artifact_rejects_external_bad_or_redirected_content(
    paper_href, paper_url, paper_content
):
    html = f"""<html><head><title>eFD: Print Periodic Transaction Report</title></head>
    <body><a href="{paper_href}">paper filing</a></body></html>"""

    class Landing:
        url = KATIE_REPORT_URL
        status_code = 200
        text = html
        content = html.encode()

    responses = [Landing()]
    if paper_url is not None:
        responses.append(
            type(
                "Paper",
                (),
                {
                    "url": paper_url,
                    "status_code": 200,
                    "text": "",
                    "content": paper_content,
                },
            )()
        )
    source = _source()
    calls = iter(responses)
    source._request_with_retry = lambda *args, **kwargs: next(calls)

    result = source._fetch_report_transactions(KATIE_REPORT_PATH)
    assert result.outcome is ReportOutcome.FAILED
    assert result.paper_artifact_sha256 is None
    assert result.paper_artifact_url is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"transaction_date": pd.NaT},
        {"transaction_date": float("nan")},
        {"transaction_date": ""},
        {"official_filing_date": pd.NaT},
        {"available_date": ""},
        {"transaction_date": "01/30/2026"},
        {"notification_date": "01/30/2026"},
        {"transaction_date": "01/01/2200"},
    ],
)
def test_date_null_domain_and_chronology_values_are_rejected(overrides):
    with pytest.raises(SenateRowValidationError) as exc_info:
        _source()._normalize([_raw_trade(**overrides)])
    assert exc_info.value.rejected_count == 1


def test_duplicate_or_blank_source_row_ids_fail_normalization():
    with pytest.raises(SenateRowValidationError) as duplicate:
        _source()._normalize([_raw_trade(), _raw_trade()])
    assert duplicate.value.accepted_count == 0
    assert duplicate.value.rejected_count == 2

    for field, value in (
        ("source_row_id", "  "),
        ("source_row_id", None),
        ("source_record_id", "  "),
    ):
        with pytest.raises(SenateRowValidationError) as blank:
            _source()._normalize([_raw_trade(**{field: value})])
        assert blank.value.rejected_count == 1


def test_duplicate_official_row_ids_fail_report_classification():
    html = """<html><head><title>eFD: Print Periodic Transaction Report</title></head>
    <table class="table-striped"><thead><tr><th>#</th><th>Transaction Date</th>
    <th>Owner</th><th>Asset Name</th><th>Type</th><th>Amount</th></tr></thead>
    <tbody><tr><td>1</td><td>01/28/2026</td><td>Self</td><td>Acme</td>
    <td>Purchase</td><td>$1,001 - $15,000</td></tr>
    <tr><td>1</td><td>01/28/2026</td><td>Self</td><td>Acme</td>
    <td>Purchase</td><td>$1,001 - $15,000</td></tr></tbody></table></html>"""

    class Response:
        url = KATIE_REPORT_URL
        status_code = 200
        text = html
        content = html.encode()

    source = _source()
    source._request_with_retry = lambda *a, **k: Response()
    result = source._fetch_report_transactions(KATIE_REPORT_PATH)
    assert result.outcome is ReportOutcome.FAILED
    assert len(result.transactions) == 2
    assert "duplicate source row IDs" in result.error_message


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
        landing_sha256="a" * 64,
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
    source.report_inventory = [_report_inventory_row()]
    frame = source._normalize([_raw_trade()])

    inserted = source.save_to_db(frame)

    assert inserted == 1
    assert len(calls) == 1
    assert calls[0]["source"] == "senate_efd"
    assert calls[0]["chamber"] == "senate"
    assert calls[0]["ingestion_generation"] == "generation-2026-08-09"
    assert calls[0]["transactions"].equals(frame)
    assert calls[0]["reports"].to_dict("records") == source.report_inventory


def test_save_rejects_duplicate_source_row_identity_and_incomplete_refresh():
    class Database:
        @staticmethod
        def persist_source_refresh(**kwargs):
            raise AssertionError("persistence must not be called")

    source = _source()
    source.db = Database()
    source.last_refresh_summary = SenateRefreshSummary(
        found=1, parsed=1, paper_only=0, unavailable=0, failed=0
    )
    source.report_inventory = [_report_inventory_row()]
    frame = source._normalize([_raw_trade()])
    duplicated = pd.concat([frame, frame], ignore_index=True)

    with pytest.raises(SenateEFDError, match="must be unique"):
        source.save_to_db(duplicated)

    invalid_date = frame.copy()
    invalid_date.loc[0, "available_date"] = pd.NaT
    with pytest.raises(SenateEFDError, match="provenance values are incomplete"):
        source.save_to_db(invalid_date)

    invalid_chronology = frame.copy()
    invalid_chronology.loc[0, "transaction_date"] = pd.Timestamp("2026-01-30")
    with pytest.raises(SenateEFDError, match="dates are incomplete or invalid"):
        source.save_to_db(invalid_chronology)

    source.last_refresh_summary = SenateRefreshSummary(
        found=2, parsed=2, paper_only=0, unavailable=0, failed=0
    )
    source.report_inventory = source.report_inventory * 2
    with pytest.raises(SenateEFDError, match="Duplicate Senate report inventory ID"):
        source.save_to_db(frame)

    source.last_refresh_summary = SenateRefreshSummary(
        found=1, parsed=0, paper_only=0, unavailable=1, failed=0
    )
    with pytest.raises(SenateEFDError, match="complete Senate report inventory"):
        source.save_to_db(frame)


@pytest.mark.parametrize(
    "ticker,candidate,origin,raw_ticker,description,asset_class",
    [
        ("JPM", None, "official", "JPM", "JPMorgan", "Stock"),
        ("JPM", None, "asset_description", "--", "JPMorgan (JPM)", "Stock"),
        (None, "ACME", "unverified", "--", "Acme Holdings (ACME)", ""),
        (None, None, "non_equity", "--", "Treasury Note", "Bond"),
        (None, None, "missing", "--", "Private holding", ""),
        (None, None, "invalid", "bad!", "Private holding", ""),
        (None, "STOCK", "invalid", "--", "Private holding (STOCK)", "Stock"),
    ],
)
def test_save_accepts_every_valid_ticker_provenance_matrix_case(
    ticker, candidate, origin, raw_ticker, description, asset_class
):
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[
        0,
        [
            "ticker",
            "ticker_candidate",
            "ticker_origin",
            "raw_ticker",
            "asset_description",
            "raw_asset_description",
            "raw_asset_class",
            "instrument_type",
        ],
    ] = [
        ticker,
        candidate,
        origin,
        raw_ticker,
        description,
        description,
        asset_class,
        source._normalize_instrument_type(asset_class),
    ]

    assert source.save_to_db(frame) == 1


@pytest.mark.parametrize(
    "ticker,candidate,origin",
    [
        ("JPM", None, "bogus"),
        (None, None, "official"),
        ("JPM", "ACME", "official"),
        ("bad!", None, "asset_description"),
        ("BOND", None, "asset_description"),
        ("ACME", "ACME", "unverified"),
        (None, "bad!", "unverified"),
        ("BOND", None, "non_equity"),
        (None, "BOND", "non_equity"),
        ("JPM", None, "missing"),
        (None, "ACME", "missing"),
        ("JPM", None, "invalid"),
        (None, "ACME", "invalid"),
    ],
)
def test_save_rejects_every_invalid_ticker_provenance_matrix_case(
    ticker, candidate, origin
):
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[0, ["ticker", "ticker_candidate", "ticker_origin"]] = [
        ticker,
        candidate,
        origin,
    ]

    with pytest.raises(SenateEFDError, match="ticker"):
        source.save_to_db(frame)


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"outcome": "bogus"}, "Unknown Senate report outcome"),
        ({"raw_row_count": 2}, "row accounting is invalid"),
        ({"raw_row_count": -1, "accepted_row_count": -1}, "row counts are invalid"),
        (
            {"raw_row_count": 2, "accepted_row_count": 2},
            "count does not match transaction rows",
        ),
        (
            {"artifact_sha256": "z" * 64, "landing_sha256": "z" * 64},
            "landing hash is invalid",
        ),
        ({"artifact_sha256": "b" * 64}, "landing hash is invalid"),
        ({"paper_artifact_sha256": "c" * 64}, "Nonpaper Senate report"),
    ],
)
def test_save_rejects_bogus_report_outcome_counts_and_hashes(overrides, error):
    source = _ready_persistence_source(_report_inventory_row(**overrides))
    frame = source._normalize([_raw_trade()])

    with pytest.raises(SenateEFDError, match=error):
        source.save_to_db(frame)


def test_save_requires_complete_report_schema_and_exact_summary_counts():
    report = _report_inventory_row()
    del report["raw_row_count"]
    source = _ready_persistence_source(report)
    frame = source._normalize([_raw_trade()])
    with pytest.raises(SenateEFDError, match="schema mismatch"):
        source.save_to_db(frame)

    source = _ready_persistence_source()
    source.last_refresh_summary = SenateRefreshSummary(
        found=1, parsed=0, paper_only=1, unavailable=0, failed=0
    )
    with pytest.raises(SenateEFDError, match="outcome counts do not match"):
        source.save_to_db(frame)


def test_save_requires_parsed_transaction_hash_to_equal_report_landing_hash():
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[0, "artifact_sha256"] = "b" * 64

    with pytest.raises(SenateEFDError, match="does not match report landing"):
        source.save_to_db(frame)


def test_save_accepts_verified_paper_without_transactions_and_rejects_with_rows():
    paper_url = "https://efdsearch.senate.gov/media/paper-filings/report.pdf"
    report = _report_inventory_row(
        outcome="paper_only",
        raw_row_count=0,
        accepted_row_count=0,
        artifact_sha256="b" * 64,
        landing_sha256="b" * 64,
        paper_artifact_sha256="c" * 64,
        paper_artifact_url=paper_url,
    )
    source = _ready_persistence_source(report)
    source.last_refresh_summary = SenateRefreshSummary(
        found=1, parsed=0, paper_only=1, unavailable=0, failed=0
    )
    assert source.save_to_db(source._normalize([])) == 0

    frame = source._normalize(
        [
            _raw_trade(
                artifact_sha256="b" * 64,
            )
        ]
    )
    with pytest.raises(SenateEFDError, match="Nonparsed Senate report"):
        source.save_to_db(frame)


@pytest.mark.parametrize(
    "column",
    [
        "member",
        "source_report_path",
        "raw_ticker",
        "raw_transaction_subtype",
        "raw_asset_description",
        "raw_asset_class",
        "raw_owner",
        "amount_raw",
    ],
)
def test_save_requires_transaction_report_and_raw_binding_columns(column):
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()]).drop(columns=[column])

    with pytest.raises(SenateEFDError, match="schema mismatch"):
        source.save_to_db(frame)


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("member", "Another Senator", "member mismatch"),
        ("source_report_path", "/search/view/ptr/wrong/", "report path mismatch"),
        ("official_filing_date", pd.Timestamp("2026-01-28"), "dates"),
        ("available_date", pd.Timestamp("2026-01-28"), "dates"),
        ("disclosure_date", pd.Timestamp("2026-01-28"), "filing date binding"),
    ],
)
def test_save_rejects_each_transaction_report_binding_mutation(column, value, error):
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[0, column] = value

    with pytest.raises(SenateEFDError, match=error):
        source.save_to_db(frame)


def test_save_rejects_report_path_record_id_and_transaction_doc_id_mutations():
    source = _ready_persistence_source(
        _report_inventory_row(report_path="/search/view/ptr/wrong/")
    )
    frame = source._normalize([_raw_trade()])
    with pytest.raises(SenateEFDError, match="path/record ID mismatch"):
        source.save_to_db(frame)

    source = _ready_persistence_source()
    frame.loc[0, "doc_id"] = "wrong-doc"
    with pytest.raises(SenateEFDError, match="doc_id mismatch"):
        source.save_to_db(frame)


def test_save_rejects_raw_subtype_and_normalized_type_disagreement():
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[0, "raw_transaction_subtype"] = "Purchase"

    with pytest.raises(SenateEFDError, match="subtype binding mismatch"):
        source.save_to_db(frame)


@pytest.mark.parametrize("mutation", ["raw_ticker", "description", "asset_class"])
def test_save_rederives_exact_ticker_triple_from_raw_fields(mutation):
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[
        0,
        [
            "ticker",
            "ticker_candidate",
            "ticker_origin",
            "raw_ticker",
            "asset_description",
            "raw_asset_description",
            "raw_asset_class",
            "instrument_type",
        ],
    ] = [
        None,
        "ACME",
        "unverified",
        "--",
        "Acme Holdings (ACME)",
        "Acme Holdings (ACME)",
        "",
        "unknown",
    ]
    if mutation == "raw_ticker":
        frame.loc[0, "raw_ticker"] = "MSFT"
    elif mutation == "description":
        frame.loc[0, ["asset_description", "raw_asset_description"]] = [
            "Beta Holdings (BETA)",
            "Beta Holdings (BETA)",
        ]
    else:
        frame.loc[0, "raw_asset_class"] = "Stock"
        frame.loc[0, "instrument_type"] = "stock"

    with pytest.raises(SenateEFDError, match="ticker/raw binding mismatch"):
        source.save_to_db(frame)


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("raw_owner", "Self", "owner binding mismatch"),
        ("amount_raw", "$15,001 - $50,000", "amount binding mismatch"),
        ("raw_asset_class", "Bond", "asset class binding mismatch"),
    ],
)
def test_save_rejects_owner_amount_and_asset_raw_binding_mutations(
    column, value, error
):
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[0, column] = value

    with pytest.raises(SenateEFDError, match=error):
        source.save_to_db(frame)


def test_normalized_persisted_column_derivation_checklist_is_exhaustive():
    frame = _source()._normalize([_raw_trade()])
    assert set(frame.columns) == set(_NORMALIZED_TRANSACTION_COLUMNS)
    assert len(_DERIVATION_COLUMNS) == len(set(_DERIVATION_COLUMNS))
    assert set(_DERIVATION_COLUMNS) == set(_NORMALIZED_TRANSACTION_COLUMNS)


@pytest.mark.parametrize(
    "report_path",
    [
        f"https://efdsearch.senate.gov{KATIE_REPORT_PATH}",
        f"{KATIE_REPORT_PATH}?download=1",
        f"{KATIE_REPORT_PATH}#fragment",
        f"/search/view/ptr/../ptr/{KATIE_REPORT_ID}/",
        f"/search/view/ptr/{KATIE_REPORT_ID}",
        f" {KATIE_REPORT_PATH}",
    ],
)
def test_save_rejects_noncanonical_or_external_report_paths(report_path):
    source = _ready_persistence_source(_report_inventory_row(report_path=report_path))
    frame = source._normalize([_raw_trade()])

    with pytest.raises(SenateEFDError, match="path/record ID mismatch"):
        source.save_to_db(frame)


def test_save_rejects_external_transaction_report_path():
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[0, "source_report_path"] = (
        f"https://efdsearch.senate.gov{KATIE_REPORT_PATH}"
    )

    with pytest.raises(SenateEFDError, match="report path mismatch"):
        source.save_to_db(frame)


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("asset_description", "Different description", "asset description mismatch"),
        ("member_key", "WRONG MEMBER", "member key mismatch"),
        ("chamber_member_key", "house:KATIE BRITT", "member key mismatch"),
    ],
)
def test_save_rejects_three_persisted_derivation_mutations(column, value, error):
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame.loc[0, column] = value

    with pytest.raises(SenateEFDError, match=error):
        source.save_to_db(frame)


def test_save_rejects_unexpected_transaction_and_report_inventory_fields():
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    frame["unexpected_transaction_field"] = "not allowed"
    with pytest.raises(SenateEFDError, match="schema mismatch.*unexpected"):
        source.save_to_db(frame)

    source = _ready_persistence_source(
        _report_inventory_row(unexpected_report_field="not allowed")
    )
    frame = source._normalize([_raw_trade()])
    with pytest.raises(SenateEFDError, match="schema mismatch.*unexpected"):
        source.save_to_db(frame)


def test_nan_raw_ticker_is_missing_not_literal_nan_ticker():
    ticker, candidate, origin = SenateEFDSource._resolve_ticker(
        float("nan"), "Private holding", ""
    )
    assert ticker is None
    assert candidate is None
    assert origin is TickerOrigin.MISSING

    source = _ready_persistence_source()
    frame = source._normalize(
        [
            _raw_trade(
                ticker=None,
                ticker_raw=float("nan"),
                ticker_candidate=None,
                ticker_origin=TickerOrigin.MISSING.value,
                asset_name="Private holding",
                asset_type="",
            )
        ]
    )
    assert pd.isna(frame.iloc[0]["raw_ticker"])
    assert source.save_to_db(frame) == 1


def test_save_rejects_inventory_member_with_surrounding_whitespace():
    source = _ready_persistence_source(_report_inventory_row(member=" Katie Britt "))
    frame = source._normalize([_raw_trade()])

    with pytest.raises(SenateEFDError, match="member must be nonblank and stripped"):
        source.save_to_db(frame)


@pytest.mark.parametrize("column", ["ticker_candidate", "member"])
def test_save_rejects_duplicate_nullable_and_nonnullable_column_labels(column):
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    duplicated = pd.concat([frame, frame[[column]]], axis=1)
    assert duplicated.columns.is_unique is False

    with pytest.raises(SenateEFDError, match="schema mismatch.*unique=False"):
        source.save_to_db(duplicated)


def test_save_rejects_noncanonical_transaction_column_order():
    source = _ready_persistence_source()
    frame = source._normalize([_raw_trade()])
    reordered = frame[[*frame.columns[1:], frame.columns[0]]]
    assert set(reordered.columns) == set(frame.columns)
    assert reordered.columns.is_unique

    with pytest.raises(SenateEFDError, match="canonical_order=False"):
        source.save_to_db(reordered)


def test_get_transactions_filters_cache_to_senate_source():
    calls = []

    class Database:
        @staticmethod
        def get_transactions(year, *, source=None):
            calls.append((year, source))
            return pd.DataFrame({"source": [source]})

    source = _source()
    source.db = Database()

    result = source.get_transactions(2026)

    assert calls == [(2026, "senate_efd")]
    assert result.iloc[0]["source"] == "senate_efd"
