from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.metadata_repository import MetadataRepository
from analyzer.parse_run_repository import ParseRunRepository
from analyzer.price_repository import PriceRepository
from analyzer.source_report_repository import SourceReportRepository
from analyzer.ticker_resolver import TickerResolver
from analyzer.transaction_repository import (
    SOURCE_TRANSACTION_COLUMNS,
    TransactionRepository,
)


logger = logging.getLogger(__name__)


class DatabaseError(AnalysisError):
    pass


class Database:
    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        try:
            if read_only:
                self.conn = duckdb.connect(str(self.db_path), read_only=True)
            else:
                self.conn = duckdb.connect(str(self.db_path))
        except duckdb.Error as e:
            raise DatabaseError(f"Failed to open database at {self.db_path}: {e}")
        if not read_only:
            self._init_schema()
        self._transactions = TransactionRepository(self.conn)
        self._prices = PriceRepository(self.conn)
        self._metadata = MetadataRepository(self.conn)
        self._parse_runs = ParseRunRepository(self.conn)
        self._source_reports = SourceReportRepository(self.conn)

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    # -- repository accessors --------------------------------------------------

    @property
    def transactions(self) -> TransactionRepository:
        return self._transactions

    @property
    def prices(self) -> PriceRepository:
        return self._prices

    @property
    def metadata(self) -> MetadataRepository:
        return self._metadata

    @property
    def parse_runs(self) -> ParseRunRepository:
        return self._parse_runs

    @property
    def source_reports(self) -> SourceReportRepository:
        return self._source_reports

    # -- schema init (stays here) ---------------------------------------------

    def _init_schema(self):
        self._init_metadata_table()
        self._init_pdf_tables()
        self._init_source_reports_table()
        self._init_transactions_table()
        self._init_prices_table()

    def _init_metadata_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                doc_id VARCHAR PRIMARY KEY,
                first_name VARCHAR,
                last_name VARCHAR,
                filing_date TIMESTAMP,
                filing_type VARCHAR,
                fetched_at TIMESTAMP
            )
        """)

    def _init_transactions_table(self) -> None:
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS tx_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY DEFAULT nextval('tx_id_seq'),
                doc_id VARCHAR,
                member VARCHAR,
                ticker VARCHAR,
                transaction_date DATE,
                disclosure_date DATE,
                transaction_type VARCHAR,
                owner_code VARCHAR,
                amount_raw VARCHAR,
                amount_midpoint DOUBLE,
                instrument_type VARCHAR,
                strike_price DOUBLE,
                expiry_date VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._ensure_transaction_columns()
        self.conn.execute(
            "UPDATE transactions SET owner_code=COALESCE(owner_code,''), amount_raw=COALESCE(amount_raw,'')"
        )
        self.conn.execute("DROP INDEX IF EXISTS idx_tx_unique")
        self.conn.execute("DROP INDEX IF EXISTS idx_tx_unique_v2")
        self.conn.execute("DROP INDEX IF EXISTS idx_tx_unique_v3")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_source_row_unique "
            "ON transactions(source, chamber, source_record_id, source_row_id, ingestion_generation)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_year ON transactions(EXTRACT(YEAR FROM disclosure_date))"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_ticker ON transactions(ticker)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_member ON transactions(member)"
        )

    def _init_prices_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker VARCHAR,
                date DATE,
                close DOUBLE,
                PRIMARY KEY (ticker, date)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")

    def _init_pdf_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pdf_parse_runs (
                doc_id VARCHAR,
                year INTEGER,
                parser_version VARCHAR,
                status VARCHAR,
                engines_attempted VARCHAR,
                raw_row_count INTEGER,
                transaction_count INTEGER,
                error_message VARCHAR,
                parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _init_source_reports_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS source_reports (
                ingestion_generation VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                chamber VARCHAR NOT NULL,
                source_record_id VARCHAR NOT NULL,
                report_path VARCHAR,
                member VARCHAR,
                official_filing_date DATE,
                outcome VARCHAR NOT NULL CHECK (
                    outcome IN ('parsed', 'paper_only', 'unavailable', 'failed')
                ),
                artifact_sha256 VARCHAR,
                landing_sha256 VARCHAR,
                paper_artifact_url VARCHAR,
                paper_artifact_sha256 VARCHAR,
                error_message VARCHAR,
                raw_row_count INTEGER NOT NULL,
                accepted_row_count INTEGER NOT NULL,
                rejected_row_count INTEGER NOT NULL,
                UNIQUE (
                    ingestion_generation, source, chamber, source_record_id
                )
            )
        """)

    def _ensure_transaction_columns(self) -> None:
        existing_columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info('transactions')").fetchall()
        }
        required_columns = {
            "owner_code": "VARCHAR",
            "amount_raw": "VARCHAR",
            "amount_midpoint": "DOUBLE",
            "instrument_type": "VARCHAR",
            "strike_price": "DOUBLE",
            "expiry_date": "VARCHAR",
            "asset_description": "VARCHAR",
            "source": "VARCHAR",
            "chamber": "VARCHAR",
            "source_record_id": "VARCHAR",
            "source_row_id": "VARCHAR",
            "source_report_path": "VARCHAR",
            "official_filing_date": "DATE",
            "available_date": "DATE",
            "notification_date": "DATE",
            "amends_source_record_id": "VARCHAR",
            "raw_transaction_subtype": "VARCHAR",
            "ticker_origin": "VARCHAR",
            "raw_ticker": "VARCHAR",
            "ticker_candidate": "VARCHAR",
            "raw_asset_class": "VARCHAR",
            "raw_asset_description": "VARCHAR",
            "raw_owner": "VARCHAR",
            "ingestion_generation": "VARCHAR",
            "artifact_sha256": "VARCHAR",
        }
        for column, column_type in required_columns.items():
            if column not in existing_columns:
                self.conn.execute(
                    f"ALTER TABLE transactions ADD COLUMN {column} {column_type}"
                )

        self.conn.execute("""
            UPDATE transactions
            SET source = 'gemini_ocr'
            WHERE source IS NULL
              AND doc_id IN (
                  SELECT doc_id FROM pdf_parse_runs
                  WHERE parser_version LIKE 'v4-gemini%'
              )
        """)
        self.conn.execute("""
            UPDATE transactions
            SET source = 'capitol_trades'
            WHERE source IS NULL
              AND doc_id LIKE 'ct-%'
        """)

    # -- delegating facade methods (backward compatibility) --------------------

    def upsert_metadata(self, df: pd.DataFrame) -> None:
        self.metadata.upsert(df)

    def get_metadata(self, year: int) -> pd.DataFrame:
        return self.metadata.get_by_year(year)

    def metadata_exists(self, year: int) -> bool:
        return self.metadata.exists(year)

    def clear_metadata(self, year: int) -> None:
        self.metadata.clear(year)

    def replace_metadata(self, year: int, df: pd.DataFrame) -> None:
        self.metadata.replace_year(year, df)

    def upsert_transactions(self, df: pd.DataFrame, *, source: str) -> int:
        return self.transactions.upsert(df, source=source)

    def replace_transactions_for_docs(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        parse_runs: list[dict] | None = None,
    ) -> None:
        """Atomically replace parsed transactions and their optional audit records."""
        doc_ids = df["doc_id"].unique().tolist()
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for doc_id in doc_ids:
                self.transactions.delete_for_doc(doc_id)
            self.transactions.upsert(df, source=source, _in_transaction=True)
            for parse_run in parse_runs or []:
                self.parse_runs.upsert(**parse_run, _in_transaction=True)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def get_transactions(self, year: int) -> pd.DataFrame:
        return self.transactions.get_by_year(year)

    def get_transactions_by_date_range(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        return self.transactions.get_by_date_range(start_date, end_date)

    def delete_transactions_for_doc(self, doc_id: str) -> None:
        self.transactions.delete_for_doc(doc_id)

    def get_transactions_for_doc(self, doc_id: str) -> pd.DataFrame:
        return self.transactions.get_for_doc(doc_id)

    def count_transactions_for_docs(self, doc_ids: list[str]) -> dict[str, int]:
        return self.transactions.count_for_docs(doc_ids)

    def transactions_exist(self, year: int) -> bool:
        return self.transactions.exists(year)

    def upsert_prices(self, df: pd.DataFrame) -> None:
        self.prices.upsert(df)

    def get_prices(
        self, tickers: list[str], start_date: date, end_date: date
    ) -> pd.DataFrame:
        return self.prices.get(tickers, start_date, end_date)

    def get_missing_price_data(
        self, tickers: list[str], start_date: date, end_date: date
    ) -> tuple[list[str], list[pd.Timestamp]]:
        return self.prices.get_missing(tickers, start_date, end_date)

    def get_entry_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
        max_staleness_days: int = 30,
        resolver: TickerResolver | None = None,
    ) -> pd.DataFrame:
        return self.prices.get_entry_prices(
            tickers,
            start_date,
            end_date,
            max_staleness_days=max_staleness_days,
            resolver=resolver,
        )

    def upsert_parse_run(
        self,
        doc_id: str,
        year: int,
        parser_version: str,
        status: str,
        engines_attempted: str,
        raw_row_count: int,
        transaction_count: int,
        error_message: str | None = None,
    ) -> None:
        self.parse_runs.upsert(
            doc_id=doc_id,
            year=year,
            parser_version=parser_version,
            status=status,
            engines_attempted=engines_attempted,
            raw_row_count=raw_row_count,
            transaction_count=transaction_count,
            error_message=error_message,
        )

    def replace_source_reports(
        self,
        generation: str,
        source: str,
        chamber: str,
        reports_df: pd.DataFrame,
    ) -> None:
        self.source_reports.replace_generation(generation, source, chamber, reports_df)

    def get_source_reports(
        self, generation: str, source: str, chamber: str
    ) -> pd.DataFrame:
        return self.source_reports.get(generation, source, chamber)

    def get_source_report_reconciliation(
        self, generation: str, source: str, chamber: str
    ) -> dict[str, int]:
        return self.source_reports.reconcile(generation, source, chamber)

    def persist_source_refresh(
        self,
        *,
        transactions: pd.DataFrame,
        reports: pd.DataFrame,
        source: str,
        chamber: str,
        ingestion_generation: str,
    ) -> int:
        """Atomically replace a complete source refresh and its report inventory."""
        self.source_reports.validate_replacement(
            ingestion_generation, source, chamber, reports
        )
        self._validate_source_refresh_transactions(
            transactions=transactions,
            reports=reports,
            source=source,
            chamber=chamber,
            ingestion_generation=ingestion_generation,
        )

        self.conn.execute("BEGIN TRANSACTION")
        try:
            inserted = self.transactions.replace_source_refresh(
                transactions,
                source=source,
                chamber=chamber,
                ingestion_generation=ingestion_generation,
                _in_transaction=True,
            )
            self.source_reports.replace_source_refresh(
                ingestion_generation,
                source,
                chamber,
                reports,
                _in_transaction=True,
            )
            self.conn.execute("COMMIT")
            return inserted
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _validate_source_refresh_transactions(
        *,
        transactions: pd.DataFrame,
        reports: pd.DataFrame,
        source: str,
        chamber: str,
        ingestion_generation: str,
    ) -> None:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

        missing = set(SOURCE_TRANSACTION_COLUMNS) - set(transactions.columns)
        if missing:
            raise ValueError(
                "source transaction columns are missing: " + ", ".join(sorted(missing))
            )
        if (
            "source" in transactions.columns
            and not transactions["source"].eq(source).all()
        ):
            raise ValueError(
                "all source transactions must match the persistence source"
            )
        if not transactions["chamber"].eq(chamber).all():
            raise ValueError(
                "all source transactions must match the persistence chamber"
            )
        if not transactions["ingestion_generation"].eq(ingestion_generation).all():
            raise ValueError(
                "all source transactions must match the ingestion generation"
            )

        required_values = [
            "doc_id",
            "source_record_id",
            "source_row_id",
            "source_report_path",
            "member",
            "transaction_date",
            "disclosure_date",
            "official_filing_date",
            "available_date",
            "transaction_type",
            "raw_transaction_subtype",
            "amount_raw",
            "raw_asset_description",
            "ticker_origin",
            "artifact_sha256",
        ]
        if transactions[required_values].isna().any().any():
            raise ValueError("source transaction provenance values are incomplete")

        for column in [
            "source_record_id",
            "source_row_id",
            "source_report_path",
            "member",
        ]:
            invalid = transactions[column].map(
                lambda value: not isinstance(value, str) or not value.strip()
            )
            if invalid.any():
                raise ValueError(f"{column} must be a non-empty string")

        duplicate_rows = transactions.duplicated(
            subset=["source_record_id", "source_row_id"], keep=False
        )
        if duplicate_rows.any():
            duplicates = transactions.loc[
                duplicate_rows, ["source_record_id", "source_row_id"]
            ].drop_duplicates()
            rendered = [
                f"{row.source_record_id}/{row.source_row_id}"
                for row in duplicates.itertuples(index=False)
            ]
            raise ValueError(
                "duplicate source row identities are not allowed: "
                + ", ".join(rendered[:10])
            )

        Database._validate_ticker_origin_matrix(transactions)

        report_index = reports.set_index("source_record_id")
        transaction_report_ids = set(transactions["source_record_id"])
        unknown_reports = transaction_report_ids - set(report_index.index)
        if unknown_reports:
            raise ValueError(
                "source transactions have no report inventory entry: "
                + ", ".join(sorted(str(value) for value in unknown_reports)[:10])
            )

        senate_path = re.compile(
            r"^/search/view/ptr/"
            r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/$"
        )
        if source == "senate_efd":
            for report in reports.itertuples(index=False):
                match = senate_path.fullmatch(str(report.report_path))
                if match is None or match.group(1) != report.source_record_id:
                    raise ValueError(
                        "Senate report_path must be the canonical path for "
                        f"source_record_id: {report.source_record_id}"
                    )

        binding_columns = [
            "source_record_id",
            "source_report_path",
            "doc_id",
            "member",
            "artifact_sha256",
            "official_filing_date",
            "available_date",
            "disclosure_date",
        ]
        for row in transactions[binding_columns].itertuples(index=False):
            report = report_index.loc[row.source_record_id]
            if row.source_report_path != report["report_path"]:
                raise ValueError(
                    "source transaction report path does not match inventory: "
                    f"{row.source_record_id}"
                )
            if row.member != report["member"]:
                raise ValueError(
                    "source transaction member does not match report inventory: "
                    f"{row.source_record_id}"
                )
            if source == "senate_efd" and row.doc_id != row.source_record_id:
                raise ValueError(
                    "Senate source transaction doc_id must equal source_record_id"
                )
            if row.artifact_sha256 != report["landing_sha256"]:
                raise ValueError(
                    "source transaction artifact hash does not match report landing hash: "
                    f"{row.source_record_id}"
                )

            report_date = pd.to_datetime(
                report["official_filing_date"], errors="coerce"
            )
            bound_dates = [
                pd.to_datetime(value, errors="coerce")
                for value in (
                    row.official_filing_date,
                    row.available_date,
                    row.disclosure_date,
                )
            ]
            if pd.isna(report_date) or any(pd.isna(value) for value in bound_dates):
                raise ValueError("report-bound dates must be valid dates")
            if any(value.date() != report_date.date() for value in bound_dates):
                raise ValueError(
                    "source transaction dates do not match report inventory: "
                    f"{row.source_record_id}"
                )

        actual_counts = transactions["source_record_id"].value_counts().to_dict()
        for report in reports.itertuples(index=False):
            actual = int(actual_counts.get(report.source_record_id, 0))
            accepted = int(report.accepted_row_count)
            if actual != accepted:
                raise ValueError(
                    "accepted transaction count does not match report inventory: "
                    f"{report.source_record_id} expected={accepted} actual={actual}"
                )
            if actual and report.outcome != "parsed":
                raise ValueError(
                    "transactions may only map to parsed report outcomes: "
                    f"{report.source_record_id}"
                )

    @staticmethod
    def _validate_ticker_origin_matrix(transactions: pd.DataFrame) -> None:
        valid_ticker = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z]{1,2})?$")
        reserved = {
            "COUPON",
            "BOND",
            "BONDS",
            "NOTE",
            "NOTES",
            "STOCK",
            "TICKER",
        }
        allowed_origins = {
            "official",
            "asset_description",
            "unverified",
            "non_equity",
            "missing",
            "invalid",
        }

        def is_null(value: object) -> bool:
            return bool(pd.isna(value))

        def is_valid(value: object) -> bool:
            return isinstance(value, str) and valid_ticker.fullmatch(value) is not None

        for row in transactions[
            ["ticker", "ticker_candidate", "ticker_origin", "raw_ticker"]
        ].itertuples(index=False):
            ticker = row.ticker
            candidate = row.ticker_candidate
            origin = row.ticker_origin
            if origin not in allowed_origins:
                raise ValueError(f"unknown ticker_origin: {origin}")
            if origin == "official":
                if not is_valid(ticker) or not is_null(candidate):
                    raise ValueError("official ticker origin has inconsistent values")
                continue
            if origin == "asset_description":
                if not is_valid(ticker) or ticker in reserved or not is_null(candidate):
                    raise ValueError(
                        "asset_description ticker origin has inconsistent values"
                    )
                continue
            if origin == "unverified":
                if (
                    not is_null(ticker)
                    or not is_valid(candidate)
                    or candidate in reserved
                ):
                    raise ValueError("unverified ticker origin has inconsistent values")
                continue
            if origin in {"non_equity", "missing"}:
                if not is_null(ticker) or not is_null(candidate):
                    raise ValueError(
                        f"{origin} ticker origin must not set ticker values"
                    )
                continue

            if not is_null(ticker):
                raise ValueError("invalid ticker origin must not set canonical ticker")
            if (
                not is_null(candidate)
                and is_valid(candidate)
                and candidate not in reserved
            ):
                raise ValueError("invalid ticker origin has a valid ticker candidate")
            if is_null(candidate):
                raw_ticker = row.raw_ticker
                if (
                    not isinstance(raw_ticker, str)
                    or not raw_ticker.strip()
                    or (is_valid(raw_ticker) and raw_ticker not in reserved)
                ):
                    raise ValueError(
                        "invalid ticker origin requires a rejected raw ticker"
                    )

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
