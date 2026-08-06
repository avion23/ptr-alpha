from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.metadata_repository import MetadataRepository
from analyzer.parse_run_repository import ParseRunRepository
from analyzer.price_repository import PriceRepository
from analyzer.ticker_resolver import TickerResolver
from analyzer.transaction_repository import TransactionRepository


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

    # -- schema init (stays here) ---------------------------------------------

    def _init_schema(self):
        self._init_metadata_table()
        self._init_pdf_tables()
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
        try:
            self.conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_unique_v2 "
                "ON transactions(doc_id, ticker, transaction_date, member, transaction_type, "
                "amount_raw, owner_code, asset_description)"
            )
        except duckdb.ConstraintException as e:
            raise RuntimeError(
                "Failed to create transactions unique index. Run "
                "`python3 scripts/purge_phantom_rows.py --execute` first."
            ) from e
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

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
