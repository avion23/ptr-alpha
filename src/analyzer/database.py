from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.exceptions import AnalysisError


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

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    def _init_schema(self):
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._ensure_transaction_columns()
        self.conn.execute("DROP INDEX IF EXISTS idx_tx_unique")
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_unique ON transactions(doc_id, ticker, transaction_date, member, transaction_type)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_year ON transactions(EXTRACT(YEAR FROM disclosure_date))"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_ticker ON transactions(ticker)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tx_member ON transactions(member)"
        )

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

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pdf_downloads (
                doc_id VARCHAR PRIMARY KEY,
                year INTEGER,
                status VARCHAR,
                status_code INTEGER,
                error_message VARCHAR,
                downloaded_at TIMESTAMP
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
        }
        for column, column_type in required_columns.items():
            if column not in existing_columns:
                self.conn.execute(f"ALTER TABLE transactions ADD COLUMN {column} {column_type}")

    def upsert_metadata(self, df: pd.DataFrame) -> None:
        self.conn.execute("""
            INSERT INTO metadata (doc_id, first_name, last_name, filing_date, filing_type, fetched_at)
            SELECT doc_id, first_name, last_name, filing_date, filing_type, fetched_at
            FROM df
            ON CONFLICT (doc_id) DO UPDATE SET
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                filing_date = EXCLUDED.filing_date,
                filing_type = EXCLUDED.filing_type,
                fetched_at = EXCLUDED.fetched_at
        """)

    def get_metadata(self, year: int) -> pd.DataFrame:
        result = self.conn.execute(
            """
            SELECT doc_id AS "DocID", first_name AS "First", last_name AS "Last",
                   filing_date AS "FilingDate", filing_type AS "FilingType"
            FROM metadata
            WHERE EXTRACT(YEAR FROM filing_date) = ?
        """,
            [year],
        ).fetchdf()
        return result

    def metadata_exists(self, year: int) -> bool:
        count = self.conn.execute(
            """
            SELECT COUNT(*) FROM metadata
            WHERE EXTRACT(YEAR FROM filing_date) = ?
        """,
            [year],
        ).fetchone()[0]
        return count > 0

    def clear_metadata(self, year: int) -> None:
        self.conn.execute(
            """
            DELETE FROM metadata
            WHERE EXTRACT(YEAR FROM filing_date) = ?
        """,
            [year],
        )
        logger.info(f"Cleared metadata for year {year}")

    def upsert_transactions(self, df: pd.DataFrame) -> None:
        df = df.copy()
        for column in ["owner_code", "amount_raw", "amount_midpoint"]:
            if column not in df.columns:
                df[column] = None
        df["created_at"] = datetime.now()

        self.conn.execute("CREATE TEMP TABLE staging_transactions AS SELECT * FROM df")
        self.conn.execute("""
            INSERT INTO transactions (
                doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                owner_code, amount_raw, amount_midpoint, created_at
            )
            SELECT doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                   owner_code, amount_raw, amount_midpoint, created_at
            FROM staging_transactions
            ON CONFLICT (doc_id, ticker, transaction_date, member, transaction_type) DO UPDATE SET
                transaction_type = EXCLUDED.transaction_type,
                disclosure_date = EXCLUDED.disclosure_date,
                owner_code = EXCLUDED.owner_code,
                amount_raw = EXCLUDED.amount_raw,
                amount_midpoint = EXCLUDED.amount_midpoint,
                created_at = EXCLUDED.created_at
        """)
        self.conn.execute("DROP TABLE staging_transactions")

    def get_transactions(self, year: int) -> pd.DataFrame:
        result = self.conn.execute(
            """
            SELECT member, ticker, transaction_date, disclosure_date, transaction_type,
                   owner_code, amount_raw, amount_midpoint
            FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
            ORDER BY disclosure_date DESC
        """,
            [year],
        ).fetchdf()
        return result

    def get_transactions_by_date_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        result = self.conn.execute(
            """
            SELECT member, ticker, transaction_date, disclosure_date, transaction_type,
                   owner_code, amount_raw, amount_midpoint
            FROM transactions
            WHERE disclosure_date BETWEEN ? AND ?
              AND transaction_date <= disclosure_date
            ORDER BY disclosure_date DESC
        """,
            [start_date, end_date],
        ).fetchdf()
        return result

    def get_entry_prices(self, tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        result = self.conn.execute(
            """
            SELECT t.member, t.ticker, t.disclosure_date, t.transaction_type,
                   t.owner_code, t.amount_midpoint, p.close AS entry_price
            FROM transactions t
            ASOF JOIN prices p
              ON t.ticker = p.ticker
              AND p.date <= t.disclosure_date
            WHERE t.ticker IN (SELECT UNNEST(?))
              AND t.disclosure_date BETWEEN ? AND ?
              AND p.close IS NOT NULL
        """,
            [tickers, start_date, end_date],
        ).fetchdf()

        return result

    def transactions_exist(self, year: int) -> bool:
        count = self.conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
        """,
            [year],
        ).fetchone()[0]
        return count > 0

    def upsert_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        df_reset = df.reset_index().copy()
        index_col_name = df_reset.columns[0]
        prices_long = df_reset.melt(
            id_vars=[index_col_name], var_name="ticker", value_name="close"
        )
        prices_long = prices_long.rename(columns={index_col_name: "date"})
        prices_long = prices_long.dropna(subset=["close"])

        self.conn.execute("""
            INSERT INTO prices (ticker, date, close)
            SELECT ticker, date, close
            FROM prices_long
            ON CONFLICT (ticker, date) DO UPDATE SET
                close = EXCLUDED.close
        """)

    def get_prices(self, tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        result = self.conn.execute(
            """
            SELECT date, ticker, close
            FROM prices
            WHERE ticker IN (SELECT UNNEST(?))
              AND date BETWEEN ? AND ?
            ORDER BY date
        """,
            [tickers, start_date, end_date],
        ).fetchdf()

        if result.empty:
            return pd.DataFrame()

        pivot = result.pivot(index="date", columns="ticker", values="close")
        return pivot

    def get_missing_price_data(self, tickers: list[str], start_date: date, end_date: date) -> tuple[list[str], list[pd.Timestamp]]:
        all_dates = pd.date_range(start_date, end_date, freq="B")
        existing = self.conn.execute(
            """
            SELECT DISTINCT ticker, date
            FROM prices
            WHERE ticker IN (SELECT UNNEST(?))
              AND date BETWEEN ? AND ?
        """,
            [tickers, start_date, end_date],
        ).fetchdf()

        if existing.empty:
            return tickers, all_dates.to_list()

        existing_tickers = set(existing["ticker"].unique())
        missing_tickers = [t for t in tickers if t not in existing_tickers]

        start_cutoff = pd.Timestamp(start_date) + pd.Timedelta(days=7)
        ticker_starts = existing.groupby("ticker")["date"].min()
        insufficient: list[str] = []
        for t in tickers:
            if t in ticker_starts.index:
                val = ticker_starts.loc[t]
                if pd.notna(val) and pd.Timestamp(val) > start_cutoff:
                    insufficient.append(t)

        need_full_fetch = missing_tickers + insufficient
        if need_full_fetch:
            return need_full_fetch, all_dates.to_list()

        existing_dates = set(existing["date"].unique())
        missing_dates = [d for d in all_dates if d not in existing_dates]
        return [], missing_dates

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
