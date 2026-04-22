import duckdb
import pandas as pd
import pathlib
import logging
from datetime import datetime
from analyzer.exceptions import AnalysisError


logger = logging.getLogger(__name__)


class DatabaseError(AnalysisError):
    pass


class Database:
    def __init__(self, db_path: str | pathlib.Path):
        self.db_path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.conn = duckdb.connect(str(self.db_path))
        except duckdb.Error as e:
            raise DatabaseError(f"Failed to open database at {self.db_path}: {e}")
        self._init_schema()

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
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
        df["created_at"] = datetime.now()

        self.conn.execute("CREATE TEMP TABLE staging_transactions AS SELECT * FROM df")
        self.conn.execute("""
            INSERT INTO transactions (doc_id, member, ticker, transaction_date, disclosure_date, transaction_type, created_at)
            SELECT doc_id, member, ticker, transaction_date, disclosure_date, transaction_type, created_at
            FROM staging_transactions
            ON CONFLICT DO NOTHING
        """)
        self.conn.execute("DROP TABLE staging_transactions")

    def get_transactions(self, year: int) -> pd.DataFrame:
        result = self.conn.execute(
            """
            SELECT member, ticker, transaction_date, disclosure_date, transaction_type
            FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
            ORDER BY disclosure_date DESC
        """,
            [year],
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

        df_reset = df.reset_index()
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

    def get_prices(self, tickers: list[str], start_date, end_date) -> pd.DataFrame:
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

    def get_missing_price_data(self, tickers: list[str], start_date, end_date) -> tuple[list[str], pd.DatetimeIndex]:
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
            return tickers, all_dates

        existing_tickers = set(existing["ticker"].unique())
        missing_tickers = [t for t in tickers if t not in existing_tickers]

        if not missing_tickers:
            existing_dates = set(existing["date"].unique())
            missing_dates = [d for d in all_dates if d not in existing_dates]
            return [], missing_dates

        return missing_tickers, all_dates

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
