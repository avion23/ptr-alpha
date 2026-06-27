from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.ticker_resolver import TickerResolver


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
        self._init_metadata_table()
        self._init_transactions_table()
        self._init_prices_table()
        self._init_pdf_tables()

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
        for column in ["owner_code", "amount_raw", "amount_midpoint", "instrument_type", "strike_price", "expiry_date", "asset_description"]:
            if column not in df.columns:
                df[column] = None
        df["created_at"] = datetime.now()

        # Check if asset_description column exists in the table
        has_asset_desc = any(
            r[0] == "asset_description"
            for r in self.conn.execute("DESCRIBE transactions").fetchall()
        )

        self.conn.execute("CREATE TEMP TABLE staging_transactions AS SELECT * FROM df")
        if has_asset_desc:
            self.conn.execute("""
                INSERT INTO transactions (
                    doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                    owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date, created_at,
                    asset_description
                )
                SELECT doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                       owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date, created_at,
                       asset_description
                FROM staging_transactions
                ON CONFLICT (doc_id, ticker, transaction_date, member, transaction_type) DO UPDATE SET
                    transaction_type = EXCLUDED.transaction_type,
                    disclosure_date = EXCLUDED.disclosure_date,
                    owner_code = EXCLUDED.owner_code,
                    amount_raw = EXCLUDED.amount_raw,
                    amount_midpoint = EXCLUDED.amount_midpoint,
                    instrument_type = EXCLUDED.instrument_type,
                    strike_price = EXCLUDED.strike_price,
                    expiry_date = EXCLUDED.expiry_date,
                    created_at = EXCLUDED.created_at,
                    asset_description = EXCLUDED.asset_description
            """)
        else:
            self.conn.execute("""
                INSERT INTO transactions (
                    doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                    owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date, created_at
                )
                SELECT doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                       owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date, created_at
                FROM staging_transactions
                ON CONFLICT (doc_id, ticker, transaction_date, member, transaction_type) DO UPDATE SET
                    transaction_type = EXCLUDED.transaction_type,
                    disclosure_date = EXCLUDED.disclosure_date,
                    owner_code = EXCLUDED.owner_code,
                    amount_raw = EXCLUDED.amount_raw,
                    amount_midpoint = EXCLUDED.amount_midpoint,
                    instrument_type = EXCLUDED.instrument_type,
                    strike_price = EXCLUDED.strike_price,
                    expiry_date = EXCLUDED.expiry_date,
                    created_at = EXCLUDED.created_at
            """)
        self.conn.execute("DROP TABLE staging_transactions")

    def delete_transactions_for_doc(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM transactions WHERE doc_id = ?", [doc_id])

    def upsert_parse_run(
        self,
        doc_id: str,
        year: int,
        parser_version: str,
        status: str,
        engines_attempted: str,
        raw_row_count: int,
        transaction_count: int,
        error_message: str = "",
    ) -> None:
        self.conn.execute("""
            INSERT INTO pdf_parse_runs (
                doc_id, year, parser_version, status, engines_attempted,
                raw_row_count, transaction_count, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [doc_id, year, parser_version, status, engines_attempted,
              raw_row_count, transaction_count, error_message])

    def get_transactions(self, year: int) -> pd.DataFrame:
        result = self.conn.execute(
            """
            SELECT member, ticker, transaction_date, disclosure_date, transaction_type,
                   owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date
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
                   owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date
            FROM transactions
            WHERE disclosure_date BETWEEN ? AND ?
              AND transaction_date <= disclosure_date
            ORDER BY disclosure_date DESC
        """,
            [start_date, end_date],
        ).fetchdf()
        return result

    def get_entry_prices(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
        max_staleness_days: int = 30,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        # Resolve tickers so the ASOF join can match both raw and resolved symbols
        resolver = TickerResolver()
        resolutions = resolver.resolve_batch(tickers)
        # Build expanded ticker set: raw tickers + their resolved yfinance symbols
        expanded_tickers: list[str] = []
        seen: set[str] = set()
        # Build raw→resolved mapping for the SQL CTE
        ticker_map_entries: list[tuple[str, str]] = []
        for raw in tickers:
            if raw not in seen:
                seen.add(raw)
                expanded_tickers.append(raw)
            resolved = resolutions[raw].price_symbol
            ticker_map_entries.append((raw, resolved))
            if resolved not in seen:
                seen.add(resolved)
                expanded_tickers.append(resolved)

        # Build VALUES clause for the ticker resolution CTE
        values_parts = [
            f"('{raw.replace(chr(39), chr(39)*2)}', '{res.replace(chr(39), chr(39)*2)}')"
            for raw, res in ticker_map_entries
        ]
        values_str = ", ".join(values_parts)

        result = self.conn.execute(
            f"""
            WITH ticker_map(raw, resolved) AS (
                VALUES {values_str}
            ),
            resolved_tickers AS (
                SELECT t.*, COALESCE(tm.resolved, t.ticker) AS resolved_ticker
                FROM transactions t
                LEFT JOIN ticker_map tm ON t.ticker = tm.raw
            )
            SELECT r.member, r.ticker, r.disclosure_date, r.transaction_type,
                   r.owner_code, r.amount_midpoint, r.instrument_type, r.strike_price, r.expiry_date,
                   p.close AS entry_price, p.date AS entry_price_date
            FROM resolved_tickers r
            ASOF JOIN prices p
              ON r.resolved_ticker = p.ticker
              AND p.date <= r.disclosure_date
            WHERE r.ticker IN (SELECT UNNEST(?))
              AND r.disclosure_date BETWEEN ? AND ?
              AND p.close IS NOT NULL
        """,
            [expanded_tickers, start_date, end_date],
        ).fetchdf()

        if not result.empty and max_staleness_days is not None:
            result["disclosure_date"] = pd.to_datetime(result["disclosure_date"])
            result["entry_price_date"] = pd.to_datetime(result["entry_price_date"])
            staleness = (result["disclosure_date"] - result["entry_price_date"]).dt.days
            stale_mask = staleness > max_staleness_days
            result.loc[stale_mask, "entry_price"] = None
            result.loc[stale_mask, "member"] = None
            result = result.dropna(subset=["member"])
            result = result.drop(columns=["entry_price_date"])
        elif not result.empty:
            result = result.drop(columns=["entry_price_date"])

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
