from __future__ import annotations

import logging
from datetime import date, datetime

import duckdb
import pandas as pd


logger = logging.getLogger(__name__)


class TransactionRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def get_by_year(self, year: int) -> pd.DataFrame:
        excluded = self.conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
              AND transaction_date IS NOT NULL
              AND transaction_date > disclosure_date
            """,
            [year],
        ).fetchone()[0]
        if excluded > 0:
            logger.debug(
                "Excluding %d transactions with transaction_date > disclosure_date "
                "(likely OCR date swap) for year %d",
                excluded,
                year,
            )
        result = self.conn.execute(
            """
            SELECT member, ticker, transaction_date, disclosure_date, transaction_type,
                   owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date
            FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
              AND (transaction_date IS NULL OR transaction_date <= disclosure_date)
            ORDER BY disclosure_date DESC
        """,
            [year],
        ).fetchdf()
        return result

    def get_by_date_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        excluded = self.conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE disclosure_date BETWEEN ? AND ?
              AND transaction_date IS NOT NULL
              AND transaction_date > disclosure_date
            """,
            [start_date, end_date],
        ).fetchone()[0]
        if excluded > 0:
            logger.debug(
                "Excluding %d transactions with transaction_date > disclosure_date "
                "(likely OCR date swap) for date range %s to %s",
                excluded,
                start_date,
                end_date,
            )
        result = self.conn.execute(
            """
            SELECT member, ticker, transaction_date, disclosure_date, transaction_type,
                   owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date
            FROM transactions
            WHERE disclosure_date BETWEEN ? AND ?
              AND (transaction_date IS NULL OR transaction_date <= disclosure_date)
            ORDER BY disclosure_date DESC
        """,
            [start_date, end_date],
        ).fetchdf()
        return result

    def upsert(self, df: pd.DataFrame, *, source: str) -> None:
        df = df.copy()
        for column in ["owner_code", "amount_raw", "amount_midpoint", "instrument_type", "strike_price", "expiry_date", "asset_description"]:
            if column not in df.columns:
                df[column] = None
        df["owner_code"] = df["owner_code"].fillna("").astype(str).replace("None", "")
        df["amount_raw"] = df["amount_raw"].fillna("").astype(str).replace("None", "")
        df["created_at"] = datetime.now()
        df["source"] = source

        dedup_key = [
            "doc_id", "ticker_key", "transaction_date", "member",
            "transaction_type", "amount_raw", "owner_code", "asset_description_key",
        ]
        dedup_df = df.copy()
        dedup_df["ticker_key"] = dedup_df["ticker"].fillna("").astype(str).replace("None", "")
        dedup_df["asset_description_key"] = dedup_df["asset_description"].fillna("").astype(str).replace("None", "")
        df = df.loc[~dedup_df.duplicated(subset=dedup_key, keep="first")].copy()
        if df.empty:
            return

        self.conn.execute("CREATE TEMP TABLE staging_transactions AS SELECT * FROM df")
        try:
            self.conn.execute("""
                CREATE TEMP TABLE filtered_staging_transactions AS
                SELECT s.*
                FROM staging_transactions s
                LEFT JOIN transactions t
                  ON t.doc_id = s.doc_id
                 AND COALESCE(CAST(t.ticker AS VARCHAR), '') = COALESCE(CAST(s.ticker AS VARCHAR), '')
                 AND t.transaction_date IS NOT DISTINCT FROM s.transaction_date
                 AND t.member IS NOT DISTINCT FROM s.member
                 AND t.transaction_type IS NOT DISTINCT FROM s.transaction_type
                 AND COALESCE(t.amount_raw, '') = COALESCE(s.amount_raw, '')
                 AND COALESCE(t.owner_code, '') = COALESCE(s.owner_code, '')
                 AND COALESCE(CAST(t.asset_description AS VARCHAR), '') = COALESCE(CAST(s.asset_description AS VARCHAR), '')
                WHERE t.id IS NULL
            """)
            self.conn.execute("""
                INSERT INTO transactions (
                    doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                    owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date, created_at,
                    asset_description, source
                )
                SELECT doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                       owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date, created_at,
                       asset_description, source
                FROM filtered_staging_transactions
                WHERE ticker IS NOT NULL
                ON CONFLICT (doc_id, ticker, transaction_date, member, transaction_type, amount_raw, owner_code, asset_description) DO UPDATE SET
                    transaction_type = EXCLUDED.transaction_type,
                    disclosure_date = EXCLUDED.disclosure_date,
                    owner_code = EXCLUDED.owner_code,
                    amount_raw = EXCLUDED.amount_raw,
                    amount_midpoint = EXCLUDED.amount_midpoint,
                    instrument_type = EXCLUDED.instrument_type,
                    strike_price = EXCLUDED.strike_price,
                    expiry_date = EXCLUDED.expiry_date,
                    created_at = EXCLUDED.created_at,
                    asset_description = EXCLUDED.asset_description,
                    source = COALESCE(transactions.source, EXCLUDED.source)
            """)
            self.conn.execute("""
                INSERT INTO transactions (
                    doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                    owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date, created_at,
                    asset_description, source
                )
                SELECT doc_id, member, ticker, transaction_date, disclosure_date, transaction_type,
                       owner_code, amount_raw, amount_midpoint, instrument_type, strike_price, expiry_date, created_at,
                       asset_description, source
                FROM filtered_staging_transactions
                WHERE ticker IS NULL
            """)
        finally:
            self.conn.execute("DROP TABLE IF EXISTS filtered_staging_transactions")
            self.conn.execute("DROP TABLE IF EXISTS staging_transactions")

    def delete_for_doc(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM transactions WHERE doc_id = ?", [doc_id])

    def count_for_docs(self, doc_ids: list[str]) -> dict[str, int]:
        if not doc_ids:
            return {}

        placeholders = ", ".join("?" for _ in doc_ids)
        rows = self.conn.execute(
            f"""
            SELECT doc_id, COUNT(*)
            FROM transactions
            WHERE doc_id IN ({placeholders})
            GROUP BY doc_id
            """,
            doc_ids,
        ).fetchall()
        return {doc_id: count for doc_id, count in rows}

    def exists(self, year: int) -> bool:
        count = self.conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
        """,
            [year],
        ).fetchone()[0]
        return count > 0
