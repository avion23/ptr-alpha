from __future__ import annotations

import logging
import re
from datetime import date, datetime

import duckdb
import pandas as pd


logger = logging.getLogger(__name__)


class AmbiguousTransactionIdentityError(ValueError):
    """Raised when a legacy economic unique key blocks identity-safe storage."""

_PROVENANCE_COLUMNS = (
    "chamber",
    "member_key",
    "chamber_member_key",
    "source_record_id",
    "source_row_id",
    "source_report_path",
    "official_filing_date",
    "available_date",
    "notification_date",
    "amends_source_record_id",
    "raw_transaction_subtype",
    "ticker_origin",
    "raw_ticker",
    "ticker_candidate",
    "raw_owner",
    "raw_asset_class",
    "raw_asset_description",
    "ingestion_generation",
    "artifact_sha256",
)
_BASE_WRITE_COLUMNS = (
    "doc_id",
    "member",
    "ticker",
    "transaction_date",
    "disclosure_date",
    "transaction_type",
    "owner_code",
    "amount_raw",
    "amount_midpoint",
    "instrument_type",
    "strike_price",
    "expiry_date",
    "created_at",
    "asset_description",
    "source",
)
_ARTIFACT_IDENTITY_COLUMNS = (
    "source",
    "chamber",
    "source_record_id",
    "source_row_id",
    "ingestion_generation",
)
_TYPE_MAP = {
    "Sale Full": "Sale",
    "Sale Partial": "Sale",
    "Partial Sale": "Sale",
}
_OWNER_MAP = {
    "DEPENDENT": "DC",
    "DEPENDENT CHILD": "DC",
    "DC": "DC",
    "SPOUSE": "SP",
    "SP": "SP",
    "JOINT": "J",
    "JT": "JT",
    "J": "J",
    "SELF": "S",
    "S": "S",
}
_INSTRUMENT_MAP = {
    "stock": "stock",
    "call": "call",
    "put": "put",
    "option": "option",
    "stock option": "option",
}


def _normalize_owner(value) -> str:
    if value is None or pd.isna(value):
        return ""
    normalized = str(value).strip().upper()
    return _OWNER_MAP.get(normalized, "")


def _normalize_instrument(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    normalized = str(value).strip().lower()
    return _INSTRUMENT_MAP.get(normalized, normalized)


def _normalize_amount(row) -> float | None:
    raw = "" if pd.isna(row.get("amount_raw")) else str(row.get("amount_raw")).strip()
    midpoint = row.get("amount_midpoint")
    if raw.endswith("-"):
        return None
    values = [
        float(value.replace(",", "")) for value in re.findall(r"\$([0-9][0-9,]*)", raw)
    ]
    if len(values) >= 2:
        return sum(values[:2]) / 2
    if len(values) == 1 and int(values[0]) % 1000 == 1:
        return None
    if midpoint is None or pd.isna(midpoint):
        return None
    return float(midpoint)


def _normalize_frame(df: pd.DataFrame, *, deduplicate: bool) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    original_type = df["transaction_type"].copy()
    if "raw_transaction_subtype" not in df.columns:
        df["raw_transaction_subtype"] = None
    raw_missing = df["raw_transaction_subtype"].isna()
    suffix_type = original_type.isin(_TYPE_MAP)
    df.loc[raw_missing & suffix_type, "raw_transaction_subtype"] = original_type
    df["transaction_type"] = df["transaction_type"].replace(_TYPE_MAP)
    if "owner_code" in df.columns:
        df["owner_code"] = df["owner_code"].map(_normalize_owner)
    if "instrument_type" in df.columns:
        df["instrument_type"] = df["instrument_type"].map(_normalize_instrument)
    if "amount_raw" in df.columns and "amount_midpoint" in df.columns:
        df["amount_midpoint"] = df.apply(_normalize_amount, axis=1)
    if deduplicate and set(_ARTIFACT_IDENTITY_COLUMNS).issubset(df.columns):
        identified = pd.Series(True, index=df.index)
        for column in _ARTIFACT_IDENTITY_COLUMNS:
            values = df[column].fillna("").astype(str).str.strip()
            identified &= values.ne("")
        replay = identified & df.duplicated(
            list(_ARTIFACT_IDENTITY_COLUMNS), keep="last"
        )
        df = df.loc[~replay].copy()

    economic_key = [
        column
        for column in (
            "member",
            "ticker",
            "transaction_date",
            "transaction_type",
            "amount_raw",
            "owner_code",
            "asset_description",
            "raw_asset_description",
        )
        if column in df.columns
    ]
    df["economic_duplicate_candidate"] = (
        df.duplicated(economic_key, keep=False) if economic_key else False
    )
    return df


SOURCE_TRANSACTION_COLUMNS = [
    "doc_id",
    "chamber",
    "source_record_id",
    "source_row_id",
    "source_report_path",
    "member",
    "member_key",
    "chamber_member_key",
    "ticker",
    "raw_ticker",
    "ticker_candidate",
    "transaction_date",
    "disclosure_date",
    "official_filing_date",
    "available_date",
    "notification_date",
    "transaction_type",
    "raw_transaction_subtype",
    "owner_code",
    "raw_owner",
    "amount_raw",
    "amount_midpoint",
    "instrument_type",
    "raw_asset_class",
    "strike_price",
    "expiry_date",
    "asset_description",
    "raw_asset_description",
    "ticker_origin",
    "amends_source_record_id",
    "ingestion_generation",
    "artifact_sha256",
]


class TransactionRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    @staticmethod
    def _source_filter(
        *, source: str | None, sources: tuple[str, ...] | None
    ) -> tuple[str, list[str]]:
        if source is not None and sources is not None:
            raise ValueError("source and sources are mutually exclusive")
        if source is not None:
            return " AND source = ?", [source]
        if sources is None:
            return "", []
        if not sources:
            raise ValueError("sources must not be empty")
        placeholders = ", ".join("?" for _ in sources)
        return f" AND source IN ({placeholders})", list(sources)  # nosec B608

    def get_by_year(
        self,
        year: int,
        *,
        source: str | None = None,
        sources: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        source_clause, source_params = self._source_filter(
            source=source, sources=sources
        )
        params: list[object] = [year, *source_params]
        excluded = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
              AND transaction_date IS NOT NULL
              AND transaction_date > disclosure_date
              {source_clause}
            """,  # nosec B608 -- source_clause is a fixed internal fragment
            params,
        ).fetchone()[0]
        if excluded > 0:
            logger.debug(
                "Excluding %d transactions with transaction_date > disclosure_date "
                "(likely OCR date swap) for year %d",
                excluded,
                year,
            )
        result = self.conn.execute(
            f"""
            SELECT *
            FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ?
              AND (transaction_date IS NULL OR transaction_date <= disclosure_date)
              {source_clause}
            ORDER BY disclosure_date DESC
            """,  # nosec B608 -- source_clause is a fixed internal fragment
            params,
        ).fetchdf()
        return _normalize_frame(result, deduplicate=True)

    def get_by_date_range(
        self,
        start_date: date,
        end_date: date,
        *,
        source: str | None = None,
        sources: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        source_clause, source_params = self._source_filter(
            source=source, sources=sources
        )
        params: list[object] = [start_date, end_date, *source_params]
        excluded = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM transactions
            WHERE disclosure_date BETWEEN ? AND ?
              AND transaction_date IS NOT NULL
              AND transaction_date > disclosure_date
              {source_clause}
            """,  # nosec B608 -- source_clause is a fixed internal fragment
            params,
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
            f"""
            SELECT *
            FROM transactions
            WHERE disclosure_date BETWEEN ? AND ?
              AND (transaction_date IS NULL OR transaction_date <= disclosure_date)
              {source_clause}
            ORDER BY disclosure_date DESC
            """,  # nosec B608 -- source_clause is a fixed internal fragment
            params,
        ).fetchdf()
        return _normalize_frame(result, deduplicate=True)

    def upsert(
        self, df: pd.DataFrame, *, source: str, _in_transaction: bool = False
    ) -> int:
        """Insert previously unseen canonical rows and return their count."""
        df = df.copy()
        for column in [
            "owner_code",
            "amount_raw",
            "amount_midpoint",
            "instrument_type",
            "strike_price",
            "expiry_date",
            "asset_description",
            "chamber",
            "source_record_id",
            "official_filing_date",
            "available_date",
            "notification_date",
            "amends_source_record_id",
            "raw_transaction_subtype",
            "ticker_origin",
            "raw_asset_class",
            "raw_asset_description",
            "ingestion_generation",
            "artifact_sha256",
        ]:
            if column not in df.columns:
                df[column] = None
        df["owner_code"] = df["owner_code"].fillna("").astype(str).replace("None", "")
        df["amount_raw"] = df["amount_raw"].fillna("").astype(str).replace("None", "")
        df["created_at"] = datetime.now()
        df["source"] = source
        df = _normalize_frame(df, deduplicate=True)

        if df.empty:
            return 0

        existing_column_types = {
            row[1]: row[2]
            for row in self.conn.execute("PRAGMA table_info('transactions')").fetchall()
        }
        write_columns = [
            column
            for column in _BASE_WRITE_COLUMNS + _PROVENANCE_COLUMNS
            if column in existing_column_types and column in df.columns
        ]
        df = df[write_columns].copy()
        staging_select = ", ".join(
            f"CAST({column} AS {existing_column_types[column]}) AS {column}"
            for column in write_columns
        )
        self.conn.execute(
            f"CREATE TEMP TABLE staging_transactions AS SELECT {staging_select} FROM df"  # nosec B608 -- identifiers/types come from the database schema
        )
        inserted_count = 0
        try:
            if not _in_transaction:
                self.conn.execute("BEGIN TRANSACTION")
            count_before = self.conn.execute(
                "SELECT COUNT(*) FROM transactions"
            ).fetchone()[0]
            self.conn.execute("""
                CREATE TEMP TABLE filtered_staging_transactions AS
                SELECT * FROM staging_transactions
            """)
            insert_columns_sql = ", ".join(write_columns)
            mutable_columns = [
                column
                for column in (
                    "disclosure_date",
                    "amount_midpoint",
                    "instrument_type",
                    "strike_price",
                    "expiry_date",
                    "created_at",
                )
                if column in write_columns
            ]
            provenance_columns = [
                column
                for column in ("source",) + _PROVENANCE_COLUMNS
                if column in write_columns
            ]
            updates = [f"{column} = s.{column}" for column in mutable_columns]
            updates.extend(
                f"{column} = CASE WHEN t.{column} IS NULL "
                f"THEN s.{column} ELSE t.{column} END"
                for column in provenance_columns
            )
            update_sql = ", ".join(updates)
            has_artifact_identity = set(_ARTIFACT_IDENTITY_COLUMNS).issubset(
                write_columns
            )
            if has_artifact_identity:
                identity_sql = " AND ".join(
                    f"s.{column} IS NOT NULL AND TRIM(s.{column}) <> '' "
                    f"AND t.{column} = s.{column}"
                    for column in _ARTIFACT_IDENTITY_COLUMNS
                )
                self.conn.execute(
                    f"""UPDATE transactions AS t SET {update_sql}
                        FROM filtered_staging_transactions AS s
                        WHERE {identity_sql}"""  # nosec B608 -- identifiers are fixed schema constants
                )
                self.conn.execute(
                    f"""INSERT INTO transactions ({insert_columns_sql})
                        SELECT {insert_columns_sql}
                        FROM filtered_staging_transactions s
                        WHERE NOT EXISTS (
                            SELECT 1 FROM transactions t WHERE {identity_sql}
                        )"""  # nosec B608 -- identifiers are fixed schema constants
                )
            else:
                self.conn.execute(
                    f"""INSERT INTO transactions ({insert_columns_sql})
                        SELECT {insert_columns_sql}
                        FROM filtered_staging_transactions"""  # nosec B608 -- identifiers are fixed schema constants
                )

            count_after = self.conn.execute(
                "SELECT COUNT(*) FROM transactions"
            ).fetchone()[0]
            inserted_count = count_after - count_before
            if not _in_transaction:
                self.conn.execute("COMMIT")
        except duckdb.ConstraintException as exc:
            if not _in_transaction:
                self.conn.execute("ROLLBACK")
            raise AmbiguousTransactionIdentityError(
                "Legacy economic unique key blocked a transaction without exact "
                "(source, chamber, source_record_id, source_row_id, "
                "ingestion_generation) identity"
            ) from exc
        except Exception:
            if not _in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        finally:
            self.conn.execute("DROP TABLE IF EXISTS filtered_staging_transactions")
            self.conn.execute("DROP TABLE IF EXISTS staging_transactions")
        return inserted_count

    def replace_source_refresh(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        chamber: str,
        ingestion_generation: str,
        _in_transaction: bool = False,
    ) -> int:
        """Replace the active source/chamber state without economic-key merging."""
        df = df[SOURCE_TRANSACTION_COLUMNS].copy()
        df["created_at"] = datetime.now()
        df["source"] = source
        self.conn.execute(
            "CREATE TEMP TABLE staging_source_transactions AS SELECT * FROM df"
        )
        succeeded = False
        try:
            if not _in_transaction:
                self.conn.execute("BEGIN TRANSACTION")
            self.conn.execute(
                "DELETE FROM transactions WHERE source = ? AND chamber = ?",
                [source, chamber],
            )
            self.conn.execute("""
                INSERT INTO transactions (
                    doc_id, member, member_key, chamber_member_key, ticker,
                    transaction_date, disclosure_date,
                    transaction_type, owner_code, amount_raw, amount_midpoint,
                    instrument_type, strike_price, expiry_date, created_at,
                    asset_description, source, chamber, source_record_id,
                    source_row_id, source_report_path, official_filing_date,
                    available_date,
                    notification_date, amends_source_record_id,
                    raw_transaction_subtype, ticker_origin, raw_ticker,
                    ticker_candidate, raw_asset_class,
                    raw_asset_description, raw_owner, ingestion_generation,
                    artifact_sha256
                )
                SELECT
                    doc_id, member, member_key, chamber_member_key, ticker,
                    transaction_date, disclosure_date,
                    transaction_type, owner_code, amount_raw, amount_midpoint,
                    instrument_type, strike_price, expiry_date, created_at,
                    asset_description, source, chamber, source_record_id,
                    source_row_id, source_report_path, official_filing_date,
                    available_date,
                    notification_date, amends_source_record_id,
                    raw_transaction_subtype, ticker_origin, raw_ticker,
                    ticker_candidate, raw_asset_class,
                    raw_asset_description, raw_owner, ingestion_generation,
                    artifact_sha256
                FROM staging_source_transactions
            """)
            if not _in_transaction:
                self.conn.execute("COMMIT")
            succeeded = True
        except Exception:
            if not _in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        finally:
            if succeeded or not _in_transaction:
                self.conn.execute("DROP TABLE IF EXISTS staging_source_transactions")
        return len(df)

    def get_for_doc(self, doc_id: str) -> pd.DataFrame:
        result = self.conn.execute(
            """
            SELECT *

            FROM transactions
            WHERE doc_id = ?
            ORDER BY id
            """,
            [doc_id],
        ).fetchdf()
        return _normalize_frame(result, deduplicate=True)

    def delete_for_doc(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM transactions WHERE doc_id = ?", [doc_id])

    def count_for_docs(self, doc_ids: list[str]) -> dict[str, int]:
        if not doc_ids:
            return {}

        placeholders = ", ".join("?" for _ in doc_ids)
        # Only generated placeholders are interpolated; doc_ids remain bound parameters.
        query = f"SELECT doc_id, COUNT(*) FROM transactions WHERE doc_id IN ({placeholders}) GROUP BY doc_id"  # nosec B608
        rows = self.conn.execute(query, doc_ids).fetchall()
        return {doc_id: count for doc_id, count in rows}

    def exists(
        self,
        year: int,
        *,
        source: str | None = None,
        sources: tuple[str, ...] | None = None,
    ) -> bool:
        source_clause, source_params = self._source_filter(
            source=source, sources=sources
        )
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM transactions
            WHERE EXTRACT(YEAR FROM disclosure_date) = ? {source_clause}
            """,  # nosec B608 -- source_clause is a fixed internal fragment
            [year, *source_params],
        ).fetchone()
        return row[0] > 0
