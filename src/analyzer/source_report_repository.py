from __future__ import annotations

from enum import StrEnum
import re

import duckdb
import pandas as pd


class SourceReportOutcome(StrEnum):
    PARSED = "parsed"
    PAPER_ONLY = "paper_only"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


SOURCE_REPORT_COLUMNS = [
    "ingestion_generation",
    "chamber",
    "source_record_id",
    "report_path",
    "member",
    "official_filing_date",
    "outcome",
    "artifact_sha256",
    "landing_sha256",
    "paper_artifact_url",
    "paper_artifact_sha256",
    "error_message",
    "raw_row_count",
    "accepted_row_count",
    "rejected_row_count",
]


class SourceReportRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def replace_generation(
        self,
        generation: str,
        chamber: str,
        reports_df: pd.DataFrame,
        *,
        _in_transaction: bool = False,
    ) -> None:
        """Atomically replace one generation/chamber report inventory."""
        self.validate_replacement(generation, chamber, reports_df)
        reports_df = reports_df[SOURCE_REPORT_COLUMNS].copy()

        if not _in_transaction:
            self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.execute(
                "DELETE FROM source_reports "
                "WHERE ingestion_generation = ? AND chamber = ?",
                [generation, chamber],
            )
            self.conn.execute("""
                INSERT INTO source_reports (
                    ingestion_generation,
                    chamber,
                    source_record_id,
                    report_path,
                    member,
                    official_filing_date,
                    outcome,
                    artifact_sha256,
                    landing_sha256,
                    paper_artifact_url,
                    paper_artifact_sha256,
                    error_message,
                    raw_row_count,
                    accepted_row_count,
                    rejected_row_count
                )
                SELECT
                    ingestion_generation,
                    chamber,
                    source_record_id,
                    report_path,
                    member,
                    CAST(official_filing_date AS DATE),
                    outcome,
                    artifact_sha256,
                    landing_sha256,
                    paper_artifact_url,
                    paper_artifact_sha256,
                    error_message,
                    CAST(raw_row_count AS INTEGER),
                    CAST(accepted_row_count AS INTEGER),
                    CAST(rejected_row_count AS INTEGER)
                FROM reports_df
            """)
            if not _in_transaction:
                self.conn.execute("COMMIT")
        except Exception:
            if not _in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    def get(self, generation: str, chamber: str) -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT
                ingestion_generation,
                chamber,
                source_record_id,
                report_path,
                member,
                official_filing_date,
                outcome,
                artifact_sha256,
                landing_sha256,
                paper_artifact_url,
                paper_artifact_sha256,
                error_message,
                raw_row_count,
                accepted_row_count,
                rejected_row_count
            FROM source_reports
            WHERE ingestion_generation = ? AND chamber = ?
            ORDER BY source_record_id
            """,
            [generation, chamber],
        ).fetchdf()

    def reconcile(self, generation: str, chamber: str) -> dict[str, int]:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS found,
                COUNT(*) FILTER (WHERE outcome = 'parsed') AS parsed,
                COUNT(*) FILTER (WHERE outcome = 'paper_only') AS paper_only,
                COUNT(*) FILTER (WHERE outcome = 'unavailable') AS unavailable,
                COUNT(*) FILTER (WHERE outcome = 'failed') AS failed
            FROM source_reports
            WHERE ingestion_generation = ? AND chamber = ?
            """,
            [generation, chamber],
        ).fetchone()
        names = ("found", "parsed", "paper_only", "unavailable", "failed")
        return dict(zip(names, (int(value) for value in row), strict=True))

    @staticmethod
    def validate_replacement(
        generation: str,
        chamber: str,
        reports_df: pd.DataFrame,
    ) -> None:
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")
        if not isinstance(chamber, str) or not chamber.strip():
            raise ValueError("chamber must be a non-empty string")

        actual_columns = set(reports_df.columns)
        expected_columns = set(SOURCE_REPORT_COLUMNS)
        missing = expected_columns - actual_columns
        unexpected = actual_columns - expected_columns
        if missing or unexpected:
            raise ValueError(
                "source report columns must match the inventory schema; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

        required = reports_df[
            ["ingestion_generation", "chamber", "source_record_id", "outcome"]
        ]
        if required.isna().any().any():
            raise ValueError(
                "source report generation, chamber, record ID, and outcome are required"
            )
        if not reports_df["ingestion_generation"].eq(generation).all():
            raise ValueError("all source reports must match the replacement generation")
        if not reports_df["chamber"].eq(chamber).all():
            raise ValueError("all source reports must match the replacement chamber")
        if (
            reports_df["source_record_id"]
            .map(lambda value: not isinstance(value, str) or not value.strip())
            .any()
        ):
            raise ValueError("source_record_id must be a non-empty string")
        duplicates = reports_df["source_record_id"].duplicated(keep=False)
        if duplicates.any():
            duplicate_ids = sorted(
                reports_df.loc[duplicates, "source_record_id"].unique()
            )
            raise ValueError(
                "duplicate source_record_id values are not allowed: "
                + ", ".join(duplicate_ids[:10])
            )

        count_columns = [
            "raw_row_count",
            "accepted_row_count",
            "rejected_row_count",
        ]
        numeric_counts = reports_df[count_columns].apply(pd.to_numeric, errors="coerce")
        if (
            numeric_counts.isna().any().any()
            or (numeric_counts < 0).any().any()
            or (numeric_counts % 1 != 0).any().any()
        ):
            raise ValueError("source report row counts must be non-negative integers")
        invalid_equation = numeric_counts["raw_row_count"].ne(
            numeric_counts["accepted_row_count"] + numeric_counts["rejected_row_count"]
        )
        if invalid_equation.any():
            bad_ids = reports_df.loc[invalid_equation, "source_record_id"].tolist()
            raise ValueError(
                "source report row reconciliation failed for: "
                + ", ".join(bad_ids[:10])
            )

        hash_columns = [
            "artifact_sha256",
            "landing_sha256",
            "paper_artifact_sha256",
        ]
        for column in hash_columns:
            hashes = reports_df[column]
            invalid_hash = hashes.notna() & ~hashes.map(
                lambda value: (
                    isinstance(value, str)
                    and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
                    if value is not None
                    else False
                )
            )
            if invalid_hash.any():
                raise ValueError(f"{column} must be a 64-character hexadecimal digest")
        parsed = reports_df["outcome"].eq(SourceReportOutcome.PARSED.value)
        parsed_hash_missing = parsed & (
            reports_df["artifact_sha256"].isna() | reports_df["landing_sha256"].isna()
        )
        if parsed_hash_missing.any():
            raise ValueError(
                "parsed reports require artifact_sha256 and landing_sha256"
            )
        parsed_hash_mismatch = parsed & reports_df["artifact_sha256"].ne(
            reports_df["landing_sha256"]
        )
        if parsed_hash_mismatch.any():
            raise ValueError("parsed artifact_sha256 must equal landing_sha256")

        paper_only = reports_df["outcome"].eq(SourceReportOutcome.PAPER_ONLY.value)
        paper_hash_missing = paper_only & (
            reports_df["landing_sha256"].isna()
            | reports_df["paper_artifact_sha256"].isna()
        )
        if paper_hash_missing.any():
            raise ValueError(
                "paper_only reports require landing_sha256 and paper_artifact_sha256"
            )

        counts = reports_df["outcome"].value_counts().to_dict()
        found = len(reports_df)
        parsed = int(counts.get(SourceReportOutcome.PARSED.value, 0))
        paper_only = int(counts.get(SourceReportOutcome.PAPER_ONLY.value, 0))
        unavailable = int(counts.get(SourceReportOutcome.UNAVAILABLE.value, 0))
        failed = int(counts.get(SourceReportOutcome.FAILED.value, 0))
        classified = parsed + paper_only + unavailable + failed
        if found != classified:
            valid_outcomes = tuple(outcome.value for outcome in SourceReportOutcome)
            unknown = sorted(
                str(value)
                for value in reports_df.loc[
                    ~reports_df["outcome"].isin(valid_outcomes), "outcome"
                ].unique()
            )
            raise ValueError(
                "source report reconciliation failed: "
                f"found={found}, parsed={parsed}, paper_only={paper_only}, "
                f"unavailable={unavailable}, failed={failed}, unknown={unknown}"
            )
        if failed or unavailable:
            raise ValueError(
                "source report replacement requires unavailable=0 and failed=0; "
                f"found={found}, parsed={parsed}, paper_only={paper_only}, "
                f"unavailable={unavailable}, failed={failed}"
            )
