from __future__ import annotations

from enum import StrEnum

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
    "error_message",
]


class SourceReportRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def replace_generation(
        self,
        generation: str,
        chamber: str,
        reports_df: pd.DataFrame,
    ) -> None:
        """Atomically replace one generation/chamber report inventory."""
        self._validate_replacement(generation, chamber, reports_df)
        reports_df = reports_df[SOURCE_REPORT_COLUMNS].copy()

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
                    error_message
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
                    error_message
                FROM reports_df
            """)
            self.conn.execute("COMMIT")
        except Exception:
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
                error_message
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
    def _validate_replacement(
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
        if failed:
            raise ValueError(
                "source report replacement requires failed=0; "
                f"found={found}, parsed={parsed}, paper_only={paper_only}, "
                f"unavailable={unavailable}, failed={failed}"
            )
