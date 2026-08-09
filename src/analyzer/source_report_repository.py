from __future__ import annotations

from enum import StrEnum
import re
from urllib.parse import urlsplit

import duckdb
import pandas as pd


def _is_official_paper_url(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    allowed_paths = (
        "/media/",
        "/search/view/paper/",
        "/search/view/paper-filing/",
    )
    return (
        parsed.scheme == "https"
        and parsed.hostname == "efdsearch.senate.gov"
        and parsed.username is None
        and parsed.password is None
        and port in (None, 443)
        and not parsed.query
        and not parsed.fragment
        and parsed.path.startswith(allowed_paths)
    )


class SourceReportOutcome(StrEnum):
    PARSED = "parsed"
    PAPER_ONLY = "paper_only"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


SOURCE_REPORT_COLUMNS = [
    "ingestion_generation",
    "source",
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
SOURCE_REPORT_INPUT_COLUMNS = [
    column for column in SOURCE_REPORT_COLUMNS if column != "source"
]


class SourceReportRepository:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def replace_generation(
        self,
        generation: str,
        source: str,
        chamber: str,
        reports_df: pd.DataFrame,
        *,
        _in_transaction: bool = False,
    ) -> None:
        """Atomically replace one generation/source/chamber report inventory."""
        self.validate_replacement(generation, source, chamber, reports_df)
        self._replace(
            generation=generation,
            source=source,
            chamber=chamber,
            reports_df=reports_df,
            replace_all_generations=False,
            in_transaction=_in_transaction,
        )

    def replace_source_refresh(
        self,
        generation: str,
        source: str,
        chamber: str,
        reports_df: pd.DataFrame,
        *,
        _in_transaction: bool = False,
    ) -> None:
        """Atomically replace the active inventory for one source/chamber."""
        self.validate_replacement(generation, source, chamber, reports_df)
        self._replace(
            generation=generation,
            source=source,
            chamber=chamber,
            reports_df=reports_df,
            replace_all_generations=True,
            in_transaction=_in_transaction,
        )

    def _replace(
        self,
        *,
        generation: str,
        source: str,
        chamber: str,
        reports_df: pd.DataFrame,
        replace_all_generations: bool,
        in_transaction: bool,
    ) -> None:
        reports_df = reports_df[SOURCE_REPORT_INPUT_COLUMNS].copy()
        reports_df["source"] = source
        reports_df = reports_df[SOURCE_REPORT_COLUMNS]

        if not in_transaction:
            self.conn.execute("BEGIN TRANSACTION")
        try:
            if replace_all_generations:
                self.conn.execute(
                    "DELETE FROM source_reports WHERE source = ? AND chamber = ?",
                    [source, chamber],
                )
            else:
                self.conn.execute(
                    "DELETE FROM source_reports "
                    "WHERE ingestion_generation = ? AND source = ? AND chamber = ?",
                    [generation, source, chamber],
                )
            self.conn.execute("""
                INSERT INTO source_reports (
                    ingestion_generation,
                    source,
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
                    source,
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
            if not in_transaction:
                self.conn.execute("COMMIT")
        except Exception:
            if not in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    def get(self, generation: str, source: str, chamber: str) -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT
                ingestion_generation,
                source,
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
            WHERE ingestion_generation = ? AND source = ? AND chamber = ?
            ORDER BY source_record_id
            """,
            [generation, source, chamber],
        ).fetchdf()

    def reconcile(self, generation: str, source: str, chamber: str) -> dict[str, int]:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS found,
                COUNT(*) FILTER (WHERE outcome = 'parsed') AS parsed,
                COUNT(*) FILTER (WHERE outcome = 'paper_only') AS paper_only,
                COUNT(*) FILTER (WHERE outcome = 'unavailable') AS unavailable,
                COUNT(*) FILTER (WHERE outcome = 'failed') AS failed
            FROM source_reports
            WHERE ingestion_generation = ? AND source = ? AND chamber = ?
            """,
            [generation, source, chamber],
        ).fetchone()
        names = ("found", "parsed", "paper_only", "unavailable", "failed")
        return dict(zip(names, (int(value) for value in row), strict=True))

    @staticmethod
    def validate_replacement(
        generation: str,
        source: str,
        chamber: str,
        reports_df: pd.DataFrame,
    ) -> None:
        if not isinstance(generation, str) or not generation.strip():
            raise ValueError("generation must be a non-empty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not isinstance(chamber, str) or not chamber.strip():
            raise ValueError("chamber must be a non-empty string")

        actual_columns = set(reports_df.columns)
        expected_columns = set(SOURCE_REPORT_INPUT_COLUMNS)
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
                    and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                    if value is not None
                    else False
                )
            )
            if invalid_hash.any():
                raise ValueError(
                    f"{column} must be a lowercase 64-character hexadecimal digest"
                )

        parsed = reports_df["outcome"].eq(SourceReportOutcome.PARSED.value)
        invalid_parsed_counts = parsed & (
            numeric_counts["accepted_row_count"].le(0)
            | numeric_counts["rejected_row_count"].ne(0)
            | numeric_counts["raw_row_count"].ne(numeric_counts["accepted_row_count"])
        )
        if invalid_parsed_counts.any():
            raise ValueError(
                "parsed reports require accepted>0, rejected=0, and raw=accepted"
            )
        parsed_hash_invalid = parsed & (
            reports_df["artifact_sha256"].isna()
            | reports_df["landing_sha256"].isna()
            | reports_df["artifact_sha256"].ne(reports_df["landing_sha256"])
        )
        if parsed_hash_invalid.any():
            raise ValueError(
                "parsed reports require artifact_sha256 equal to landing_sha256"
            )

        paper_only = reports_df["outcome"].eq(SourceReportOutcome.PAPER_ONLY.value)
        invalid_paper_counts = paper_only & numeric_counts[count_columns].ne(0).any(
            axis=1
        )
        if invalid_paper_counts.any():
            raise ValueError("paper_only reports require all row counts to equal zero")
        invalid_paper_hashes = paper_only & (
            reports_df["artifact_sha256"].isna()
            | reports_df["landing_sha256"].isna()
            | reports_df["paper_artifact_sha256"].isna()
            | reports_df["artifact_sha256"].ne(reports_df["landing_sha256"])
        )
        if invalid_paper_hashes.any():
            raise ValueError(
                "paper_only reports require artifact=landing and paper artifact hashes"
            )
        invalid_paper_urls = paper_only & ~reports_df["paper_artifact_url"].map(
            _is_official_paper_url
        )
        if invalid_paper_urls.any():
            raise ValueError("paper_only reports require an official Senate paper URL")

        nonpaper = ~paper_only
        forbidden_paper_fields = nonpaper & (
            reports_df["paper_artifact_url"].notna()
            | reports_df["paper_artifact_sha256"].notna()
        )
        if forbidden_paper_fields.any():
            raise ValueError("non-paper reports must not set paper artifact fields")

        classified = (
            parsed
            | paper_only
            | reports_df["outcome"].isin(
                [
                    SourceReportOutcome.UNAVAILABLE.value,
                    SourceReportOutcome.FAILED.value,
                ]
            )
        )
        other_invalid_equation = (
            classified
            & ~parsed
            & ~paper_only
            & (
                numeric_counts["raw_row_count"].ne(
                    numeric_counts["accepted_row_count"]
                    + numeric_counts["rejected_row_count"]
                )
            )
        )
        if other_invalid_equation.any():
            raise ValueError("source report row reconciliation failed")

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

        invalid_paths = reports_df["report_path"].map(
            lambda value: not isinstance(value, str) or not value.strip()
        )
        if invalid_paths.any():
            raise ValueError("report_path must be a non-empty string")
        invalid_members = reports_df["member"].map(
            lambda value: (
                not isinstance(value, str)
                or not value.strip()
                or value != value.strip()
            )
        )
        if invalid_members.any():
            raise ValueError("member must be a non-empty stripped string")
        filing_dates = pd.to_datetime(
            reports_df["official_filing_date"], errors="coerce"
        )
        if filing_dates.isna().any():
            raise ValueError("official_filing_date must be a valid date")
