#!/usr/bin/env python3
"""Read-only post-rebuild generation audit.

Verifies the invariants a full House/Senate rebuild must leave in a staged
DuckDB database. The audit never writes: the database is opened read-only.

Checks (each violation is reported exactly, and the process exits nonzero):

  C1  parse-count == persisted-rows   every provenance-bound ``pdf_parse_runs``
      row must reconcile with the transactions actually persisted under the
      same (doc, source, generation, artifact) identity; terminal-bad statuses
      (error/failed/rejected/zero_rows without rows) must not hide persisted
      rows (fail-closed).
  C2  source_row_id integrity         deterministic (house_pdf) and OCR
      (gemini_ocr) rows must carry a non-blank source_row_id, unique per
      (source, chamber, source_record_id, source_row_id, ingestion_generation),
      and the enforcing unique index must exist.
  C3  no transaction after disclosure no row may have transaction_date >
      disclosure_date.
  C4  duplicate policy                exact-identity replays are the only
      permitted dedup (no duplicate identity tuples, dedup removes nothing);
      economic duplicates survive ingestion and are flagged
      (economic_duplicate_candidate) exactly as the production read path
      computes them.
  C5  source_reports equation         per (ingestion_generation, source,
      chamber): found == parsed + no_txs + paper_only + unavailable + failed,
      and each report's row counts reconcile (parsed: raw==accepted>0,
      rejected=0; no_txs/paper_only: all zero; others:
      raw==accepted+rejected). Accepted rows must reconcile to transactions
      under the same source identity.
  C6  completeness                    a source_reports group is complete only
      when failed == 0 AND unavailable == 0.
  C7  pinned PDF canaries             20030977 -> 224, 20033737 -> 16,
      20033921 -> 15 persisted rows.
  C8  scans fail-closed / House       a House generation is activated
      (parse_status='complete') only with zero unresolved artifacts (every
      artifact has a terminal success/no_txs run bound to it and exactly one
      source report); canonical House/OCR rows must come from the latest
      semantically accepted complete generation of their archive year;
      persisted House/OCR rows must be artifact-bound.
  C9  chronology / date-domain        dates within [1900-01-01, today+1];
      canonical rows must carry transaction_date and disclosure_date; when both
      are present disclosure_date must be >= transaction_date; producer-
      guaranteed ordering against official_filing_date / notification_date is
      verified for the sources that guarantee it.

Usage:
    python3 scripts/audit_generation.py PATH/TO/STAGED.duckdb [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import duckdb

# Pinned parser canaries from tests/test_parsing.py: the production cascade
# emits exactly these row counts for these PDFs, so a faithful rebuild must
# persist exactly these per-document row counts.
PINNED_CANARY_COUNTS = {
    "20030977": 224,
    "20033737": 16,
    "20033921": 15,
}

MIN_DATE = date(1900, 1, 1)
MAX_DATE = date.today() + timedelta(days=1)

# Parse-run statuses that count as a resolved terminal outcome. Anything else
# (error, rejected, zero_rows, unknown) must fail closed.
TERMINAL_STATUSES = ("success", "no_txs")

# Sources produced by the deterministic parser cascade vs the Gemini OCR path.
DETERMINISTIC_PARSER_FAMILIES = ("v5-deterministic",)
OCR_PARSER_FAMILIES = ("v5-gemini-validated",)

_ECONOMIC_KEY = [
    "member",
    "ticker",
    "transaction_date",
    "transaction_type",
    "amount_raw",
    "owner_code",
    "asset_description",
    "raw_asset_description",
]
_IDENTITY_KEY = [
    "source",
    "chamber",
    "source_record_id",
    "source_row_id",
    "ingestion_generation",
]


@dataclass
class CheckResult:
    name: str
    description: str
    violations: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def _run_source_for_parser(parser_version: str | None) -> str | None:
    if not parser_version:
        return None
    lowered = str(parser_version).lower()
    if "gemini" in lowered:
        return "gemini_ocr"
    return "house_pdf"


def _is_official_paper_url(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == "efdsearch.senate.gov"
        and parsed.username is None
        and parsed.password is None
        and port in (None, 443)
        and not parsed.query
        and not parsed.fragment
        and parsed.path.startswith(
            ("/media/", "/search/view/paper/", "/search/view/paper-filing/")
        )
    )


def _table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = ? AND table_schema = 'main'",
            [table],
        ).fetchone()
        is not None
    )


def _view_exists(conn: duckdb.DuckDBPyConnection, view: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM information_schema.views "
            "WHERE table_name = ? AND table_schema = 'main'",
            [view],
        ).fetchone()
        is not None
    )


# ── C1 ────────────────────────────────────────────────────────────────────────
def check_parse_counts_match_persisted(
    conn: duckdb.DuckDBPyConnection,
) -> CheckResult:
    result = CheckResult(
        "parse_counts_match_persisted",
        "every provenance-bound parse run reconciles with persisted rows",
    )
    if not _table_exists(conn, "pdf_parse_runs") or not _table_exists(
        conn, "transactions"
    ):
        result.violations.append("pdf_parse_runs or transactions table missing")
        return result

    runs = conn.execute(
        """
        SELECT doc_id, parser_version, status, raw_row_count, transaction_count,
               artifact_sha256, ingestion_generation
        FROM pdf_parse_runs
        WHERE ingestion_generation IS NOT NULL
          AND artifact_sha256 IS NOT NULL
        ORDER BY doc_id, parser_version
        """
    ).fetchall()
    skipped = int(
        conn.execute(
            "SELECT COUNT(*) FROM pdf_parse_runs "
            "WHERE ingestion_generation IS NULL OR artifact_sha256 IS NULL"
        ).fetchone()[0]
    )
    if skipped:
        result.info.append(
            f"skipped {skipped} legacy parse run(s) without generation/artifact binding"
        )

    for doc_id, parser_version, status, raw_count, tx_count, sha, generation in runs:
        source = _run_source_for_parser(parser_version)
        if source is None:
            result.violations.append(
                f"{doc_id}: unrecognized parser_version {parser_version!r}"
            )
            continue
        persisted = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM transactions
                WHERE doc_id = ?
                  AND source = ?
                  AND ingestion_generation IS NOT DISTINCT FROM ?
                  AND artifact_sha256 IS NOT DISTINCT FROM ?
                """,
                [doc_id, source, generation, sha],
            ).fetchone()[0]
        )
        if status in TERMINAL_STATUSES:
            if tx_count != persisted:
                result.violations.append(
                    f"{doc_id} ({parser_version}, {generation}): "
                    f"run.transaction_count={tx_count} but persisted={persisted}"
                )
            if status == "success" and (
                raw_count is None
                or raw_count <= 0
                or tx_count is None
                or tx_count <= 0
            ):
                result.violations.append(
                    f"{doc_id} ({parser_version}, {generation}): "
                    "status='success' requires positive raw_row_count and "
                    "transaction_count"
                )
            if status == "no_txs" and (
                raw_count != 0 or tx_count != 0
            ):
                result.violations.append(
                    f"{doc_id} ({parser_version}, {generation}): "
                    f"status='no_txs' but raw_row_count={raw_count} "
                    f"transaction_count={tx_count}"
                )
        elif status in ("zero_rows", "error", "failed", "rejected", "invalidated"):
            if tx_count != 0:
                result.violations.append(
                    f"{doc_id} ({parser_version}, {generation}): "
                    f"status={status!r} but run.transaction_count={tx_count}"
                )
            if persisted != 0:
                result.violations.append(
                    f"{doc_id} ({parser_version}, {generation}): "
                    f"status={status!r} but {persisted} persisted row(s) remain "
                    "(scans must fail closed)"
                )
        else:
            result.violations.append(
                f"{doc_id} ({parser_version}, {generation}): "
                f"unknown parse status {status!r}"
            )
        if raw_count is not None and raw_count < 0:
            result.violations.append(
                f"{doc_id} ({parser_version}): negative raw_row_count={raw_count}"
            )
    return result


# ── C2 ────────────────────────────────────────────────────────────────────────
def check_source_row_ids(
    conn: duckdb.DuckDBPyConnection,
) -> CheckResult:
    result = CheckResult(
        "source_row_id_integrity",
        "deterministic/OCR rows have non-blank source_row_id, unique per identity",
    )
    if not _table_exists(conn, "transactions"):
        result.violations.append("transactions table missing")
        return result

    blank = conn.execute(
        """
        SELECT doc_id, source, source_row_id, ingestion_generation
        FROM transactions
        WHERE source IN ('house_pdf', 'gemini_ocr')
          AND (source_row_id IS NULL OR TRIM(source_row_id) = '')
        ORDER BY doc_id
        LIMIT 25
        """
    ).fetchall()
    for doc_id, source, row_id, generation in blank:
        result.violations.append(
            f"{doc_id} ({source}, gen={generation}): blank source_row_id "
            f"{row_id!r}"
        )
    if blank:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM transactions
                WHERE source IN ('house_pdf', 'gemini_ocr')
                  AND (source_row_id IS NULL OR TRIM(source_row_id) = '')
                """
            ).fetchone()[0]
        )
        if total > len(blank):
            result.violations.append(
                f"...and {total - len(blank)} more blank source_row_id row(s)"
            )

    duplicated = conn.execute(
        """
        SELECT source, chamber, source_record_id, source_row_id,
               ingestion_generation, COUNT(*) AS n
        FROM transactions
        WHERE source IN ('house_pdf', 'gemini_ocr')
          AND source_row_id IS NOT NULL AND TRIM(source_row_id) <> ''
        GROUP BY 1, 2, 3, 4, 5
        HAVING COUNT(*) > 1
        ORDER BY 3, 4
        LIMIT 25
        """
    ).fetchall()
    for source, chamber, record_id, row_id, generation, n in duplicated:
        result.violations.append(
            f"{record_id} ({source}/{chamber}, row={row_id!r}, "
            f"gen={generation}): {n} rows share one source_row_id"
        )
    if duplicated:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT source, chamber, source_record_id, source_row_id,
                           ingestion_generation
                    FROM transactions
                    WHERE source IN ('house_pdf', 'gemini_ocr')
                      AND source_row_id IS NOT NULL AND TRIM(source_row_id) <> ''
                    GROUP BY 1, 2, 3, 4, 5
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        if total > len(duplicated):
            result.violations.append(
                f"...and {total - len(duplicated)} more duplicated identity tuple(s)"
            )

    indexes = {
        str(row[0])
        for row in conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE table_name = 'transactions'"
        ).fetchall()
    }
    if "idx_tx_source_row_unique" not in indexes:
        result.violations.append(
            "unique index idx_tx_source_row_unique missing on transactions"
        )
    return result


# ── C3 ────────────────────────────────────────────────────────────────────────
def check_no_transaction_after_disclosure(
    conn: duckdb.DuckDBPyConnection,
) -> CheckResult:
    result = CheckResult(
        "no_transaction_after_disclosure",
        "no row has transaction_date > disclosure_date",
    )
    if not _table_exists(conn, "transactions"):
        result.violations.append("transactions table missing")
        return result
    rows = conn.execute(
        """
        SELECT doc_id, source, source_row_id, transaction_date, disclosure_date
        FROM transactions
        WHERE transaction_date IS NOT NULL AND disclosure_date IS NOT NULL
          AND transaction_date > disclosure_date
        ORDER BY doc_id
        LIMIT 25
        """
    ).fetchall()
    for doc_id, source, row_id, tx_date, disc_date in rows:
        result.violations.append(
            f"{doc_id} ({source}, row={row_id}): transaction_date={tx_date} "
            f"> disclosure_date={disc_date}"
        )
    if rows:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM transactions
                WHERE transaction_date IS NOT NULL AND disclosure_date IS NOT NULL
                  AND transaction_date > disclosure_date
                """
            ).fetchone()[0]
        )
        if total > len(rows):
            result.violations.append(f"...and {total - len(rows)} more row(s)")
    return result


# ── C4 ────────────────────────────────────────────────────────────────────────
def check_duplicate_policy(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    from analyzer.transaction_repository import _normalize_frame

    result = CheckResult(
        "duplicate_policy",
        "exact-identity replay is the only dedup; economic duplicates flagged",
    )
    if not _table_exists(conn, "transactions"):
        result.violations.append("transactions table missing")
        return result

    # (a) No exact-identity replay tuples may exist.
    duplicated = conn.execute(
        f"""
        SELECT {', '.join(_IDENTITY_KEY)}, COUNT(*) AS n
        FROM transactions
        WHERE {' AND '.join(f'{column} IS NOT NULL' for column in _IDENTITY_KEY)}
        GROUP BY {', '.join(_IDENTITY_KEY)}
        HAVING COUNT(*) > 1
        ORDER BY 3, 4
        LIMIT 25
        """,
    ).fetchall()
    for source, chamber, record_id, row_id, generation, n in duplicated:
        result.violations.append(
            f"exact-identity replay not deduplicated: {record_id} "
            f"({source}/{chamber}, row={row_id!r}, gen={generation}): {n} rows"
        )
    if duplicated:
        total = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT {', '.join(_IDENTITY_KEY)}
                    FROM transactions
                    WHERE {' AND '.join(f'{column} IS NOT NULL' for column in _IDENTITY_KEY)}
                    GROUP BY {', '.join(_IDENTITY_KEY)}
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        if total > len(duplicated):
            result.violations.append(
                f"...and {total - len(duplicated)} more duplicate identity tuple(s)"
            )

    # (b) Economic duplicates survive ingestion and are flagged exactly as the
    # production read path flags them, within each ingest batch.
    df = conn.execute(
        """
        SELECT doc_id, source, chamber, source_record_id, source_row_id,
               ingestion_generation, artifact_sha256, member, ticker,
               transaction_date, transaction_type, amount_raw, owner_code,
               asset_description, raw_asset_description
        FROM transactions
        ORDER BY source, ingestion_generation, doc_id, source_row_id
        """
    ).fetchdf()
    for column in (
        "member",
        "ticker",
        "transaction_type",
        "amount_raw",
        "owner_code",
        "asset_description",
        "raw_asset_description",
    ):
        df[column] = df[column].fillna("")
    df["_batch"] = (
        df["source"].fillna("")
        + "\x1f"
        + df["ingestion_generation"].fillna("").astype(str)
        + "\x1f"
        + df["artifact_sha256"].fillna("").astype(str)
    )
    for batch, batch_df in df.groupby("_batch", sort=False):
        identified = batch_df[_IDENTITY_KEY].fillna("").apply(
            lambda series: series.str.strip() if series.dtype == object else series
        )
        replay_scope = batch_df.loc[
            (identified != "").all(axis=1), batch_df.columns != "_batch"
        ].copy()
        if replay_scope.empty:
            continue
        try:
            normalized = _normalize_frame(replay_scope, deduplicate=True)
        except Exception as exc:  # noqa: BLE001 - audit must fail closed
            result.violations.append(
                f"batch {batch}: _normalize_frame failed: {type(exc).__name__}: {exc}"
            )
            continue
        if len(normalized) != len(replay_scope):
            result.violations.append(
                f"batch {batch}: replay dedup removed "
                f"{len(replay_scope) - len(normalized)} row(s) "
                "(exact-identity replay only; economic duplicates must survive)"
            )
        kept = replay_scope.loc[normalized.index]
        expected_flag = kept.duplicated(_ECONOMIC_KEY, keep=False)
        actual_flag = (
            normalized["economic_duplicate_candidate"].astype(bool).to_numpy()
        )
        mismatch = int((expected_flag.to_numpy() != actual_flag).sum())
        if mismatch:
            result.violations.append(
                f"batch {batch}: {mismatch} row(s) have a wrong "
                "economic_duplicate_candidate flag"
            )
    return result


# ── C5 ────────────────────────────────────────────────────────────────────────
def check_source_report_equation(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    result = CheckResult(
        "source_reports_equation",
        "found == parsed + no_txs + paper_only + unavailable + failed per group",
    )
    if not _table_exists(conn, "source_reports"):
        result.info.append("source_reports table missing; nothing to audit")
        return result
    transactions_available = _table_exists(conn, "transactions")
    if not transactions_available:
        result.violations.append(
            "transactions table missing; source-report row reconciliation is unavailable"
        )
    groups = conn.execute(
        """
        SELECT ingestion_generation, source, chamber,
               COUNT(*) AS found,
               COUNT(*) FILTER (WHERE outcome = 'parsed') AS parsed,
               COUNT(*) FILTER (WHERE outcome = 'no_txs') AS no_txs,
               COUNT(*) FILTER (WHERE outcome = 'paper_only') AS paper_only,
               COUNT(*) FILTER (WHERE outcome = 'unavailable') AS unavailable,
               COUNT(*) FILTER (WHERE outcome = 'failed') AS failed
        FROM source_reports
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).fetchall()
    if not groups:
        result.info.append("no source_reports groups present")
    for (
        generation,
        source,
        chamber,
        found,
        parsed,
        no_txs,
        paper,
        unavail,
        failed,
    ) in groups:
        classified = parsed + no_txs + paper + unavail + failed
        if found != classified:
            result.violations.append(
                f"source={source} chamber={chamber} gen={generation}: "
                f"found={found} != parsed={parsed}+no_txs={no_txs}+"
                f"paper_only={paper}+"
                f"unavailable={unavail}+failed={failed} ({classified})"
            )

    # Per-report row-count reconciliation (mirrors validate_replacement).
    bad = conn.execute(
        """
        SELECT ingestion_generation, source, chamber, source_record_id, outcome,
               raw_row_count, accepted_row_count, rejected_row_count
        FROM source_reports
        WHERE raw_row_count IS NULL
           OR accepted_row_count IS NULL
           OR rejected_row_count IS NULL
           OR outcome = 'parsed'
          AND NOT (raw_row_count = accepted_row_count
                   AND accepted_row_count > 0
                   AND rejected_row_count = 0)
           OR outcome = 'no_txs'
          AND NOT (raw_row_count = 0 AND accepted_row_count = 0
                  AND rejected_row_count = 0)
           OR outcome = 'paper_only'
          AND NOT (raw_row_count = 0 AND accepted_row_count = 0
                   AND rejected_row_count = 0)
           OR outcome IN ('unavailable', 'failed')
          AND NOT (raw_row_count = accepted_row_count + rejected_row_count)
        ORDER BY ingestion_generation, source_record_id
        LIMIT 25
        """
    ).fetchall()
    for generation, source, chamber, record_id, outcome, raw, accepted, rejected in bad:
        result.violations.append(
            f"source={source} chamber={chamber} gen={generation} "
            f"{record_id}: outcome={outcome} raw={raw} accepted={accepted} "
            f"rejected={rejected}"
        )

    invalid_no_txs_provenance = conn.execute(
        """
        SELECT ingestion_generation, source, chamber, source_record_id,
               artifact_sha256, landing_sha256
        FROM source_reports
        WHERE outcome = 'no_txs'
          AND (
              COALESCE(lower(chamber), '') <> 'house'
              OR
              artifact_sha256 IS NULL
              OR landing_sha256 IS NULL
              OR artifact_sha256 IS DISTINCT FROM landing_sha256
              OR NOT regexp_full_match(artifact_sha256, '[0-9a-f]{64}')
              OR NOT regexp_full_match(landing_sha256, '[0-9a-f]{64}')
          )
        ORDER BY ingestion_generation, source, source_record_id
        LIMIT 25
        """
    ).fetchall()
    for generation, source, chamber, record_id, artifact, landing in (
        invalid_no_txs_provenance
    ):
        result.violations.append(
            f"source={source} chamber={chamber} gen={generation} "
            f"{record_id}: no_txs report has invalid House provenance "
            f"(artifact={artifact!r}, landing={landing!r})"
        )

    invalid_parsed_provenance = conn.execute(
        """
        SELECT ingestion_generation, source, chamber, source_record_id,
               artifact_sha256, landing_sha256
        FROM source_reports
        WHERE outcome = 'parsed'
          AND (
              artifact_sha256 IS NULL
              OR landing_sha256 IS NULL
              OR artifact_sha256 IS DISTINCT FROM landing_sha256
              OR NOT regexp_full_match(artifact_sha256, '[0-9a-f]{64}')
              OR NOT regexp_full_match(landing_sha256, '[0-9a-f]{64}')
          )
        ORDER BY ingestion_generation, source, source_record_id
        LIMIT 25
        """
    ).fetchall()
    for generation, source, chamber, record_id, artifact, landing in (
        invalid_parsed_provenance
    ):
        result.violations.append(
            f"source={source} chamber={chamber} gen={generation} "
            f"{record_id}: parsed report has invalid artifact provenance "
            f"(artifact={artifact!r}, landing={landing!r})"
        )

    invalid_nonpaper_fields = conn.execute(
        """
        SELECT ingestion_generation, source, chamber, source_record_id, outcome
        FROM source_reports
        WHERE outcome <> 'paper_only'
          AND (paper_artifact_url IS NOT NULL OR paper_artifact_sha256 IS NOT NULL)
        ORDER BY ingestion_generation, source, source_record_id
        LIMIT 25
        """
    ).fetchall()
    for generation, source, chamber, record_id, outcome in invalid_nonpaper_fields:
        result.violations.append(
            f"source={source} chamber={chamber} gen={generation} "
            f"{record_id}: outcome={outcome} sets paper artifact fields"
        )

    paper_reports = conn.execute(
        """
        SELECT ingestion_generation, source, chamber, source_record_id,
               artifact_sha256, landing_sha256, paper_artifact_url,
               paper_artifact_sha256
        FROM source_reports
        WHERE outcome = 'paper_only'
        ORDER BY ingestion_generation, source, source_record_id
        LIMIT 25
        """
    ).fetchall()
    for (
        generation,
        source,
        chamber,
        record_id,
        artifact,
        landing,
        paper_url,
        paper_sha,
    ) in paper_reports:
        invalid_scope = (
            not isinstance(chamber, str) or chamber.strip().lower() != "senate"
        )
        invalid_hashes = (
            not isinstance(artifact, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact) is None
            or not isinstance(landing, str)
            or re.fullmatch(r"[0-9a-f]{64}", landing) is None
            or artifact != landing
            or not isinstance(paper_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", paper_sha) is None
        )
        if invalid_scope or invalid_hashes or not _is_official_paper_url(paper_url):
            result.violations.append(
                f"source={source} chamber={chamber} gen={generation} "
                f"{record_id}: paper_only report has invalid Senate paper "
                f"provenance (artifact={artifact!r}, landing={landing!r}, "
                f"paper_sha={paper_sha!r}, paper_url={paper_url!r})"
            )

    if not transactions_available:
        return result

    # A report's accepted count must bind to rows from the same source,
    # chamber, generation, and source record.  A House OCR parse is therefore
    # not allowed to reconcile against a report accidentally stored as
    # ``house_pdf`` (or vice versa).
    report_row_mismatches = conn.execute(
        """
        SELECT r.ingestion_generation, r.source, r.chamber,
               r.source_record_id, r.outcome, r.accepted_row_count,
               COUNT(t.id) AS actual
        FROM source_reports r
        LEFT JOIN transactions t
          ON t.ingestion_generation = r.ingestion_generation
         AND t.source = r.source
         AND lower(t.chamber) = lower(r.chamber)
         AND t.source_record_id = r.source_record_id
         AND t.artifact_sha256 IS NOT DISTINCT FROM r.artifact_sha256
        GROUP BY 1, 2, 3, 4, 5, 6
        HAVING r.accepted_row_count != COUNT(t.id)
        ORDER BY 1, 2, 3, 4
        LIMIT 25
        """
    ).fetchall()
    for generation, source, chamber, record_id, outcome, accepted, actual in (
        report_row_mismatches
    ):
        result.violations.append(
            f"source={source} chamber={chamber} gen={generation} "
            f"{record_id}: outcome={outcome} accepted={accepted} "
            f"persisted={actual}"
        )
    if report_row_mismatches:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT r.ingestion_generation, r.source, r.chamber,
                           r.source_record_id
                    FROM source_reports r
                    LEFT JOIN transactions t
                      ON t.ingestion_generation = r.ingestion_generation
                     AND t.source = r.source
                     AND lower(t.chamber) = lower(r.chamber)
                     AND t.source_record_id = r.source_record_id
                     AND t.artifact_sha256 IS NOT DISTINCT FROM r.artifact_sha256
                    GROUP BY 1, 2, 3, 4, r.accepted_row_count
                    HAVING r.accepted_row_count != COUNT(t.id)
                )
                """
            ).fetchone()[0]
        )
        if total > len(report_row_mismatches):
            result.violations.append(
                f"...and {total - len(report_row_mismatches)} more "
                "source-report row mismatch(es)"
            )

    # Conversely, every canonical House/OCR transaction must have a matching
    # report row.  This catches an inventory accidentally emitted under the
    # wrong source even when that report advertises accepted_row_count=0.
    orphan_transactions = conn.execute(
        """
        SELECT t.source, t.chamber, t.ingestion_generation,
               t.source_record_id, COUNT(*) AS rows
        FROM transactions t
        LEFT JOIN source_reports r
          ON r.ingestion_generation = t.ingestion_generation
         AND r.source = t.source
         AND lower(r.chamber) = lower(t.chamber)
         AND r.source_record_id = t.source_record_id
         AND r.artifact_sha256 IS NOT DISTINCT FROM t.artifact_sha256
        WHERE t.source IN ('house_pdf', 'gemini_ocr')
          AND r.source_record_id IS NULL
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 2, 3, 4
        LIMIT 25
        """
    ).fetchall()
    for source, chamber, generation, record_id, rows in orphan_transactions:
        result.violations.append(
            f"source={source} chamber={chamber} gen={generation} "
            f"{record_id}: {rows} persisted transaction row(s) have no "
            "matching source report"
        )
    if orphan_transactions:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT t.source, t.chamber, t.ingestion_generation,
                           t.source_record_id
                    FROM transactions t
                    LEFT JOIN source_reports r
                      ON r.ingestion_generation = t.ingestion_generation
                     AND r.source = t.source
                     AND lower(r.chamber) = lower(t.chamber)
                     AND r.source_record_id = t.source_record_id
                     AND r.artifact_sha256 IS NOT DISTINCT FROM t.artifact_sha256
                    WHERE t.source IN ('house_pdf', 'gemini_ocr')
                      AND r.source_record_id IS NULL
                    GROUP BY 1, 2, 3, 4
                )
                """
            ).fetchone()[0]
        )
        if total > len(orphan_transactions):
            result.violations.append(
                f"...and {total - len(orphan_transactions)} more "
                "transaction source-report orphan group(s)"
            )
    return result


# ── C6 ────────────────────────────────────────────────────────────────────────
def check_completeness(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    result = CheckResult(
        "completeness",
        "completeness requires failed == 0 AND unavailable == 0 per group",
    )
    if not _table_exists(conn, "source_reports"):
        result.info.append("source_reports table missing; nothing to audit")
        return result
    incomplete = conn.execute(
        """
        SELECT ingestion_generation, source, chamber,
               COUNT(*) FILTER (WHERE outcome = 'failed') AS failed,
               COUNT(*) FILTER (WHERE outcome = 'unavailable') AS unavailable
        FROM source_reports
        GROUP BY 1, 2, 3
        HAVING failed > 0 OR unavailable > 0
        ORDER BY 1, 2, 3
        """
    ).fetchall()
    for generation, source, chamber, failed, unavailable in incomplete:
        result.violations.append(
            f"source={source} chamber={chamber} gen={generation} incomplete: "
            f"failed={failed}, unavailable={unavailable}"
        )
    return result


# ── C7 ────────────────────────────────────────────────────────────────────────
def check_canary_counts(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    result = CheckResult(
        "pinned_pdf_canaries",
        "pinned canary docs persist exactly their pinned row counts",
    )
    if not _table_exists(conn, "transactions"):
        result.violations.append("transactions table missing")
        return result
    if not _table_exists(conn, "pdf_parse_runs"):
        result.info.append("pdf_parse_runs table missing; row-count check only")
    for doc_id, pinned in sorted(PINNED_CANARY_COUNTS.items()):
        persisted = int(
            conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE doc_id = ?",
                [doc_id],
            ).fetchone()[0]
        )
        if persisted != pinned:
            result.violations.append(
                f"canary {doc_id}: persisted {persisted} rows, pinned {pinned}"
            )
            continue
        run = conn.execute(
            """
            SELECT parser_version, status, raw_row_count
            FROM pdf_parse_runs
            WHERE doc_id = ? AND status = 'success'
            ORDER BY parsed_at DESC, parser_version
            LIMIT 1
            """,
            [doc_id],
        ).fetchone()
        if run is not None and int(run[2]) != pinned:
            result.violations.append(
                f"canary {doc_id}: latest success run raw_row_count={run[2]} "
                f"({run[0]}, {run[1]}), pinned {pinned}"
            )
    return result


# ── C8 ────────────────────────────────────────────────────────────────────────
def _get_unresolved_house_doc_ids(
    conn: duckdb.DuckDBPyConnection,
    archive_year: int,
    generation_id: str,
) -> list[str]:
    """Apply the same authoritative House completeness contract as Database."""
    rows = conn.execute(
        """
        SELECT scope.doc_id
        FROM house_generation_metadata scope
        LEFT JOIN house_pdf_artifacts artifact
          ON artifact.archive_year = scope.archive_year
         AND artifact.generation_id = scope.generation_id
         AND artifact.doc_id = scope.doc_id
        WHERE scope.archive_year = ? AND scope.generation_id = ?
          AND scope.filing_type = 'P'
          AND (
            artifact.doc_id IS NULL
            OR NOT EXISTS (
                SELECT 1
                FROM pdf_parse_runs parse_run
                WHERE parse_run.doc_id = artifact.doc_id
                  AND parse_run.artifact_sha256 = artifact.artifact_sha256
                  AND parse_run.ingestion_generation = artifact.generation_id
                  AND parse_run.status IN ('success', 'no_txs')
                  AND COALESCE(parse_run.transaction_count, 0) = (
                      SELECT COUNT(*)
                      FROM transactions tx_count
                      WHERE tx_count.doc_id = artifact.doc_id
                        AND tx_count.artifact_sha256 = artifact.artifact_sha256
                        AND tx_count.ingestion_generation = artifact.generation_id
                        AND tx_count.source = CASE
                            WHEN LOWER(COALESCE(parse_run.parser_version, '')) LIKE '%gemini%'
                            THEN 'gemini_ocr'
                            ELSE 'house_pdf'
                        END
                  )
                  AND (
                      parse_run.status = 'no_txs'
                      OR (
                          COALESCE(parse_run.transaction_count, 0) > 0
                          AND NOT EXISTS (
                              SELECT 1
                              FROM transactions tx_invalid
                              WHERE tx_invalid.doc_id = artifact.doc_id
                                AND tx_invalid.artifact_sha256 = artifact.artifact_sha256
                                AND tx_invalid.ingestion_generation = artifact.generation_id
                                AND tx_invalid.source = CASE
                                    WHEN LOWER(COALESCE(parse_run.parser_version, '')) LIKE '%gemini%'
                                    THEN 'gemini_ocr'
                                    ELSE 'house_pdf'
                                END
                                AND (
                                    tx_invalid.transaction_date IS NULL
                                    OR tx_invalid.disclosure_date IS NULL
                                    OR tx_invalid.transaction_date > tx_invalid.disclosure_date
                                    OR (
                                        tx_invalid.notification_date IS NOT NULL
                                        AND tx_invalid.notification_date < tx_invalid.transaction_date
                                    )
                                    OR tx_invalid.chamber IS NULL
                                    OR TRIM(tx_invalid.chamber) = ''
                                    OR tx_invalid.source_record_id IS NULL
                                    OR TRIM(tx_invalid.source_record_id) = ''
                                    OR tx_invalid.source_row_id IS NULL
                                    OR TRIM(tx_invalid.source_row_id) = ''
                                    OR tx_invalid.official_filing_date IS NULL
                                )
                          )
                      )
                  )
            )
          )
        ORDER BY scope.doc_id
        """,
        [archive_year, generation_id],
    ).fetchall()
    return [str(row[0]) for row in rows]


def check_house_generation_activation(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    result = CheckResult(
        "house_generation_activation",
        "complete generations have zero unresolved artifacts; canonical rows "
        "come from the latest semantically accepted complete generation and "
        "every artifact has one "
        "matching House source report",
    )
    if not _table_exists(conn, "house_archive_generations"):
        result.info.append("house_archive_generations table missing; nothing to audit")
        return result
    if not _table_exists(conn, "house_pdf_artifacts"):
        result.violations.append("house_pdf_artifacts table missing")
    if not _table_exists(conn, "house_generation_metadata"):
        result.violations.append("house_generation_metadata table missing")
    if not _table_exists(conn, "pdf_parse_runs"):
        result.violations.append("pdf_parse_runs table missing")
    if not _table_exists(conn, "source_reports"):
        result.violations.append("source_reports table missing")
    if result.violations:
        return result

    generations = conn.execute(
        "SELECT archive_year, generation_id, parse_status, promoted_at "
        "FROM house_archive_generations "
        "ORDER BY archive_year, promoted_at ASC NULLS FIRST, generation_id"
    ).fetchall()
    latest_accepted: dict[int, str] = {}
    for archive_year, generation_id, parse_status, _promoted_at in generations:
        semantic_unresolved = _get_unresolved_house_doc_ids(
            conn, archive_year, generation_id
        )
        artifact_unresolved = conn.execute(
            """
            SELECT a.doc_id
            FROM house_pdf_artifacts a
            LEFT JOIN pdf_parse_runs p
              ON p.doc_id = a.doc_id
             AND p.artifact_sha256 = a.artifact_sha256
             AND p.ingestion_generation = a.generation_id
             AND p.status IN ('success', 'no_txs')
            WHERE a.archive_year = ? AND a.generation_id = ?
            GROUP BY a.doc_id
            HAVING COUNT(p.doc_id) = 0
            ORDER BY a.doc_id
            """,
            [archive_year, generation_id],
        ).fetchall()
        unresolved = sorted(
            {
                *semantic_unresolved,
                *(str(row[0]) for row in artifact_unresolved),
            }
        )
        if parse_status == "complete" and unresolved:
            doc_ids = ", ".join(unresolved[:10])
            result.violations.append(
                f"archive {archive_year} generation {generation_id} marked "
                f"'complete' but has {len(unresolved)} unresolved artifact(s): "
                f"{doc_ids}"
            )
        elif unresolved and parse_status != "complete":
            result.info.append(
                f"archive {archive_year} generation {generation_id} correctly "
                f"incomplete ({len(unresolved)} unresolved artifact(s))"
            )
        elif parse_status == "complete" and not semantic_unresolved:
            # The view uses this same semantic predicate before choosing the
            # latest generation, so an invalid newer marker cannot shadow an
            # earlier accepted generation.
            latest_accepted[archive_year] = str(generation_id)

        # A complete House generation must have one source-report row for
        # every authoritative artifact, including terminal no_txs documents.
        # Match the artifact digest as well as the document ID so a report
        # cannot silently bind to a different downloaded document.
        if parse_status == "complete":
            artifact_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM house_pdf_artifacts
                    WHERE archive_year = ? AND generation_id = ?
                    """,
                    [archive_year, generation_id],
                ).fetchone()[0]
            )
            report_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM source_reports
                    WHERE ingestion_generation = ?
                      AND source IN ('house_pdf', 'gemini_ocr')
                      AND lower(chamber) = 'house'
                    """,
                    [generation_id],
                ).fetchone()[0]
            )
            if report_count != artifact_count:
                result.violations.append(
                    f"archive {archive_year} generation {generation_id}: "
                    f"source_reports={report_count} but authoritative "
                    f"artifacts={artifact_count}"
                )

            missing_reports = conn.execute(
                """
                SELECT a.doc_id
                FROM house_pdf_artifacts a
                LEFT JOIN source_reports r
                  ON r.ingestion_generation = a.generation_id
                 AND r.source IN ('house_pdf', 'gemini_ocr')
                 AND lower(r.chamber) = 'house'
                 AND r.source_record_id = a.doc_id
                 AND r.artifact_sha256 IS NOT DISTINCT FROM a.artifact_sha256
                WHERE a.archive_year = ? AND a.generation_id = ?
                  AND r.source_record_id IS NULL
                ORDER BY a.doc_id
                LIMIT 25
                """,
                [archive_year, generation_id],
            ).fetchall()
            if missing_reports:
                preview = ", ".join(str(row[0]) for row in missing_reports)
                result.violations.append(
                    f"archive {archive_year} generation {generation_id}: "
                    f"{len(missing_reports)} authoritative artifact(s) lack "
                    f"a matching House source report: {preview}"
                )

            extra_reports = conn.execute(
                """
                SELECT r.source, r.source_record_id, r.artifact_sha256
                FROM source_reports r
                LEFT JOIN house_pdf_artifacts a
                  ON a.generation_id = r.ingestion_generation
                 AND a.doc_id = r.source_record_id
                 AND a.artifact_sha256 IS NOT DISTINCT FROM r.artifact_sha256
                WHERE r.ingestion_generation = ?
                  AND r.source IN ('house_pdf', 'gemini_ocr')
                  AND lower(r.chamber) = 'house'
                  AND a.doc_id IS NULL
                ORDER BY r.source, r.source_record_id
                LIMIT 25
                """,
                [generation_id],
            ).fetchall()
            if extra_reports:
                preview = ", ".join(
                    f"{source}:{record_id}" for source, record_id, _sha in extra_reports
                )
                result.violations.append(
                    f"archive {archive_year} generation {generation_id}: "
                    f"{len(extra_reports)} House source report(s) lack a "
                    f"matching authoritative artifact: {preview}"
                )

            duplicate_reports = conn.execute(
                """
                SELECT source_record_id, COUNT(*) AS n
                FROM source_reports
                WHERE ingestion_generation = ?
                  AND source IN ('house_pdf', 'gemini_ocr')
                  AND lower(chamber) = 'house'
                GROUP BY source_record_id
                HAVING COUNT(*) > 1
                ORDER BY source_record_id
                LIMIT 25
                """,
                [generation_id],
            ).fetchall()
            if duplicate_reports:
                preview = ", ".join(
                    f"{record_id}({count})" for record_id, count in duplicate_reports
                )
                result.violations.append(
                    f"archive {archive_year} generation {generation_id}: "
                    f"duplicate House source reports: {preview}"
                )

            # Every authoritative artifact must have exactly one terminal
            # parser run and one report with the same source family and
            # semantic outcome. Count/doc/artifact coverage alone cannot
            # detect a zero-row report moved between House parser sources.
            artifact_rows = conn.execute(
                """
                SELECT doc_id, artifact_sha256
                FROM house_pdf_artifacts
                WHERE archive_year = ? AND generation_id = ?
                ORDER BY doc_id
                """,
                [archive_year, generation_id],
            ).fetchall()
            terminal_runs = conn.execute(
                """
                SELECT doc_id, parser_version, status, artifact_sha256
                FROM pdf_parse_runs
                WHERE ingestion_generation = ?
                  AND status IN ('success', 'no_txs')
                """,
                [generation_id],
            ).fetchall()
            reports = conn.execute(
                """
                SELECT source, source_record_id, outcome, artifact_sha256
                FROM source_reports
                WHERE ingestion_generation = ?
                  AND source IN ('house_pdf', 'gemini_ocr')
                  AND lower(chamber) = 'house'
                """,
                [generation_id],
            ).fetchall()
            runs_by_artifact: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for doc_id, parser_version, status, artifact_sha in terminal_runs:
                if doc_id is None or artifact_sha is None:
                    continue
                source = _run_source_for_parser(parser_version)
                if source is None:
                    continue
                outcome = "parsed" if status == "success" else "no_txs"
                runs_by_artifact.setdefault(
                    (str(doc_id), str(artifact_sha)), []
                ).append((source, outcome))
            reports_by_artifact: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for source, record_id, outcome, artifact_sha in reports:
                if record_id is None or artifact_sha is None:
                    continue
                reports_by_artifact.setdefault(
                    (str(record_id), str(artifact_sha)), []
                ).append((str(source), str(outcome)))
            for doc_id, artifact_sha in artifact_rows:
                key = (str(doc_id), str(artifact_sha))
                run_bindings = runs_by_artifact.get(key, [])
                report_bindings = reports_by_artifact.get(key, [])
                if len(run_bindings) != 1:
                    result.violations.append(
                        f"archive {archive_year} generation {generation_id}: "
                        f"artifact {doc_id} has {len(run_bindings)} matching "
                        "terminal parser run(s)"
                    )
                if len(report_bindings) != 1:
                    result.violations.append(
                        f"archive {archive_year} generation {generation_id}: "
                        f"artifact {doc_id} has {len(report_bindings)} matching "
                        "House source report(s)"
                    )
                if len(run_bindings) == 1 and len(report_bindings) == 1:
                    if run_bindings[0] != report_bindings[0]:
                        result.violations.append(
                            f"archive {archive_year} generation {generation_id}: "
                            f"artifact {doc_id} parser/report binding mismatch: "
                            f"run={run_bindings[0]} report={report_bindings[0]}"
                        )


    # House/OCR rows persisted under a generation must be artifact-bound to a
    # terminal run of the matching parser family.
    rows = conn.execute(
        """
        SELECT t.doc_id, t.source, t.ingestion_generation, t.artifact_sha256
        FROM transactions t
        WHERE t.source IN ('house_pdf', 'gemini_ocr')
          AND t.ingestion_generation IS NOT NULL
          AND t.artifact_sha256 IS NOT NULL
        ORDER BY t.doc_id
        LIMIT 50
        """
    ).fetchall()
    unbound = []
    for doc_id, source, generation, sha in rows:
        family = (
            "NOT LIKE '%gemini%'"
            if source == "house_pdf"
            else "LIKE '%gemini%'"
        )
        bound = conn.execute(
            f"""
            SELECT COUNT(*) FROM pdf_parse_runs
            WHERE doc_id = ? AND ingestion_generation = ?
              AND artifact_sha256 = ?
              AND parser_version {family}
              AND status IN ('success', 'no_txs')
            """,
            [doc_id, generation, sha],
        ).fetchone()[0]
        if not bound:
            unbound.append((doc_id, source, generation))
    if unbound:
        preview = ", ".join(
            f"{doc}({source},{generation})" for doc, source, generation in unbound[:10]
        )
        result.violations.append(
            f"{len(unbound)} persisted House/OCR row(s) are not artifact-bound "
            f"to a terminal run: {preview}"
        )

    # Canonical House/OCR rows must come from the latest semantically accepted
    # complete generation.
    if not _view_exists(conn, "canonical_transactions"):
        result.violations.append("canonical_transactions view missing")
        return result
    if not _table_exists(conn, "metadata"):
        result.info.append("metadata table missing; canonical-source check skipped")
        return result
    canonical_house = conn.execute(
        """
        SELECT t.doc_id, t.source, t.ingestion_generation, m.archive_year
        FROM canonical_transactions t
        JOIN metadata m ON m.doc_id = t.doc_id
        WHERE t.source IN ('house_pdf', 'gemini_ocr')
        ORDER BY t.doc_id
        """
    ).fetchall()
    for doc_id, source, generation, archive_year in canonical_house:
        latest_complete = latest_accepted.get(archive_year)
        if latest_complete is None:
            continue
        if generation != latest_complete:
            result.violations.append(
                f"{doc_id} ({source}): canonical rows bound to generation "
                f"{generation!r}, but archive {archive_year} latest accepted "
                f"complete generation is {latest_complete!r}"
            )
    return result


# ── C9 ────────────────────────────────────────────────────────────────────────
def check_date_domain(conn: duckdb.DuckDBPyConnection) -> CheckResult:
    result = CheckResult(
        "date_domain",
        "dates parse and stay within [1900-01-01, today+1]; canonical rows "
        "carry transaction_date and disclosure_date",
    )
    if not _table_exists(conn, "transactions"):
        result.violations.append("transactions table missing")
        return result

    date_columns = [
        "transaction_date",
        "disclosure_date",
        "official_filing_date",
        "available_date",
        "notification_date",
    ]
    checks = [
        f"{column} IS NOT NULL AND {column} < CAST(? AS DATE)" for column in date_columns
    ] + [
        f"{column} IS NOT NULL AND {column} > CAST(? AS DATE)" for column in date_columns
    ]
    sql = (
        "SELECT doc_id, source, source_row_id, transaction_date, disclosure_date, "
        "official_filing_date, available_date, notification_date "
        f"FROM transactions WHERE {' OR '.join(checks)} "
        "ORDER BY doc_id LIMIT 25"
    )
    out_of_domain = conn.execute(
        sql, [str(MIN_DATE)] * len(date_columns) + [str(MAX_DATE)] * len(date_columns)
    ).fetchall()
    for row in out_of_domain:
        doc_id, source, row_id = row[0], row[1], row[2]
        result.violations.append(
            f"{doc_id} ({source}, row={row_id}): date out of domain "
            f"[{MIN_DATE}, {MAX_DATE}]: "
            + ", ".join(
                f"{column}={value}"
                for column, value in zip(date_columns, row[3:], strict=True)
                if value is not None
            )
        )
    if out_of_domain:
        total = int(
            conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE "
                + " OR ".join(checks),
                [str(MIN_DATE)] * len(date_columns)
                + [str(MAX_DATE)] * len(date_columns),
            ).fetchone()[0]
        )
        if total > len(out_of_domain):
            result.violations.append(f"...and {total - len(out_of_domain)} more row(s)")

    if not _view_exists(conn, "canonical_transactions"):
        result.info.append("canonical_transactions view missing; null-date check skipped")
        return result
    null_dates = conn.execute(
        """
        SELECT doc_id, source, source_row_id, transaction_date, disclosure_date
        FROM canonical_transactions
        WHERE transaction_date IS NULL OR disclosure_date IS NULL
        ORDER BY doc_id
        LIMIT 25
        """
    ).fetchall()
    for doc_id, source, row_id, tx_date, disc_date in null_dates:
        result.violations.append(
            f"{doc_id} ({source}, row={row_id}): canonical row missing a date "
            f"(transaction_date={tx_date}, disclosure_date={disc_date})"
        )
    if null_dates:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM canonical_transactions
                WHERE transaction_date IS NULL OR disclosure_date IS NULL
                """
            ).fetchone()[0]
        )
        if total > len(null_dates):
            result.violations.append(f"...and {total - len(null_dates)} more row(s)")

    # Producer-guaranteed ordering against filing/notification dates.
    ordering = conn.execute(
        """
        SELECT doc_id, source, source_row_id, transaction_date, disclosure_date,
               official_filing_date, notification_date
        FROM transactions
        WHERE transaction_date IS NOT NULL AND disclosure_date IS NOT NULL
          AND (
            -- Senate/legacy/capitol/house guarantee disclosure <= official filing
            (source IS DISTINCT FROM 'gemini_ocr'
             AND official_filing_date IS NOT NULL
             AND disclosure_date > official_filing_date)
            OR
            -- Senate guarantees the notification chain; every source keeps
            -- notification chronologically after the transaction
            (notification_date IS NOT NULL
             AND notification_date < transaction_date)
            OR
            (chamber ILIKE 'senate'
             AND notification_date IS NOT NULL
             AND official_filing_date IS NOT NULL
             AND notification_date > official_filing_date)
          )
        ORDER BY doc_id
        LIMIT 25
        """,
    ).fetchall()
    for doc_id, source, row_id, tx_date, disc_date, filing_date, notif_date in ordering:
        result.violations.append(
            f"{doc_id} ({source}, row={row_id}): chronology broken "
            f"(transaction_date={tx_date}, disclosure_date={disc_date}, "
            f"official_filing_date={filing_date}, "
            f"notification_date={notif_date})"
        )
    if ordering:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM transactions
                WHERE transaction_date IS NOT NULL AND disclosure_date IS NOT NULL
                  AND (
                    (source IS DISTINCT FROM 'gemini_ocr'
                     AND official_filing_date IS NOT NULL
                     AND disclosure_date > official_filing_date)
                    OR
                    (notification_date IS NOT NULL
                     AND notification_date < transaction_date)
                    OR
                    (chamber ILIKE 'senate'
                     AND notification_date IS NOT NULL
                     AND official_filing_date IS NOT NULL
                     AND notification_date > official_filing_date)
                  )
                """
            ).fetchone()[0]
        )
        if total > len(ordering):
            result.violations.append(f"...and {total - len(ordering)} more row(s)")
    return result


CHECKS = [
    check_parse_counts_match_persisted,
    check_source_row_ids,
    check_no_transaction_after_disclosure,
    check_duplicate_policy,
    check_source_report_equation,
    check_completeness,
    check_canary_counts,
    check_house_generation_activation,
    check_date_domain,
]


def run_audit(conn: duckdb.DuckDBPyConnection) -> list[CheckResult]:
    """Run every check; a raised exception fails that check closed."""
    results: list[CheckResult] = []
    for check in CHECKS:
        result = CheckResult(check.__name__, check.__doc__ or "")
        try:
            result = check(conn)
        except Exception as exc:  # noqa: BLE001 - audits must fail closed
            result.violations.append(
                f"check crashed: {type(exc).__name__}: {exc}"
            )
        results.append(result)
    return results


def audit_database(db_path: str | Path) -> list[CheckResult]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return run_audit(conn)
    finally:
        conn.close()


def _render(results: list[CheckResult]) -> str:
    lines = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"[{status}] {result.name}: {result.description}")
        for violation in result.violations:
            lines.append(f"    violation: {violation}")
        for info in result.info:
            lines.append(f"    info: {info}")
    failed = sum(not result.passed for result in results)
    lines.append(f"{len(results) - failed}/{len(results)} checks passed")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only post-rebuild generation audit of a staged DuckDB "
            "database (never writes)."
        )
    )
    parser.add_argument("db_path", type=Path, help="staged DuckDB database path")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write machine-readable results to this path",
    )
    args = parser.parse_args(argv)

    if not args.db_path.exists():
        print(f"audit error: database not found: {args.db_path}", file=sys.stderr)
        return 2
    try:
        results = audit_database(args.db_path)
    except duckdb.Error as exc:
        print(f"audit error: cannot open {args.db_path} read-only: {exc}", file=sys.stderr)
        return 2

    print(_render(results))
    payload = {
        "db_path": str(args.db_path),
        "passed": all(result.passed for result in results),
        "checks": [
            {
                "name": result.name,
                "description": result.description,
                "passed": result.passed,
                "violations": result.violations,
                "info": result.info,
            }
            for result in results
        ],
    }
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
