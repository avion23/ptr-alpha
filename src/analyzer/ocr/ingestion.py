#!/usr/bin/env python3
"""Artifact-bound House OCR work selection, validation, and persistence.

Batch engine dispatch belongs to runner. Operational GLM staging and audit
policy remain executable tooling; their shared per-document file-publication
protocol lives here so wrappers use one lock without reversing dependencies.
"""

import datetime
import fcntl
import glob
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import duckdb
from pathlib import Path

from analyzer.persistence.database import Database
from analyzer.ocr.common import (
    GEMINI_PARSER_VERSION,
    MODEL,
    GeminiOutputError as GeminiOutputError,
    inspect_cached_response,
    parse_gemini_output,
    pdf_sha256,
    validate_transactions,
)

from analyzer.ocr.paths import DATA_DIR, STAGING_ROOT
from analyzer.ocr.normalization import (
    CACHE_ROW_COLUMNS, canonical_row_projection, normalize_rows,
    typed_instrument_fields,
)

DB_PATH = str(DATA_DIR / "congress.duckdb")
PROGRESS_PATH = str(DATA_DIR / "ocr_progress_gemini_manual.json")
COOLDOWN = 3  # seconds between requests (Lite model allows rapid fire)
# Authoritative unresolved-work manifest (FP review finding 8):
# ``get_ocr_work_items`` is the single source of truth for what still needs
# an OCR reading; --export-work-manifest freezes that selection to a JSON
# file the staging drain (glm_stage stage --from-manifest) consumes, so the
# pixel pipeline drains the real queue instead of re-scanning PDF dirs.
WORK_MANIFEST_SCHEMA = "glm-work-manifest-v1"
WORK_MANIFEST_DEFAULT_PATH = str(STAGING_ROOT / "work-manifest.json")
# Engine-B-primary single-engine drain (OCR_ENGINE=visionb): GLM reads the
# backlog alone — engine A (Gemini) is never contacted, conserving its free
# quota for current-era docs. Readings land as STAGING records
# (data/staging/glm-backlog, via scripts.glm_stage); canonical ingestion of
# GLM rows happens ONLY through ``glm_stage promote`` after the pixel audit,
# and promote inserts through this module's gemini_ocr source family, insert
# path, and validation.
GLM_PARSER_VERSION = "v1-glm53-vision"
GLM_ENGINES_ATTEMPTED = "visionb"
GLM_STAGING_SUBDIR = Path("staging") / "glm-backlog"
OCR_ENGINE_ENV = "OCR_ENGINE"
REQUIRED_OCR_SCHEMA_COLUMNS = {
    "chamber",
    "source_record_id",
    "source_row_id",
    "official_filing_date",
    "available_date",
    "notification_date",
    "amends_source_record_id",
    "raw_transaction_subtype",
    "ticker_origin",
    "raw_ticker",
    "ticker_candidate",
    "raw_asset_class",
    "raw_owner",
    "raw_asset_description",
    "ingestion_generation",
    "artifact_sha256",
}

OCR_INSERT_COLUMNS = [
    "doc_id",
    "member",
    "ticker",
    "transaction_date",
    "disclosure_date",
    "transaction_type",
    "amount_raw",
    "amount_midpoint",
    "owner_code",
    "asset_description",
    "source",
    "chamber",
    "source_record_id",
    "source_row_id",
    "official_filing_date",
    "available_date",
    "notification_date",
    "amends_source_record_id",
    "raw_transaction_subtype",
    "ticker_origin",
    "raw_ticker",
    "ticker_candidate",
    "raw_asset_class",
    "raw_asset_description",
    "raw_owner",
    "ingestion_generation",
    "artifact_sha256",
    "instrument_type",
    "strike_price",
    "expiry_date",
    "acceptance_datetime",
    "source_report_path",
    "member_key",
    "chamber_member_key",
]


def validate_ticker_provenance(row):
    origin = row["ticker_origin"]
    ticker = row["ticker"]
    raw_ticker = row["raw_ticker"]
    candidate = row["ticker_candidate"]
    if origin == "official":
        if not ticker or ticker != raw_ticker or candidate is not None:
            raise ValueError("official ticker provenance is inconsistent")
        return
    if origin == "asset_description":
        if not ticker or raw_ticker is not None or candidate is not None:
            raise ValueError("asset-description ticker provenance is inconsistent")
        return
    if origin == "unverified":
        if ticker is not None or not raw_ticker or candidate != raw_ticker:
            raise ValueError("unverified ticker provenance is inconsistent")
        return
    if origin == "not_reported":
        if ticker is not None or raw_ticker is not None or candidate is not None:
            raise ValueError("not-reported ticker provenance is inconsistent")
        return
    raise ValueError(f"invalid ticker_origin: {origin}")


def require_ocr_schema(conn):
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info('transactions')").fetchall()
    }
    missing = sorted(REQUIRED_OCR_SCHEMA_COLUMNS - existing)
    if missing:
        raise RuntimeError(
            "OCR ingestion requires the provenance schema; missing columns: "
            + ", ".join(missing)
        )
    return existing


# Amount range midpoint estimates (for amount_midpoint column)
AMOUNT_MIDPOINTS = {
    "A": 8000,
    "B": 32500,
    "C": 75000,
    "D": 175000,
    "E": 375000,
    "F": 750000,
    "G": 3000000,
    "H": 15000000,
    "I": 37500000,
    "J": 50000000,
    "K": 1000000,
}


def get_ocr_work_items(
    *,
    db_path: str = DB_PATH,
    data_dir: str | Path | None = None,
    year: int | None = None,
    parser_version: str = GEMINI_PARSER_VERSION,
    require_schema: bool = True,
    conn=None,
):
    """Select work from current artifacts, validated cache, schema, and DB rows.

    Years are House archive years (also the canonical acquisition/insert key),
    not filing-date years. This keeps default and year-scoped work identical
    for late filings instead of emitting two identities for the same document.
    """
    if conn is None:
        from analyzer.persistence.ownership import resolve_read_path
        owned = duckdb.connect(str(resolve_read_path(db_path)), read_only=True)
        try:
            return get_ocr_work_items(db_path=db_path, data_dir=data_dir, year=year,
                                      parser_version=parser_version, require_schema=require_schema, conn=owned)
        finally:
            owned.close()
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info('transactions')").fetchall()
    }
    schema_ready = REQUIRED_OCR_SCHEMA_COLUMNS <= existing_columns
    if require_schema and not schema_ready:
        require_ocr_schema(conn)
    base = Path(data_dir) if data_dir is not None else Path(db_path).parent
    generation_work = []
    if schema_ready and year is not None:
        db = Database.__new__(Database)
        db.conn = conn
        try:
            generation = db.get_latest_house_generation(year)
            scoped_ptrs = 0
            if generation is not None:
                scoped_ptrs = int(
                    db.conn.execute(
                        """
                        SELECT COUNT(*) FROM house_generation_metadata
                        WHERE archive_year = ? AND generation_id = ?
                          AND filing_type = 'P'
                        """,
                        [year, generation],
                    ).fetchone()[0]
                )
            if generation is not None and scoped_ptrs > 0:
                unresolved = db.get_unresolved_house_doc_ids(year, generation)
                generation_work = [
                    (
                        doc_id,
                        year,
                        str(base / str(year) / "pdfs" / f"{doc_id}.pdf"),
                    )
                    for doc_id in unresolved
                ]
        finally:
            pass  # borrowed coordinator connection

    if not schema_ready:
        legacy_rows = conn.execute(
            """
            WITH deterministic AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY doc_id ORDER BY parsed_at DESC
                ) AS rn
                FROM pdf_parse_runs
                WHERE NOT (
                    LOWER(COALESCE(parser_version, '')) LIKE '%gemini%'
                    OR LOWER(COALESCE(parser_version, '')) LIKE '%ling%'
                    OR LOWER(COALESCE(parser_version, '')) LIKE '%gpt%'
                    OR LOWER(COALESCE(parser_version, '')) LIKE '%openrouter%'
                    OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini/%'
                    OR LOWER(COALESCE(engines_attempted, '')) LIKE 'openrouter/%'
                    OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gpt-%'
                )
            )
            SELECT CAST(m.doc_id AS VARCHAR),
                   COALESCE(m.archive_year, CAST(EXTRACT(YEAR FROM m.filing_date) AS INTEGER))
            FROM metadata m
            JOIN deterministic d ON d.doc_id = m.doc_id AND d.rn = 1
            WHERE m.filing_type = 'P'
              AND d.status IN ('zero_rows', 'error', 'rejected', 'partial')
              AND (? IS NULL OR COALESCE(m.archive_year, EXTRACT(YEAR FROM m.filing_date)) = ?)
            ORDER BY EXTRACT(YEAR FROM m.filing_date), m.doc_id
            """,
            [year, year],
        ).fetchall()
        return [
            (
                doc_id,
                doc_year,
                str(base / str(doc_year) / "pdfs" / f"{doc_id}.pdf"),
            )
            for doc_id, doc_year in legacy_rows
        ]
    rows = conn.execute(
        """
        WITH deterministic AS (
            -- Dual-read runs (parser_version v6-dual-validated, engines
            -- 'gemini+%') intentionally count as deterministic recency here:
            -- after an AGREE (success) or DISAGREE (invalid) decision the doc
            -- must leave the OCR work list instead of being re-read nightly.
            -- Engine-B-primary runs (parser_version v1-glm53-vision, engines
            -- 'visionb') count the same way: a GLM success/no_txs run is done
            -- and must leave the work list; only an 'error' run keeps the doc
            -- in work for the next nightly retry.
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY doc_id ORDER BY parsed_at DESC
            ) AS rn
            FROM pdf_parse_runs
            WHERE NOT (
                LOWER(COALESCE(parser_version, '')) LIKE '%gemini%'
                OR LOWER(COALESCE(parser_version, '')) LIKE '%ling%'
                OR LOWER(COALESCE(parser_version, '')) LIKE '%gpt%'
                OR LOWER(COALESCE(parser_version, '')) LIKE '%openrouter%'
                OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini/%'
                OR LOWER(COALESCE(engines_attempted, '')) LIKE 'openrouter/%'
                OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gpt-%'
            )
        ), current_ocr AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY doc_id ORDER BY parsed_at DESC
            ) AS rn
            FROM pdf_parse_runs
            WHERE parser_version = ?
        ), tx AS (
            SELECT doc_id,
                   COUNT(*) FILTER (
                       WHERE source = 'gemini_ocr'
                             AND (chamber = 'House' OR chamber IS NULL)
                             AND ingestion_generation = ?
                   ) AS current_ocr_row_count,
                   COUNT(*) FILTER (
                       WHERE source = 'gemini_ocr'
                         AND (chamber = 'House' OR chamber IS NULL)
                         AND (ingestion_generation IS NULL OR ingestion_generation <> ?)
                   ) AS stale_ocr_row_count
            FROM transactions
            GROUP BY doc_id
        )
        SELECT CAST(m.doc_id AS VARCHAR),
               COALESCE(m.archive_year, CAST(EXTRACT(YEAR FROM m.filing_date) AS INTEGER)),
               o.status,
               o.raw_row_count,
               o.transaction_count,
               COALESCE(tx.current_ocr_row_count, 0),
               COALESCE(tx.stale_ocr_row_count, 0)
        FROM metadata m
        JOIN deterministic d ON d.doc_id = m.doc_id AND d.rn = 1
        LEFT JOIN current_ocr o ON o.doc_id = m.doc_id AND o.rn = 1
        LEFT JOIN tx ON tx.doc_id = m.doc_id
        WHERE m.filing_type = 'P'
          AND d.status IN ('zero_rows', 'error', 'rejected', 'partial')
          AND (? IS NULL OR COALESCE(m.archive_year, EXTRACT(YEAR FROM m.filing_date)) = ?)
        ORDER BY EXTRACT(YEAR FROM m.filing_date), m.doc_id
        """,
        [parser_version, parser_version, parser_version, year, year],
    ).fetchall()
    cache_dir = str(base / "gemini_cache")
    unresolved = []
    for (
        doc_id,
        doc_year,
        status,
        recorded_raw_count,
        recorded_count,
        current_ocr_row_count,
        stale_ocr_row_count,
    ) in rows:
        pdf_path = str(base / str(doc_year) / "pdfs" / f"{doc_id}.pdf")
        try:
            cached = inspect_cached_response(
                doc_id, pdf_path, cache_dir, parser_version
            )
        except (OSError, ValueError):
            cached = None
        if cached is None or not schema_ready:
            unresolved.append((doc_id, doc_year, pdf_path))
            continue
        parsed_count = len(cached.parsed.transactions)
        try:
            ingestion_generation = _resolve_ingestion_generation(
                conn, doc_id, doc_year, cached.pdf_sha256
            )
        except RuntimeError:
            unresolved.append((doc_id, doc_year, pdf_path))
            continue
        bound_run = conn.execute(
            """
            SELECT status, raw_row_count, transaction_count
            FROM pdf_parse_runs
            WHERE doc_id = ? AND parser_version = ?
              AND artifact_sha256 = ? AND ingestion_generation = ?
            ORDER BY parsed_at DESC LIMIT 1
            """,
            [
                doc_id,
                parser_version,
                cached.pdf_sha256,
                ingestion_generation,
            ],
        ).fetchone()
        if bound_run is None:
            status = None
            recorded_raw_count = None
            recorded_count = None
        else:
            status, recorded_raw_count, recorded_count = bound_run
        matching_rows = conn.execute(
            """
            SELECT source_row_id, raw_asset_description, raw_transaction_subtype,
                   transaction_date, notification_date, amount_raw, owner_code, raw_owner,
                   instrument_type, strike_price, expiry_date
            FROM transactions
            WHERE doc_id = ? AND source = 'gemini_ocr'
              AND (chamber = 'House' OR chamber IS NULL)
              AND ingestion_generation = ? AND artifact_sha256 = ?
            ORDER BY source_row_id
            """,
            [doc_id, ingestion_generation, cached.pdf_sha256],
        ).fetchall()
        current_ocr_row_count = len(matching_rows)
        stale_ocr_row_count = conn.execute(
            """
            SELECT COUNT(*) FROM transactions
            WHERE doc_id = ? AND source = 'gemini_ocr'
              AND (chamber = 'House' OR chamber IS NULL)
              AND NOT (
                  ingestion_generation IS NOT DISTINCT FROM ?
                  AND artifact_sha256 IS NOT DISTINCT FROM ?
              )
            """,
            [doc_id, ingestion_generation, cached.pdf_sha256],
        ).fetchone()[0]
        bound_metadata = conn.execute(
            "SELECT filing_date, first_name, last_name FROM house_generation_metadata "
            "WHERE archive_year = ? AND generation_id = ? AND doc_id = ? AND filing_type = 'P'",
            [doc_year, ingestion_generation, doc_id],
        ).fetchone()
        if bound_metadata is None:
            unresolved.append((doc_id, doc_year, pdf_path))
            continue
        normalized, findings = normalize_rows(
            doc_id, cached.parsed.member, cached.parsed.transactions,
            bound_metadata[0], " ".join(part for part in bound_metadata[1:] if part), cached.pdf_sha256,
        )
        expected_persistence, projection_errors = _normalize_ocr_rows(
            doc_id, cached.parsed.member, normalized, bound_metadata[0],
            artifact_sha256=cached.pdf_sha256, ingestion_generation=ingestion_generation,
        )
        expected_rows = [canonical_row_projection(row) for row in expected_persistence]
        matching_rows = [canonical_row_projection(dict(zip(CACHE_ROW_COLUMNS, row))) for row in matching_rows]
        success_matches = (
            status == "success"
            and not findings
            and not projection_errors
            and recorded_raw_count == cached.parsed.raw_row_count
            and recorded_count == parsed_count
            and current_ocr_row_count == parsed_count
            and stale_ocr_row_count == 0
            and matching_rows == sorted(expected_rows)
        )
        no_txs_matches = (
            status == "no_txs"
            and cached.parsed.no_transactions
            and current_ocr_row_count == 0
            and stale_ocr_row_count == 0
        )
        if not (success_matches or no_txs_matches):
            unresolved.append((doc_id, doc_year, pdf_path))
    return sorted(set(unresolved) | set(generation_work))


def get_zero_row_pdfs():
    """Backward-compatible all-year work selection from current DB state."""
    return get_ocr_work_items()


def build_work_manifest(
    *,
    db_path: str = DB_PATH,
    data_dir: str | Path | None = None,
    year: int | None = None,
    conn=None,
) -> dict:
    """The authoritative unresolved OCR queue as an artifact-bound manifest.

    Exports exactly ``get_ocr_work_items`` — the same selection every drain
    mode resolves against — binding each entry to the LOCAL artifact's
    sha256 and its House ingestion generation at export time (both are
    None when the artifact is missing from disk or not bound to the
    current generation: the doc is unresolved precisely because that
    binding is absent or unconfirmed, and the record stays honest instead
    of guessing). ``glm_stage stage --from-manifest`` consumes the frozen
    file so cron drains an inspectable queue rather than a directory scan
    that includes text-born docs and re-admits promoted docs. New ``year``
    groups use archive years; additive ``archive_year`` keeps artifact identity
    explicit while old manifests may retain filing-year staging groups.
    """
    if conn is None:
        from analyzer.persistence.ownership import resolve_read_path
        owned = duckdb.connect(str(resolve_read_path(db_path)), read_only=True)
        try:
            return build_work_manifest(db_path=db_path, data_dir=data_dir, year=year, conn=owned)
        finally:
            owned.close()
    base = Path(data_dir) if data_dir is not None else Path(db_path).parent
    items = []
    for doc_id, doc_year, pdf_path in get_ocr_work_items(
        db_path=db_path, data_dir=base, year=year, conn=conn
    ):
        entry = {
            "doc_id": str(doc_id),
            "year": int(doc_year),
            "archive_year": int(doc_year),
            "artifact_sha256": None,
            "ingestion_generation": None,
        }
        local_pdf = Path(pdf_path)
        if local_pdf.exists():
            entry["artifact_sha256"] = pdf_sha256(local_pdf)
        items.append(entry)
    if items:
        for entry in items:
            if entry["artifact_sha256"] is None:
                continue
            try:
                entry["ingestion_generation"] = _resolve_ingestion_generation(
                    conn, entry["doc_id"], entry["year"], entry["artifact_sha256"]
                )
            except RuntimeError:
                entry["ingestion_generation"] = None
    return {
        "schema": WORK_MANIFEST_SCHEMA,
        "generated_at_utc": (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        ),
        "db_path": str(db_path),
        "year": int(year) if year is not None else None,
        "item_count": len(items),
        "items": items,
    }


@dataclass(frozen=True)
class ApprovedDataRoot:
    """One operation's approved directory identity, not a re-resolvable path."""
    path: Path
    declared_path: Path
    fd: int

    def __fspath__(self):
        # Existing Path-based readers/writers must follow this same directory,
        # even if its original pathname is renamed/replaced during the job.
        # The parent PID also keeps subprocess output paths usable (CLOEXEC).
        return f"/proc/{os.getpid()}/fd/{self.fd}"


@contextmanager
def approved_data_root(data_dir):
    if isinstance(data_dir, ApprovedDataRoot):
        yield data_dir  # nested helpers borrow; only the outer owner closes
        return
    declared = Path(os.path.abspath(data_dir))
    root = declared.resolve()
    with ExitStack() as stack:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open("/", flags)
        stack.callback(os.close, fd)
        for component in root.parts[1:]:
            fd = os.open(component, flags, dir_fd=fd)
            stack.callback(os.close, fd)
        yield ApprovedDataRoot(root, declared, fd)


def read_work_artifact(data_dir, path: str | Path) -> bytes:
    """Read one regular file via the operation's pinned root FD, never symlinks."""
    with approved_data_root(data_dir) as root, ExitStack() as stack:
        candidate = Path(os.path.abspath(path))
        try:
            relative = candidate.relative_to(Path(root))
        except ValueError:
            try:
                relative = candidate.relative_to(root.path)
            except ValueError as exc:
                try:
                    relative = candidate.relative_to(root.declared_path)
                except ValueError:
                    raise RuntimeError("artifact path escapes the approved data root") from exc
        fd = root.fd
        for component in relative.parts[:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            stack.callback(os.close, fd)
        artifact_fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(artifact_fd, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise RuntimeError("artifact is not a regular file")
            return source.read()


def local_work_artifact(conn, data_dir: str | Path, item: dict) -> tuple[Path, dict]:
    """Resolve old/new queue identities without scanning for same-ID PDFs.

    ``year`` is the staging group; metadata's archive year locates the House
    artifact. Crossing years requires a recorded, current acquisition. Legacy
    same-year unbound manifests remain readable, but never authorize a
    canonical skip. Explicit hashes/generations are constraints, not hints.
    """
    if not isinstance(data_dir, ApprovedDataRoot):
        with approved_data_root(data_dir) as root:
            return local_work_artifact(conn, root, item)
    doc_id, group_year = str(item["doc_id"]), int(item["year"])
    archive_year = item.get("archive_year")
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT archive_year FROM metadata WHERE doc_id = ?", [doc_id]
            ).fetchone()
        except duckdb.CatalogException:
            row = None  # legacy synthetic/manual DB without House metadata
        if row and row[0] is not None:
            if archive_year is not None and archive_year != row[0]:
                raise RuntimeError("manifest archive_year disagrees with House metadata")
            archive_year = row[0]
    archive_year = archive_year if archive_year is not None else group_year
    if not isinstance(archive_year, int) or isinstance(archive_year, bool) or not 1000 <= archive_year <= 9999:
        raise RuntimeError("invalid artifact archive_year")
    if Path(doc_id).name != doc_id or doc_id in ("", ".", ".."):
        raise RuntimeError("invalid artifact doc_id")
    path = data_dir.path / str(archive_year) / "pdfs" / f"{doc_id}.pdf"
    identity = {"archive_year": archive_year, "artifact_sha256": None, "ingestion_generation": None}
    try:
        pdf_bytes = read_work_artifact(data_dir, path)
    except FileNotFoundError:
        return path, identity
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    expected = item.get("artifact_sha256", item.get("pdf_sha256"))
    if expected is not None and digest != expected:
        raise RuntimeError("local artifact does not match expected SHA-256")
    generation = None
    if conn is not None:
        try:
            generation = _resolve_ingestion_generation(conn, doc_id, archive_year, digest)
        except (RuntimeError, duckdb.CatalogException):
            pass
    if item.get("ingestion_generation") is not None and generation != item["ingestion_generation"]:
        raise RuntimeError("artifact is not bound to the expected current House generation")
    if archive_year != group_year and generation is None:
        raise RuntimeError("cross-year artifact lacks a current recorded House acquisition")
    identity.update(artifact_sha256=digest, ingestion_generation=generation, pdf_bytes=pdf_bytes)
    return path, identity


def export_work_manifest(
    path: str | Path = WORK_MANIFEST_DEFAULT_PATH, **kwargs
) -> dict:
    """Atomically write ``build_work_manifest`` to ``path``; returns it."""
    envelope = build_work_manifest(**kwargs)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp_path, manifest_path)
    return envelope


def load_progress(path: str | Path = PROGRESS_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "errors": [], "no_txs": []}


def save_progress(progress, path: str | Path = PROGRESS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, path)


def mark_progress(progress, doc_id, status):
    """Record one confirmed terminal/current result without stale overlaps."""
    key = "completed" if status == "success" else status
    if key not in {"completed", "errors", "no_txs"}:
        raise ValueError(f"unsupported progress status: {status}")
    doc_id = str(doc_id)
    for values in progress.values():
        while doc_id in values:
            values.remove(doc_id)
    progress[key].append(doc_id)


def parse_output(output, *, expected_page_count=None):
    """Parse a schema-validated Gemini response into the legacy tuple API."""
    parsed = parse_gemini_output(output, expected_page_count=expected_page_count)
    return parsed.member, parsed.transactions


def normalize_date(date_str):
    """Convert strict MM/DD[/YY] or MM-DD[-YY] dates to YYYY-MM-DD."""
    if not date_str:
        return None
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", date_str.strip())
    if not m:
        return None
    month, day, year = m.groups()
    month_i, day_i = int(month), int(day)
    if month_i < 1 or month_i > 12 or day_i < 1 or day_i > 31:
        return None
    if len(year) == 2:
        year = "20" + year if int(year) < 50 else "19" + year
    year_i = int(year)
    try:
        datetime.date(year_i, month_i, day_i)
    except ValueError:
        return None
    return f"{year}-{month_i:02d}-{day_i:02d}"


def extract_ticker(asset, *, explicit_only=False):
    """Extract stock ticker from common House asset formats."""
    if not asset:
        return None
    text = asset.upper()
    patterns = [
        r"\bTICKER\s*[:=]\s*([A-Z]{1,5}(?:\.[AB])?)\b",
    ]
    if not explicit_only:
        patterns.extend([
            r"\(([A-Z]{1,5}(?:\.[AB])?)\)",
            r"\$([A-Z]{1,5}(?:\.[AB])?)\b",
        ])
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            ticker = match.group(1)
            if ticker not in {"LLC", "LP", "LLP", "INC", "ETF", "ST", "MF", "OT"}:
                return ticker
    return None


# Hardcoded mapping of common company names/keywords → ticker symbols
# Used when OCR asset descriptions lack explicit ticker symbols.
COMPANY_TICKER_MAP = {
    # Mega-cap tech
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "adobe": "ADBE",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "intel": "INTC",
    "amd": "AMD",
    "broadcom": "AVGO",
    "cisco": "CSCO",
    "qualcomm": "QCOM",
    "ibm": "IBM",
    "tx": "TXN",
    "intuit": "INTU",
    "paypal": "PYPL",
    "shopify": "SHOP",
    "uber": "UBER",
    "lyft": "LYFT",
    "snap": "SNAP",
    "pinterest": "PINS",
    "robinhood": "HOOD",
    "coinbase": "COIN",
    "block": "SQ",
    "square": "SQ",
    "zoom": "ZM",
    "crowdstrike": "CRWD",
    "palo alto": "PANW",
    "cloudflare": "NET",
    "datadog": "DDOG",
    "mongodb": "MDB",
    "snowflake": "SNOW",
    "databricks": "DBX",
    "twilio": "TWLO",
    "spotify": "SPOT",
    "roku": "ROKU",
    "palantir": "PLTR",
    "unity": "U",
    "roblox": "RBLX",
    "doximity": "DOCS",
    # Finance / Banking
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "goldman sachs": "GS",
    "goldman": "GS",
    "morgan stanley": "MS",
    "wells fargo": "WFC",
    "citigroup": "C",
    "citi": "C",
    "us bancorp": "USB",
    "truist": "TFC",
    "pnc": "PNC",
    "charles schwab": "SCHW",
    "schwab": "SCHW",
    "barclays": "BCS",
    "hsbc": "HSBC",
    "ubs": "UBS",
    "credit suisse": "CS",
    "american express": "AXP",
    "visa": "V",
    "mastercard": "MA",
    "blackrock": "BLK",
    "blackstone": "BX",
    "vanguard": "VT",
    "fidelity": "FNF",
    "state street": "STT",
    "northern trust": "NTRS",
    "t rowe price": "TROW",
    "berkshire": "BRK",
    "berkshire hathaway": "BRK",
    "capital one": "COF",
    "discover": "DFS",
    "synchrony": "SYF",
    "stock exchange": "ICE",
    "cme group": "CME",
    "nasdaq": "NDAQ",
    "intercontinental exchange": "ICE",
    "spglobal": "SPGI",
    "s&p global": "SPGI",
    "moody": "MCO",
    "moody's": "MCO",
    # Healthcare / Pharma
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "pfizer": "PFE",
    "unitedhealth": "UNH",
    "united health": "UNH",
    "abbvie": "ABBV",
    "merck": "MRK",
    "abbott": "ABT",
    "amgen": "AMGN",
    "gilead": "GILD",
    "bristol-myers": "BMY",
    "bristol myers": "BMY",
    "eli lilly": "LLY",
    "lilly": "LLY",
    "regeneron": "REGN",
    "vertex": "VRTX",
    "moderna": "MRNA",
    "biogen": "BIIB",
    "vertex pharmaceuticals": "VRTX",
    "cigna": "CI",
    "humana": "HUM",
    "anthem": "ELV",
    "elevance": "ELV",
    "centene": "CNC",
    "aetna": "AET",
    "mcdermott": "MCD",
    "medtronic": "MDT",
    "baxter": "BAX",
    "becton dickinson": "BDX",
    "stryker": "SYK",
    "zimmer biomet": "ZBH",
    "intuitive surgical": "ISRG",
    "intuitive": "ISRG",
    "hologic": "HOLX",
    "idexx": "IDXX",
    "charles river": "CRL",
    # Consumer / Retail
    "walmart": "WMT",
    "costco": "COST",
    "target": "TGT",
    "home depot": "HD",
    "lowes": "LOW",
    "ikea": "INGKA",
    "dollar general": "DG",
    "dollar tree": "DLTR",
    "best buy": "BBY",
    "nordstrom": "JWN",
    "macys": "M",
    "macy's": "M",
    "kohl's": "KSS",
    "tjx": "TJX",
    "ross stores": "ROST",
    "lululemon": "LULU",
    "nike": "NKE",
    "adidas": "ADDYY",
    "puma": "PUMAY",
    "under armour": "UAA",
    "gap": "GPS",
    "old navy": "GPS",
    "zara": "ITX",
    "h&m": "HNNMY",
    "starbucks": "SBUX",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "chipotle": "CMG",
    "yum": "YUM",
    "yum brands": "YUM",
    "coca-cola": "KO",
    "coca cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "procter": "PG",
    "procter & gamble": "PG",
    "unilever": "UL",
    "colgate": "CL",
    "kellogg's": "K",
    "kellogg": "K",
    "general mills": "GIS",
    "campbell": "CPB",
    "conagra": "CAG",
    "kraft heinz": "KHC",
    "mondelez": "MDLZ",
    "mars": "MWWC",
    "nestle": "NSRGY",
    # Energy / Oil & Gas
    "exxon": "XOM",
    "exxon mobil": "XOM",
    "chevron": "CVX",
    "conocophillips": "COP",
    "shell": "SHEL",
    "bp": "BP",
    "occidental": "OXY",
    "occidental petroleum": "OXY",
    "devon": "DVN",
    "devon energy": "DVN",
    "marathon": "MPC",
    "marathon petroleum": "MPC",
    "valero": "VLO",
    "phillips 66": "PSX",
    "sunoco": "SUN",
    "nextera": "NEE",
    "nextera energy": "NEE",
    "duke energy": "DUK",
    "southern company": "SO",
    "dominion": "D",
    "dominion energy": "D",
    "american electric": "AEP",
    "first solar": "FSLR",
    "enphase": "ENPH",
    "solar edge": "SEDG",
    # Industrials / Aerospace
    "boeing": "BA",
    "lockheed": "LMT",
    "lockheed martin": "LMT",
    "raytheon": "RTX",
    "rtx": "RTX",
    "northrop grumman": "NOC",
    "general dynamics": "GD",
    "l3harris": "LHX",
    "transdigm": "TDG",
    "honeywell": "HON",
    "3m": "MMM",
    "caterpillar": "CAT",
    "deere": "DE",
    "john deere": "DE",
    "general electric": "GE",
    "siemens": "SIEGY",
    "schneider electric": "SU.PA",
    "emerson": "EMR",
    "parker hannifin": "PH",
    "rockwell": "ROK",
    "illinois tool": "ITW",
    "union pacific": "UNP",
    "burlington northern": "BRK",
    "csx": "CSX",
    "norfolk southern": "NSC",
    "fedex": "FDX",
    "ups": "UPS",
    "xpo": "XPO",
    "j.b. hunt": "JBHT",
    "waste management": "WM",
    "republic services": "RSG",
    "cintas": "CTAS",
    "aramark": "ARMK",
    # Telecom / Media
    "at&t": "T",
    "at&t inc": "T",
    "verizon": "VZ",
    "t-mobile": "TMUS",
    "comcast": "CMCSA",
    "charter": "CHTR",
    "charter communications": "CHTR",
    "dish": "DISH",
    "dish network": "DISH",
    "disney": "DIS",
    "warner bros": "WBD",
    "warner discovery": "WBD",
    "paramount": "PARA",
    "fox": "FOX",
    "fox corporation": "FOX",
    "nbcuniversal": "CMCSA",
    "viacom": "VIA",
    "live nation": "LYV",
    "spotify technology": "SPOT",
    # Real Estate / REITs
    "prologis": "PLD",
    "american tower": "AMT",
    "equinix": "EQIX",
    "realty income": "O",
    "public storage": "PSA",
    "welltower": "WELL",
    "avalonbay": "AVB",
    "easterly government": "DEA",
    "digital realty": "DLR",
    "crown castle": "CCI",
    "simon property": "SPG",
    "tanger": "SKT",
    # Food / Beverage
    "starbucks corporation": "SBUX",
    "monster beverage": "MNST",
    "constellation": "STZ",
    "brown forman": "BF.B",
    "domino's": "DPZ",
    "dominos": "DPZ",
    "dard": "DRI",
    "darden": "DRI",
    "wingstop": "WING",
    "sweetgreen": "SG",
    "shake shack": "SHAK",
    "caesars": "CZR",
    "caesars entertainment": "CZR",
    "las vegas sands": "LVS",
    "mgm": "MGM",
    "wynn": "WYNN",
    "melco": "MLCO",
    # EV / Auto
    "rivian": "RIVN",
    "lucid": "LCID",
    "lucid motors": "LCID",
    "nio": "NIO",
    "xpeng": "XPEV",
    "li auto": "LI",
    "toyota": "TM",
    "honda": "HMC",
    "hyundai": "HYMTF",
    "ford": "F",
    "general motors": "GM",
    "gm": "GM",
    "stellantis": "STLA",
    "ferrari": "RACE",
    "porsche": "POAHY",
    "lamborghini": "VOW3.DE",
    # Semiconductor
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "samsung": "SSNLF",
    "asml": "ASML",
    "arm holdings": "ARM",
    "arm": "ARM",
    "micron": "MU",
    "micron technology": "MU",
    "on semiconductor": "ON",
    "onsemi": "ON",
    "marvell": "MRVL",
    "marvell technology": "MRVL",
    "analog devices": "ADI",
    "maxim": "MXIM",
    "microchip": "MCHP",
    "nxpi": "NXPI",
    "nvidia corporation": "NVDA",
    "amd inc": "AMD",
    # Software / SaaS
    "microsoft corporation": "MSFT",
    "salesforce inc": "CRM",
    "servicenow": "NOW",
    "workday": "WDAY",
    "adobe inc": "ADBE",
    "vmware": "VMW",
    "broadcom inc": "AVGO",
    "synopsys": "SNPS",
    "cadence": "CDNS",
    "ansys": "ANSS",
    "splunk": "SPLK",
    "zscaler": "ZS",
    "okta": "OKTA",
    "cloudflare inc": "NET",
    "dynatrace": "DT",
    "new relic": "NEWR",
    "jfrog": "FROG",
    "confluent": "CFLT",
    "elastic": "ESTC",
    "hashicorp": "HCP",
    "gitlab": "GTLB",
    "atlassian": "TEAM",
    "hubspot": "HUBS",
    "zendesk": "ZEN",
    "freshworks": "FRSH",
    "monday.com": "MNDY",
    "c3.ai": "AI",
    "c3 ai": "AI",
    "openai": "PRIV",
    "anthropic": "PRIV",
    "databricks inc": "DBX",
    # Travel / Hospitality
    "marriott": "MAR",
    "marriott international": "MAR",
    "hilton": "HLT",
    "hilton worldwide": "HLT",
    "airbnb": "ABNB",
    "booking": "BKNG",
    "booking holdings": "BKNG",
    "expedia": "EXPE",
    "tripadvisor": "TRIP",
    "american airlines": "AAL",
    "delta": "DAL",
    "delta air lines": "DAL",
    "united airlines": "UAL",
    "southwest": "LUV",
    "southwest airlines": "LUV",
    "jetblue": "JBLU",
    "spirit airlines": "SAVE",
    "carnival": "CCL",
    "royal caribbean": "RCL",
    "norwegian": "NCLH",
    # Misc / Other
    "berkshire hathaway inc": "BRK",
    "berkshire hathaway class a": "BRK.A",
    "berkshire hathaway class b": "BRK.B",
    "leidos": "LDOS",
    "booz allen": "BAH",
    "accenture": "ACN",
    "deloitte": "PRIVATE",
    "pwc": "PRIVATE",
    "ey": "PRIVATE",
    "kpmg": "PRIVATE",
    "mckinsey": "PRIVATE",
    "spacex": "PRIV",
    "stripe": "PRIV",
    "airtable": "PRIV",
    "canva": "PRIV",
    "notion": "PRIV",
    "figma": "PRIVATE",
    "discord": "PRIV",
    "tiktok": "PRIV",
    "bytedance": "PRIV",
    "temu": "PDD",
    "pinduoduo": "PDD",
    "alibaba": "BABA",
    "baba": "BABA",
    "jd.com": "JD",
    "baidu": "BIDU",
    "tencent": "TCEHY",
    "xiaomi": "XIACF",
    "huawei": "PRIVATE",
    "boeing company": "BA",
    "the boeing company": "BA",
    "lockheed martin corp": "LMT",
    "raytheon technologies": "RTX",
    "lennar": "LEN",
    "pultegroup": "PHM",
    "d.r. horton": "DHI",
    "dr horton": "DHI",
    "meritage": "MTH",
    "toll brothers": "TOL",
    "nvr": "NVR",
    "pulte": "PHM",
    "dream finders": "DFH",
    "green brick": "GRBK",
    "schlumberger": "SLB",
    "slb": "SLB",
    "halliburton": "HAL",
    "baker hughes": "BKR",
    "weatherford": "WFRD",
    "crown holdings": "CCK",
    "sealed air": "SEE",
    "ball corporation": "BLL",
    "silgan": "SLGN",
    "verisk": "VRSK",
    "willis towers watson": "WTW",
    "marsh": "MMC",
    "marsh & mclennan": "MMC",
    "aon": "AON",
    "gallagher": "AJG",
    "arthur j": "AJG",
    "progressive": "PGR",
    "allstate": "ALL",
    "chubb": "CB",
    "hartford": "HIG",
    "travelers": "TRV",
    "metlife": "MET",
    "prudential": "PRU",
    "lincoln national": "LNC",
    "unum": "UNM",
    "aflac": "AFL",
    "cigna group": "CI",
    "hca healthcare": "HCA",
    "tenet": "TCP",
    "tenet healthcare": "TCP",
    "universal health": "UHS",
    "community health": "CYH",
    "davita": "DVA",
    "encompass health": "EHC",
    "waste connections": "WCN",
    "clean harcbors": "CLH",
    "stericycle": "SRCL",
    "republic services inc": "RSG",
    "deckers": "DECK",
    "columbia sportswear": "COLM",
    "ralph lauren": "RL",
    "capri": "CPRI",
    "tapestry": "TPR",
    "coach": "TPR",
    "kate spade": "CPRI",
    "michael kors": "CPRI",
    "hugo boss": "BOSS.DE",
    "burberry": "BRBY.L",
    "lvmh": "MC.PA",
    "hermes": "RMS.PA",
    "kering": "KER.PA",
    "christian dior": "CDI.PA",
    "prada": "1913.HK",
    "toyota motor": "TM",
    "honda motor": "HMC",
    "volkswagen": "VOW3.DE",
    "bmw": "BAMXF",
    "mercedes": "MBG.DE",
    "stellantis nv": "STLA",
    "qualcomm inc": "QCOM",
    "broadcom limited": "AVGO",
    "advanced micro devices": "AMD",
    "micron technology inc": "MU",
    "applovin": "APP",
    "applovin corp": "APP",
    "sea limited": "SE",
    "sea": "SE",
    "grab holdings": "GRAB",
    "grab": "GRAB",
    "sofi": "SOFI",
    "sofi technologies": "SOFI",
    "affirm": "AFRM",
    "affirm holdings": "AFRM",
    "upstart": "UPST",
    "upstart holdings": "UPST",
    "lendingclub": "LC",
    "lending club": "LC",
    "chime": "PRIV",
    "nubank": "NU",
    "nubank holdings": "NU",
    "nu holdings": "NU",
    "rocket companies": "RKT",
    "rocket mortgage": "RKT",
    "uwm": "UWMC",
    "united wholesale mortgage": "UWMC",
    "zillow": "Z",
    "zillow group": "Z",
    "redfin": "RDFN",
    "redfin corp": "RDFN",
    "opendoor": "OPEN",
    "opendoor technologies": "OPEN",
    "we work": "WE",
    "wework": "WE",
    "peloton": "PTON",
    "peloton interactive": "PTON",
    "whatnot": "PRIV",
    "poshmark": "POSH",
    "depop": "ETSY",
    "etsy": "ETSY",
    "etsy inc": "ETSY",
    "ebay": "EBAY",
    "ebay inc": "EBAY",
    "craigslist": "PRIVATE",
    "facebook marketplace": "META",
    "amazon marketplace": "AMZN",
    "amazon web services": "AMZN",
    "aws": "AMZN",
    "microsoft azure": "MSFT",
    "azure": "MSFT",
    "google cloud": "GOOGL",
    "gcp": "GOOGL",
    "openai chatgpt": "PRIV",
    "chatgpt": "PRIV",
    "midjourney": "PRIV",
    "stability ai": "PRIV",
    "cathie wood": "ARKK",
    "ark invest": "ARKK",
    "ark innovation": "ARKK",
    "ark genomics": "ARKG",
    "vanguard s&p 500": "VOO",
    "vanguard total": "VT",
    "ishares": "IVV",
    "spdr s&p": "SPY",
    "qqq": "QQQ",
    "invesco qqq": "QQQ",
    "ark": "ARKK",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "solana": "SOL-USD",
    "crypto": "BTC-USD",
    "treasury": "TLT",
    "treasury bond": "TLT",
    "i bonds": "GOVT",
    "tips": "TIP",
    "real estate investment trust": "VNQ",
    "reit": "VNQ",
    "vanguard real estate": "VNQ",
    "jpmorgan chase": "JPM",
    "jpmorgan chase & co": "JPM",
    "jpmorgan chase and co": "JPM",
    "bank of new york": "BK",
    "bnymellon": "BK",
    "state street corp": "STT",
    "northern trust corp": "NTRS",
    "visa inc": "V",
    "mastercard incorporated": "MA",
    "mastercard inc": "MA",
    "booz allen hamilton": "BAH",
    "booz allen hamilton holding": "BAH",
    "leidos holdings": "LDOS",
    "leidos inc": "LDOS",
    "general dynamics corp": "GD",
    "l3harris technologies": "LHX",
    "transdigm group": "TDG",
    "transdigm holdings": "TDG",
    "northrop grumman corp": "NOC",
}


def resolve_ticker(asset):
    """Resolve ticker from asset description text using keyword matching.

    Tries substring matching against COMPANY_TICKER_MAP keys. Returns
    the longest matching key's ticker (most specific match), or None.
    """
    if not asset:
        return None
    text = asset.lower()
    best_ticker = None
    best_len = 0
    for name, ticker in COMPANY_TICKER_MAP.items():
        if len(name) > best_len and name in text:
            best_ticker = ticker
            best_len = len(name)
    return best_ticker


def get_filing_date(conn, doc_id):
    row = conn.execute(
        "SELECT filing_date FROM metadata WHERE doc_id = ?",
        [str(doc_id)],
    ).fetchone()
    return row[0] if row else None


def get_metadata_member(conn, doc_id):
    row = conn.execute(
        "SELECT first_name, last_name FROM metadata WHERE doc_id = ?",
        [str(doc_id)],
    ).fetchone()
    if not row:
        return None
    return " ".join(part for part in row if part).strip() or None


def _resolve_ingestion_generation(
    conn, doc_id: str, year: int, artifact_sha256: str
) -> str:
    row = conn.execute(
        """
        SELECT artifact.generation_id
        FROM house_pdf_artifacts AS artifact
        JOIN house_archive_generations AS generation
          ON generation.archive_year = artifact.archive_year
         AND generation.generation_id = artifact.generation_id
        WHERE artifact.doc_id = ?
          AND artifact.archive_year = ?
          AND artifact.artifact_sha256 = ?
          AND artifact.generation_id = (
              SELECT generation_id FROM house_archive_generations
              WHERE archive_year = artifact.archive_year
              ORDER BY promoted_at DESC, generation_id DESC LIMIT 1
          )
        ORDER BY artifact.acquired_at DESC, artifact.generation_id DESC
        LIMIT 1
        """,
        [str(doc_id), year, artifact_sha256],
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "OCR artifact is not bound to an acquired House ingestion generation"
        )
    return str(row[0])


def resolve_ingestion_generation(
    db_path: str, doc_id: str, year: int, artifact_sha256: str, *, conn=None,
) -> str:
    if conn is not None:
        return _resolve_ingestion_generation(conn, doc_id, year, artifact_sha256)
    from analyzer.persistence.ownership import resolve_read_path
    conn = duckdb.connect(str(resolve_read_path(db_path)), read_only=True)
    try:
        return _resolve_ingestion_generation(conn, doc_id, year, artifact_sha256)
    finally:
        conn.close()


# GLM staging publication is shared by the operational OCR adapter, the
# staging command, and the pixel-audit recorder. Keep its per-document lock,
# archive, and atomic replacement on the OCR side of the package boundary.
_staging_publication_state = threading.local()


def glm_staging_root(data_dir: str | Path) -> Path:
    return Path(data_dir) / GLM_STAGING_SUBDIR


def glm_staging_record_path(data_dir: str | Path, year: int, doc_id: str) -> Path:
    return glm_staging_root(data_dir) / str(year) / f"{doc_id}.json"


def glm_staging_attempts_dir(data_dir: str | Path, year: int) -> Path:
    return glm_staging_root(data_dir) / str(year) / "attempts"


def archived_glm_staging_attempts(
    data_dir: str | Path, year: int, doc_id: str
) -> list[tuple[int, Path]]:
    directory = glm_staging_attempts_dir(data_dir, year)
    if not directory.exists():
        return []
    prefix = f"{glob.escape(str(doc_id))}.attempt-"
    attempts = []
    for path in directory.glob(f"{prefix}*.json"):
        suffix = path.name[len(str(doc_id)) + len(".attempt-") : -len(".json")]
        if suffix.isdigit():
            attempts.append((int(suffix), path))
    return sorted(attempts)


@contextmanager
def staging_document_lock(data_dir: str | Path, year: int, doc_id: str):
    """Reentrant per-document publication lock shared by all GLM writers."""
    root = glm_staging_root(data_dir).resolve()
    key = (str(root), int(year), str(doc_id))
    held = getattr(_staging_publication_state, "held", None)
    if held is None:
        held = _staging_publication_state.held = {}
    if key in held:
        held[key][1] += 1
        try:
            yield
        finally:
            held[key][1] -= 1
        return

    lock_name = hashlib.sha256(str(doc_id).encode("utf-8")).hexdigest()
    lock_path = root / ".publication-locks" / str(year) / f"{lock_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        held[key] = [lock_file, 1]
        try:
            yield
        finally:
            del held[key]
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _archive_glm_staging_attempt_unlocked(
    data_dir: str | Path, year: int, doc_id: str
) -> Path | None:
    live = glm_staging_record_path(data_dir, year, doc_id)
    if not live.exists():
        return None
    prior = archived_glm_staging_attempts(data_dir, year, doc_id)
    next_number = prior[-1][0] + 1 if prior else 0
    target_dir = glm_staging_attempts_dir(data_dir, year)
    target_dir.mkdir(parents=True, exist_ok=True)
    while True:
        target = target_dir / f"{doc_id}.attempt-{next_number}.json"
        try:
            # Hard-link the exact old inode; never replace an existing archive.
            os.link(live, target)
        except FileExistsError:
            prior = archived_glm_staging_attempts(data_dir, year, doc_id)
            next_number = max(next_number + 1, prior[-1][0] + 1 if prior else 0)
            continue
        live.unlink()
        return target


def archive_glm_staging_attempt(
    data_dir: str | Path, year: int, doc_id: str
) -> Path | None:
    with staging_document_lock(data_dir, year, doc_id):
        return _archive_glm_staging_attempt_unlocked(data_dir, year, doc_id)


def _atomic_write_glm_staging_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def replace_glm_staging_record(data_dir: str | Path, record: dict) -> Path:
    """Atomically replace mutable metadata while holding its document lock."""
    year, doc_id = int(record["year"]), str(record["doc_id"])
    path = glm_staging_record_path(data_dir, year, doc_id)
    with staging_document_lock(data_dir, year, doc_id):
        _atomic_write_glm_staging_json(path, record)
    return path


def publish_glm_staging_record(data_dir: str | Path, record: dict) -> tuple[Path, Path | None]:
    """Archive any live attempt and atomically publish a producer record."""
    year, doc_id = int(record["year"]), str(record["doc_id"])
    path = glm_staging_record_path(data_dir, year, doc_id)
    with staging_document_lock(data_dir, year, doc_id):
        archived = _archive_glm_staging_attempt_unlocked(data_dir, year, doc_id)
        _atomic_write_glm_staging_json(path, record)
    return path, archived


def record_parse_run(
    conn,
    doc_id,
    year,
    status,
    raw_count,
    tx_count,
    error_message="",
    parser_version=GEMINI_PARSER_VERSION,
    artifact_sha256: str | None = None,
    ingestion_generation: str | None = None,
    engines_attempted: str = MODEL,
):
    identity = [
        str(doc_id),
        parser_version,
        artifact_sha256,
        ingestion_generation,
    ]
    archive_rows(
        conn, "pdf_parse_runs",
        "doc_id = ? AND parser_version = ? "
        "AND artifact_sha256 IS NOT DISTINCT FROM ? "
        "AND ingestion_generation IS NOT DISTINCT FROM ?", identity,
    )
    conn.execute(
        """
        DELETE FROM pdf_parse_runs
        WHERE doc_id = ? AND parser_version = ?
          AND artifact_sha256 IS NOT DISTINCT FROM ?
          AND ingestion_generation IS NOT DISTINCT FROM ?
        """,
        identity,
    )
    conn.execute(
        """
        INSERT INTO pdf_parse_runs (
            doc_id, year, parser_version, status, engines_attempted,
            raw_row_count, transaction_count, error_message,
            artifact_sha256, ingestion_generation, parsed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(doc_id),
            year,
            parser_version,
            status,
            engines_attempted,
            raw_count,
            tx_count,
            error_message[:1000],
            artifact_sha256,
            ingestion_generation,
            datetime.datetime.now(),
        ],
    )
    # New schemas bind acceptance to the exact persisted source rowset rather
    # than relying on transaction-start versus wall-clock timestamp ordering.
    columns = {r[1] for r in conn.execute("PRAGMA table_info('pdf_parse_runs')").fetchall()}
    if "rowset_created_at" in columns:
        conn.execute(
            """UPDATE pdf_parse_runs SET rowset_created_at = (
                SELECT MAX(created_at) FROM transactions WHERE doc_id = ?
                  AND artifact_sha256 IS NOT DISTINCT FROM ?
                  AND ingestion_generation IS NOT DISTINCT FROM ? AND source = 'gemini_ocr')
                WHERE doc_id = ? AND parser_version = ?
                  AND artifact_sha256 IS NOT DISTINCT FROM ?
                  AND ingestion_generation IS NOT DISTINCT FROM ?""",
            [str(doc_id), artifact_sha256, ingestion_generation, *identity],
        )


def _retain_partial_ocr_result(
    conn,
    doc_id,
    year,
    transactions,
    row_rejections,
    *,
    parser_version,
    artifact_sha256,
    ingestion_generation,
):
    """Append rejected extraction evidence; this table is intentionally append-only."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ocr_partial_results (
            doc_id VARCHAR,
            year INTEGER,
            parser_version VARCHAR,
            artifact_sha256 VARCHAR,
            ingestion_generation VARCHAR,
            transactions_json JSON,
            rejections_json JSON,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO ocr_partial_results (
            doc_id, year, parser_version, artifact_sha256,
            ingestion_generation, transactions_json, rejections_json
        ) VALUES (?, ?, ?, ?, ?, CAST(? AS JSON), CAST(? AS JSON))
        """,
        [
            str(doc_id), year, parser_version, artifact_sha256,
            ingestion_generation, json.dumps(transactions, default=str),
            json.dumps(row_rejections, sort_keys=True),
        ],
    )


def archive_rows(conn, table, predicate, parameters):
    """Archive exactly the rows a transactional mutation is about to retire."""
    history = {"transactions": "transaction_versions", "pdf_parse_runs": "parse_run_versions"}[table]
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {history} AS "
        f"SELECT *, CURRENT_TIMESTAMP AS retired_at FROM {table} WHERE FALSE"
    )
    conn.execute(
        f"INSERT INTO {history} SELECT *, CURRENT_TIMESTAMP FROM {table} WHERE {predicate}",
        parameters,
    )


def record_failed_ocr_attempt(
    conn,
    doc_id,
    year,
    raw_count,
    error_message,
    *,
    parser_version: str = GEMINI_PARSER_VERSION,
    artifact_sha256: str | None = None,
    ingestion_generation: str | None = None,
    engine_model: str = MODEL,
    receipt=None,
) -> None:
    """Record a failed OCR attempt and retire rows for that exact artifact."""
    if _receipt_replay(conn, receipt, doc_id, year, artifact_sha256, ingestion_generation, parser_version) is not None:
        return
    conn.execute("BEGIN TRANSACTION")
    try:
        if artifact_sha256 and ingestion_generation:
            archive_rows(conn, "transactions",
                          "source = 'gemini_ocr' AND doc_id = ? "
                          "AND (chamber = 'House' OR chamber IS NULL) "
                          "AND ingestion_generation = ? AND artifact_sha256 = ?",
                          [str(doc_id), ingestion_generation, artifact_sha256])
            conn.execute(
                """
                DELETE FROM transactions
                WHERE source = 'gemini_ocr' AND doc_id = ?
                  AND (chamber = 'House' OR chamber IS NULL)
                  AND ingestion_generation = ? AND artifact_sha256 = ?
                """,
                [str(doc_id), ingestion_generation, artifact_sha256],
            )
        record_parse_run(
            conn,
            doc_id,
            year,
            "error",
            raw_count,
            0,
            error_message,
            parser_version=parser_version,
            artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation,
            engines_attempted=engine_model,
        )
        _finish_receipt(conn, receipt, "needs_review", error_message)
        conn.execute("COMMIT")
    except Exception:
        cleanup_ocr_resource(lambda: conn.execute("ROLLBACK"))
        raise


@dataclass(frozen=True)
class InsertOutcome:
    """The explicit committed outcome of one insert_transactions call.

    ``committed`` is true only when a transaction actually COMMITed (rows or
    a no_txs parse run). ``complete`` is true only when everything the caller
    handed in was committed with no rejections — a partial insert (row-level
    rejections kept out) or a member/row rejection is never "complete", so
    callers cannot mistake partial output for a finished document.
    """

    status: str  # success | partial | no_txs | rejected | error
    inserted: int = 0
    expected: int = 0
    committed: bool = False
    rejections: dict | None = None
    detail: str | None = None

    @property
    def complete(self) -> bool:
        return (
            self.committed
            and self.status in ("success", "no_txs")
            and self.inserted == self.expected
        )

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "inserted": self.inserted,
            "expected": self.expected,
            "committed": self.committed,
            "rejections": self.rejections,
            "detail": self.detail,
        }


def _finish_receipt(conn, receipt, state, detail=None):
    if receipt is not None:
        from analyzer.persistence.work_repository import WorkRepository
        repository = WorkRepository(conn)
        if state in {"accepted", "verified_empty"}:
            from analyzer.ocr.review import assert_review_supersession

            # Final authoritative check inside the same SQL transaction as rows
            # and receipt. The committed binding identifies the revoked decision
            # this fresh operator review explicitly supersedes.
            assert_review_supersession(repository, receipt["result_id"], detail["review_binding"])
        if "admission_decision" in receipt:
            from dataclasses import replace
            from analyzer.ocr.contracts import ExtractionOutcome
            from analyzer.ocr.workers import _publish

            decision = receipt["admission_decision"]
            detail = detail if isinstance(detail, dict) else {"reason": detail}
            source_proposal = detail.get("source_proposal")
            source_lineage = detail.get("source_review_lineage")
            if source_proposal is not None:
                from analyzer.ocr.review import digest, POLICY
                if (detail.get("review_binding", {}).get("authority") != POLICY["ai_authority"]
                        or source_lineage["original_raw_result_id"] != receipt["result_id"]
                        or digest(decision["result"].to_dict()) != receipt["result_id"]
                        or source_lineage["source_origin"] != "independent-ai-source-proposal-v1"
                        or source_lineage["extracted_raw_count"] != len(decision["result"].rows)
                        or source_lineage["source_proposal_sha256"] != digest(source_proposal)
                        or source_lineage["verified_source_count"] != len(source_proposal["rows"])
                        or source_lineage["derived_outcome"] != ("verified_empty" if state == "verified_empty" else "candidate")):
                    raise ValueError("source-reviewed decision lineage does not bind original receipt and derived rows")
            bundle = replace(
                decision["result"],
                rows=source_proposal["rows"] if source_proposal is not None else decision["result"].rows,
                outcome=ExtractionOutcome({"accepted": "candidate", "verified_empty": "verified_empty",
                                           "needs_review": "needs_review", "blocked": "blocked"}[state]),
                evidence={**decision["result"].evidence,
                          **({"source_review_lineage": source_lineage,
                              "source_proposal": source_proposal} if source_proposal is not None else {}),
                          "admission_decision": {
                    "predecessor_result_id": decision["predecessor"]["result_id"],
                    "raw_result_id": receipt["result_id"], "state": state,
                    "detail_sha256": hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest(),
                }},
            )
            path = _publish(bundle, decision["spool"])
            repository.append_admission_decision(
                decision["predecessor"], bundle, path, state, detail,
                in_transaction=True,
            )
            return
        repository.finish(receipt, state, detail=json.dumps(detail, sort_keys=True)
                          if isinstance(detail, dict) else detail, in_transaction=True)


def _refuse_rolled_back_admission(conn, receipt, reason):
    """Record refusal only; a failed replacement must not retire baseline rows."""
    conn.execute("BEGIN TRANSACTION")
    try:
        _finish_receipt(conn, receipt, "needs_review", {"reason": reason})
        conn.execute("COMMIT")
    except Exception:
        cleanup_ocr_resource(lambda: conn.execute("ROLLBACK"))
        raise


def _receipt_replay(conn, receipt, doc_id, year, artifact_sha256, ingestion_generation, parser_version):
    """Validate mutation scope and stop replay before touching rows or history."""
    if receipt is None:
        return None
    from analyzer.persistence.work_repository import WorkRepository
    repository = WorkRepository(conn)
    work = repository.validate_receipt(receipt)
    if (str(doc_id), year, artifact_sha256, ingestion_generation) != (
        work.doc_id, work.archive_year, work.artifact_sha256, work.generation_id,
    ):
        raise ValueError("OCR mutation scope does not match result receipt")
    if not repository.is_finished(receipt):
        return None
    if "admission_decision" in receipt:
        return None  # New decision; the finished extraction receipt stays immutable.
    state, detail = conn.execute(
        "SELECT state, detail FROM extraction_results WHERE result_id = ?",
        [receipt["result_id"]],
    ).fetchone()
    run = conn.execute(
        "SELECT status, raw_row_count, transaction_count FROM pdf_parse_runs "
        "WHERE doc_id = ? AND parser_version = ? AND artifact_sha256 = ? "
        "AND ingestion_generation = ? ORDER BY parsed_at DESC LIMIT 1",
        [str(doc_id), parser_version, artifact_sha256, ingestion_generation],
    ).fetchone()
    if run and ((state == "accepted" and run[0] == "success") or
                (state == "verified_empty" and run[0] == "no_txs")):
        return InsertOutcome(status=run[0], inserted=run[2], expected=run[1], committed=True, detail=detail)
    return InsertOutcome(status="already_finished", detail=state)


def insert_outcome_from_report(
    report: dict | None, *, inserted: int, expected: int
) -> InsertOutcome:
    """The InsertOutcome for an insert_transactions call.

    Reads the ``report`` dict the call was given; if the callee ignored it
    (monkeypatched fakes, older callers), fall back to a conservative
    outcome derived from the committed count alone: a positive count is a
    committed but UNVERIFIED full insert only when it matches ``expected``,
    and zero is an error — never silently "complete".
    """
    outcome = (report or {}).get("outcome")
    if isinstance(outcome, InsertOutcome):
        return outcome
    if (report or {}).get("outcome") is not None:
        return InsertOutcome(
            status="success" if inserted == expected and inserted else "error",
            inserted=inserted,
            expected=expected,
            committed=bool(inserted),
        )
    if expected == 0 and inserted == 0:
        # Empty input committed nothing by definition; only the callee knows
        # whether a no_txs run was recorded, so treat it as unverified.
        return InsertOutcome(status="error", inserted=0, expected=0)
    if inserted and inserted == expected:
        return InsertOutcome(
            status="success", inserted=inserted, expected=expected, committed=True
        )
    return InsertOutcome(
        status="error", inserted=inserted, expected=expected, committed=bool(inserted)
    )


def cleanup_ocr_resource(cleanup) -> None:
    """Surface cleanup failures unless another exception already owns the failure."""
    import sys

    active_error = sys.exc_info()[1]
    try:
        cleanup()
    except Exception:
        if active_error is None:
            raise
        logging.getLogger(__name__).exception("Secondary OCR cleanup failure")


def _insert_transactions(
    conn,
    doc_id,
    year,
    member,
    transactions,
    *,
    parser_version: str = GEMINI_PARSER_VERSION,
    raw_count: int | None = None,
    artifact_sha256: str | None = None,
    ingestion_generation: str | None = None,
    engine_model: str = MODEL,
    checkpoint: bool = True,
    force_partial_canonical: bool = False,
    receipt=None,
    evidence=None,
    admission=None,
) -> InsertOutcome:
    """Validate and dispatch a generation-bound rowset; partials stay blocked."""

    replay = _receipt_replay(conn, receipt, doc_id, year, artifact_sha256, ingestion_generation, parser_version)
    if replay is not None:
        return replay
    from analyzer.ocr.review import has_review_authority
    if not has_review_authority(admission):
        return InsertOutcome(status="needs_review", detail="grounded API source review required")
    from analyzer.ocr.normalization import NORMALIZATION_RECIPE
    from analyzer.ocr.review import digest
    if (receipt is None or admission["binding"]["raw_result_id"] != receipt["result_id"]
            or admission["binding"]["normalization_recipe"] != NORMALIZATION_RECIPE
            or admission["candidate"]["rows"] != transactions
            or digest(admission["projection"]) != admission["binding"]["final_projection_sha256"]):
        raise ValueError("admission does not bind this receipt, normalization and final rowset")
    require_ocr_schema(conn)
    if not artifact_sha256 or not ingestion_generation:
        raise RuntimeError(
            "OCR insertion requires artifact_sha256 and ingestion_generation"
        )
    input_expected = raw_count if raw_count is not None else len(transactions)
    if (force_partial_canonical or input_expected != len(transactions)
            or input_expected != len(admission["projection"])):
        raise ValueError("review authorizes only the complete unchanged rowset; partial forcing is forbidden")
    bound_generation = _resolve_ingestion_generation(conn, doc_id, year, artifact_sha256)
    if bound_generation != ingestion_generation:
        raise RuntimeError("OCR insertion generation does not match the current artifact")
    metadata = conn.execute(
        "SELECT filing_date, first_name, last_name FROM house_generation_metadata "
        "WHERE archive_year = ? AND generation_id = ? AND doc_id = ? AND filing_type = 'P'",
        [year, ingestion_generation, str(doc_id)],
    ).fetchone()
    if metadata is None:
        raise RuntimeError("OCR insertion requires generation-bound filing metadata")
    filing_date = metadata[0]
    expected_member = " ".join(part for part in metadata[1:] if part).strip()
    bound_work = admission["binding"]["work"]
    from analyzer.ocr.common import _strict_date
    if (_strict_date(filing_date).isoformat(), expected_member) != (bound_work["filing_date"], bound_work["member"]):
        raise ValueError("review metadata no longer matches the official generation")
    _, identity_rejections = validate_transactions(doc_id, member, [], filing_date, expected_member)
    if identity_rejections:
        record_failed_ocr_attempt(
            conn, doc_id, year, raw_count or len(transactions),
            json.dumps(identity_rejections, sort_keys=True),
            parser_version=parser_version, artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation, engine_model=engine_model,
            receipt=receipt,
        )
        # Member-level rejection: a no_txs-looking zero is NOT a no_txs.
        return InsertOutcome(
            status="rejected", inserted=0, expected=input_expected,
            rejections=dict(identity_rejections),
        )
    if not transactions:
        if receipt is not None and not raw_count:
            from analyzer.persistence.work_repository import WorkRepository
            work = WorkRepository(conn).validate_receipt(receipt)
            if not admission["review_binding"] or work.page_count != bound_work["page_count"]:
                raise ValueError("verified empty admission requires explicit evidence for every artifact page")
        status = "error" if raw_count else "no_txs"
        message = "semantic_zero_after_raw_rows" if raw_count else ""
        if raw_count:
            record_failed_ocr_attempt(
                conn, doc_id, year, raw_count, message,
                parser_version=parser_version,
                artifact_sha256=artifact_sha256,
                ingestion_generation=ingestion_generation,
                engine_model=engine_model,
                receipt=receipt,
            )
            return InsertOutcome(
                status="rejected", inserted=0, expected=input_expected, detail=message,
            )
        conn.execute("BEGIN TRANSACTION")
        try:
            if status == "no_txs":
                archive_rows(conn, "transactions",
                              "source = 'gemini_ocr' AND doc_id = ? "
                              "AND (chamber = 'House' OR chamber IS NULL)", [str(doc_id)])
                conn.execute(
                    "DELETE FROM transactions WHERE source = 'gemini_ocr' "
                    "AND doc_id = ? AND (chamber = 'House' OR chamber IS NULL)",
                    [str(doc_id)],
                )
            record_parse_run(
                conn,
                doc_id,
                year,
                status,
                raw_count or 0,
                0,
                message,
                parser_version=parser_version,
                artifact_sha256=artifact_sha256,
                ingestion_generation=ingestion_generation,
                engines_attempted=engine_model,
            )
            detail = _verified_final_admission(conn, doc_id, admission)
            _finish_receipt(conn, receipt, "verified_empty", detail)
            conn.execute("COMMIT")
        except Exception as exc:
            cleanup_ocr_resource(lambda: conn.execute("ROLLBACK"))
            _refuse_rolled_back_admission(conn, receipt, str(exc))
            raise
        return InsertOutcome(
            status="no_txs", inserted=0, expected=0, committed=True
        )

    # Persist the exact projection already normalized under the review lock.
    return _replace_accepted_ocr_rows(
        conn, doc_id, year, admission["projection"], input_expected,
        parser_version=parser_version,
        artifact_sha256=artifact_sha256, ingestion_generation=ingestion_generation,
        engine_model=engine_model, checkpoint=checkpoint,
        receipt=receipt,
        admission=admission,
    )


def _normalize_ocr_rows(
    doc_id, member, transactions, filing_date, *, artifact_sha256, ingestion_generation,
) -> tuple[list[dict], list[str]]:
    """Build persistence rows without making acceptance or database decisions."""
    errors = []
    rows: list[dict] = []
    for source_row_number, tx in enumerate(transactions, start=1):
        try:
            from analyzer.ocr.common import _strict_date
            parsed_date = _strict_date(tx["date"])
            parsed_notification = _strict_date(tx.get("notif_date"))
            tx_date = parsed_date.isoformat() if parsed_date else None
            notification_date = parsed_notification.isoformat() if parsed_notification else None
            disclosure_date = filing_date or notification_date or tx_date
            disclosed_ticker = extract_ticker(tx["asset"], explicit_only=True)
            ticker_candidate = None
            if disclosed_ticker:
                ticker = disclosed_ticker
                raw_ticker = disclosed_ticker
                ticker_origin = "official"
            elif ticker := extract_ticker(tx["asset"]):
                raw_ticker = None
                ticker_origin = "asset_description"
            else:
                ticker_candidate = resolve_ticker(tx["asset"])
                ticker = None
                raw_ticker = ticker_candidate
                ticker_origin = "unverified" if ticker_candidate else "not_reported"
            if not tx_date:
                errors.append(f"bad transaction date: {tx['date']}")
                continue

            values = {
                "doc_id": str(doc_id),
                "member": tx.get("member") or member,
                "ticker": ticker,
                "transaction_date": tx_date,
                "disclosure_date": disclosure_date,
                "transaction_type": tx["type"],
                "amount_raw": tx.get("amount_raw", tx.get("amount_letter")),
                "amount_midpoint": tx["amount_midpoint"],
                "owner_code": tx.get("owner_code"),
                "asset_description": tx["asset"][:500],
                "source": "gemini_ocr",
                **typed_instrument_fields(tx),
                "expiry_date": tx.get("expiry_date"),
            }
            page_number = tx.get("page_number")
            row_identity = (
                f"{doc_id}:page:{page_number}:row:{source_row_number}"
                if page_number is not None
                else f"{doc_id}:row:{source_row_number}"
            )
            authoritative_provenance = {
                "chamber": "House",
                "source_record_id": str(doc_id),
                "source_row_id": row_identity,
                "official_filing_date": filing_date,
                "available_date": filing_date,
                "notification_date": notification_date,
                "raw_transaction_subtype": tx["type"],
                "ticker_origin": ticker_origin,
                "raw_ticker": raw_ticker,
                "ticker_candidate": ticker_candidate,
                "raw_asset_class": tx.get("raw_asset_class", tx.get("raw_instrument_type", tx.get("instrument_type")) or "Not separately reported"),
                "raw_owner": tx.get("raw_owner", tx.get("owner")),
                "raw_asset_description": tx["asset"][:500],
                "ingestion_generation": ingestion_generation,
                "artifact_sha256": artifact_sha256,
            }
            values.update(authoritative_provenance)
            row = {column: values.get(column) for column in OCR_INSERT_COLUMNS}
            validate_ticker_provenance(row)
            rows.append(row)
        except Exception as exc:
            errors.append(f"{tx.get('asset', '?')}: {exc}")
    return rows, errors


def canonical_ocr_projection(rows):
    """Exact SQL-typed business columns, preserving ordered row occurrences."""
    from analyzer.ocr.common import _strict_date

    date_columns = {"transaction_date", "disclosure_date", "official_filing_date",
                    "available_date", "notification_date"}
    projected = []
    for row in rows:
        item = {}
        for column in OCR_INSERT_COLUMNS:
            value = row.get(column)
            if value is None:
                item[column] = None
            elif column in date_columns:
                parsed = _strict_date(value)
                if parsed is None:
                    raise ValueError(f"invalid final date: {column}")
                item[column] = parsed.isoformat()
            elif column in {"amount_midpoint", "strike_price"}:
                item[column] = float(value)
            elif column == "acceptance_datetime":
                item[column] = value.isoformat() if hasattr(value, "isoformat") else str(value)
            else:
                item[column] = str(value)
        projected.append(item)
    return projected


def final_ocr_projection(work, normalized_rows):
    """Pure projection used by frozen review and the canonical insert alike."""
    rows, errors = _normalize_ocr_rows(
        work.doc_id, work.member, normalized_rows, work.filing_date,
        artifact_sha256=work.artifact_sha256, ingestion_generation=work.generation_id,
    )
    if errors:
        raise ValueError("invalid persistence projection: " + "; ".join(errors))
    return canonical_ocr_projection(rows)


def _verified_final_admission(conn, doc_id, admission):
    """Inside the row transaction, compare actual scoped SQL rows before receipt."""
    from analyzer.ocr.review import digest

    columns = {row[1] for row in conn.execute("PRAGMA table_info('transactions')").fetchall()}
    if columns != set(OCR_INSERT_COLUMNS) | {"id", "created_at"}:
        raise ValueError("transaction columns changed; update the final review projection")
    stored = conn.execute(
        f"SELECT {', '.join(OCR_INSERT_COLUMNS)} FROM transactions "
        "WHERE source = 'gemini_ocr' AND doc_id = ? AND (chamber = 'House' OR chamber IS NULL)",
        [str(doc_id)],
    ).fetchall()
    actual = canonical_ocr_projection([dict(zip(OCR_INSERT_COLUMNS, row)) for row in stored])
    expected = admission["projection"]
    by_identity = lambda row: row["source_row_id"]
    if sorted(actual, key=by_identity) != sorted(expected, key=by_identity):
        raise ValueError("final SQL rows differ from the source-reviewed projection")
    # Store the reviewed occurrence order, after comparison with actual SQL values.
    indexed = {row["source_row_id"]: row for row in actual}
    actual = [indexed[row["source_row_id"]] for row in expected]
    if len(indexed) != len(expected) or digest(actual) != admission["binding"]["final_projection_sha256"]:
        raise ValueError("final SQL row count or digest differs from review")
    detail = {"schema": "house-api-final-admission-v1",
            "normalization_recipe": admission["binding"]["normalization_recipe"],
            "policy_identity": admission["binding"]["policy_identity"],
            "policy_sha256": admission["binding"]["policy_sha256"],
            "raw_result_id": admission["binding"]["raw_result_id"],
            "final_projection": actual, "final_projection_sha256": digest(actual),
            "final_projection_count": len(actual), "review_binding": admission["review_binding"],
             "audit_binding": admission.get("audit_binding")}
    if admission.get("source_proposal") is not None:
        detail["source_proposal"] = admission["source_proposal"]
        detail["source_review_lineage"] = admission["source_review_lineage"]
    return detail


def _replace_accepted_ocr_rows(
    conn, doc_id, year, rows, input_count, *,
    parser_version, artifact_sha256, ingestion_generation,
    engine_model, checkpoint, receipt=None, admission=None,
) -> InsertOutcome:
    """Own accepted-rowset replacement and historical repair in one transaction."""
    count = 0
    committed = False
    staging_table = "staging_gemini_ocr_transactions"
    try:
        conn.execute(f"DROP TABLE IF EXISTS {staging_table}")
        conn.execute(
            f"CREATE TEMP TABLE {staging_table} AS "
            "SELECT * FROM transactions WHERE FALSE"
        )
        for row in rows:
            columns = list(row)
            placeholders = ", ".join("?" for _ in columns)
            conn.execute(
                f"INSERT INTO {staging_table} ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                [row[column] for column in columns],
            )
        columns = OCR_INSERT_COLUMNS
        conn.execute("BEGIN TRANSACTION")
        invalid_house_rows = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM transactions
                WHERE source = 'house_pdf'
                  AND doc_id = ?
                  AND ingestion_generation = ?
                  AND artifact_sha256 = ?
                  AND (
                      transaction_date IS NULL
                      OR disclosure_date IS NULL
                      OR transaction_date > disclosure_date
                      OR (
                          notification_date IS NOT NULL
                          AND notification_date < transaction_date
                      )
                      OR chamber IS NULL OR TRIM(chamber) = ''
                      OR source_record_id IS NULL OR TRIM(source_record_id) = ''
                      OR source_row_id IS NULL OR TRIM(source_row_id) = ''
                      OR official_filing_date IS NULL
                  )
                """,
                [str(doc_id), ingestion_generation, artifact_sha256],
            ).fetchone()[0]
        )
        if invalid_house_rows:
            archive_rows(conn, "transactions",
                          "source = 'house_pdf' AND doc_id = ? "
                          "AND ingestion_generation = ? AND artifact_sha256 = ?",
                          [str(doc_id), ingestion_generation, artifact_sha256])
            # Only non-OCR terminal runs are invalidated below.
            archive_rows(conn, "pdf_parse_runs",
                          "doc_id = ? AND ingestion_generation = ? AND artifact_sha256 = ? "
                          "AND NOT (LOWER(COALESCE(parser_version, '')) LIKE '%gemini%' "
                          "OR LOWER(COALESCE(parser_version, '')) LIKE '%ling%' "
                          "OR LOWER(COALESCE(parser_version, '')) LIKE '%gpt%' "
                          "OR LOWER(COALESCE(parser_version, '')) LIKE '%openrouter%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini/%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'openrouter/%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gpt-%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini+%' "
                          "OR LOWER(COALESCE(engines_attempted, '')) LIKE 'visionb%') "
                          "AND status IN ('success', 'no_txs')",
                          [str(doc_id), ingestion_generation, artifact_sha256])
            conn.execute(
                """
                DELETE FROM transactions
                WHERE source = 'house_pdf'
                  AND doc_id = ?
                  AND ingestion_generation = ?
                  AND artifact_sha256 = ?
                """,
                [str(doc_id), ingestion_generation, artifact_sha256],
            )
            conn.execute(
                """
                UPDATE pdf_parse_runs
                SET status = 'invalidated', transaction_count = 0, parsed_at = ?,
                    error_message = 'superseded by validated gemini_ocr fallback after semantic validation failure'
                WHERE doc_id = ?
                  AND ingestion_generation = ?
                  AND artifact_sha256 = ?
                  AND NOT (
                      LOWER(COALESCE(parser_version, '')) LIKE '%gemini%'
                      OR LOWER(COALESCE(parser_version, '')) LIKE '%ling%'
                      OR LOWER(COALESCE(parser_version, '')) LIKE '%gpt%'
                      OR LOWER(COALESCE(parser_version, '')) LIKE '%openrouter%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini/%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'openrouter/%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gpt-%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'gemini+%'
                      OR LOWER(COALESCE(engines_attempted, '')) LIKE 'visionb%'
                  )
                  AND status IN ('success', 'no_txs')
                """,
                [datetime.datetime.now(), str(doc_id), ingestion_generation, artifact_sha256],
            )
        identity_columns = [
            "source",
            "chamber",
            "source_record_id",
            "source_row_id",
            "ingestion_generation",
        ]
        mutable_columns = [
            column for column in columns if column not in identity_columns
        ]
        identity_join = " AND ".join(
            f"target.{column} = staged.{column}" for column in identity_columns
        )
        updates = ", ".join(f"{column} = staged.{column}" for column in mutable_columns)
        archive_rows(conn, "transactions",
                      f"EXISTS (SELECT 1 FROM {staging_table} AS staged WHERE "
                      + identity_join.replace("target.", "transactions.") + ")", [])
        conn.execute(
            f"UPDATE transactions AS target SET {updates}, created_at = ? "
            f"FROM {staging_table} AS staged WHERE {identity_join}", [datetime.datetime.now()]
        )
        conn.execute(
            f"INSERT INTO transactions ({', '.join(columns)}) "
            f"SELECT {', '.join(f'staged.{column}' for column in columns)} "
            f"FROM {staging_table} AS staged WHERE NOT EXISTS ("
            f"SELECT 1 FROM transactions AS target WHERE {identity_join})"
        )
        stale_identity_join = " AND ".join(
            f"staged.{column} = target.{column}" for column in identity_columns
        )
        archive_rows(conn, "transactions",
                      "source = 'gemini_ocr' AND doc_id = ? "
                      "AND (chamber = 'House' OR chamber IS NULL) "
                      f"AND NOT EXISTS (SELECT 1 FROM {staging_table} AS staged WHERE "
                      + stale_identity_join.replace("target.", "transactions.") + ")", [str(doc_id)])
        conn.execute(
            "DELETE FROM transactions AS target "
            "WHERE target.source = 'gemini_ocr' AND target.doc_id = ? "
            "AND (target.chamber = 'House' OR target.chamber IS NULL) "
            f"AND NOT EXISTS (SELECT 1 FROM {staging_table} AS staged "
            f"WHERE {stale_identity_join})",
            [str(doc_id)],
        )
        count = len(rows)
        record_parse_run(
            conn,
            doc_id,
            year,
            "success",
            input_count,
            count,
            "",
            parser_version=parser_version,
            artifact_sha256=artifact_sha256,
            ingestion_generation=ingestion_generation,
            engines_attempted=engine_model,
        )
        detail = _verified_final_admission(conn, doc_id, admission)
        _finish_receipt(conn, receipt, "accepted", detail)
        conn.execute("COMMIT")
        committed = True
        if checkpoint:
            conn.execute("CHECKPOINT")
        return InsertOutcome(
            status="success",
            inserted=count,
            expected=input_count,
            committed=True,
        )
    except Exception as exc:
        if not committed:
            cleanup_ocr_resource(lambda: conn.execute("ROLLBACK"))
            _refuse_rolled_back_admission(conn, receipt, str(exc))
        raise
    finally:
        cleanup_ocr_resource(lambda: conn.execute(f"DROP TABLE IF EXISTS {staging_table}"))


def insert_transactions_outcome(
    doc_id, year, member, transactions, *, db_path: str = DB_PATH,
    parser_version: str = GEMINI_PARSER_VERSION, raw_count: int | None = None,
    artifact_sha256: str | None = None, ingestion_generation: str | None = None,
    engine_model: str = MODEL, checkpoint: bool = True,
    force_partial_canonical: bool = False,
    conn=None, receipt=None, evidence=None, admission=None,
) -> InsertOutcome:
    """Return the committed outcome; partials are evidence-only unless forced.

    The explicit force override retains an incomplete outcome and warning.
    Every branch shares this connection lifetime, including schema failures.
    """
    from analyzer.ocr.review import has_review_authority
    if not has_review_authority(admission):
        return InsertOutcome(status="needs_review", detail="grounded API source review required")
    with ExitStack() as resources:
        if conn is None:
            from analyzer.persistence.ownership import database_owner
            resources.enter_context(database_owner(db_path, "standalone OCR insertion"))
            conn = duckdb.connect(db_path)
            resources.callback(lambda: cleanup_ocr_resource(conn.close))
        return _insert_transactions(
            conn, doc_id, year, member, transactions,
            parser_version=parser_version, raw_count=raw_count,
            artifact_sha256=artifact_sha256, ingestion_generation=ingestion_generation,
            engine_model=engine_model, checkpoint=checkpoint,
            force_partial_canonical=force_partial_canonical,
            receipt=receipt, evidence=evidence, admission=admission,
        )


def insert_transactions(doc_id, year, member, transactions, *, report=None, **kwargs):
    """Legacy integer/report adapter; new callers use insert_transactions_outcome."""
    outcome = insert_transactions_outcome(doc_id, year, member, transactions, **kwargs)
    if report is not None:
        report["outcome"] = outcome
    return outcome.inserted


def validate_for_insert(conn, doc_id, member, transactions):
    filing_date = get_filing_date(conn, doc_id)
    expected_member = get_metadata_member(conn, doc_id)
    return validate_transactions(
        doc_id, member, transactions, filing_date, expected_member
    )
