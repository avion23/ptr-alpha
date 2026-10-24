"""OCR batch orchestration; persistence belongs to ingestion."""
import argparse
import os
import time
from dataclasses import replace
from pathlib import Path
import duckdb
from analyzer.ocr.paths import DATA_DIR
from analyzer.vision_engines import (
    DualReadDoc,
    UNRESOLVED,
    VisionBConfig,
    VisionBUnavailable,
    dual_read_enabled,
    read_document_with_engine_b,
    run_dual_read_for_doc,
)
from analyzer.ocr.common import (
    ArtifactMetadata,
    GEMINI_PARSER_VERSION,
    MODEL,
    GeminiOutputError,
    ParsedGeminiOutput,
    call_gemini,
    is_quota_error,
    parse_gemini_output,
    snapshot_pdf,
)
from analyzer.ocr.ingestion import (
    COOLDOWN,
    DB_PATH,
    OCR_ENGINE_ENV,
    WORK_MANIFEST_DEFAULT_PATH,
    export_work_manifest,
    get_ocr_work_items,
    get_zero_row_pdfs,
    insert_transactions_outcome,
    load_progress,
    mark_progress,
    record_failed_ocr_attempt,
    resolve_ingestion_generation,
    save_progress
)


def _require_successful_response_metadata(
    metadata: ArtifactMetadata | None,
) -> ParsedGeminiOutput:
    if metadata is None:
        raise RuntimeError("successful Gemini response requires artifact metadata")
    parsed = metadata.parsed
    if parsed is None:
        raise RuntimeError("successful Gemini response requires validated output")
    return parsed


def _ocr_engine_mode(env=None) -> str:
    """Resolve the nightly drain's engine mode: 'dual', 'visionb', 'gemini'.

    Precedence: DUAL_READ=1 forces dual regardless of OCR_ENGINE; otherwise
    OCR_ENGINE selects — 'visionb' (GLM-primary single engine), 'dual'
    (alias of DUAL_READ=1), or unset/'gemini' (the historical Gemini-only
    drain, the default). Anything else is a typo and must fail loudly rather
    than silently drain the backlog with the wrong engine's quota.
    """
    source = os.environ if env is None else env
    if dual_read_enabled(source):
        return "dual"
    engine = str(source.get(OCR_ENGINE_ENV) or "").strip().lower()
    if engine in ("", "gemini"):
        return "gemini"
    if engine in ("visionb", "dual"):
        return engine
    raise ValueError(
        f"unsupported {OCR_ENGINE_ENV}: {engine!r} "
        "(expected 'gemini', 'visionb', or 'dual')"
    )


def call_vision_b(
    pdf_path: str,
    doc_id: str | None = None,
    *,
    config: VisionBConfig | None = None,
    page_images=None,
) -> tuple[str | None, str, ArtifactMetadata | None]:
    """Engine-B-primary read with the call_gemini contract.

    Hashes and page-counts the same immutable snapshot bytes the artifact
    binding is checked against, asks engine B once with the shared extraction
    prompt, and validates the completion with the shared parser. No response
    cache. Programmatic helper only: the OCR_ENGINE=visionb batch drain now
    routes through ``scripts.glm_stage``'s staging functions (canonical
    ingestion of GLM rows happens ONLY via ``glm_stage promote``), so the
    resume point for that drain is the staging record, not a parse run.
    Transport, configuration, and schema failures return
    (None, error, metadata) — the batch-level quota stop is the caller's
    is_quota_error check.
    """
    metadata = None
    try:
        with snapshot_pdf(pdf_path) as snapshot:
            metadata = ArtifactMetadata(
                snapshot.sha256,
                snapshot.page_count,
                snapshot.byte_count,
                str(Path(pdf_path)),
            )
            doc = DualReadDoc(
                str(doc_id or Path(pdf_path).stem), 0, str(snapshot.path)
            )
            try:
                output = read_document_with_engine_b(
                    doc,
                    config=config,
                    page_images=page_images,
                    page_count=snapshot.page_count,
                )
            except VisionBUnavailable as exc:
                return None, str(exc), metadata
            try:
                parsed = parse_gemini_output(
                    output, expected_page_count=snapshot.page_count
                )
            except GeminiOutputError as exc:
                return None, f"invalid_response: {exc}", metadata
            return output, "", replace(metadata, parsed=parsed)
    except Exception as exc:
        return None, str(exc), metadata


def main(*, staging_runner=None) -> int:
    parser = argparse.ArgumentParser(
        description="OCR zero-row House PTR PDFs with Gemini"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached Gemini responses"
    )
    parser.add_argument(
        "--force-partial-canonical",
        action="store_true",
        help=(
            "OWNER OVERRIDE: commit partially validated Gemini OCR rows to "
            "canonical (prints a warning and keeps the attempt incomplete)"
        ),
    )
    parser.add_argument(
        "--export-work-manifest",
        nargs="?",
        const=WORK_MANIFEST_DEFAULT_PATH,
        default=None,
        metavar="PATH",
        help=(
            "export get_ocr_work_items (the authoritative unresolved queue) "
            "as an artifact-bound JSON manifest and exit, without draining "
            f"(default path: {WORK_MANIFEST_DEFAULT_PATH}; consumed by "
            "glm_stage.py stage --from-manifest)"
        ),
    )
    args = parser.parse_args()

    if args.export_work_manifest:
        # Mode-independent: exporting the queue must work even when the
        # engine-B env is unusable (the cron wrapper exports BEFORE the
        # drain; a failure there means canonical is unavailable).
        envelope = export_work_manifest(
            args.export_work_manifest,
            db_path=DB_PATH,
            data_dir=os.path.dirname(DB_PATH) or ".",
        )
        print(
            f"work manifest {envelope['schema']}: {envelope['item_count']} "
            f"unresolved docs -> {args.export_work_manifest}"
        )
        return 0

    mode = _ocr_engine_mode()
    if args.force_partial_canonical and mode != "gemini":
        print(
            "ERROR: --force-partial-canonical is supported only in Gemini "
            "single-engine mode; refusing this run.",
            flush=True,
        )
        print("\n=== OCR RUN INCOMPLETE ===", flush=True)
        return 1
    vision_b_config = None
    parser_version, engine_model = GEMINI_PARSER_VERSION, MODEL
    if mode == "visionb":
        # GLM-primary mode must fail closed on an unusable engine-B setup:
        # silently draining with engine A would spend the scarce Gemini
        # quota this mode exists to conserve.
        try:
            vision_b_config = VisionBConfig.from_env()
        except VisionBUnavailable as exc:
            print(f"  visionb unusable: {exc}; stopping batch", flush=True)
            print("\n=== OCR RUN INCOMPLETE ===", flush=True)
            return 1
        if vision_b_config is None:
            print(
                "  visionb requires VISION_B_BASE_URL and VISION_B_MODEL; "
                "stopping batch",
                flush=True,
            )
            print("\n=== OCR RUN INCOMPLETE ===", flush=True)
            return 1
    if mode == "dual":
        # A dual run without a usable engine B would be unresolved for every
        # doc in the batch: alert (config defect) and stop before spending
        # engine A on readings nothing may persist.
        try:
            dual_config = VisionBConfig.from_env()
        except VisionBUnavailable as exc:
            print(
                f"ALERT: dual read requested but engine B is unusable: {exc}; "
                "stopping batch",
                flush=True,
            )
            print("\n=== OCR RUN INCOMPLETE ===", flush=True)
            return 1
        if dual_config is None:
            print(
                "ALERT: dual read requested but VISION_B_BASE_URL/"
                "VISION_B_MODEL are not configured; stopping batch",
                flush=True,
            )
            print("\n=== OCR RUN INCOMPLETE ===", flush=True)
            return 1

    pdfs = get_zero_row_pdfs()
    progress = load_progress()
    remaining = pdfs
    print(f"Current unresolved OCR work: {len(remaining)}")
    if mode == "visionb":
        if staging_runner is None:
            raise RuntimeError("visionb staging requires scripts/ocr_zero_rows.py; use glm_stage promote after audit")
        staging_failed = bool(
            staging_runner(remaining, vision_b_config, data_dir=Path(DB_PATH).parent)
        )
        print(
            "\n=== OCR RUN INCOMPLETE ==="
            if staging_failed
            else "\n=== OCR RUN FINISHED ===",
            flush=True,
        )
        print(
            "Total inserted: 0 (visionb mode stages only; canonical "
            "ingestion of GLM rows happens via glm_stage promote)",
            flush=True,
        )
        return 1 if staging_failed else 0

    total_inserted = 0
    run_failed = False
    for i, (doc_id, year, path) in enumerate(remaining):
        idx = i + 1
        print(f"\n[{idx}/{len(remaining)}] {doc_id} ({year})...", flush=True)

        time.sleep(COOLDOWN)

        if mode == "dual":
            # Opt-in dual vision read (DUAL_READ=1 / OCR_ENGINE=dual, engine
            # B configured): engine A (Gemini) is gated against engine B per
            # doc. AGREE and DISAGREE are persisted or quarantined by the
            # orchestrator (per-doc commit inside insert_transactions/
            # record_disagreement); engine-B absence is UNRESOLVED — nothing
            # persists A-only, quota stops the batch, other outages land an
            # error run that retries the doc next night.
            outcome = run_dual_read_for_doc(
                DualReadDoc(
                    str(doc_id),
                    year,
                    path,
                    data_dir=os.path.dirname(DB_PATH) or ".",
                    db_path=DB_PATH,
                ),
                refresh=args.refresh,
            )
            total_inserted += outcome.inserted
            if outcome.handled:
                mark_progress(progress, doc_id, outcome.progress_key)
                save_progress(progress)
                if outcome.progress_key == "errors":
                    run_failed = True
                print(
                    f"  DUAL {outcome.decision.status}: inserted={outcome.inserted}",
                    flush=True,
                )
                continue
            if outcome.decision.status == UNRESOLVED:
                # Engine-B quota stop (unhandled == nothing persisted).
                print("  Engine B quota exhausted; stopping batch", flush=True)
                run_failed = True
                break
            output, error, artifact_metadata = (
                outcome.output,
                outcome.error,
                outcome.metadata,
            )
        else:
            output, error, artifact_metadata = call_gemini(
                path, doc_id=doc_id, refresh=args.refresh
            )
        # A_ERROR passthrough carries error=None only when engine A
        # succeeded; the single-engine path always has a string here.
        if is_quota_error(error or ""):
            print("  Quota exhausted; stopping batch", flush=True)
            run_failed = True
            break
        ingestion_generation = None
        if artifact_metadata is not None:
            try:
                ingestion_generation = resolve_ingestion_generation(
                    DB_PATH, doc_id, year, artifact_metadata.sha256
                )
            except RuntimeError as exc:
                output, error = None, str(exc)
        if output is None or error:
            conn = duckdb.connect(DB_PATH)
            try:
                record_failed_ocr_attempt(
                    conn,
                    doc_id,
                    year,
                    0,
                    error,
                    parser_version=parser_version,
                    engine_model=engine_model,
                    artifact_sha256=(
                        artifact_metadata.sha256 if artifact_metadata else None
                    ),
                    ingestion_generation=ingestion_generation,
                )
            finally:
                conn.close()
            mark_progress(progress, doc_id, "errors")
            save_progress(progress)
            run_failed = True
            print(f"  ERROR: {error}", flush=True)
            continue

        parsed = _require_successful_response_metadata(artifact_metadata)
        member, transactions = parsed.member, parsed.transactions
        insert_outcome = None
        if not transactions:
            insert_outcome = insert_transactions_outcome(
                doc_id,
                year,
                member,
                [],
                db_path=DB_PATH,
                raw_count=0,
                parser_version=parser_version,
                engine_model=engine_model,
                artifact_sha256=artifact_metadata.sha256,
                ingestion_generation=ingestion_generation,
            )
            # A member rejection is not a no_txs: only a committed no_txs
            # run may close the doc as "no transactions found".
            if insert_outcome.complete:
                mark_progress(progress, doc_id, "no_txs")
                save_progress(progress)
                print("  No transactions found", flush=True)
            else:
                mark_progress(progress, doc_id, "errors")
                save_progress(progress)
                run_failed = True
                print(
                    "  ERROR: no-transaction run not committed "
                    f"({insert_outcome.status})",
                    flush=True,
                )
            continue

        raw_count = len(transactions)
        insert_outcome = insert_transactions_outcome(
            doc_id,
            year,
            member,
            transactions,
            db_path=DB_PATH,
            raw_count=raw_count,
            parser_version=parser_version,
            engine_model=engine_model,
            artifact_sha256=artifact_metadata.sha256,
            ingestion_generation=ingestion_generation,
            force_partial_canonical=args.force_partial_canonical,
        )
        total_inserted += insert_outcome.inserted
        # Partial or rejected output is never "complete": mark errors so the
        # operator sees it, mirroring the dual path.
        mark_progress(
            progress,
            doc_id,
            "success" if insert_outcome.complete else "errors",
        )
        save_progress(progress)
        print(f"  Member: {member}", flush=True)
        print(
            f"  Inserted: {insert_outcome.inserted}/{raw_count} transactions "
            f"(total: {total_inserted})",
            flush=True,
        )
        if not insert_outcome.complete:
            run_failed = True
            print(
                f"  ERROR: insert incomplete ({insert_outcome.status}); "
                "partial output is not marked complete",
                flush=True,
            )
        for tx in transactions[:3]:
            print(
                f"    {tx['asset']} | {tx['type']} | {tx['date']} | {tx['amount_letter']}",
                flush=True,
            )

    print(
        "\n=== OCR RUN INCOMPLETE ==="
        if run_failed
        else "\n=== OCR RUN FINISHED ===",
        flush=True,
    )
    print(f"Total inserted: {total_inserted}", flush=True)
    return 1 if run_failed else 0


def run_gemini_ocr_for_year(
    year: int,
    data_dir: str = str(DATA_DIR),
    refresh: bool = False,
    force_partial_canonical: bool = False,
):
    """Process unresolved deterministic OCR candidates for one year."""
    if force_partial_canonical and dual_read_enabled():
        print(
            "ERROR: force_partial_canonical is supported only in Gemini "
            "single-engine mode; refusing this run.",
            flush=True,
        )
        return 0
    db_path = os.path.join(data_dir, "congress.duckdb")
    progress_path = Path(data_dir) / "ocr_progress_gemini_manual.json"
    remaining = get_ocr_work_items(db_path=db_path, data_dir=data_dir, year=year)
    print(f"Current unresolved OCR work for {year}: {len(remaining)}")
    if dual_read_enabled():
        # Same contract as main()'s dual mode: a dual run without a usable
        # engine B is unresolved for every doc — alert the config defect and
        # stop rather than draining A-only.
        try:
            dual_config = VisionBConfig.from_env()
        except VisionBUnavailable as exc:
            print(
                f"ALERT: dual read requested but engine B is unusable: {exc}; "
                "stopping batch",
                flush=True,
            )
            return 0
        if dual_config is None:
            print(
                "ALERT: dual read requested but VISION_B_BASE_URL/"
                "VISION_B_MODEL are not configured; stopping batch",
                flush=True,
            )
            return 0
    progress = load_progress(progress_path)

    total_inserted = 0
    for i, (doc_id, yr, path) in enumerate(remaining):
        print(f"\n[{i + 1}/{len(remaining)}] {doc_id} ({yr})...", flush=True)
        time.sleep(COOLDOWN)
        if dual_read_enabled():
            # Opt-in dual vision read (DUAL_READ=1): engine A (Gemini) is
            # gated against engine B per doc. AGREE/DISAGREE are persisted or
            # quarantined by the orchestrator; engine-B absence is UNRESOLVED
            # — quota stops the batch, other outages land an error run and
            # are retried next time. Nothing persists A-only.
            outcome = run_dual_read_for_doc(
                DualReadDoc(str(doc_id), yr, path, data_dir=data_dir, db_path=db_path),
                refresh=refresh,
            )
            total_inserted += outcome.inserted
            if outcome.handled:
                mark_progress(progress, doc_id, outcome.progress_key)
                save_progress(progress, progress_path)
                print(
                    f"  DUAL {outcome.decision.status}: inserted={outcome.inserted}",
                    flush=True,
                )
                continue
            if outcome.decision.status == UNRESOLVED:
                # Engine-B quota stop (unhandled == nothing persisted).
                print("  Engine B quota exhausted; stopping batch", flush=True)
                break
            output, error, artifact_metadata = (
                outcome.output,
                outcome.error,
                outcome.metadata,
            )
        else:
            output, error, artifact_metadata = call_gemini(
                path,
                doc_id=doc_id,
                refresh=refresh,
                cache_dir=os.path.join(data_dir, "gemini_cache"),
                parser_version=GEMINI_PARSER_VERSION,
            )
        # A_ERROR passthrough carries error=None (engine A succeeded).
        if is_quota_error(error or ""):
            print("  Quota exhausted; stopping batch", flush=True)
            break
        ingestion_generation = None
        if artifact_metadata is not None:
            try:
                ingestion_generation = resolve_ingestion_generation(
                    db_path, doc_id, yr, artifact_metadata.sha256
                )
            except RuntimeError as exc:
                output, error = None, str(exc)
        if output is None or error:
            conn = duckdb.connect(db_path)
            try:
                record_failed_ocr_attempt(
                    conn,
                    doc_id,
                    yr,
                    0,
                    error,
                    artifact_sha256=(
                        artifact_metadata.sha256 if artifact_metadata else None
                    ),
                    ingestion_generation=ingestion_generation,
                )
            finally:
                conn.close()
            mark_progress(progress, doc_id, "errors")
            save_progress(progress, progress_path)
            print(f"  ERROR: {error}", flush=True)
            continue

        parsed = _require_successful_response_metadata(artifact_metadata)
        member, transactions = parsed.member, parsed.transactions
        if not transactions:
            no_tx_outcome = insert_transactions_outcome(
                doc_id,
                yr,
                member,
                [],
                db_path=db_path,
                raw_count=0,
                parser_version=GEMINI_PARSER_VERSION,
                artifact_sha256=artifact_metadata.sha256,
                ingestion_generation=ingestion_generation,
            )
            if no_tx_outcome.complete:
                mark_progress(progress, doc_id, "no_txs")
                save_progress(progress, progress_path)
                print("  No transactions found", flush=True)
            else:
                mark_progress(progress, doc_id, "errors")
                save_progress(progress, progress_path)
                print(
                    "  ERROR: no-transaction run not committed "
                    f"({no_tx_outcome.status})",
                    flush=True,
                )
            continue

        raw_count = len(transactions)
        insert_outcome = insert_transactions_outcome(
            doc_id,
            yr,
            member,
            transactions,
            db_path=db_path,
            parser_version=GEMINI_PARSER_VERSION,
            raw_count=raw_count,
            artifact_sha256=artifact_metadata.sha256,
            ingestion_generation=ingestion_generation,
            force_partial_canonical=force_partial_canonical,
        )
        if not insert_outcome.complete:
            # Rejected or partial: never call the output complete.
            mark_progress(progress, doc_id, "errors")
            save_progress(progress, progress_path)
            print(
                f"  ERROR: insert incomplete ({insert_outcome.status}); "
                f"inserted={insert_outcome.inserted}/{raw_count}",
                flush=True,
            )
            continue
        total_inserted += insert_outcome.inserted
        mark_progress(progress, doc_id, "success")
        save_progress(progress, progress_path)
        print(f"  Inserted: {insert_outcome.inserted}/{raw_count}", flush=True)

    return total_inserted


if __name__ == "__main__":
    main()
