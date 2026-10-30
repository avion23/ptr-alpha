"""Fail closed on new transaction readers, including callers of thin loaders.

No database is opened. Inventory Python ASTs in src/ and scripts/ (not docs or
comments). Exemptions below are exact path:qualified-function -> site counts,
with an inline reason; never exempt a whole module. A new site in an existing
exempt function also requires review. Caller-owned loaders remain monitored at
every call site. Prefer canonical_decision/canonical_as_of to new exemptions.

This is an architectural tripwire, not SQL/data-flow verification: dynamic SQL,
reflection, and cross-module aliases still require review and behavioral tests.
"""

import ast
from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TABLE_READ = re.compile(r"\b(?:FROM|JOIN)\s+(?:\w+\.)?[\"`]?"
                        r"(?:canonical_transactions|transactions)\b", re.I)

# Caller-owned scope: keep these names in the call-site inventory, not just SQL.
LOADERS = {
    "get_transactions", "get_house_transactions", "get_transactions_by_date_range",
    "get_skill_history", "get_by_year", "get_house_by_year", "get_by_date_range",
    "load_member_trades", "_member_variants", "_load_actor_members",
    "_unpromoted_transaction_source", "_load_unpromoted_rows",
    "_analysis_transactions", "get_entry_prices",
    "get_for_doc", "get_transactions_for_doc",
}

# Exact exemptions, with reasons. Counts are AST sites, not source line numbers.
EXEMPT = {
    "scripts/audit_generation.py:_get_unresolved_house_doc_ids": 1,  # Current ingestion audit.
    "scripts/purge_unaccepted_house_rowsets.py:<module>": 2,  # One-time contamination report (report-only default; reads the leak it measures).
    "scripts/purge_unaccepted_house_rowsets.py:main": 1,  # Read-only preview before acquiring maintenance ownership.
    "scripts/purge_unaccepted_house_rowsets.py:_purge_owned": 4,  # Owner-held contamination cleanup with transaction_versions archival.
    "scripts/audit_generation.py:check_canary_counts": 1,  # Current canary inventory.
    "scripts/audit_generation.py:check_date_domain": 6,  # Validate stored date domains, not decisions.
    "scripts/audit_generation.py:check_duplicate_policy": 1,  # Economic-duplicate audit.
    "scripts/audit_generation.py:check_source_identity_duplicates": 2,  # Shared source-identity invariant.
    "scripts/audit_generation.py:check_house_generation_activation": 2,  # Current activation audit.
    "scripts/audit_generation.py:check_no_transaction_after_disclosure": 2,  # Stored-row invariant.
    "scripts/audit_generation.py:check_parse_counts_match_persisted": 1,  # Ingestion reconciliation.
    "scripts/audit_generation.py:check_source_report_equation": 4,  # Ingestion reconciliation.
    "scripts/audit_generation.py:check_source_row_ids": 4,  # Stored identity audit.
    "scripts/backfill_tickers.py:_backfill_owned": 3,  # Owner-held current repair with row archival.
    "scripts/cleanup_tickers.py:_cleanup_owned": 8,  # Owner-held current ticker repair and counts.
    "scripts/ocr_triage.py:load_unresolved_artifacts": 1,  # Current OCR work queue.
    "src/analyzer/ocr/ingestion.py:record_failed_ocr_attempt": 1,  # Ingestion provenance.
    "src/analyzer/ocr/ingestion.py:get_ocr_work_items": 3,  # Current OCR work queue.
    "src/analyzer/ocr/ingestion.py:_insert_transactions": 1,  # Ingestion no-txs row retirement.
    "src/analyzer/ocr/ingestion.py:_replace_accepted_ocr_rows": 5,  # Persistence/deduplication/history repair.
    "src/analyzer/ocr/ingestion.py:record_parse_run": 1,  # Ingestion provenance.
    "scripts/purge_phantom_rows.py:<module>": 1,  # Duplicate-detection SQL constant.
    "scripts/purge_phantom_rows.py:purge_phantom_rows": 3,  # Current duplicate repair.
    "scripts/rebuild_staged.py:_canonical_view_diff_counts": 1,  # View-equivalence audit.
    "scripts/rebuild_staged.py:_house_inventory_rows": 1,  # Ingestion inventory.
    "scripts/rebuild_staged.py:finalize": 2,  # Ingestion finalization counts.
    "scripts/rebuild_staged.py:prices": 1,  # Price-download universe, not decision selection.
    "scripts/rebuild_staged.py:repair_audit_gaps": 9,  # Current ingestion repair.
    "scripts/rebuild_staged.py:_verify_shared_database_audits": 2,  # Staged chronology and source identity only.
    "scripts/rebuild_staged.py:_verify_senate_phase": 4,  # Senate generation/report reconciliation only.
    "scripts/reconcile_blotter.py:get_congressional": 1,  # Current blotter reconciliation, no as-of API.
    "scripts/reconcile_untickered.py:_canonical_rows": 1,  # Current untickered reconciliation.
    "scripts/refresh_prices.py:_source_tickers": 1,  # Download universe, not decision selection.
    "scripts/refresh_prices.py:_recent_active_tickers": 1,  # Dividend-download universe, not decision selection.
    "scripts/reparse_all.py:_persisted_house_generation_counts": 1,  # Ingestion reconciliation.
    "src/analyzer/actors.py:_load_actor_members": 1,  # Caller-owned: scoped load_skill_table; name monitored.
    "src/analyzer/persistence/database.py:Database._init_canonical_transactions_view": 4,  # Scope implementation itself.
    "src/analyzer/persistence/database.py:Database._init_transactions_table": 1,  # Schema/history initialization.
    "src/analyzer/persistence/database.py:Database.get_entry_prices": 1,  # Caller-owned repository forwarding.
    "src/analyzer/persistence/database.py:Database.get_house_transactions": 1,  # Caller-owned repository forwarding.
    "src/analyzer/persistence/database.py:Database.get_transactions": 1,  # Caller-owned repository forwarding.
    "src/analyzer/persistence/database.py:Database.get_transactions_by_date_range": 1,  # Caller-owned repository forwarding.
    "src/analyzer/persistence/database.py:Database.get_transactions_for_doc": 1,  # Current document inspection.
    "src/analyzer/persistence/database.py:Database.get_unresolved_house_doc_ids": 1,  # Current ingestion work queue.
    "src/analyzer/persistence/database.py:Database.mark_house_generation_parse_complete": 2,  # Ingestion reconciliation.
    "src/analyzer/persistence/database.py:Database.promote_house_archive": 3,  # Current generation promotion.
    "src/analyzer/persistence/database.py:Database.replace_transactions_for_docs": 7,  # Persistence/history maintenance.
    "src/analyzer/download.py:HouseTransactionSource.get_transactions": 1,  # Caller-owned source forwarding.
    "src/analyzer/download.py:preserve_existing_fields": 1,  # Reparse repair preserves existing fields.
    "src/analyzer/form13f.py:ThirteenFSource.get_transactions": 1,  # Caller-owned source forwarding.
    "src/analyzer/form4.py:Form4Source.get_transactions": 1,  # Caller-owned source forwarding.
    "src/analyzer/freshness.py:form4_freshness": 1,  # Monitoring aggregate: newest form4 disclosure date, not decision selection.
    "src/analyzer/freshness.py:house_ingestion_freshness": 1,  # Monitoring aggregate: newest House disclosure date, not decision selection.
    "src/analyzer/house_refresh.py:complete_house_refresh": 1,  # Refresh reporting and generation finalization, not decision selection.
    "src/analyzer/house_refresh.py:house_refresh_counts": 2,  # Refresh before/after population counts, not decision selection.
    "src/analyzer/persistence/parse_run_repository.py:ParseRunRepository.upsert": 1,  # Ingestion timestamp provenance.
    "src/analyzer/pipeline.py:_analysis_transactions": 1,  # Caller-owned: scoped prepare_analysis_data.
    "src/analyzer/positions.py:_distinct_member_names": 2,  # Caller-owned identity expansion, batched _member_variants; name monitored.
    "src/analyzer/positions.py:_unpromoted_transaction_source": 1,  # Caller-owned SQL source; only _load_unpromoted_rows uses it; cutoff selects created/retired versions, absent scope uses current rows.
    "src/analyzer/positions.py:_load_unpromoted_rows": 2,  # Caller-owned raw loader; batch is @canonical_decision, and historical load_member_trades callers are scoped and monitored below.
    "src/analyzer/positions.py:load_member_trades": 4,  # Caller-owned member loader; historical callers are @canonical_decision or CLI canonical_as_of; loader name remains monitored.
    "src/analyzer/persistence/price_repository.py:PriceRepository.get_entry_prices": 2,  # Caller-owned repository loader.
    "src/analyzer/row_history.py:accept_corrections": 2,  # Acceptance-history maintenance itself.
    "src/analyzer/senate_efd.py:SenateEFDSource.get_transactions": 1,  # Caller-owned source forwarding.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.count_for_docs": 1,  # Current document audit.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.delete_for_doc": 2,  # Persistence/history maintenance.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.exists": 1,  # Current ingestion cache check.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.get_by_date_range": 1,  # Caller-owned repository loader.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.get_by_year": 1,  # Caller-owned repository loader.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.get_for_doc": 1,  # Current document inspection.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.get_house_by_year": 1,  # Caller-owned repository forwarding.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.get_skill_history": 1,  # Caller-owned skill loader.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.house_exists": 1,  # Current ingestion cache check.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.merge_source_refresh": 3,  # Persistence/deduplication.
    "src/analyzer/persistence/transaction_repository.py:TransactionRepository.upsert": 4,  # Persistence/deduplication.
    "src/analyzer/persistence/transaction_repository.py:drop_already_stored": 1,  # Ingestion deduplication.
    "src/analyzer/vision_engines.py:_record_dual_failure": 1,  # Shared source-bound ingestion retirement for disagreed or unresolved dual readings.
}


def _name(node):
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")


def inventory(source, path):
    """Yield (owner, line, protected) for SQL strings and loader calls."""
    tree = ast.parse(source, filename=path)

    def walk(node, owner="<module>", protected=False, scopes=frozenset()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            owner = node.name if owner == "<module>" else f"{owner}.{node.name}"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                protected = any(isinstance(d, ast.Call) and
                                _name(d.func) == "canonical_decision"
                                for d in node.decorator_list)
                # The existing helpers bind conditional canonical_as_of contexts.
                scopes = frozenset(
                    target.id for stmt in ast.walk(node)
                    if isinstance(stmt, ast.Assign)
                    and any(isinstance(c, ast.Call) and _name(c.func) == "canonical_as_of"
                            for c in ast.walk(stmt.value))
                    for target in stmt.targets if isinstance(target, ast.Name)
                )
        if isinstance(node, (ast.With, ast.AsyncWith)):
            protected = protected or any(
                (isinstance(item.context_expr, ast.Name) and item.context_expr.id in scopes)
                or (isinstance(item.context_expr, ast.Call)
                    and _name(item.context_expr.func) == "canonical_as_of")
                for item in node.items
            )
        # JoinedStr must be inspected as one SQL fragment, without counting its
        # constituent Constants again (SELECT ... FROM {dynamic} is out of scope).
        if isinstance(node, (ast.Constant, ast.JoinedStr)):
            text = (node.value if isinstance(node, ast.Constant) else " ".join(
                part.value for part in node.values if isinstance(part, ast.Constant)
                and isinstance(part.value, str)))
            if isinstance(text, str) and TABLE_READ.search(text):
                yield f"{path}:{owner}", node.lineno, protected
            return
        if isinstance(node, ast.Call):
            if _name(node.func) in LOADERS:
                yield f"{path}:{owner}", node.lineno, protected
            elif _name(node.func) == "decision_history":
                # This helper owns its scope; inventory its backtest/CLI callers
                # as well as checking the helper's underlying read above.
                yield f"{path}:{owner}", node.lineno, True
        for child in ast.iter_child_nodes(node):
            yield from walk(child, owner, protected, scopes)

    yield from walk(tree)


def test_asof_consumer_inventory():
    sites = [site for directory in ("src", "scripts")
             for path in sorted((ROOT / directory).rglob("*.py"))
             for site in inventory(path.read_text(), path.relative_to(ROOT).as_posix())]
    unscoped = Counter(owner for owner, _, protected in sites if not protected)
    errors = [f"{owner.rsplit(':', 1)[0]}:{line}: {owner.split(':', 1)[1]} "
              "reads transaction history without canonical decision scope"
              for owner, line, protected in sites
              if not protected and unscoped[owner] != EXEMPT.get(owner)]
    stale = {owner: count for owner, count in EXEMPT.items()
             if unscoped[owner] != count}
    assert not errors and not stale, "\n".join([
        "Scope the reader or review an exact exemption with an inline reason.",
        *errors, f"Changed/stale exemptions: {stale}" if stale else "",
    ])


def test_inventory_detects_new_unscoped_readers():
    for source in (
        'def added(db, as_of):\n return db.conn.execute("SELECT * FROM canonical_transactions")',
        'def added(db, as_of):\n return db.get_transactions_by_date_range(None, as_of)',
        'def added(db, as_of):\n return db.transactions.get_skill_history(as_of, ())',
        'def added(db, as_of):\n return load_member_trades(db, "member")',
        'def added(db, as_of):\n return _load_unpromoted_rows(db, ["member"])',
        'def added(db, as_of):\n return _unpromoted_transaction_source(db)',
    ):
        sites = list(inventory(source, "synthetic.py"))
        assert sites and not any(protected for _, _, protected in sites)


def test_inventory_recognizes_only_scoped_sites():
    source = '''
@canonical_decision("as_of")
def decorated(db, as_of):
    return db.get_transactions(2025)

def context(db, as_of):
    scope = db.canonical_as_of(as_of) if hasattr(db, "canonical_as_of") else nullcontext()
    with scope:
        db.get_transactions(2025)
    db.get_transactions(2025)

def direct(db, as_of):
    with db.canonical_as_of(as_of):
        db.conn.execute(f"SELECT * FROM canonical_transactions WHERE id = {1}")
'''
    assert [protected for _, _, protected in inventory(source, "synthetic.py")] == [
        True, True, False, True,
    ]
