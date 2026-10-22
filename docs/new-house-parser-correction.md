# Deterministic House row correction (2026-10-22)

This correction retains printed notification dates and linked economic descriptions,
classifies explicit Treasury debt and private SAFEs without calling them common
stock, and excludes the new `private` instrument at common-stock admission. No
weights, priors, source documents, OCR prompts, or canonical data are changed.
Missing notification dates remain missing; disclosure/filing dates are not substitutes.

The internal layout row's original first five cells are unchanged. Notification
Date is appended before optional Source Row ID (which remains last). Both table
headers and the pdfplumber flattened-row unpacking/mapping are updated. All Python
callers of `_parse_pdftotext_lines` and both match helpers were searched.

## Cache identity and controlled migration

`download._PARSE_VERSION` remains `v5-deterministic` intentionally for this isolated
fresh-five correction. `ParseRunRepository.get_cached_doc_ids` matches parser
version, archive year, ingestion generation, artifact SHA-256 and terminal status.
A global version bump would invalidate historical terminal reuse and could trigger
OCR outside this authorization. The five new filings have no existing canonical
parse runs according to the upstream refresh inventory supplied for this task.

**The old and corrected payload recipes are not identical.** Existing cached runs
are retained as historical materializations, not certified reproductions of this
corrected parser, and do not thereby acquire notification dates or repaired types.
Before historical correction or exact recipe-level replay is claimed, the owner
must approve a distinct parser version and controlled migration/reparse scope
(including OCR budget and promotion gates), or separately record a code-revision
recipe identity. This patch does not implement that migration or mark cached rows
as corrected. New verification evidence must include the candidate Git revision
and code hashes, not the coarse parser-version string alone.

This is limited to explicit SAFE/Treasury/bond identity and description retention.
It is not a comprehensive ETF, crypto, fund, or private-security taxonomy, nor
does it authorize ingest, staged promotion, publication, or deployment.
