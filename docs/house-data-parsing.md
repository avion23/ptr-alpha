# House data parsing

This document describes the current House PTR ingestion implementation. It is not a claim that every House filing can be reconstructed perfectly.

## Data flow

```text
House annual index ZIP
        |
        v
validated metadata generation
        |
        v
PTR PDF acquisition + artifact SHA-256
        |
        v
per-document parser cascade
        |
        v
consolidation + metadata attribution
        |
        v
atomic transaction + parse-run persistence
        |
        v
complete generation -> canonical_transactions
```

### 1. Metadata

`HouseTransactionSource.fetch_metadata()` reads the annual House index ZIP and normalizes the tab-separated metadata table.

Validation includes:

- UTF-8 BOM handling with Windows-1252 fallback for historical names;
- required `DocID`, `First`, `Last`, and `FilingDate` headers;
- duplicate-header rejection;
- duplicate/blank document-ID checks;
- row-width validation;
- filing-date validation;
- ambiguous ZIP text-member rejection.

House archive generations are tracked explicitly. An incomplete generation is not made canonical merely because some files or rows exist.

### 2. PDF acquisition

PTR metadata rows (`FilingType == "P"`) drive the required PDF set.

Downloads:

- require HTTP success;
- require PDF signature/trailer validation;
- write through a temporary file and atomic rename;
- record artifact SHA-256 and HTTP metadata;
- retain generation identity;
- fail an incomplete archive acquisition instead of silently calling it complete.

Cached PDFs are reused only when they pass local PDF validation.

### 3. Deterministic parser cascade

`_parse_pdf_worker()` compares text-layer parsers first:

1. pdfplumber
2. Camelot lattice
3. Camelot stream
4. `pdftotext`

Each backend produces normalized transaction candidates through `parse_pdf_table()`.

Important current behavior:

- all four text engines are attempted;
- pdfplumber, lattice, stream, and Tesseract aggregate rows from every returned table;
- `pdftotext` aggregates rows from every returned table instead of stopping at the first non-empty table;
- Docling likewise aggregates every returned table;
- candidate diagnostics record engine row count, simple field-completeness quality, and elapsed time at debug level;
- candidate multisets are compared by normalized transaction identity;
- row-count/identity disagreement is recorded in `engines_attempted` telemetry.

If pdfplumber's candidate multiset is contained in pdftotext's result, pdftotext is accepted. If pdftotext is contained in pdfplumber, pdfplumber is accepted. This is the cheap trusted-agreement path.

Otherwise the parse is uncertain and OCR corroboration is required.

### 4. OCR path

OCR is materially more expensive than text extraction:

5. Docling, unless `PTR_SKIP_DOCLING=1`
6. Tesseract

If Tesseract returns a candidate set, text/OCR candidates are reconciled while preserving the maximum observed multiplicity for each normalized transaction identity.

If no backend establishes complete enough coverage, the document raises `ParserCascadeError`. It is quarantined as a per-document parse failure instead of aborting the entire annual pool.

A per-document wall-clock watchdog bounds the accepted cascade. Docling/Tesseract subprocesses also have their own bounded execution paths. `PTR_SKIP_DOCS` is an explicit operator escape hatch for known pathological documents; skipped documents are recorded as failures, not successes.

The optional Gemini pass is separate. It targets unresolved deterministic parses, validates the response contract, records provenance, and caches the model response. Gemini output is never treated as semantically correct merely because it is valid JSON.

## Table parsing

`parse_pdf_table()` handles heterogeneous House layouts:

- searches introductory rows for a header;
- recognizes one- and two-row headers;
- maps asset/type/date/owner/amount columns;
- merges bounded continuation rows;
- extracts ticker, type, dates, owner code, amount interval/midpoint, instrument type, strike/expiry, and raw asset description;
- rejects rows that cannot establish a transaction type/date;
- preserves source-row identity where an extractor provides it.

Ticker extraction is intentionally conservative. Company-name fallback mappings and parser-artifact mappings are not equivalent to exchange verification; downstream candidate selection also applies asset and provenance checks.

## Consolidation and persistence

`consolidate_transactions()` joins parsed document IDs to House metadata, assigns member/disclosure identity, normalizes dates, and rejects impossible rows.

For reparses, `preserve_existing_fields()` can carry forward an existing ticker or raw amount only when the old value is unambiguous for the transaction identity. It does not guess across conflicting historical rows.

`Database.replace_transactions_for_docs()` is the persistence boundary:

- attempted document IDs and replacement document IDs are explicit and separate;
- deterministic ambiguous `zero_rows` results do **not** delete existing House rows;
- nonzero/verified replacement rows are deleted and reinserted for the same source/generation;
- replacement rows and parse-run telemetry are committed in the **same DuckDB transaction**;
- the parse run's `transaction_count` is overwritten with the actual persisted source-generation count before commit;
- any insert/parse-run failure rolls back the replacement transaction.

`raw_row_count` is the number of parser candidates before consolidation. It is no longer a hard-coded zero.

## Cheap parsing vs expensive parsing

The useful operational distinction is cost, not parser prestige.

| Class | Engines | Typical cost | Use |
| --- | --- | ---: | --- |
| Text layer | pdfplumber, Camelot, pdftotext | milliseconds to seconds | default path |
| Raster/OCR | Tesseract | seconds to minutes | uncertain/scanned documents |
| Large OCR pipeline | Docling | seconds to minutes with high memory | difficult scans |
| External model | Gemini OCR | API latency/quota/cost | explicit recovery pass |

The text path should do as much work as possible, but **uncertainty must not be converted into confidence merely to avoid OCR cost**.

## Logging and diagnostics

Run:

```bash
ptr-alpha --verbose parse --year 2026
ptr-alpha --verbose parse --year 2026 --gemini-ocr
```

Useful parser debug records now include bounded messages such as:

```text
Parser pdfplumber: rows=17 quality=0.941 elapsed=0.083s doc=20012345.pdf
Parser pdftotext: rows=17 quality=1.000 elapsed=0.022s doc=20012345.pdf
```

The annual save summary reports parsed PDFs, zero-row documents, persisted transaction counts, and canonical/raw database totals. Third-party library noise is not a substitute for these bounded project-level diagnostics.

After a parse, inspect latest outcomes:

```sql
WITH latest AS (
  SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) AS rn
  FROM pdf_parse_runs
  WHERE year = 2026
)
SELECT status, count(*) AS documents,
       sum(raw_row_count) AS raw_rows,
       sum(transaction_count) AS persisted_rows
FROM latest
WHERE rn = 1
GROUP BY status
ORDER BY status;
```

Check parse-run/database agreement:

```sql
WITH latest AS (
  SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) AS rn
  FROM pdf_parse_runs
  WHERE year = 2026
)
SELECT l.doc_id, l.status, l.raw_row_count, l.transaction_count,
       count(t.id) AS stored_generation_rows
FROM latest l
LEFT JOIN transactions t
  ON t.doc_id = l.doc_id
 AND t.ingestion_generation = l.ingestion_generation
 AND t.source = 'house_pdf'
WHERE l.rn = 1
GROUP BY l.doc_id, l.status, l.raw_row_count, l.transaction_count
HAVING count(t.id) <> l.transaction_count
ORDER BY l.doc_id;
```

Check date reversals hidden from normal reads:

```sql
SELECT doc_id, member, ticker, transaction_date, disclosure_date
FROM transactions
WHERE transaction_date > disclosure_date
ORDER BY disclosure_date, doc_id;
```

## What parser success means

A parser success means the accepted cascade produced rows that survived normalization and were persisted consistently. It does **not** prove that the filing had no omitted rows or that every ticker/amount/owner field is semantically correct.

The canonical risk register is [`house-ingestion-error-catalog.md`](house-ingestion-error-catalog.md). The dated [`HOUSE_PARSER_AUDIT.md`](HOUSE_PARSER_AUDIT.md) is historical corpus evidence and should not be read as the current parser architecture.
