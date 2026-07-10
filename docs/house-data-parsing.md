# House Data Parsing

This document describes the implementation in `src/analyzer`, not a guarantee about the
accuracy or completeness of House disclosures or extracted records.

## Data flow and ownership

1. `HouseTransactionSource.fetch_metadata()` downloads the annual House ZIP, reads the first
   text member, and passes its tab-separated content to `normalize_house_metadata()`.
2. Metadata normalization strips a UTF-8 BOM, requires `DocID`, `First`, `Last`, and
   `FilingDate`, rejects duplicate headers and empty data, skips malformed-width rows, and
   rejects invalid filing dates. A refresh replaces the year's metadata in one DuckDB
   transaction, so download or validation failure leaves the cached year intact.
3. `fetch_and_cache_pdfs()` selects `FilingType == "P"`, downloads PDFs concurrently, rejects
   non-200 and non-`%PDF-` responses, writes a temporary file, then atomically renames it.
   Existing files are reused only when nonempty and beginning with `%PDF-`.
4. `parse_cached_pdfs()` intersects metadata with files in `data/<year>/pdfs`, builds a
   `DocID`-to-member lookup, and parses PDFs in a multiprocessing pool.
5. Each extraction backend produces table-like rows. `parse_pdf_table()` detects a header in
   the first three rows, maps one- or two-row headers, and otherwise assumes columns 0, 1, and
   2 are asset, type, and date. It merges split rows, then extracts ticker, transaction type,
   date, owner, amount, instrument/option details, and the original asset description.
6. `consolidate_transactions()` joins results to metadata by PDF stem/`DocID`, supplies member
   and filing date, coerces dates, and drops records whose transaction or disclosure date is
   invalid. Documents without matching metadata or a member name are skipped with warnings.
7. `preserve_existing_fields()` carries forward usable fields from an earlier parse where the
   new parse is weaker. `replace_transactions_for_docs()` deletes and reinserts all included
   documents in one transaction; any insertion error rolls back all those deletions.

Metadata, parse-run tracking, and transactions are separate concerns:
`MetadataRepository` stores the filing index, `ParseRunRepository` records extraction outcomes,
and `TransactionRepository` deduplicates and persists normalized transactions. The
`Database` class is their facade.

## Parser cascade

`_parse_pdf_worker()` tries these backends in order:

1. pdfplumber
2. Camelot lattice
3. Camelot stream
4. `pdftotext`
5. Docling OCR, unless `PTR_SKIP_DOCLING=1`
6. Tesseract OCR

All four text-layer parsers are considered unless one returns a result with quality at least
0.7. Quality is the fraction of extracted rows containing both a transaction date and amount
midpoint; it does not score ticker correctness, row completeness, owner, type, or semantic
accuracy. Below the threshold, the highest-quality result wins, with row count as the
tiebreaker. OCR is reached only when every text-layer parser returns zero rows, so a nonempty
but incomplete text parse prevents OCR. Docling returns immediately on nonempty output;
Tesseract is last.

The optional `--gemini-ocr` pass is separate from this deterministic cascade. It targets the
latest `pdf_parse_runs` entries marked `zero_rows` or `error`, validates model output, records
its own parse run, and caches responses under `data/gemini_cache/`. It uses external API quota
and should be audited like any nondeterministic extraction.

## Persistence and replacement boundaries

- Annual metadata refresh is atomic and removes stale metadata rows for that filing year.
- A PDF download becomes visible only after its temporary file is fully written.
- A nonempty batch of consolidated documents is replaced atomically. Rows for documents not
  present in that DataFrame are untouched.
- Zero-row documents are deliberately not deleted. If they already have transactions, the
  parser logs that those rows may be stale.
- Parse-run upsert is atomic per document, but parse-run records are written before the batch
  transaction replacement. Therefore a `success` parse-run can coexist with old transaction
  rows if later consolidation or persistence fails.
- House upserts deduplicate on document, ticker, transaction date, member, type, amount,
  owner, and asset description. Similar rows that differ in any key field remain distinct.
- Reads exclude records where transaction date is later than disclosure date as likely OCR
  date swaps; those records remain stored and are reported only at debug log level.

## Error register and corpus evidence

The severity-ranked [`house-ingestion-error-catalog.md`](house-ingestion-error-catalog.md) is
the canonical potential-error register. The dated
[`HOUSE_PARSER_AUDIT.md`](HOUSE_PARSER_AUDIT.md) records observed results from a full local
PDF-corpus run. Keep potential risks separate from observed run results: a successful parse is
not proof of semantic correctness, while a zero-row result is not proof that a filing has no
transactions.

## Operations and diagnostics

Run with debug logging and preserve the complete output:

```bash
ptr-alpha --verbose parse --year 2025
ptr-alpha --verbose parse --year 2025 --gemini-ocr
```

The CLI exit status detects pipeline failure, not semantic completeness. Query DuckDB after
each run (replace the year as needed):

```sql
-- Latest outcome per document.
WITH latest AS (
  SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) AS rn
  FROM pdf_parse_runs WHERE year = 2025
)
SELECT status, count(*) AS documents, sum(transaction_count) AS rows
FROM latest WHERE rn = 1 GROUP BY status ORDER BY status;

-- Zero/error documents and the engines actually attempted.
WITH latest AS (
  SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) AS rn
  FROM pdf_parse_runs WHERE year = 2025
)
SELECT doc_id, status, parser_version, engines_attempted, transaction_count,
       error_message, parsed_at
FROM latest WHERE rn = 1 AND status <> 'success' ORDER BY doc_id;

-- Successful parse-run records with no currently stored rows.
WITH latest AS (
  SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) AS rn
  FROM pdf_parse_runs WHERE year = 2025
)
SELECT l.doc_id, l.transaction_count, count(t.id) AS stored_rows
FROM latest l LEFT JOIN transactions t USING (doc_id)
WHERE l.rn = 1 AND l.status = 'success'
GROUP BY l.doc_id, l.transaction_count
HAVING count(t.id) = 0 OR count(t.id) <> l.transaction_count;

-- Stored rows hidden from normal reads because dates are reversed.
SELECT doc_id, member, ticker, transaction_date, disclosure_date
FROM transactions
WHERE transaction_date > disclosure_date
ORDER BY disclosure_date, doc_id;

-- Coverage of PTR metadata by a latest parse-run record.
WITH latest AS (
  SELECT *, row_number() OVER (PARTITION BY doc_id ORDER BY parsed_at DESC) AS rn
  FROM pdf_parse_runs
)
SELECT m.doc_id, m.first_name, m.last_name, m.filing_date
FROM metadata m LEFT JOIN latest l ON l.doc_id = m.doc_id AND l.rn = 1
WHERE extract(year FROM m.filing_date) = 2025
  AND m.filing_type = 'P' AND l.doc_id IS NULL
ORDER BY m.doc_id;
```

Also compare metadata PTR count, valid local PDF count, latest successful/zero/error counts,
stored rows per document, and totals against the prior run. Spot-check PDFs from every winning
engine, all zero-row documents, unusually high/low row counts, missing amounts/tickers, unknown
owner codes, options, and date reversals. Keep the database and `data/gemini_cache/` backed up
before bulk reparsing; do not treat a stable row count as proof of correct extraction.
