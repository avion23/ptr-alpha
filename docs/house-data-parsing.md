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

## Potential error modes

Treat all of the following as audit candidates, even when the command exits successfully.

### Source and metadata

- Network timeout, HTTP failure, rate limiting, cached HTTP response, HTML/error content, a
  changed House URL or ZIP layout, corrupt ZIP, or the wrong first text member.
- Empty, one-line, wrong-delimiter, renamed/missing/duplicate columns, BOM/encoding damage,
  malformed row widths, invalid filing dates, missing names, incorrect filing type, duplicate
  `DocID`, or a filing assigned to the wrong year.
- Metadata refresh can be internally valid but incomplete; atomic replacement then faithfully
  replaces the year with that incomplete upstream index.
- A PDF may be absent from the index, indexed as non-PTR, renamed, withdrawn, amended, or
  associated with metadata that does not match its contents.

### Download and local files

- Missing directory, permission/disk-full error, timeout, non-200 response, HTML masquerading
  as a download, truncated/corrupt/encrypted PDF, or a file with a valid header but unreadable
  body. The validity check verifies only existence, nonzero size, and the first five bytes.
- Cached files are not checksummed or compared with the current upstream version, so a valid-
  header stale or silently revised PDF can be reused.
- Concurrent worker/OCR memory exhaustion, process termination, unavailable system binaries,
  optional Python dependency failure, or platform-specific extraction differences.

### Tables, rows, and cells

- Backend returns no table, the wrong table, duplicated tables/pages, partial pages, reordered
  rows, collapsed columns, or incorrectly segmented cells.
- Header occurs after row 3, uses an unknown spelling, has fewer than two recognized labels,
  or a two-row header is mistaken for data. Headerless fallback assumes a fixed three-column
  layout and can silently mis-map other layouts.
- A disclosure with only one extracted row is rejected because `parse_pdf_table()` requires at
  least two table rows.
- Continuation logic merges only the next row and may join unrelated rows or fail on three-way
  splits. It activates when type and date are both missing; other partial splits can be lost.
- Unknown transaction labels, malformed or ambiguous dates, two-digit-year normalization,
  OCR digit swaps, missing tickers, ticker punctuation, fund/bond/private assets, and company-
  name substring matches can cause false negatives or false positives.
- The ticker blacklist and static company-name map are heuristic and can become stale. Company
  renames, share classes, foreign listings, symbols that are ordinary words, and ticker reuse
  require manual verification.
- Owner codes outside known values are truncated to eight characters. Amount regex/range
  parsing can miss OCR variants or infer the wrong midpoint. Generic stock options may be
  classified as calls. Strike and expiry parsing can be absent or wrong.
- A row is retained only when both type and date parse. Consolidation later drops invalid dates
  without a per-row error record. Missing ticker rows can be stored, but downstream analysis
  may be unable to use them until a ticker is resolved.

### Selection, joins, and persistence

- Any PDF stem that differs from metadata `DocID`, missing lookup entry, or blank member name
  causes all its extracted transactions to be skipped.
- The 0.7 quality threshold can select a shorter/incomplete parser result; quality ignores
  correctness. Any nonempty text result prevents the slower OCR backends.
- Carry-forward matching can preserve an old field onto an incorrectly matched new row, while
  changed identity fields can prevent preservation. Deduplication can merge genuinely distinct
  transactions or retain extraction duplicates whose key fields differ slightly.
- Zero-row reparses intentionally preserve prior rows, which protects data from destructive
  parsing failures but can retain records that an amended filing removed.
- Parse-run status, transactions, and optional Gemini progress/cache are not committed as one
  unit. A crash can leave their states inconsistent. `raw_row_count` is currently recorded as
  zero by the deterministic pipeline and is not a useful completeness metric.
- External scripts can use different parser versions, selection rules, or write paths. Record
  `source`, `parser_version`, and command used when comparing results.

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
