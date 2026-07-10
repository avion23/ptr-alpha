# House ingestion and parsing error catalog

This catalog covers the House metadata-to-database path. It is an operational risk register, not a claim that every malformed disclosure can be recovered. Severity describes possible data impact: **critical** can corrupt or broadly erase stored data, **high** can silently omit or misattribute transactions, **medium** degrades fields or a subset of filings, and **low** primarily affects observability or availability.

## Pipeline and failure boundaries

`HouseTransactionSource` downloads the annual metadata ZIP, normalizes its TSV, filters PTR filings, downloads PDFs atomically, runs the parser cascade, joins rows to member metadata, preserves selected resolved fields, and atomically replaces rows for successfully parsed documents. The cascade tries pdfplumber, Camelot lattice/stream, pdftotext, Docling, then Tesseract. Table rows pass through header/column detection, continuation merging, and cell extraction.

## Error catalog

| Area / error condition | Severity | Detection | Current behavior or mitigation | Residual risk |
|---|---|---|---|---|
| Metadata HTTP failure, timeout, non-200 response | High | Exception and command failure | Existing cached rows remain; refresh replacement occurs only after download and validation | Cache may be stale; no retry/backoff in this layer |
| HTTP cache serves an obsolete but valid response | High | Compare `fetched_at`, upstream index, and prior snapshot | Cache expires after one hour | A revised upstream index can be missed within the cache window |
| Response is not a ZIP, ZIP is corrupt, or has no `.txt` member | High | ZIP/`ParsingError` propagated as `DataSourceError` | No database replacement | Uppercase `.TXT` is not recognized; first matching member is selected without semantic validation |
| Metadata is empty, header-only, or lacks identity/date columns | High | `ParsingError` | Rejected before persistence | A syntactically valid but semantically wrong text member may pass |
| UTF-8 BOM on first metadata header | Medium | Regression test | BOM is removed | Other encodings are decoded as UTF-8 with invalid bytes silently discarded |
| Duplicate metadata headers | High | `ParsingError` | Entire file rejected | None known |
| Metadata row has fewer columns than header | High | Warning with dropped-row count | Malformed row is skipped | Silent omission if warnings are not monitored; no doc IDs in warning |
| Metadata row has extra columns or embedded tabs | High | Not explicitly detected | Extra fields are truncated | Column shift can misattribute names/dates/doc IDs |
| Invalid filing date | High | Coercion then row drop; all-invalid file raises | Bad rows excluded | Partial loss is not summarized or thresholded |
| Duplicate or blank DocID | High | Not explicitly validated here | Passed to repository upsert | Collisions/blank IDs can overwrite or create bad PDF paths depending on DB constraints |
| Wrong/missing `FilingType` | High | Later filtering or missing-column exception | Only exact PTR value is parsed | Case/schema drift can omit every filing |
| Filing is absent, withdrawn, amended, renamed, assigned to another year, or indexed as non-PTR | High | Reconcile metadata and PDFs with the House source | Exact PTR metadata drives selection | Cached transactions/PDFs can outlive upstream changes; no amendment lineage is modeled |
| Metadata refresh contains fewer valid rows than prior snapshot | Critical | No count/coverage guard | Valid refresh atomically replaces the year | A truncated but syntactically valid upstream file can delete cached metadata |
| PDF HTTP failure/timeout | High | Download result and summary count | Existing valid PDF is retained; failed new file is not installed | Overall fetch does not fail solely because individual downloads fail |
| HTTP 200 contains HTML/error bytes | High | `%PDF-` signature check | Rejected | Signature-only checks cannot detect truncated/corrupt PDFs |
| Process dies while writing PDF | Medium | `.pdf.tmp` may remain | Temporary write plus atomic `os.replace` protects final file | Temporary files are not cleaned automatically |
| PDF directory is unwritable or storage is full | High | `OSError`/download error result and summary | Failed temporary write is not installed as the final PDF | Other downloads can succeed, leaving incomplete year coverage |
| Existing PDF begins `%PDF-` but is truncated/corrupt | High | Parser engines fail or yield zero rows | Old DB rows are retained and warning is emitted | Download is skipped, so corruption is not self-healed |
| Missing PDF directory / no files | High | `DataSourceError` | Parse stops without transaction replacement | Missing subset of PDFs is only visible through counts/logs |
| Parser dependency absent (pdftotext, Ghostscript, OCR/Docling) | Medium | Engine debug logs; usually zero rows | Cascade continues to other engines | Debug-only diagnostics and `engines_attempted` do not distinguish unavailable from no match |
| Dependency/version/platform differences change extraction | High | Re-run fixtures/corpus under pinned environments and compare per-document output | Deterministic ordering within one environment | Native PDF tools and OCR can emit different tables without an exception |
| Parser dependency hangs or exhausts memory | High | pdftotext has 15s timeout; process/OS signals otherwise | Multiprocessing isolates workers to a degree | pdfplumber/Camelot/Docling/Tesseract have no per-engine timeout here; worker failure may abort the batch |
| Password/encryption or unsupported PDF features | High | Engine exceptions at debug level / zero rows | Cascade tries alternate engines | All engines can fail silently into `zero_rows` |
| Parser engine returns some plausible rows but misses others | High | No ground-truth count; quality checks date+amount completeness | Low-quality text results are compared; zero rows triggers OCR | A >=0.7-quality early engine wins even if a later engine would find more rows; OCR is skipped whenever text engines find any rows |
| Different engines return conflicting rows | High | Not detected | One engine wins; results are not reconciled | Wrong ticker/date/type/amount can look valid and persist |
| Multiple tables/pages | High | Backend-specific | pdfplumber/lattice/Tesseract aggregate tables | Stream, pdftotext, Docling return the first table producing rows and can omit later tables |
| Header absent | Medium | No header found | Default columns 0/1/2; first row retained, including a one-row table | Nonstandard column order is silently misparsed |
| Header appears after first three rows | High | Not detected | Treated as headerless data | Transactions may be lost or fields shifted |
| Two-row header resembles data | Medium | Heuristic core-column test | May merge headers and skip both rows | False header classification can skip first transaction |
| Unknown, misspelled, or reordered columns | High | Core mapping fallback | Defaults to asset/type/date positions | Owner/amount can be lost; incorrect dates/types can attach to another field |
| Short/ragged row | Medium | Missing cells yield `None`; `IndexError` suppressed | Invalid rows are dropped | No per-row error count or source coordinates |
| Split transaction spans more than two rows | High | Not detected | Only one following continuation row is merged | Transaction omitted or asset truncated |
| Split row already contains a ticker | Medium | Regression test | It is merged when type/date are on the next row | A next row belonging to another transaction can be consumed if it supplies plausible fields |
| Duplicate transaction rows/pages | High | No parser-level deduplication | Sent to repository | Database key/upsert semantics may collapse some duplicates; genuine same-day same-type trades are hard to distinguish |
| Ticker punctuation/length outside regex or company map | Medium | Ticker becomes null | Transaction may still persist | Foreign/exotic tickers and OCR variants are missed |
| Parenthesized non-ticker or OCR fragment resembles ticker | High | Small blacklist only | Candidate is accepted | False tickers remain possible; no exchange/security validation at parse time |
| Company-name fallback matches inside another word | High | Regression test | Alphanumeric boundaries are now required | Ambiguous standalone names such as Block, Gap, Target, Arm, Shell, and Ford can still be ordinary words |
| Company-name fallback map is stale/wrong | High | Manual tests only | Longest bounded name wins | Corporate actions, share classes, and international listings can be misresolved |
| Ticker is missing but row otherwise parses | Medium | Query null tickers and compare source PDF | Row can be stored for later resolution | Downstream analysis may ignore or be unable to price the transaction |
| Transaction type new/garbled/unsupported | High | Extractor returns null; row dropped | Recognizes P/S/E and common words/partial suffix | Gifts, exercises, and OCR substitutions may be omitted or mapped incorrectly |
| Date is malformed or impossible | High | Regex accepts shape; later pandas coercion drops impossible dates | Invalid consolidated row is removed | Dropped-row count is not logged; two-digit years use fixed 1950 pivot |
| Transaction date after disclosure or implausibly old/future | High | Debug exclusion count and direct database query | Persisted if parseable; normal repository reads exclude dates after disclosure | Bad rows remain stored, other implausible dates are not excluded, and direct SQL consumers can use them |
| Owner code unknown or OCR-corrupt | Medium | Not validated | Cleaned value truncated to eight characters | Arbitrary uppercase asset text can still be stored as owner if columns are shifted |
| Amount missing or OCR-corrupt | Medium | Quality score checks midpoint presence | Raw/midpoint null allowed; old raw amount may be preserved on reparse | Transactions persist without size; malformed bracket can produce a plausible wrong midpoint |
| Option classification/detail extraction error | Medium | No cross-field validation | Best-effort parsing; defaults to stock | Expiry/strike can be absent or wrong and instrument can be misclassified |
| Asset description over 500 characters | Low | Deterministic truncation | Bounded storage | Useful identifier/details beyond limit are lost; description is not consolidated into final dataframe |
| Metadata lookup missing for parsed doc | High | Warning with doc ID/count | All its transactions are skipped | Partial batch can still be saved; no failure threshold |
| Member first/last absent | High | Warning | Transactions skipped | Name changes/suffixes are not a stable member identity |
| Duplicate DocID maps to different members | Critical | Not detected in lookup construction | Last dictionary entry wins | Transactions can be attributed to the wrong member |
| Consolidated required transaction keys absent | Medium | `KeyError` aborts save | No transaction replacement has occurred yet | Parse-run records may already have been written, producing misleading status |
| Consolidated date coercion drops rows | High | Not reported | Invalid transaction/disclosure dates removed | Partial data loss can be silent and can turn a document into no replacement while other docs replace |
| Fresh parse returns zero rows for a previously stored doc | High | Parse-run `zero_rows` and stale-row warning | Existing transactions are retained | Stale data can survive indefinitely and parser regression is not a hard failure |
| Fresh parse finds fewer nonzero rows than stored | Critical | No comparison/threshold | Successfully parsed doc is atomically replaced | Partial parse can delete valid historical rows |
| Reparse loses resolved ticker or raw amount | High | Regression tests | Unambiguous existing values are carried forward by member/date/type identity | Other fields are not preserved; same-day same-type trades are ambiguous; changed identity prevents carry-forward |
| Database insert/replacement fails | Critical | Exception | Per-call transaction wraps delete+upsert and rolls back | Parse-run upserts occur before the replacement transaction and are not rolled back with it |
| Batch has valid rows for some docs and zero/invalid rows for others | High | Warnings/parse runs | Valid docs replace; zero-row docs remain stale | The database becomes a mixed-generation snapshot |
| Abrupt process termination between metadata, PDFs, parse runs and transactions | High | Operational monitoring | Individual metadata/transaction replacement operations are transactional | End-to-end pipeline has no single transaction or generation marker |
| Gemini OCR output is nondeterministic, quota-limited, malformed, or served from a stale cache | High | Validation, model parse run, cache review, and PDF spot-check | Optional pass targets latest zero/error documents and caches validated responses | Validation cannot establish semantic correctness; model/cache/progress/database are not one transaction |
| External maintenance script uses different selection, parser version, or write path | High | Record command, `parser_version`, and `source`; compare document-level counts | Provenance columns distinguish common sources | Scripts are not forced through one end-to-end generation boundary and legacy rows can have null source |
| Logs/debug diagnostics not retained | Medium | External | Parse-run stores attempted engine names and counts | `raw_row_count` is always zero; exceptions are generally not recorded per engine/document |
| Upstream layout/schema changes | High | Row counts and warnings only | Multiple heuristic engines/fallbacks | No fixture/contract check against current House format and no anomaly threshold |

## Operator checks

Before treating a refresh as complete, compare metadata/PTR/PDF/document/transaction counts to the previous run, inspect all `zero_rows` and missing-metadata warnings, verify parser dependencies, and sample disclosures against the source PDFs. Keep a database backup because a plausible partial parse can legitimately pass current validation and replace a document with fewer rows.

Tests cover deterministic parser and transactional regressions, but they cannot prove extraction completeness. Current parse-run telemetry should not be used as a raw-input audit trail because `raw_row_count` is always recorded as zero.
