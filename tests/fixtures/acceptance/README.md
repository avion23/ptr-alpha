# SEC acceptance-clock evidence

Captured directly from SEC submissions JSON and complete accession `.txt` files
in one bounded run (5 requests total, at least 130 ms between requests), with
the `SEC_USER_AGENT` request header. `captured.json` retains the exact per-filing
JSON records and both source URLs; the `.txt` files are unmodified responses.

| Accession | Header (Eastern local) | JSON acceptanceDateTime | JSON wall-clock minus header |
| --- | --- | --- | --- |
| 0001268406-25-000027 | 2025-10-15 21:43:51 EDT | 2025-10-16T05:43:51.000Z | 8 hours |
| 0001193125-26-352200 | 2026-08-14 16:05:04 EDT | 2026-08-15T00:05:04.000Z | 8 hours |
| 0001193125-26-054580 | 2026-02-17 16:05:04 EST | 2026-02-18T02:05:04.000Z | 10 hours |

These observations equal *twice* the seasonal Eastern-to-UTC offset, not
ordinary UTC representations of the header time. Converting the JSON as UTC
still leaves it four hours late in summer and five hours late in winter. This
does not establish SEC's internal transformation or a universal correction.
We deliberately do not repair JSON by any inferred fixed or seasonal offset.

The authoritative knowledge time is the filing's own ACCEPTANCE-DATETIME:
the receipt event recorded in its submission header (also consistent with the
filing-detail page in the reported review), not the discrepant JSON rendering.
Form 4 reads its already-downloaded submission; 13F fetches the complete
submission alongside its information-table XML. Both store the header verbatim
as a naive Eastern-local TIMESTAMP. Missing headers yield NULL; JSON is never
a fallback. Neither filing date nor insertion time is a substitute.

Only Form 4 and 13F populate this column. House/Senate sources never populate
it and remain NULL. Writable schema migration preserves legacy NULLs; read-only
opens do not migrate. Deduplication does not enrich existing rows: already
stored rows with NULL stay NULL on refetch (no backfill in this change).
