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

`download._PARSE_VERSION` is `v6-deterministic-context`: new parse runs identify
the corrected recipe. Default cache lookup accepts only that version, with the
existing archive-year, ingestion-generation, actual artifact SHA-256 and cacheable
status checks. It never implicitly treats a v5 payload as a v6 result.

For the owner-approved **2026 refresh only**, set the run-local environment value
`PTR_REUSE_LEGACY_V5_CACHE_YEAR=2026`. Unset it after that operation; do not persist
it in service configuration. This accepts unchanged `v5-deterministic` cached
materializations only when the requested archive year equals the explicit value,
using the same generation/artifact/status checks as the primary lookup. No other
parser versions or old generations are searched. An absent value disables legacy
reuse; a present non-four-digit/zero year raises `ValueError`. `force=True` ignores
all caches and the opt-in. A different archive year does not inherit the opt-in.

Logs record the requested year, new producer version, force mode, configured
opt-in year, whether it applies, and the count/generation/original producer of
legacy-only skips. Skipping performs no save, relabel, or payload rewrite: reused
v5 rows retain their original versions and values. **They are not corrected or
reproduced under v6.** The five uncached new filings must be parsed and recorded
as v6; the flag does not manufacture cache hits for them. The refresh report must
retain the explicit policy/logs and pin the executing Git revision/code hashes.

No all-year historical reparse, OCR budget expansion, or migration is authorized.
Such work requires a separately approved scope and unchanged promotion gates.

This is limited to explicit SAFE/Treasury/bond identity and description retention.
It is not a comprehensive ETF, crypto, fund, or private-security taxonomy, nor
does it authorize ingest, staged promotion, publication, or deployment.
