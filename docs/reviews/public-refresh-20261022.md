# Public-source refresh — 2026-10-22

## Status

This is a **partial source refresh**, not full-feed certification. The SEC
row-cohort publication, additive price recovery and both global decision
paths are verified. **Buying conclusion: no new purchase recommended now.**
No trades were placed or code pushed.

## Completed source work

- Cleanup: `2d582ae4`. Reviewed House context/date fixes: `395a5b39`,
  `04f4bfc0`. Reviewed SEC XML-date compatibility: `81dceee1`.
- The seven-day SEC sweep retained 2,965 logical requests: 2,962 HTTP 200
  responses and three index HTTP 403 responses (October 17, 18 and 22).
  Four completed business-day indices contained 2,958 Form 4 entries,
  below the explicit 5,000-entry cap. Repeated issuer/owner entries are not
  independent trades or corroboration.
- The original accepted frame had 1,286 observations, representing 642
  distinct rows: 592 sales and 50 purchases. Its earlier
  `accepted_purchase_rows` label was inaccurate; it counted both directions.
- Strict XML `xs:date` support recovered two ASPI **sales**, preserving their
  printed calendar dates. All 1,623 previously successful parser invocations
  reproduced their original accepted values offline.
- A capture-driver pathname collision had overwritten 274 intermediate
  per-invocation Parquets with later URL aliases. All 274 were reconstructed
  to their **original recorded SHA-256**, under separate filenames. Original
  ledgers, source responses and accepted-frame bytes were not overwritten.
- The corrected proposal contained 644 distinct rows. Independent source
  identity review admitted 610 (584 sales, 26 purchases), holding 34
  unsupported/unresolved identities. The holds include preferred shares,
  funds, depositary securities, unspecified classes and malformed/placeholder
  symbols; no substitute tickers were invented.
- Existing `Form4Source.save_to_db` published **430 new rows**. The other
  **180 matching rows** were already present and were not overwritten.
  Canonical row count after this step: **78,388**. Affected filings were not
  marked complete merely because some rows passed admission.

The 34-row hold applies to **this publication cohort**, not a retroactive
database cleanup. Six held source keys already exist in older canonical
generations, involving BZUN, STRR, UMH and UXIN. They were neither republished
nor deleted. Those legacy security-identity issues remain explicit manual
recommendation exclusions; the refresh does not certify them repaired.

## House refresh remains incomplete

The 2026 official ZIP matched SHA-256
`252f9374f1b3ac656f951624b075dd113369bfb9d6eb94a3a431cf1c251fbd68`.
Its parser retained 1,755 valid-date metadata entries and 435 PTR PDFs;
37 metadata entries had invalid dates. All five new PDFs matched their
previously reviewed PDF hashes; their six rendered source images were inspected.

The refresh explicitly reused 347 matching v5 materializations without
relabeling their original parser versions or changing transaction-critical
values. Six older documents gained 209 deterministic rows. Comparing the
other 34 canonical columns, excluding only database ID and archive ingestion
generation, found no previous trade values removed or changed.

Generation `20261022T143823678350-ef3db367b4f7` finished **incomplete**;
the refresh exited 1. All five new filings failed cross-engine completeness
checks and have no rows in the resulting canonical view. Direct pdftotext and
pdfplumber agreement alone did not authorize bypassing contradictory Camelot
readings. The isolated v7 Camelot correction remains separate from production
until a suitable maintenance plan permits deployment. Final isolated
candidate `f75d35d357cc2bfb384358ecf718bb6798fafa5c` passed independent
re-review after six fail-closed counterexamples were repaired. Both workers
reproduce all 12 source-matching rows with no OCR; this does not certify
the broader generation or authorize deployment during scoring.

Senate access remains blocked; its latest local disclosure is October 1.
The manager submissions checks found no new Q3 13F reports. Existing 13F
security-mapping defects are not certified repaired.

## Prices and recommendation constraints

The identified historical price worker was gracefully interrupted, not
reported as successful. Its preserved database contains 7,289,220 valid price
rows across 2,661 tickers, with 1,369 reaching October 21. Source-generation
checks admit 2,660 tickers; QLEIX lacks generation metadata and is excluded.
Recovery uses existing additive merge and overlap/comparability gates.

The additive merge committed 10,408 price rows, reaching October 21, but its
global post-check exited 1 on **1,087 preexisting invalid observations** in
JGCCF/OCLCF. A read-only before/after comparison found zero newly introduced
invalid rows and no prior price rows changed or removed. This is not a claim
that historical prices are clean: those observations remain preserved, their
two tickers are excluded from manual buy advice. Exact generation/value,
source-generation identity and consumption re-verification subsequently
passed, with no old transactions or dividends changed or removed.
The failed run and stopped dependent steps are retained, not relabeled as an
original successful operation.

The focused APLD/DX/RAPP follow-up then added another 4,950 price rows, with
zero overlap conflicts or old-price changes. Final price count is
**8,670,231**, latest October 21. Generation states are comparable for 1,597
tickers and unknown for 1,066; unknown bases remain blocked, not forcibly
reconciled. These counts are not exchange-wide freshness certification.

`90eaef54` replaces per-row price-generation inserts with one bulk SQL
statement. Financial rows/header hashes matched the previous implementation
on valid unique-date inputs; 84 focused tests and three subtests passed on
main. The 1,000-row disk benchmark was 7.50 seconds versus 0.048 seconds;
this is a fixture measurement, not a full-pipeline speed guarantee.

A separate three-name quote fetch returned October 21 adjusted closes:
APLD $20.98, DX $11.05 and RAPP $25.06. These are not live executable quotes,
and source freshness alone does not establish canonical comparability.

Additional recommendation checks:

- **APLD:** the close exceeds 1.05 times both newly reported purchase prices
  ($18.04 and $18.86), before any separate actor references are considered.
  Its October 7 release reports $6.4 billion debt and a $221 million quarterly
  continuing-operations loss attributable to common shareholders. Do not
  chase solely because insiders purchased.
- **DX:** distinguish the three admitted common-share purchases from the
  quarantined preferred purchase. Its October 19 release reports an 11.5%
  quarterly economic loss, $10.91 book value per common share and 9.0-times
  leverage. Insider buying does not remove mortgage-spread/funding risk.
- **BORR:** the linked source footnote describes a cashless option exercise
  and sale proceeds funding exercise price/taxes despite printed code `P`.
  Preserve that contradiction; do not recommend it as a verified discretionary
  open-market purchase. This contextual hold does not rewrite the source code
  or canonical direction, and is distinct from source-identity admission.

Both recommendation paths retain their original rules: full-universe weighted
`setups` and 28-day/three-distinct-buyer consensus. No ticker/member-scoped
rerun substitutes for filtering the global output. Strategy parameters,
the $3,000 ceiling and OCR population/pixel promotion gates remain unchanged;
13F corroboration stays disabled. Positive deployable net alpha remains
unproven. A buy-screen exclusion is not a sell instruction, and 90 days is
an evaluation horizon, not an agreed liquidation rule.

## Completed global rebuild and buying conclusion

Both commands exited 0 under main code `90eaef5419615b11629ae3552c9049c370f6b790`,
with `--as-of 2026-10-22`, no member/ticker scope and no unpromoted filings:

| Path | Result | Receipt verification |
| --- | --- | --- |
| Weighted setups | 32,040 evidence rows; 113 ranked tickers; 11,762 blocked actor rows | Integrity and recorded-recipe replay pass |
| 28-day, three-distinct-buyer consensus | 1,333 disclosures; 55 preliminary eligible ticker candidates; **zero positive candidates** | Integrity passes; snapshot-only, not independent scorer replay |

The prior receipt ledger bytes remain unchanged; two new content-addressed
receipts were appended. All 113 ranked names have an explicit comparable
price-generation state. This does not certify their issuer identity,
freshness, economic interpretation or investment suitability.

The weighted top three are BA (4.94), BABA (3.90) and ADBE (3.64). Their
receipt flags show no fresh evidence; their quotes are October 20. They are
persistent estimated-holdings rankings, not three newly verified buy signals.
The output also retains known identity-sensitive SPCX exposure; it is not a
manual buy recommendation simply because 13F corroboration is disabled.

**No new purchase is recommended from this run.** The absence of positive
consensus results applies to available admitted data, not all upstream
filings or the whole market. Historical net-alpha evidence remains adverse
or inconclusive; the refresh did not establish a deployable edge.

**DX is watch-only**, not a $3,000 buy instruction: its three verified
common-share purchases produce weighted rank 47, score 0.888527, at the
October 21 close of $11.05. The business-risk checks above remain material.
APLD's two new officer purchases are blocked above entry tolerance. RAPP's
new officer purchase is blocked below the existing ranking floor. BORR's
officer event is already blocked for no open position, independently of the
additional cashless-exercise contextual hold. No sell/exit instruction is
implied for positions the user already owns.

Receipts under `data/recommendations/2026-10-22/`:

- Weighted: `61d38669b4259db57ebc01317721be0fd6ed596f41b77ac8810cb503fcc33ef1.json`.
- Consensus: `fefbdeebaee53d43046b4214b057e8de227a92064c104eeb0e4a98839bb22729.json`.

## Evidence

Detailed local receipts are under
`/tmp/opencode/upstream-score-refresh-20261022-l19cbknh/`.
Verified durable copies are under `data/backups/public-refresh/`, including:

- `20261022-precanonical-source-evidence.tar.gz` — 52 verified members.
- `20261022-form4-capture-recovery-and-admission.tar.gz` — 4,371 members;
  SHA-256 `e75421579719725f90bc613bb013ff07b37e14b781f6cf1894b61b88d79c4ef0`.
- `20261022-form4-admitted-and-quarantine.tar.gz` — six members;
  SHA-256 `76c5694ab70f0e030b87518a470ae21525368cdda55c5fb8d0b78ce331c2697e`.
- `20261022-business-risk-and-focused-price-evidence.tar.gz` — 11 members;
  SHA-256 `772a4ae1f0477eb454c4192141669418591518fc010577250aad2486f6127b6f`.
- `20261022-reviewed-isolated-house-v7-candidate.tar.gz` — eight members;
  approved code/fixtures only, not deployed source or admitted transactions.
- `20261022-final-refresh-receipts-and-evidence.tar.gz` — final run reports,
  original failed-operation outputs, two decision receipts, receipt ledger
  and this report. Its separate manifest records every member and archive hash.

Some older harness-managed stdout files were unavailable when the final
archive was assembled; its availability record names those gaps. No claim of
byte recovery is made for missing harness output. Captured source artifacts,
the critical post-merge failure output and a fresh final focused test run are
preserved separately from that limitation.

Consistent, hash-verified database backups precede the House, SEC and price
maintenance steps. These are local backups, not an off-host durability claim.
The final database is preserved as
`congress-after-partial-refresh-and-scoring-20261022.duckdb` (78,388 canonical
transactions; 8,670,231 prices), SHA-256
`f42ad672b49ce01f2af170e3084681a058c1eba24d409611c1c67c166563eede`.
The original interrupted price source is separately retained as
`prices-interrupted-sweep-20261022.duckdb`, byte-identical SHA-256
`a624abeefa7245fa35c438237b5dc0e423fa4a1290184e86d08174bf53f9ee4a`.
