# FP review: OCR pipeline setup and workflow (Astra/GPT-6 Astra)

2026-10-18. Reviewed clean HEAD `00450b30` (+ working tree). Objective:
maximize correctly recovered canonical transactions; correctness and explicit
uncertainty outrank throughput. No repo edits; staging data observed changing
during review (drain was live).

## Requirements verdicts
| Requirement | Status |
|---|---|
| Correct canonical transactions | **FAIL** — findings 2-4 |
| Fail-closed publication | **FAIL** — findings 1, 5, 7 |
| Defensible audit acceptance | **FAIL** — current report upper bound 71.08%; owner budget UNKNOWN |
| Reproducibility and recovery | **FAIL** — finding 6 |
| Bounded, observable operation | **FAIL** — findings 8, 10-13 |

## Architecture ruling
ADOPT the staging→audit→promote separation; REJECT its current implementation
as fail-closed. No replacement database, GPU, paid model, or orchestration
framework justified. Sound design: deterministic processing → artifact-bound
unresolved-work manifest → retained OCR candidate → exact-candidate
adjudication → one mandatory transactional publisher. Current-era scans
require actual dual agreement or manual adjudication including outages.
Backlog: individually verified documents considered independently;
extrapolation to unaudited documents requires a frozen extraction recipe,
representative acceptance evidence, and an explicit owner policy.

## Findings
1. **[CRITICAL]** Audit approval optional (`require_audit_pass=False` default,
   `glm_stage.py:493-500,531-542,708-711`); with the flag it checks only
   `audit=="pass"`, never the sample report. Separately
   `ocr_zero_rows.py:1840-1845,1909-1913,1976-1988` still provides GLM-only
   direct-to-canonical ingestion. Fix: mandatory approval at the shared GLM
   publication boundary; direct the legacy route to staging.
2. **[CRITICAL]** Canonical acceptance leaks across source rowsets:
   `database.py:443-529` accepts via one source's parse run but `:541-555`
   admits ALL House/OCR rows for the accepted artifact. Reproduced: one
   accepted deterministic run returned its row PLUS an unaccepted OCR row
   ("WRONG OTHER SECURITY (MSFT)"). Fix: bind admission to the accepted
   rowset; one authoritative rowset per filing; superseded evidence preserved
   outside the selection; same rule live and historical.
3. **[CRITICAL]** The deterministic cascade accepts demonstrated corruption:
   `_parse_text_only_worker(data/2017/pdfs/20006438.pdf)` accepted 16 rows,
   omitted visible ETP and PSXP transactions, duplicated lots;
   `pdfplumber_parser.py:40-45,76-94` merges unmatched observations;
   `parser_cascade.py:327-365` accepts subset relationships;
   `parsing/cells.py:77-81` turns `FILING STATUS:` into ticker `STATUS`.
   Fix: repair physical-row extraction; quarantine unexplained
   count/content disagreement; preserve legitimate identical lots via source
   coordinates.
4. **[CRITICAL]** Promotion can consume a different result than audited, or a
   failed empty insertion: audited-PASS 20006470 has 22 rows;
   `validate_transactions()` retains 20 (two impossible chronologies
   rejected — source-data issues); `ocr_zero_rows.py:1404-1434,1693-1706`
   records partial insertion as success; `glm_stage.py:582-600` accepts any
   positive count and deletes staging; `no_txs` conflates with member
   rejection. Fix: explicit committed outcomes; retain rejected rows.
5. **[CRITICAL]** "Both readers must agree" is false during outages:
   `vision_engines.py:316-358` returns A-only for missing config/timeouts/
   HTTP failures/malformed envelopes (offline probe: A_ONLY persisted on B
   HTTP 500); the year runner loses B-quota batch-stop. Fix: one strict dual
   policy for both callers — unavailable is unresolved, not agreement.
6. **[CRITICAL]** Mutable staging + unbound verdicts: verdicts stamp whatever
   record exists; records overwrite; DPI absent; page identities discarded;
   invalid responses never reach raw_output; hashing reads the live PDF
   separately. W3 correction: the three "not configured" errors were real
   config-missing (model null). Fix: versioned attempts, snapshot inputs
   once, bind verdicts to the exact reviewed attempt, retain evidence after
   promotion.
7. **[CRITICAL]** The API client accepts truncated completions (no
   `finish_reason` check — a `length`-truncated response was accepted) and
   discards non-200 bodies (destroying error classification); transient
   failures become persistent error files that selection permanently skips.
   Z.AI error codes distinguish temporary overload from exhausted
   quota/invalid config. Fix: reject incomplete completions; retain error
   metadata; bounded retries with jitter and reusable renders; stop globally
   on config/auth defects.
8. **[MAJOR]** Staging doesn't drain the unresolved queue: selects every
   local PDF absent a staging file (text-born docs included — which is why
   the audit sample was all text-born); cron omits `--check-canonical`;
   promotion makes docs eligible again. Fix: reuse authoritative unresolved
   selection as an artifact-bound manifest; persist disposition.
9. **[CRITICAL]** The audit report can manufacture acceptance: latest verdict
   replaces earlier failures; missing verdicts count as non-failures;
   omission detection only by row-count inequality; promotion removes the
   rows needed for comparison; equal per-year samples pooled without
   population weighting; recipe/DPI changes untracked. Probes: 299 null
   verdicts → budget_met=True; 299 failures then passes → zero failures.
   Fix: artifact-bound observations, row-multiset comparison, preserve
   original failures, frozen eligible lot + uniform random sample, blinded
   transcription, human-adjudicated checks of the LLM auditor.
10. **[MAJOR]** The 6h/2-stream ceiling is not enforced: six separate
    10,000s years = 16h40m; quota returns 0 → wrapper continues; CLI accepts
    workers>2; no shared lock vs overlapping runs; `printf '---'` bug.
11. **[MAJOR]** Long canonical lock ownership blocks unrelated operations;
    handling inconsistent (verified RO failure during all-years refresh).
    `glm_stage.py status` already works without canonical.
12. **[MAJOR]** Monitoring doesn't observe the new pipeline: doctor checks
    retired-style staging DBs, not glm-backlog staging; an entirely broken
    OCR drain can appear successful (stage exits 0 with doc errors; cron
    returns only doctor status).
13. **[MAJOR]** Scheduled acquisition and historical scope don't match the
    stated goal: no current-year House acquisition cron; backlog covers
    2013-2018 while local PDFs span 2015-2026; missing-year dirs silently
    yield zero. Also reconcile which recommendation path is authoritative.

## Prioritized fix order
P0: publication bypasses + rowset leakage 1→2→5→7; quarantine deterministic
defects and fix audited-vs-inserted outcomes 3→4.
P1: attempts/verdict binding + audit-report validity 6→9; queue selection +
transient retries 8→7 before further bulk inference.
P2: runtime limits, lock coordination, ingestion cadence, monitoring
10→11→13→12. Then freeze the recipe and run a scan-representative holdout;
300-DPI crops only for a named legibility failure (300 DPI alone did NOT fix
the character errors — Sol re-audit confirmed identical failures).

## UNKNOWNs
Live canonical contamination magnitude (canonical was locked); actual
scanned-backlog size and the true scanned/text split; owner-approved
material-error budget; sustained subscription throughput; provider payload
limits. Existing evidence establishes neither scan accuracy nor a DPI cause
for W3.
