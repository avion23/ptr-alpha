# FP decision review: GLM-5.3-flash as backlog OCR engine (Sol/GPT-6.1)

2026-10-18. Full text of the first-principles decision review (phases 0-6),
lightly formatted. Verdicts: ADOPT-WITH-CONDITIONS on GLM-first; REJECT on
unconditional trust and on the "no alternative" claim; key corrections to
the pilot's claims included.

## 0 — Requirements · ADOPT-WITH-CONDITIONS
- $0 incremental; CPU-only box; historical data lower-value: REAL constraints; subscription capacity UNKNOWN.
- Protect canonical/backtests from invented or materially incorrect rows: independent pixel-ground-truth audit of ALL persisted fields; acceptance budget explicitly approved — UNKNOWN (owner hasn't set a budget).
- Beat existing drain without weakening deterministic parsing: measured sustained throughput >190 docs/day across quota resets — UNKNOWN.
- Objective: maximize useful, trustworthy historical recovery — not documents processed. Poison-vs-gap asymmetry ⇒ correctness outranks drain speed; marginal historical value permits leaving gaps.

## 1 — First principles · Q1 ADOPT-WITH-CONDITIONS
REAL: ParseBench Tables GLM 89.18 vs Gemini Lite 85.48 (+3.70); GLM Content
Faithfulness 88.34 vs Gemini 89.46. UNKNOWN: superiority on PTR scans —
Tables measures structural accuracy of merged cells/hierarchical headers,
not end-to-end correctness of transaction tuples; simple structure removes
benchmark difficulty but not degraded-pixel, date, amount, or
row-association errors. Use for candidate selection, not certification.
FALSE in the published snapshot: GLM is rank 36 by overall score when
sorted properly, not rank 25 (site display order; no effect on the
decision).

## 2 — Logic · Q2 REJECT unconditional trust
REAL as a documented measurement: 11 documents, 10 AGREE, one A_ONLY, zero
disagreements; independent correctness was NOT measured. The pilot licenses
endpoint/schema interoperability, not safe extraction of 3,931 documents.
- Correction: the production gate compares asset, type, transaction date,
  notification date, AND amount — including multiplicity (vision_agreement.py),
  which is stronger than the pilot's (type,date)-only check.
- "Both must agree before persistence" is FALSE for the complete
  architecture: B absent/unavailable explicitly permits A-only persistence,
  and A failure never attempts B. Availability fallback removes the
  protection exactly when it matters.
- Grounding generated text cannot independently establish what the pixels
  say; the text-born n=161 result does not validate scanned documents.
- Single-engine extraction can introduce both poison and gaps; dual
  quarantine converts detected discrepancies into gaps, but shared mistakes
  still escape.

## 3 — Mathematics · Q2 REJECT "≥95% established"
10/10 paired agreements: two-sided 95% Wilson interval [72.25%, 100%];
Jeffreys [78.28%, 99.995%]; one-sided exact 95% lower bound 74.11%. 59/59
would establish agreement above 95% under independent, representative
sampling — and that is agreement, not correctness. The A_ONLY fallback is
excluded from that denominator; paired-agreement yield across attempted
docs is 10/11.
Different model families do NOT establish error independence:
P(A wrong, B wrong) = P(A wrong)·P(B wrong | A wrong) — shared
illegibility/template errors can produce identical wrong rows; the
independence multiplication is unsupported.

## 4 — CS/algorithms · Q3 ADOPT-WITH-CONDITIONS
Adopt GLM-first backlog reading; REJECT immediate unaudited single-engine
canonical promotion. Preserve existing validation; quota/schema/empty-output
failures must remain unresolved rather than becoming successful empty
filings; stop on quota with no paid overflow. Pilot bounded batches before
committing the whole drain. "Dual halves throughput" is UNKNOWN — at 190
Gemini docs/day, 3,931 docs take 20.7 days; GLM five-hour windows establish
no competing capacity. Require measured >190/day across resets before
claiming a throughput win.

## 5 — Data design · Q5 ADOPT-WITH-CONDITIONS
The value-at-stake split holds: backlog GLM-only candidates should be
STAGED — explicitly excluded from canonical/backtests until an acceptance
rule passes — retaining PDF identity, raw response, model/prompt/parser
identity, validation outcome, audit status. Current-era scans: require
actual dual agreement or manual review, INCLUDING reader outages — not
silent single-engine fallback.

## 6 — Empirical · Q4 REJECT the universal "no alternative" claim
REAL: none of the ParseBench-listed alternatives scores Tables >89.2.
UNKNOWN: whether any $0 usable alternative exceeds it on this box (feasibility
research covered by a parallel agent). Promotion condition: the owner must
approve an error budget — an illustrative <1% materially-wrong-document rate
requires 299 independently sampled documents with ZERO material errors for a
one-sided exact 95% bound (0.997%) — and even that permits substantial
backlog contamination. Audit against pixels; count omissions separately;
stratify by year/template/scan-quality.

## Ruling
- GLM-first provisional drain: YES (bounded batches).
- Flip all historical canonical ingestion to GLM alone: NOT YET.
- Strict dual/manual review for current-era scans: YES.
- One measurement that could change the canonical decision: a
  representative, independently adjudicated GLM material-error rate against
  an owner-approved budget.
