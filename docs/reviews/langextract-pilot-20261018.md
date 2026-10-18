# LangExtract pilot: grounded extraction vs deterministic ground truth

2026-10-18. Question: does Google's LangExtract (Apache-2.0, source-grounded
LLM extraction) earn a place in the parse pipeline — and where?

## Setup

20 2026 House docs already parsed by the deterministic cascade (v5-
deterministic, 1-62 rows each — ground truth held), pdftotext text, LangExtract
v1.7.1 with `gemini-3.1-flash-lite` (the model already in use), 1 extraction
pass, transaction schema (asset verbatim, type, date), grounding filter on
(`char_interval is None` → dropped). 13 docs usable (7 PDFs absent from the
2026 cache dir); 161 ground-truth transactions.

## Results

- **(type, date) pair precision: 161/161 = 100%.** Every LangExtract
  transaction matched a deterministic ground-truth row.
- **(type, date) pair recall: 161/161 = 100%.** No ground-truth row missed
  (docs up to 62 rows).
- **Grounding filter: 143/161 kept (88.8%).** The 18 dropped extractions were
  ungrounded duplicates (locatable nowhere in the source); dropping them cost
  zero recall — mechanical dedup + anti-hallucination in one step.
- **Asset text: verbatim match** with canonical's stored `asset_description`
  (spot-checked side-by-side; the strict-triple check that read 0 was
  comparison strictness, corrected by inspection).

## Where it fits (and does not)

- **Does NOT replace the deterministic cascade** — pdfplumber/camelot/
  pdftotext are exact, free, quota-independent, and remain the primary path
  (~92% of 2026 docs).
- **Fits as the trust layer on the LLM path**: for rows produced from
  scanned/zero-row PDFs via the vision pass, require grounded extraction —
  ungrounded rows go to quarantine, never silently into canonical. Store the
  character offsets; `lx.visualize()` gives the manual-review queue a
  click-to-highlight review surface (2026 has 8 manual-review docs).
- **Caveats**: text-in only (scanned docs still need the vision pass first);
  1-pass quota cost is comparable to the current prompt; n=161 rows on
  text-born docs is a small sample — the scanned-backlog population (the
  actual hard case) needs the same A/B before any accuracy claim there.

## Decision

Adopt as the LLM-path validation layer: wire grounded extraction into the
Gemini OCR cascade as a second opinion (disagreement with the vision-pass
rows → quarantine + HTML), and use the grounding filter as the hallucination
gate for scan-born rows. Revisit accuracy on true scanned images with the
same A/B once the backlog drain surfaces enough dual-parsed docs.
