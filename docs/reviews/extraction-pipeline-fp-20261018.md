# First-principles review: the document extraction pipeline (LangExtract, vLLM)

2026-10-18. Question: should the extraction pipeline adopt LangExtract, and
what role should a vLLM-served model play? Phases in order; evidence or
UNKNOWN, never "likely".

## 0. Requirements

| # | Requirement | Check | Status |
|---|---|---|---|
| R1 | No fabricated row enters canonical, ever | grounding gate + dual-model agreement + quarantine | PARTIAL — deterministic path safe by construction; vision path heuristics-only today |
| R2 | Every stored row traceable to its source | char offsets (text stage); pixel provenance (vision stage) | PARTIAL — grounding validates extraction-vs-text, NOT text-vs-pixels |
| R3 | Recall on parseable docs ≥ deterministic level | A/B vs v5-deterministic ground truth | MEASURED: 100% precision/recall, n=161 text-born rows (langextract-pilot-20261018.md); scanned population UNKNOWN |
| R4 | Backlog drains without a quota ceiling | throughput measurement | UNKNOWN pending vLLM endpoint |
| R5 | Reproducibility: parser_version pins engine+prompt+model | parse_runs schema | PASS (existing discipline; extend to new engines) |
| R6 | Deterministic cascade stays primary for text-born PDFs | unchanged by adoption | PASS by design |

Conflicts: throughput (quota-free local inference) vs accuracy assurance
( cloud model quality) — resolved by *dual independent reads + agreement
gate*, not by picking one.

## 1. First principles

The pipeline estimates transaction tuples from two information classes:

- **(a) Text-born PDFs**: an exact text layer exists. Extraction is
  deterministic parsing; inference adds risk, zero information. LLMs have no
  role here beyond an optional cross-check.
- **(b) Scanned images**: no text layer. Vision inference is *required* — the
  question is which model serves it and how its output is verified.

The core principle: **inference is acceptable exactly where its claims are
independently checkable.** LangExtract's grounding converts structure-stage
inference ("these fields came from this text") into lookup-verifiable claims.
It does NOT verify the vision stage (text-vs-pixels): grounding anchors to the
vision model's *own* text, so a hallucinated reading passes grounding
circularly. Vision-stage fabrication is caught only by an *independent second
vision read*.

## 2. Logic — pipeline graph after adoption

```
PDF → deterministic cascade → success ⇒ canonical (unchanged, ~92% of docs)
  ↓ fail
rasterize → vision read A (Gemini, cloud)     ┐ independent reads
          → vision read B (vLLM-served model) ┘ (different model families)
  → both structure via LangExtract (grounded, ungrounded → quarantine)
  → agree ⇒ canonical · disagree ⇒ quarantine + review HTML
  → endpoint B unavailable ⇒ fall back to A alone (current behavior)
```

Every new artifact is consumed: rows → canonical; disagreements → review
queue; HTML → operator. Nothing unconsumed.

## 3. Mathematics

- Hallucination escape probability: current = P(halluc) × P(heuristic miss).
  With grounding: escape requires the hallucinated text to *also* appear in
  the source (≈0 for fabricated tickers; duplicates dedupe naturally —
  measured: 18 ungrounded drops, zero recall cost). With dual independent
  vision reads + agreement gate: escape requires *both* models to fabricate
  the *same* wrong tuple — for independent model families this multiplies
  small probabilities.
- Throughput: 3,931 backlog docs. vLLM + Qwen2.5-VL-7B-class on one 24GB GPU
  ≈ 1–3 s/page → backlog in ~2–6 h. Gemini at quota: ~3 weeks. LangExtract
  adds 1–3 structure calls per doc (or runs against the same vLLM endpoint
  free).

## 4. CS/ALGORITHMS

vLLM exposes an OpenAI-compatible API; LangExtract consumes custom providers
(and litellm-style `gemini/...` routing already exists in
`gemini_ocr_common.py`). Integration shape: endpoint URL + model id as
config; parser_version `v6-vllm-validated`; response cache keyed by
(artifact_sha256, model_id, prompt_version) — the cache-envelope machinery
already exists. Fallback ordering: deterministic → vLLM → Gemini →
quarantine. Batching: vLLM throughput scales with batch; nightly cron
unchanged.

## 5. DATA DESIGN

Provenance: store char offsets + both models' raw readings for disagreed
docs (side table keyed by doc_id; no canonical schema change). Cache:
immutable, keyed as above. Never store "consensus" edits — store both reads
and the decision.

## 6. EMPIRICAL

Measured tonight: LangExtract text-born A/B = 100% precision/recall (n=161),
grounding 88.8% kept (18 ungrounded duplicates dropped, zero recall cost),
asset text verbatim-identical (langextract-pilot-20261018.md). NOT measured:
scanned-image population accuracy (the actual hard case), vLLM throughput on
this workload, dual-read agreement rate. Pre-registered pilot: 30 backlog
docs dual-read (A/B models) + the 13-doc text A/B re-run through vLLM;
adopt if agreement ≥95% on text-born and backlog throughput ≥500 docs/day.

## Decision update 2 (2026-10-18 evening): engine B is LIVE via the Z.AI
## subscription — dual-read pilot passed at 100%

The operator's existing Z.AI subscription exposed `glm-5.3-flash` with
vision (ParseBench Tables 89.2 — better than engine A's 85.5) at $0
incremental. Compare-only dual-read pilot on 11 scanned backlog docs
(2016–2017 scans): **AGREE 10, DISAGREE 0, A_ONLY 1** (B returned nothing
for one doc; fell back exactly as designed) — **agreement 100%**, bar ≥95%.
Zero errors. The gate is no longer hypothetical: both engines read real
rotten scans and normalized to identical transaction multisets.

Wiring in progress: `main()` (the nightly backlog cron path) routes through
the dual-read orchestrator when `DUAL_READ=1` + engine-B env are present;
credentials live outside the repo (`~/.config/ptr-backlog/env`, 0600,
sourced by `cron/backlog_ocr.sh` — never committed). The MLX/Mac endpoint
spec (`mlx-vision-endpoint-spec-20261018.md`) becomes a third reader /
fallback for a future GPU-less-Mac scenario, not a requirement.

## Decision update (2026-10-18, post-ParseBench)

ParseBench (parsebench.ai, 147 methods) changes decision 2: the operator
declined paid tiers, and the free-optimum analysis shows the current engine
(Gemini 3.1 Flash Lite, Tables 85.5, 0.3¢/page) is already top-quartile —
within ~4 points of the paid table leaders (LlamaParse Agentic Plus 93.4 at
$450/backlog, Opus 5.5 94.3 at ~$490), and AHEAD of every CPU-viable open
model (PaddleOCR-VL 67.8, GLM-OCR 66.1) and of LlamaParse's free-tier mode
(Cost Effective, 84.2, which also drains slower than the existing Gemini
nightly quota). PTR filings are the easy end of the benchmark (simple
columns; the separating hard cases — merged cells, cross-page tables,
charts — do not occur in them).

**Final: keep the existing nightly Gemini drain as the scanned-backlog
engine. No spend, no new machinery.** Decision 1 (LangExtract grounding
gate on the LLM path) stands unchanged. The vLLM/MLX second-reader spec
(mlx-vision-endpoint-spec-20261018.md) stays on the shelf for a future GPU
box; the dual-read client built for it is engine-agnostic and costs nothing
to keep.
