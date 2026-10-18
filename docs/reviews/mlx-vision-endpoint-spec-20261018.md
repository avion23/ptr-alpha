# MLX vision endpoint spec: independent second reader (Model B)

2026-10-18. Adaptation spec for the operator's Apple Silicon (`statement2csv`,
MLX) box serving an independent second vision read of scanned PTR filings,
per `docs/reviews/extraction-pipeline-fp-20261018.md`. Phase 1 requires **zero
statement2csv code**: `mlx-vlm` already ships an OpenAI-compatible FastAPI
vision server (`/v1/chat/completions` with base64 `image_url` parts, JSON
schema structured outputs, bearer auth, `/health`, `/v1/metrics`).
(mlx-vlm server verified: github.com/Blaizzy/mlx-vlm; `mlx_lm.server` is
text-only.)

## Model recommendation (MLX ecosystem)

| role | model | size / RAM | notes |
|---|---|---|---|
| **PRIMARY** | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` | 5.65 GB weights, ~7–9 GB peak | strongest DocVQA-class here; follows the existing PTR prompt directly; needs ≥16 GB Mac; ~5–20 s/page single-stream (prefill-bound; UNKNOWN until measured on target Mac) |
| **FALLBACK** | `mlx-community/GLM-OCR-4bit` (MIT) | 1.25 GB, ~2–3 GB peak | OmniDocBench V1.5 #1; parse-only (fixed prompts) → degraded mode feeding the client's deterministic parser |
| evaluated, held | PaddleOCR-VL-1.6 (Apache-2.0) | 0.9B | no advantage over GLM-OCR here |
| ruled out | MinerU-class | — | not MLX-native (own vLLM/SGLang pipeline) |
| excluded | olmOCR 2 / DeepSeek-OCR / Chandra | 3–7B | need CUDA |

Different model family from Gemini on both choices — the agreement gate is
meaningful.

## API contract (stock server; do not invent a protocol for phase 1)

Launch on the Mac:
```bash
caffeinate -is uv run --with mlx-vlm==0.7.8 python -m mlx_vlm.server \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --host 0.0.0.0 --port 8080 --api-key "$PTR_VISION_TOKEN" --max-pixels 4194304
```
- Request: `POST /v1/chat/completions`, `Authorization: Bearer <token>`, one
  `image_url` part per page (base64 `data:image/png`), temperature 0, the SAME
  PTR prompt text as `scripts/gemini_ocr_common.py` (prompt stays versioned on
  the Linux side; `PROMPT_SHA256` cache keys carry over).
- Response: standard chat-completions; feed `choices[0].message.content` to
  the existing `parse_gemini_output()` (preamble-tolerant, page-coverage
  checks). Optional hardening: `response_format json_schema` (multimodal-
  supported; not combinable with `--draft-model`).
- Client (this box): rasterize via `pdftoppm -r 200 -png` (installed); the Mac
  needs no PDF dependency. Client timeout 180 s/page (600 s multi-image).
- Error contract: connection refused/timeout (Mac asleep) =
  `reader_unavailable` → Gemini-only proceeds, doc queued `pending_dual_read`
  (by design, not failure). 401/403 = config error, alert. 400 = our bug.
  200-but-unparseable = 1 retry at higher max_tokens, then B-failed.
- Networking: Tailscale preferred; token enforced regardless. Reachability:
  `http://<mac-tailscale-name>:8080/v1`.

## statement2csv adaptation (phase 2 only)

Phase 1: none. Phase 2 (if the Mac should own PDF→rows): reuse its
rasterization + prompt + validation; add a ~100-line FastAPI `/ocr` wrapper
(base64 pages in, validated rows JSON + parser_version out); replace file I/O
with HTTP. Client-side new module `scripts/vision_mlx.py` beside
`gemini_ocr_common.py` with its own cache dir and parser_version
(`v1-mlx-qwen25vl`).

## Effort and risks

Phase 1 (stock server + this box's reader + agreement gate): 3–5 h. Phase 2
(`/ocr` in statement2csv): +6–10 h.

Top risks: (1) agreement-gate pass rate on checkbox direction attribution —
run Qwen2.5-VL-4bit over a 20–30 doc gold set BEFORE wiring the gate;
(2) M-series throughput is prefill-bound and unbenchmarked for this
model/chip — page-wise fan-out and async backfill semantics absorb it;
(3) mlx-vlm release churn — pin `mlx-vlm==0.7.8`, keep the smoke-test curl,
client treats the endpoint as optional-at-runtime (breakage degrades to
Gemini-only).

Runbook (on the Mac): (1) one-time model download via the launch command;
(2) daily serve under `caffeinate -is`; (3) `curl -H "Authorization: Bearer
<token>" http://<mac>:8080/health`.
