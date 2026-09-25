# Run log: full rescrape → merge → promote (2026-09-21 → 2026-09-25)

Live DB `data/congress.duckdb` went 77,301 → 95,577 → **119,213 transactions**
(canonical 43,134 → 57,999; prices 2,710,658 → 6,337,398).

## Pipeline
1. Rescraped House PTR 2015–2026 + Senate eFD into staging generation
   `data/.staging/rescrape/rescrape-20260921T224104/` (isolated `--data-dir`
   per year/window; live DB untouched throughout).
2. House parse: tuned Tesseract cascade (see perf fixes below); Gemini OCR
   direct path (`scripts/ocr_zero_rows.py::run_gemini_ocr_for_year`) for
   cascade-immune docs — 2015: 118→8,581 tx, 2023: 2,449→8,637 tx.
   Free-tier quota (500 req/day, gemini-3.1-flash-lite) gates the ~4.4k-doc
   remainder; daily 04:00 UTC cron (`data/logs/gemini-cron/daily.sh`) grinds it.
3. Senate: 14,865 canonical tx across disjoint clean windows (monthly isolation
   for 2015–2017). 12 reports stay fail-closed (malformed rows / duplicate row
   IDs — diagnosed unrecoverable, validation intact).
4. Consolidated staging → `final/congress.duckdb` (25,730 tx), refreshed prices
   (5.53M rows, snapshot manifest), then MERGED (never swapped: staging was a
   subset of live): merge-1 77,301→95,577 (+18,276), merge-2 95,577→119,213
   (+23,636), additive-only, NULL-safe anti-join verified 0 missing both times.

## Perf fixes (committed)
- `e53a8d44` OMP_THREAD_LIMIT=1 default (12-thread storm on 4 cores → 4× speedup)
- `f4752f2` OSD thumbnail 600→1200px (600px broke OSD on sparse forms)
- Found: `config.toml` beats env, so `DATA__PARALLEL_WORKERS` was silently
  ignored — override via per-run workdir config.toml.

## Revert
Backups: `data/congress.duckdb.backup-prerescape-20260921T224104`,
`...backup-premerge-20260922T192700`, `...backup-premerge2-20260925T193708`.
Revert: `mv <backup> data/congress.duckdb` with no process holding it.

## Scheduled
- Prices refresh: Monday 02:00 UTC (`data/logs/prices-cron/weekly.sh`).
- Gemini OCR grind: daily 04:00 UTC (`data/logs/gemini-cron/daily.sh`).
- Merge cron OCR gains to live on demand (additive merge pattern, manifests
  in `.../final/merge_manifest*.json`).
