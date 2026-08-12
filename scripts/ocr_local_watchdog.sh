#!/bin/bash
# Daily-resume watchdog for the local free OCR second pass (House PTR).
#
# The sweep is resumable: each document is staged atomically as it completes
# and --skip-staged replays only the remainder.  Year order is ASCENDING
# (2015-2020 priority first per the go-live plan, then recent years); the
# manifest is re-assembled at the end of every cycle so luna-finalize can
# consume a consistent snapshot at any time.
#
# Usage:
#   OCR_LOCAL_SWEEP_DATA=<gen-live dir> OCR_LOCAL_SWEEP_DB=<staged duckdb> \
#     OCR_LOCAL_SWEEP_MANIFEST=<rebuild2 manifest> \
#     OCR_LOCAL_SWEEP_WORKERS=3 ./scripts/ocr_local_watchdog.sh
set -u
cd "$(dirname "$0")/.."
PY=${OCR_LOCAL_SWEEP_PY:-/opt/homebrew/bin/python3.14}
DATA=${OCR_LOCAL_SWEEP_DATA:?set OCR_LOCAL_SWEEP_DATA (staged generation dir with <year>/pdfs)}
DB=${OCR_LOCAL_SWEEP_DB:?set OCR_LOCAL_SWEEP_DB (staged generation congress.duckdb)}
MANIFEST=${OCR_LOCAL_SWEEP_MANIFEST:?set OCR_LOCAL_SWEEP_MANIFEST (rebuild2 manifest.json)}
WORKERS=${OCR_LOCAL_SWEEP_WORKERS:-3}
OUT=.staging/ocr2-local/gen-live-20260810
LOG=${OCR_LOCAL_SWEEP_LOG:-/tmp/ocr_local_sweep.log}
TARGET=3445
MAX_CYCLES=24
# Ascending priority: 2015-2020 first (2,698 docs, largest unresolved share),
# then recent years.
YEARS="2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026"
for CYCLE in $(seq 1 $MAX_CYCLES); do
  echo "=== cycle $CYCLE $(date +%Y-%m-%dT%H:%M:%S%z) order=asc ===" | tee -a "$LOG"
  for YEAR in $YEARS; do
    $PY scripts/ocr_local_sweep.py sweep \
      --data-dir "$DATA" --db "$DB" --manifest "$MANIFEST" \
      --out "$OUT" --workers "$WORKERS" --years "$YEAR" --skip-staged \
      >> "$LOG" 2>&1
  done
  $PY scripts/ocr_local_sweep.py sweep --merge-only --out "$OUT" \
    --data-dir "$DATA" --db "$DB" --manifest "$MANIFEST" >> "$LOG" 2>&1
  # Reconcile no_txs evidence on the shared root (older sweep instances stage
  # cover-classified docs as no_txs; those must stay unresolved fail-closed).
  $PY scripts/ocr_local_reclassify_no_txs.py --out "$OUT" >> "$LOG" 2>&1
  STAGED=$(ls "$OUT/docs/" 2>/dev/null | wc -l | tr -d " ")
  echo "=== cycle $CYCLE staged=$STAGED/$TARGET $(date +%H:%M:%S) ===" | tee -a "$LOG"
  if [ "$STAGED" -ge "$TARGET" ]; then
    echo "ALL DOCS STAGED - DONE" | tee -a "$LOG"
    exit 0
  fi
  echo "=== cycle $CYCLE probing again in 3600s ===" | tee -a "$LOG"
  sleep 3600
done
echo "WATCHDOG EXHAUSTED MAX_CYCLES without completing $TARGET" | tee -a "$LOG"
exit 2
