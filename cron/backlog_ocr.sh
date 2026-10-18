#!/usr/bin/env bash
# Daily bounded Gemini OCR drain of the zero-row PDF backlog, writing
# directly into canonical (scripts/ocr_zero_rows.py inserts per doc; the
# batch stops itself cleanly when the Gemini quota is exhausted — a normal,
# expected outcome, not a failure). Verification step: scripts/doctor.py at
# the end (canonical consumption/verification per the cron contract).
set -u

readonly repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly log_root="$repo/data/logs/ocr-backlog-cron"
readonly git_sha="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
run_date="$(date -u +%Y%m%d)"
readonly run_date
readonly run_dir="$log_root/$run_date"

mkdir -p "$run_dir"
exec >>"$run_dir/run.log" 2>&1
printf 'ocr backlog start: %s UTC (repo %s @ %s)\n' \
    "$(date -u --iso-8601=seconds)" "$repo" "$git_sha"
cd "$repo"

# Engine-B dual-read credentials live outside the repo (0600, operator-
# managed). Missing file means a plain single-engine drain, exactly as
# before — not an error.
if [ -f "${HOME}/.config/ptr-backlog/env" ]; then
    # shellcheck disable=SC1091  # machine-local operator file
    . "${HOME}/.config/ptr-backlog/env"
fi
# Mark the (possibly non-exporting) operator file for the child interpreter.
export DUAL_READ VISION_B_BASE_URL VISION_B_MODEL VISION_B_TIMEOUT VISION_B_API_KEY
engine_b="no"
if [ -n "${VISION_B_BASE_URL:-}" ] && [ -n "${VISION_B_MODEL:-}" ]; then
    engine_b="yes"
fi
printf 'dual-read: DUAL_READ=%s engine_b_configured=%s\n' \
    "${DUAL_READ:-}" "$engine_b"

timeout 21600 python3 scripts/ocr_zero_rows.py
ocr_rc=$?
printf 'ocr backlog done: exit=%d\n' "$ocr_rc"

# Consumption/verification step: canonical health after the inserts.
python3 scripts/doctor.py
doctor_rc=$?
printf 'doctor exit=%d\n' "$doctor_rc"

# Prune to the last 7 run dirs.
find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '????????' |
    sort -r | tail -n +8 | xargs -r rm -rf --
exit "$doctor_rc"
