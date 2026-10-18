#!/usr/bin/env bash
# Nightly bounded GLM OCR drain of the zero-row PDF backlog into STAGING
# (scripts/glm_stage.py — GLM-only rows are staged, excluded from canonical
# until the pixel audit passes; see docs/reviews/glm-engine-fp-review-20261018.md).
# Verification step: scripts/doctor.py at the end (cron contract).
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

# Engine-B credentials + mode (outside the repo; missing file = no-op drain)
if [ -f "${HOME}/.config/ptr-backlog/env" ]; then
    . "${HOME}/.config/ptr-backlog/env"
fi
export DUAL_READ VISION_B_BASE_URL VISION_B_MODEL VISION_B_API_KEY VISION_B_TIMEOUT
printf 'dual-read: DUAL_READ=%s engine_b_configured=%s\n' \
    "${DUAL_READ:-<unset>}" "$([ -n "${VISION_B_API_KEY:-}" ] && echo yes || echo no)"
cd "$repo"

# Drain year-by-year into staging; quota exhaustion stops each year cleanly.
for year in 2017 2016 2015 2014 2013 2018; do
    printf '--- staging year %s ---\n' "$year"
    timeout 10000 python3 scripts/glm_stage.py stage --year "$year" --limit 1500 \
        || printf 'year %s stage exit=%d (quota/timeout tolerated)\n' "$year" "$?"
done

# Consumption/verification step: canonical health after the drain.
python3 scripts/doctor.py
doctor_rc=$?
printf 'doctor exit=%d\n' "$doctor_rc"

# Prune to the last 7 run dirs.
find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '????????' |
    sort -r | tail -n +8 | xargs -r rm -rf --
exit "$doctor_rc"
