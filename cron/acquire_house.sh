#!/usr/bin/env bash
# Current-year House PTR acquisition: fetch → parse → Gemini OCR for zero-row
# docs (the pipeline's primary purpose — catching CURRENT skilled-insider
# filings). NOT installed by default: it writes canonical, so it must not
# overlap `refresh --all-years` or the manual drains. Proposed schedule:
# twice weekly (e.g. 04:30 UTC Tue/Sat) after the backlog drain completes.
# Activate with: bash cron/install.sh -y   (after adding to cron/crontab.txt)
set -u

readonly repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly log_root="$repo/data/logs/acquire-house-cron"
readonly git_sha="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
run_date="$(date -u +%Y%m%d)"
readonly run_date
readonly run_dir="$log_root/$run_date"
year="$(date -u +%Y)"
readonly year

mkdir -p "$run_dir"
exec >>"$run_dir/run.log" 2>&1
printf 'acquire-house start: %s UTC (repo %s @ %s, year %s)\n' \
    "$(date -u --iso-8601=seconds)" "$repo" "$git_sha" "$year"
cd "$repo"

# Refuse to run while another canonical writer is active (best-effort: the
# all-years refresh and the backlog drain hold long write windows).
if pgrep -f 'refresh --all-years|glm_stage.py stage' >/dev/null 2>&1; then
    printf 'another canonical writer is active; skipping this window\n'
    exit 0
fi

timeout 10000 python3 -m analyzer.cli refresh --year "$year" --gemini-ocr
rc=$?
printf 'acquire-house done: exit=%d\n' "$rc"

# Prune to the last 7 run dirs.
find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '????????' |
    sort -r | tail -n +8 | xargs -r rm -rf --
exit "$rc"
