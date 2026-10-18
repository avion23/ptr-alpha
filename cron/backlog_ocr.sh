#!/usr/bin/env bash
# Nightly bounded GLM OCR drain of the zero-row PDF backlog into STAGING
# (scripts/glm_stage.py — GLM-only rows are staged, excluded from canonical
# until the pixel audit passes; see docs/reviews/glm-engine-fp-review-20261018.md).
# Verification step: scripts/doctor.py at the end (cron contract).
# Bounding (review 2026-10-18 finding 10): ONE whole-run deadline shared
# across the year loop (per-year 10000s timeouts summed to 16h40m), a
# single-instance flock (cron vs manual drain = 2x the 2-stream budget), and
# quota-stop propagation (a dead quota makes later years fast-fail).
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

# Single-instance lock: the nightly 03:30 cron and an operator's manual run
# must never overlap (that doubles the 2-stream budget). Non-blocking: a busy
# lock means the same workload is already draining, so exit 0 quietly.
if command -v flock >/dev/null 2>&1; then
    exec 9>"$log_root/drain.lock"
    if ! flock -n 9; then
        printf '%s\n' "another drain is running (lock: $log_root/drain.lock); exiting"
        exit 0
    fi
else
    printf '%s\n' "WARNING: flock unavailable; running WITHOUT single-instance lock"
fi

# ONE whole-run deadline: every year invocation draws down a single 6h
# budget; the tail reserve keeps the doctor/prune steps inside it too.
readonly run_budget_seconds=21600
readonly deadline_epoch=$(( $(date -u +%s) + run_budget_seconds ))
readonly year_budget_cap=10000
readonly tail_reserve_seconds=300

# Engine-B credentials + mode (outside the repo; missing file = no-op drain)
if [ -f "${HOME}/.config/ptr-backlog/env" ]; then
    . "${HOME}/.config/ptr-backlog/env"
fi
export DUAL_READ VISION_B_BASE_URL VISION_B_MODEL VISION_B_API_KEY VISION_B_TIMEOUT
printf 'dual-read: DUAL_READ=%s engine_b_configured=%s\n' \
    "${DUAL_READ:-<unset>}" "$([ -n "${VISION_B_API_KEY:-}" ] && echo yes || echo no)"
cd "$repo"

# Drain year-by-year into staging. Each year's timeout is
# min(year_budget_cap, remaining run budget); when the run budget is nearly
# spent the loop stops instead of starting another year. glm_stage.py stops a
# quota-exhausted year cleanly with exit 0 and reports the leftover count in
# its "Remaining unstaged candidates: N" output — N>0 means the quota is dead
# and every later year would fast-fail, so propagate and stop the whole loop.
# Z.AI subscription allows 2 concurrent vision streams.
for year in 2017 2016 2015 2014 2013 2018; do
    now_epoch="$(date -u +%s)"
    remaining_seconds=$(( deadline_epoch - now_epoch ))
    if [ "$remaining_seconds" -le "$tail_reserve_seconds" ]; then
        printf '%s\n' "run deadline reached (remaining=${remaining_seconds}s <= ${tail_reserve_seconds}s reserve); stopping year loop"
        break
    fi
    year_budget=$(( remaining_seconds - tail_reserve_seconds ))
    if [ "$year_budget" -gt "$year_budget_cap" ]; then
        year_budget=$year_budget_cap
    fi
    printf '%s\n' "--- staging year $year (timeout=${year_budget}s, run remaining=${remaining_seconds}s) ---"
    year_log="$run_dir/stage-$year.log"
    timeout "$year_budget" python3 scripts/glm_stage.py stage --year "$year" --limit 1500 --workers 2 \
        >"$year_log" 2>&1
    stage_rc=$?
    cat "$year_log"
    if [ "$stage_rc" -ne 0 ]; then
        printf 'year %s stage exit=%d (quota/timeout tolerated)\n' "$year" "$stage_rc"
    fi
    remaining_candidates="$(grep -oE '(remaining=|Remaining unstaged candidates: )[0-9]+' "$year_log" | grep -oE '[0-9]+' | tail -n 1 || true)"
    if [ -n "$remaining_candidates" ] && [ "$remaining_candidates" -gt 0 ]; then
        printf '%s\n' "quota exhausted with $remaining_candidates unstaged candidates left; stopping year loop (later years would fast-fail on the same quota)"
        break
    fi
done

# Durability: snapshot the staging corpus so a disk failure can never return
# the backlog to nirvana (cheap: JSON records, a few MB).
if [ -d "$repo/data/staging/glm-backlog" ]; then
    backup_root="$repo/data/backups/glm-backlog"
    mkdir -p "$backup_root"
    if tar -czf "$backup_root/glm-backlog-$run_date.tar.gz" \
        -C "$repo/data/staging" glm-backlog 2>/dev/null; then
        printf '%s\n' "staging backup: $backup_root/glm-backlog-$run_date.tar.gz"
    else
        printf '%s\n' "WARNING: staging backup failed"
    fi
    find "$backup_root" -name 'glm-backlog-*.tar.gz' | sort -r | tail -n +15 | xargs -r rm -f --
fi

# Consumption/verification step: canonical health after the drain.
python3 scripts/doctor.py
doctor_rc=$?
printf 'doctor exit=%d\n' "$doctor_rc"

# Prune to the last 7 run dirs.
find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '????????' |
    sort -r | tail -n +8 | xargs -r rm -rf --
exit "$doctor_rc"
