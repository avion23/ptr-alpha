#!/usr/bin/env bash
# Bounded unified OCR coordinator run. Local work and the two-stream GLM lane
# share canonical's owner; GLM quota exhaustion does not stop the local lane.
# Exit 0 means this invocation and its checks succeeded, not archive completion.
set -u

readonly repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly log_root="$repo/data/logs/ocr-backlog-cron"
readonly git_sha="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
readonly run_date="$(date -u +%Y%m%d)"
readonly run_dir="$log_root/$run_date"
mkdir -p "$run_dir"
exec >>"$run_dir/run.log" 2>&1
printf 'ocr backlog start: %s UTC (repo %s @ %s)\n' \
    "$(date -u --iso-8601=seconds)" "$repo" "$git_sha"

# This wrapper lock limits duplicate scheduled invocations; the database owner
# lease, not flock or process-name inspection, enforces canonical exclusivity.
if ! command -v flock >/dev/null 2>&1; then
    printf '%s\n' "ERROR: flock unavailable; refusing an unbounded duplicate drain"
    exit 1
fi
exec 9>"$log_root/drain.lock"
if ! flock -n 9; then
    printf '%s\n' "another drain is running (lock: $log_root/drain.lock); exiting"
    exit 0
fi

readonly run_budget_seconds=21600
readonly tail_reserve_seconds=300
readonly deadline_epoch=$(( $(date -u +%s) + run_budget_seconds ))

# Existing provider configuration is read, never changed or printed.
if [ -f "${HOME}/.config/ptr-backlog/env" ]; then
    . "${HOME}/.config/ptr-backlog/env"
fi
export DUAL_READ VISION_B_BASE_URL VISION_B_MODEL VISION_B_API_KEY VISION_B_TIMEOUT
cd "$repo"

# One coordinator inventories authoritative database work for all years. No
# separate GLM staging loop, directory-scan fallback, or quota-output regex.
printf '%s\n' '--- coordinator start (local=3 gemini=0 glm=2) ---'
timeout "$((run_budget_seconds - tail_reserve_seconds))" python3 - "$repo/data" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
from analyzer.ocr.coordinator import run_pipeline

print(json.dumps(run_pipeline(sys.argv[1], local_workers=3, gemini_workers=0,
                              glm_workers=2), default=str, sort_keys=True))
PY
coordinator_rc=$?
printf 'coordinator exit=%d (retry/blocked work may remain; see structured status above)\n' "$coordinator_rc"

# Preserve the staged GLM pixel-audit corpus, not a claimed canonical backup.
# The durable coordinator spool and database snapshots have separate lifetimes.
backup_rc=0
if [ -d "$repo/data/staging/glm-backlog" ]; then
    backup_root="$repo/data/backups/glm-backlog"
    mkdir -p "$backup_root"
    remaining_seconds=$(( deadline_epoch - $(date -u +%s) ))
    if [ "$remaining_seconds" -gt 0 ] && timeout "$remaining_seconds" \
        tar -czf "$backup_root/glm-backlog-$run_date.tar.gz" \
        -C "$repo/data/staging" glm-backlog; then
        printf '%s\n' "staging backup: $backup_root/glm-backlog-$run_date.tar.gz"
        find "$backup_root" -name 'glm-backlog-*.tar.gz' | sort -r | tail -n +15 | xargs -r rm -f --
    else
        printf '%s\n' "ERROR: staging backup failed or run deadline reached"
        backup_rc=1
    fi
fi
printf 'backup exit=%d\n' "$backup_rc"

remaining_seconds=$(( deadline_epoch - $(date -u +%s) ))
if [ "$remaining_seconds" -gt 0 ]; then
    timeout "$remaining_seconds" python3 scripts/doctor.py
    doctor_rc=$?
else
    printf '%s\n' "ERROR: run deadline reached before doctor; verification not performed"
    doctor_rc=1
fi
printf 'doctor exit=%d\n' "$doctor_rc"

find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '????????' |
    sort -r | tail -n +8 | xargs -r rm -rf --
if [ "$coordinator_rc" -ne 0 ]; then
    exit "$coordinator_rc"
fi
if [ "$backup_rc" -ne 0 ]; then
    exit "$backup_rc"
fi
exit "$doctor_rc"
