#!/usr/bin/env bash
# Daily fail-loud freshness doctor (see scripts/doctor.py). Tracked source of
# truth; the live crontab must match cron/crontab.txt. Escalation path when
# doctor fails (no MTA on this box — mail/sendmail are absent):
#   1. nonzero exit -> cron logs it to /var/log/syslog,
#   2. failure summary appended to the operator's central cron log
#      (default $HOME/logs/ptr-alerts.log, beside reap-cron.err etc.),
#   3. sticky flag data/alerts/ACTIVE re-created on every failing run and
#      removed only by a later PASS — delete it to acknowledge.
# A heartbeat (data/alerts/doctor-last-run) proves this cron fired; doctor
# itself fails when the heartbeat goes stale (dead-man switch).
set -u

readonly repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly log_root="$repo/data/logs/doctor-cron"
readonly alert_dir="$repo/data/alerts"
readonly alert_flag="$alert_dir/ACTIVE"
readonly heartbeat="$alert_dir/doctor-last-run"
readonly alert_log="${DOCTOR_ALERT_LOG:-$HOME/logs/ptr-alerts.log}"
readonly git_sha="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
run_date="$(date -u +%Y%m%d)"
readonly run_date
readonly run_dir="$log_root/$run_date"

mkdir -p "$run_dir" "$alert_dir"
exec >>"$run_dir/run.log" 2>&1
printf 'doctor cron start: %s UTC (repo %s @ %s)\n' \
    "$(date -u --iso-8601=seconds)" "$repo" "$git_sha"
date -u +%s > "$heartbeat"
cd "$repo"

# A transient failure must not pin yesterday's flag forever: clear first,
# re-create below if this run also fails.
rm -f -- "$alert_flag"

python3 scripts/doctor.py
rc=$?
printf 'doctor cron done: exit=%d\n' "$rc"

if (( rc != 0 )); then
    {
        printf 'FAIL %s repo=%s sha=%s run_log=%s\n' \
            "$(date -u --iso-8601=seconds)" "$repo" "$git_sha" "$run_dir/run.log"
        sed -n '1,40p' "$run_dir/run.log"
    } >>"$alert_log" 2>/dev/null ||
        printf 'doctor FAIL %s: alert log %s unwritable\n' \
            "$(date -u --iso-8601=seconds)" "$alert_log" >&2
    printf 'doctor FAILED at %s (sha %s); see %s\n' \
        "$(date -u --iso-8601=seconds)" "$git_sha" "$run_dir/run.log" >"$alert_flag"
fi

# Prune to the last 7 run dirs.
find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '????????' |
    sort -r | tail -n +8 | xargs -r rm -rf --
exit "$rc"
