#!/usr/bin/env bash
# Weekly price/dividend refresh -> merged into canonical the same night.
# Tracked source of truth; the live crontab must match cron/crontab.txt
# (scripts/doctor.py enforces this daily). Outputs stay under data/, which
# stays gitignored. Contract: cron/README.md and
# docs/reviews/cron-contract-20261018.md.
set -Eeuo pipefail

readonly repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly log_root="$repo/data/logs/prices-cron"
run_date="$(date -u +%Y%m%d)"
readonly run_date
readonly run_dir="$log_root/$run_date"
readonly temp_db="$run_dir/prices.duckdb"
readonly report="$run_dir/report.json"
readonly snapshot_dir="$run_dir/snapshot"
readonly git_sha="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"

mkdir -p "$run_dir"
exec >>"$run_dir/run.log" 2>&1

printf 'prices cron start: %s UTC (repo %s @ %s)\n' \
    "$(date -u --iso-8601=seconds)" "$repo" "$git_sha"
cd "$repo"

prune_runs() {
    local -a run_names=()
    local run_name kept=0

    mapfile -t run_names < <(
        find "$log_root" -mindepth 1 -maxdepth 1 -type d -name '????????' -printf '%f\n' |
            sort -r
    )
    for run_name in "${run_names[@]}"; do
        [[ "$run_name" =~ ^[0-9]{8}$ ]] || continue
        ((kept += 1))
        if (( kept > 4 )); then
            printf 'prune: %s\n' "$log_root/$run_name"
            rm -rf -- "${log_root:?}/$run_name"
        fi
    done
}

# --merge-into is the consumption step: the refresh is only successful when
# its prices and dividends are verifiably present in canonical afterwards
# (refresh_prices re-reads canonical and exits nonzero on any missing key).
if python3 scripts/refresh_prices.py \
    --source-db data/congress.duckdb \
    --db "$temp_db" \
    --merge-into "$repo/data/congress.duckdb" \
    --report "$report"; then
    printf 'refresh completed and verified in canonical\n'
else
    refresh_rc=$?
    printf 'refresh failed: exit=%d; snapshot skipped\n' "$refresh_rc"
    prune_runs
    exit "$refresh_rc"
fi

if python3 scripts/snapshot_prices.py \
    --db "$temp_db" \
    --generation "prices-cron-$run_date" \
    --out "$snapshot_dir"; then
    printf 'snapshot completed\n'
else
    snapshot_rc=$?
    printf 'snapshot failed: exit=%d\n' "$snapshot_rc"
    prune_runs
    exit "$snapshot_rc"
fi

prune_runs
printf 'prices cron complete: %s UTC\n' "$(date -u --iso-8601=seconds)"
