#!/usr/bin/env bash
# Weekly price/dividend refresh -> merged into canonical the same night.
# Tracked source of truth; the live crontab must match cron/crontab.txt
# (scripts/doctor.py enforces this daily). Outputs stay under data/, which
# stays gitignored. Contract: cron/README.md and
# docs/reviews/cron-contract-20261018.md.
set -Eeuo pipefail

readonly repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly log_root="$repo/data/logs/prices-cron"
mkdir -p "$log_root"
readonly run_dir="$(mktemp -d "$log_root/$(date -u +%Y%m%dT%H%M%S)-XXXXXX")"
readonly run_id="${run_dir##*/}"
readonly temp_db="$run_dir/prices.duckdb"
readonly report="$run_dir/report.json"
readonly snapshot_dir="$run_dir/snapshot"
readonly git_sha="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"

exec >>"$run_dir/run.log" 2>&1

printf 'prices cron start: %s UTC (repo %s @ %s)\n' \
    "$(date -u --iso-8601=seconds)" "$repo" "$git_sha"
cd "$repo"
printf 'attempt retention: automatic pruning disabled; operator-explicit GC only\n'

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
    exit "$refresh_rc"
fi

if python3 scripts/snapshot_prices.py \
    --db "$temp_db" \
    --generation "prices-cron-$run_id" \
    --out "$snapshot_dir"; then
    printf 'snapshot completed\n'
else
    snapshot_rc=$?
    printf 'snapshot failed: exit=%d\n' "$snapshot_rc"
    exit "$snapshot_rc"
fi

printf 'prices cron complete: %s UTC\n' "$(date -u --iso-8601=seconds)"
