#!/usr/bin/env bash
# Install the repo's managed cron block (cron/crontab.txt) into the live
# crontab. Idempotent: the previous managed block (between BEGIN/END
# markers) is replaced, everything outside it is preserved byte-for-byte.
# Without -y this is a dry run: it prints the diff and changes nothing.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/.." && pwd)"
apply=0
[[ "${1:-}" == "-y" ]] && apply=1

block="$(sed "s|@REPO@|$repo|g" "$here/crontab.txt")"
current="$(crontab -l 2>/dev/null || true)"
stripped="$(printf '%s\n' "$current" |
    awk '$0 == "# BEGIN insider-trading (managed block; rewritten by cron/install.sh)" {skip=1}
         $0 == "# END insider-trading" {skip=0; next}
         !skip')"
candidate="$(printf '%s\n%s\n' "$stripped" "$block" | sed '/./,$!d')"

printf '%s\n' '--- live crontab' "$current" '--- candidate crontab' "$candidate" |
    sed -n '1,200p'
if (( ! apply )); then
    printf 'dry run: no changes made (use -y to apply)\n'
    exit 0
fi

backup="/tmp/opencode/crontab.backup-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$(dirname "$backup")"
printf '%s\n' "$current" >"$backup"
printf '%s\n' "$candidate" | crontab -
printf 'installed. previous crontab saved to %s\n' "$backup"
