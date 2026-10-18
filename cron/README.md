# Cron contract (tracked operational infra)

The 2026-10-18 incident: two cron wrappers living under gitignored `data/`
produced artifacts no process ever consumed, for weeks, with exit code 0.
Root causes and fixes: `docs/reviews/cron-contract-20261018.md` (analysis)
and `docs/reviews/merge-gap-20261018.md` (symptom disposition).

## The contract

> Every automated writer's output is consumed by canonical within its
> cadence, and any break is loud within one day.

Made checkable by four mechanisms, none of which rely on vigilance:

1. **Wrappers are tracked here.** Every repo cron job lives in `cron/`,
   derives its repo root from `${BASH_SOURCE[0]}` (no machine-specific
   paths inside the wrapper), stamps the checked-out git sha into its
   run log, and writes outputs under gitignored `data/`.
   `tests/test_cron_contract.py` enforces all of it at PR time.
2. **Writers verify consumption before exiting 0.** The weekly wrapper
   passes `--merge-into` and `refresh_prices` re-reads canonical after the
   merge and exits nonzero if any refreshed key is still missing.
   `scripts/doctor.py` re-checks the end products daily from the outside.
3. **One deployment source of truth.** `cron/crontab.txt` is the only
   sanctioned crontab content for this repo; `cron/install.sh` applies it
   idempotently (managed BEGIN/END block, rest of the crontab untouched),
   and the doctor's `crontab.contract` check fails within a day if the
   live crontab drifts from the file.
4. **Failures route somewhere concrete.** This box has no MTA (no
   mail/sendmail), so exit codes alone would vanish into syslog. On
   doctor failure the wrapper (a) exits nonzero, (b) appends the failure
   summary to `$HOME/logs/ptr-alerts.log` beside the operator's
   other cron logs, and (c) re-creates the sticky flag
   `data/alerts/ACTIVE` (delete it to acknowledge; a later PASS clears it
   automatically). A heartbeat file (`data/alerts/doctor-last-run`,
   written by the wrapper before each run) makes a dead doctor cron fail
   the `doctor.alive` check — and if the doctor cron itself dies, the
   only remaining detector is external; see "Residual limits".

## Files

- `cron/crontab.txt` — source of truth (`@REPO@` placeholder).
- `cron/install.sh` — dry-run diff / `-y` apply with timestamped backup.
- `cron/weekly_prices.sh` — Mondays 02:00 UTC: refresh + merge + snapshot.
- `cron/daily_doctor.sh` — daily 05:00 UTC: doctor + escalation wiring.

Historical wrappers remain under `data/logs/*` (gitignored, retired); the
disabled Gemini staging cron's disposition is in
`docs/reviews/merge-gap-20261018.md`.

## Residual limits

- If the doctor cron itself is removed/broken *and* the heartbeat check
  cannot run (doctor gone), nothing in-repo detects it. An external
  uptime pinger on `data/alerts/doctor-last-run` freshness would close
  that last gap and is deliberately out of scope here.
- `data/alerts/ACTIVE` requires a human (or a future tool) to read it;
  the flag exists so any repo-side command can surface it cheaply.
