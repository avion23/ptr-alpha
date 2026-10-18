# Cron contract: root-cause review and systemic fix

2026-10-18. Follow-up to `docs/reviews/merge-gap-20261018.md` (the symptom
fix). First-principles review of why the gap existed and persisted, and the
minimum mechanisms that make recurrence impossible by construction. No commit:
review only.

## New facts found beyond the disposition doc (all verified 2026-10-18 ~01:15 UTC)

1. **The landed merge has never run.** `merge_temp_into_canonical` attached
   the temp DB with a bound parameter (`ATTACH ? AS merge_src`), which is a
   `ParserException` on duckdb 1.5.5. The disposition's claim that the
   20261012 temp was "superseded by a fresh refresh+merge on 2026-10-18" is
   false: canonical `dividends` is still 0 rows, canonical prices max is
   still 2026-10-15 (a successful 10-18 merge would contain the 10-16
   session), and no merge line exists in any run log. The symptom fix was
   itself an unverified writer — the incident class, one level up.
2. **The doctor cron had not fired yet** when reviewed (no
   `data/logs/doctor-cron/` run dirs; first fire 05:00 UTC). At review time
   nothing consumed doctor's nonzero exit either: this box has no MTA (no
   mail/sendmail), so cron mail does not exist and exit codes vanish into
   syslog.
3. The live 20261012 weekly temp DB carries 0 dividend rows (pre-R9 code),
   so even a successful merge of it could not feed the ex-div gate.

## Root causes

| # | Root cause | Status |
|---|---|---|
| R1 | Operational infra untracked: `data/` is gitignored, so cron wrappers were unreviewed, untestable, invisible to repo audits | ADDRESSED-BY-YOUR-FIX (tracked `cron/`, tripwire test) |
| R2 | Success measured at the wrong boundary: every writer exited 0 on its own artifact, never on canonical consumption | ADDRESSED-BY-YOUR-FIX (merge read-back verification; writer-scoped `--merge-into` requirement enforced by test) |
| R3 | Monitoring measured the wrong surface: `health` saw only chamber coverage; end products (prices/dividends/form4/receipts) were measured nowhere | ADDRESSED-BY-SYMPTOM-FIX for cron (doctor), ADDRESSED-BY-YOUR-FIX for the operator surface (health end-to-end table, one shared implementation) |
| R4 | Consumer before producer: R9 gate shipped reading canonical dividends while no code path populated canonical | ADDRESSED-BY-SYMPTOM-FIX in intent (merge wiring), but inert until the ATTACH bug was fixed; first real producer run is Mon 2026-10-19 02:00 UTC |
| R5 | Redundant parallel path (staging OCR) born alongside the canonical path | ADDRESSED-BY-SYMPTOM-FIX (retired; staging kept as artifact) |
| R6 | Fixes themselves unverified: the merge shipped untested and crashed on first use; no test would notice | ADDRESSED-BY-YOUR-FIX (merge/verify unit tests, doctor integration tests) |
| R7 | Nothing consumes a monitoring failure (no MTA; cron exit codes unread) | ADDRESSED-BY-YOUR-FIX (alert log + sticky flag + dead-man heartbeat; residual below) |

## Detection-latency budget (worst case, failure → detection)

| Failure mode | Before today | After landed fix | After this fix |
|---|---|---|---|
| Weekly merge silently missing | never | ≤ ~10d 3h (artifact age >10d) + prices lag ≤8 sessions (~11d) | ≤ 24h (dividends/populated), plus read-back fails the writer the same night |
| Prices stop refreshing | never | ≤ ~11d | ≤ 24h (doctor) + writer exits nonzero same night |
| Dividends starved (R9 gate off) | never | ≤ 24h | ≤ 24h |
| Doctor cron itself dead | never | never (still open) | wrapper death ≤ 36h (heartbeat); doctor-line removal needs an external watcher — residual |
| Live crontab drifts from intent | never | never | ≤ 24h (`crontab.contract`) |

The bound still exceeds one day only for removal of the monitoring cron
itself; closing that requires an off-box watcher on
`data/alerts/doctor-last-run` freshness (deliberately out of scope).

## The fix (this working tree, uncommitted)

- `cron/` — tracked wrappers (`weekly_prices.sh`, `daily_doctor.sh`),
  `crontab.txt` source of truth, idempotent `install.sh`, README with the
  contract; wrappers derive the repo root from `BASH_SOURCE`, stamp the git
  sha, write outputs under gitignored `data/`.
- `scripts/refresh_prices.py` — `verify_consumption()` re-reads canonical
  after every merge and fails the writer on any missing key; also fixes the
  `ATTACH ?` crash (literal, quote-escaped attach path).
- `scripts/doctor.py` — freshness checks now delegate to
  `src/analyzer/freshness.py` (shared with `health`); new `doctor.alive`
  dead-man heartbeat check and `crontab.contract` drift check.
- `src/analyzer/cli.py` — `health` prints the end-to-end freshness table
  (report-only by default, `--strict-freshness` folds failures into exit 1).
- Tests: `test_cron_contract.py` (tripwire: tracked, portable, executable,
  consumption step, escalation wired), `test_freshness.py`,
  `test_refresh_verify_consumed.py`, `test_doctor.py`, health additions in
  `test_cli.py`; one exact as-of-inventory exemption for the monitoring
  aggregate.
- Live crontab migrated to the tracked wrappers (backup:
  `/tmp/opencode/crontab.before-cron-contract-20261018`; all non-repo lines
  byte-identical; verified idempotent by `cron/install.sh` dry run).

## Residual

- Canonical dividends remain 0 until the first producer run with working
  merge (Mon 2026-10-19 02:00 UTC) or a manual
  `refresh_prices --merge-into`; doctor/health fail loudly until then.
- Legacy weekly temps (20260928/20260929/20261005) and September staging
  year DBs stay on disk, flagged by `artifacts.consumed` until pruned.
- Off-box watcher for a dead doctor cron: open.
