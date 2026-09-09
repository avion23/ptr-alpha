# Changelog

## Unreleased

### Breaking changes

- `score_ticker_by_buyers` now defaults to consensus scoring, requires an explicit
  `as_of_date` in that mode, and no longer accepts the unused member-skill,
  uncertainty-penalty, or solo-buyer posterior-gate parameters.
- Removed `_lookup_buyer_posterior_lift` from the member-ranking package API.
- `MemberSkillPosterior` now exposes only estimable member effects and effective
  information. `score_members_for_ticker` was replaced by
  `score_member_posteriors`, which accepts unique member identities only.
- `bayesian_quality` scoring was removed. Unknown scoring modes now fail instead
  of silently falling back to another score.
- Production consensus scoring is now the distinct recent-buyer count; the hidden
  exponential recency coefficient and backtest-only crash/lag multipliers were removed.
- Live ticker analysis no longer requires historical price/outcome construction. Single-
  ticker and multi-ticker analysis use the same disclosure window and buyer threshold.
- CLI backtests now evaluate the declared fixed horizon; adaptive OU holding periods are
  no longer injected into production replay. Portfolio simulation no longer exposes
  unused horizon/training/threshold knobs.
- `backtest_recommendations` no longer accepts an unused price frame.

### Fixed

- Aggregate all pdftotext and Docling tables instead of returning after the first
  non-empty table; parser debug logs now include bounded per-engine timing/quality.
- Use canonical official-source provenance for ordinary ticker eligibility instead of
  requiring sparsely populated `ticker_origin` metadata; date-gate known reused symbols
  so pre-listing disclosures cannot borrow another security's historical price series.
- Preserve real buyers when rows are only flagged as economic duplicate candidates;
  exact same-day signal display duplicates are collapsed separately.
- Avoid import-time database discovery in `scripts/reconcile_blotter.py`.
- Price acquisition skips quarantined parser tokens and summarizes ticker resolution;
  CLI logging suppresses per-symbol yfinance error spam in favor of project-level counts.
- Signal calendar construction now precomputes NYSE sessions per batch, eliminating the
  medium-size performance regression.
- Removed exact-machine runtime fingerprinting as a final-evaluation correctness gate.
- Read-oriented commands no longer write implicit price snapshots; `snapshot` remains the
  explicit persistence command.

### Documentation

- Reconciled README, prediction architecture, House parser flow, ingestion risk catalog,
  and historical audit/ADR labeling with the current implementation.
- Marked `docs/reviews/` as archived review evidence rather than current product guidance.
