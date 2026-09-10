# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

Impeccable classification only: PTR Alpha currently has no graphical web or native interface. Its real user-facing surfaces are terminal output plus CSV/JSON/report artifacts.

## Users

PTR Alpha is for engineers and quantitative researchers evaluating public congressional financial disclosures. The primary job is to separate information that was publicly actionable at a historical decision time from retrospective research outcomes, without backdating knowledge to the private transaction date.

## Product Purpose

PTR Alpha ingests official House and Senate disclosures, normalizes them into one canonical DuckDB model, produces current consensus buy candidates, and retrospectively evaluates the same production decision rule under executable market-session semantics.

The production rule is intentionally small. Within a 28-calendar-day public-disclosure window, count distinct canonical congressional buyers of the same eligible public equity. The default minimum is three buyers and `signal_score` is exactly the distinct-buyer count.

Member analysis is descriptive research only. It cannot authorize production recommendations.

## Positioning

The product's defining mechanism is its public-time boundary: disclosure time controls what can enter a decision. Delayed filings become actionable when they first become public. Private transaction time may describe the underlying trade but cannot move information into an earlier decision set.

Live analysis, replay, and validation share one production consensus rule rather than parallel scoring authorities.

## Operating Context

- Official House filing acquisition and PDF parsing.
- Official Senate eFD ingestion.
- DuckDB persistence with chamber/source/generation provenance.
- Read-only live analysis, historical replay, validation, and portfolio evaluation.
- Market-price acquisition for retrospective outcomes, never as an input to the live consensus score.
- Capitol Trades data as reconciliation evidence only, not canonical transaction authority.
- CLI commands and generated console/CSV/JSON output; no current graphical frontend.

## Capabilities and Constraints

- Public disclosure time is the information boundary.
- Default consensus window: 28 calendar days.
- Default consensus minimum: 3 distinct canonical buyers.
- Production scorer provenance: `identity_free_distinct_buyer_count_v2`.
- `signal_score = distinct recent buyer count`.
- Explicit options, funds, bonds, private assets, and quarantined ticker artifacts are excluded.
- Ticker aliases and renames resolve to one economic security and to the symbol tradable on the decision date.
- Historical execution enters on the first expected NYSE session after disclosure.
- Fixed-horizon exit is the expected NYSE session on or before entry plus the declared calendar-day horizon.
- Missing required price endpoints remain unavailable rather than using stale or padded fallback pricing.
- Scheduled no-trade dates represent cash return 0 when benchmark support exists.
- Validation evaluates only the production consensus family and keeps retrospective evidence distinct from any reserved final holdout.
- Hashes may identify artifacts but do not authorize execution or establish correctness.
- Live data freshness depends on the latest successfully ingested local House and Senate disclosures.

## Brand Commitments

The product name is PTR Alpha. User-facing copy and engineering interfaces should be compact, explicit, utilitarian, and easy to grep. Preserve a regular libcurl/Quake-style tool identity: no decorative UI, faux-premium language, generic gradients, unnecessary status pills, or verbose explanatory copy when a precise field or state is enough.

## Evidence on Hand

- Official-source House and Senate ingestion code and provenance in `src/analyzer/`.
- Canonical DuckDB schema and repository code.
- Current production scorer in `src/analyzer/member_ranking/buyer_scoring.py`.
- Replay and validation in `src/analyzer/backtest/` and `src/analyzer/validation.py`.
- Parser behavior documented in `docs/house-data-parsing.md`.
- Historical audit evidence under `docs/HOUSE_PARSER_AUDIT.md` and `docs/reviews/`; those are evidence, not current authority.
- Automated regression and integration tests under `tests/`.

## Product Principles

1. Public time before private hindsight.
2. One production decision rule and one validation authority.
3. Fail closed when required identity, chronology, provenance, or execution support is unresolved.
4. Prefer simple observable contracts and deletion over compatibility layers or speculative research architecture.
5. Keep descriptive research clearly separate from production authorization.
