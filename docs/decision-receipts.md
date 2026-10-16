# Decision receipts: ledger-anchored publication

Every production decision boundary publishes a receipt. This note describes the
mechanism, the guarantees, and — honestly — the residual risk.

## Layout

```
<data-dir>/recommendations/
├── receipts.log                        # append-only ledger, one line per publication
└── <YYYY-MM-DD>/<decision_id>.json     # one self-contained receipt per distinct decision
```

- `<decision_id>` is the SHA-256 of the receipt's canonical payload
  **excluding** the `manifest_hash` field. `manifest_hash` equals it, so the
  content, its internal digest and the file name all bind to the same value.
  An identical rerun resolves to the same id and is a true no-op (no rewrite,
  no new ledger line); a genuinely different decision on the same date gets
  its own receipt file and its own ledger line, and is logged as recorded —
  never swallowed as a "conflict". An empty decision occupies only its own id
  and never blocks later receipts.
- `receipts.log` carries one canonical-JSON line per publication:
  `{decision_date, decision_id, manifest_hash, prev_line_hash, published_at,
  git_sha}`. Lines are appended with `O_APPEND` + `fsync`, followed by a
  directory `fsync`. `prev_line_hash` is the canonical hash of the previous
  line (`0…0` at genesis), forming a hash chain. Every line also satisfies
  `decision_id == manifest_hash`; a line where they diverge has been rewritten.

## Provenance

- Setups candidates carry canonical row identities: `eligible_events` rows
  include `source_record_id` / `source_row_id` (explicit NULL when a legacy
  producer has no ids — nothing is invented), and `holdings_candidates` rows
  include `filing_refs`: the deduplicated, canonically sorted
  `{source, source_record_id, source_row_id}` of every contributing lot.
  Receipts therefore reconstruct input provenance, not just scoring. Adding
  these columns does not change ranking (scoring never reads them); the CLI
  output is byte-identical with and without recording, and is asserted so in
  tests.
- `git_sha` is resolved from the **executing package's checkout**
  (`git -C <analyzer package dir> rev-parse HEAD`), with `GIT_SHA` from the
  environment as fallback and NULL when unresolvable. Provenance fails
  closed; it never records an unrelated repository's revision.
- `analyze --ticker` / `--mode tickers` receipts (`kind: ticker_consensus`)
  freeze the served buyer/score tables. They are anchored in the same ledger
  and verify by content hash, but they do not claim an offline replay: the
  consensus scorer is identity-free by design and requires the database.

## Verification

`verify_decision(path)` fails closed. It recomputes the receipt's canonical
hash and matches it against BOTH the internal `manifest_hash` AND the ledger
entry; verifies the full chain linkage of `receipts.log`; re-scores the frozen
setups inputs and compares ranked/blocked/eligibility and the price
reference. A receipt with no ledger entry, or a missing/corrupt ledger, is a
mismatch — never a pass. Publication shares one code path for all boundaries
(`_publish`); a receipt write failure is logged and never raised, so trading
output and exit status are unaffected.

## Residual risk (read this before claiming immutability)

The ledger does **not** provide cryptographic immutability. The operating
system owner can rewrite both the receipt and the ledger, chmod permissions
notwithstanding (`0444` receipts are advisory). A sufficiently determined
attacker can regenerate a consistent receipt and recompute the entire chain
history. What the ledger buys is detectability: any tampering must rewrite
published chain history — every subsequent line's `prev_line_hash`, and every
receipt id published after the change — leaving a coherent but forged history
that no longer matches any independently retained copy of `receipts.log`.
Anchor or export the ledger (or its tail hash) outside the writer's control
(e.g. append-only log shipping, a WORM store, or a signed checkpoint) to
convert that detectable forgery into a prevented one.
