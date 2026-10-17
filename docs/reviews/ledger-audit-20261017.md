# Ledger provenance audit: the "Bogus Member" receipts

2026-10-17. Follow-up on the replay-protocol design review's UNKNOWN #3.

**Finding: operator smoke runs, not contamination of any decision.** The
three manifests with `scope.members == ["Bogus Member"]` in
`data/recommendations/2026-10-16/` were written by hand while the receipt
feature was being developed on 2026-10-16:

| receipt | published (UTC) | git | ranked | note |
|---|---|---|---:|---|
| `f2081d2a` | 17:51:34 | `b7331992` | 0 | 2 min after the receipts merge |
| `1b19c306` | 19:04:25 | `b7331992` | 332 | pre-`38f3d89b`: bogus explicit member still saw the full event universe — the exact bug `38f3d89b` ("empty only when explicitly scoped") fixed at 20:20 |
| `633c5bbe` | 20:28:31 | `38f3d89b` | 0 | post-fix rerun of the same smoke |

**Chain integrity: PASS via the production verifier** — `verify_decision`
returns `agreement=True, mismatches=[]` for all three plus the real
`734927d4` receipt (a hand-rolled raw-line hash check reports False for
every line; the ledger hashes canonical lines, `decision_snapshot.py:115-119`
— use the shipped verifier, not ad-hoc math).

**Recurrence: impossible from the current suite.** The CLI smoke test passes
`--no-receipt` (`tests/test_cli_smoke.py:53-55`); the snapshot test writes to
`--data-dir tmp_path` (`tests/test_decision_snapshot.py:366-371`).

**Action: none.** The ledger is append-only and the entries are
self-identifying (`members: ["Bogus Member"]`); they stay as historical
record — `1b19c306` additionally documents the pre-`38f3d89b` universe-leak
bug's existence. Any future smoke test must keep using `--no-receipt` or a
temporary data dir.
