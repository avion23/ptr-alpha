# OpenCode adversarial review verification

Date: 2026-07-10

## Invocation

`opencode run -m opencode/big-pickle ... -f /tmp/opencode-adversarial-input.txt`

The attachment was 1,941,452 bytes and contained 163 tracked text files: production Python, scripts, tests, configuration, documentation, and the fresh test output. OpenCode inspected the commit range `82a1d06..783b5a1`. The readable response is preserved in `opencode-adversarial-review-response.txt`.

## Independent verification

- F1/F6, empty consolidation lacks audit rows: confirmed behavior, but not a regression in the reviewed range. `_save_parse_results` deliberately rejects a batch that persists no transactions. Persisting “success” records separately before raising would recreate the audit/data inconsistency that commit `9e25b82` fixed. A future design could atomically record batch failures as `error`; no patch was made without an explicit batch-failure model.
- F2, zero-row documents are deleted: retracted by OpenCode and independently rejected. Replacement document IDs come only from the consolidated DataFrame, so zero-row document IDs are not deleted.
- F3, log/simple-return mixing: rejected as a correctness bug. `core.py` intentionally computes midpoint-weighted log returns for both the security and SPY; subtraction therefore produces log alpha in a consistent space. Exponentiating only the security as suggested would make alpha inconsistent. The output name/documentation could be clearer, but changing units would alter the model contract and requires a separate migration.
- F4, parser blacklist drops valid one-letter tickers: known precision/recall tradeoff, not newly introduced. The parser blocks ambiguous single-letter transaction/owner artifacts in noisy PDF cells. The claimed `$100K -> K` case is false because the regex requires a letter immediately after `$`. Existing regression tests assert the ambiguity policy. No patch was made.
- F5, `US` cleanup regression: rejected. The reviewed fix intentionally stopped destructive spelling-only deletion because a token cannot be classified from spelling alone. Reintroducing it would violate the cleanup safety invariant.
- F7, unused `grp`: confirmed harmless dead local, pre-existing and unrelated. It is a lint cleanup, not an adversarial correctness finding.

## Result

No new correctness regression in `82a1d06..783b5a1` survived independent verification. No production changes were justified by this review.
