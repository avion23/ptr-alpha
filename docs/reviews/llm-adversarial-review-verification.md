# LLM adversarial review verification

Two successful `llm` reviews were independently verified. Readable prompt,
response, and usage copies are stored in the adjacent review files with trailing
whitespace normalized. Their original byte-exact forms, including the separate
AST-normalized all-source response, are preserved in
`llm-review-byte-exact-artifacts.tar.gz`.

## Verified defects fixed

- `scripts/cleanup_tickers.py` nulled symbols including `UNIT`, `TECH`, `EAST`, `WEST`, and `LAKE` solely by spelling even though they can be valid tickers. The destructive set and update loop were removed; a regression protects ambiguous valid symbols.
- `HouseTransactionSource._save_parse_results` wrote `success` parse-run records before transaction replacement. A failed replacement could leave a false-success audit record. Parse-run writes now occur only after replacement succeeds. Regressions cover replacement failure and an all-zero batch that aborts before persistence.

## Rejected or already-fixed findings

- Parser-cascade data loss was not established. The proposed partial-table signal does not exist reliably in the current parser API.
- DuckDB contention/races were asserted without a matching concurrent write path; workers read while the queue-owned writer serializes mutations.
- Gemini transaction replacement already records deletion, insertion, and success atomically. Parse-run-only error/no-result paths do not mutate transactions.
- `$100K` cannot match `_extract_ticker`'s dollar-ticker regex because the first character after `$` must be a letter.
- A fixed row-count threshold for reparsing was rejected because valid parser improvements can merge duplicates and missing rows cannot be inferred safely from count alone. Replacement is transactional.
- Kelly NaN propagation and SPY double weighting were already fixed and covered by tests.
- Log-return conversion, event-driven Sharpe annualization, delisting policy, episode construction, and UTF-16 metadata require product or data evidence before changing semantics.
- OCR orientation and the residual zero-row corpus are already documented; neither review supplied a new reproducible recovery method.

## Integration verification

`uv run --extra dev pytest -q` passes 806 tests. Ruff passes for all changed Python files, and `git diff --check` passes.
