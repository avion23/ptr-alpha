# Integrated LLM adversarial verification

Readable command and response copies from the required follow-up `llm`
invocation are stored in the adjacent integrated-review files with trailing
whitespace normalized. Their byte-exact originals are preserved in
`llm-review-byte-exact-artifacts.tar.gz`. Its prompt included the integrated
diff from `82a1d06`, both prior LLM responses and the verification response,
relevant full implementation/tests, and the `806 passed` result.

Two findings were verified and fixed:

- `cleanup_tickers` still treated valid `US`, `NEW`, and `SP` symbols as cash/junk. The fallback deletion loop was removed and the cash/fund classifier no longer classifies those symbols by spelling. A regression covers all three.
- Transaction replacement and parse-run publication were ordered but not atomic. `Database.replace_transactions_for_docs` now accepts parse-run records and persists them in its transaction. `ParseRunRepository.upsert` supports participating in a caller-owned transaction. A failure-injection regression proves a malformed audit record rolls back transaction replacement.

The weak cleanup test was strengthened through the classifier behavior. The `raw_row_count=0` observation was not introduced by this patch and cannot be populated from the current parser worker result, which exposes normalized transactions rather than raw rows; inventing a count would make the audit less reliable.

Verification: targeted tests passed `94 passed`; full `uv run --extra dev pytest -q` passed `807 passed in 9.66s`; Ruff passed all changed Python files; `git diff --check` passed.
