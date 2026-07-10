# LLM adversarial review verification

Model: `gemini/gemini-flash-lite-latest` via `llm`.

- Finding 1, parser cascade data loss: not established. The cascade intentionally accepts complete text-parser results and retains the highest-quality partial result; the proposed notion of a partial table has no reliable signal in the current parser API.
- Finding 2, OCR parallel race: not established. Worker DB access is read-only and the writer owns mutations. The queue already serializes writes.
- Finding 3, non-atomic persistence: false. `insert_transactions` wraps deletion, insertion, and the success parse-run record in one DuckDB transaction. Parse-run-only error/no-result paths do not mutate transactions.
- Finding 4, ticker cleanup corruption: verified. `UNIT`, `TECH`, `EAST`, `WEST`, and `LAKE` can be real symbols, yet the cleanup script nulled them solely by spelling. The destructive set and update loop were removed and the regression test now protects ambiguous valid symbols.
- Finding 5, Kelly NaN propagation: already fixed and covered by `test_estimate_win_loss_nan_does_not_propagate`.
- Finding 6, SPY double weighting: already fixed in `_compute_derived_arrays` and covered by existing signal tests.

Verification: `uv run --extra dev pytest -q` passes 805 tests. `git diff --check` passes. Ruff still reports three pre-existing unused imports in `tests/test_bug_fixes.py`; `scripts/cleanup_tickers.py` is clean.
