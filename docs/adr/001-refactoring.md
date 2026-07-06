# ADR-001: Codebase Refactoring — Frozen Dataclasses, Repository Pattern, Module Split, Result Types

## Status

Accepted

## Date

2026-07-06

## Context

The ptr-alpha codebase (congressional insider trading analysis tool) had accumulated several architectural issues:

1. **Shallow freeze problem**: 14 dataclasses used `field(default_factory=list)` with `__post_init__` converting to tuple, but the freeze was only shallow — nested mutable objects could still be mutated.

2. **God modules**: `datasources.py` (650 lines) and `database.py` (602 lines) contained too many responsibilities.

3. **No result types**: Pipeline functions mixed computation with presentation (print statements, table formatting). No standardized way to propagate success/failure.

4. **Mutable constants**: `signals/constants.py` had incorrect `Final` annotations on values that were actually mutated by `validation.py` sweep.

## Decision

### 1. Frozen Dataclasses (Phase 1)
- Converted all `list` fields to `tuple` in frozen dataclasses
- Removed `__post_init__` conversion hacks
- Verified immutability with 18 new tests

### 2. Repository Pattern (Phase 2a)
Split `database.py` into four focused repositories:
- `transaction_repository.py` — Transaction CRUD
- `price_repository.py` — Price data CRUD
- `metadata_repository.py` — Metadata CRUD
- `parse_run_repository.py` — Parse run tracking

`database.py` became a thin facade (~276 lines) delegating to repositories. Fixed `upsert_parse_run` idempotency and stale-price filter nullification bug.

### 3. Datasources Split (Phase 2b)
Split `datasources.py` into:
- `parser_cascade.py` — PDF parser cascade logic
- `download.py` — House PDF download and caching
- `price_source.py` — Price data sourcing (yfinance + cache)

`datasources.py` kept as backward-compatible re-export wrapper (25 lines).

### 4. Pipeline Refactor (Phase 3)
- Separated computation from presentation
- Pipeline functions return `DataResult` with data dicts
- CLI handles all formatting/printing
- `DisplayMode` and `_save_results` moved to `cli.py`
- `_load_sector_data` moved to `sector_data.py` as public `load_sector_data`
- `_analyze_by_sector` moved to `analysis.py` as public `analyze_by_sector`

### 5. Result Types (Phase 1)
Added to `exceptions.py`:
- `StepResult` — success/error/duration for operation tracking
- `DataResult[T]` — generic result with data payload

### 6. Constants Fix (Adversarial Review)
- Removed incorrect `Final` from `BAYES_PRIOR_STRENGTH` and `DECAY_LAMBDA`
- Changed `core.py` from snapshot import to module attribute access for runtime propagation

### 7. Logging & API Cleanup (Adversarial Review)
- Converted 14 f-string logger calls to lazy `%s` formatting in `pipeline.py`
- Converted 22 `print()` calls to `logger.info()` in `validation.py`
- Converted 17 `print()` calls to `logger` in `matched_control.py`
- Renamed `_prepare_analysis_data` to `prepare_analysis_data` in `pipeline.py` (public API)

## Consequences

### Positive
- All 726 tests pass
- Clear separation of concerns (repository pattern, pipeline computation vs CLI presentation)
- Standardized result propagation via StepResult/DataResult
- Proper lazy logging throughout
- Backward-compatible re-exports for existing consumers

### Negative
- More files to navigate (but each is focused and small)
- Some indirection in database.py facade

### Risks Mitigated
- Shallow freeze to deep freeze via tuple conversion
- God modules to focused repositories and modules
- Mixed presentation to clean computation/presentation split
- Incorrect Final annotations to correct runtime behavior

## Alternatives Considered

1. **Keep monolithic modules**: Rejected — continued maintenance burden
2. **Use attrs instead of dataclasses**: Rejected — stdlib preferred, existing tests use dataclasses
3. **Return exceptions instead of Result types**: Rejected — Result types are more explicit and composable
