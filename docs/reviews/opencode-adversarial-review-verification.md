# OpenCode adversarial review verification

Date: 2026-07-10

## Invocation

The attachment was constructed from the checkout at `783b5a1` with:

```bash
uv run python - <<'PY'
from pathlib import Path
import subprocess

files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
extensions = {".py", ".toml", ".md", ".txt", ".yaml", ".yml", ".json"}
selected = [path for path in files if Path(path).suffix in extensions]
output = Path("/tmp/opencode-adversarial-input.txt")
with output.open("w", encoding="utf-8") as stream:
    stream.write(
        "ADVERSARIAL REVIEW INPUT: ALL TRACKED TEXT SOURCE, TESTS, "
        "CONFIGURATION, DOCUMENTATION, AND RESULTS\n"
    )
    for path in selected:
        try:
            source = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        stream.write(f"\n\n===== FILE: {path} =====\n{source}")
print(len(selected), output.stat().st_size)
PY
```

The successful review invocation was:

```bash
opencode run -m opencode/big-pickle --title 'PTR adversarial review 2025-07-10' \
  "$(cat docs/reviews/opencode-adversarial-review-prompt.md)" \
  -f /tmp/opencode-adversarial-input.txt 2>&1 \
  | tee docs/reviews/opencode-adversarial-review-response.txt
```

The attachment was 1,941,452 bytes and contained 163 tracked text files:
production Python, scripts, tests, configuration, documentation, and the fresh
test output. Its SHA-256 was
`52e497527dee5d993ca798eb8b795ef065b82a546af1abfce3a4a0ad0bffa8b8`.
OpenCode inspected the commit range `82a1d06..783b5a1`.

The stream returned after exploration, so the recorded session was prompted once
more with:

```bash
opencode run -s ses_0b5cefde2ffeXhHuHeWoYLX8iP -m opencode/big-pickle \
  "Finish the adversarial review now. Return the requested final actionable findings only. Do not make edits or run more broad exploration; independently reason from what you have read." \
  2>&1 | tee -a docs/reviews/opencode-adversarial-review-response.txt
```

The response artifact contains the complete substantive findings at lines
356-433, followed by three asynchronously flushed exploration lines ending in
an incomplete sentence. It is preserved verbatim but is not a complete transport
transcript. The resumed command emitted no separate final response.

## Independent verification

- F1/F6, empty consolidation lacks audit rows: confirmed behavior, but not a regression in the reviewed range. `_save_parse_results` deliberately rejects a batch that persists no transactions. Persisting “success” records separately before raising would recreate the audit/data inconsistency that commit `9e25b82` fixed. A future design could atomically record batch failures as `error`; no patch was made without an explicit batch-failure model.
- F2, zero-row documents are deleted: retracted by OpenCode and independently rejected. Replacement document IDs come only from the consolidated DataFrame, so zero-row document IDs are not deleted.
- F3, log/simple-return mixing: rejected as a correctness bug. `core.py` intentionally computes midpoint-weighted log returns for both the security and SPY; subtraction therefore produces log alpha in a consistent space. Exponentiating only the security as suggested would make alpha inconsistent. The output name/documentation could be clearer, but changing units would alter the model contract and requires a separate migration.
- F4, parser blacklist drops valid one-letter tickers: known precision/recall tradeoff, not newly introduced. The parser blocks ambiguous single-letter transaction/owner artifacts in noisy PDF cells. The claimed `$100K -> K` case is false because the regex requires a letter immediately after `$`. Existing regression tests assert the ambiguity policy. No patch was made.
- F5, `US` cleanup regression: rejected. The reviewed fix intentionally stopped destructive spelling-only deletion because a token cannot be classified from spelling alone. Reintroducing it would violate the cleanup safety invariant.
- F7, unused `grp`: confirmed harmless dead local, pre-existing and unrelated. It is a lint cleanup, not an adversarial correctness finding.

## Result

No new correctness regression in `82a1d06..783b5a1` survived independent verification. No production changes were justified by this review.
