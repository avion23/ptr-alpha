"""Disabled legacy in-sample parameter sweep.

Winner claims from a single reused window are not valid out-of-sample evidence.
Use ``ptr-alpha validate`` for purged, corrected nested validation. The locked
post-2025 final phase is intentionally unavailable to this script.
"""

from __future__ import annotations

import sys

from analyzer.validation import SweepResult, run_single_backtest  # noqa: F401


def main() -> None:
    print(
        "sweep.py is disabled: use `ptr-alpha validate` for purged corrected "
        "validation; no in-sample winner is reported.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
