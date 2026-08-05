from __future__ import annotations

import pandas as pd

from analyzer._memo import clear_all_caches, df_memoize


def test_df_memoize_is_bounded_repeats_calls_and_clears() -> None:
    calls = 0

    @df_memoize
    def row_count(frame: pd.DataFrame) -> int:
        nonlocal calls
        calls += 1
        return len(frame)

    frame = pd.DataFrame({"value": [1, 2, 3]})

    assert row_count(frame) == 3
    assert row_count(frame) == 3
    assert calls == 1
    assert row_count.cache_info().maxsize == 2048
    assert row_count.cache_info().currsize == 1

    clear_all_caches()

    assert row_count.cache_info().currsize == 0
    assert row_count(frame) == 3
    assert calls == 2
