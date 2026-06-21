"""Smoke tests for analyzer._memo module (df_memoize helper)."""
import unittest

import pandas as pd

from analyzer._memo import df_memoize


class TestDfMemoize(unittest.TestCase):

    def setUp(self):
        self.df_a = pd.DataFrame({"x": [1, 2, 3]})
        self.df_b = pd.DataFrame({"x": [1, 2, 3]})  # same content, different id

    def test_caches_by_dataframe_identity(self):
        @df_memoize
        def f(df):
            return {"sum": int(df["x"].sum())}

        # First call: computed
        r1 = f(self.df_a)
        # Second call with same object: cached (hits cache info)
        r2 = f(self.df_a)
        self.assertEqual(r1, r2)
        self.assertEqual(f.cache_info().hits, 1)
        self.assertEqual(f.cache_info().misses, 1)

    def test_distinct_dataframes_separate_cache_entries(self):
        @df_memoize
        def f(df):
            return int(df["x"].sum())

        f(self.df_a)
        f(self.df_b)
        # Different id() -> separate cache entry
        self.assertEqual(f.cache_info().misses, 2)

    def test_default_returns_copy_of_dataframe(self):
        @df_memoize
        def f(df):
            return df

        result = f(self.df_a)
        # copy=True (default) -> caller gets a different object
        self.assertIsNot(result, self.df_a)

    def test_copy_false_returns_cached_dataframe(self):
        @df_memoize(copy=False)
        def f(df):
            return df

        result1 = f(self.df_a)
        result2 = f(self.df_a)
        # copy=False -> identical id across calls
        self.assertIs(result1, result2)

    def test_scalar_args_pass_through(self):
        @df_memoize
        def f(x, y):
            return x + y

        self.assertEqual(f(1, 2), 3)
        self.assertEqual(f(1, 2), 3)
        self.assertEqual(f.cache_info().hits, 1)

    def test_cache_clear_resets(self):
        @df_memoize
        def f(df):
            return int(df["x"].sum())

        f(self.df_a)
        f.cache_clear()
        f(self.df_a)
        # After clear, stats reset to 0, so misses reflects only post-clear calls.
        self.assertEqual(f.cache_info().hits, 0)
        self.assertEqual(f.cache_info().misses, 1)

    def test_decorator_with_no_args(self):
        @df_memoize
        def f(x):
            return x * 2

        self.assertEqual(f(5), 10)


if __name__ == "__main__":
    unittest.main()