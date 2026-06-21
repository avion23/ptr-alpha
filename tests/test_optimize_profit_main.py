"""Smoke tests for optimize_profit.main module.

`main.py` is the sweep-driver entry point: imports submodules, defines
`main()`. Importing it doesn't run the sweep; we just verify the public
surface.
"""
import unittest
from pathlib import Path


class TestMainImports(unittest.TestCase):

    def test_module_imports(self):
        import optimize_profit.main
        self.assertTrue(callable(optimize_profit.main.main))

    def test_module_file_exists(self):
        path = Path(__file__).resolve().parent.parent / "optimize_profit" / "main.py"
        self.assertTrue(path.exists())

    def test_private_helpers_callable(self):
        import optimize_profit.main as m
        self.assertTrue(callable(m._load_data))
        self.assertTrue(callable(m._compute_signals_and_precompute))
        self.assertTrue(callable(m._run_sweep))
        self.assertTrue(callable(m._maybe_log_progress))


class TestMaybeLogProgress(unittest.TestCase):

    def test_does_not_crash_on_first_combo(self):
        import io
        import contextlib
        import time
        import optimize_profit.main as m
        params = {
            "scoring_fn": "shrunk_alpha", "top_n": 5,
            "min_buyers": 2, "allocation": "equal",
        }
        metrics = {
            "total_return_pct": 12.3, "sharpe": 1.5,
            "max_drawdown_pct": -10.0, "win_rate_pct": 60.0,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # First combo (i=0) always logs. Use a t0 a few seconds ago
            # so elapsed > 0 and the rate calculation doesn't divide by zero.
            m._maybe_log_progress(0, 100, params, metrics, t0=time.time() - 5)
        out = buf.getvalue()
        self.assertIn("shrunk_alpha", out)
        self.assertIn("[  1/100]", out)

    def test_mid_sweep_does_not_log(self):
        import io
        import contextlib
        import time
        import optimize_profit.main as m
        params = {
            "scoring_fn": "shrunk_alpha", "top_n": 5,
            "min_buyers": 2, "allocation": "equal",
        }
        metrics = {
            "total_return_pct": 12.3, "sharpe": 1.5,
            "max_drawdown_pct": -10.0, "win_rate_pct": 60.0,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # i=5, total=100 -> not at boundary, no log
            m._maybe_log_progress(5, 100, params, metrics, t0=time.time() - 5)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
