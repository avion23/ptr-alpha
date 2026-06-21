"""Smoke tests for member_profitability.reporting module."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


class TestReportingImports(unittest.TestCase):

    def test_module_imports(self):
        import member_profitability.reporting
        self.assertTrue(callable(member_profitability.reporting.build_output_dict))
        self.assertTrue(callable(member_profitability.reporting.best_predictors))
        self.assertTrue(callable(member_profitability.reporting.write_output))
        self.assertTrue(callable(member_profitability.reporting.serialize_numpy))


class TestBuildOutputDict(unittest.TestCase):

    def test_returns_sections(self):
        from member_profitability.reporting import build_output_dict
        sigs = pd.DataFrame({
            "member": ["A"],
            "ticker": ["X"],
            "disclosure_date": pd.to_datetime(["2024-01-01"]),
        })
        all_tx = pd.DataFrame({"ticker": ["X"]})
        out = build_output_dict(
            sigs=sigs, all_tx=all_tx, all_tickers=["X"], windows=[],
            valid_windows=0, all_wf=pd.DataFrame(),
            correlations={}, tier_results={}, trade_count_analysis={},
            position_results=[], combined_results={},
        )
        for key in [
            "analysis_config", "spearman_correlations", "tier_analysis",
            "trade_count_reliability", "position_sizing_grid",
            "combined_metrics", "recommendations",
        ]:
            self.assertIn(key, out)


class TestSerializeNumpy(unittest.TestCase):

    def test_handles_numpy_types(self):
        import numpy as np
        from member_profitability.reporting import serialize_numpy
        self.assertEqual(serialize_numpy(np.int64(5)), 5)
        self.assertEqual(serialize_numpy(np.float64(1.5)), 1.5)
        self.assertEqual(serialize_numpy(np.array([1, 2, 3])), [1, 2, 3])
        self.assertEqual(serialize_numpy("plain"), "plain")


class TestWriteOutput(unittest.TestCase):

    def test_writes_valid_json(self):
        from member_profitability.reporting import write_output
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "out.json"
            payload = {"x": 1, "nested": {"y": [1, 2, 3]}}
            buf = io.StringIO()
            with redirect_stdout(buf):
                write_output(payload, path)
            self.assertTrue(path.exists())
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, payload)


class TestBestPredictors(unittest.TestCase):

    def test_picks_metric_with_highest_abs_correlation(self):
        from member_profitability.reporting import best_predictors
        correlations = {
            "weak": {"mean_spearman": 0.05, "n_windows": 5},
            "strong": {"mean_spearman": 0.5, "n_windows": 5},
        }
        out = best_predictors(
            correlations=correlations,
            combined_results={},
            tier_results={},
            position_results=[],
        )
        self.assertEqual(out["best_single_predictor"]["metric"], "strong")
        self.assertTrue(len(out["key_findings"]) > 0)


if __name__ == "__main__":
    unittest.main()
