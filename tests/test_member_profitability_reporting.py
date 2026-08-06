"""Smoke tests for member_profitability.reporting module."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


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
