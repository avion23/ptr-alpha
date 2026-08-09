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


class TestQualifiedResearchLanguage(unittest.TestCase):
    def test_negative_results_are_not_called_optimal_or_profitable(self):
        from member_profitability.reporting import best_predictors

        result = best_predictors(
            correlations={"negative": {"mean_spearman": -0.4, "n_windows": 5}},
            combined_results={"negative_combo": [-0.3, -0.2]},
            tier_results={"negative": {"alpha_lift": -2.0, "n_windows": 5}},
            position_results={
                "selected_candidate": {"top_n": 1, "min_buyers": 2},
                "status": "nonpositive_holdout",
            },
        )

        self.assertIsNone(result["best_single_predictor"])
        self.assertIsNone(result["leading_combined_metric"])
        self.assertIsNone(result["leading_positive_tier"])
        self.assertEqual(result["holdout_status"], "nonpositive_holdout")
        rendered = json.dumps(result).lower()
        self.assertNotIn("optimal", rendered)
        self.assertNotIn("profitable", rendered)

    def test_positive_small_holdout_is_explicitly_not_robust(self):
        from member_profitability.position_sizing import _holdout_status

        status = _holdout_status(
            {
                "n_eligible_recommendations": 2,
                "n_evaluable_recommendations": 2,
                "n_missing_outcome_recommendations": 0,
                "n_evaluable_decision_dates": 2,
                "mean_excess_return_pct": 5.0,
                "one_sided_p_value": 0.01,
            }
        )

        self.assertEqual(status, "positive_holdout_not_robust")

    def test_json_writer_rejects_nan(self):
        from member_profitability.reporting import write_output

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_output({"invalid": float("nan")}, Path(tmp) / "out.json")


if __name__ == "__main__":
    unittest.main()
