"""Safety and determinism tests for the research-only model option."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyzer.research import (
    DeterministicRegularizedTabularBaseline,
    DynamicHierarchicalBaseline,
    ResearchOnlyError,
    compute_spy_factor_residual_outcomes,
)


def _factor_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=9, freq="D")
    factor = np.array([0.0, 0.01, -0.005, 0.02, 0.01, 0.20, 0.03, -0.01, 0.02])
    spy = 100.0 * np.cumprod(1.0 + factor)
    # The asset has a beta of two before entry.  On the entry date the factor
    # jumps, deliberately making an inclusive fit observably wrong.
    asset_factor = factor.copy()
    asset_factor[5] = 0.60
    asset = 100.0 * np.cumprod(1.0 + 2.0 * asset_factor)
    prices = pd.DataFrame({"AAPL": asset, "SPY": spy}, index=dates)
    factors = pd.DataFrame({"SPY": factor}, index=dates)
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "entry_date": [dates[5]],
            "exit_date": [dates[7]],
            "ticker": ["AAPL"],
        }
    )
    return events, prices, factors


class PointInTimeOutcomeTests(unittest.TestCase):
    def test_factor_fit_is_strictly_pre_entry(self):
        events, prices, factors = _factor_fixture()
        result = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=factors,
            factor_columns=["SPY"],
            min_factor_observations=3,
        ).iloc[0]

        self.assertEqual(result["label_status"], "ok")
        self.assertTrue(result["factor_fit_is_pre_entry"])
        self.assertLess(result["factor_fit_end_date"], result["entry_date"])
        # The helper fits log returns while this fixture supplies simple
        # returns, so the finite-sample beta is close rather than exactly 2.
        self.assertAlmostEqual(result["factor_betas"][0], 2.0, delta=0.03)

    def test_future_price_and_factor_rows_do_not_change_label(self):
        events, prices, factors = _factor_fixture()
        base = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=factors,
            min_factor_observations=3,
        ).iloc[0]

        later_dates = pd.date_range("2024-01-10", periods=3, freq="D")
        extended_prices = pd.concat(
            [
                prices,
                pd.DataFrame(
                    {
                        "AAPL": [500.0, 501.0, 502.0],
                        "SPY": [400.0, 401.0, 402.0],
                    },
                    index=later_dates,
                ),
            ]
        )
        extended_factors = pd.concat(
            [factors, pd.DataFrame({"SPY": [0.5, -0.4, 0.3]}, index=later_dates)]
        )
        extended = compute_spy_factor_residual_outcomes(
            events,
            extended_prices,
            factor_returns=extended_factors,
            min_factor_observations=3,
        ).iloc[0]
        for column in (
            "raw_return",
            "spy_return",
            "factor_expected_return",
            "factor_residual_return",
            "factor_intercept",
        ):
            self.assertEqual(float(base[column]), float(extended[column]))

    def test_simple_and_log_factor_inputs_have_the_same_log_fit(self):
        events, prices, factors = _factor_fixture()
        simple = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=factors,
            factor_columns=["SPY"],
            min_factor_observations=3,
        ).iloc[0]
        log_factors = np.log1p(factors)
        logged = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=log_factors,
            factor_columns=["SPY"],
            factor_returns_are_log=True,
            min_factor_observations=3,
        ).iloc[0]

        self.assertEqual(simple["label_status"], "ok")
        self.assertEqual(simple["factor_input_return_unit"], "simple")
        self.assertEqual(logged["factor_input_return_unit"], "log")
        self.assertEqual(simple["factor_return_unit"], "log")
        self.assertAlmostEqual(
            float(simple["factor_expected_return"]),
            float(logged["factor_expected_return"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(simple["factor_residual_return"]),
            float(logged["factor_residual_return"]),
            places=12,
        )

    def test_factor_unit_declaration_is_explicit_and_conflicts_are_rejected(self):
        events, prices, factors = _factor_fixture()
        result = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=factors,
            factor_columns=["SPY"],
            factor_return_unit="simple",
            min_factor_observations=3,
        ).iloc[0]
        self.assertEqual(result["factor_input_return_unit"], "simple")
        self.assertEqual(result["factor_return_unit"], "log")
        self.assertEqual(result["asset_return_unit"], "log")
        self.assertEqual(result["outcome_return_unit"], "simple")
        with self.assertRaises(ValueError):
            compute_spy_factor_residual_outcomes(
                events,
                prices,
                factor_returns=factors,
                factor_return_unit="simple",
                factor_returns_are_log=True,
                min_factor_observations=3,
            )

    def test_incomplete_future_factor_path_is_not_labeled(self):
        events, prices, factors = _factor_fixture()
        incomplete = factors.drop(index=factors.index[7])
        result = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=incomplete,
            factor_columns=["SPY"],
            min_factor_observations=3,
        ).iloc[0]
        self.assertEqual(result["label_status"], "future_factor_data_unavailable")
        self.assertTrue(pd.isna(result["factor_residual_return"]))

    def test_interior_factor_calendar_gap_is_not_labeled(self):
        events, prices, factors = _factor_fixture()
        incomplete = factors.drop(index=factors.index[6])
        result = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=incomplete,
            factor_columns=["SPY"],
            min_factor_observations=3,
        ).iloc[0]
        self.assertEqual(result["label_status"], "future_factor_data_unavailable")
        self.assertFalse(bool(result["factor_calendar_complete"]))
        self.assertEqual(int(result["factor_calendar_expected_rows"]), 2)
        self.assertEqual(int(result["factor_calendar_observed_rows"]), 1)

    def test_explicit_factor_calendar_catches_gaps_in_sparse_price_panels(self):
        events, prices, factors = _factor_fixture()
        dates = prices.index
        sparse_prices = pd.concat([prices.loc[: dates[5]], prices.loc[[dates[7]]]])
        incomplete = factors.drop(index=factors.index[6])
        result = compute_spy_factor_residual_outcomes(
            events,
            sparse_prices,
            factor_returns=incomplete,
            factor_columns=["SPY"],
            factor_calendar=dates,
            min_factor_observations=3,
        ).iloc[0]
        self.assertEqual(result["label_status"], "future_factor_data_unavailable")
        self.assertFalse(bool(result["factor_calendar_complete"]))
        self.assertEqual(int(result["factor_calendar_expected_rows"]), 2)
        self.assertEqual(int(result["factor_calendar_observed_rows"]), 1)

    def test_outcome_factor_schema_rejects_realized_factor_names(self):
        events, prices, factors = _factor_fixture()
        with self.assertRaises(ValueError):
            compute_spy_factor_residual_outcomes(
                events,
                prices,
                factor_returns=factors.assign(future_return=factors["SPY"]),
                min_factor_observations=3,
            )

    def test_sparse_price_endpoints_still_sum_daily_factor_path(self):
        events, prices, factors = _factor_fixture()
        dates = prices.index
        sparse_prices = pd.concat([prices.loc[: dates[5]], prices.loc[[dates[7]]]])
        sparse = compute_spy_factor_residual_outcomes(
            events,
            sparse_prices,
            factor_returns=factors,
            factor_columns=["SPY"],
            min_factor_observations=3,
        ).iloc[0]
        dense = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=factors,
            factor_columns=["SPY"],
            min_factor_observations=3,
        ).iloc[0]
        self.assertEqual(sparse["label_status"], "ok")
        self.assertAlmostEqual(
            float(sparse["factor_expected_return"]),
            float(dense["factor_expected_return"]),
            places=12,
        )

    def test_label_cutoff_is_strictly_before_cutoff(self):
        events, prices, factors = _factor_fixture()
        result = compute_spy_factor_residual_outcomes(
            events,
            prices,
            factor_returns=factors,
            min_factor_observations=3,
            label_as_of=prices.index[7],
        ).iloc[0]
        self.assertEqual(result["label_status"], "future_label_rejected")


def _training_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["a1", "a2", "b1", "b2"],
            "entry_date": pd.to_datetime(["2024-01-01"] * 4),
            "label_available_date": pd.to_datetime(["2024-01-02"] * 4),
            "member": ["A", "A", "B", "B"],
            "sector": ["Tech"] * 4,
            "momentum": [1.0, 1.0, -1.0, -1.0],
            "factor_residual_return": [10.0, 10.0, 0.0, 0.0],
        }
    )


class BaselineSafetyTests(unittest.TestCase):
    def test_hierarchical_baseline_partially_pools_and_handles_unseen_groups(self):
        training = _training_fixture()
        model = DynamicHierarchicalBaseline(
            horizon_days=1,
            min_time_observations=10,
            feature_columns=(),
        ).fit(training, cutoff="2024-01-10")
        predictions = model.predict(
            pd.DataFrame(
                {
                    "event_id": ["known", "unseen"],
                    "entry_date": ["2024-01-11", "2024-01-11"],
                    "member": ["A", "NEW_MEMBER"],
                    "sector": ["Tech", "NEW_SECTOR"],
                }
            )
        )

        known = float(predictions.loc[0, "prediction_mean"])
        unseen = float(predictions.loc[1, "prediction_mean"])
        self.assertGreater(known, unseen)
        self.assertLess(known, 10.0)  # A's two observations are shrunk toward 5.
        self.assertTrue(np.isfinite(predictions["prediction_std"]).all())
        self.assertIn("pooled", predictions.loc[1, "pooling_status"])

    def test_regularized_tabular_baseline_is_deterministic(self):
        training = _training_fixture()
        query = pd.DataFrame(
            {
                "event_id": ["q1", "q2"],
                "entry_date": ["2024-01-11", "2024-01-12"],
                "member": ["A", "NEVER_SEEN"],
                "sector": ["Tech", "NewSector"],
                "momentum": [0.5, -0.5],
            }
        )
        first = DeterministicRegularizedTabularBaseline(
            horizon_days=1, alpha=2.0, max_bins=4
        ).fit(training, cutoff="2024-01-10").predict(query)
        second = DeterministicRegularizedTabularBaseline(
            horizon_days=1, alpha=2.0, max_bins=4
        ).fit(training, cutoff="2024-01-10").predict(query)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first["research_only"] == True).all())  # noqa: E712
        self.assertTrue((first["deployment_authorized"] == False).all())  # noqa: E712

    def test_future_rows_leakage_columns_and_duplicate_events_are_rejected(self):
        training = _training_fixture()
        future = training.iloc[[0]].copy()
        future["event_id"] = "future"
        future["entry_date"] = "2025-01-01"
        future["label_available_date"] = "2025-01-02"
        future["factor_residual_return"] = 999.0
        with_future = pd.concat([training, future], ignore_index=True)
        model = DynamicHierarchicalBaseline(
            horizon_days=1, min_time_observations=10, feature_columns=()
        ).fit(with_future, cutoff="2024-01-10")
        self.assertEqual(model.provenance.rejected_future_event_rows, 1)
        self.assertEqual(model.provenance.fit_rows, len(training))

        leaking = training.assign(future_return=123.0)
        with self.assertRaises(ValueError):
            model.fit(leaking, cutoff="2024-01-10")

        duplicate = pd.concat([training, training.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            model.fit(duplicate, cutoff="2024-01-10")

    def test_research_model_cannot_authorize_deployment(self):
        model = DynamicHierarchicalBaseline()
        self.assertFalse(model.deployment_authorized)
        self.assertTrue(model.research_only)
        self.assertFalse(hasattr(model, "can_deploy"))
        with self.assertRaises(ResearchOnlyError):
            model.authorize_deployment()

    def test_group_columns_cannot_be_reused_as_dynamic_features(self):
        training = _training_fixture()
        automatic = DynamicHierarchicalBaseline(
            group_columns=("member",), feature_columns=None
        ).fit(training, cutoff="2024-01-10")
        self.assertNotIn("member", automatic.provenance.feature_columns)
        with self.assertRaises(ValueError):
            DynamicHierarchicalBaseline(
                group_columns=("member",), feature_columns=("member",)
            ).fit(training, cutoff="2024-01-10")

    def test_prediction_uncertainty_is_predictive_and_provenance_is_fold_local(self):
        model = DynamicHierarchicalBaseline(
            horizon_days=1, min_time_observations=10, feature_columns=()
        ).fit(_training_fixture(), cutoff="2024-01-10")
        prediction = model.predict(
            pd.DataFrame(
                {
                    "event_id": ["q1"],
                    "entry_date": ["2024-01-11"],
                    "member": ["A"],
                    "sector": ["Tech"],
                }
            )
        ).iloc[0]

        self.assertEqual(
            prediction["uncertainty_semantics"],
            "predictive_standard_deviation_for_1_day_horizon",
        )
        self.assertLess(float(prediction["lower_95"]), float(prediction["upper_95"]))
        self.assertEqual(model.provenance.as_dict()["fit_cutoff"], "2024-01-10T00:00:00")
        self.assertEqual(model.provenance.as_dict()["time_column"], "entry_date")
        self.assertEqual(model.provenance.as_dict()["strict_future"], False)

    def test_provenance_records_normalized_training_and_fitted_parameters(self):
        training = _training_fixture()
        first = DynamicHierarchicalBaseline(
            horizon_days=1, min_time_observations=10, feature_columns=()
        ).fit(training, cutoff="2024-01-10")
        second = DynamicHierarchicalBaseline(
            horizon_days=1, min_time_observations=10, feature_columns=()
        ).fit(training.sample(frac=1.0, random_state=7), cutoff="2024-01-10")

        provenance = first.provenance.as_dict()
        self.assertNotEqual(provenance["training_data_hash"], "unavailable")
        self.assertNotEqual(provenance["model_parameter_hash"], "unavailable")
        self.assertEqual(
            provenance["training_data_hash"],
            second.provenance.training_data_hash,
        )
        self.assertEqual(
            provenance["model_parameter_hash"],
            second.provenance.model_parameter_hash,
        )
        self.assertEqual(
            first.provenance.provenance_hash,
            provenance["provenance_id"],
        )

        prediction = first.predict(
            pd.DataFrame(
                {
                    "event_id": ["q1"],
                    "entry_date": ["2024-01-11"],
                    "member": ["A"],
                    "sector": ["Tech"],
                }
            )
        ).iloc[0]
        self.assertEqual(prediction["training_data_hash"], provenance["training_data_hash"])
        self.assertEqual(prediction["model_parameter_hash"], provenance["model_parameter_hash"])

    def test_prediction_uncertainty_is_tied_to_fitted_horizon(self):
        training = _training_fixture()
        model = DynamicHierarchicalBaseline(
            horizon_days=2, min_time_observations=10, feature_columns=()
        ).fit(training, cutoff="2024-01-10")
        self.assertEqual(model.provenance.uncertainty_horizon_days, 2)
        query = pd.DataFrame(
            {
                "event_id": ["q1"],
                "entry_date": ["2024-01-11"],
                "member": ["A"],
                "sector": ["Tech"],
                "horizon_days": [2],
            }
        )
        prediction = model.predict(query).iloc[0]
        self.assertEqual(prediction["horizon_days"], 2)
        self.assertEqual(prediction["uncertainty_horizon_days"], 2)
        self.assertEqual(
            prediction["uncertainty_target"], "factor_residual_return"
        )
        with self.assertRaises(ValueError):
            model.predict(query.assign(horizon_days=1))

    def test_cutoff_is_strict_for_events_and_label_availability(self):
        training = _training_fixture()
        boundary = training.iloc[[0]].copy()
        boundary["event_id"] = "boundary"
        boundary["label_available_date"] = "2024-01-10"
        combined = pd.concat([training, boundary], ignore_index=True)

        model = DynamicHierarchicalBaseline(
            horizon_days=1, min_time_observations=10, feature_columns=()
        ).fit(combined, cutoff="2024-01-10")
        self.assertEqual(model.provenance.rejected_future_label_rows, 1)
        self.assertEqual(model.provenance.fit_rows, len(training))
        with self.assertRaises(ValueError):
            DynamicHierarchicalBaseline(
                horizon_days=1,
                min_time_observations=10,
                feature_columns=(),
                strict_future=True,
            ).fit(combined, cutoff="2024-01-10")

    def test_group_dimensions_are_unique_after_normalization(self):
        for baseline in (
            DynamicHierarchicalBaseline,
            DeterministicRegularizedTabularBaseline,
        ):
            with self.assertRaises(ValueError):
                baseline(group_columns=("member", " MEMBER "))

    def test_prediction_realized_columns_are_rejected(self):
        training = _training_fixture()
        model = DynamicHierarchicalBaseline(
            horizon_days=1, min_time_observations=10, feature_columns=()
        ).fit(training, cutoff="2024-01-10")
        query = pd.DataFrame(
            {
                "event_id": ["q1"],
                "entry_date": ["2024-01-11"],
                "member": ["A"],
                "sector": ["Tech"],
                "exit_date": ["2024-01-12"],
            }
        )
        with self.assertRaises(ValueError):
            model.predict(query)

    def test_custom_label_availability_is_reserved_and_provenanced(self):
        training = _training_fixture().rename(
            columns={"label_available_date": "available_at"}
        )
        model = DynamicHierarchicalBaseline(
            horizon_days=1,
            min_time_observations=10,
            feature_columns=None,
            label_available_column="available_at",
        ).fit(training, cutoff="2024-01-10")
        self.assertNotIn("available_at", model.provenance.feature_columns)
        self.assertEqual(model.provenance.label_available_column, "available_at")


if __name__ == "__main__":
    unittest.main()
