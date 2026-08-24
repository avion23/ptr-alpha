"""Focused tests for pure executable targets and calibration boundaries."""

import numpy as np
import pandas as pd
import pytest

from analyzer.calibration import fit_affine_calibration
from analyzer.execution_costs import (
    ExecutionCosts,
    executable_return_from_prices,
    executable_return_pct,
)
from analyzer.target_frame import build_target_frame


def test_execution_costs_validate_endpoint_multipliers():
    costs = ExecutionCosts(entry_bps=10, exit_bps=20)

    assert costs.entry_multiplier == pytest.approx(1.001)
    assert costs.exit_multiplier == pytest.approx(0.998)
    assert executable_return_pct(10.0, costs) == pytest.approx(
        ((1.10 * 0.998 / 1.001) - 1.0) * 100.0
    )
    # Negative percentage labels remain valid simple returns; the percent
    # helper must not apply the decimal -1.0 lower bound before scaling.
    assert executable_return_pct(-5.0, costs) == pytest.approx(
        ((0.95 * 0.998 / 1.001) - 1.0) * 100.0
    )
    assert executable_return_from_prices(100.0, 110.0, costs) == pytest.approx(
        1.10 * 0.998 / 1.001 - 1.0
    )

    with pytest.raises(ValueError):
        ExecutionCosts(entry_bps=-1)
    with pytest.raises(ValueError):
        ExecutionCosts(exit_bps=10_000)
    with pytest.raises(ValueError):
        ExecutionCosts(entry_bps=10, buy_bps=20)


def test_target_frame_uses_existing_exact_endpoint_labels():
    costs = ExecutionCosts(entry_slippage_bps=10, exit_slippage_bps=20)
    labels = pd.DataFrame(
        {
            "event_id": ["a"],
            "total_return_pct": [10.0],
            # The existing gross alpha implies a same-endpoint benchmark
            # return of 6%, not an independently looked-up endpoint.
            "total_spy_alpha_pct": [4.0],
            "label_window_end": pd.to_datetime(["2024-02-01"]),
            "window_complete": [True],
        }
    )

    result = build_target_frame(labels, costs, as_of_date="2024-02-02")
    expected_net = (1.10 * 0.998 / 1.001 - 1.0) * 100.0
    expected_alpha = expected_net - 6.0

    assert result.loc[0, "net_return_pct"] == pytest.approx(expected_net)
    assert result.loc[0, "net_alpha_pct"] == pytest.approx(expected_alpha)
    assert bool(result.loc[0, "target_available"])
    # Gross labels remain untouched and are still available for diagnostics.
    assert result.loc[0, "total_return_pct"] == 10.0
    assert result.loc[0, "total_spy_alpha_pct"] == 4.0


def test_target_frame_masks_immature_nan_and_incomplete_rows():
    labels = pd.DataFrame(
        {
            "event_id": ["mature", "partial", "future", "missing"],
            "total_return_pct": [10.0, 25.0, 40.0, np.nan],
            "total_spy_alpha_pct": [4.0, 5.0, 6.0, np.nan],
            "label_window_end": pd.to_datetime(
                ["2024-01-10", "2024-01-10", "2024-03-10", pd.NaT]
            ),
            "window_complete": [True, False, True, True],
        }
    )

    result = build_target_frame(labels, as_of_date="2024-02-01")
    assert result["target_available"].tolist() == [True, False, False, False]
    assert result["label_mature"].tolist() == [True, False, False, False]
    assert pd.notna(result.loc[0, "net_return_pct"])
    assert result.loc[1:, "net_return_pct"].isna().all()
    assert result.loc[1:, "net_alpha_pct"].isna().all()
    assert len(result) == len(labels)

    mature_only = build_target_frame(
        labels, as_of_date="2024-02-01", drop_incomplete=True
    )
    assert mature_only["event_id"].tolist() == ["mature"]


def test_affine_calibration_reports_uncertainty_and_probability_positive():
    frame = pd.DataFrame(
        {
            "prediction": [-1.0, 0.0, 1.0, 2.0],
            "net_return_pct": [-1.0, 1.0, 3.0, 5.0],
            "as_of": pd.to_datetime(["2024-01-01"] * 4),
            "label_window_end": pd.to_datetime(["2024-01-10"] * 4),
            "target_available": [True] * 4,
        }
    )

    calibration = fit_affine_calibration(frame, fit_cutoff="2024-02-01")
    forecast = calibration.predict_one(0.5)

    assert calibration.intercept == pytest.approx(1.0)
    assert calibration.slope == pytest.approx(2.0)
    assert calibration.n_observations == 4
    assert forecast.expected_value == pytest.approx(2.0)
    assert forecast.predictive_std >= 0.0
    assert 0.0 <= forecast.probability_positive <= 1.0
    assert forecast.probability_positive > 0.5

    predicted = calibration.predict_frame(pd.DataFrame({"prediction": [-1.0, 1.0]}))
    assert {"calibrated_mean", "predictive_std", "probability_positive"}.issubset(
        predicted.columns
    )
    assert predicted.loc[0, "probability_positive"] < 0.5
    assert predicted.loc[1, "probability_positive"] > 0.5


def test_future_rows_cannot_change_target_or_calibration_before_cutoff():
    base = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "prediction": [0.0, 1.0],
            "total_return_pct": [2.0, 4.0],
            "total_spy_alpha_pct": [1.0, 2.0],
            "as_of": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "label_window_end": pd.to_datetime(["2024-01-10", "2024-01-11"]),
            "window_complete": [True, True],
        }
    )
    future = pd.DataFrame(
        {
            "event_id": ["future"],
            "prediction": [10_000.0],
            "total_return_pct": [10_000.0],
            "total_spy_alpha_pct": [10_000.0],
            "as_of": pd.to_datetime(["2025-01-01"]),
            "label_window_end": pd.to_datetime(["2025-04-01"]),
            "window_complete": [True],
        }
    )
    costs = ExecutionCosts(entry_bps=10, exit_bps=20)

    first_targets = build_target_frame(base, costs, as_of_date="2024-02-01")
    extended_targets = build_target_frame(
        pd.concat([base, future], ignore_index=True), costs, as_of_date="2024-02-01"
    )
    pd.testing.assert_frame_equal(
        first_targets,
        extended_targets.iloc[: len(base)].reset_index(drop=True),
    )

    first_fit = fit_affine_calibration(
        first_targets.assign(prediction=base["prediction"]),
        fit_cutoff="2024-02-01",
    )
    extended_fit = fit_affine_calibration(
        extended_targets.assign(
            prediction=pd.concat(
                [base["prediction"], future["prediction"]], ignore_index=True
            )
        ),
        fit_cutoff="2024-02-01",
    )
    assert extended_fit.intercept == pytest.approx(first_fit.intercept)
    assert extended_fit.slope == pytest.approx(first_fit.slope)
    assert extended_fit.n_observations == first_fit.n_observations
    assert extended_fit.predict_one(0.25) == first_fit.predict_one(0.25)
