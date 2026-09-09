"""Compatibility facade for leakage-safe affine calibration."""

from analyzer.calibration import (
    AffineCalibration,
    AffineCalibrator,
    CalibratedPrediction,
    FrozenAffineCalibration,
    LeakageSafeAffineCalibrator,
    affine_calibration,
    calibrate_predictions,
    fit_affine_calibration,
    fit_calibration,
)

__all__ = [
    "AffineCalibration",
    "AffineCalibrator",
    "CalibratedPrediction",
    "FrozenAffineCalibration",
    "LeakageSafeAffineCalibrator",
    "affine_calibration",
    "calibrate_predictions",
    "fit_affine_calibration",
    "fit_calibration",
]
