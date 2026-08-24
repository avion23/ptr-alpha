"""Pure prediction-plane primitives.

This facade keeps the three independent boundaries discoverable from one
module while leaving the legacy validation and execution shells untouched.
"""

from analyzer.calibration import (
    AffineCalibration,
    AffineCalibrator,
    CalibratedPrediction,
    FrozenAffineCalibration,
    LeakageSafeAffineCalibrator,
    calibrate_predictions,
    fit_affine_calibration,
    fit_calibration,
)
from analyzer.execution_costs import (
    DEFAULT_EXECUTION_COSTS,
    ExecutionCosts,
    apply_execution_costs,
    apply_execution_costs_pct,
    executable_alpha_pct,
    executable_return,
    executable_return_from_prices,
    executable_return_pct,
    net_executable_return,
    net_executable_return_pct,
)
from analyzer.target_frame import (
    build_executable_target_frame,
    build_mature_target_frame,
    build_target_frame,
    build_training_target_frame,
    mature_label_mask,
    target_frame,
)

__all__ = [
    "AffineCalibration",
    "AffineCalibrator",
    "CalibratedPrediction",
    "DEFAULT_EXECUTION_COSTS",
    "ExecutionCosts",
    "FrozenAffineCalibration",
    "LeakageSafeAffineCalibrator",
    "apply_execution_costs",
    "apply_execution_costs_pct",
    "build_executable_target_frame",
    "build_mature_target_frame",
    "build_target_frame",
    "build_training_target_frame",
    "calibrate_predictions",
    "executable_alpha_pct",
    "executable_return",
    "executable_return_from_prices",
    "executable_return_pct",
    "fit_affine_calibration",
    "fit_calibration",
    "mature_label_mask",
    "net_executable_return",
    "net_executable_return_pct",
    "target_frame",
]
