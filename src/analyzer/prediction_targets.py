"""Compatibility facade for pure executable target construction."""

from analyzer.execution_costs import (
    DEFAULT_EXECUTION_COSTS,
    ExecutionCosts,
    executable_alpha_pct,
    executable_return,
    executable_return_from_prices,
    executable_return_pct,
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
    "DEFAULT_EXECUTION_COSTS",
    "ExecutionCosts",
    "build_executable_target_frame",
    "build_mature_target_frame",
    "build_target_frame",
    "build_training_target_frame",
    "executable_alpha_pct",
    "executable_return",
    "executable_return_from_prices",
    "executable_return_pct",
    "mature_label_mask",
    "target_frame",
]
