"""Compatibility exports for the research-only baseline package.

Use :mod:`analyzer.research` for new code.  This module intentionally contains
no production integration and exists only to make the research helpers easy to
discover from a flat ``analyzer`` namespace.
"""

from analyzer.research.baselines import (
    DeterministicRegularizedTabularBaseline,
    DeterministicTabularBaseline,
    DynamicHierarchicalBaseline,
    RegularizedTabularBaseline,
    ResearchModelProvenance,
    ResearchOnlyError,
)
from analyzer.research.outcomes import (
    FactorFitMetadata,
    build_spy_factor_residual_outcomes,
    compute_spy_factor_residual_outcomes,
    point_in_time_spy_factor_residual_outcomes,
)

__all__ = [
    "DeterministicRegularizedTabularBaseline",
    "DeterministicTabularBaseline",
    "DynamicHierarchicalBaseline",
    "FactorFitMetadata",
    "RegularizedTabularBaseline",
    "ResearchModelProvenance",
    "ResearchOnlyError",
    "build_spy_factor_residual_outcomes",
    "compute_spy_factor_residual_outcomes",
    "point_in_time_spy_factor_residual_outcomes",
]
