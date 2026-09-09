"""Compatibility exports for the research-only baseline package.

Use :mod:`analyzer.research` for new code.  This module intentionally contains
no production integration and exists only to make the research helpers easy to
discover from a flat ``analyzer`` namespace.
"""

from analyzer.research.baselines import (
    DeterministicRegularizedTabularBaseline,
    DynamicHierarchicalBaseline,
    ResearchModelProvenance,
    ResearchOnlyError,
)
from analyzer.research.outcomes import (
    FactorFitMetadata,
    compute_spy_factor_residual_outcomes,
)

__all__ = [
    "DeterministicRegularizedTabularBaseline",
    "DynamicHierarchicalBaseline",
    "FactorFitMetadata",
    "ResearchModelProvenance",
    "ResearchOnlyError",
    "compute_spy_factor_residual_outcomes",
]
