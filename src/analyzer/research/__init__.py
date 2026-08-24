"""Research-only outcome and baseline helpers.

Nothing in this package is wired into live scoring, validation, portfolio
simulation, or deployment.  See :mod:`analyzer.research.outcomes` and
:mod:`analyzer.research.baselines` for the explicit point-in-time boundaries.
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
