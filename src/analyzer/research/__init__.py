"""Research-only outcome and baseline helpers.

Nothing in this package is wired into live scoring, validation, portfolio
simulation, or deployment.  See :mod:`analyzer.research.outcomes` and
:mod:`analyzer.research.baselines` for the explicit point-in-time boundaries.
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
