from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class AnalyzerError(Exception):
    pass

class DataSourceError(AnalyzerError):
    pass

class ParsingError(AnalyzerError):
    pass

class AnalysisError(AnalyzerError):
    pass

class ConfigurationError(AnalyzerError):
    pass


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of a pipeline step execution."""
    success: bool
    error: Exception | None = None

@dataclass(frozen=True, slots=True)
class DataResult(Generic[T]):
    """Result of a computation that returns data."""
    success: bool
    data: T | None = None
    error: Exception | None = None

    @classmethod
    def ok(cls, data: T) -> DataResult[T]:
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: Exception) -> DataResult[T]:
        return cls(success=False, error=error)
