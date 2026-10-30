"""JSON-only work and result contracts; workers receive no database location."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID


class ExtractionLane(str, Enum):
    LOCAL = "local"
    GEMINI = "gemini"
    GLM = "glm"


class ExtractionOutcome(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED_EMPTY = "verified_empty"
    RETRY_WAIT = "retry_wait"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


def _text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _identity(doc_id, archive_year, generation_id, artifact_sha256, extraction_recipe, lane):
    for name, value in (
        ("doc_id", doc_id), ("generation_id", generation_id),
        ("extraction_recipe", extraction_recipe),
    ):
        _text(name, value)
    if type(archive_year) is not int or not 1000 <= archive_year <= 9999:
        raise ValueError("archive_year must be a four-digit integer")
    if not isinstance(artifact_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
        raise ValueError("artifact_sha256 must be a lowercase SHA-256 hex digest")
    if not isinstance(lane, ExtractionLane):
        raise ValueError("lane must be an ExtractionLane")
    return [doc_id, archive_year, generation_id, artifact_sha256, extraction_recipe, lane.value]


def task_key(
    doc_id: str,
    archive_year: int,
    generation_id: str,
    artifact_sha256: str,
    extraction_recipe: str,
    lane: ExtractionLane,
) -> str:
    """Stable replay identity, independent of attempt UUID and artifact location."""
    import json

    identity = _identity(doc_id, archive_year, generation_id, artifact_sha256, extraction_recipe, lane)
    return hashlib.sha256(json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def _json_value(value: object) -> None:
    """Reject non-JSON objects rather than silently coercing raw row fields."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("rows and evidence must contain only finite JSON-native values")


@dataclass(frozen=True)
class WorkEnvelope:
    task_id: str
    attempt_id: str
    doc_id: str
    archive_year: int
    generation_id: str
    artifact_sha256: str
    artifact_path: str
    member: str
    filing_date: str
    page_count: int
    extraction_recipe: str
    lane: ExtractionLane

    def __post_init__(self) -> None:
        _identity(self.doc_id, self.archive_year, self.generation_id, self.artifact_sha256, self.extraction_recipe, self.lane)
        for name in ("task_id", "attempt_id", "artifact_path", "member", "filing_date"):
            _text(name, getattr(self, name))
        try:
            UUID(self.attempt_id)
        except ValueError as exc:
            raise ValueError("attempt_id must be a caller-provided UUID") from exc
        if type(self.page_count) is not int or self.page_count < 1:
            raise ValueError("page_count must be a positive integer")

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "attempt_id": self.attempt_id,
            "doc_id": self.doc_id, "archive_year": self.archive_year,
            "generation_id": self.generation_id, "artifact_sha256": self.artifact_sha256,
            "artifact_path": self.artifact_path, "member": self.member,
            "filing_date": self.filing_date, "page_count": self.page_count,
            "extraction_recipe": self.extraction_recipe, "lane": self.lane.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WorkEnvelope:
        values = dict(data)
        values["lane"] = ExtractionLane(values["lane"])
        return cls(**values)


@dataclass(frozen=True)
class ResultBundle:
    work: WorkEnvelope
    outcome: ExtractionOutcome
    rows: list[dict]
    raw_response: str | None = None
    evidence: dict = field(default_factory=dict)
    detail: str | None = None
    retry_after: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.work, WorkEnvelope):
            raise ValueError("work must be a WorkEnvelope")
        if not isinstance(self.outcome, ExtractionOutcome):
            raise ValueError("outcome must be an ExtractionOutcome")
        if type(self.rows) is not list or any(type(row) is not dict for row in self.rows):
            raise ValueError("rows must be a list of raw row dictionaries")
        if type(self.evidence) is not dict:
            raise ValueError("evidence must be a dictionary")
        _json_value(self.rows)
        _json_value(self.evidence)
        for name in ("raw_response", "detail"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        if self.retry_after is not None and (
            type(self.retry_after) not in (int, float)
            or not math.isfinite(self.retry_after) or self.retry_after < 0
        ):
            raise ValueError("retry_after must be finite nonnegative seconds or None")

    def to_dict(self) -> dict:
        from copy import deepcopy

        return {
            "work": self.work.to_dict(), "outcome": self.outcome.value,
            "rows": deepcopy(self.rows), "raw_response": self.raw_response,
            "evidence": deepcopy(self.evidence), "detail": self.detail,
            "retry_after": self.retry_after,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResultBundle:
        from copy import deepcopy

        values = deepcopy(data)
        values["work"] = WorkEnvelope.from_dict(values["work"])
        values["outcome"] = ExtractionOutcome(values["outcome"])
        return cls(**values)
