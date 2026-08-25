"""Canonical identity for finite ordered parameter-grid families.

The parameter names and the order of their candidate values are part of an
experiment's identity.  This is deliberate: changing either changes the
enumerated trial IDs and therefore changes the multiple-testing family.  The
identity is based only on the declared grid, never on observed result rows.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any


FAMILY_SCHEMA_VERSION = 1
FAMILY_PROVENANCE = "canonical_ordered_grid_enumeration_v1"


def _deep_freeze(value: Any) -> Any:
    """Recursively detach and freeze JSON-compatible container values."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=_canonical_json)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("grid values must be finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"grid value {value!r} is not JSON serializable")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _ordered_grid(
    grid: Mapping[str, Iterable[Any]],
    *,
    parameter_order: Sequence[str] | None = None,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("experiment grid must be a non-empty mapping")

    names = tuple(str(name) for name in grid)
    if len(set(names)) != len(names):
        raise ValueError("experiment grid parameter names must be unique")
    if parameter_order is not None:
        requested = tuple(str(name) for name in parameter_order)
        if len(requested) != len(names) or set(requested) != set(names):
            raise ValueError("parameter_order must contain every parameter once")
        names = requested

    by_name = {str(name): values for name, values in grid.items()}
    normalized: list[tuple[str, tuple[Any, ...]]] = []
    for name in names:
        values = by_name[name]
        if isinstance(values, (str, bytes)):
            raise TypeError(f"grid values for {name!r} must be an iterable")
        if isinstance(values, (set, frozenset)):
            values = sorted(values, key=_canonical_json)
        elif not isinstance(values, Sequence):
            if not isinstance(values, Iterable):
                raise TypeError(f"grid values for {name!r} must be iterable")
            values = tuple(values)
        else:
            values = tuple(values)
        if not values:
            raise ValueError(f"grid parameter {name!r} must not be empty")
        normalized.append(
            (name, tuple(_deep_freeze(_json_safe(value)) for value in values))
        )
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class GridTrial:
    """One deterministic configuration and its stable specification hash."""

    trial_id: int
    config: Mapping[str, Any]
    trial_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.trial_id, bool) or not isinstance(self.trial_id, int):
            raise TypeError("trial_id must be an integer")
        frozen = _deep_freeze(_json_safe(self.config))
        if not isinstance(frozen, Mapping):
            raise TypeError("trial config must be a mapping")
        object.__setattr__(self, "config", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "config": _json_safe(self.config),
            "trial_sha256": self.trial_sha256,
        }


@dataclass(frozen=True, slots=True)
class GridFamily:
    """An immutable, ordered grid family and its canonical SHA-256."""

    grid: tuple[tuple[str, tuple[Any, ...]], ...]
    trials: tuple[GridTrial, ...]
    family_sha256: str
    provenance: str = FAMILY_PROVENANCE

    def __post_init__(self) -> None:
        frozen_grid = tuple(
            (
                str(name),
                tuple(_deep_freeze(_json_safe(value)) for value in values),
            )
            for name, values in self.grid
        )
        object.__setattr__(self, "grid", frozen_grid)
        object.__setattr__(self, "trials", tuple(self.trials))

    @property
    def family_hash(self) -> str:
        return self.family_sha256

    @property
    def family_size(self) -> int:
        return len(self.trials)

    @property
    def size(self) -> int:
        return self.family_size

    @property
    def parameter_order(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.grid)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "provenance": self.provenance,
            "family_provenance": self.provenance,
            "family_sha256": self.family_sha256,
            "family_hash": self.family_sha256,
            "family_size": self.family_size,
            "parameter_order": list(self.parameter_order),
            "trial_ids": [trial.trial_id for trial in self.trials],
            "trial_spec_sha256": [
                trial.trial_sha256 for trial in self.trials
            ],
            "ordered_grid": [
                {
                    "parameter": name,
                    "values": _json_safe(values),
                }
                for name, values in self.grid
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata(),
            "trials": [trial.to_dict() for trial in self.trials],
        }


def build_family(
    grid: Mapping[str, Iterable[Any]],
    *,
    parameter_order: Sequence[str] | None = None,
) -> GridFamily:
    """Materialize and hash the full ordered Cartesian product."""
    ordered = _ordered_grid(grid, parameter_order=parameter_order)
    names = tuple(name for name, _ in ordered)
    values = tuple(candidates for _, candidates in ordered)
    trial_payloads = [
        {
            "trial_id": trial_id,
            "config": dict(zip(names, combination)),
            "parameter_order": list(names),
        }
        for trial_id, combination in enumerate(itertools.product(*values))
    ]
    family_payload = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "provenance": FAMILY_PROVENANCE,
        "parameter_order": list(names),
        "ordered_grid": [
            {"parameter": name, "values": list(candidates)}
            for name, candidates in ordered
        ],
        "trials": trial_payloads,
    }
    family_digest = _sha256_json(family_payload)
    trials = tuple(
        GridTrial(
            trial_id=trial["trial_id"],
            config=trial["config"],
            trial_sha256=trial_spec_sha256(
                family_digest,
                trial["trial_id"],
                trial["config"],
                names,
            ),
        )
        for trial in trial_payloads
    )
    return GridFamily(ordered, trials, family_digest)


def trial_spec_sha256(
    family_sha256: str,
    trial_id: int,
    config: Mapping[str, Any],
    parameter_order: Sequence[str],
) -> str:
    """Hash one trial's actual ordered configuration within a family."""
    if isinstance(trial_id, bool) or not isinstance(trial_id, int):
        raise TypeError("trial_id must be an integer")
    return _sha256_json(
        {
            "family_sha256": family_sha256,
            "trial_id": trial_id,
            "config": config,
            "parameter_order": list(parameter_order),
        }
    )


__all__ = [
    "FAMILY_PROVENANCE",
    "FAMILY_SCHEMA_VERSION",
    "GridFamily",
    "GridTrial",
    "build_family",
    "trial_spec_sha256",
]
