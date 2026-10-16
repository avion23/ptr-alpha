"""Immutable, self-contained receipts for the setups decision boundary.

Replay starts at scoring, not ingestion: the exact eligible candidate rows,
actor weights, watchlist and dated quotes are frozen here. No database is opened.
Hashes detect accidental tampering, not a malicious rewrite of file and hash.
Receipts live at data/recommendations/YYYY-MM-DD/decision.json; call
verify_decision(path) to compare the served result with the current scorer.
The embedded price reference is content-addressed over exactly the dated quotes
used by scoring (not a claim that the entire historical price cache is frozen).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from analyzer.price_snapshot import _get_git_sha, create_snapshot
from analyzer.setups import score


class DecisionConflict(ValueError):
    """A different decision has already been published for this date."""


def _freeze(value):
    if value is pd.NA:
        return {"$scalar": "NA"}
    if value is pd.NaT:
        return {"$scalar": "NaT"}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _freeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_freeze(v) for v in value]
    if hasattr(value, "item"):
        return _freeze(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"$scalar": str(value)}
    return value


def _thaw(value):
    if isinstance(value, dict):
        if set(value) == {"$scalar"}:
            tag = value["$scalar"]
            if tag in {"NA", "NaT"}:
                return {"NA": pd.NA, "NaT": pd.NaT}[tag]
            return float(tag)
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_thaw(v) for v in value]
    return value


def _json(value):
    return json.dumps(_freeze(value), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _frame(frame):
    return _freeze({"columns": list(frame.columns), "rows": frame.values.tolist()})


def _derive(inputs):
    frozen = _thaw(inputs)
    frame = frozen["candidates"]
    candidates = pd.DataFrame(frame["rows"], columns=frame["columns"], dtype=object)
    quotes = {k: tuple(v) for k, v in frozen["quotes"].items()}
    audit = {}
    ranked, blocked = score(
        candidates,
        frozen["weights"],
        quotes,
        watchlist=frozen["watchlist"],
        audit=audit,
    )
    rows = ranked.to_dict("records")
    for rank, row in enumerate(rows, 1):
        row.update(rank=rank, **audit[row["ticker"]])
    return _freeze(
        {"ranked": rows, "blocked": blocked.to_dict("records"), "eligibility": audit}
    )


def _price_reference(inputs, as_of):
    quotes = _thaw(inputs["quotes"])
    prices = pd.DataFrame(
        {
            ticker: pd.Series({pd.Timestamp(day): value})
            for ticker, (value, day) in quotes.items()
        }
    )
    if prices.empty:
        prices.index = pd.DatetimeIndex([])
    start = prices.index.min().date() if not prices.empty else as_of
    snapshot = create_snapshot(None, sorted(quotes), start, as_of, prices=prices)
    # Content-addressed reference avoids random UUID/time changes on reruns.
    return {
        "snapshot_id": "sha256:" + snapshot.value_hash,
        "value_hash": snapshot.value_hash,
        "coverage_by_ticker": snapshot.coverage_by_ticker,
    }


def _check_decision_date(inputs, as_of):
    frame = inputs["candidates"]
    if frame["rows"]:
        index = frame["columns"].index("as_of")
        if any(pd.Timestamp(row[index]).date() != as_of for row in frame["rows"]):
            raise ValueError("candidate as_of differs from decision_date")
    if any(
        pd.Timestamp(quote[1]).date() > as_of for quote in inputs["quotes"].values()
    ):
        raise ValueError("quote is after decision_date")


def record_decision(
    data_dir,
    as_of,
    candidates,
    weights,
    quotes,
    watchlist,
    ranked=None,
    blocked=None,
    *,
    scope=None,
):
    """Publish once; identical content is a no-op, different content conflicts.

    All input rows (including identities, flags, entry refs and corroborators)
    are frozen, not just winners. Publication is atomic and never replaces a file.
    """
    as_of = date.fromisoformat(str(as_of))
    inputs = _freeze(
        {
            "candidates": _frame(candidates),
            "weights": weights,
            "quotes": quotes,
            "watchlist": sorted(watchlist),
        }
    )
    _check_decision_date(inputs, as_of)
    outputs = _derive(inputs)
    if ranked is not None and blocked is not None:
        # Verify that freezing has not altered the actual served decision.
        replay = [{k: row[k] for k in ranked.columns} for row in outputs["ranked"]]
        if replay != _freeze(ranked.to_dict("records")) or outputs[
            "blocked"
        ] != _freeze(blocked.to_dict("records")):
            raise DecisionConflict("frozen inputs do not reproduce served decision")
    manifest = {
        "schema_version": 1,
        "decision_date": as_of.isoformat(),
        "git_sha": _get_git_sha(),
        "scope": scope or {},
        "inputs": inputs,
        "input_rows_hash": _hash(inputs["candidates"]),
        "price_snapshot": _price_reference(inputs, as_of),
        **outputs,
    }
    manifest["manifest_hash"] = _hash(manifest)
    content = _json(manifest)
    path = Path(data_dir) / "recommendations" / as_of.isoformat() / "decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hard-link a complete, fsynced temporary file with no replace operation.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".decision-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != content:
                raise DecisionConflict(f"decision conflict: {path}") from None
    finally:
        os.unlink(temporary)
    return path


def record_decision_safely(*args, **kwargs):
    """Receipt failures must never change trading output or exit status."""
    try:
        return record_decision(*args, **kwargs)
    except Exception:
        logging.getLogger(__name__).exception("Decision receipt was not recorded")
        return None


def verify_decision(path):
    """Re-score frozen inputs; report mismatch without touching canonical data."""
    try:
        manifest = json.loads(Path(path).read_text())
        digest = manifest.pop("manifest_hash")
        mismatches = []
        if _hash(manifest) != digest:
            mismatches.append("manifest_hash")
        if _hash(manifest["inputs"]["candidates"]) != manifest["input_rows_hash"]:
            mismatches.append("input_rows_hash")
        _check_decision_date(
            manifest["inputs"], date.fromisoformat(manifest["decision_date"])
        )
        expected = _derive(manifest["inputs"])
        for key, value in expected.items():
            if manifest[key] != value:
                mismatches.append(key)
        if manifest["price_snapshot"] != _price_reference(
            manifest["inputs"], date.fromisoformat(manifest["decision_date"])
        ):
            mismatches.append("price_snapshot")
        return {"agreement": not mismatches, "mismatches": mismatches}
    except Exception as exc:
        return {"agreement": False, "mismatches": [f"invalid receipt: {exc}"]}
