"""Immutable, self-contained receipts for production decision boundaries.

Replay starts at scoring, not ingestion: the exact eligible candidate rows,
actor weights, watchlist and dated quotes are frozen here. No database is
opened. Candidate rows carry their canonical filing identities (source_record_id,
source_row_id and, for holdings, the contributing lot/filing references), so a
receipt reconstructs input provenance, not just scoring.

Every publication is anchored OUTSIDE the writer-controlled receipt file:

- One receipt per distinct decision at
  ``recommendations/<date>/<decision_id>.json`` where ``decision_id`` is the
  hash of the canonical payload EXCLUDING ``manifest_hash`` (content
  addressing: an identical rerun is a true no-op, a genuinely different
  decision on the same date gets its own receipt, its own ledger line and is
  logged as recorded — never swallowed as a conflict).
- An append-only ledger ``recommendations/receipts.log``: one canonical-JSON
  line per publication, ``{decision_date, decision_id, manifest_hash,
  prev_line_hash, published_at, git_sha}``, forming a hash chain
  (``prev_line_hash`` links to the previous line's canonical hash), written
  with O_APPEND + fsync followed by a directory fsync. O_APPEND serializes
  byte placement, NOT predecessor selection, so the whole read-append-verify
  cycle runs under an exclusive ``fcntl.flock`` on ``receipts.log.lock``
  beside the ledger: concurrent writers serialize instead of both linking to
  the same tail.

``verify_decision`` reports hash-chain integrity separately from replay under
the receipt's anchored scorer/parser recipe. Pre-recipe receipts are integrity
checked only and are never retroactively replayed under current code.

Residual risk (documented honestly in docs/decision-receipts.md): the process
owner can rewrite both files. The ledger does not provide cryptographic
immutability; it forces any tampering to rewrite detectable chain history —
a much louder, detectable forgery.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from analyzer.price_snapshot import _get_git_sha, create_snapshot
from analyzer.setups import score


class DecisionConflict(ValueError):
    """A published receipt file collided with different content."""


class LedgerError(ValueError):
    """The receipt ledger is missing, unreadable or its chain is broken."""


class _ReplayUnavailable(ValueError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


LEDGER_NAME = "receipts.log"
GENESIS_HASH = "0" * 64
LEDGER_KEYS = (
    "decision_date",
    "decision_id",
    "manifest_hash",
    "prev_line_hash",
    "published_at",
    "git_sha",
)
SCORER_NAME = "analyzer.setups.score"
SCORER_VERSION = "3"
_REPLAYABLE_SCORER_VERSIONS = frozenset({"3"})
PARSER_NAME = "analyzer.decision_snapshot"
PARSER_VERSION = "1"
_CONTROL = "enable_13f_corroboration"
DEFAULT_SCORE_PARAMETERS = {"entry_tol": 1.05, "event_half_life_days": 90}


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


def _canonical_line(entry):
    return (
        json.dumps(_freeze(entry), sort_keys=True, separators=(",", ":"),
                   allow_nan=False)
        + "\n"
    )


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _line_hash(entry):
    return hashlib.sha256(_canonical_line(entry).encode()).hexdigest()


def _frame(frame):
    return _freeze({"columns": list(frame.columns), "rows": frame.values.tolist()})


def _derive(inputs, recipe=None):
    frozen = _thaw(inputs)
    frame = frozen["candidates"]
    candidates = pd.DataFrame(frame["rows"], columns=frame["columns"], dtype=object)
    quotes = {k: tuple(v) for k, v in frozen["quotes"].items()}
    audit = {}
    scorer = (recipe or {}).get("scorer", {})
    scorer_parameters = scorer.get("parameters")
    if (
        not isinstance(scorer_parameters, dict)
        or _CONTROL not in scorer_parameters
        or type(scorer_parameters[_CONTROL]) is not bool
    ):
        raise ValueError(f"recipe.scorer.parameters.{_CONTROL} must be a bool")
    parameters = {**DEFAULT_SCORE_PARAMETERS, **scorer_parameters}
    ranked, blocked = score(
        candidates,
        frozen["weights"],
        quotes,
        entry_tol=parameters["entry_tol"],
        event_half_life_days=parameters["event_half_life_days"],
        watchlist=frozen["watchlist"],
        enable_13f_corroboration=scorer_parameters[_CONTROL],
        audit=audit,
        ex_dividends=frozen.get("ex_dividends"),
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


# --------------------------------------------------------------------------
# Append-only receipt ledger (the anchor outside the writer-controlled file)
# --------------------------------------------------------------------------


def _ledger_path(recommendations):
    return Path(recommendations) / LEDGER_NAME


def _ledger_lock_path(recommendations):
    return Path(recommendations) / (LEDGER_NAME + ".lock")


@contextmanager
def _ledger_lock(recommendations):
    """Exclusive interprocess lock held across one publication's whole cycle.

    O_APPEND serializes byte placement, not predecessor selection: without
    this lock two writers can read the same tail and both append a line whose
    ``prev_line_hash`` points at it, breaking the chain. Closing the
    descriptor (on any exit path, including a crash) releases the lock, so a
    dead writer can never wedge later publications.
    """
    fd = os.open(
        str(_ledger_lock_path(recommendations)), os.O_WRONLY | os.O_CREAT, 0o600
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _load_ledger(recommendations):
    """Return [(entry, line_hash), ...]; raise LedgerError on any defect.

    The whole chain is verified on every read: each line must parse, carry
    exactly the ledger keys, and link to the previous line's canonical hash.
    A missing ledger fails verification; publication creates the first ledger
    entry or raises so the serving path can stop without an unanchored receipt.
    """
    path = _ledger_path(recommendations)
    entries = []
    previous = GENESIS_HASH
    try:
        text = path.read_text()
    except FileNotFoundError:
        return entries
    except OSError as exc:
        raise LedgerError(f"ledger unreadable: {exc}") from None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"ledger line {number} is not JSON: {exc}") from None
        if not isinstance(entry, dict) or set(entry) != set(LEDGER_KEYS):
            raise LedgerError(f"ledger line {number} has unexpected fields")
        # Content addressing makes decision_id and manifest_hash one and the
        # same value; a line where they diverge has been rewritten.
        if entry["decision_id"] != entry["manifest_hash"]:
            raise LedgerError(
                f"ledger line {number} decision_id does not match its manifest_hash"
            )
        if entry["prev_line_hash"] != previous:
            raise LedgerError(f"ledger line {number} breaks the hash chain")
        previous = _line_hash(entry)
        entries.append((entry, previous))
    return entries


def _append_ledger_entry(recommendations, ledger, manifest, decision_id):
    entry = {
        "decision_date": manifest["decision_date"],
        "decision_id": decision_id,
        "manifest_hash": manifest["manifest_hash"],
        "prev_line_hash": ledger[-1][1] if ledger else GENESIS_HASH,
        "published_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_sha": manifest.get("git_sha"),
    }
    line = _canonical_line(entry).encode()
    fd = os.open(
        str(_ledger_path(recommendations)), os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_directory(recommendations)
    return entry


def _fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_receipt(path, content):
    """Hard-link a complete, fsynced temporary file; never replace a receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
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
                raise DecisionConflict(f"receipt collision: {path}") from None
        else:
            os.chmod(path, 0o444)
        _fsync_directory(path.parent)
    finally:
        os.unlink(temporary)


def _publish(data_dir, manifest):
    """Content-addressed publication shared by every decision boundary.

    ``decision_id`` is the hash of the canonical payload EXCLUDING
    ``manifest_hash``; identical reruns resolve to the same id and are a
    true no-op. Different decisions on a date each get their own receipt
    file and ledger line, and every additional distinct decision for a date
    is logged as recorded (visible, never swallowed). The whole cycle —
    ledger load, duplicate detection, receipt publication, append+fsync and
    final verification — runs under the interprocess ledger lock, because
    O_APPEND does not serialize predecessor selection between writers.
    """
    decision_id = _hash(manifest)
    manifest["manifest_hash"] = decision_id
    content = _json(manifest)
    recommendations = Path(data_dir) / "recommendations"
    recommendations.mkdir(parents=True, exist_ok=True)
    with _ledger_lock(recommendations):
        ledger = _load_ledger(recommendations)
        already = [
            entry
            for entry, _ in ledger
            if entry.get("decision_id") == decision_id
            and entry.get("decision_date") == manifest["decision_date"]
        ]
        path = recommendations / manifest["decision_date"] / f"{decision_id}.json"
        if path.exists():
            if path.read_text() != content:
                raise DecisionConflict(f"receipt collision: {path}")
        else:
            _write_receipt(path, content)
        if already:
            logging.getLogger(__name__).debug(
                "Decision %s already recorded; idempotent no-op", decision_id
            )
            return _verify_published(path)
        _append_ledger_entry(recommendations, ledger, manifest, decision_id)
        logger = logging.getLogger(__name__)
        siblings = {
            entry.get("decision_id")
            for entry, _ in ledger
            if entry.get("decision_date") == manifest["decision_date"]
        }
        if siblings:
            logger.info(
                "Recorded decision %s for %s; %d distinct decisions now recorded "
                "for this date",
                decision_id,
                manifest["decision_date"],
                len(siblings) + 1,
            )
        else:
            logger.info(
                "Recorded decision %s for %s", decision_id, manifest["decision_date"]
            )
        return _verify_published(path)


def _verify_published(path):
    result = verify_decision(path)
    if not result["agreement"]:
        raise LedgerError(
            "published receipt failed verification: " + ", ".join(result["mismatches"])
        )
    return path


# --------------------------------------------------------------------------
# Decision boundaries
# --------------------------------------------------------------------------


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
    ex_dividends=None,
    dividend_source="caller-provided",
    entry_tol=1.05,
    event_half_life_days=90,
    enable_13f_corroboration: bool | None = None,
):
    """Record one setups decision as an immutable, self-contained receipt.

    All input rows (including identities, flags, entry refs and corroborators)
    are frozen, not just winners. Identical content is an idempotent no-op;
    different content on the same date is recorded separately and visibly.
    """
    from analyzer.positions import _13f_enabled

    enable_13f_corroboration = _13f_enabled(enable_13f_corroboration)
    as_of = date.fromisoformat(str(as_of))
    inputs = _freeze(
        {
            "candidates": _frame(candidates),
            "weights": weights,
            "quotes": quotes,
            "watchlist": sorted(watchlist),
            "ex_dividends": ex_dividends,
        }
    )
    _check_decision_date(inputs, as_of)
    scorer_parameters = {
        "entry_tol": entry_tol,
        "event_half_life_days": event_half_life_days,
        _CONTROL: enable_13f_corroboration,
    }
    recipe = {
        "scorer": {
            "name": SCORER_NAME,
            "version": SCORER_VERSION,
            "parameters": scorer_parameters,
        },
        "parser": {"name": PARSER_NAME, "version": PARSER_VERSION},
        "dividends": {
            "source": dividend_source,
            "input_hash": _hash(inputs["ex_dividends"]),
        },
        "frame": {
            "input_hash": _hash(inputs["candidates"]),
            "columns": list(candidates.columns),
            "row_count": len(candidates),
        },
    }
    outputs = _derive(inputs, recipe)
    if ranked is not None and blocked is not None:
        # Verify that freezing has not altered the actual served decision.
        replay = [{k: row[k] for k in ranked.columns} for row in outputs["ranked"]]
        if replay != _freeze(ranked.to_dict("records")) or outputs[
            "blocked"
        ] != _freeze(blocked.to_dict("records")):
            raise DecisionConflict("frozen inputs do not reproduce served decision")
    price_snapshot = _price_reference(inputs, as_of)
    manifest = {
        "schema_version": 3,
        "kind": "setups",
        "decision_date": as_of.isoformat(),
        "git_sha": _get_git_sha(),
        "scope": scope or {},
        "inputs": inputs,
        "input_rows_hash": _hash(inputs["candidates"]),
        "price_snapshot": price_snapshot,
        "recipe": {
            **recipe,
            "price_snapshot_id": price_snapshot["snapshot_id"],
        },
        **outputs,
    }
    return _publish(data_dir, manifest)


def record_ticker_decision(data_dir, as_of, inputs, outputs, *, scope=None):
    """Record a consensus-ticker decision (analyze --ticker/--mode tickers).

    The served buyer/score tables are frozen as the receipt content. These
    receipts anchor what was served in the ledger like every other decision;
    unlike setups receipts they do not claim an offline replay, because the
    consensus scorer is identity-free by design and needs the database.
    """
    as_of = date.fromisoformat(str(as_of))
    frozen_inputs = {
        str(name): _frame(value) if isinstance(value, pd.DataFrame) else _freeze(value)
        for name, value in (inputs or {}).items()
    }
    frozen_outputs = {
        str(name): _frame(value) if isinstance(value, pd.DataFrame) else _freeze(value)
        for name, value in (outputs or {}).items()
    }
    manifest = {
        "schema_version": 3,
        "kind": "ticker_consensus",
        "decision_date": as_of.isoformat(),
        "git_sha": _get_git_sha(),
        "scope": scope or {},
        "inputs": frozen_inputs,
        "input_rows_hash": _hash(frozen_inputs),
        "outputs": frozen_outputs,
        "output_rows_hash": _hash(frozen_outputs),
        "recipe": {
            "scorer": {
                "name": "analyzer.ticker_consensus",
                "version": "1",
                "parameters": {"scope": scope or {}},
            },
            "parser": {"name": PARSER_NAME, "version": PARSER_VERSION},
            "dividends": {"source": "not_applicable", "input_hash": _hash(None)},
            "frame": {"input_hash": _hash(frozen_inputs)},
            "price_snapshot_id": None,
        },
    }
    return _publish(data_dir, manifest)


def record_decision_safely(*args, **kwargs):
    """Log publication failures and propagate them to the serving boundary."""
    try:
        return record_decision(*args, **kwargs)
    except Exception:
        logging.getLogger(__name__).exception("Decision receipt was not recorded")
        raise


def record_ticker_decision_safely(*args, **kwargs):
    """Log publication failures and propagate them to the serving boundary."""
    try:
        return record_ticker_decision(*args, **kwargs)
    except Exception:
        logging.getLogger(__name__).exception("Decision receipt was not recorded")
        raise


def date_index(data_dir, as_of):
    """Every decision_id recorded in the ledger for a date, in ledger order."""
    day = date.fromisoformat(str(as_of)).isoformat()
    return [
        entry["decision_id"]
        for entry, _ in _load_ledger(Path(data_dir) / "recommendations")
        if entry["decision_date"] == day
    ]


def verify_decision(path, *, current_dividends=None):
    """Check hash-chain integrity separately from recorded-recipe replay.

    Receipts predating recipe anchors are integrity checked only and explicitly
    reported as ``pre-recipe anchor``; their outputs are never compared with
    today's scorer. ``current_dividends`` is an optional diagnostic override
    for testing/replaying against a later dividend snapshot. The default always
    replays the dividend input frozen in the receipt.
    """
    integrity_mismatches = []
    replay_mismatches = []
    replay_status = "fail"
    stage = "integrity"
    try:
        manifest = json.loads(Path(path).read_text())
        digest = manifest.pop("manifest_hash")
        if _hash(manifest) != digest:
            integrity_mismatches.append("manifest_hash")
        try:
            ledger = _load_ledger(Path(path).parent.parent)
        except LedgerError as exc:
            integrity_mismatches.append(f"ledger: {exc}")
        else:
            entries = [
                entry for entry, _ in ledger if entry.get("decision_id") == digest
            ]
            if not entries:
                integrity_mismatches.append("receipt_not_in_ledger")
            else:
                if any(entry.get("manifest_hash") != digest for entry in entries):
                    integrity_mismatches.append("ledger_manifest_hash")
                if not any(
                    entry.get("decision_date") == manifest.get("decision_date")
                    for entry in entries
                ):
                    integrity_mismatches.append("ledger_decision_date")
        if manifest.get("kind") == "ticker_consensus":
            if _hash(manifest["inputs"]) != manifest["input_rows_hash"]:
                integrity_mismatches.append("inputs")
            if _hash(manifest["outputs"]) != manifest["output_rows_hash"]:
                integrity_mismatches.append("outputs")
            replay_status = (
                "snapshot-only" if "recipe" in manifest else "pre-recipe anchor"
            )
        elif "recipe" not in manifest:
            replay_status = "pre-recipe anchor"
        else:
            if integrity_mismatches:
                replay_status = "not-run"
            else:
                stage = "replay"
                recipe = manifest["recipe"]
                scorer = recipe.get("scorer", {})
                parser = recipe.get("parser", {})
                if (
                    not isinstance(scorer, dict)
                    or scorer.get("name") != SCORER_NAME
                    or not isinstance(scorer.get("version"), str)
                    or scorer["version"] not in _REPLAYABLE_SCORER_VERSIONS
                ):
                    raise _ReplayUnavailable(
                        "unsupported-version", "recipe.scorer_version"
                    )
                if (
                    not isinstance(parser, dict)
                    or parser.get("name") != PARSER_NAME
                    or parser.get("version") != PARSER_VERSION
                ):
                    raise _ReplayUnavailable(
                        "unsupported-version", "recipe.parser_version"
                    )
                inputs = manifest["inputs"]
                as_of = date.fromisoformat(manifest["decision_date"])
                _check_decision_date(inputs, as_of)
                if _hash(inputs["candidates"]) != manifest["input_rows_hash"]:
                    replay_mismatches.append("input_rows_hash")
                frame_anchor = recipe.get("frame", {})
                candidate_frame = inputs["candidates"]
                if (
                    frame_anchor.get("input_hash") != _hash(candidate_frame)
                    or frame_anchor.get("columns") != candidate_frame["columns"]
                    or frame_anchor.get("row_count") != len(candidate_frame["rows"])
                ):
                    replay_mismatches.append("recipe.frame")
                if recipe.get("dividends", {}).get("input_hash") != _hash(
                    inputs.get("ex_dividends")
                ):
                    replay_mismatches.append("recipe.dividends")
                if recipe.get("price_snapshot_id") != manifest["price_snapshot"].get(
                    "snapshot_id"
                ):
                    replay_mismatches.append("recipe.price_snapshot_id")
                if manifest["price_snapshot"] != _price_reference(inputs, as_of):
                    replay_mismatches.append("price_snapshot")
                if replay_mismatches:
                    raise _ReplayUnavailable("fail", "invalid frozen replay anchors")
                if current_dividends is not None:
                    inputs = {**inputs, "ex_dividends": _freeze(current_dividends)}
                expected = _derive(inputs, recipe)
                for key, value in expected.items():
                    if manifest[key] != value:
                        replay_mismatches.append(key)
                replay_status = "fail" if replay_mismatches else "pass"
        integrity_ok = not integrity_mismatches
        replay_ok = replay_status in {"pass", "snapshot-only", "pre-recipe anchor"}
        mismatches = integrity_mismatches + replay_mismatches
        return {
            "agreement": integrity_ok and replay_ok,
            "mismatches": mismatches,
            "integrity": {
                "agreement": integrity_ok,
                "mismatches": integrity_mismatches,
            },
            "replay": {"status": replay_status, "mismatches": replay_mismatches},
        }
    except _ReplayUnavailable as exc:
        if stage == "replay":
            if str(exc) not in replay_mismatches:
                replay_mismatches.append(str(exc))
            integrity_ok = not integrity_mismatches
            return {
                "agreement": False,
                "mismatches": integrity_mismatches + replay_mismatches,
                "integrity": {
                    "agreement": integrity_ok,
                    "mismatches": integrity_mismatches,
                },
                "replay": {"status": exc.status, "mismatches": replay_mismatches},
            }
        integrity_mismatches.append(f"invalid receipt: {exc}")
        return {
            "agreement": False,
            "mismatches": integrity_mismatches + replay_mismatches,
            "integrity": {"agreement": False, "mismatches": integrity_mismatches},
            "replay": {"status": "fail", "mismatches": replay_mismatches},
        }
    except Exception as exc:
        if stage == "replay":
            replay_mismatches.append(f"replay_error: {exc}")
            integrity_ok = not integrity_mismatches
            return {
                "agreement": False,
                "mismatches": integrity_mismatches + replay_mismatches,
                "integrity": {
                    "agreement": integrity_ok,
                    "mismatches": integrity_mismatches,
                },
                "replay": {"status": "fail", "mismatches": replay_mismatches},
            }
        integrity_mismatches.append(f"invalid receipt: {exc}")
        return {
            "agreement": False,
            "mismatches": integrity_mismatches + replay_mismatches,
            "integrity": {"agreement": False, "mismatches": integrity_mismatches},
            "replay": {"status": "fail", "mismatches": replay_mismatches},
        }
