"""Chronological backtest of the system's load-bearing assumptions.

Phase 1 (skill): label members using only pre-2023 disclosures whose 90-day
outcomes matured before each decision date. Skilled = mean excess > +3pp with
>=5 unique episodes. Hyperactive = point-in-time breadth > 150.

Phase 2 (cohorts): score from point-in-time inputs and the last quote on or
  before each decision date, through the production actor-weight path
  (weights_for_candidates), so research weights are live weights. Subsequent
  executable fills and forward outcomes live in separate artifacts and never
  enter the scored snapshot; they are produced by the production evaluator
  (evaluate_backtest) with declared costs at exact NYSE endpoints. A missing
  expected entry, exit, or same-session SPY quote stays unavailable — a
  nearby session is never substituted.

Usage:
  python3 scripts/backtest_assumptions.py skill --frame {disclosure,ingestion}
  python3 scripts/backtest_assumptions.py run 2023-01-15 --frame ingestion
  python3 scripts/backtest_assumptions.py capture 2027-01-15 --frame ingestion
  python3 scripts/backtest_assumptions.py attach-outcomes 2027-01-15 --frame ingestion
  python3 scripts/backtest_assumptions.py analyze-capture 2027-01-15 --frame ingestion
  python3 scripts/backtest_assumptions.py analyze --frame disclosure
  python3 scripts/backtest_assumptions.py sweep --frame disclosure
  python3 scripts/backtest_assumptions.py full --frame ingestion
run never rescores frozen walk-forward dates; capture is D+1 and outcomes/analysis
remain sealed until 2028-11-01. Historical run refuses degenerate frames (<50
events or <20 candidate tickers).
--frame has no default anywhere: the frame is never implicit again. Ingestion
replays the live canonical_as_of(decision) scope; disclosure replays today's
complete database filtered by disclosure_date <= decision. Every artifact
records the frame that produced it (frame column + manifest_<D>.json).
Results land in data/backtest_assumptions/.
Frozen D+1 inputs publish as walkforward/captures/<decision>/ bundles;
legacy flat captures remain readable without being rewritten. Their attached
outcomes live at walkforward/outcomes_<decision>.csv.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import closing, contextmanager, nullcontext
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    for _entry in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
        if _entry not in sys.path:
            sys.path.insert(0, _entry)

import numpy as np
import pandas as pd
from scipy import stats

from analyzer import positions
from analyzer.actors import actor_id, weights_for_candidates
from analyzer.backtest.evaluate import evaluate_backtest
from analyzer.persistence.database import Database
from analyzer.manager_watchlist import watchlist_actor_ids
from analyzer.member_names import canonical_member_key
from analyzer.pipeline import eligible_events
from analyzer.positions import (
    coalesce_actor_variants,
    dedupe_member_names,
    gate_closed_events,
    holdings_candidates,
)
from analyzer.persistence.price_repository import next_nyse_session, previous_nyse_session
from analyzer.setups import score as score_setups

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "backtest_assumptions"
DB_PATH = Path("/home/ubuntu/insider-trading/data/congress.duckdb")
HORIZON = 90
MAX_ENTRY_GAP_DAYS = 4
TOLERANCES = [1.00, 1.03, 1.05, 1.10, 1.30, 1.50, 2.00, math.inf]

# Frozen live-scoring controls (docs/strategy-charter.md). Passed explicitly
# so replay scoring can never drift from live scoring, and so a stray
# INSIDER_ENABLE_13F_CORROBORATION environment variable can never flip a
# research run: in replay, 13F corroboration stays off (fail closed).
SCORING_ENTRY_TOL = 1.05
SCORING_EVENT_HALF_LIFE_DAYS = 90
SCORING_13F_CORROBORATION = False

# Explicitly declared execution costs for research outcome labels: the
# production evaluator's defaults (analyzer.backtest.evaluate), stated here
# so an upstream default change is a visible diff in this file too.
ENTRY_SLIPPAGE_BPS = 10.0
EXIT_SLIPPAGE_BPS = 10.0

# Calendar-day slack around decision + HORIZON when fetching the label price
# window. The evaluator still requires exact NYSE endpoints inside it; the
# slack only bounds the query, never a substituted fill.
OUTCOME_PRICE_SLACK_DAYS = 7

# Replay frames (docs/strategy-charter.md, "Replay frames"). Never pooled,
# never implicit: every command and every artifact names its frame.
FRAMES = ("disclosure", "ingestion")
OUTCOME_COLUMNS = [
    "ticker", "as_of", "fill_date", "exit_date", "fill_px",
    "entry_eligibility", "entry_eligible", "fwd", "spy", "excess",
    "outcome_status", "exclusion_reason", "frame",
]

MIN_FRAME_EVENTS = 50
MIN_FRAME_TICKERS = 20

DECISIONS = [
    date(2023, 1, 15),
    date(2023, 7, 15),
    date(2024, 1, 15),
    date(2024, 7, 15),
    date(2025, 1, 15),
    date(2025, 7, 15),
    date(2026, 1, 15),
    date(2026, 7, 1),
]

WALK_FORWARD_BLOCK = (
    date(2027, 1, 15),
    date(2027, 7, 15),
    date(2028, 1, 15),
    date(2028, 7, 15),
)
UNSEAL_AFTER = date(2028, 11, 1)


def check_sealed(decision: date, today: date | None = None):
    """Refuse frozen walk-forward block dates before the unseal date."""
    if today is None:
        today = date.today()
    if decision in WALK_FORWARD_BLOCK and today < UNSEAL_AFTER:
        raise SystemExit(
            f"SEALED BLOCK: {decision} is a frozen walk-forward decision date"
            " defined docs/strategy-charter.md; unseal >= 2028-11-01"
        )


def check_capture_date(decision: date, today: date | None = None):
    """Allow only a D+1, label-free capture for preregistered dates."""
    if today is None:
        today = date.today()
    if decision not in WALK_FORWARD_BLOCK:
        raise SystemExit(f"NOT A FROZEN DATE: {decision} is not in the walk-forward block")
    if today != decision + timedelta(days=1):
        raise SystemExit(
            f"CAPTURE WINDOW: {decision} inputs must be captured on "
            f"{decision + timedelta(days=1)} (today is {today})"
        )


def check_unsealed(today: date | None = None):
    """Keep frozen outcomes and analysis sealed until the charter date."""
    if today is None:
        today = date.today()
    if today < UNSEAL_AFTER:
        raise SystemExit(
            f"SEALED ANALYSIS: frozen outcomes remain sealed until "
            f"{UNSEAL_AFTER}; analysis is not permitted before then"
        )


def check_frame(
    events_count: int,
    tickers_count: int,
    decision: date,
    frame: str = "ingestion",
):
    """Refuse degenerate replay frames before any artifact is written."""
    if events_count >= MIN_FRAME_EVENTS and tickers_count >= MIN_FRAME_TICKERS:
        return
    raise SystemExit(
        f"DEGENERATE FRAME: {decision} {frame} frame has {events_count} "
        f"eligible events (minimum {MIN_FRAME_EVENTS}) and {tickers_count} "
        f"candidate tickers (minimum {MIN_FRAME_TICKERS}); refusing to write "
        "artifacts. Use a decision date after the database's first ingestion "
        "watermark."
    )


def sha256_file(path: Path) -> str:
    """Chunked sha256 so hashing the live DB does not load it whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walkforward_dir() -> Path:
    return OUT_DIR / "walkforward"


def _capture_paths(decision: date) -> tuple[Path, Path, Path, bool]:
    """Resolve new immutable bundles, retaining read access to legacy flat files."""
    bundle = _walkforward_dir() / "captures" / str(decision)
    bundled = os.path.lexists(bundle)
    legacy = _legacy_capture_is_complete(decision)
    if bundled and legacy:
        raise SystemExit(
            f"CAPTURE CONFLICT: {decision} has both bundle and legacy inputs"
        )
    if bundled:
        return (
            bundle / "cohorts.csv",
            bundle / "actors.csv",
            bundle / "manifest.json",
            True,
        )
    return (
        OUT_DIR / f"cohorts_{decision}.csv",
        OUT_DIR / f"actors_{decision}.csv",
        OUT_DIR / f"manifest_{decision}.json",
        False,
    )


def _capture_seal(manifest_path: Path, manifest: dict) -> dict[str, str]:
    """Bind labels to the artifact identities returned by capture validation."""
    hashes = manifest.get("artifact_sha256") if isinstance(manifest, dict) else None
    names = ("cohorts.csv", "actors.csv")
    if not isinstance(hashes, dict) or any(
        not isinstance(hashes.get(name), str) for name in names
    ):
        raise SystemExit(f"INVALID CAPTURE: {manifest_path.name} has no verifiable seal")
    return {**{name: hashes[name] for name in names}, "manifest.json": sha256_file(manifest_path)}


def _outcome_binding_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _validate_outcome_panel(
    cohorts: pd.DataFrame,
    outcomes: pd.DataFrame,
    decision: date,
    frame: str,
    *,
    require_exit_date: bool,
) -> None:
    required = {
        "ticker", "as_of", "fill_date", "fill_px", "entry_eligibility",
        "entry_eligible", "fwd", "spy", "excess", "outcome_status",
        "exclusion_reason", "frame",
    }
    if require_exit_date:
        required.add("exit_date")
    if not required.issubset(outcomes.columns):
        raise SystemExit(f"INVALID OUTCOME: {decision} is missing required columns")
    if (
        not outcomes["as_of"].astype(str).eq(str(decision)).all()
        or not outcomes["frame"].astype(str).eq(frame).all()
    ):
        raise SystemExit(f"INVALID OUTCOME: {decision} decision/frame mismatch")
    join_outcomes(cohorts, outcomes, "frozen cohort")

    expected_entry = next_nyse_session(pd.Timestamp(decision)).date()
    for row in outcomes.itertuples(index=False):
        status = str(row.outcome_status)
        if status not in {"available", "excluded"}:
            raise SystemExit(f"INVALID OUTCOME: {decision} has unknown outcome status")
        has_fill_date = pd.notna(row.fill_date)
        has_fill_px = pd.notna(row.fill_px)
        if has_fill_date != has_fill_px:
            raise SystemExit(f"INVALID OUTCOME: {decision} has incomplete fill data")
        fill_date = None
        if has_fill_date:
            parsed_fill = pd.to_datetime(row.fill_date, errors="coerce")
            if pd.isna(parsed_fill) or parsed_fill.date() != expected_entry:
                raise SystemExit(f"INVALID OUTCOME: {decision} fill is not the expected next session")
            fill_date = parsed_fill.date()
            try:
                fill_px = float(row.fill_px)
            except (TypeError, ValueError):
                fill_px = float("nan")
            if not math.isfinite(fill_px) or fill_px <= 0:
                raise SystemExit(f"INVALID OUTCOME: {decision} has invalid fill price")

        has_exit_date = hasattr(row, "exit_date") and pd.notna(row.exit_date)
        if require_exit_date and status == "available" and not has_exit_date:
            raise SystemExit(f"INVALID OUTCOME: {decision} available label has no exit date")
        if has_exit_date:
            if fill_date is None:
                raise SystemExit(f"INVALID OUTCOME: {decision} exit has no fill")
            parsed_exit = pd.to_datetime(row.exit_date, errors="coerce")
            expected_exit = previous_nyse_session(
                pd.Timestamp(fill_date) + pd.Timedelta(days=HORIZON)
            ).date()
            if pd.isna(parsed_exit) or parsed_exit.date() != expected_exit:
                raise SystemExit(f"INVALID OUTCOME: {decision} exit violates the {HORIZON}-day horizon")

        values = []
        for field in ("fwd", "spy", "excess"):
            try:
                value = float(getattr(row, field))
            except (TypeError, ValueError):
                value = float("nan")
            values.append(value)
        if status == "available":
            if fill_date is None or not all(math.isfinite(value) for value in values):
                raise SystemExit(f"INVALID OUTCOME: {decision} available label is incomplete")
            if not math.isclose(values[2], values[0] - values[1], abs_tol=0.02):
                raise SystemExit(f"INVALID OUTCOME: {decision} return and excess disagree")
        elif any(pd.notna(getattr(row, field)) for field in ("fwd", "spy", "excess")):
            raise SystemExit(f"INVALID OUTCOME: {decision} excluded label contains returns")
        elif pd.isna(row.exclusion_reason) or not str(row.exclusion_reason).strip():
            raise SystemExit(f"INVALID OUTCOME: {decision} excluded label has no reason")


def _read_bound_outcome(
    path: Path,
    cohort_path: Path,
    decision: date,
    frame: str,
    capture: dict[str, str],
) -> pd.DataFrame:
    binding_path = _outcome_binding_path(path)
    try:
        binding = json.loads(binding_path.read_text())
        panel, _ = read_panel_csv(path, "outcome", frame=frame)
        cohorts, _ = read_panel_csv(cohort_path, "cohort", frame=frame)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"INVALID OUTCOME: {decision} sidecar is unreadable or unbound") from exc
    if not isinstance(binding, dict):
        raise SystemExit(f"INVALID OUTCOME: {decision} sidecar is not bound to this capture")
    expected = {
        "format": 1,
        "decision": str(decision),
        "frame": frame,
        "producer": "attach-outcomes",
        "label_contract": "production_outcomes",
        "labels_generated_on": binding.get("labels_generated_on"),
        "capture_sha256": capture,
        "outcome_sha256": sha256_file(path),
    }
    try:
        generated_on = date.fromisoformat(str(binding["labels_generated_on"]))
    except (KeyError, TypeError, ValueError):
        generated_on = None
    if (
        binding != expected
        or generated_on is None
        or generated_on < UNSEAL_AFTER
        or generated_on > date.today()
    ):
        raise SystemExit(f"INVALID OUTCOME: {decision} sidecar is not bound to this capture")
    raw_columns = pd.read_csv(path, nrows=0).columns
    if "frame" not in raw_columns:
        raise SystemExit(f"INVALID OUTCOME: {decision} bound sidecar has no frame column")
    _validate_outcome_panel(
        cohorts, panel, decision, frame, require_exit_date=True
    )
    return panel


def _same_frozen_outcomes(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    keys = ["as_of", "ticker", "frame"]
    values = [
        "fill_date", "exit_date", "fill_px", "entry_eligibility", "entry_eligible",
        "fwd", "spy", "excess", "outcome_status", "exclusion_reason",
    ]
    columns = [*keys, *(column for column in values if column in left and column in right)]
    left = left[columns].sort_values(keys).reset_index(drop=True)
    right = right[columns].sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        return False
    numeric = {"fill_px", "entry_eligible", "fwd", "spy", "excess"}
    for column in columns:
        if column in numeric:
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
            if not np.array_equal(a, b, equal_nan=True):
                return False
        elif not left[column].astype("string").fillna("").equals(
            right[column].astype("string").fillna("")
        ):
            return False
    return True


def _capture_outcome_path(
    decision: date,
    bundled: bool,
    *,
    frame: str,
    cohort_path: Path,
    capture: dict[str, str],
) -> Path:
    sidecar = _walkforward_dir() / f"outcomes_{decision}.csv"
    legacy = OUT_DIR / f"outcomes_{decision}.csv"
    if not os.path.lexists(sidecar):
        return sidecar if bundled else legacy
    sidecar_panel = _read_bound_outcome(
        sidecar, cohort_path, decision, frame, capture
    )
    if not os.path.lexists(legacy):
        return sidecar

    legacy_valid = _legacy_outcome_is_complete(legacy, cohort_path, decision, frame)
    if _outcome_binding_path(legacy).exists():
        _read_bound_outcome(legacy, cohort_path, decision, frame, capture)
    if legacy_valid:
        legacy_panel, _ = read_panel_csv(legacy, "outcome", frame=frame)
        if not _same_frozen_outcomes(legacy_panel, sidecar_panel):
            raise SystemExit(
                f"CONFLICTING FROZEN OUTCOMES: {decision} has valid legacy and sidecar labels"
            )
        return legacy
    return sidecar


def _legacy_outcome_is_complete(
    path: Path, cohort_path: Path, decision: date, frame: str
) -> bool:
    """Distinguish a completed flat legacy output from a retryable partial file."""
    try:
        cohorts, _ = read_panel_csv(cohort_path, "cohort", frame=frame)
        outcomes, _ = read_panel_csv(path, "outcome", frame=frame)
        required = {
            "ticker", "as_of", "fill_date", "fill_px", "entry_eligibility",
            "entry_eligible", "fwd", "spy", "excess", "outcome_status",
            "exclusion_reason", "frame",
        }
        if not required.issubset(outcomes.columns):
            return False
        if not outcomes["as_of"].astype(str).eq(str(decision)).all():
            return False
        _validate_outcome_panel(
            cohorts, outcomes, decision, frame, require_exit_date=False
        )
        return True
    except (OSError, ValueError, KeyError, SystemExit):
        return False


def _outcome_write_path(
    decision: date, bundled: bool, cohort_path: Path, frame: str
) -> Path:
    sidecar = _walkforward_dir() / f"outcomes_{decision}.csv"
    legacy = OUT_DIR / f"outcomes_{decision}.csv"
    if os.path.lexists(legacy) and _legacy_outcome_is_complete(
        legacy, cohort_path, decision, frame
    ):
        return legacy
    if bundled or os.path.lexists(sidecar) or os.path.lexists(legacy):
        # Preserve any partial legacy attempt; a retry is bound to the verified
        # capture by its companion receipt and never replaces the old bytes.
        return sidecar
    return legacy


def _validate_capture(
    decision: date,
    frame: str,
    cohort_path: Path,
    actor_path: Path,
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
) -> dict:
    """Validate a capture before labels; skip hashes only to preserve legacy evidence."""
    try:
        manifest = json.loads(manifest_path.read_text())
        cohorts = pd.read_csv(cohort_path)
        actors = pd.read_csv(actor_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"INVALID CAPTURE: {decision} artifacts are unreadable") from exc
    if (
        not isinstance(manifest, dict)
        or not manifest.get("capture_only")
        or manifest.get("decision") != str(decision)
        or manifest.get("frame") != frame
        or not {"ticker", "as_of", "frame"}.issubset(cohorts.columns)
        or not {"ticker", "as_of", "frame"}.issubset(actors.columns)
        or len(cohorts) != manifest.get("rows")
        or not cohorts["as_of"].astype(str).eq(str(decision)).all()
        or not actors["as_of"].astype(str).eq(str(decision)).all()
        or not cohorts["frame"].astype(str).eq(frame).all()
        or not actors["frame"].astype(str).eq(frame).all()
    ):
        raise SystemExit(f"INVALID CAPTURE: {decision} does not match frame {frame}")
    if "actor_rows" in manifest and len(actors) != manifest["actor_rows"]:
        raise SystemExit(f"INVALID CAPTURE: {decision} actor rows do not match manifest")
    if verify_hashes:
        expected_hashes = manifest.get("artifact_sha256")
        names = ("cohorts.csv", "actors.csv")
        if not isinstance(expected_hashes, dict) or any(
            not isinstance(expected_hashes.get(name), str) for name in names
        ):
            raise SystemExit(
                f"UNVERIFIED CAPTURE: {decision} has no complete artifact hash binding"
            )
        for name, path in zip(names, (cohort_path, actor_path)):
            if expected_hashes[name] != sha256_file(path):
                raise SystemExit(f"INVALID CAPTURE: {decision} {name} hash mismatch")
    return manifest


def _legacy_capture_is_complete(decision: date) -> bool:
    """Recognize complete legacy evidence for overwrite guards, not label trust."""
    cohort_path = OUT_DIR / f"cohorts_{decision}.csv"
    actor_path = OUT_DIR / f"actors_{decision}.csv"
    manifest_path = OUT_DIR / f"manifest_{decision}.json"
    if not all(path.is_file() for path in (cohort_path, actor_path, manifest_path)):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            return False
        frame = manifest.get("frame")
        if frame not in FRAMES:
            return False
        _validate_capture(
            decision,
            frame,
            cohort_path,
            actor_path,
            manifest_path,
            verify_hashes=False,
        )
        return True
    except (OSError, ValueError, SystemExit):
        return False


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def next_session_close(db, ticker: str, after: date):
    row = db.conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? AND date > ?"
        " AND close IS NOT NULL ORDER BY date LIMIT 1",
        [ticker, after],
    ).fetchone()
    if not row or (row[0] - after).days > MAX_ENTRY_GAP_DAYS:
        return None
    try:
        price = float(row[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return (row[0], price)


def decision_close(db, ticker: str, decision: date):
    """Last usable close on or before the decision cutoff."""
    row = db.conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? AND date <= ?"
        " AND close IS NOT NULL ORDER BY date DESC LIMIT 1",
        [ticker, decision],
    ).fetchone()
    if not row:
        return None
    try:
        price = float(row[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return (row[0], price)


def outcome_exit_date(entry_date: date) -> date:
    return previous_nyse_session(entry_date + timedelta(days=HORIZON)).date()


def fwd_excess(db, ticker: str, entry_date: date, entry_px: float):
    """Phase-1 skill-episode metric: 90-day close-to-close excess vs SPY.

    Skill episodes keep their documented historical definition (first close
    after disclosure, SPY on or before entry). Phase-2 research outcome
    labels do not flow through here: they come from production_outcomes().
    """
    end = outcome_exit_date(entry_date)
    if end <= entry_date:
        return None, None, None
    out = db.conn.execute(
        "SELECT close FROM prices WHERE ticker = ? AND date = ?",
        [ticker, end],
    ).fetchone()
    spy_in = db.conn.execute(
        "SELECT close FROM prices WHERE ticker = 'SPY' AND date <= ?"
        " ORDER BY date DESC LIMIT 1",
        [entry_date],
    ).fetchone()
    spy_out = db.conn.execute(
        "SELECT close FROM prices WHERE ticker = 'SPY' AND date = ?", [end]
    ).fetchone()
    if not out or not spy_in or not spy_out:
        return None, None, None
    try:
        out_px, spy_in_px, spy_out_px = map(
            float, (out[0], spy_in[0], spy_out[0])
        )
    except (TypeError, ValueError):
        return None, None, None
    if any(
        not math.isfinite(price) or price <= 0
        for price in (entry_px, out_px, spy_in_px, spy_out_px)
    ):
        return None, None, None
    ret = (out_px / entry_px - 1) * 100
    spy = (spy_out_px / spy_in_px - 1) * 100
    return ret, spy, ret - spy


def production_outcomes(db, tickers, decision: date) -> dict[str, dict]:
    """Label decision -> HORIZON outcomes with the production evaluator.

    Research labels are production labels (backtest/evaluate.py): the exact
    expected next NYSE session for entry, the exact expected exit session,
    and the exact same-session SPY quotes at both endpoints, with the
    declared costs above. A missing quote stays unavailable — no nearby
    session is ever substituted. The harness keeps its declared vocabulary:
    no resolved fill -> "fill_unavailable"; a resolved fill whose exit or
    benchmark quote is missing -> "outcome_price_unavailable".
    """
    labels: dict[str, dict] = {}
    symbols = sorted({str(ticker).strip().upper() for ticker in tickers if ticker})
    if not symbols:
        return labels
    window_end = decision + timedelta(days=HORIZON + OUTCOME_PRICE_SLACK_DAYS)
    prices = db.get_prices([*symbols, "SPY"], decision, window_end)
    recommendations = pd.DataFrame(
        {"ticker": symbols, "instrument_type": "stock"}
    )
    evaluated = evaluate_backtest(
        recommendations,
        prices,
        pd.Timestamp(decision),
        HORIZON,
        entry_slippage_bps=ENTRY_SLIPPAGE_BPS,
        exit_slippage_bps=EXIT_SLIPPAGE_BPS,
    )
    for row in evaluated.itertuples(index=False):
        entry_date = row.bt_entry_date
        entry_price = row.bt_entry_price
        resolved_fill = (
            entry_date is not None
            and entry_price is not None
            and not pd.isna(entry_price)
        )
        if row.bt_coverage == "complete":
            fill_date: str | None = str(entry_date)
            fill_px: float | None = float(entry_price)
            fwd = float(row.bt_return_pct)
            spy = float(row.bt_spy_return_pct)
            excess = float(row.bt_alpha_pct)
            exclusion = None
        else:
            if resolved_fill:
                fill_date, fill_px = str(entry_date), float(entry_price)
                exclusion = "outcome_price_unavailable"
            else:
                fill_date, fill_px = None, None
                exclusion = "fill_unavailable"
            fwd = spy = excess = None
        labels[str(row.ticker).strip().upper()] = {
            "fill_date": fill_date,
            "exit_date": (
                str(row.bt_exit_date)
                if row.bt_coverage == "complete" and row.bt_exit_date is not None
                else None
            ),
            "fill_px": fill_px,
            "fwd": fwd,
            "spy": spy,
            "excess": excess,
            "exclusion": exclusion,
        }
    return labels


def outcome_label(labels: dict[str, dict], ticker) -> dict:
    """Fetch one guaranteed label; absence would be a harness bug, not data."""
    return labels.get(str(ticker).strip().upper()) or {
        "fill_date": None,
        "exit_date": None,
        "fill_px": None,
        "fwd": None,
        "spy": None,
        "excess": None,
        "exclusion": "fill_unavailable",
    }


def breadth_by_actor(
    db, decision: date, frame_scope: date | None = None
) -> dict[str, int]:
    scope_date = frame_scope if frame_scope is not None else decision
    with db.canonical_as_of(scope_date):
        rows = db.conn.execute(
            "SELECT DISTINCT source, member, ticker FROM canonical_transactions"
            " WHERE disclosure_date <= ? AND member IS NOT NULL AND ticker IS NOT NULL",
            [decision],
        ).fetchall()
    kinds = {
        "house_pdf": "congress",
        "gemini_ocr": "congress",
        "senate_efd": "congress",
        "form4": "officer",
        "13f": "manager",
    }
    symbols: dict[str, set[str]] = {}
    for source, member, ticker in rows:
        kind = kinds.get(source)
        if not kind or not isinstance(member, str) or not isinstance(ticker, str):
            continue
        normalized = canonical_member_key(member) if kind == "congress" else member
        actor = actor_id(kind, normalized)
        symbols.setdefault(actor, set()).add(ticker.upper())
    return {actor: len(tickers) for actor, tickers in symbols.items()}


@contextmanager
def cached_position_inputs():
    load_trades = positions.load_member_trades
    load_prices = positions.load_price_history
    trades_cache = {}
    prices_cache = {}

    def cached_trades(db, member: str, *, include_unpromoted: bool = False):
        key = (id(db), getattr(db, "_canonical_as_of", None), member, include_unpromoted)
        if key not in trades_cache:
            trades_cache[key] = load_trades(
                db, member, include_unpromoted=include_unpromoted
            )
        return trades_cache[key]

    def cached_prices(db, ticker: str):
        key = (id(db), ticker)
        if key not in prices_cache:
            prices_cache[key] = load_prices(db, ticker)
        return prices_cache[key]

    positions.load_member_trades = cached_trades
    positions.load_price_history = cached_prices
    try:
        yield
    finally:
        positions.load_member_trades = load_trades
        positions.load_price_history = load_prices


def skill_episodes(db, decision, frame_scope: date | None = None):
    ev = eligible_events(db, decision, frame_scope=frame_scope)
    ev = ev[pd.to_datetime(ev["disclosure_date"]) < pd.Timestamp("2023-01-01")]
    ev = ev.drop_duplicates(["actor_id", "ticker", "disclosure_date"])
    ev = ev.loc[ev["actor_id"].astype(str).str.startswith("congress:")]
    episodes = []
    for r in ev.itertuples():
        disclosure = pd.to_datetime(r.disclosure_date).date()
        entry = next_session_close(db, str(r.ticker), disclosure)
        if not entry:
            continue
        exit_date = outcome_exit_date(entry[0])
        _, _, excess = fwd_excess(db, str(r.ticker), entry[0], entry[1])
        if excess is not None:
            episodes.append({"actor": str(r.actor_id), "ticker": str(r.ticker),
                             "disclosure_date": disclosure, "exit_date": exit_date,
                             "excess": excess})
    return episodes


def cmd_skill(*, frame: str):
    if frame not in FRAMES:
        raise SystemExit(
            f"unknown frame {frame!r}; expected one of {', '.join(FRAMES)}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(DB_PATH, read_only=True)
    try:
        by_decision = {}
        for decision in DECISIONS:
            scoped_as_of = date.today() if frame == "disclosure" else decision
            episodes = skill_episodes(db, decision, frame_scope=scoped_as_of)
            matured = [item for item in episodes if item["exit_date"] < decision]
            stats_by_actor = {}
            for item in matured:
                stats_by_actor.setdefault(item["actor"], []).append(item["excess"])
            stats_rows = [
                {
                    "actor": actor,
                    "n": len(rets),
                    "mean_excess": sum(rets) / len(rets),
                }
                for actor, rets in sorted(stats_by_actor.items())
                if len(rets) >= 5
            ]
            labels = {
                item["actor"]
                for item in stats_rows
                if item["mean_excess"] > 3.0
            }
            breadth = breadth_by_actor(db, decision, frame_scope=scoped_as_of)
            by_decision[str(decision)] = {
                "labels": sorted(labels),
                "stats": stats_rows,
                "breadth": breadth,
            }
        with (OUT_DIR / "skill.json").open("w") as output:
            json.dump({"frame": frame, "by_decision": by_decision}, output)
        for decision, values in by_decision.items():
            print(decision, "skilled members:", len(values["labels"]), flush=True)
    finally:
        db.close()


def cmd_run(
    decision: date,
    db=None,
    *,
    frame: str,
    capture_only: bool = False,
    capture_output_dir: Path | None = None,
):
    if capture_only and capture_output_dir is None:
        raise ValueError("capture-only runs require a private staging directory")
    if capture_only:
        check_capture_date(decision)
    else:
        check_sealed(decision)
        if decision in WALK_FORWARD_BLOCK:
            raise SystemExit(
                f"FROZEN INPUTS: {decision} must not be rescored; capture at D+1 "
                "and attach outcomes only with attach-outcomes after unseal"
            )
    if frame not in FRAMES:
        raise SystemExit(
            f"unknown frame {frame!r}; expected one of {', '.join(FRAMES)}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    owns_db = db is None
    if owns_db:
        db = Database(DB_PATH, read_only=True)
    scoped_as_of = date.today() if frame == "disclosure" else decision
    with closing(db) if owns_db else nullcontext(), db.canonical_as_of(scoped_as_of):
        if capture_only:
            skilled, hyper = set(), set()
        else:
            skill = json.loads((OUT_DIR / "skill.json").read_text())
            if skill.get("frame") != frame:
                raise SystemExit(
                    f"skill.json frame {skill.get('frame')!r} does not match "
                    f"requested frame {frame!r}; regenerate with skill --frame {frame}"
                )
            point_in_time = skill["by_decision"][str(decision)]
            skilled = set(point_in_time["labels"])
            hyper = {
                actor for actor, count in point_in_time["breadth"].items()
                if count > 150
            }
        events_all = eligible_events(db, decision, frame_scope=scoped_as_of)
        tickers = sorted(events_all["ticker"].astype("string").str.upper().unique())
        keep = []
        for ticker in tickers:
            sub = events_all.loc[
                events_all["ticker"].astype("string").str.upper() == ticker
            ]
            if sub["actor_id"].nunique() >= 2:
                keep.append(ticker)
        if not capture_only:
            check_frame(len(events_all), len(keep), decision, frame)
        print(decision, "tickers:", len(keep), flush=True)
        holdings_cache: dict = {}
        # Production weight path caches its point-in-time skill table per
        # (cutoff, actor set); the cache lives for one replay decision only.
        skill_cache: dict = {}
        # One production evaluation for the whole frame: exact next-session
        # entries, exact exits, exact same-session SPY quotes. Captures stay
        # label-free (outcomes attach later through the same evaluator).
        outcome_labels = (
            production_outcomes(db, keep, decision) if not capture_only else {}
        )

        def holdings_for(member: str):
            if member not in holdings_cache:
                holdings_cache[member] = holdings_candidates(db, member, decision)
            return holdings_cache[member]

        watchlist = watchlist_actor_ids()
        rows, actor_rows, outcomes = [], [], []
        for i, symbol in enumerate(keep):
            events = events_all.loc[
                events_all["ticker"].astype("string").str.upper() == symbol
            ].copy()
            kinds: dict[str, str] = {}
            for actor_id in events["actor_id"].dropna():
                text = str(actor_id)
                if ":" in text:
                    kind, _, name = text.partition(":")
                    if name:
                        kinds.setdefault(name, kind)
            frames, congress_frames = [], []
            for member in dedupe_member_names(sorted(kinds)):
                lots = holdings_for(member)
                if lots.empty:
                    continue
                lots = lots.loc[
                    lots["ticker"].astype("string").str.upper() == symbol
                ].copy()
                if not lots.empty:
                    frames.append(lots)
                    congress_frames.append(lots.loc[lots["kind"] == "congress"].copy())
            positions_frame = (
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            )
            events = coalesce_actor_variants(events, positions_frame)
            events = gate_closed_events(events, positions_frame)
            candidates = pd.concat(
                [part for part in congress_frames + [events] if not part.empty],
                ignore_index=True,
            )
            quote = decision_close(db, symbol, decision)
            quote_date, quote_px = quote if quote else (None, None)
            if candidates.empty:
                ranked = pd.DataFrame()
                blocked = pd.DataFrame(columns=["ticker", "actor_id", "reason"])
                score_reason = "no eligible candidates"
            else:
                weights = weights_for_candidates(
                    db, candidates, decision, skill_table_cache=skill_cache
                )
                ranked, blocked = score_setups(
                    candidates,
                    weights,
                    {symbol: (quote_px, quote_date) if quote else None},
                    entry_tol=SCORING_ENTRY_TOL,
                    event_half_life_days=SCORING_EVENT_HALF_LIFE_DAYS,
                    watchlist=watchlist,
                    enable_13f_corroboration=SCORING_13F_CORROBORATION,
                )
                score_reason = None if not ranked.empty else "; ".join(
                    sorted(set(blocked.loc[blocked["ticker"] == symbol, "reason"].astype(str)))
                ) or "not ranked"

            refs = (
                pd.to_numeric(candidates["entry_ref"], errors="coerce")
                if not candidates.empty else pd.Series(dtype=float)
            )
            valid_refs = refs.notna() & np.isfinite(refs) & (refs > 0)
            init = candidates.loc[valid_refs].copy() if not candidates.empty else candidates
            actor_ids = set(init["actor_id"].astype(str)) if not init.empty else set()
            kinds_hit = set(init["kind"].astype(str)) if not init.empty else set()
            has_13f = bool(not candidates.empty and (candidates["source"] == "13f").any())
            lags = (
                pd.to_datetime(decision) - pd.to_datetime(init["disclosure_date"])
            ).dt.days.dropna() if not init.empty else pd.Series(dtype=float)
            fresh_frame = init.loc[lags.between(0, 30)] if not lags.empty else init.iloc[0:0]
            fresh_actors = set(fresh_frame["actor_id"].astype(str))
            max_valid_ref = float(refs.loc[valid_refs].max()) if valid_refs.any() else None
            for actor, actor_candidates in candidates.groupby("actor_id", sort=True):
                actor_refs = pd.to_numeric(actor_candidates["entry_ref"], errors="coerce")
                actor_valid = actor_refs.notna() & np.isfinite(actor_refs) & (actor_refs > 0)
                actor_reference = (
                    actor_candidates.loc[actor_refs.loc[actor_valid].idxmax()]
                    if actor_valid.any() else actor_candidates.iloc[0]
                )
                actor_lags = (
                    pd.to_datetime(decision)
                    - pd.to_datetime(actor_candidates["disclosure_date"])
                ).dt.days.dropna()
                blocked_for_actor = (
                    blocked.loc[
                        (blocked["ticker"] == symbol)
                        & (blocked["actor_id"].astype(str) == str(actor)),
                        "reason",
                    ]
                    if not blocked.empty else pd.Series(dtype=object)
                )
                actor_row = {
                    "as_of": str(decision),
                    "ticker": symbol,
                    "actor_id": str(actor),
                    "kind": str(actor_reference["kind"]),
                    "entry_ref": (
                        float(actor_reference["entry_ref"]) if actor_valid.any() else None
                    ),
                    "lag_days": int(actor_lags.min()) if not actor_lags.empty else None,
                    "entry_reference_status": "valid" if actor_valid.any() else "missing",
                    "score_block_reason": "; ".join(
                        sorted(set(blocked_for_actor.dropna().astype(str)))
                    ) or None,
                }
                if not capture_only:
                    actor_row.update(skilled=str(actor) in skilled, hyper=str(actor) in hyper)
                actor_rows.append(actor_row)

            row = {
                "ticker": symbol,
                "as_of": str(decision),
                "decision_quote_date": str(quote_date) if quote_date else None,
                "decision_quote_px": quote_px,
                "ranked": bool(not ranked.empty),
                "score": float(ranked["score"].max()) if not ranked.empty else None,
                "score_exclusion_reason": score_reason,
                "n_actors": len(actor_ids),
                "has_officer": "officer" in kinds_hit,
                "has_13f": has_13f,
                "fresh": bool(fresh_actors),
                "min_lag": int(lags.min()) if not lags.empty else None,
                "max_valid_ref": max_valid_ref,
            }
            if not capture_only:
                row.update(
                    fresh_skilled=bool(fresh_actors & skilled),
                    skilled_present=bool(actor_ids & skilled),
                    hyper_present=bool(actor_ids & hyper),
                    hyper_only=bool(actor_ids and actor_ids <= hyper),
                )
                label = outcome_label(outcome_labels, symbol)
                fill_date = label["fill_date"]
                fill_px = label["fill_px"]
                ret = label["fwd"]
                spy = label["spy"]
                excess = label["excess"]
                exclusion = label["exclusion"]
                eligible_fill = (
                    bool(fill_px <= SCORING_ENTRY_TOL * refs.loc[valid_refs].max())
                    if fill_px is not None and valid_refs.any() else None
                )
                outcomes.append(
                    {
                        "ticker": symbol,
                        "as_of": str(decision),
                        "fill_date": str(fill_date) if fill_date else None,
                        "exit_date": label["exit_date"],
                        "fill_px": fill_px,
                        "entry_eligibility": (
                            "eligible" if eligible_fill else "ineligible"
                        ) if eligible_fill is not None else "excluded: no fill or valid reference",
                        "entry_eligible": eligible_fill,
                        "fwd": ret,
                        "spy": spy,
                        "excess": excess,
                        "outcome_status": "available" if excess is not None else "excluded",
                        "exclusion_reason": exclusion or (
                            "no_valid_entry_reference" if eligible_fill is None else None
                        ),
                        "frame": frame,
                    }
                )
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(keep)}", flush=True)

        cohort_columns = [
            "ticker", "as_of", "decision_quote_date", "decision_quote_px",
            "ranked", "score", "score_exclusion_reason", "n_actors",
            "has_officer", "has_13f", "fresh", "min_lag", "max_valid_ref",
        ]
        if not capture_only:
            cohort_columns += [
                "fresh_skilled", "skilled_present", "hyper_present", "hyper_only",
            ]
        cohort_frame = pd.DataFrame(rows, columns=cohort_columns)
        cohort_frame["frame"] = frame
        cohort_path = (
            capture_output_dir / "cohorts.csv"
            if capture_only
            else OUT_DIR / f"cohorts_{decision}.csv"
        )
        cohort_frame.to_csv(cohort_path, index=False)
        actor_columns = [
            "as_of", "ticker", "actor_id", "kind", "entry_ref", "lag_days",
            "entry_reference_status", "score_block_reason",
        ]
        if not capture_only:
            actor_columns += ["skilled", "hyper"]
        actor_frame = pd.DataFrame(actor_rows, columns=actor_columns)
        actor_frame["frame"] = frame
        actor_path = (
            capture_output_dir / "actors.csv"
            if capture_only
            else OUT_DIR / f"actors_{decision}.csv"
        )
        actor_frame.to_csv(actor_path, index=False)
        if not capture_only:
            pd.DataFrame(
                outcomes,
                columns=OUTCOME_COLUMNS,
            ).to_csv(OUT_DIR / f"outcomes_{decision}.csv", index=False)
        manifest = {
            "decision": str(decision),
            "frame": frame,
            "scoped_as_of": scoped_as_of.isoformat(),
            "events": len(events_all),
            "keep_tickers": len(keep),
            "rows": len(rows),
            "db_path": str(db.db_path),
            "db_sha256": sha256_file(db.db_path),
            "git_sha": git_sha(),
        }
        if capture_only:
            manifest["capture_only"] = True
            manifest["capture_format"] = 1
            manifest["actor_rows"] = len(actor_frame)
            manifest["artifact_sha256"] = {
                "cohorts.csv": sha256_file(cohort_path),
                "actors.csv": sha256_file(actor_path),
            }
        manifest_path = (
            capture_output_dir / "manifest.json"
            if capture_only
            else OUT_DIR / f"manifest_{decision}.json"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        if capture_only:
            for path in (cohort_path, actor_path, manifest_path):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            _fsync_directory(capture_output_dir)
        print("wrote", len(rows), "rows", flush=True)


def _write_json_exclusive(value: dict, path: Path) -> None:
    """Create a receipt once, accepting only an identical interrupted retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise SystemExit(f"OUTCOME BINDING: {path.name} is unreadable") from exc
        if existing != value:
            raise SystemExit(f"OUTCOME BINDING: {path.name} binds different bytes or capture")
        return
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError):
            raise SystemExit(f"OUTCOME BINDING: {path.name} is unreadable") from exc
        if existing != value:
            raise SystemExit(f"OUTCOME BINDING: {path.name} binds different bytes or capture") from exc
    _fsync_directory(path.parent)
    temporary.unlink()
    _fsync_directory(path.parent)


def _write_csv_exclusive(
    frame: pd.DataFrame, path: Path, binding: dict
) -> None:
    """Publish a complete CSV and a receipt bound to its capture and bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    receipt = {
        **binding,
        "outcome_sha256": sha256_file(temporary),
    }
    _write_json_exclusive(receipt, _outcome_binding_path(path))
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise SystemExit(f"OUTCOME OVERWRITE: {path.name} already exists") from exc
    _fsync_directory(path.parent)
    temporary.unlink()
    _fsync_directory(path.parent)


def cmd_capture(decision: date, db=None, *, frame: str):
    """Stage and atomically publish one immutable, label-free capture bundle."""
    check_capture_date(decision)
    if frame not in FRAMES:
        raise SystemExit(f"unknown frame {frame!r}; expected one of {', '.join(FRAMES)}")
    capture_parent = _walkforward_dir() / "captures"
    target = capture_parent / str(decision)
    if os.path.lexists(target) or _legacy_capture_is_complete(decision):
        raise SystemExit(
            f"CAPTURE OVERWRITE: {decision} capture artifacts already exist"
        )
    capture_parent.mkdir(parents=True, exist_ok=True)
    # Failed attempts remain under private, unique staging names and never
    # occupy the canonical date path, so they cannot block a valid retry.
    staging = Path(tempfile.mkdtemp(prefix=f".{decision}.", dir=capture_parent))
    cmd_run(
        decision,
        db,
        frame=frame,
        capture_only=True,
        capture_output_dir=staging,
    )
    _validate_capture(
        decision,
        frame,
        staging / "cohorts.csv",
        staging / "actors.csv",
        staging / "manifest.json",
    )
    if os.path.lexists(target) or _legacy_capture_is_complete(decision):
        raise SystemExit(f"CAPTURE OVERWRITE: {decision} capture appeared during staging")
    try:
        os.rename(staging, target)
    except OSError as exc:
        if os.path.lexists(target):
            raise SystemExit(f"CAPTURE OVERWRITE: {decision} capture appeared during staging") from exc
        raise
    _fsync_directory(capture_parent)


def cmd_attach_outcomes(decision: date, db=None, *, frame: str):
    """Attach labels after unseal without reloading or rescoring decisions."""
    check_unsealed()
    if decision not in WALK_FORWARD_BLOCK:
        raise SystemExit(f"NOT A FROZEN DATE: {decision} is not in the walk-forward block")
    if frame not in FRAMES:
        raise SystemExit(f"unknown frame {frame!r}; expected one of {', '.join(FRAMES)}")
    cohort_path, actor_path, manifest_path, bundled = _capture_paths(decision)
    if not cohort_path.exists() or not manifest_path.exists():
        raise SystemExit(f"MISSING CAPTURE: no frozen input snapshot for {decision}")
    capture_manifest = _validate_capture(
        decision, frame, cohort_path, actor_path, manifest_path
    )
    capture_seal = _capture_seal(manifest_path, capture_manifest)
    outcome_path = _outcome_write_path(decision, bundled, cohort_path, frame)
    if os.path.lexists(outcome_path):
        raise SystemExit(f"OUTCOME OVERWRITE: {outcome_path.name} already exists")
    owns_db = db is None
    if owns_db:
        db = Database(DB_PATH, read_only=True)
    try:
        outcomes = []
        captured = pd.read_csv(cohort_path)
        tickers_by_decision: dict[date, set[str]] = {}
        for row in captured.itertuples(index=False):
            tickers_by_decision.setdefault(
                date.fromisoformat(str(row.as_of)), set()
            ).add(str(row.ticker))
        labels_by_decision = {
            decision_date: production_outcomes(
                db, sorted(tickers), decision_date
            )
            for decision_date, tickers in sorted(tickers_by_decision.items())
        }
        for row in captured.itertuples(index=False):
            symbol = str(row.ticker)
            decision_date = date.fromisoformat(str(row.as_of))
            label = outcome_label(labels_by_decision[decision_date], symbol)
            fill_date = label["fill_date"]
            fill_px = label["fill_px"]
            ret = label["fwd"]
            spy = label["spy"]
            excess = label["excess"]
            exclusion = label["exclusion"]
            ref = pd.to_numeric(pd.Series([row.max_valid_ref]), errors="coerce").iloc[0]
            eligible_fill = (
                bool(fill_px <= SCORING_ENTRY_TOL * ref)
                if fill_px is not None and pd.notna(ref) and ref > 0 else None
            )
            outcomes.append(
                {
                    "ticker": symbol,
                    "as_of": str(decision_date),
                    "fill_date": str(fill_date) if fill_date else None,
                    "exit_date": label["exit_date"],
                    "fill_px": fill_px,
                    "entry_eligibility": (
                        "eligible" if eligible_fill else "ineligible"
                    ) if eligible_fill is not None else "excluded: no fill or valid reference",
                    "entry_eligible": eligible_fill,
                    "fwd": ret,
                    "spy": spy,
                    "excess": excess,
                    "outcome_status": "available" if excess is not None else "excluded",
                    "exclusion_reason": exclusion or (
                        "no_valid_entry_reference" if eligible_fill is None else None
                    ),
                    "frame": frame,
                }
            )
        binding = {
            "format": 1,
            "decision": str(decision),
            "frame": frame,
            "producer": "attach-outcomes",
            "label_contract": "production_outcomes",
            "labels_generated_on": str(date.today()),
            "capture_sha256": capture_seal,
        }
        current_manifest = _validate_capture(
            decision, frame, cohort_path, actor_path, manifest_path
        )
        if _capture_seal(manifest_path, current_manifest) != capture_seal:
            raise SystemExit(f"INVALID CAPTURE: {decision} changed while outcomes were attached")
        _write_csv_exclusive(
            pd.DataFrame(outcomes, columns=OUTCOME_COLUMNS), outcome_path, binding
        )
    finally:
        if owns_db:
            db.close()


def cmd_analyze_capture(decision: date, *, frame: str) -> pd.DataFrame:
    """Join a frozen input snapshot to outcomes only after the formal unseal."""
    check_unsealed()
    if frame not in FRAMES:
        raise SystemExit(f"unknown frame {frame!r}; expected one of {', '.join(FRAMES)}")
    cohort_path, actor_path, manifest_path, bundled = _capture_paths(decision)
    if not all(path.exists() for path in (cohort_path, actor_path, manifest_path)):
        raise SystemExit(f"MISSING FROZEN DATA: capture and attach outcomes for {decision}")
    capture_manifest = _validate_capture(
        decision, frame, cohort_path, actor_path, manifest_path
    )
    capture_seal = _capture_seal(manifest_path, capture_manifest)
    outcome_path = _capture_outcome_path(
        decision,
        bundled,
        frame=frame,
        cohort_path=cohort_path,
        capture=capture_seal,
    )
    if not outcome_path.exists():
        raise SystemExit(f"MISSING FROZEN DATA: capture and attach outcomes for {decision}")
    cohorts, _ = read_panel_csv(cohort_path, "cohort", frame=frame)
    outcomes, _ = read_panel_csv(outcome_path, "outcome", frame=frame)
    bound = outcome_path == _walkforward_dir() / f"outcomes_{decision}.csv"
    has_binding = _outcome_binding_path(outcome_path).exists()
    if bound or has_binding:
        _read_bound_outcome(
            outcome_path, cohort_path, decision, frame, capture_seal
        )
    _validate_outcome_panel(
        cohorts,
        outcomes,
        decision,
        frame,
        require_exit_date=bound or has_binding,
    )
    current_manifest = _validate_capture(
        decision, frame, cohort_path, actor_path, manifest_path
    )
    if _capture_seal(manifest_path, current_manifest) != capture_seal:
        raise SystemExit(f"INVALID CAPTURE: {decision} changed during outcome analysis")
    result = join_outcomes(cohorts, outcomes, "frozen cohort")
    print(result.to_string(index=False), flush=True)
    return result


def clustered_t(frame: pd.DataFrame, group: str) -> float | None:
    values = frame[["excess", "as_of", group]].dropna()
    if values[group].nunique() != 2 or values["as_of"].nunique() < 2:
        return None
    y = values["excess"].to_numpy(dtype=float)
    treatment = values[group].astype(bool).to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(values)), treatment))
    n, k = design.shape
    if n <= k:
        return None
    bread = np.linalg.inv(design.T @ design)
    residuals = y - design @ (bread @ design.T @ y)
    meat = np.zeros((k, k))
    for _, indices in values.groupby("as_of").groups.items():
        positions = values.index.get_indexer(indices)
        cluster_score = design[positions].T @ residuals[positions]
        meat += np.outer(cluster_score, cluster_score)
    clusters = values["as_of"].nunique()
    covariance = (clusters / (clusters - 1)) * ((n - 1) / (n - k)) * bread @ meat @ bread
    variance = covariance[1, 1]
    if not math.isfinite(variance) or variance <= 0:
        return None
    standard_error = math.sqrt(variance)
    return float((bread @ design.T @ y)[1] / standard_error)


def compare_groups(
    frame: pd.DataFrame, name: str, group: str, label: str, other: str
) -> dict:
    values = frame[["excess", "as_of", group]].dropna()
    positive = values.loc[values[group].astype(bool), "excess"].astype(float)
    negative = values.loc[~values[group].astype(bool), "excess"].astype(float)
    welch = (
        stats.ttest_ind(positive, negative, equal_var=False).statistic
        if len(positive) and len(negative)
        else float("nan")
    )
    return {
        "comparison": name,
        "group_a": label,
        "n_a": len(positive),
        "mean_a": float(positive.mean()) if len(positive) else float("nan"),
        "group_b": other,
        "n_b": len(negative),
        "mean_b": float(negative.mean()) if len(negative) else float("nan"),
        "welch_t": float(welch),
        "date_clustered_t": clustered_t(values, group),
    }


def read_panel_csv(
    path: Path, kind: str, *, frame: str | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Read a panel under an explicit frame, including asserted legacy files."""
    panel = pd.read_csv(path)
    if "frame" not in panel.columns:
        if frame not in FRAMES:
            raise SystemExit(
                f"{kind} CSV {path.name} has no frame column; provide an explicit "
                "--frame assertion or regenerate with --frame"
            )
        # The required CLI --frame is the operator's explicit provenance
        # assertion for pre-frame artifacts; never infer or pool it silently.
        panel["frame"] = frame
        return panel, [frame]
    values = sorted(set(panel["frame"].dropna().astype(str)))
    if frame is not None and values and values != [frame]:
        raise SystemExit(
            f"{kind} CSV {path.name} frame {values} does not match requested {frame}"
        )
    return panel, values or ([frame] if frame in FRAMES else [])


def refuse_mixed_frames(frame_values: dict[str, list[str]]) -> None:
    """A pooled estimate that mixes frames is invalid regardless of t-stat."""
    if len(frame_values) > 1:
        detail = ", ".join(
            f"{value} <- {', '.join(names)}"
            for value, names in sorted(frame_values.items())
        )
        raise SystemExit(f"mixed frames in panel: {detail}")


def join_outcomes(inputs: pd.DataFrame, outcomes: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Require exact input/outcome coverage; unavailable labels stay as rows."""
    merged = inputs.merge(
        outcomes,
        on=["as_of", "ticker", "frame"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_outcome"),
    )
    unmatched = merged.loc[merged["_merge"] != "both", ["as_of", "ticker", "_merge"]]
    if not unmatched.empty:
        raise SystemExit(
            f"{kind} inputs/outcomes do not match; "
            f"{len(unmatched)} rows are missing from one side"
        )
    return merged.drop(columns="_merge")


def cmd_analyze(*, frame: str):
    if frame not in FRAMES:
        raise SystemExit(f"unknown frame {frame!r}; expected one of {', '.join(FRAMES)}")
    for decision in WALK_FORWARD_BLOCK:
        _, _, path, _ = _capture_paths(decision)
        if path.exists() and json.loads(path.read_text()).get("capture_only"):
            check_unsealed()
    files = [OUT_DIR / f"cohorts_{decision}.csv" for decision in DECISIONS]
    outcome_files = [OUT_DIR / f"outcomes_{decision}.csv" for decision in DECISIONS]
    missing_outcomes = [path.name for path in outcome_files if not path.exists()]
    if missing_outcomes:
        raise SystemExit("missing outcome CSVs: " + ", ".join(missing_outcomes))
    frames, outcomes = [], []
    frame_values: dict[str, list[str]] = {}
    for path in (path for path in files if path.exists()):
        panel, values = read_panel_csv(path, "cohort", frame=frame)
        for value in values:
            frame_values.setdefault(value, []).append(path.name)
        frames.append(panel)
    for path in outcome_files:
        panel, values = read_panel_csv(path, "outcome", frame=frame)
        for value in values:
            frame_values.setdefault(value, []).append(path.name)
        outcomes.append(panel)
    if len(frames) != len(DECISIONS):
        raise SystemExit("missing cohort CSVs; run all decision dates first")
    refuse_mixed_frames(frame_values)
    observations = pd.concat(frames, ignore_index=True)
    outcome_panel = pd.concat(outcomes, ignore_index=True)
    observations = join_outcomes(observations, outcome_panel, "cohort")
    eligible = observations.loc[
        observations["entry_eligible"].fillna(False).astype(bool)
    ]
    comparisons = [
        compare_groups(
            observations, "skilled vs not", "skilled_present", "skilled", "not skilled"
        ),
        compare_groups(
            observations,
            "fresh+skilled same actor vs rest",
            "fresh_skilled",
            "fresh+skilled",
            "rest",
        ),
        compare_groups(
            eligible,
            "fresh+skilled same actor among eligible",
            "fresh_skilled",
            "fresh+skilled",
            "other eligible",
        ),
        compare_groups(
            observations,
            "eligible vs ineligible",
            "entry_eligible",
            "eligible",
            "ineligible",
        ),
    ]
    summary = pd.DataFrame(comparisons)
    summary["frame"] = frame
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def sweep_comparison(frame: pd.DataFrame, group: str, outcome: str) -> dict:
    values = frame[[outcome, "as_of", group]].dropna().rename(
        columns={outcome: "excess"}
    )
    eligible = values.loc[values[group].astype(bool), "excess"].astype(float)
    ineligible = values.loc[~values[group].astype(bool), "excess"].astype(float)
    welch = (
        stats.ttest_ind(eligible, ineligible, equal_var=False).statistic
        if len(eligible) and len(ineligible)
        else float("nan")
    )
    return {
        "n_elig": len(eligible),
        "n_inelig": len(ineligible),
        "mean_elig": float(eligible.mean()) if len(eligible) else float("nan"),
        "mean_inelig": float(ineligible.mean()) if len(ineligible) else float("nan"),
        "hit_rate_elig": float((eligible > 0).mean()) if len(eligible) else float("nan"),
        "hit_rate_inelig": float((ineligible > 0).mean()) if len(ineligible) else float("nan"),
        "welch_t": float(welch),
        "date_clustered_t": clustered_t(values, group),
    }


def cmd_sweep(*, frame: str):
    if frame not in FRAMES:
        raise SystemExit(f"unknown frame {frame!r}; expected one of {', '.join(FRAMES)}")
    for decision in WALK_FORWARD_BLOCK:
        _, _, path, _ = _capture_paths(decision)
        if path.exists() and json.loads(path.read_text()).get("capture_only"):
            check_unsealed()
    cohort_files = [OUT_DIR / f"cohorts_{decision}.csv" for decision in DECISIONS]
    actor_files = [OUT_DIR / f"actors_{decision}.csv" for decision in DECISIONS]
    outcome_files = [OUT_DIR / f"outcomes_{decision}.csv" for decision in DECISIONS]
    missing = [
        path.name for path in cohort_files + actor_files + outcome_files if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "missing cohort, actor, or outcome CSVs; run all decision dates first: "
            + ", ".join(missing)
        )

    frame_values: dict[str, list[str]] = {}
    cohort_frames, actor_frames, outcome_frames = [], [], []
    for kind, files, sink in (
        ("cohort", cohort_files, cohort_frames),
        ("actor", actor_files, actor_frames),
        ("outcome", outcome_files, outcome_frames),
    ):
        for path in files:
            panel, values = read_panel_csv(path, kind, frame=frame)
            for value in values:
                frame_values.setdefault(value, []).append(path.name)
            sink.append(panel)
    refuse_mixed_frames(frame_values)
    cohorts = pd.concat(cohort_frames, ignore_index=True)
    actors = pd.concat(actor_frames, ignore_index=True)
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    cohorts = join_outcomes(cohorts, outcomes, "cohort")
    if actors.duplicated(["as_of", "ticker", "actor_id"]).any():
        raise SystemExit("actor CSVs contain duplicate decision+ticker+actor rows")
    actor_refs = pd.to_numeric(actors["entry_ref"], errors="coerce")
    actors = actors.loc[
        actor_refs.notna() & np.isfinite(actor_refs) & (actor_refs > 0)
    ].copy()
    actors = actors.merge(
        outcomes[["as_of", "ticker", "excess"]],
        on=["as_of", "ticker"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if actors["_merge"].ne("both").any():
        raise SystemExit("actor rows have no matching cohort outcome")
    actors = actors.drop(columns="_merge")
    actor_means = actors.groupby("actor_id", as_index=False)["excess"].mean()
    actor_means = actor_means.rename(columns={"excess": "actor_mean"})
    actors = actors.merge(actor_means, on="actor_id", validate="many_to_one")
    demeaned = actors.groupby(["as_of", "ticker"], as_index=False).agg(
        actor_excess=("excess", "first"), actor_mean=("actor_mean", "mean"),
        actor_rows=("actor_id", "size"),
    )
    if actors.groupby(["as_of", "ticker"])["excess"].nunique().gt(1).any():
        raise SystemExit("actor rows do not share one ticker-level excess")
    demeaned["y_demeaned"] = demeaned["actor_excess"] - demeaned["actor_mean"]
    cohorts["max_valid_ref"] = pd.to_numeric(cohorts["max_valid_ref"], errors="coerce")
    cohorts = cohorts.merge(
        demeaned[["as_of", "ticker", "actor_excess", "y_demeaned", "actor_rows"]],
        on=["as_of", "ticker"],
        how="left",
        validate="one_to_one",
    )
    valid_ref = np.isfinite(cohorts["max_valid_ref"]) & (cohorts["max_valid_ref"] > 0)
    has_actor_rows = cohorts["actor_rows"].notna()
    if not np.array_equal(valid_ref.to_numpy(), has_actor_rows.to_numpy()):
        raise SystemExit("actor CSVs do not cover every cohort with a valid reference")
    if cohorts.loc[valid_ref & cohorts["excess"].notna(), "y_demeaned"].isna().any():
        raise SystemExit("actor rows do not support within-actor demeaning")
    if not np.allclose(
        cohorts.loc[has_actor_rows, "excess"],
        cohorts.loc[has_actor_rows, "actor_excess"],
        equal_nan=True,
    ):
        raise SystemExit("actor and cohort excess values differ")
    cohorts = cohorts.drop(columns=["actor_excess", "actor_rows"])
    cohorts["fill_px"] = pd.to_numeric(cohorts["fill_px"], errors="coerce")
    ranked = cohorts["ranked"].astype(str).str.lower().eq("true")
    at_105 = (
        valid_ref
        & np.isfinite(cohorts["fill_px"])
        & (cohorts["fill_px"] <= 1.05 * cohorts["max_valid_ref"])
    )
    if not np.array_equal(
        at_105.to_numpy(),
        cohorts["entry_eligible"].astype(str).str.lower().eq("true").to_numpy(),
    ):
        raise SystemExit("1.05 re-bucketing differs from persisted entry eligibility")
    rows = []
    for tolerance in TOLERANCES:
        cohorts["eligible"] = (
            valid_ref
            & np.isfinite(cohorts["fill_px"])
            & (cohorts["fill_px"] <= tolerance * cohorts["max_valid_ref"])
        )
        for sample, selected in (("all", cohorts), ("ranked", cohorts.loc[ranked])):
            for variant, outcome in (
                ("raw", "excess"),
                ("within_actor", "y_demeaned"),
            ):
                rows.append(
                    {
                        "tolerance": "inf" if math.isinf(tolerance) else f"{tolerance:.2f}",
                        "sample": sample,
                        "outcome": variant,
                        **sweep_comparison(selected, "eligible", outcome),
                    }
                )
    summary = pd.DataFrame(rows)
    summary["frame"] = frame
    summary.to_csv(OUT_DIR / "sweep.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def cmd_full(*, frame: str):
    cmd_skill(frame=frame)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(DB_PATH, read_only=True)
    try:
        with cached_position_inputs():
            for decision in DECISIONS:
                cmd_run(decision, db, frame=frame)
    finally:
        db.close()
    cmd_analyze(frame=frame)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronological backtest of the system's load-bearing assumptions."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("skill", "analyze", "sweep"):
        child = subparsers.add_parser(name)
        _add_frame_flag(child)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("decision", type=date.fromisoformat)
    _add_frame_flag(run_parser)
    for name in ("capture", "attach-outcomes", "analyze-capture"):
        child = subparsers.add_parser(name)
        child.add_argument("decision", type=date.fromisoformat)
        _add_frame_flag(child)
    _add_frame_flag(subparsers.add_parser("full"))
    return parser


def _add_frame_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frame",
        choices=FRAMES,
        required=True,
        help="replay frame; no default, the frame is never implicit",
    )


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "skill":
        cmd_skill(frame=args.frame)
    elif args.mode == "run":
        with cached_position_inputs():
            cmd_run(args.decision, frame=args.frame)
    elif args.mode == "analyze":
        cmd_analyze(frame=args.frame)
    elif args.mode == "sweep":
        cmd_sweep(frame=args.frame)
    elif args.mode == "capture":
        cmd_capture(args.decision, frame=args.frame)
    elif args.mode == "attach-outcomes":
        cmd_attach_outcomes(args.decision, frame=args.frame)
    elif args.mode == "analyze-capture":
        cmd_analyze_capture(args.decision, frame=args.frame)
    elif args.mode == "full":
        cmd_full(frame=args.frame)
