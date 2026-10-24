"""Frame flag semantics, self-describing outputs, and downstream guards.

Covers the 2026-10-17 design items on scripts/backtest_assumptions.py:
--frame is required everywhere and never implicit; the disclosure frame
replays today's complete database filtered by disclosure <= decision while
the ingestion frame replays canonical_as_of(decision); every artifact names
its frame; analyze/sweep refuse frame-less or frame-mixed panels.
"""

import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from analyzer.persistence.database import Database
from analyzer.pipeline import eligible_events
from analyzer.positions import holdings_candidates
from tests.test_asof_v2 import load_script, seed
from tests.test_backtest_guard import FRAME_TICKERS, install_skill_fixture, seed_frame

DECISION = date(2026, 6, 20)
FROZEN_TODAY = date(2026, 10, 17)

COHORT_COLUMNS = [
    "ticker", "as_of", "decision_quote_date", "decision_quote_px", "ranked",
    "score", "score_exclusion_reason", "n_actors", "has_officer", "has_13f",
    "fresh", "min_lag", "max_valid_ref", "fresh_skilled", "skilled_present",
    "hyper_present", "hyper_only", "frame",
]
ACTOR_COLUMNS = [
    "as_of", "ticker", "actor_id", "kind", "entry_ref", "lag_days",
    "entry_reference_status", "score_block_reason",
    "skilled", "hyper", "frame",
]
OUTCOME_COLUMNS = [
    "ticker", "as_of", "fill_date", "exit_date", "fill_px",
    "entry_eligibility", "entry_eligible", "fwd", "spy", "excess",
    "outcome_status", "exclusion_reason", "frame",
]
MANIFEST_KEYS = {
    "decision", "frame", "scoped_as_of", "events", "keep_tickers", "rows",
    "db_path", "db_sha256", "git_sha",
}


class FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 17)


class CaptureDate(date):
    @classmethod
    def today(cls):
        return cls(2027, 1, 16)


class UnsealDate(date):
    @classmethod
    def today(cls):
        return cls(2028, 11, 1)


class NextUnsealDate(UnsealDate):
    @classmethod
    def today(cls):
        return cls(2028, 11, 2)


@pytest.fixture()
def frame_script():
    return load_script("backtest_assumptions")


def seed_prices(db, tickers, start="2026-06-01", end="2026-12-31", close=100.0):
    """Flat price history so every ticker fills at `close` with zero drift."""
    sessions = pd.bdate_range(start, end)
    rows = [
        (ticker, session.date(), close)
        for ticker in [*tickers, "SPY"]
        for session in sessions
    ]
    db.conn.executemany("INSERT INTO prices VALUES (?, ?, ?)", rows)


def write_synthetic_capture(out_dir, script, decision, tickers=("AAPL",)):
    out_dir.mkdir(parents=True, exist_ok=True)
    cohorts = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "as_of": str(decision),
                "frame": "ingestion",
                "max_valid_ref": 100.0,
            }
            for ticker in tickers
        ],
        columns=["ticker", "as_of", "frame", "max_valid_ref"],
    )
    actors = pd.DataFrame(columns=["ticker", "as_of", "frame"])
    cohort_path = out_dir / f"cohorts_{decision}.csv"
    actor_path = out_dir / f"actors_{decision}.csv"
    manifest_path = out_dir / f"manifest_{decision}.json"
    cohorts.to_csv(cohort_path, index=False)
    actors.to_csv(actor_path, index=False)
    manifest = {
        "decision": str(decision),
        "frame": "ingestion",
        "scoped_as_of": str(decision),
        "events": len(tickers),
        "keep_tickers": len(tickers),
        "rows": len(tickers),
        "db_path": "synthetic",
        "db_sha256": "0" * 64,
        "git_sha": "synthetic",
        "capture_only": True,
        "capture_format": 1,
        "actor_rows": 0,
        "artifact_sha256": {
            "cohorts.csv": script.sha256_file(cohort_path),
            "actors.csv": script.sha256_file(actor_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest))
    return cohort_path, actor_path, manifest_path


def write_synthetic_bundle(out_dir, script, decision, tickers=("AAPL",)):
    capture_dir = out_dir / "walkforward" / "captures" / str(decision)
    paths = write_synthetic_capture(capture_dir, script, decision, tickers)
    targets = tuple(
        capture_dir / name for name in ("cohorts.csv", "actors.csv", "manifest.json")
    )
    for source, target in zip(paths, targets):
        source.rename(target)
    return targets


def synthetic_outcomes(script, decision, *, fill_date=None, fwd=2.0, excess=1.0):
    fill_date = fill_date or script.next_nyse_session(pd.Timestamp(decision)).date()
    exit_date = script.previous_nyse_session(
        pd.Timestamp(fill_date) + pd.Timedelta(days=script.HORIZON)
    ).date()
    return pd.DataFrame(
        [{
            "ticker": "AAPL",
            "as_of": str(decision),
            "fill_date": str(fill_date),
            "exit_date": str(exit_date),
            "fill_px": 100.0,
            "entry_eligibility": "eligible",
            "entry_eligible": True,
            "fwd": fwd,
            "spy": 1.0,
            "excess": excess,
            "outcome_status": "available",
            "exclusion_reason": None,
            "frame": "ingestion",
        }],
        columns=OUTCOME_COLUMNS,
    )


def synthetic_outcome_binding(script, decision, capture_paths):
    manifest = script._validate_capture(
        decision, "ingestion", *capture_paths
    )
    return {
        "format": 1,
        "decision": str(decision),
        "frame": "ingestion",
        "producer": "attach-outcomes",
        "label_contract": "production_outcomes",
        "labels_generated_on": str(script.date.today()),
        "capture_sha256": script._capture_seal(capture_paths[2], manifest),
    }


def disclosure_visible_tickers(db, decision, today):
    """The direct disclosure-frame oracle: today's DB, disclosure <= decision,
    candidate-rich (>= 2 distinct buyers) — the set cmd_run must reproduce."""
    with db.canonical_as_of(today):
        rows = db.conn.execute(
            "SELECT ticker FROM canonical_transactions"
            " WHERE disclosure_date <= ? AND ticker IS NOT NULL"
            " GROUP BY ticker HAVING COUNT(DISTINCT member) >= 2",
            [decision],
        ).fetchall()
    return {str(row[0]) for row in rows}


# --- frame flag and frame plumbing ---


def test_frames_constant_and_cli_requires_frame_everywhere(frame_script):
    assert frame_script.FRAMES == ("disclosure", "ingestion")
    parser = frame_script.build_parser()
    for argv in (
        ["run", "2023-01-15"],
        ["skill"],
        ["analyze"],
        ["sweep"],
        ["full"],
    ):
        with pytest.raises(SystemExit):  # missing --frame: the frame is never implicit
            parser.parse_args(argv)
    with pytest.raises(SystemExit):  # not a frame choice
        parser.parse_args(["run", "2023-01-15", "--frame", "hindsight"])
    for mode in ("skill", "analyze", "sweep", "full"):
        args = parser.parse_args([mode, "--frame", "disclosure"])
        assert args.frame == "disclosure"
    args = parser.parse_args(["run", "2023-01-15", "--frame", "ingestion"])
    assert args.decision == date(2023, 1, 15)
    assert args.frame == "ingestion"


def test_check_frame_message_names_the_declared_frame(frame_script):
    with pytest.raises(SystemExit, match="DEGENERATE FRAME") as excinfo:
        frame_script.check_frame(49, 19, DECISION, "disclosure")
    assert "disclosure frame" in str(excinfo.value)
    assert frame_script.check_frame(50, 20, DECISION, "disclosure") is None


def test_eligible_events_frame_scope_overrides_inner_scope_only(tmp_path):
    """The disclosure frame widens the canonical scope, never the disclosure
    filter: rows are today's, selected by disclosure <= decision."""
    db = Database(tmp_path / "db.duckdb")
    try:
        seed(db, promoted="2026-08-01", created="2026-08-02",
             accepted="2026-10-01", disclosure="2023-01-03")
        decision = date(2023, 1, 15)
        assert eligible_events(db, decision).empty
        rows = eligible_events(db, decision, frame_scope=date(2026, 10, 17))
        assert rows["ticker"].tolist() == ["JPM"]
        assert (pd.to_datetime(rows["disclosure_date"]) <= pd.Timestamp(decision)).all()
        assert eligible_events(db, decision).empty  # scope restored after the call
    finally:
        db.close()


def test_disclosure_frame_scope_is_preserved_by_nested_holdings_loader(tmp_path):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS[:1])
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(db, FRAME_TICKERS[:1], start="2026-06-01", end="2026-06-30")
        with db.canonical_as_of(FROZEN_TODAY):
            holdings = holdings_candidates(db, "Jane Doe", DECISION)
        assert holdings["ticker"].tolist() == [FRAME_TICKERS[0]]
        assert db._canonical_as_of is None
    finally:
        db.close()


def test_skill_episodes_and_breadth_obey_frame_scope(tmp_path, monkeypatch, frame_script):
    db = Database(tmp_path / "db.duckdb")
    try:
        seed(db, promoted="2026-08-01", created="2026-08-02",
             accepted="2026-10-01", disclosure="2022-12-03")
        monkeypatch.setattr(
            frame_script, "next_session_close", lambda *args: (date(2022, 12, 5), 100)
        )
        monkeypatch.setattr(frame_script, "fwd_excess", lambda *args: (0, 0, 5.0))
        decision = date(2023, 1, 15)
        today = date(2026, 10, 17)
        assert frame_script.skill_episodes(db, decision) == []
        episodes = frame_script.skill_episodes(db, decision, frame_scope=today)
        assert [episode["ticker"] for episode in episodes] == ["JPM"]
        assert frame_script.breadth_by_actor(db, decision) == {}
        breadth = frame_script.breadth_by_actor(db, decision, frame_scope=today)
        assert list(breadth.values()) == [1]
        assert next(iter(breadth)).startswith("congress:")
    finally:
        db.close()


def test_skill_artifact_records_the_explicit_frame(tmp_path, monkeypatch, frame_script):
    db_path = tmp_path / "empty.duckdb"
    db = Database(db_path)
    db.close()
    out_dir = tmp_path / "out"
    monkeypatch.setattr(frame_script, "DB_PATH", db_path)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", FrozenDate)

    frame_script.cmd_skill(frame="disclosure")

    assert json.loads((out_dir / "skill.json").read_text())["frame"] == "disclosure"


# --- disclosure run succeeds where the ingestion frame is empty ---


def test_disclosure_run_succeeds_and_matches_direct_filtered_query(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    install_skill_fixture(out_dir, DECISION, frame="ingestion")
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", FrozenDate)
    db = Database(tmp_path / "frame.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS)  # watermarks: promoted 06-30, accepted 07-02
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(db, FRAME_TICKERS)

        # Ingestion frame at the decision date: every watermark postdates it,
        # the live frame is empty and must be refused as degenerate.
        with pytest.raises(SystemExit, match="DEGENERATE FRAME") as excinfo:
            frame_script.cmd_run(DECISION, db=db, frame="ingestion")
        assert "0 eligible events" in str(excinfo.value)
        assert list(out_dir.glob("*.csv")) == []

        # Disclosure frame: same DB, same decision — today's complete
        # database filtered by disclosure <= decision must produce rows.
        install_skill_fixture(out_dir, DECISION, frame="disclosure")
        frame_script.cmd_run(DECISION, db=db, frame="disclosure")

        cohorts = pd.read_csv(out_dir / f"cohorts_{DECISION}.csv")
        assert cohorts["frame"].eq("disclosure").all()
        expected = disclosure_visible_tickers(db, DECISION, FROZEN_TODAY)
        assert set(cohorts["ticker"].astype(str)) == expected
        assert len(cohorts) == len(FRAME_TICKERS)
        actors = pd.read_csv(out_dir / f"actors_{DECISION}.csv")
        assert actors["frame"].eq("disclosure").all()
    finally:
        db.close()


# --- self-describing outputs ---


def test_frame_is_last_column_and_artifacts_are_self_describing(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    install_skill_fixture(out_dir, DECISION, frame="disclosure")
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", FrozenDate)
    db = Database(tmp_path / "frame.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS)
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(db, FRAME_TICKERS)
        frame_script.cmd_run(DECISION, db=db, frame="disclosure")
    finally:
        db.close()

    assert list(
        pd.read_csv(out_dir / f"cohorts_{DECISION}.csv").columns
    ) == COHORT_COLUMNS
    assert list(
        pd.read_csv(out_dir / f"actors_{DECISION}.csv").columns
    ) == ACTOR_COLUMNS
    assert list(
        pd.read_csv(out_dir / f"outcomes_{DECISION}.csv").columns
    ) == OUTCOME_COLUMNS

    manifest = json.loads((out_dir / f"manifest_{DECISION}.json").read_text())
    assert set(manifest) == MANIFEST_KEYS  # no wall-clock field (I1)
    assert manifest["decision"] == str(DECISION)
    assert manifest["frame"] == "disclosure"
    assert manifest["scoped_as_of"] == str(FROZEN_TODAY)
    assert manifest["events"] == 75
    assert manifest["keep_tickers"] == len(FRAME_TICKERS)
    assert manifest["rows"] == len(FRAME_TICKERS)
    assert manifest["db_path"] == str(tmp_path / "frame.duckdb")
    assert len(manifest["db_sha256"]) == 64
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert manifest["git_sha"] == head


def test_rerun_is_byte_identical(tmp_path, monkeypatch, frame_script):
    out_dir = tmp_path / "out"
    install_skill_fixture(out_dir, DECISION, frame="disclosure")
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", FrozenDate)
    db = Database(tmp_path / "frame.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS)
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(db, FRAME_TICKERS)

        def snapshot():
            frame_script.cmd_run(DECISION, db=db, frame="disclosure")
            return {
                path.name: path.read_bytes()
                for path in sorted(out_dir.glob("*"))
                if path.name != "skill.json"
            }

        first = snapshot()
        second = snapshot()
    finally:
        db.close()
    assert first == second
    assert set(first) == {
        f"cohorts_{DECISION}.csv", f"actors_{DECISION}.csv",
        f"outcomes_{DECISION}.csv",
        f"manifest_{DECISION}.json",
    }


def test_post_decision_price_perturbation_cannot_change_scored_snapshot(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    install_skill_fixture(out_dir, DECISION, frame="disclosure")
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", FrozenDate)
    db = Database(tmp_path / "frame.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS)
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(db, FRAME_TICKERS)
        frame_script.cmd_run(DECISION, db=db, frame="disclosure")
        before = tuple(
            (out_dir / f"{name}_{DECISION}.csv").read_bytes()
            for name in ("cohorts", "actors")
        )

        db.conn.execute(
            "UPDATE prices SET close = close * 7 WHERE date > ?", [DECISION]
        )
        frame_script.cmd_run(DECISION, db=db, frame="disclosure")
        after = tuple(
            (out_dir / f"{name}_{DECISION}.csv").read_bytes()
            for name in ("cohorts", "actors")
        )
    finally:
        db.close()
    assert before == after


def test_missing_outcomes_are_retained_and_unranked_scores_are_null(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    install_skill_fixture(out_dir, DECISION, frame="disclosure")
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", FrozenDate)
    monkeypatch.setattr(
        frame_script,
        "weights_for_candidates",
        lambda db, candidates, as_of, *, skill_table_cache=None: {
            str(actor): 0.01 for actor in candidates["actor_id"]
        },
    )
    db = Database(tmp_path / "frame.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS)
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(db, FRAME_TICKERS, start="2026-06-01", end="2026-06-22")
        frame_script.cmd_run(DECISION, db=db, frame="disclosure")
    finally:
        db.close()

    cohorts = pd.read_csv(out_dir / f"cohorts_{DECISION}.csv")
    outcomes = pd.read_csv(out_dir / f"outcomes_{DECISION}.csv")
    assert len(cohorts) == len(FRAME_TICKERS)
    assert cohorts["score"].isna().all()
    assert cohorts["score_exclusion_reason"].str.contains("evidence floor").all()
    assert len(outcomes) == len(FRAME_TICKERS)
    assert outcomes["excess"].isna().all()
    assert outcomes["exclusion_reason"].eq("outcome_price_unavailable").all()


@pytest.mark.parametrize(
    "missing_ticker, exclusion",
    [("AAPL", "fill_unavailable"), ("SPY", "outcome_price_unavailable")],
)
def test_production_outcomes_do_not_substitute_later_entry_or_benchmark_quotes(
    frame_script, missing_ticker, exclusion
):
    db = Database(":memory:")
    try:
        seed_prices(db, ["AAPL"])
        complete = frame_script.production_outcomes(db, ["AAPL"], DECISION)["AAPL"]
        assert complete["fill_date"] == "2026-06-22"
        assert complete["excess"] is not None
        db.conn.execute(
            "DELETE FROM prices WHERE ticker = ? AND date = ?",
            [missing_ticker, date(2026, 6, 22)],
        )
        # June 23 and all later quotes remain available, but cannot replace
        # the production evaluator's exact June 22 entry endpoint.
        label = frame_script.production_outcomes(db, ["AAPL"], DECISION)["AAPL"]
        assert label["exclusion"] == exclusion
        assert label["fwd"] is label["spy"] is label["excess"] is None
        if missing_ticker == "AAPL":
            assert label["fill_date"] is label["fill_px"] is None
        else:
            assert label["fill_date"] == complete["fill_date"]
            assert label["fill_px"] == complete["fill_px"]
    finally:
        db.close()


def test_frozen_capture_is_label_free_write_once_and_analysis_sealed(
    tmp_path, monkeypatch, frame_script
):
    decision = date(2027, 1, 15)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", CaptureDate)
    db = Database(tmp_path / "capture.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS[:3])
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(
            db, FRAME_TICKERS[:3], start="2026-06-01", end="2027-01-30"
        )

        frame_script.cmd_capture(decision, db=db, frame="ingestion")
        capture_dir = out_dir / "walkforward" / "captures" / str(decision)
        cohort_path = capture_dir / "cohorts.csv"
        capture = pd.read_csv(cohort_path)
        actor_capture = pd.read_csv(capture_dir / "actors.csv")
        assert len(capture) == 3
        assert not {"fwd", "spy", "excess", "fill_px", "skilled_present"} & set(capture)
        assert not {"skilled", "hyper", "excess"} & set(actor_capture)
        assert not (out_dir / "skill.json").exists()
        assert not (out_dir / f"outcomes_{decision}.csv").exists()
        manifest_path = capture_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["capture_only"]
        assert manifest["artifact_sha256"] == {
            "cohorts.csv": frame_script.sha256_file(cohort_path),
            "actors.csv": frame_script.sha256_file(capture_dir / "actors.csv"),
        }

        with pytest.raises(SystemExit, match="SEALED ANALYSIS"):
            frame_script.cmd_analyze_capture(decision, frame="ingestion")
        with pytest.raises(SystemExit, match="SEALED ANALYSIS"):
            frame_script.cmd_analyze(frame="ingestion")
        with pytest.raises(SystemExit, match="SEALED ANALYSIS"):
            frame_script.cmd_sweep(frame="ingestion")
        with pytest.raises(SystemExit, match="SEALED ANALYSIS"):
            frame_script.cmd_attach_outcomes(decision, db=db, frame="ingestion")
        with pytest.raises(SystemExit, match="CAPTURE OVERWRITE"):
            frame_script.cmd_capture(decision, db=db, frame="ingestion")

        frozen_inputs = cohort_path.read_bytes()
        monkeypatch.setattr(frame_script, "date", UnsealDate)
        with pytest.raises(SystemExit, match="FROZEN INPUTS"):
            frame_script.cmd_run(decision, db=db, frame="ingestion")
        frame_script.cmd_attach_outcomes(decision, db=db, frame="ingestion")
        attached = frame_script.cmd_analyze_capture(decision, frame="ingestion")
        assert len(attached) == 3
        assert attached["exclusion_reason"].eq("outcome_price_unavailable").all()
        assert cohort_path.read_bytes() == frozen_inputs
        assert (out_dir / "walkforward" / f"outcomes_{decision}.csv").is_file()
    finally:
        db.close()


def test_capture_analysis_rejects_conflicts_and_impossible_chronology(
    tmp_path, monkeypatch, frame_script
):
    decision = date(2027, 1, 15)
    monkeypatch.setattr(frame_script, "date", UnsealDate)

    conflict_dir = tmp_path / "conflict"
    conflict_dir.mkdir()
    monkeypatch.setattr(frame_script, "OUT_DIR", conflict_dir)
    capture_paths = write_synthetic_capture(conflict_dir, frame_script, decision)
    legacy = synthetic_outcomes(frame_script, decision)
    legacy.to_csv(conflict_dir / f"outcomes_{decision}.csv", index=False)
    valid = frame_script.cmd_analyze_capture(decision, frame="ingestion")
    assert valid[["fwd", "spy", "excess"]].iloc[0].tolist() == [2.0, 1.0, 1.0]

    sidecar_path = conflict_dir / "walkforward" / f"outcomes_{decision}.csv"
    sidecar = synthetic_outcomes(frame_script, decision, fwd=3.0, excess=2.0)
    frame_script._write_csv_exclusive(
        sidecar,
        sidecar_path,
        synthetic_outcome_binding(frame_script, decision, capture_paths),
    )
    with pytest.raises(SystemExit, match="CONFLICTING FROZEN OUTCOMES"):
        frame_script.cmd_analyze_capture(decision, frame="ingestion")

    chronology_dir = tmp_path / "chronology"
    chronology_dir.mkdir()
    monkeypatch.setattr(frame_script, "OUT_DIR", chronology_dir)
    capture_paths = write_synthetic_capture(chronology_dir, frame_script, decision)
    synthetic_outcomes(frame_script, decision).to_csv(
        chronology_dir / f"outcomes_{decision}.csv", index=False
    )
    invalid = synthetic_outcomes(
        frame_script, decision, fill_date=decision, fwd=1000.0, excess=999.0
    )
    sidecar_path = chronology_dir / "walkforward" / f"outcomes_{decision}.csv"
    frame_script._write_csv_exclusive(
        invalid,
        sidecar_path,
        synthetic_outcome_binding(frame_script, decision, capture_paths),
    )
    with pytest.raises(SystemExit, match="fill is not the expected next session"):
        frame_script.cmd_analyze_capture(decision, frame="ingestion")

    exit_dir = tmp_path / "exit-horizon"
    exit_dir.mkdir()
    capture_paths = write_synthetic_capture(exit_dir, frame_script, decision)
    synthetic_outcomes(frame_script, decision).to_csv(
        exit_dir / f"outcomes_{decision}.csv", index=False
    )
    invalid_exit = synthetic_outcomes(frame_script, decision)
    invalid_exit.loc[0, "exit_date"] = str(
        decision + pd.Timedelta(days=frame_script.HORIZON - 1)
    )
    sidecar_path = exit_dir / "walkforward" / f"outcomes_{decision}.csv"
    frame_script._write_csv_exclusive(
        invalid_exit,
        sidecar_path,
        synthetic_outcome_binding(frame_script, decision, capture_paths),
    )
    monkeypatch.setattr(frame_script, "OUT_DIR", exit_dir)
    with pytest.raises(SystemExit, match="exit violates the 90-day horizon"):
        frame_script.cmd_analyze_capture(decision, frame="ingestion")


def test_bundled_outcomes_share_sidecar_authority_and_reject_legacy_conflicts(
    tmp_path, monkeypatch, frame_script
):
    decision = date(2027, 1, 15)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", UnsealDate)
    capture_paths = write_synthetic_bundle(out_dir, frame_script, decision)
    legacy = synthetic_outcomes(frame_script, decision).drop(columns=["exit_date"])
    legacy_path = out_dir / f"outcomes_{decision}.csv"
    legacy.to_csv(legacy_path, index=False)

    sidecar_path = out_dir / "walkforward" / f"outcomes_{decision}.csv"
    capture_path = frame_script._outcome_write_path(
        decision, True, capture_paths[0], "ingestion"
    )
    reader_path = frame_script._capture_outcome_path(
        decision,
        True,
        frame="ingestion",
        cohort_path=capture_paths[0],
        capture=frame_script._capture_seal(
            capture_paths[2],
            frame_script._validate_capture(
                decision, "ingestion", *capture_paths
            ),
        ),
    )
    assert capture_path == reader_path == sidecar_path
    assert not sidecar_path.exists()

    frame_script._write_csv_exclusive(
        synthetic_outcomes(frame_script, decision),
        sidecar_path,
        synthetic_outcome_binding(frame_script, decision, capture_paths),
    )
    assert frame_script._capture_outcome_path(
        decision,
        True,
        frame="ingestion",
        cohort_path=capture_paths[0],
        capture=frame_script._capture_seal(
            capture_paths[2],
            frame_script._validate_capture(decision, "ingestion", *capture_paths),
        ),
    ) == sidecar_path
    assert frame_script.cmd_analyze_capture(
        decision, frame="ingestion"
    )["excess"].tolist() == [1.0]

    conflicting = synthetic_outcomes(
        frame_script, decision, fwd=3.0, excess=2.0
    ).drop(columns=["exit_date"])
    conflicting.to_csv(legacy_path, index=False)
    with pytest.raises(SystemExit, match="CONFLICTING FROZEN OUTCOMES"):
        frame_script.cmd_analyze_capture(decision, frame="ingestion")


def test_legacy_unbound_outcomes_are_explicitly_reported(
    tmp_path, monkeypatch, frame_script, capsys
):
    decision = date(2027, 1, 15)
    out_dir = tmp_path / "legacy"
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", UnsealDate)
    write_synthetic_capture(out_dir, frame_script, decision)
    legacy = synthetic_outcomes(frame_script, decision).drop(columns=["exit_date"])
    legacy.to_csv(out_dir / f"outcomes_{decision}.csv", index=False)

    result = frame_script.cmd_analyze_capture(decision, frame="ingestion")
    assert result.attrs["outcome_contract"] == "legacy-unbound"
    assert "OUTCOME CONTRACT: legacy-unbound" in capsys.readouterr().out
    assert not (out_dir / f"outcomes_{decision}.json").exists()


def test_empty_frozen_capture_keeps_a_valid_bound_empty_outcome(
    tmp_path, monkeypatch, frame_script
):
    decision = date(2027, 1, 15)
    out_dir = tmp_path / "empty"
    write_synthetic_capture(out_dir, frame_script, decision, tickers=())
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", UnsealDate)
    db = Database(":memory:")
    try:
        frame_script.cmd_attach_outcomes(decision, db=db, frame="ingestion")
        result = frame_script.cmd_analyze_capture(decision, frame="ingestion")
        assert result.empty
        assert pd.read_csv(out_dir / f"outcomes_{decision}.csv").columns.tolist() == OUTCOME_COLUMNS
    finally:
        db.close()


def test_capture_failure_keeps_partial_attempt_private_and_allows_retry(
    tmp_path, monkeypatch, frame_script
):
    decision = date(2027, 1, 15)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", CaptureDate)
    db = Database(tmp_path / "capture-fault.duckdb")
    legacy_partial = out_dir / f"cohorts_{decision}.csv"
    legacy_partial.parent.mkdir(parents=True)
    legacy_partial.write_bytes(b"operator-owned partial bytes\n")
    original_to_csv = pd.DataFrame.to_csv
    try:
        seed_frame(db, tickers=FRAME_TICKERS[:3])
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(db, FRAME_TICKERS[:3], start="2026-06-01", end="2027-01-30")

        def fail_after_partial(self, path_or_buf=None, *args, **kwargs):
            if Path(path_or_buf).name == "cohorts.csv":
                Path(path_or_buf).write_bytes(b"partial staged capture\n")
                raise OSError("injected capture serialization failure")
            return original_to_csv(self, path_or_buf, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial)
        with pytest.raises(OSError, match="injected capture"):
            frame_script.cmd_capture(decision, db=db, frame="ingestion")

        capture_dir = out_dir / "walkforward" / "captures" / str(decision)
        assert not capture_dir.exists()
        assert legacy_partial.read_bytes() == b"operator-owned partial bytes\n"
        attempts = list((out_dir / "walkforward" / "captures").glob(f".{decision}.*"))
        assert len(attempts) == 1
        assert (attempts[0] / "cohorts.csv").read_bytes() == b"partial staged capture\n"

        monkeypatch.setattr(pd.DataFrame, "to_csv", original_to_csv)
        frame_script.cmd_capture(decision, db=db, frame="ingestion")
        assert legacy_partial.read_bytes() == b"operator-owned partial bytes\n"
        frozen = pd.read_csv(capture_dir / "cohorts.csv")
        assert set(frozen["ticker"].astype(str)) == set(FRAME_TICKERS[:3])
        first_bytes = {
            path.name: path.read_bytes() for path in capture_dir.iterdir()
        }
        with pytest.raises(SystemExit, match="CAPTURE OVERWRITE"):
            frame_script.cmd_capture(decision, db=db, frame="ingestion")
        assert first_bytes == {
            path.name: path.read_bytes() for path in capture_dir.iterdir()
        }
    finally:
        db.close()


def test_capture_requires_complete_matching_preexisting_hash_binding(
    tmp_path, frame_script
):
    decision = date(2027, 1, 15)
    cohort_path = tmp_path / "cohorts.csv"
    actor_path = tmp_path / "actors.csv"
    manifest_path = tmp_path / "manifest.json"
    cohort_path.write_text(
        f"ticker,as_of,frame\nORIGINAL,{decision},ingestion\n"
    )
    actor_path.write_text(
        f"ticker,as_of,frame\nORIGINAL,{decision},ingestion\n"
    )
    hashes = {
        "cohorts.csv": frame_script.sha256_file(cohort_path),
        "actors.csv": frame_script.sha256_file(actor_path),
    }
    manifest = {
        "capture_only": True,
        "decision": str(decision),
        "frame": "ingestion",
        "rows": 1,
        "actor_rows": 1,
    }

    # The old validator accepted this same-row-count mutation without a seal.
    cohort_path.write_text(f"ticker,as_of,frame\nREPLACED,{decision},ingestion\n")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="UNVERIFIED CAPTURE"):
        frame_script._validate_capture(
            decision, "ingestion", cohort_path, actor_path, manifest_path
        )

    cohort_path.write_text(f"ticker,as_of,frame\nORIGINAL,{decision},ingestion\n")
    for binding in (
        None,
        {"cohorts.csv": hashes["cohorts.csv"]},
        {**hashes, "actors.csv": "0" * 64},
    ):
        sealed_manifest = {**manifest, "artifact_sha256": binding} if binding else manifest
        manifest_path.write_text(json.dumps(sealed_manifest))
        expected = "hash mismatch" if binding and len(binding) == 2 else "UNVERIFIED CAPTURE"
        with pytest.raises(SystemExit, match=expected):
            frame_script._validate_capture(
                decision, "ingestion", cohort_path, actor_path, manifest_path
            )

    manifest_path.write_text(json.dumps({**manifest, "artifact_sha256": hashes}))
    assert frame_script._validate_capture(
        decision, "ingestion", cohort_path, actor_path, manifest_path
    )["artifact_sha256"] == hashes


def test_legacy_capture_during_staging_is_preserved_and_not_shadowed(
    tmp_path, monkeypatch, frame_script
):
    decision = date(2027, 1, 15)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", CaptureDate)

    def write_capture(directory, ticker, *, legacy):
        directory.mkdir(parents=True, exist_ok=True)
        if legacy:
            cohort = directory / f"cohorts_{decision}.csv"
            actor = directory / f"actors_{decision}.csv"
            manifest_path = directory / f"manifest_{decision}.json"
        else:
            cohort = directory / "cohorts.csv"
            actor = directory / "actors.csv"
            manifest_path = directory / "manifest.json"
        cohort.write_text(f"ticker,as_of,frame\n{ticker},{decision},ingestion\n")
        actor.write_text(f"ticker,as_of,frame\n{ticker},{decision},ingestion\n")
        manifest = {
            "capture_only": True,
            "decision": str(decision),
            "frame": "ingestion",
            "rows": 1,
            "actor_rows": 1,
        }
        if not legacy:
            manifest["artifact_sha256"] = {
                "cohorts.csv": frame_script.sha256_file(cohort),
                "actors.csv": frame_script.sha256_file(actor),
            }
        manifest_path.write_text(json.dumps(manifest))

    def publish_legacy_while_staging(
        _decision, _db, *, frame, capture_only, capture_output_dir
    ):
        assert frame == "ingestion" and capture_only
        write_capture(out_dir, "OLD_LEGACY", legacy=True)
        write_capture(capture_output_dir, "NEW_BUNDLE", legacy=False)

    monkeypatch.setattr(frame_script, "cmd_run", publish_legacy_while_staging)
    with pytest.raises(SystemExit, match="capture appeared during staging"):
        frame_script.cmd_capture(decision, frame="ingestion")

    capture_dir = out_dir / "walkforward" / "captures" / str(decision)
    legacy_paths = (
        out_dir / f"cohorts_{decision}.csv",
        out_dir / f"actors_{decision}.csv",
        out_dir / f"manifest_{decision}.json",
    )
    legacy_bytes = {path: path.read_bytes() for path in legacy_paths}
    assert not capture_dir.exists()
    cohort_path, actor_path, manifest_path, bundled = frame_script._capture_paths(
        decision
    )
    assert not bundled
    assert (cohort_path, actor_path, manifest_path) == legacy_paths
    assert pd.read_csv(cohort_path).loc[0, "ticker"] == "OLD_LEGACY"

    monkeypatch.setattr(frame_script, "date", UnsealDate)
    with pytest.raises(SystemExit, match="UNVERIFIED CAPTURE"):
        frame_script.cmd_attach_outcomes(decision, frame="ingestion")
    outcome_path = out_dir / f"outcomes_{decision}.csv"
    outcome_path.write_text("legacy evidence\n")
    with pytest.raises(SystemExit, match="UNVERIFIED CAPTURE"):
        frame_script.cmd_analyze_capture(decision, frame="ingestion")
    assert {path: path.read_bytes() for path in legacy_paths} == legacy_bytes
    assert outcome_path.read_bytes() == b"legacy evidence\n"

    # If a publisher races past its final preflight, readers reject ambiguity.
    write_capture(capture_dir, "NEW_BUNDLE", legacy=False)
    with pytest.raises(SystemExit, match="CAPTURE CONFLICT"):
        frame_script._capture_paths(decision)
    assert {path: path.read_bytes() for path in legacy_paths} == legacy_bytes


def test_outcome_failure_never_publishes_partial_target_and_retry_is_write_once(
    tmp_path, monkeypatch, frame_script
):
    decision = date(2027, 1, 15)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", CaptureDate)
    db = Database(tmp_path / "outcome-fault.duckdb")
    original_to_csv = pd.DataFrame.to_csv
    try:
        seed_frame(db, tickers=FRAME_TICKERS[:3])
        db.conn.execute("UPDATE transactions SET amount_midpoint = 10000.0")
        seed_prices(db, FRAME_TICKERS[:3], start="2026-06-01", end="2027-01-30")
        frame_script.cmd_capture(decision, db=db, frame="ingestion")
        capture_dir = out_dir / "walkforward" / "captures" / str(decision)
        for name, legacy_name in (
            ("cohorts.csv", f"cohorts_{decision}.csv"),
            ("actors.csv", f"actors_{decision}.csv"),
            ("manifest.json", f"manifest_{decision}.json"),
        ):
            (out_dir / legacy_name).write_bytes((capture_dir / name).read_bytes())
        legacy_capture_bytes = {
            name: (out_dir / name).read_bytes()
            for name in (
                f"cohorts_{decision}.csv",
                f"actors_{decision}.csv",
                f"manifest_{decision}.json",
            )
        }
        # Exercise the read-only legacy layout; publication uses a separate
        # retry sidecar and does not touch an old partial canonical outcome.
        retry_dir = out_dir / "walkforward-retry"
        monkeypatch.setattr(frame_script, "_walkforward_dir", lambda: retry_dir)
        with pytest.raises(SystemExit, match="CAPTURE OVERWRITE"):
            frame_script.cmd_capture(decision, db=db, frame="ingestion")
        assert legacy_capture_bytes == {
            name: (out_dir / name).read_bytes() for name in legacy_capture_bytes
        }
        monkeypatch.setattr(frame_script, "date", UnsealDate)
        legacy_outcome = out_dir / f"outcomes_{decision}.csv"
        legacy_outcome.write_bytes(b"operator-owned partial outcome\n")
        outcome_path = retry_dir / f"outcomes_{decision}.csv"

        def fail_after_partial(self, path_or_buf=None, *args, **kwargs):
            path_or_buf.write("partial outcomes\n")
            raise OSError("injected outcome serialization failure")

        monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_partial)
        with pytest.raises(OSError, match="injected outcome"):
            frame_script.cmd_attach_outcomes(decision, db=db, frame="ingestion")
        assert not outcome_path.exists()
        assert legacy_outcome.read_bytes() == b"operator-owned partial outcome\n"
        failed_attempts = list(
            outcome_path.parent.glob(f".{outcome_path.name}.*.tmp")
        )
        assert len(failed_attempts) == 1
        assert failed_attempts[0].read_bytes() == b"partial outcomes\n"

        monkeypatch.setattr(pd.DataFrame, "to_csv", original_to_csv)
        frame_script.cmd_attach_outcomes(decision, db=db, frame="ingestion")
        assert len(
            frame_script.cmd_analyze_capture(decision, frame="ingestion")
        ) == 3
        first_bytes = outcome_path.read_bytes()
        legacy_bytes = legacy_outcome.read_bytes()
        with pytest.raises(SystemExit, match="OUTCOME OVERWRITE"):
            frame_script.cmd_attach_outcomes(decision, db=db, frame="ingestion")
        assert outcome_path.read_bytes() == first_bytes
        assert legacy_outcome.read_bytes() == legacy_bytes
    finally:
        db.close()


def test_receipt_first_outcome_retry_reuses_immutable_receipt_across_days(
    tmp_path, monkeypatch, frame_script
):
    decision = date(2027, 1, 15)
    out_dir = tmp_path / "outcome-retry"
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    monkeypatch.setattr(frame_script, "date", UnsealDate)
    capture_paths = write_synthetic_bundle(out_dir, frame_script, decision)
    capture_bytes = {path: path.read_bytes() for path in capture_paths}
    outcome_path = out_dir / "walkforward" / f"outcomes_{decision}.csv"
    receipt_path = frame_script._outcome_binding_path(outcome_path)
    db = Database(":memory:")
    real_link = frame_script.os.link

    def fail_csv_publish(source, target, *args, **kwargs):
        if Path(target) == outcome_path:
            raise OSError("injected crash after receipt publication")
        return real_link(source, target, *args, **kwargs)

    try:
        monkeypatch.setattr(frame_script.os, "link", fail_csv_publish)
        with pytest.raises(OSError, match="after receipt publication"):
            frame_script.cmd_attach_outcomes(decision, db=db, frame="ingestion")
        original_receipt = receipt_path.read_bytes()
        assert json.loads(original_receipt)["labels_generated_on"] == "2028-11-01"
        assert not outcome_path.exists()
        assert capture_bytes == {path: path.read_bytes() for path in capture_paths}

        monkeypatch.setattr(frame_script.os, "link", real_link)
        monkeypatch.setattr(frame_script, "date", NextUnsealDate)
        frame_script.cmd_attach_outcomes(decision, db=db, frame="ingestion")
        assert receipt_path.read_bytes() == original_receipt
        assert frame_script.sha256_file(outcome_path) == json.loads(
            original_receipt
        )["outcome_sha256"]
        assert capture_bytes == {path: path.read_bytes() for path in capture_paths}
        assert len(frame_script.cmd_analyze_capture(
            decision, frame="ingestion"
        )) == 1

        changed_binding = json.loads(original_receipt)
        changed_binding["capture_sha256"] = {"cohorts.csv": "different"}
        with pytest.raises(SystemExit, match="OUTCOME BINDING"):
            frame_script._write_json_exclusive(changed_binding, receipt_path)
        original_binding = {
            key: value
            for key, value in json.loads(original_receipt).items()
            if key != "outcome_sha256"
        }
        original_binding["labels_generated_on"] = str(NextUnsealDate.today())
        with pytest.raises(SystemExit, match="OUTCOME BINDING"):
            frame_script._write_csv_exclusive(
                pd.read_csv(outcome_path).assign(excess=123.0),
                outcome_path,
                original_binding,
            )
        assert receipt_path.read_bytes() == original_receipt
        assert capture_bytes == {path: path.read_bytes() for path in capture_paths}
    finally:
        db.close()


# --- downstream guards: analyze / sweep ---


def write_panel(out_dir, decisions, name_template, rows, frame):
    for decision in decisions:
        panel = pd.DataFrame(rows)
        if "as_of" in panel.columns:
            panel = panel.loc[panel["as_of"].astype(str) == str(decision)]
        if frame is not None:
            panel["frame"] = frame
        panel.to_csv(out_dir / name_template.format(decision), index=False)


def test_analyze_requires_frame_column(tmp_path, monkeypatch, frame_script):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    rows = [
        {"as_of": "2023-01-15", "ticker": "AAPL",
         "skilled_present": False, "fresh_skilled": False},
    ]
    write_panel(out_dir, frame_script.DECISIONS, "cohorts_{}.csv", rows, None)
    with pytest.raises(SystemExit, match="explicit --frame") as excinfo:
        frame_script.read_panel_csv(
            out_dir / "cohorts_2023-01-15.csv", "cohort"
        )
    assert "cohort CSV cohorts_2023-01-15.csv" in str(excinfo.value)


def test_analyze_refuses_mixed_frames_but_accepts_uniform_panel(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    decisions = frame_script.DECISIONS
    rows = [
        {"as_of": str(decision), "ticker": "AAPL",
         "skilled_present": False, "fresh_skilled": False}
        for decision in decisions
    ]
    write_panel(out_dir, decisions, "cohorts_{}.csv", rows, "disclosure")
    outcomes = [
        {
            "as_of": str(decision), "ticker": "AAPL",
            "excess": None if decision == decisions[0] else 1.0,
            "entry_eligible": None if decision == decisions[0] else True,
            "exclusion_reason": (
                "outcome_price_unavailable" if decision == decisions[0] else None
            ),
        }
        for decision in decisions
    ]
    write_panel(out_dir, decisions, "outcomes_{}.csv", outcomes, "disclosure")
    frame_script.cmd_analyze(frame="disclosure")
    summary = pd.read_csv(out_dir / "summary.csv")
    assert summary["frame"].eq("disclosure").all()
    assert pd.isna(
        pd.read_csv(out_dir / f"outcomes_{decisions[0]}.csv").iloc[0]["excess"]
    )

    flipped = pd.read_csv(out_dir / f"cohorts_{decisions[-1]}.csv")
    flipped["frame"] = "ingestion"
    flipped.to_csv(out_dir / f"cohorts_{decisions[-1]}.csv", index=False)
    with pytest.raises(SystemExit, match="does not match requested"):
        frame_script.cmd_analyze(frame="disclosure")


def test_sweep_requires_frame_column_on_cohort_and_actor_csvs(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    decisions = frame_script.DECISIONS
    cohort_rows = [
        {"as_of": str(decision), "ticker": "AAPL",
         "ranked": True,
         "max_valid_ref": 100.0}
        for decision in decisions
    ]
    actor_rows = [
        {"as_of": str(decision), "ticker": "AAPL", "actor_id": "congress:X",
         "entry_ref": 100.0, "kind": "congress", "lag_days": 1}
        for decision in decisions
    ]
    write_panel(out_dir, decisions, "cohorts_{}.csv", cohort_rows, "ingestion")
    write_panel(out_dir, decisions, "actors_{}.csv", actor_rows, "ingestion")
    write_panel(
        out_dir, decisions, "outcomes_{}.csv",
        [{"as_of": str(d), "ticker": "AAPL", "excess": 1.0,
          "entry_eligible": True, "fill_px": 100.0} for d in decisions],
        "ingestion",
    )
    frame_script.cmd_sweep(frame="ingestion")
    assert pd.read_csv(out_dir / "sweep.csv")["frame"].eq("ingestion").all()

    write_panel(out_dir, decisions, "actors_{}.csv", actor_rows, "ingestion")
    write_panel(out_dir, decisions, "cohorts_{}.csv", cohort_rows, None)
    frame_script.cmd_sweep(frame="ingestion")


def test_sweep_refuses_mixed_frames_across_the_whole_panel(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    decisions = frame_script.DECISIONS
    cohort_rows = [
        {"as_of": str(decision), "ticker": "AAPL",
         "ranked": True,
         "max_valid_ref": 100.0}
        for decision in decisions
    ]
    actor_rows = [
        {"as_of": str(decision), "ticker": "AAPL", "actor_id": "congress:X",
         "entry_ref": 100.0, "kind": "congress", "lag_days": 1}
        for decision in decisions
    ]
    write_panel(out_dir, decisions, "cohorts_{}.csv", cohort_rows, "ingestion")
    write_panel(out_dir, decisions, "actors_{}.csv", actor_rows, "ingestion")
    write_panel(
        out_dir, decisions, "outcomes_{}.csv",
        [{"as_of": str(d), "ticker": "AAPL", "excess": 1.0,
          "entry_eligible": True, "fill_px": 100.0} for d in decisions],
        "ingestion",
    )

    flipped = pd.read_csv(out_dir / f"actors_{decisions[3]}.csv")
    flipped["frame"] = "disclosure"
    flipped.to_csv(out_dir / f"actors_{decisions[3]}.csv", index=False)
    with pytest.raises(SystemExit, match="does not match requested"):
        frame_script.cmd_sweep(frame="ingestion")

    # Keep the existing missing-file check: a hole in the panel still fails.
    (out_dir / f"actors_{decisions[3]}.csv").unlink()
    with pytest.raises(SystemExit, match="missing cohort, actor, or outcome CSVs"):
        frame_script.cmd_sweep(frame="ingestion")
