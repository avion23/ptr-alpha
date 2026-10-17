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

import pandas as pd
import pytest

from analyzer.database import Database
from analyzer.pipeline import eligible_events
from tests.test_asof_v2 import load_script, seed
from tests.test_backtest_guard import FRAME_TICKERS, install_skill_fixture, seed_frame

DECISION = date(2026, 6, 20)
FROZEN_TODAY = date(2026, 10, 17)

COHORT_COLUMNS = [
    "ticker", "as_of", "ranked", "score", "n_actors", "has_officer",
    "has_13f", "fresh", "fresh_skilled", "min_lag", "entry_px",
    "max_valid_ref", "entry_eligibility", "entry_eligible", "fwd", "spy",
    "excess", "skilled_present", "hyper_present", "hyper_only", "frame",
]
ACTOR_COLUMNS = [
    "as_of", "ticker", "actor_id", "kind", "entry_ref", "lag_days",
    "skilled", "hyper", "excess", "frame",
]
MANIFEST_KEYS = {
    "decision", "frame", "scoped_as_of", "events", "keep_tickers", "rows",
    "db_path", "db_sha256", "git_sha",
}


class FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 17)


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


# --- disclosure run succeeds where the ingestion frame is empty ---


def test_disclosure_run_succeeds_and_matches_direct_filtered_query(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    install_skill_fixture(out_dir, DECISION)
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
    install_skill_fixture(out_dir, DECISION)
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
    install_skill_fixture(out_dir, DECISION)
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
        f"manifest_{DECISION}.json",
    }


# --- downstream guards: analyze / sweep ---


def write_panel(out_dir, decisions, name_template, rows, frame):
    for decision in decisions:
        panel = pd.DataFrame(rows)
        if frame is not None:
            panel["frame"] = frame
        panel.to_csv(out_dir / name_template.format(decision), index=False)


def test_analyze_requires_frame_column(tmp_path, monkeypatch, frame_script):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    rows = [
        {"as_of": "2023-01-15", "ticker": "AAPL", "excess": 1.0,
         "entry_eligible": True, "skilled_present": False, "fresh_skilled": False},
    ]
    write_panel(out_dir, frame_script.DECISIONS, "cohorts_{}.csv", rows, None)
    with pytest.raises(SystemExit, match="no frame column") as excinfo:
        frame_script.cmd_analyze()
    assert "cohort CSV cohorts_2023-01-15.csv" in str(excinfo.value)


def test_analyze_refuses_mixed_frames_but_accepts_uniform_panel(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    decisions = frame_script.DECISIONS
    rows = [
        {"as_of": str(decision), "ticker": "AAPL", "excess": 1.0,
         "entry_eligible": True, "skilled_present": False, "fresh_skilled": False}
        for decision in decisions
    ]
    write_panel(out_dir, decisions, "cohorts_{}.csv", rows, "disclosure")
    frame_script.cmd_analyze()  # uniform panel passes and still writes summary
    assert (out_dir / "summary.csv").exists()

    flipped = pd.read_csv(out_dir / f"cohorts_{decisions[-1]}.csv")
    flipped["frame"] = "ingestion"
    flipped.to_csv(out_dir / f"cohorts_{decisions[-1]}.csv", index=False)
    with pytest.raises(SystemExit, match="mixed frames in panel:") as excinfo:
        frame_script.cmd_analyze()
    assert "disclosure" in str(excinfo.value) and "ingestion" in str(excinfo.value)


def test_sweep_requires_frame_column_on_cohort_and_actor_csvs(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    decisions = frame_script.DECISIONS
    cohort_rows = [
        {"as_of": str(decision), "ticker": "AAPL", "excess": 1.0,
         "entry_eligible": True, "ranked": True, "entry_px": 100.0,
         "max_valid_ref": 100.0}
        for decision in decisions
    ]
    actor_rows = [
        {"as_of": str(decision), "ticker": "AAPL", "actor_id": "congress:X",
         "excess": 1.0}
        for decision in decisions
    ]
    write_panel(out_dir, decisions, "cohorts_{}.csv", cohort_rows, "ingestion")
    write_panel(out_dir, decisions, "actors_{}.csv", actor_rows, None)
    with pytest.raises(SystemExit, match="no frame column") as excinfo:
        frame_script.cmd_sweep()
    assert "actor CSV actors_2023-01-15.csv" in str(excinfo.value)

    write_panel(out_dir, decisions, "actors_{}.csv", actor_rows, "ingestion")
    write_panel(out_dir, decisions, "cohorts_{}.csv", cohort_rows, None)
    with pytest.raises(SystemExit, match="no frame column") as excinfo:
        frame_script.cmd_sweep()
    assert "cohort CSV cohorts_2023-01-15.csv" in str(excinfo.value)


def test_sweep_refuses_mixed_frames_across_the_whole_panel(
    tmp_path, monkeypatch, frame_script
):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(frame_script, "OUT_DIR", out_dir)
    decisions = frame_script.DECISIONS
    cohort_rows = [
        {"as_of": str(decision), "ticker": "AAPL", "excess": 1.0,
         "entry_eligible": True, "ranked": True, "entry_px": 100.0,
         "max_valid_ref": 100.0}
        for decision in decisions
    ]
    actor_rows = [
        {"as_of": str(decision), "ticker": "AAPL", "actor_id": "congress:X",
         "excess": 1.0}
        for decision in decisions
    ]
    write_panel(out_dir, decisions, "cohorts_{}.csv", cohort_rows, "ingestion")
    write_panel(out_dir, decisions, "actors_{}.csv", actor_rows, "ingestion")

    flipped = pd.read_csv(out_dir / f"actors_{decisions[3]}.csv")
    flipped["frame"] = "disclosure"
    flipped.to_csv(out_dir / f"actors_{decisions[3]}.csv", index=False)
    with pytest.raises(SystemExit, match="mixed frames in panel:") as excinfo:
        frame_script.cmd_sweep()
    assert f"actors_{decisions[3]}.csv" in str(excinfo.value)

    # Keep the existing missing-file check: a hole in the panel still fails.
    (out_dir / f"actors_{decisions[3]}.csv").unlink()
    with pytest.raises(SystemExit, match="missing cohort or actor CSVs"):
        frame_script.cmd_sweep()
