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
    "ticker", "as_of", "fill_date", "fill_px", "entry_eligibility",
    "entry_eligible", "fwd", "spy", "excess", "outcome_status",
    "exclusion_reason", "frame",
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
    monkeypatch.setattr(frame_script, "compute_weight", lambda *args: 0.01)
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
        cohort_path = out_dir / f"cohorts_{decision}.csv"
        capture = pd.read_csv(cohort_path)
        actor_capture = pd.read_csv(out_dir / f"actors_{decision}.csv")
        assert len(capture) == 3
        assert not {"fwd", "spy", "excess", "fill_px", "skilled_present"} & set(capture)
        assert not {"skilled", "hyper", "excess"} & set(actor_capture)
        assert not (out_dir / "skill.json").exists()
        assert not (out_dir / f"outcomes_{decision}.csv").exists()
        assert json.loads((out_dir / f"manifest_{decision}.json").read_text())["capture_only"]

        with pytest.raises(SystemExit, match="SEALED ANALYSIS"):
            frame_script.cmd_analyze_capture(decision, frame="ingestion")
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
