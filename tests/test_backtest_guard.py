"""Degenerate-frame and sealed-block guards for scripts/backtest_assumptions.

Replays the measured incident: a run whose decision date precedes every
ingestion watermark sees an empty canonical view and used to write a
1-byte cohorts CSV with exit code 0. The guards must refuse loudly at the
cause, before anything is written.
"""

import json
from datetime import date

import pandas as pd
import pytest

from analyzer.database import Database
from tests.test_asof_v2 import load_script

MEMBERS = ["Jane Doe", "Bob Smith", "Carol Jones"]
# Letters-only synthetic tickers: _VALID_TICKER_RE rejects digits, and the
# resolver passes unknown but well-formed symbols through as "unverified".
FRAME_TICKERS = [f"Z{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(25)]


@pytest.fixture(scope="module")
def guard_script():
    return load_script("backtest_assumptions")


def seed_frame(db, *, tickers, members=MEMBERS, promoted="2026-06-30",
               created="2026-07-01", accepted="2026-07-02",
               disclosure="2026-06-15"):
    """Seed one house doc per ticker with distinct congress members.

    Mirrors tests/test_asof_v2.py::seed but bulk: every doc carries one row
    per member, so each ticker has >= 2 distinct buyer actors and counts as
    candidate-rich in cmd_run.
    """
    rows, runs = [], []
    for i, ticker in enumerate(tickers):
        doc = f"guard-doc-{i}"
        db.conn.execute(
            "INSERT INTO house_archive_generations "
            "(archive_year, generation_id, promoted_at) VALUES (?, ?, ?)",
            [2026, doc, promoted],
        )
        db.conn.execute(
            "INSERT INTO house_generation_metadata "
            "(archive_year, generation_id, doc_id, filing_type) VALUES (?, ?, ?, 'P')",
            [2026, doc, doc],
        )
        db.conn.execute(
            "INSERT INTO house_pdf_artifacts "
            "(archive_year, generation_id, doc_id, artifact_sha256) VALUES (?, ?, ?, 'sha')",
            [2026, doc, doc],
        )
        for j, member in enumerate(members):
            rows.append(dict(
                doc_id=doc, member=member, ticker=ticker,
                transaction_date=disclosure, disclosure_date=disclosure,
                transaction_type="Purchase", chamber="house",
                source_record_id=doc, source_row_id=f"row{j}",
                official_filing_date=disclosure,
                ingestion_generation=doc, artifact_sha256="sha",
            ))
        runs.append(dict(
            doc_id=doc, year=2026, parser_version="pdftotext", status="success",
            engines_attempted="pdftotext", raw_row_count=len(members),
            transaction_count=len(members), artifact_sha256="sha",
            ingestion_generation=doc,
        ))
    db.upsert_transactions(pd.DataFrame(rows), source="house_pdf")
    db.conn.execute("UPDATE transactions SET created_at = ?", [created])
    for run in runs:
        db.parse_runs.upsert(**run)
    db.conn.execute(
        "UPDATE pdf_parse_runs SET parsed_at = ?, rowset_created_at = NULL",
        [accepted],
    )


def install_skill_fixture(out_dir, decision, frame="ingestion"):
    """cmd_run reads OUT_DIR/skill.json before the guard; provide a stub."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skill.json").write_text(json.dumps(
        {"frame": frame, "by_decision": {
            str(decision): {"labels": [], "breadth": {}}
        }}
    ))


# --- pure-function tests: check_frame ---


def test_check_frame_refuses_short_events_with_full_message(guard_script):
    with pytest.raises(SystemExit, match="DEGENERATE FRAME") as excinfo:
        guard_script.check_frame(49, 20, date(2023, 1, 15))
    message = str(excinfo.value)
    assert "2023-01-15" in message
    assert "ingestion" in message
    assert "49 eligible events" in message and "minimum 50" in message
    assert "20 candidate tickers" in message and "minimum 20" in message
    assert "first ingestion watermark" in message


def test_check_frame_refuses_short_tickers(guard_script):
    with pytest.raises(SystemExit, match="DEGENERATE FRAME"):
        guard_script.check_frame(50, 19, date(2023, 7, 15))
    with pytest.raises(SystemExit, match="DEGENERATE FRAME"):
        guard_script.check_frame(0, 0, date(2023, 1, 15))


def test_check_frame_passes_at_and_above_thresholds(guard_script):
    assert guard_script.check_frame(50, 20, date(2023, 1, 15)) is None
    assert guard_script.check_frame(500, 200, date(2027, 1, 15)) is None


# --- pure-function tests: check_sealed ---


def test_check_sealed_refuses_every_block_date_before_unseal(guard_script):
    for decision in guard_script.WALK_FORWARD_BLOCK:
        with pytest.raises(SystemExit, match="SEALED BLOCK") as excinfo:
            guard_script.check_sealed(decision, today=date(2026, 10, 17))
        message = str(excinfo.value)
        assert str(decision) in message
        assert "docs/strategy-charter.md" in message
        assert "unseal >= 2028-11-01" in message
    # Still sealed inside the block window itself.
    with pytest.raises(SystemExit, match="SEALED BLOCK"):
        guard_script.check_sealed(date(2028, 7, 15), today=date(2028, 7, 16))


def test_check_sealed_allows_nonblock_dates_and_unseal_boundary(guard_script):
    assert guard_script.check_sealed(
        date(2023, 1, 15), today=date(2026, 10, 17)
    ) is None
    assert guard_script.check_sealed(
        date(2026, 7, 1), today=date(2027, 1, 16)
    ) is None
    # On/after the unseal date the block is runnable again.
    assert guard_script.check_sealed(
        date(2027, 1, 15), today=date(2028, 11, 1)
    ) is None


# --- call-site tests: cmd_run ordering and wiring ---


@pytest.mark.parametrize("decision,accepted,expected_events", [
    # Visible but tiny frame: counts are real, below both minimums.
    (date(2026, 7, 15), "2026-07-02", 9),
    # The measured incident: every watermark postdates the decision date,
    # the scoped canonical view is empty, nothing was ever written.
    (date(2026, 1, 15), "2026-07-02", 0),
])
def test_cmd_run_refuses_degenerate_frame_before_writing_anything(
    tmp_path, monkeypatch, guard_script, decision, accepted, expected_events
):
    out_dir = tmp_path / "out"
    install_skill_fixture(out_dir, decision)
    monkeypatch.setattr(guard_script, "OUT_DIR", out_dir)
    db = Database(tmp_path / "frame.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS[:3], accepted=accepted)
        with pytest.raises(SystemExit, match="DEGENERATE FRAME") as excinfo:
            guard_script.cmd_run(decision, db=db, frame="ingestion")
        assert f"{expected_events} eligible events" in str(excinfo.value)
    finally:
        db.close()
    assert list(out_dir.glob("cohorts_*.csv")) == []
    assert list(out_dir.glob("actors_*.csv")) == []
    assert list(out_dir.glob("manifest_*.json")) == []


def test_cmd_run_invokes_frame_guard_with_computed_counts_before_writes(
    tmp_path, monkeypatch, guard_script
):
    out_dir = tmp_path / "out"
    decision = date(2026, 7, 15)
    install_skill_fixture(out_dir, decision)
    monkeypatch.setattr(guard_script, "OUT_DIR", out_dir)
    db = Database(tmp_path / "frame.duckdb")
    try:
        seed_frame(db, tickers=FRAME_TICKERS)  # 75 events / 25 candidate tickers
        calls = []

        class GuardPassed(Exception):
            pass

        def recording_guard(events, tickers, seen_decision, frame="ingestion"):
            calls.append((events, tickers, seen_decision, frame))
            raise GuardPassed

        monkeypatch.setattr(guard_script, "check_frame", recording_guard)
        with pytest.raises(GuardPassed):
            guard_script.cmd_run(decision, db=db, frame="ingestion")
        assert calls == [(75, 25, decision, "ingestion")]
    finally:
        db.close()
    # Stopped at the guard: no cohort, actor, or manifest artifact was opened.
    assert list(out_dir.glob("cohorts_*.csv")) == []
    assert list(out_dir.glob("actors_*.csv")) == []
    assert list(out_dir.glob("manifest_*.json")) == []


def test_cmd_run_refuses_sealed_block_before_touching_disk(
    tmp_path, monkeypatch, guard_script
):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(guard_script, "OUT_DIR", out_dir)
    # Safety net: a regression past the seal check must not open the real DB.
    monkeypatch.setattr(guard_script, "DB_PATH", tmp_path / "unused.duckdb")

    class FrozenDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 17)

    monkeypatch.setattr(guard_script, "date", FrozenDate)
    with pytest.raises(SystemExit, match="SEALED BLOCK"):
        guard_script.cmd_run(date(2027, 1, 15), frame="ingestion")
    assert not out_dir.exists()
