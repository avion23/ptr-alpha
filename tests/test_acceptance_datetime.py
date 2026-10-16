from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from analyzer.database import Database
from analyzer.form4 import _acceptance_datetime
from analyzer.transaction_repository import acceptance_datetime_violations


def test_init_and_legacy_read_only_migration(tmp_path):
    path = tmp_path / "legacy.duckdb"
    with Database(path) as db:
        assert dict((r[1], r[2]) for r in db.conn.execute(
            "PRAGMA table_info('transactions')"
        ).fetchall())["acceptance_datetime"] == "TIMESTAMP"
        db.conn.execute("INSERT INTO transactions (doc_id, source, owner_code, amount_raw) VALUES ('legacy', 'form4', '', '')")
        # Simulate the pre-migration schema, including a retired row.
        db.conn.execute("INSERT INTO transaction_versions SELECT *, CURRENT_TIMESTAMP FROM transactions")
        db.conn.execute("DROP VIEW canonical_transactions")
        for (index,) in db.conn.execute("SELECT index_name FROM duckdb_indexes() WHERE table_name = 'transactions'").fetchall():
            db.conn.execute(f"DROP INDEX {index}")
        for table in ("transactions", "transaction_versions"):
            db.conn.execute(f"ALTER TABLE {table} DROP COLUMN acceptance_datetime")
    before = path.read_bytes()
    with Database(path, read_only=True) as db:
        assert db.conn.execute("SELECT doc_id FROM canonical_transactions").fetchall() == [("legacy",)]
        assert "acceptance_datetime" not in db.conn.execute("SELECT * FROM transactions").fetchdf()
    assert path.read_bytes() == before
    with Database(path) as db:
        assert db.conn.execute("SELECT acceptance_datetime FROM transactions").fetchall() == [(None,)]
        assert db.conn.execute("SELECT acceptance_datetime FROM transaction_versions").fetchall() == [(None,)]
        base = db.conn.execute("SELECT * FROM transactions LIMIT 0").description
        history = db.conn.execute("SELECT * FROM transaction_versions LIMIT 0").description
        assert [c[0] for c in history] == [c[0] for c in base] + ["retired_at"]
        db.conn.execute("INSERT INTO transaction_versions SELECT *, CURRENT_TIMESTAMP FROM transactions")
        with db.canonical_as_of(date(2026, 9, 2)):
            db.conn.execute("SELECT * FROM canonical_transactions").fetchall()
    with Database(path, read_only=True) as db:
        assert db.conn.execute("SELECT acceptance_datetime FROM transactions").fetchall() == [(None,)]


@pytest.mark.parametrize("accession, expected", [
    ("0001268406-25-000027", datetime(2025, 10, 15, 21, 43, 51)),
    ("0001193125-26-352200", datetime(2026, 8, 14, 16, 5, 4)),
    ("0001193125-26-054580", datetime(2026, 2, 17, 16, 5, 4)),
])
def test_captured_header_clock(accession, expected):
    header = (Path(__file__).parent / "fixtures/acceptance" / f"{accession}.txt").read_bytes()
    assert _acceptance_datetime(content=header) == expected
    assert _acceptance_datetime(content=header.replace(b"<ACCEPTANCE-DATETIME>", b"ACCEPTANCE-DATETIME\t")) == expected
    assert _acceptance_datetime() is None


@pytest.mark.parametrize("accepted, violates", [
    (None, False),
    ("2026-09-01T20:30:00", False),
    ("2026-09-02T00:00:00", False),
    ("2026-09-02T00:00:01", True),
    ("2026-08-30T00:00:00", False),
    ("2026-08-29T23:59:59", True),
])
def test_acceptance_invariant(accepted, violates):
    df = pd.DataFrame([{
        "acceptance_datetime": accepted,
        "disclosure_date": date(2026, 9, 1),
        "transaction_date": date(2026, 8, 31),
    }])
    assert bool(acceptance_datetime_violations(df).iloc[0]) == violates


def test_invariant_reports_without_coercion(tmp_path, caplog):
    accepted = datetime(2026, 9, 3)
    with Database(tmp_path / "test.duckdb") as db:
        df = pd.DataFrame([{
            "doc_id": "bad-clock", "member": "Officer", "ticker": "TEST",
            "transaction_date": date(2026, 8, 31),
            "disclosure_date": date(2026, 9, 1),
            "transaction_type": "Purchase", "acceptance_datetime": accepted,
        }])
        db.upsert_transactions(df, source="form4")
        assert "invariant violated for 1 rows" in caplog.text
        assert db.conn.execute("SELECT acceptance_datetime FROM transactions").fetchone() == (accepted,)
