from datetime import date

import pandas as pd
import pytest

from analyzer.database import Database
from analyzer.pipeline import eligible_events
from analyzer import replay as replay_module


@pytest.mark.parametrize('read_only', [False, True])
def test_canonical_asof_preserves_live_and_restores_historical_generations(tmp_path, read_only, monkeypatch):
    path = tmp_path / 'asof.duckdb'
    db = Database(path)
    for generation, promoted, ticker in [
        ('august', '2026-08-25 18:53:44', 'JPM'),
        ('october', '2026-10-09 22:56:07', None),
    ]:
        db.conn.execute(
            "INSERT INTO house_archive_generations (archive_year, generation_id, parse_status, promoted_at) "
            "VALUES (2026, ?, 'complete', ?)", [generation, promoted],
        )
        db.conn.execute(
            "INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) "
            "VALUES (2026, ?, '20034202', 'same-pdf')", [generation],
        )
        db.conn.execute(
            "INSERT INTO house_generation_metadata (archive_year, generation_id, doc_id, filing_type) "
            "VALUES (2026, ?, '20034202', 'P')", [generation],
        )
        db.upsert_transactions(pd.DataFrame([{
            'doc_id': '20034202', 'member': 'Jane Doe', 'ticker': ticker,
            'transaction_date': date(2026, 3, 6), 'disclosure_date': date(2026, 3, 18),
            'transaction_type': 'Purchase', 'source_record_id': '20034202',
            'source_row_id': 'pdftotext:l26', 'chamber': 'house',
            'official_filing_date': date(2026, 3, 18),
            'ingestion_generation': generation, 'artifact_sha256': 'same-pdf',
        }]), source='house_pdf')
        db.conn.execute("UPDATE transactions SET created_at = ? WHERE ingestion_generation = ?", [promoted, generation])
        db.conn.execute(
            "INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, transaction_count, "
            "artifact_sha256, ingestion_generation, parsed_at) VALUES "
            "('20034202', 2026, 'pdftotext', 'success', 1, 'same-pdf', ?, ?)",
            [generation, promoted],
        )
    db.close()
    db = Database(path, read_only=read_only)
    try:
        live = db.conn.execute('SELECT * FROM canonical_transactions').fetchall()
        assert db.get_transactions(2026)['ticker'].tolist() == [None]
        with db.canonical_as_of(date(2026, 8, 26)):
            assert db.get_transactions(2026)['ticker'].tolist() == ['JPM']
            with db.canonical_as_of(date(2026, 10, 15)):
                assert db.conn.execute('SELECT * FROM canonical_transactions').fetchall() == live
            assert db.get_transactions(2026)['ticker'].tolist() == ['JPM']
        assert db.conn.execute('SELECT * FROM canonical_transactions').fetchall() == live
        # The public historical event loader scopes generation selection itself.
        assert eligible_events(db, date(2026, 8, 26))['ticker'].tolist() == ['JPM']
        assert eligible_events(db, date(2026, 10, 15)).empty
        # Replay must retain the scope through holdings and skill scoring,
        # not just through the event loader's own nested context.
        def candidates(scoped_db, *_):
            assert scoped_db.get_transactions(2026)['ticker'].tolist() == ['JPM']
            return pd.DataFrame()

        def score(scoped_db, *_):
            assert scoped_db.get_transactions(2026)['ticker'].tolist() == ['JPM']
            return 1.0, False, 1

        monkeypatch.setattr(replay_module, '_load_dependencies', lambda: (None,) * 4)
        monkeypatch.setattr(replay_module, '_ticker_candidates', candidates)
        monkeypatch.setattr(replay_module, '_score', score)
        replay_module.replay(db, ['JPM'], [date(2026, 8, 26)])
        assert db.conn.execute('SELECT * FROM canonical_transactions').fetchall() == live
        with pytest.raises(RuntimeError), db.canonical_as_of(date(2026, 8, 26)):
            raise RuntimeError('restore on failure')
        assert db.conn.execute('SELECT * FROM canonical_transactions').fetchall() == live
        with db.canonical_as_of(date(2026, 8, 24)):
            assert db.get_transactions(2026).empty
    finally:
        db.close()


def test_future_complete_generation_does_not_hide_historical_legacy_rows(tmp_path):
    db = Database(tmp_path / 'legacy.duckdb')
    try:
        db.upsert_transactions(pd.DataFrame([{
            'doc_id': 'legacy', 'member': 'Jane Doe', 'ticker': 'JPM',
            'transaction_date': date(2026, 3, 6), 'disclosure_date': date(2026, 3, 18),
            'transaction_type': 'Purchase',
        }]), source='house_pdf')
        db.conn.execute("UPDATE transactions SET created_at = '2026-03-18'")
        db.conn.execute(
            "INSERT INTO house_archive_generations (archive_year, generation_id, parse_status, promoted_at) "
            "VALUES (2026, 'future', 'complete', '2026-10-09')"
        )
        db.conn.execute(
            "INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256) "
            "VALUES (2026, 'future', 'legacy', 'empty-pdf')"
        )
        assert db.get_transactions(2026).empty
        with db.canonical_as_of(date(2026, 8, 26)):
            assert db.get_transactions(2026)['ticker'].tolist() == ['JPM']
        assert db.get_transactions(2026).empty
    finally:
        db.close()
