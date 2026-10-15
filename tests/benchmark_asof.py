"""Controlled offline replay benchmark; run with PYTHONPATH=<revision>/src.

Usage: python tests/benchmark_asof.py FIXTURE_DB OUTPUT_CSV [--seed]
Never accepts an existing database for seeding.
"""
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import resource
import sys
import time

import numpy as np
import pandas as pd

from analyzer.database import Database
from analyzer.pipeline import BacktestParams, run_backtest_pipeline


path = Path(sys.argv[1])
if "--seed" in sys.argv:
    assert not path.exists()
    db = Database(path)
    db.conn.execute("""
        INSERT INTO transactions (doc_id, member, ticker, transaction_date,
            disclosure_date, transaction_type, amount_midpoint, source, chamber,
            source_record_id, source_row_id, ingestion_generation, artifact_sha256,
            official_filing_date, created_at, instrument_type)
        SELECT 'doc' || (i // 25), 'Member ' || (i % 31),
            ['JPM','MSFT','AAPL','NVDA','AMZN'][1 + (i % 5)],
            DATE '2022-12-04' + (i % 390)::INTEGER,
            DATE '2022-12-04' + (i % 390)::INTEGER,
            'Purchase', 8000, 'house_pdf', 'House', 'doc' || (i // 25),
            'row' || i, 'gen', 'sha' || (i // 25),
            DATE '2022-12-04' + (i % 390)::INTEGER,
            TIMESTAMP '2022-01-01', 'stock'
        FROM range(25145) r(i)
    """)
    db.conn.execute("""
        INSERT INTO house_archive_generations
            (archive_year, generation_id, promoted_at, parse_status)
        VALUES (2023, 'gen', '2022-01-01', 'incomplete')
    """)
    db.conn.execute("""
        INSERT INTO house_pdf_artifacts (archive_year, generation_id, doc_id, artifact_sha256)
        SELECT DISTINCT 2023, 'gen', doc_id, artifact_sha256 FROM transactions
    """)
    db.conn.execute("""
        INSERT INTO house_generation_metadata (archive_year, generation_id, doc_id, filing_type)
        SELECT DISTINCT 2023, 'gen', doc_id, 'P' FROM transactions
    """)
    db.conn.execute("""
        INSERT INTO pdf_parse_runs (doc_id, year, parser_version, status, engines_attempted,
            transaction_count, artifact_sha256, ingestion_generation, parsed_at)
        SELECT doc_id, 2023, 'pdftotext', 'success', 'pdftotext', count(*),
            artifact_sha256, 'gen', TIMESTAMP '2022-01-02'
        FROM transactions GROUP BY doc_id, artifact_sha256
    """)
    db.close()

db = Database(path, read_only=True)
dates = pd.bdate_range("2022-01-01", "2024-06-01")
prices = pd.DataFrame({ticker: 100 + np.arange(len(dates)) * (0.01 + i * 0.001)
                       for i, ticker in enumerate(["JPM", "MSFT", "AAPL", "NVDA", "AMZN", "SPY"])}, index=dates)
start = time.perf_counter()
result = run_backtest_pipeline(
    BacktestParams(date(2023, 1, 1), date(2023, 12, 27)),
    SimpleNamespace(db=db), SimpleNamespace(get_prices=lambda *args: prices),
)
elapsed = time.perf_counter() - start
assert result.success, result
for name in ("combined", "summary", "date_observations"):
    if name in result.data:
        result.data[name].to_csv(sys.argv[2] + "." + name, index=False)
print(f"seconds={elapsed:.3f} max_rss_mib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}")
db.close()
