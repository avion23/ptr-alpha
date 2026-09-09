import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from analyzer.database import Database
from analyzer.models import TickerOrigin


def make_entry_prices(transactions_df, prices_df):
    prices_long = prices_df.stack().reset_index(name="price")
    prices_long.columns = ["price_date", "ticker", "price"]
    prices_long = prices_long.sort_values("price_date")
    trans_sorted = transactions_df.sort_values("disclosure_date")
    merged = pd.merge_asof(
        trans_sorted,
        prices_long,
        left_on="disclosure_date",
        right_on="price_date",
        by="ticker",
    ).dropna(subset=["price"])
    optional_columns = [
        col for col in ("owner_code", "amount_midpoint") if col in merged.columns
    ]
    return (
        merged[
            [
                "member",
                "ticker",
                "disclosure_date",
                "transaction_type",
                "price",
                *optional_columns,
            ]
        ]
        .rename(columns={"price": "entry_price"})
        .reset_index(drop=True)
    )


def make_raw_trade(ingestion_generation, **overrides):
    report_id = "37900303-65bf-467d-962b-76555d510b28"
    report_path = f"/search/view/ptr/{report_id}/"
    trade = {
        "doc_id": report_id,
        "source_record_id": report_id,
        "source_row_id": "official:1",
        "source_report_path": report_path,
        "senator": "Katie Britt",
        "filed_date": pd.Timestamp("2026-01-29"),
        "official_filing_date": pd.Timestamp("2026-01-29"),
        "available_date": pd.Timestamp("2026-01-29"),
        "notification_date": "01/29/2026",
        "amends_source_record_id": None,
        "ticker": "JPM",
        "ticker_raw": "JPM",
        "ticker_candidate": None,
        "ticker_origin": TickerOrigin.OFFICIAL.value,
        "transaction_date": "01/28/2026",
        "type": "Sale (Full)",
        "transaction_subtype_raw": "Sale (Full)",
        "owner": "SP",
        "owner_raw": "Spouse",
        "amount_range": "$1,001 - $15,000",
        "amount_range_raw": "$1,001 - $15,000",
        "asset_name": "JPMorgan Chase & Co. Common Stock",
        "asset_type": "Stock",
        "ingestion_generation": ingestion_generation,
        "artifact_sha256": "a" * 64,
    }
    trade.update(overrides)
    return trade


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.duckdb"
        self.db = Database(self.db_path)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp_dir)
