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


def make_entry_prices(transactions_df, prices_df):
    prices_long = prices_df.stack().reset_index(name="price")
    prices_long.columns = ["price_date", "ticker", "price"]
    prices_long = prices_long.sort_values("price_date")
    trans_sorted = transactions_df.sort_values("disclosure_date")
    merged = pd.merge_asof(
        trans_sorted, prices_long,
        left_on="disclosure_date", right_on="price_date", by="ticker"
    ).dropna(subset=["price"])
    optional_columns = [
        col for col in ("owner_code", "amount_midpoint") if col in merged.columns
    ]
    return merged[["member", "ticker", "disclosure_date", "transaction_type", "price", *optional_columns]].rename(
        columns={"price": "entry_price"}
    ).reset_index(drop=True)


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.tmp_dir) / "test.duckdb"
        self.db = Database(self.db_path)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp_dir)
