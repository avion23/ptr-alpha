import unittest
from datetime import date
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from analyzer.pipeline import (
    TickerScoringParams,
    prepare_analysis_data,
    prepare_live_analysis_data,
    run_recent_ticker_scoring,
)
from analyzer.cli import _save_results
from analyzer.exceptions import DataSourceError
from analyzer.models import AnalysisMode










class TestPrepareAnalysisData(unittest.TestCase):

    @patch("analyzer.pipeline.analysis.calculate_signal_potential")
    def testprepare_analysis_data(self, mock_calc_signals):
        mock_transactions = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-01-15"]),
            "transaction_type": ["Purchase"],
        })

        mock_prices = pd.DataFrame({
            "AAPL": [100.0],
            "SPY": [200.0],
        }, index=pd.to_datetime(["2024-01-15"]))

        mock_entry_prices = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-01-15"]),
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        })

        mock_signals = pd.DataFrame({
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "signal_type": ["Purchase"],
            "horizon_days": [90],
        })

        mock_calc_signals.return_value = mock_signals

        mock_tx_source = MagicMock()
        mock_tx_source.get_transactions.return_value = mock_transactions
        mock_tx_source.db.get_entry_prices.return_value = mock_entry_prices

        mock_price_source = MagicMock()
        mock_price_source.get_prices.return_value = mock_prices

        trades, prices, signals = prepare_analysis_data(mock_tx_source, mock_price_source, 2024, [90])

        mock_tx_source.get_transactions.assert_called_once_with(2024)
        mock_tx_source.db.get_entry_prices.assert_called_once()
        mock_price_source.get_prices.assert_called_once()

        self.assertEqual(len(trades), 1)
        self.assertEqual(len(signals), 1)
        mock_calc_signals.assert_called_once_with(mock_entry_prices, mock_prices, [90])

    @patch("analyzer.pipeline.analysis.calculate_signal_potential")
    def testprepare_analysis_data_empty_trades_raises(self, mock_calc_signals):
        mock_tx_source = MagicMock()
        mock_tx_source.get_transactions.return_value = pd.DataFrame()

        mock_price_source = MagicMock()

        with self.assertRaises(DataSourceError):
            prepare_analysis_data(mock_tx_source, mock_price_source, 2024, [90])

        mock_calc_signals.assert_not_called()


class TestRecentTickerScoring(unittest.TestCase):

    @patch("analyzer.pipeline.analysis.calculate_signal_potential")
    def test_live_training_loads_history_and_caps_all_data_at_as_of(self, mock_calc):
        source = MagicMock()
        prices = MagicMock()
        as_of = pd.Timestamp("2026-07-10")
        source.db.get_transactions_by_date_range.return_value = pd.DataFrame({
            "member": ["Historical", "Recent", "Future"],
            "ticker": ["AAPL", "MSFT", "TSLA"],
            "disclosure_date": pd.to_datetime([
                "2024-08-01", "2026-07-01", "2026-07-11",
            ]),
            "transaction_type": ["Purchase"] * 3,
        })
        prices.get_prices.return_value = pd.DataFrame(
            {"AAPL": [100.0], "MSFT": [200.0], "SPY": [500.0]},
            index=[as_of],
        )
        source.db.get_entry_prices.return_value = pd.DataFrame()
        mock_calc.return_value = pd.DataFrame({
            "disclosure_date": pd.to_datetime(["2024-08-01", "2026-07-01"]),
        })

        trades, _, signals = prepare_live_analysis_data(
            source, prices, (180,), as_of, 1095,
        )

        self.assertEqual(set(trades["member"]), {"Historical", "Recent"})
        self.assertEqual(len(signals), 2)
        _, query_end = source.db.get_transactions_by_date_range.call_args.args
        self.assertEqual(query_end, as_of)
        _, price_start, price_end = prices.get_prices.call_args.args
        self.assertLess(price_start, pd.Timestamp("2024-08-01"))
        self.assertEqual(price_end, as_of)

    @patch("analyzer.pipeline.analysis.rank_members")
    @patch("analyzer.pipeline.analysis.score_ticker_by_buyers")
    @patch("analyzer.pipeline.prepare_live_analysis_data")
    def test_does_not_present_negative_scores_as_buys(self, mock_prepare, mock_score, mock_rank):
        trades = pd.DataFrame({
            "member": ["Alice", "Bob"], "ticker": ["AAPL", "AAPL"],
            "disclosure_date": pd.to_datetime(["2024-05-01", "2024-05-02"]),
            "transaction_type": ["Purchase", "Purchase"],
        })
        mock_prepare.return_value = (trades, pd.DataFrame(), pd.DataFrame())
        mock_rank.return_value = pd.DataFrame({"member": ["Alice"]})
        mock_score.return_value = pd.DataFrame({"ticker": ["AAPL"], "signal_score": [-1.0]})

        result = run_recent_ticker_scoring(
            MagicMock(), MagicMock(),
            TickerScoringParams(year=2024, horizons=[90], days_back=30, min_buyers=2, as_of_date=date(2024, 5, 15)),
        )

        self.assertTrue(result.data["result"].empty)

    @patch("analyzer.pipeline.analysis.rank_members")
    @patch("analyzer.pipeline.analysis.score_ticker_by_buyers")
    @patch("analyzer.pipeline.prepare_live_analysis_data")
    def test_scores_use_recent_trades_not_full_year_trades(self, mock_prepare, mock_score, mock_rank):
        today = pd.Timestamp.today().normalize()
        trades = pd.DataFrame({
            "member": ["Alice", "Bob", "Old Buyer"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "disclosure_date": [today - pd.Timedelta(days=2), today - pd.Timedelta(days=1), today - pd.Timedelta(days=100)],
            "transaction_type": ["Purchase", "Purchase", "Purchase"],
        })
        signals = pd.DataFrame({
            "member": ["Alice", "Bob", "Old Buyer"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "signal_type": ["Purchase", "Purchase", "Purchase"],
            "horizon_days": [90, 90, 90],
            "decayed_return_pct": [1.0, 1.0, 1.0],
            "peak_potential_pct": [1.0, 1.0, 1.0],
            "spy_alpha_pct": [1.0, 1.0, 1.0],
        })
        rankings = pd.DataFrame({
            "member": ["Alice", "Bob", "Old Buyer"],
            "avg_spy_alpha_pct": [1.0, 1.0, 100.0],
            "purchase_trades": [1, 1, 1],
        })
        mock_prepare.return_value = (trades, pd.DataFrame(), signals)
        mock_rank.return_value = rankings
        mock_score.return_value = pd.DataFrame({
            "ticker": ["AAPL"],
            "signal_score": [1.0],
        })

        result = run_recent_ticker_scoring(MagicMock(), MagicMock(), TickerScoringParams(year=today.year, horizons=[90], days_back=30, min_buyers=2))

        self.assertTrue(result)
        scored_trades = mock_score.call_args.args[1]
        self.assertEqual(set(scored_trades["member"]), {"Alice", "Bob"})
        self.assertNotIn("Old Buyer", set(scored_trades["member"]))

    @patch("analyzer.pipeline.analysis.rank_members")
    @patch("analyzer.pipeline.analysis.score_ticker_by_buyers")
    @patch("analyzer.pipeline.prepare_live_analysis_data")
    def test_excludes_future_trades_and_non_positive_scores(self, mock_prepare, mock_score, mock_rank):
        today = pd.Timestamp.today().normalize()
        trades = pd.DataFrame({
            "member": ["Alice", "Bob", "Future"],
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "disclosure_date": [today - pd.Timedelta(days=2), today - pd.Timedelta(days=1), today + pd.Timedelta(days=1)],
            "transaction_type": ["Purchase"] * 3,
        })
        signals = pd.DataFrame({
            "member": ["Alice"], "ticker": ["OLD"], "signal_type": ["Purchase"],
            "horizon_days": [90], "decayed_return_pct": [1.0],
        })
        mock_prepare.return_value = (trades, pd.DataFrame(), signals)
        mock_rank.return_value = pd.DataFrame()
        mock_score.return_value = pd.DataFrame({"ticker": ["AAPL"], "signal_score": [-1.0]})

        result = run_recent_ticker_scoring(
            MagicMock(), MagicMock(),
            TickerScoringParams(year=today.year, horizons=[90], days_back=30, min_buyers=2),
        )

        self.assertTrue(result.data["result"].empty)
        self.assertEqual(result.data["as_of_date"], today.date())
        self.assertNotIn("Future", set(mock_score.call_args.args[1]["member"]))


class TestSaveResults(unittest.TestCase):


    def test_csv_output_creates_file(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_spy_alpha_pct": [10.5],
            "bayes_win_prob": [0.8],
            "peak_hit_rate_pct": [70.0],
            "sharpe_ratio": [1.2],
            "bayes_factor": [2.1],
            "purchase_trades": [5],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", AnalysisMode.MEMBER_RANKINGS, None, data_dir)

            filepath = data_dir / "member_rankings.csv"
            self.assertTrue(filepath.exists())

            saved = pd.read_csv(filepath)
            self.assertEqual(len(saved), 1)
            self.assertIn("member", saved.columns)

    def test_csv_output_member_filter_filename(self):
        table = pd.DataFrame({
            "ticker": ["AAPL"],
            "spy_alpha_pct": [10.0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", AnalysisMode.MEMBER_SIGNALS, "John Doe", data_dir)

            filepath = data_dir / "john_doe_signals.csv"
            self.assertTrue(filepath.exists())

    def test_csv_output_top_signals_filename(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "spy_alpha_pct": [10.0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", AnalysisMode.TOP_SIGNALS, None, data_dir)

            filepath = data_dir / "top_signals.csv"
            self.assertTrue(filepath.exists())

    def test_fallback_columns(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_loss_avoided_pct": [12.5],
            "median_loss_avoided_pct": [10.0],
            "sale_trades": [3],
            "sharpe_ratio": [1.1],
            "bayes_win_prob": [0.75],
            "bayes_factor": [2.0],
            "avg_spy_alpha_pct": [8.0],
            "col_a": [1],
            "col_b": [2],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", AnalysisMode.SALE_RANKINGS, None, data_dir)

            saved = pd.read_csv(data_dir / "sale_rankings.csv")
            self.assertIn("avg_loss_avoided_pct", saved.columns)
            self.assertIn("median_loss_avoided_pct", saved.columns)
            self.assertNotIn("col_a", saved.columns)
            self.assertNotIn("col_b", saved.columns)

    def test_sales_csv_output_uses_sale_rankings_filename(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_loss_avoided_pct": [12.5],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", AnalysisMode.SALE_RANKINGS, None, data_dir)

            self.assertTrue((data_dir / "sale_rankings.csv").exists())
            self.assertFalse((data_dir / "member_rankings.csv").exists())

    def test_sales_columns_included_in_display(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_loss_avoided_pct": [12.5],
            "median_loss_avoided_pct": [10.0],
            "sale_trades": [3],
            "sharpe_ratio": [1.1],
            "bayes_win_prob": [0.75],
            "bayes_factor": [2.0],
            "avg_spy_alpha_pct": [8.0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", AnalysisMode.SALE_RANKINGS, None, data_dir)

            saved = pd.read_csv(data_dir / "sale_rankings.csv")
            self.assertIn("avg_loss_avoided_pct", saved.columns)
            self.assertIn("median_loss_avoided_pct", saved.columns)
            self.assertIn("sale_trades", saved.columns)
            self.assertIn("avg_spy_alpha_pct", saved.columns)

    def test_sales_columns_excluded_ranking_columns(self):
        table = pd.DataFrame({
            "member": ["Alice"],
            "avg_loss_avoided_pct": [12.5],
            "median_loss_avoided_pct": [10.0],
            "sale_trades": [3],
            "sharpe_ratio": [1.1],
            "bayes_win_prob": [0.75],
            "bayes_factor": [2.0],
            "avg_spy_alpha_pct": [8.0],
        })

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _save_results(table, "csv", AnalysisMode.SALE_RANKINGS, None, data_dir)

            saved = pd.read_csv(data_dir / "sale_rankings.csv")
            self.assertNotIn("purchase_trades", saved.columns)
            self.assertNotIn("peak_hit_rate_pct", saved.columns)




if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Pytest-style tests (use caplog fixture — cannot use unittest.TestCase here)
# ---------------------------------------------------------------------------
