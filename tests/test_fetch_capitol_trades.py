"""Smoke tests for scripts.fetch_capitol_trades module."""
import unittest
from unittest.mock import MagicMock, patch


class TestFetchCapitolTrades(unittest.TestCase):



    def test_fetch_all_trades_paginates(self):
        from scripts import fetch_capitol_trades

        # Mock response for two pages
        page1 = MagicMock()
        page1.json.return_value = {"trades": [{"id": 1}, {"id": 2}], "pages": 2, "total": 3}
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {"trades": [{"id": 3}], "pages": 2, "total": 3}
        page2.raise_for_status = MagicMock()

        with patch.object(fetch_capitol_trades.requests, "Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.headers = {}
            mock_session.get.side_effect = [page1, page2]
            mock_session_cls.return_value = mock_session

            with patch.object(fetch_capitol_trades.time, "sleep"):
                trades = fetch_capitol_trades.fetch_all_trades()

        self.assertEqual(len(trades), 3)
        self.assertEqual(trades[0]["id"], 1)
        self.assertEqual(trades[2]["id"], 3)


if __name__ == "__main__":
    unittest.main()