from datetime import date, timedelta
from itertools import pairwise
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from analyzer import actors, pipeline, positions, replay, setups

AS_OF = date(2026, 6, 30)
BEFORE_JUNE = date(2026, 5, 1)
MEMBERS = [
    "Byron Donalds",
    "Gilbert Cisneros",
    "Charles J Fleischmann",
    "Rohit Khanna",
]
_TRANSACTION_COLUMNS = (
    "member",
    "ticker",
    "transaction_type",
    "transaction_date",
    "disclosure_date",
    "amount_midpoint",
    "amount_raw",
    "source",
    "instrument_type",
    "source_record_id",
    "amends_source_record_id",
    "ticker_origin",
    "raw_asset_class",
    "asset_description",
    "raw_asset_description",
)


class _MemoryDB:
    def __init__(self, transactions, prices):
        self.conn = duckdb.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)"
        )
        self.conn.executemany("INSERT INTO prices VALUES (?, ?, ?)", prices)
        self.conn.execute(
            """
            CREATE TABLE canonical_transactions (
                member VARCHAR, ticker VARCHAR, transaction_type VARCHAR,
                transaction_date DATE, disclosure_date DATE,
                amount_midpoint DOUBLE, amount_raw VARCHAR, source VARCHAR,
                instrument_type VARCHAR, source_record_id VARCHAR,
                amends_source_record_id VARCHAR, ticker_origin VARCHAR,
                raw_asset_class VARCHAR, asset_description VARCHAR,
                raw_asset_description VARCHAR
            )
            """
        )
        self.conn.executemany(
            "INSERT INTO canonical_transactions VALUES (" + ", ".join("?" * 15) + ")",
            [
                tuple(row[column] for column in _TRANSACTION_COLUMNS)
                for row in transactions
            ],
        )

    def get_transactions_by_date_range(self, start_date, end_date):
        return self.conn.execute(
            """
            SELECT * FROM canonical_transactions
            WHERE disclosure_date BETWEEN ? AND ?
            ORDER BY disclosure_date DESC
            """,
            [start_date, end_date],
        ).fetchdf()


def _scenario():
    dates = pd.date_range("2026-03-02", "2026-06-30", freq="B")
    close_by_date = {
        day.date(): 190.0 - index * 0.45 for index, day in enumerate(dates)
    }
    prices = [("ZTS", day, close) for day, close in close_by_date.items()]

    transactions = []

    def add_trade(member, transaction_type, day, disclosure, amount, amount_raw):
        transactions.append(
            {
                "member": member,
                "ticker": "ZTS",
                "transaction_type": transaction_type,
                "transaction_date": day,
                "disclosure_date": disclosure,
                "amount_midpoint": amount,
                "amount_raw": amount_raw,
                "source": "house_pdf",
                "instrument_type": None,
                "source_record_id": f"{member}-{day}-{transaction_type}",
                "amends_source_record_id": None,
                "ticker_origin": None,
                "raw_asset_class": None,
                "asset_description": None,
                "raw_asset_description": None,
            }
        )

    buys = [
        (
            "Byron Donalds",
            date(2026, 6, 9),
            date(2026, 6, 18),
            25000,
            "$15,001 - $50,000",
        ),
        (
            "Gilbert Cisneros",
            date(2026, 6, 16),
            date(2026, 6, 23),
            10000,
            "$1,001 - $15,000",
        ),
        (
            "Charles J Fleischmann",
            date(2026, 6, 9),
            date(2026, 6, 18),
            32500,
            "$15,001 - $50,000",
        ),
        (
            "Rohit Khanna",
            date(2026, 4, 9),
            date(2026, 4, 24),
            15000,
            "$15,001 - $50,000",
        ),
        (
            "Rohit Khanna",
            date(2026, 5, 12),
            date(2026, 5, 22),
            15000,
            "$15,001 - $50,000",
        ),
        (
            "Rohit Khanna",
            date(2026, 6, 9),
            date(2026, 6, 18),
            15000,
            "$15,001 - $50,000",
        ),
    ]
    for member, day, disclosure, amount, amount_raw in buys:
        add_trade(member, "Purchase", day, disclosure, amount, amount_raw)

    khanna_buys = [row for row in buys if row[0] == "Rohit Khanna"]
    khanna_sales = []
    for member, buy_day, _, amount, amount_raw in khanna_buys:
        sell_day = buy_day + timedelta(days=20)
        sell_price = close_by_date[sell_day]
        buy_price = close_by_date[buy_day]
        sale_amount = amount / buy_price * sell_price * 1.01
        add_trade(
            member,
            "Sale",
            sell_day,
            min(sell_day + timedelta(days=7), AS_OF),
            sale_amount,
            amount_raw,
        )
        khanna_sales.append(sell_day)

    return _MemoryDB(transactions, prices), close_by_date, khanna_buys, khanna_sales


def test_zts_june_setup_replays_and_respects_point_in_time(monkeypatch):
    db, close_by_date, khanna_buys, khanna_sales = _scenario()
    scored_candidates = {}
    june_actors = {actors.actor_id("congress", member) for member in MEMBERS[:3]}

    def capture_score(candidates, weights, current_prices):
        decision_date = pd.Timestamp(candidates.iloc[0]["as_of"]).date()
        scored_candidates[decision_date] = candidates.copy()
        return setups.score(candidates, weights, current_prices)

    monkeypatch.setattr(
        replay,
        "_load_dependencies",
        lambda: (
            pipeline,
            positions,
            actors,
            SimpleNamespace(score=capture_score),
        ),
    )

    try:
        prices = list(close_by_date.values())
        assert prices[0] > prices[-1]
        assert all(left > right for left, right in pairwise(prices))
        assert all(
            0 < (sale - buy[1]).days <= 30
            for buy, sale in zip(khanna_buys, khanna_sales)
        )

        june_result = replay.replay(
            db, ["ZTS"], [AS_OF], horizon_days=90, members=MEMBERS
        ).iloc[0]
        candidates = scored_candidates[AS_OF]
        assert june_result["score"] > 0
        assert june_result["n_actors"] >= 3

        event_rows = candidates.loc[~candidates["position_evidence"]]
        expected_events = {
            (actors.actor_id("congress", "Byron Donalds"), date(2026, 6, 9)),
            (actors.actor_id("congress", "Gilbert Cisneros"), date(2026, 6, 16)),
            (
                actors.actor_id("congress", "Charles J Fleischmann"),
                date(2026, 6, 9),
            ),
        }
        expected_events.update(
            (actors.actor_id("congress", "Rohit Khanna"), row[1]) for row in khanna_buys
        )
        observed_events = {
            (row.actor_id, pd.Timestamp(row.event_date).date())
            for row in event_rows.itertuples()
        }
        assert observed_events == expected_events
        for row in event_rows.itertuples():
            day = pd.Timestamp(row.event_date).date()
            assert row.entry_ref == pytest.approx(close_by_date[day])

        khanna_id = actors.actor_id("congress", "Rohit Khanna")
        khanna_rows = event_rows.loc[event_rows["actor_id"] == khanna_id]
        assert set(pd.to_datetime(khanna_rows["event_date"]).dt.date) == {
            row[1] for row in khanna_buys
        }
        assert not candidates.loc[
            candidates["actor_id"] == khanna_id, "position_evidence"
        ].any()

        weights = {
            row.actor_id: actors.compute_weight(str(row.kind), 0, 0.0, str(row.source))
            for row in candidates.itertuples()
        }

        def actor_contribution(actor_id):
            ranked, _ = setups.score(
                candidates.loc[candidates["actor_id"] == actor_id],
                weights,
                {"ZTS": close_by_date[AS_OF]},
            )
            return float(ranked.iloc[0]["score"])

        holder_contribution = actor_contribution(
            actors.actor_id("congress", "Byron Donalds")
        )
        churner_contribution = actor_contribution(khanna_id)
        assert holder_contribution > churner_contribution

        corroboration = pd.DataFrame(
            [
                {
                    "ticker": "ZTS",
                    "actor_id": "manager:13F ONLY",
                    "kind": "manager",
                    "source": "13f",
                    "entry_ref": None,
                    "event_date": AS_OF,
                    "disclosure_date": AS_OF,
                    "corroboration": True,
                    "position_evidence": False,
                    "as_of": AS_OF,
                }
            ]
        )
        corroborated, _ = setups.score(
            corroboration,
            {"manager:13F ONLY": 0.6},
            {"ZTS": close_by_date[AS_OF]},
        )
        assert corroborated.iloc[0]["score"] == 0
        assert corroborated.iloc[0]["n_actors"] == 0

        early_result = replay.replay(
            db, ["ZTS"], [BEFORE_JUNE], horizon_days=90, members=MEMBERS
        ).iloc[0]
        early_candidates = scored_candidates[BEFORE_JUNE]
        assert not set(early_candidates["actor_id"]) & june_actors
        assert not (
            pd.to_datetime(early_candidates["event_date"]).dt.date > BEFORE_JUNE
        ).any()
        assert early_result["n_actors"] < 3
    finally:
        db.conn.close()
