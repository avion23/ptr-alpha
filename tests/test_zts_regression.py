from datetime import date
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from analyzer import actors, pipeline, positions, replay, setups

JUNE_30 = date(2026, 6, 30)
JULY_31 = date(2026, 7, 31)
AUGUST_3 = date(2026, 8, 3)
SEPTEMBER_4 = date(2026, 9, 4)
MEMBERS = [
    "Byron Donalds",
    "Gilbert Cisneros",
    "Charles J Fleischmann",
    "Rohit Khanna",
]

# Actual ZTS closes from data/congress.duckdb's prices table; only dates used
# for entry and as-of prices are included, not a generated price curve.
REAL_CLOSES = {
    date(2026, 5, 27): 78.9297866821289,
    date(2026, 6, 2): 75.86117553710938,
    date(2026, 6, 9): 81.63095092773438,
    date(2026, 6, 16): 78.78083038330078,
    date(2026, 6, 30): 71.36254119873047,
    date(2026, 7, 31): 77.29000091552734,
    AUGUST_3: 77.0999984741211,
}
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
            "INSERT INTO canonical_transactions VALUES ("
            + ", ".join("?" * len(_TRANSACTION_COLUMNS))
            + ")",
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


def _trade(
    member,
    transaction_date,
    disclosure_date,
    transaction_type,
    amount_midpoint,
    amount_raw,
    source,
    source_record_id,
):
    return {
        "member": member,
        "ticker": "ZTS",
        "transaction_type": transaction_type,
        "transaction_date": transaction_date,
        "disclosure_date": disclosure_date,
        "amount_midpoint": amount_midpoint,
        "amount_raw": amount_raw,
        "source": source,
        "instrument_type": None,
        "source_record_id": source_record_id,
        "amends_source_record_id": None,
        "ticker_origin": None,
        "raw_asset_class": None,
        "asset_description": "Zoetis Inc. (ZTS)",
        "raw_asset_description": "Zoetis Inc. (ZTS)",
    }


def _scenario(include_later_row):
    transactions = [
        # Real Rohit Khanna purchase from filing 9116142.
        _trade(
            "Rohit Khanna",
            date(2026, 5, 27),
            date(2026, 6, 9),
            "Purchase",
            8000.0,
            "A",
            "gemini_ocr",
            "9116142",
        ),
        # These three buys were disclosed in July, not June.
        _trade(
            "Gilbert Cisneros",
            date(2026, 6, 16),
            date(2026, 7, 2),
            "Purchase",
            8000.5,
            "$1,001 - $15,000",
            "house_pdf",
            "20034906",
        ),
        _trade(
            "Charles J Fleischmann",
            date(2026, 6, 9),
            date(2026, 7, 8),
            "Purchase",
            75000.0,
            "C",
            "gemini_ocr",
            "9116212",
        ),
        _trade(
            "Byron Donalds",
            date(2026, 6, 9),
            date(2026, 7, 15),
            "Purchase",
            8000.5,
            "$1,001 - $15,000",
            "house_pdf",
            "20034968",
        ),
    ]
    prices = [
        ("ZTS", day, close)
        for day, close in REAL_CLOSES.items()
        if include_later_row or day != AUGUST_3
    ]
    if include_later_row:
        # Real Rohit Khanna purchase: 2026-08-03, disclosed 2026-09-04.
        transactions.append(
            _trade(
                "Rohit Khanna",
                AUGUST_3,
                SEPTEMBER_4,
                "Purchase",
                8000.0,
                "A",
                "gemini_ocr",
                "9116328",
            )
        )
    return _MemoryDB(transactions, prices)


def test_replay_does_not_count_requested_mccaul_variant_twice():
    db = _MemoryDB(
        [
            _trade(
                "Michael T. McCaul",
                date(2026, 5, 27),
                date(2026, 6, 9),
                "Purchase",
                8000.0,
                "A",
                "gemini_ocr",
                "mccaul-buy",
            )
        ],
        [("ZTS", day, close) for day, close in REAL_CLOSES.items()],
    )
    try:
        event_only = replay.replay(
            db, ["ZTS"], [JUNE_30], horizon_days=90, members=[]
        ).iloc[0]
        requested_variant = replay.replay(
            db,
            ["ZTS"],
            [JUNE_30],
            horizon_days=90,
            members=["Michael McCaul"],
        ).iloc[0]
    finally:
        db.conn.close()

    assert event_only["n_actors"] == 1
    assert event_only["score"] == 0.0
    assert requested_variant["n_actors"] == 1
    assert requested_variant["score"] == event_only["score"]
    assert bool(requested_variant["blocked"])


def test_replay_coalesces_filed_name_variants_from_eligible_events():
    db = _MemoryDB(
        [
            _trade(
                "Michael T. McCaul",
                date(2026, 5, 27),
                date(2026, 6, 9),
                "Purchase",
                8000.0,
                "A",
                "gemini_ocr",
                "mccaul-full-buy",
            ),
            _trade(
                "Michael McCaul",
                date(2026, 6, 2),
                date(2026, 6, 16),
                "Purchase",
                8000.0,
                "A",
                "gemini_ocr",
                "mccaul-short-buy",
            ),
        ],
        [("ZTS", day, close) for day, close in REAL_CLOSES.items()],
    )
    try:
        events = pipeline.eligible_events(db, JUNE_30)
        result = replay.replay(
            db, ["ZTS"], [JUNE_30], horizon_days=90, members=[]
        ).iloc[0]
    finally:
        db.conn.close()

    assert events["actor_id"].nunique() == 2
    assert result["n_actors"] == 1
    assert bool(result["blocked"])


def test_zts_june_july_results_use_disclosure_dates_and_ignore_later_rows(
    monkeypatch,
):
    captured = {}
    ranked_by_snapshot = {}
    current_snapshot = None

    def capture_score(candidates, weights, current_prices):
        decision_date = pd.Timestamp(candidates.iloc[0]["as_of"]).date()
        key = (current_snapshot, decision_date)
        captured[key] = (candidates.copy(), current_prices.copy())
        ranked, blocked = setups.score(candidates, weights, current_prices)
        ranked_by_snapshot[key] = (ranked.copy(), blocked.copy())
        return ranked, blocked

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
    without_later_row = _scenario(include_later_row=False)
    with_later_row = _scenario(include_later_row=True)
    try:
        current_snapshot = "before"
        before = replay.replay(
            without_later_row,
            ["ZTS"],
            [JUNE_30, JULY_31],
            horizon_days=90,
            members=MEMBERS,
        )
        current_snapshot = "after"
        after = replay.replay(
            with_later_row,
            ["ZTS"],
            [JUNE_30, JULY_31],
            horizon_days=90,
            members=MEMBERS,
        )
        assert with_later_row.conn.execute(
            "SELECT close FROM prices WHERE ticker = 'ZTS' AND date = ?",
            [AUGUST_3],
        ).fetchone() == (REAL_CLOSES[AUGUST_3],)
    finally:
        without_later_row.conn.close()
        with_later_row.conn.close()

    # Adding the later purchase and its real close leaves both earlier replays
    # identical at the result-frame level.
    pd.testing.assert_frame_equal(before, after)
    expected_position_actors = {
        JUNE_30: {actors.actor_id("congress", "Rohit Khanna")},
        JULY_31: {
            actors.actor_id("congress", member)
            for member in MEMBERS
        },
    }
    expected_events = {
        JUNE_30: {
            (actors.actor_id("congress", "Rohit Khanna"), date(2026, 5, 27))
        },
        JULY_31: {
            (actors.actor_id("congress", "Rohit Khanna"), date(2026, 5, 27)),
            (actors.actor_id("congress", "Gilbert Cisneros"), date(2026, 6, 16)),
            (
                actors.actor_id("congress", "Charles J Fleischmann"),
                date(2026, 6, 9),
            ),
            (actors.actor_id("congress", "Byron Donalds"), date(2026, 6, 9)),
        },
    }
    expected_current = {
        JUNE_30: REAL_CLOSES[JUNE_30],
        JULY_31: REAL_CLOSES[JULY_31],
    }
    for decision_date in (JUNE_30, JULY_31):
        before_candidates, before_prices = captured[("before", decision_date)]
        after_candidates, after_prices = captured[("after", decision_date)]
        pd.testing.assert_frame_equal(before_candidates, after_candidates)
        assert before_prices == after_prices == {"ZTS": expected_current[decision_date]}
        disclosures = pd.to_datetime(after_candidates["disclosure_date"]).dt.date
        assert not (disclosures > decision_date).any()
        assert not (
            pd.to_datetime(after_candidates["event_date"]).dt.date == AUGUST_3
        ).any()
        assert not (disclosures == SEPTEMBER_4).any()

        position_rows = after_candidates.loc[after_candidates["position_evidence"]]
        assert set(position_rows["actor_id"]) == expected_position_actors[decision_date]
        event_rows = after_candidates.loc[~after_candidates["position_evidence"]]
        observed_events = {
            (row.actor_id, pd.Timestamp(row.event_date).date())
            for row in event_rows.itertuples()
        }
        assert observed_events == expected_events[decision_date]
        for row in event_rows.itertuples():
            event_date = pd.Timestamp(row.event_date).date()
            assert row.entry_ref == REAL_CLOSES[event_date]
        ranked, blocked = ranked_by_snapshot[("after", decision_date)]
        if decision_date == JUNE_30:
            # One weak actor: visible in blocked, never ranked.
            assert ranked.empty
            assert blocked["reason"].tolist() == [
                "score below ranking floor (1.00)"
            ]
        else:
            assert set(ranked.iloc[0]["actors"]) == expected_position_actors[
                decision_date
            ]

    june = after.loc[after["as_of"] == JUNE_30].iloc[0]
    assert june["n_actors"] == 1
    assert june["score"] == 0.0
    assert bool(june["blocked"])
    june_ranked, june_blocked = ranked_by_snapshot[("after", JUNE_30)]
    assert june_ranked.empty
    assert june_blocked["reason"].tolist() == [
        "score below ranking floor (1.00)"
    ]

    july = after.loc[after["as_of"] == JULY_31].iloc[0]
    assert july["n_actors"] == 4
    assert july["score"] == pytest.approx(2.08)
    assert not bool(july["blocked"])
    july_ranked, july_blocked = ranked_by_snapshot[("after", JULY_31)]
    assert july_ranked.iloc[0]["score"] == pytest.approx(2.08)
    assert july_blocked.empty


def test_fully_closed_holder_does_not_contribute():
    db = _MemoryDB(
        [
            _trade(
                "Sell-Now Trap",
                date(2026, 6, 2),
                date(2026, 6, 9),
                "Purchase",
                8000.5,
                "$1,001 - $15,000",
                "house_pdf",
                "test-buy",
            ),
            _trade(
                "Sell-Now Trap",
                date(2026, 6, 16),
                date(2026, 6, 23),
                "Sale Full",
                8000.5,
                "$1,001 - $15,000",
                "house_pdf",
                "test-full-sale",
            ),
        ],
        [("ZTS", day, close) for day, close in REAL_CLOSES.items() if day != AUGUST_3],
    )
    try:
        before_sale = positions.holdings_candidates(
            db, "Sell-Now Trap", date(2026, 6, 15)
        )
        assert len(before_sale) == 1
        assert before_sale.iloc[0]["entry_ref"] == pytest.approx(
            REAL_CLOSES[date(2026, 6, 2)]
        )

        candidates = positions.holdings_candidates(db, "Sell-Now Trap", JUNE_30)
        assert candidates.empty
        result = replay.replay(
            db,
            ["ZTS"],
            [JUNE_30],
            horizon_days=90,
            members=["Sell-Now Trap"],
        )
        assert result.iloc[0]["score"] == 0.0
        assert bool(result.iloc[0]["blocked"])
    finally:
        db.conn.close()
