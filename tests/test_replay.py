from datetime import date
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from analyzer import replay as replay_module

AS_OF = date(2024, 1, 10)


def _events(disclosure=AS_OF):
    return pd.DataFrame(
        [
            {
                "ticker": "ABC",
                "actor_id": "congress:ALICE",
                "kind": "congress",
                "source": "house_pdf",
                "entry_ref": 90.0,
                "event_date": date(2024, 1, 5),
                "disclosure_date": disclosure,
                "corroboration": False,
                "as_of": AS_OF,
            }
        ]
    )


def _install_scoring(monkeypatch, events):
    ranked = pd.DataFrame(
        [{"ticker": "ABC", "score": 2.5, "actors": ["congress:ALICE"],
          "n_actors": 1, "current": 100.0, "reasons": ["fixture"]}]
    )
    blocked = pd.DataFrame(columns=["ticker", "actor_id", "reason"])
    monkeypatch.setattr(
        replay_module,
        "_load_dependencies",
        lambda: (
            SimpleNamespace(eligible_events=lambda db, as_of, **kw: events),
            SimpleNamespace(
                holdings_candidates=lambda db, member, as_of, **kw: pd.DataFrame(
                    columns=["ticker", "actor_id", "kind", "source", "entry_ref",
                             "event_date", "disclosure_date", "corroboration",
                             "as_of"]
                )
            ),
            SimpleNamespace(
                compute_weight=lambda kind, n, hits, source: 0.6
            ),
            SimpleNamespace(score=lambda candidates, weights, prices: (ranked, blocked)),
        ),
    )


def _price_db(rows):
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE prices (ticker VARCHAR, date DATE, close DOUBLE)")
    if rows:
        db.executemany("INSERT INTO prices VALUES (?, ?, ?)", rows)
    return db


def test_replay_schema_and_missing_outcome_is_none(monkeypatch):
    _install_scoring(monkeypatch, _events())
    db = _price_db([("ABC", AS_OF, 100.0), ("SPY", AS_OF, 400.0)])

    result = replay_module.replay(db, ["ABC"], [AS_OF], members=[], horizon_days=30)

    assert list(result.columns) == [
        "ticker",
        "as_of",
        "score",
        "n_actors",
        "fwd_return_pct",
        "spy_return_pct",
        "excess_pct",
        "blocked",
    ]
    assert result.loc[0, "score"] == 2.5
    assert result.loc[0, "n_actors"] == 1
    assert result.loc[0, "fwd_return_pct"] is None
    assert result.loc[0, "excess_pct"] is None


def test_future_disclosure_never_reaches_scoring(monkeypatch):
    _install_scoring(monkeypatch, _events(disclosure=date(2024, 1, 11)))
    dependencies = replay_module._load_dependencies()
    dependencies[-1].score = lambda *_: pytest.fail("future event reached scoring")
    db = _price_db([])

    with pytest.raises(replay_module.ReplayError, match="disclosure after"):
        replay_module.replay(db, ["ABC"], [AS_OF], members=[])


def test_future_prices_are_not_used_for_decision(monkeypatch):
    _install_scoring(monkeypatch, _events())
    db = _price_db(
        [
            ("ABC", AS_OF, 100.0),
            ("ABC", date(2024, 1, 11), 110.0),
            ("SPY", AS_OF, 400.0),
            ("SPY", date(2024, 1, 11), 404.0),
        ]
    )

    result = replay_module.replay(db, ["ABC"], [AS_OF], members=[], horizon_days=30)

    assert result.loc[0, "fwd_return_pct"] == pytest.approx(10.0)
    assert result.loc[0, "spy_return_pct"] == pytest.approx(1.0)
    assert result.loc[0, "excess_pct"] == pytest.approx(9.0)
