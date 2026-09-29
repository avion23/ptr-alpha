from datetime import date

import pandas as pd
import pytest

from analyzer.setups import score


AS_OF = date(2024, 6, 1)


def _row(
    ticker="AAPL",
    actor_id="congress:PELOSI",
    *,
    entry_ref=100.0,
    event_date=date(2024, 5, 30),
    disclosure_date=date(2024, 5, 30),
    corroboration=False,
    source="house",
):
    return {
        "ticker": ticker,
        "actor_id": actor_id,
        "kind": actor_id.split(":", 1)[0],
        "source": source,
        "entry_ref": entry_ref,
        "event_date": event_date,
        "disclosure_date": disclosure_date,
        "corroboration": corroboration,
        "as_of": AS_OF,
    }


def test_two_actors_outscore_one_and_corroboration_is_bounded():
    candidates = pd.DataFrame(
        [
            _row(actor_id="officer:BURKE CEO", event_date=date(2024, 3, 1)),
            _row(actor_id="congress:PELOSI", event_date=date(2024, 3, 1)),
            _row(
                actor_id="manager:FUND A",
                corroboration=True,
                source="13f",
            ),
            _row(
                actor_id="manager:FUND B",
                corroboration=True,
                source="13f",
            ),
            _row(
                ticker="MSFT",
                actor_id="congress:SMITH",
                event_date=date(2024, 3, 1),
            ),
        ]
    )

    ranked, blocked = score(
        candidates,
        {"officer:BURKE CEO": 1.0, "congress:PELOSI": 1.0, "congress:SMITH": 1.0},
        {"AAPL": 100.0, "MSFT": 100.0},
    )

    assert ranked.set_index("ticker").loc["AAPL", "score"] == pytest.approx(3.0)
    assert ranked.set_index("ticker").loc["MSFT", "score"] == pytest.approx(1.0)
    assert blocked["reason"].tolist() == ["corroboration only", "corroboration only"]


def test_corroboration_only_has_zero_score_and_is_blocked():
    ranked, blocked = score(
        pd.DataFrame([_row(corroboration=True, source="13f")]),
        {},
        {"AAPL": 100.0},
    )

    assert ranked.iloc[0]["score"] == 0.0
    assert ranked.iloc[0]["actors"] == []
    assert blocked.to_dict("records") == [
        {
            "ticker": "AAPL",
            "actor_id": "congress:PELOSI",
            "reason": "corroboration only",
        }
    ]


def test_over_tolerance_actor_is_blocked_and_not_counted():
    ranked, blocked = score(
        pd.DataFrame([_row()]),
        {"congress:PELOSI": 2.0},
        {"AAPL": 106.0},
    )

    assert ranked.iloc[0]["score"] == 0.0
    assert blocked.iloc[0]["actor_id"] == "congress:PELOSI"
    assert blocked.iloc[0]["reason"] == "above entry tolerance"


def test_missing_weight_defaults_to_zero():
    ranked, _ = score(pd.DataFrame([_row()]), {}, {"AAPL": 100.0})

    assert ranked.iloc[0]["score"] == 0.0
    assert ranked.iloc[0]["actors"] == ["congress:PELOSI"]


def test_missing_entry_reference_is_blocked():
    ranked, blocked = score(
        pd.DataFrame([_row(entry_ref=None)]),
        {},
        {"AAPL": 100.0},
    )

    assert ranked.iloc[0]["score"] == 0.0
    assert blocked.iloc[0]["reason"] == "no entry reference"


def test_reasons_and_decay_are_deterministic():
    candidates = pd.DataFrame(
        [
            _row(actor_id="officer:BURKE CEO", entry_ref=135.0),
            _row(
                actor_id="congress:PELOSI",
                entry_ref=169.11,
                event_date=date(2024, 3, 1),
                disclosure_date=date(2024, 3, 1),
            ),
        ]
    )
    weights = {"officer:BURKE CEO": 1.0, "congress:PELOSI": 1.0}
    first, _ = score(candidates, weights, {"AAPL": 141.02})
    second, _ = score(candidates, weights, {"AAPL": 141.02})

    assert (
        first.iloc[0]["reasons"]
        == second.iloc[0]["reasons"]
        == [
            "PELOSI buy @169.11, now 141.02 (-16.6%)",
            "BURKE CEO buy @135.00, now 141.02 (+4.5%)",
        ]
    )
    assert first.iloc[0]["score"] == pytest.approx(1.0 + 0.5 ** (2 / 90))


def test_fresh_initiation_decays_from_disclosure_date():
    candidates = pd.DataFrame(
        [
            _row(
                event_date=date(2024, 5, 31),
                disclosure_date=date(2024, 5, 31),
            )
        ]
    )

    ranked, _ = score(candidates, {"congress:PELOSI": 1.0}, {"AAPL": 100.0})

    assert ranked.iloc[0]["score"] == pytest.approx(0.5 ** (1 / 90))
