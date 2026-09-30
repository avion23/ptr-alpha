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
    position_evidence=False,
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
        "position_evidence": position_evidence,
        "as_of": AS_OF,
    }


def test_two_actors_outscore_one_and_corroboration_is_bounded():
    candidates = pd.DataFrame(
        [
            _row(actor_id="officer:BURKE CEO", event_date=date(2024, 3, 1), position_evidence=True),
            _row(actor_id="congress:PELOSI", event_date=date(2024, 3, 1), position_evidence=True),
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
                position_evidence=True,
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

    assert ranked.empty
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

    assert ranked.empty
    assert blocked.iloc[0]["actor_id"] == "congress:PELOSI"
    assert blocked.iloc[0]["reason"] == "above entry tolerance"


def test_missing_weight_defaults_to_zero():
    ranked, blocked = score(pd.DataFrame([_row()]), {}, {"AAPL": 100.0})

    assert ranked.empty
    assert blocked.iloc[0]["reason"] == "no actor weight"


def test_zero_weight_actor_does_not_corroborate_or_count():
    candidates = pd.DataFrame(
        [
            _row(actor_id="congress:WEAK", position_evidence=True),
            _row(actor_id="officer:ZERO"),
            _row(
                ticker="MSFT",
                actor_id="officer:STRONG",
                position_evidence=True,
            ),
            _row(ticker="MSFT", actor_id="officer:ZERO"),
        ]
    )
    ranked, blocked = score(
        candidates,
        {"congress:WEAK": 0.52, "officer:STRONG": 1.0, "officer:ZERO": 0.0},
        {"AAPL": 100.0, "MSFT": 100.0},
    )

    assert ranked.to_dict("records") == [
        {
            "ticker": "MSFT",
            "score": 1.0,
            "actors": ["officer:STRONG"],
            "n_actors": 1,
            "current": 100.0,
            "reasons": ["STRONG buy @100.00, now 100.00 (+0.0%)"],
        }
    ]
    assert blocked.to_dict("records") == [
        {
            "ticker": "AAPL",
            "actor_id": "officer:ZERO",
            "reason": "no actor weight",
        },
        {
            "ticker": "AAPL",
            "actor_id": "congress:WEAK",
            "reason": "score below ranking floor (1.00)",
        },
        {
            "ticker": "MSFT",
            "actor_id": "officer:ZERO",
            "reason": "no actor weight",
        },
    ]


def test_missing_entry_reference_is_blocked():
    ranked, blocked = score(
        pd.DataFrame([_row(entry_ref=None)]),
        {},
        {"AAPL": 100.0},
    )

    assert ranked.empty
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
                position_evidence=True,
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
            "PELOSI buy @169.11 (est.), now 141.02 (-16.6%)",
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
            ),
            _row(
                actor_id="congress:CRUZ",
                event_date=date(2024, 5, 31),
                disclosure_date=date(2024, 5, 31),
            ),
        ]
    )

    ranked, _ = score(
        candidates,
        {"congress:PELOSI": 1.0, "congress:CRUZ": 1.0},
        {"AAPL": 100.0},
    )

    assert ranked.iloc[0]["score"] == pytest.approx(2 * 0.5 ** (1 / 90))


def test_no_decay_cliff_between_day_60_and_61():
    old = pd.DataFrame(
        [
            _row(event_date=date(2024, 3, 31), disclosure_date=date(2024, 3, 31)),
            _row(
                actor_id="congress:CRUZ",
                event_date=date(2024, 3, 31),
                disclosure_date=date(2024, 3, 31),
            ),
        ]
    )
    new = pd.DataFrame(
        [
            _row(event_date=date(2024, 4, 1), disclosure_date=date(2024, 4, 1)),
            _row(
                actor_id="congress:CRUZ",
                event_date=date(2024, 4, 1),
                disclosure_date=date(2024, 4, 1),
            ),
        ]
    )
    # as_of 2024-06-01: disclosures 62 vs 61 days old; continuous decay
    # must not jump the way a 60-day position-evidence switch did.
    weights = {"congress:PELOSI": 1.0, "congress:CRUZ": 1.0}
    s_old, _ = score(old, weights, {"AAPL": 100.0})
    s_new, _ = score(new, weights, {"AAPL": 100.0})
    assert abs(s_old.iloc[0]["score"] - s_new.iloc[0]["score"]) < 0.1


def test_position_evidence_holds_full_weight():
    row = _row(event_date=date(2023, 1, 1), disclosure_date=date(2023, 1, 5))
    row["position_evidence"] = True
    ranked, _ = score(
        pd.DataFrame([row]), {"congress:PELOSI": 1.0}, {"AAPL": 100.0}
    )
    assert ranked.iloc[0]["score"] == pytest.approx(1.0)


def test_officer_reference_is_not_labeled_estimate():
    ranked, _ = score(
        pd.DataFrame(
            [
                _row(actor_id="officer:BURKE CEO", source="form4"),
                _row(actor_id="officer:OTHER CEO", source="form4"),
            ]
        ),
        {"officer:BURKE CEO": 1.0, "officer:OTHER CEO": 1.0},
        {"AAPL": 100.0},
    )
    assert all("(est.)" not in reason for reason in ranked.iloc[0]["reasons"])


def test_corroborated_lone_initiator_ranks_but_uncorroborated_does_not():
    lone = pd.DataFrame(
        [
            _row(
                actor_id="officer:BURKE CEO",
                source="form4",
                event_date=date(2024, 3, 1),
                position_evidence=True,
            ),
        ]
    )
    backed = pd.DataFrame(
        [
            _row(
                actor_id="officer:BURKE CEO",
                source="form4",
                event_date=date(2024, 3, 1),
                position_evidence=True,
            ),
            _row(actor_id="manager:FUND A", corroboration=True, source="13f"),
        ]
    )
    weights = {"officer:BURKE CEO": 0.6}

    ranked_lone, _ = score(lone, weights, {"AAPL": 100.0})
    assert ranked_lone.empty

    ranked_backed, _ = score(backed, weights, {"AAPL": 100.0})
    assert len(ranked_backed) == 1
    assert ranked_backed.iloc[0]["actors"] == ["officer:BURKE CEO"]
