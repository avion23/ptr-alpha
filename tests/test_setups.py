from datetime import date

import pandas as pd
import pytest

from analyzer.setups import is_ranked, score
from analyzer.positions import build_positions


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
    holding_state=None,
    shares_held=None,
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
        "holding_state": holding_state,
        "shares_held": shares_held,
        "as_of": AS_OF,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_score_never_ranks(value):
    assert not is_ranked(value, 1)
    assert not is_ranked(value, 2, corroborated=True)


def test_non_equity_markers_neither_rank_nor_contaminate_stock_basis():
    days = ["2024-05-28", "2024-05-29", "2024-05-30", "2024-05-31"]
    rows = [
        ("BTC", "Bitcoin [CT]", "2024-05-29", 10000.0),
        ("BILL", "US Treas Bill [GS]", "2024-05-29", 10000.0),
        ("UNIT", "UNITED STATES TREAS ... [GS]", "2024-05-29", 10000.0),
        ("ALGN", "GO Bonds [GS]", "2024-05-29", 10000.0),
        ("FNMA", "FEDERAL NATIONAL MORTGAGE ASSOCIATION POOL ... [CS]", "2024-05-29", 10000.0),
        ("TZA", "TZA ETF [OT]", "2024-05-29", 10000.0),
        # Neither a misbound BTC token nor a [ST]-marked crypto trust is equity.
        ("IBIT", "BTC [CT]", "2024-05-28", 10000.0),
        ("IBIT", "iShares Bitcoin Trust ETF (IBIT) [ST]", "2024-05-29", 10000.0),
        ("SPCX", "SpaceX pre-IPO purchase [OT]", "2024-05-29", 10000.0),
        ("SPCX", "SpaceX Class A Common Stock [ST]", "2024-05-29", 10000.0),
        ("AAPL", "Apple Common Stock", "2024-05-29", 10000.0),
    ]
    trades = pd.DataFrame(
        [
            {
                "member": "Nancy Pelosi",
                "ticker": ticker,
                "transaction_type": "Purchase",
                "transaction_date": day,
                "disclosure_date": day,
                "amount_midpoint": amount,
                "source": "form4",
                # Simulate pre-fix canonical rows, which all said stock.
                "instrument_type": "stock",
                "asset_description": description,
            }
            for ticker, description, day, amount in rows
        ]
    )
    histories = {
        ticker: pd.DataFrame(
            {"close": [200.0 if ticker == "IBIT" and day == days[0] else 100.0
                        for day in days]},
            index=pd.to_datetime(days),
        )
        for ticker in trades["ticker"].unique()
    }

    positions = build_positions(trades, histories, as_of=AS_OF).set_index("ticker")
    assert set(positions.index) == {"SPCX", "AAPL"}

    candidates = pd.DataFrame(
        [
            _row(
                ticker,
                actor_id="congress:PELOSI",
                entry_ref=position.cost_basis,
                event_date=position.last_activity,
                disclosure_date=position.disclosure_date,
                position_evidence=True,
            )
            for ticker, position in positions.iterrows()
        ]
    )
    ranked, _ = score(
        candidates,
        {"congress:PELOSI": 1.0},
        {ticker: 100.0 for ticker in positions.index},
        enable_13f_corroboration=False,
    )
    assert set(ranked["ticker"]) == {"SPCX", "AAPL"}


def test_finite_contributions_overflowing_the_sum_do_not_rank():
    candidates = pd.DataFrame(
        [_row(actor_id=actor, position_evidence=True) for actor in ("congress:A", "congress:B")]
    )
    ranked, blocked = score(
        candidates,
        {"congress:A": 1e308, "congress:B": 1e308},
        {"AAPL": (100.0, AS_OF)},
        enable_13f_corroboration=False,
    )

    assert ranked.empty
    assert blocked["reason"].tolist() == ["non-finite setup score"] * 2


def test_two_actors_outscore_one_and_corroboration_is_bounded():
    candidates = pd.DataFrame(
        [
            _row(actor_id="officer:BURKE CEO", event_date=date(2024, 3, 1), position_evidence=True),
            _row(actor_id="congress:PELOSI", event_date=date(2024, 3, 1), position_evidence=True),
            _row(
                actor_id="manager:FUND A",
                corroboration=True,
                source="13f",
                holding_state="held",
                shares_held=100,
            ),
            _row(
                actor_id="manager:FUND B",
                corroboration=True,
                source="13f",
                holding_state="held",
                shares_held=50,
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
        watchlist={"manager:FUND A", "manager:FUND B"},
        enable_13f_corroboration=True,
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


def test_old_position_evidence_is_blocked_not_merely_descriptive():
    stale = _row(
        actor_id="congress:OLD",
        position_evidence=True,
        disclosure_date=date(2020, 1, 1),
    )
    fresh = _row(
        actor_id="congress:NEW",
        position_evidence=True,
        disclosure_date=date(2024, 5, 30),
    )
    ranked, _ = score(
        pd.DataFrame([stale, fresh]),
        {"congress:OLD": 1.0, "congress:NEW": 1.0},
        {"AAPL": 100.0},
    )
    assert len(ranked) == 1
    assert ranked.iloc[0]["actors"] == ["congress:NEW"]
    assert ranked.iloc[0]["has_fresh_evidence"]
    ranked_stale, blocked_stale = score(
        pd.DataFrame([stale, _row(actor_id="congress:OLD2",
                                  position_evidence=True,
                                  disclosure_date=date(2020, 6, 1))]),
        {"congress:OLD": 1.0, "congress:OLD2": 1.0},
        {"AAPL": 100.0},
    )
    assert ranked_stale.empty
    assert blocked_stale["reason"].tolist() == [
        "evidence older than 2 months",
        "evidence older than 2 months",
    ]


def test_missing_weight_defaults_to_zero():
    ranked, blocked = score(pd.DataFrame([_row()]), {}, {"AAPL": 100.0})

    assert ranked.empty
    assert blocked.iloc[0]["reason"] == "no actor weight"


def test_preblocked_positive_entry_does_not_initiate_or_contaminate_other_rows():
    blocked_a = _row(
        actor_id="congress:A",
        ticker="AAPL",
        entry_ref=100.0,
        position_evidence=True,
    )
    blocked_a["blocked_reason"] = "upstream row restriction"
    usable_a = _row(
        actor_id="congress:A",
        ticker="AAPL",
        entry_ref=120.0,
        position_evidence=True,
    )
    usable_b = _row(
        actor_id="congress:B",
        ticker="AAPL",
        entry_ref=100.0,
        position_evidence=True,
    )
    blocked_msft = _row(
        actor_id="congress:C",
        ticker="MSFT",
        entry_ref=100.0,
        position_evidence=True,
    )
    blocked_msft["blocked_reason"] = "MSFT source restriction"
    audit = {}

    ranked, blocked = score(
        pd.DataFrame([blocked_a, usable_a, usable_b, blocked_msft]),
        {"congress:A": 0.6, "congress:B": 0.6, "congress:C": 1.0},
        {"AAPL": 100.0, "MSFT": None},
        audit=audit,
    )

    assert ranked["ticker"].tolist() == ["AAPL"]
    row = ranked.iloc[0]
    assert row["score"] == pytest.approx(1.2)
    assert row["actors"] == ["congress:A", "congress:B"]
    assert audit["AAPL"]["members"] == [
        {
            "member": "congress:A",
            "entry_ref": 120.0,
            "weight": 0.6,
            "contribution": 0.6,
        },
        {
            "member": "congress:B",
            "entry_ref": 100.0,
            "weight": 0.6,
            "contribution": 0.6,
        },
    ]
    assert blocked.set_index("ticker").loc["MSFT", "reason"] == (
        "MSFT source restriction"
    )


def test_preblocked_fresh_entry_does_not_refresh_old_position_evidence():
    stale = _row(
        actor_id="congress:A",
        disclosure_date=date(2024, 1, 1),
        position_evidence=True,
    )
    fresh_blocked = _row(
        actor_id="congress:A",
        disclosure_date=date(2024, 5, 30),
        position_evidence=False,
    )
    fresh_blocked["blocked_reason"] = "upstream row restriction"

    ranked, blocked = score(
        pd.DataFrame([stale, fresh_blocked]),
        {"congress:A": 1.0},
        {"AAPL": (100.0, AS_OF)},
    )

    assert ranked.empty
    assert blocked.to_dict("records") == [
        {
            "ticker": "AAPL",
            "actor_id": "congress:A",
            "reason": "upstream row restriction",
        }
    ]


@pytest.mark.parametrize("reason", [None, float("nan"), "", "   "])
def test_empty_block_reason_does_not_block_a_usable_entry(reason):
    candidate = _row(position_evidence=True)
    candidate["blocked_reason"] = reason

    ranked, blocked = score(
        pd.DataFrame([candidate]), {"congress:PELOSI": 1.0}, {"AAPL": 100.0}
    )

    assert ranked["ticker"].tolist() == ["AAPL"]
    assert blocked.empty


@pytest.mark.parametrize("enabled", [False, True])
def test_preblocked_positive_held_13f_never_corroborates(enabled):
    officer = _row(
        actor_id="officer:CEO",
        source="form4",
        position_evidence=True,
    )
    manager = _row(
        actor_id="manager:FUND A",
        source="13f",
        corroboration=True,
        holding_state="held",
        shares_held=100,
    )
    manager["blocked_reason"] = "upstream 13F restriction"

    ranked, blocked = score(
        pd.DataFrame([officer, manager]),
        {"officer:CEO": 0.6},
        {"AAPL": 100.0},
        watchlist={"manager:FUND A"},
        enable_13f_corroboration=enabled,
    )

    assert ranked.empty
    assert blocked.set_index("actor_id").loc[
        "officer:CEO", "reason"
    ] == "score below ranking floor (1.00)"


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
            "reasons": ["STRONG basis @100.00, now 100.00 (+0.0%)"],
            "min_disclosure_lag_days": 2,
            "has_fresh_evidence": True,
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


def test_epsilon_contribution_does_not_promote_another_actor():
    candidates = pd.DataFrame(
        [
            _row(actor_id="officer:REAL", position_evidence=True),
            _row(actor_id="congress:DUST", position_evidence=True),
        ]
    )

    ranked, blocked = score(
        candidates,
        {"officer:REAL": 0.52, "congress:DUST": 1e-15},
        {"AAPL": 100.0},
    )

    assert ranked.empty
    assert blocked.set_index("actor_id").loc[
        "congress:DUST", "reason"
    ] == "below usable evidence floor (0.05)"


def test_quote_older_than_five_calendar_days_is_blocked():
    candidates = pd.DataFrame([_row(position_evidence=True)])

    ranked, blocked = score(
        candidates,
        {"congress:PELOSI": 1.0},
        {"AAPL": (100.0, date(2024, 5, 26))},
    )

    assert ranked.empty
    assert blocked.to_dict("records") == [
        {"ticker": "AAPL", "actor_id": "congress:PELOSI", "reason": "stale quote"}
    ]

    fresh_ranked, _ = score(
        candidates,
        {"congress:PELOSI": 1.0},
        {"AAPL": (100.0, date(2024, 5, 27))},
    )
    assert fresh_ranked.iloc[0]["score"] == pytest.approx(1.0)


def test_future_quote_is_blocked_with_explicit_reason():
    audit = {}
    ranked, blocked = score(
        pd.DataFrame([_row(position_evidence=True)]),
        {"congress:PELOSI": 1.0},
        {"AAPL": (100.0, date(2024, 6, 2))},
        audit=audit,
    )

    assert ranked.empty
    assert blocked.to_dict("records") == [
        {"ticker": "AAPL", "actor_id": "congress:PELOSI", "reason": "future quote"}
    ]
    assert audit["AAPL"]["future_quote"] is True


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
                disclosure_date=date(2024, 4, 1),
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
            "PELOSI basis @169.11 (est.), now 141.02 (-16.6%)",
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


def test_within_window_event_decay_is_continuous_near_cutoff():
    day_61 = pd.DataFrame(
        [
            _row(event_date=date(2024, 4, 1), disclosure_date=date(2024, 4, 1)),
            _row(
                actor_id="congress:CRUZ",
                event_date=date(2024, 4, 1),
                disclosure_date=date(2024, 4, 1),
            ),
        ]
    )
    day_60 = pd.DataFrame(
        [
            _row(event_date=date(2024, 4, 2), disclosure_date=date(2024, 4, 2)),
            _row(
                actor_id="congress:CRUZ",
                event_date=date(2024, 4, 2),
                disclosure_date=date(2024, 4, 2),
            ),
        ]
    )
    # Both disclosures are within the calendar-month window ending 2024-06-01.
    weights = {"congress:PELOSI": 1.0, "congress:CRUZ": 1.0}
    s_61, _ = score(day_61, weights, {"AAPL": 100.0})
    s_60, _ = score(day_60, weights, {"AAPL": 100.0})
    assert abs(s_61.iloc[0]["score"] - s_60.iloc[0]["score"]) < 0.1


def test_position_evidence_holds_full_weight():
    row = _row(event_date=date(2023, 1, 1), disclosure_date=date(2024, 4, 1))
    row["position_evidence"] = True
    ranked, _ = score(
        pd.DataFrame([row]), {"congress:PELOSI": 1.0}, {"AAPL": 100.0}
    )
    assert ranked.iloc[0]["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("position_evidence", [False, True])
@pytest.mark.parametrize(
    ("disclosure_date", "eligible"),
    [(date(2024, 4, 1), True), (date(2024, 3, 31), False)],
)
def test_calendar_month_cutoff_gates_position_and_event_evidence(
    position_evidence, disclosure_date, eligible
):
    candidates = pd.DataFrame(
        [
            _row(
                actor_id="officer:A",
                disclosure_date=disclosure_date,
                position_evidence=position_evidence,
            )
        ]
    )
    ranked, blocked = score(candidates, {"officer:A": 2.0}, {"AAPL": 100.0})

    if eligible:
        assert ranked.iloc[0]["score"] == pytest.approx(
            2.0 if position_evidence else 2.0 * 0.5 ** (61 / 90)
        )
        assert blocked.empty
    else:
        assert ranked.empty
        assert blocked.to_dict("records") == [
            {
                "ticker": "AAPL",
                "actor_id": "officer:A",
                "reason": "evidence older than 2 months",
            }
        ]


def test_evidence_cutoff_uses_historical_as_of_date():
    row = _row(
        actor_id="officer:HISTORICAL",
        disclosure_date=date(2026, 1, 9),
        position_evidence=True,
    )
    row["as_of"] = date(2026, 3, 10)
    ranked, blocked = score(
        pd.DataFrame([row]),
        {"officer:HISTORICAL": 1.0},
        {"AAPL": (100.0, date(2026, 3, 10))},
    )
    assert ranked.empty
    assert blocked.iloc[0]["reason"] == "evidence older than 2 months"

    row["disclosure_date"] = date(2026, 1, 10)
    ranked, blocked = score(
        pd.DataFrame([row]),
        {"officer:HISTORICAL": 1.0},
        {"AAPL": (100.0, date(2026, 3, 10))},
    )
    assert ranked.iloc[0]["score"] == pytest.approx(1.0)
    assert blocked.empty


def test_recent_event_evidence_keeps_90_day_half_life_inside_window():
    disclosure_date = date(2024, 4, 2)  # 60 days before AS_OF; cutoff is Apr 1.
    candidates = pd.DataFrame(
        [
            _row(actor_id=actor, disclosure_date=disclosure_date)
            for actor in ("officer:A", "officer:B")
        ]
    )
    ranked, blocked = score(
        candidates,
        {"officer:A": 1.0, "officer:B": 1.0},
        {"AAPL": 100.0},
    )
    assert ranked.iloc[0]["score"] == pytest.approx(2 * 0.5 ** (60 / 90))
    assert ranked.iloc[0]["actors"] == ["officer:A", "officer:B"]
    assert blocked.empty


def test_reason_verb_labels_basis_for_positions_and_buy_for_events():
    """position_evidence rows report the FIFO cost basis, not a trade price."""
    ranked, _ = score(
        pd.DataFrame(
            [
                _row(actor_id="congress:HOLDING", position_evidence=True),
                _row(actor_id="officer:EVENT", source="form4"),
            ]
        ),
        {"congress:HOLDING": 1.0, "officer:EVENT": 1.0},
        {"AAPL": 100.0},
    )

    assert ranked.iloc[0]["reasons"] == [
        "HOLDING basis @100.00 (est.), now 100.00 (+0.0%)",
        "EVENT buy @100.00, now 100.00 (+0.0%)",
    ]


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
            _row(
                actor_id="manager:FUND A", corroboration=True, source="13f",
                event_date=date(2024, 3, 31), disclosure_date=date(2024, 5, 15),
                holding_state="held", shares_held=120,
            ),
        ]
    )
    weights = {"officer:BURKE CEO": 0.6}
    watchlist = {"manager:FUND A"}

    ranked_lone, _ = score(lone, weights, {"AAPL": 100.0}, watchlist=watchlist)
    assert ranked_lone.empty

    audit = {}
    ranked_backed, _ = score(
        backed,
        weights,
        {"AAPL": 100.0},
        watchlist=watchlist,
        enable_13f_corroboration=True,
        audit=audit,
    )
    assert len(ranked_backed) == 1
    assert ranked_backed.iloc[0]["actors"] == ["officer:BURKE CEO"]
    assert ranked_backed.iloc[0]["score"] == pytest.approx(0.9)
    assert audit["AAPL"]["corroborated_13f"] is True


def test_environment_opt_in_enables_13f_but_defaults_off(monkeypatch):
    backed = pd.DataFrame(
        [
            _row(
                actor_id="officer:PTR MEMBER",
                source="form4",
                entry_ref=100.0,
                disclosure_date=AS_OF,
            ),
            _row(
                actor_id="manager:FUND A",
                source="13f",
                corroboration=True,
                event_date=date(2024, 3, 31),
                disclosure_date=AS_OF,
                holding_state="held",
                shares_held=80,
            ),
        ]
    )
    weights = {"officer:PTR MEMBER": 0.6}
    prices = {"AAPL": 100.0}
    monkeypatch.delenv("INSIDER_ENABLE_13F_CORROBORATION", raising=False)

    default_ranked, _ = score(
        backed, weights, prices, watchlist={"manager:FUND A"}
    )
    assert default_ranked.empty

    monkeypatch.setenv("INSIDER_ENABLE_13F_CORROBORATION", "1")
    audit = {}
    opt_in_ranked, _ = score(
        backed, weights, prices, watchlist={"manager:FUND A"}, audit=audit
    )
    assert opt_in_ranked.iloc[0]["score"] == pytest.approx(0.9)
    assert audit["AAPL"]["13f_corroboration_enabled"] is True


def test_non_watchlist_corroboration_does_not_promote():
    backed = pd.DataFrame(
        [
            _row(
                actor_id="officer:BURKE CEO",
                source="form4",
                event_date=date(2024, 3, 1),
                position_evidence=True,
            ),
            _row(actor_id="manager:RANDOM FUND", corroboration=True, source="13f",
                 holding_state="held", shares_held=100),
        ]
    )
    ranked, _ = score(
        backed,
        {"officer:BURKE CEO": 0.6},
        {"AAPL": 100.0},
        watchlist={"manager:FUND A"},
        enable_13f_corroboration=True,
    )
    assert ranked.empty


def test_stale_corroboration_does_not_promote():
    backed = pd.DataFrame(
        [
            _row(
                actor_id="officer:BURKE CEO",
                source="form4",
                event_date=date(2024, 3, 1),
                position_evidence=True,
            ),
            _row(
                actor_id="manager:FUND A",
                corroboration=True,
                source="13f",
                disclosure_date=date(2023, 1, 1),
                holding_state="held",
                shares_held=100,
            ),
        ]
    )
    ranked, _ = score(
        backed,
        {"officer:BURKE CEO": 0.6},
        {"AAPL": 100.0},
        watchlist={"manager:FUND A"},
        enable_13f_corroboration=True,
    )
    assert ranked.empty


def test_missing_watchlist_means_no_corroboration():
    backed = pd.DataFrame(
        [
            _row(
                actor_id="officer:BURKE CEO",
                source="form4",
                event_date=date(2024, 3, 1),
                position_evidence=True,
            ),
            _row(actor_id="manager:FUND A", corroboration=True, source="13f",
                 holding_state="held", shares_held=100),
        ]
    )
    ranked, _ = score(backed, {"officer:BURKE CEO": 0.6}, {"AAPL": 100.0})
    assert ranked.empty


def test_closed_manager_does_not_corroborate_lone_officer():
    officer = _row(
        actor_id="officer:BURKE CEO",
        source="form4",
        position_evidence=True,
    )
    manager = _row(
        actor_id="manager:FUND A",
        corroboration=True,
        source="13f",
        holding_state="held",
        shares_held=100,
    )
    manager["blocked_reason"] = (
        "corroboration unavailable: no current 13F share snapshot"
    )

    ranked, blocked = score(
        pd.DataFrame([officer, manager]),
        {"officer:BURKE CEO": 0.30},
        {"AAPL": 100.0},
        watchlist={"manager:FUND A"},
        enable_13f_corroboration=True,
    )

    assert ranked.empty
    assert blocked.set_index("actor_id").loc[
        "officer:BURKE CEO", "reason"
    ] == "score below ranking floor (1.00)"


@pytest.mark.parametrize(
    ("entry_ref", "current"),
    [
        (0, 100),
        (-1, 100),
        (float("inf"), 100),
        (100, 0),
        (100, float("inf")),
        (100, -1),
        (0, 0),
    ],
)
def test_invalid_price_domains_do_not_rank_or_crash(entry_ref, current):
    ranked, _ = score(
        pd.DataFrame([_row(entry_ref=entry_ref, position_evidence=True)]),
        {"congress:PELOSI": 1.0},
        {"AAPL": current},
    )

    assert ranked.empty
