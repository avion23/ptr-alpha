from datetime import date

import numpy as np
import pandas as pd
import pytest

from analyzer.persistence.price_repository import next_nyse_session, previous_nyse_session
from scripts.evaluate_rule import (
    _cluster_stats,
    _current_rule_refs,
    build_drop_report,
    assert_no_future_rows,
    evaluation_verdict,
    evaluate_synthetic_cohort,
)


def test_leakage_check_rejects_planted_future_input():
    frame = pd.DataFrame(
        {"disclosure_date": ["2025-01-02"], "event_date": ["2025-01-03"]}
    )
    with pytest.raises(ValueError, match="look-ahead: event_date"):
        assert_no_future_rows(frame, date(2025, 1, 2))


def test_eligibility_reference_uses_live_age_and_quarantine_gates():
    frame = pd.DataFrame(
        {
            "entry_ref": [100.0, 150.0, 200.0],
            "disclosure_date": ["2024-12-01", "2024-10-01", "2024-12-01"],
            "blocked_reason": [None, None, "quarantined marker"],
            "corroboration": [False, False, False],
        }
    )
    assert _current_rule_refs(frame, date(2025, 1, 15)).tolist() == [True, False, False]


def test_cluster_stats_use_decision_date_as_independent_unit():
    frame = pd.DataFrame(
        {
            "as_of": ["2025-01-15", "2025-01-15", "2025-04-15", "2025-04-15"],
            "excess": [0.0, 10.0, 10.0, 10.0],
        }
    )
    result = _cluster_stats(frame)
    assert result["clusters"] == 2
    assert result["n_ticker_dates"] == 4
    assert result["mean_cluster_pct"] == pytest.approx(7.5)
    assert result["date_clustered_t"] == pytest.approx(3.0)


def test_verdict_distinguishes_insufficient_clusters_from_failed_test():
    assert evaluation_verdict({"clusters": 1, "mean_cluster_pct": 3.0}) == (
        "NOT VALIDATABLE with available certified history: 1 usable cluster(s)"
    )
    assert evaluation_verdict(
        {"clusters": 5, "mean_cluster_pct": 1.0, "ci95_low_pct": 0.1}
    ) == "PASS"
    assert evaluation_verdict(
        {"clusters": 5, "mean_cluster_pct": 1.0, "ci95_low_pct": -0.1}
    ) == "NOT VALIDATED"


def test_drop_report_accounts_for_each_exclusion_reason():
    cohorts = pd.DataFrame(
        [
            {
                "as_of": "2025-01-15", "cohort_group": "eligibility_unknown",
                "ticker": "AAA", "actor_id": "congress:a",
                "drop_unquotable_reason": "no_valid_entry_reference",
                "outcome_status": "available", "exclusion_reason": None,
            },
            {
                "as_of": "2025-01-15", "cohort_group": "eligibility_unknown",
                "ticker": "BBB", "actor_id": "congress:b",
                "drop_unquotable_reason": None,
                "outcome_status": "uncertifiable", "exclusion_reason": "BBB:generation_unknown",
            },
            {
                "as_of": "2025-01-15", "cohort_group": "ineligible",
                "ticker": "CCC", "actor_id": "congress:c",
                "drop_unquotable_reason": None,
                "outcome_status": "outcome_missing", "exclusion_reason": "exit_quote_unavailable",
            },
        ]
    )
    report = build_drop_report(cohorts)
    counts = report.set_index(["drop_class", "reason"])["n_ticker_actor"].to_dict()
    assert counts == {
        ("outcome_missing", "exit_quote_unavailable"): 1,
        ("uncertifiable", "BBB:generation_unknown"): 1,
        ("unquotable", "no_valid_entry_reference"): 1,
    }


def test_tiny_cohort_matches_production_costed_excess_hand_calculation():
    decision = date(2025, 1, 3)
    entry = next_nyse_session(pd.Timestamp(decision)).date()
    exit_day = previous_nyse_session(
        pd.Timestamp(entry) + pd.Timedelta(days=90)
    ).date()
    candidates = pd.DataFrame(
        [
            {
                "ticker": "AAA", "actor_id": f"congress:{actor}",
                "kind": "congress", "source": "house_pdf", "entry_ref": 100.0,
                "as_of": decision, "disclosure_date": date(2024, 12, 20),
                "event_date": date(2024, 12, 10), "corroboration": False,
                "position_evidence": False, "blocked_reason": None,
            }
            for actor in ("alpha", "beta")
        ]
    )
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0], "SPY": [500.0, 525.0]},
        index=pd.to_datetime([entry, exit_day]),
    )
    actor_rows, ticker_rows = evaluate_synthetic_cohort(
        candidates,
        {"congress:alpha": 0.52, "congress:beta": 0.52},
        {"AAA": (100.0, decision)},
        prices,
        decision,
    )

    stock = round(((110.0 * 0.999) / (100.0 * 1.001) - 1) * 100, 2)
    spy = round(((525.0 * 0.999) / (500.0 * 1.001) - 1) * 100, 2)
    expected_excess = round(stock - spy, 2)
    assert len(actor_rows) == 2
    assert len(ticker_rows) == 1
    assert actor_rows["excess"].tolist() == [expected_excess, expected_excess]
    assert actor_rows["cohort_group"].tolist() == ["ranked_eligible", "ranked_eligible"]
    assert np.isfinite(actor_rows["excess"].astype(float)).all()
