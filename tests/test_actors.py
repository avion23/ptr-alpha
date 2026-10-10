import pytest
import pandas as pd

from analyzer.actors import (
    SOURCE_PRIORS,
    actor_id,
    age_weight,
    compute_actor_weights,
    compute_skill_table,
    compute_weight,
)


@pytest.mark.parametrize(
    ("source", "prior"),
    [(source, prior) for source, prior in SOURCE_PRIORS.items()],
)
def test_unknown_actor_gets_source_prior(source, prior):
    expected = 0.0 if prior is None else prior
    assert compute_weight("manager", 0, 0.0, source) == expected


def test_13f_never_initiates_even_with_matured_hits():
    assert compute_weight("manager", 10, 10.0, "13f") == 0.0


def test_weight_shrinks_toward_prior_more_at_low_sample_size():
    low_n = compute_weight("congress", 1, 1.0, "house_pdf")
    high_n = compute_weight("congress", 1000, 1000.0, "house_pdf")

    assert SOURCE_PRIORS["house_pdf"] < low_n < high_n < 1.0
    assert abs(low_n - SOURCE_PRIORS["house_pdf"]) < abs(1.0 - SOURCE_PRIORS["house_pdf"])
    assert abs(high_n - 1.0) < abs(low_n - 1.0)


def test_unknown_actor_is_discounted_and_validated_positive_skill_keeps_prior():
    candidates = pd.DataFrame(
        [
            {"actor_id": "congress:UNKNOWN", "kind": "congress", "source": "house_pdf"},
            {"actor_id": "congress:SKILLED", "kind": "congress", "source": "house_pdf"},
            {"actor_id": "officer:UNKNOWN", "kind": "officer", "source": "form4"},
        ]
    )
    skill_table = pd.DataFrame(
        [{"actor_id": "congress:SKILLED", "n_matured": 3, "mean_excess_pct": 1.2}]
    )

    weights = compute_actor_weights(candidates, skill_table)

    assert weights["congress:UNKNOWN"] == pytest.approx(0.52 * 0.5)
    assert weights["congress:SKILLED"] == SOURCE_PRIORS["house_pdf"]
    assert weights["officer:UNKNOWN"] == pytest.approx(0.60 * 0.5)


def test_skill_table_excludes_future_disclosures_and_unmatured_outcomes():
    as_of = pd.Timestamp("2024-04-02")
    disclosures = pd.DataFrame(
        [
            {"actor_id": "congress:ALICE", "ticker": "AAA", "source": "house_pdf", "disclosure_date": "2024-01-01"},
            {"actor_id": "congress:ALICE", "ticker": "BBB", "source": "house_pdf", "disclosure_date": "2024-01-02"},
            {"actor_id": "congress:ALICE", "ticker": "CCC", "source": "house_pdf", "disclosure_date": "2024-01-03"},
            {"actor_id": "congress:ALICE", "ticker": "DDD", "source": "house_pdf", "disclosure_date": "2024-04-02"},
            {"actor_id": "congress:ALICE", "ticker": "EEE", "source": "house_pdf", "disclosure_date": "2024-04-03"},
        ]
    )
    prices = pd.DataFrame(
        [
            ("AAA", "2024-01-02", 100.0), ("AAA", "2024-03-28", 110.0),
            ("BBB", "2024-01-03", 100.0), ("BBB", "2024-04-01", 110.0),
            ("CCC", "2024-01-04", 100.0), ("CCC", "2024-04-02", 110.0),
            ("DDD", "2024-04-03", 100.0), ("DDD", "2024-07-01", 110.0),
            ("EEE", "2024-04-04", 100.0), ("EEE", "2024-07-02", 110.0),
            ("SPY", "2024-01-02", 100.0), ("SPY", "2024-03-28", 105.0),
            ("SPY", "2024-01-03", 100.0), ("SPY", "2024-04-01", 105.0),
            ("SPY", "2024-01-04", 100.0), ("SPY", "2024-04-02", 105.0),
        ],
        columns=["ticker", "date", "close"],
    )

    skill_table = compute_skill_table(disclosures, prices, as_of)

    assert skill_table.to_dict("records") == [
        {"actor_id": "congress:ALICE", "n_matured": 2, "mean_excess_pct": pytest.approx(5.0)}
    ]


def test_cli_and_replay_use_the_same_actor_weight_path():
    from analyzer import cli, replay
    from analyzer.actors import weights_for_candidates

    assert cli.weights_for_candidates is weights_for_candidates
    assert replay._load_dependencies()[2].weights_for_candidates is weights_for_candidates


def test_age_weight_halves_at_half_life():
    assert age_weight(365) == 0.5
    assert age_weight(730, half_life_days=365) == 0.25


def test_actor_id_uses_kind_specific_name_normalization():
    assert actor_id("congress", "Dr. Michael T. McCaul Jr.") == "congress:MICHAEL T MCCAUL"
    assert actor_id("officer", "  Jane   DOE ") == "officer:JANE DOE"
    assert actor_id("manager", "  Jane   DOE ") == "manager:JANE DOE"
