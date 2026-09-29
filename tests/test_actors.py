import pytest

from analyzer.actors import SOURCE_PRIORS, actor_id, age_weight, compute_weight


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


def test_age_weight_halves_at_half_life():
    assert age_weight(365) == 0.5
    assert age_weight(730, half_life_days=365) == 0.25


def test_actor_id_uses_kind_specific_name_normalization():
    assert actor_id("congress", "Dr. Michael T. McCaul Jr.") == "congress:MICHAEL T MCCAUL"
    assert actor_id("officer", "  Jane   DOE ") == "officer:JANE DOE"
    assert actor_id("manager", "  Jane   DOE ") == "manager:JANE DOE"
