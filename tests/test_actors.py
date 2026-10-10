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
from analyzer.database import Database
from analyzer.price_repository import next_nyse_session, previous_nyse_session


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


def test_mccaul_skill_matches_exact_name_variant_only(tmp_path):
    from analyzer.actors import weights_for_candidates

    db = Database(tmp_path / "mccaul.duckdb")
    disclosures = [pd.Timestamp("2023-01-03") + pd.Timedelta(days=day) for day in range(3)]
    db.conn.executemany(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, owner_code, amount_raw, amount_midpoint, source
        ) VALUES (?, ?, ?, ?, ?, 'Purchase', '', '$1,001 - $15,000', 8000, 'house_pdf')
        """,
        [
            (f"mccaul-{index}", "Michael T. McCaul", ticker, day.date(), day.date())
            for index, (ticker, day) in enumerate(zip(("AAPL", "MSFT", "GOOG"), disclosures))
        ],
    )
    prices = {}
    for ticker, disclosed in zip(("AAPL", "MSFT", "GOOG"), disclosures):
        entry = next_nyse_session(disclosed).date()
        exit_date = previous_nyse_session(disclosed + pd.Timedelta(days=90)).date()
        prices[(ticker, entry)] = 100.0
        prices[(ticker, exit_date)] = 120.0
        prices[("SPY", entry)] = 100.0
        prices[("SPY", exit_date)] = 110.0
    db.conn.executemany(
        "INSERT INTO prices (ticker, date, close) VALUES (?, ?, ?)",
        [(ticker, day, close) for (ticker, day), close in prices.items()],
    )
    candidates = pd.DataFrame(
        [
            {
                "actor_id": "congress:MICHAEL MCCAUL",
                "kind": "congress",
                "source": "house_pdf",
            },
            {
                "actor_id": "congress:MICHAEL T MCCAUL",
                "kind": "congress",
                "source": "house_pdf",
            },
        ]
    )

    try:
        weights = weights_for_candidates(db, candidates, pd.Timestamp("2024-05-01"))
    finally:
        db.close()

    assert weights == {
        "congress:MICHAEL MCCAUL": pytest.approx(0.26),
        "congress:MICHAEL T MCCAUL": SOURCE_PRIORS["house_pdf"],
    }


def test_accented_surname_skill_history_is_found(tmp_path):
    from analyzer.actors import weights_for_candidates

    db = Database(tmp_path / "accented-member.duckdb")
    disclosures = [
        pd.Timestamp("2023-01-03") + pd.Timedelta(days=day) for day in range(3)
    ]
    tickers = ("AAPL", "MSFT", "GOOG")
    db.conn.executemany(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, owner_code, amount_raw, amount_midpoint, source
        ) VALUES (?, 'Nydia Velázquez', ?, ?, ?, 'Purchase', '', '$1,001 - $15,000', 8000, 'house_pdf')
        """,
        [
            (f"velazquez-{ticker}", ticker, day.date(), day.date())
            for ticker, day in zip(tickers, disclosures)
        ],
    )
    prices = {}
    for ticker, disclosed in zip(tickers, disclosures):
        entry = next_nyse_session(disclosed).date()
        exit_date = previous_nyse_session(disclosed + pd.Timedelta(days=90)).date()
        prices[(ticker, entry)] = 100.0
        prices[(ticker, exit_date)] = 120.0
        prices[("SPY", entry)] = 100.0
        prices[("SPY", exit_date)] = 100.0
    db.conn.executemany(
        "INSERT INTO prices (ticker, date, close) VALUES (?, ?, ?)",
        [(ticker, day, close) for (ticker, day), close in prices.items()],
    )
    candidates = pd.DataFrame(
        [
            {
                "actor_id": actor_id("congress", "Nydia Velázquez"),
                "kind": "congress",
                "source": "house_pdf",
            }
        ]
    )

    try:
        weights = weights_for_candidates(db, candidates, pd.Timestamp("2024-05-01"))
    finally:
        db.close()

    assert weights["congress:NYDIA VELAZQUEZ"] == SOURCE_PRIORS["house_pdf"]


@pytest.mark.parametrize(
    ("member", "kind", "source"),
    [("SØRENSEN", "congress", "house_pdf"), ("Weiß", "officer", "form4")],
)
def test_unicode_actor_id_normalization_matches_history_prefilter(
    tmp_path, member, kind, source
):
    from analyzer.actors import weights_for_candidates

    db = Database(tmp_path / f"unicode-{kind}.duckdb")
    _insert_positive_skill_history(db, member, source, ("AAPL", "MSFT", "GOOG"))
    candidates = pd.DataFrame(
        [{"actor_id": actor_id(kind, member), "kind": kind, "source": source}]
    )

    try:
        weights = weights_for_candidates(db, candidates, pd.Timestamp("2024-05-01"))
    finally:
        db.close()

    assert weights[actor_id(kind, member)] == SOURCE_PRIORS[source]


def test_coalesced_john_andrew_zts_event_does_not_inherit_john_alan_skill(tmp_path):
    from analyzer import positions
    from analyzer.actors import weights_for_candidates

    db = Database(tmp_path / "andrew-alan-zts.duckdb")
    _insert_positive_skill_history(db, "John Alan Smith", "house_pdf", ("ZTS",) * 3)
    held = pd.DataFrame(
        [
            {
                "ticker": "ZTS",
                "actor_id": actor_id("congress", "John Alan Smith"),
                "kind": "congress",
                "source": "house_pdf",
                "position_evidence": True,
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "ticker": "ZTS",
                "actor_id": actor_id("congress", "John Andrew Smith"),
                "kind": "congress",
                "source": "house_pdf",
                "position_evidence": False,
            }
        ]
    )

    try:
        coalesced = positions.coalesce_actor_variants(events, held)
        candidates = pd.concat([held, coalesced], ignore_index=True)
        weights = weights_for_candidates(
            db, candidates, pd.Timestamp("2024-05-01")
        )
    finally:
        db.close()

    assert (
        coalesced.iloc[0]["actor_id"] == actor_id("congress", "John Andrew Smith")
        and weights[actor_id("congress", "John Andrew Smith")] == pytest.approx(0.26)
    )


def _insert_positive_skill_history(db, member, source, tickers):
    disclosures = [
        pd.Timestamp("2023-01-03") + pd.Timedelta(days=index)
        for index in range(len(tickers))
    ]
    db.conn.executemany(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, owner_code, amount_raw, amount_midpoint, source
        ) VALUES (?, ?, ?, ?, ?, 'Purchase', '', '$1,001 - $15,000', 8000, ?)
        """,
        [
            (
                f"{source}-{member}-{index}",
                member,
                ticker,
                day.date(),
                day.date(),
                source,
            )
            for index, (ticker, day) in enumerate(zip(tickers, disclosures))
        ],
    )
    prices = {}
    for ticker, disclosed in zip(tickers, disclosures):
        entry = next_nyse_session(disclosed).date()
        exit_date = previous_nyse_session(disclosed + pd.Timedelta(days=90)).date()
        prices[(ticker, entry)] = 100.0
        prices[(ticker, exit_date)] = 120.0
        prices[("SPY", entry)] = 100.0
        prices[("SPY", exit_date)] = 110.0
    db.conn.executemany(
        "INSERT INTO prices (ticker, date, close) VALUES (?, ?, ?)",
        [(ticker, day, close) for (ticker, day), close in prices.items()],
    )


def test_single_actor_skill_history_query_returns_only_matching_rows(tmp_path, monkeypatch):
    from analyzer.actors import load_skill_table

    db = Database(tmp_path / "bounded-skill-history.duckdb")
    records = [
        ("target", "Target Member", "AAPL", "house_pdf"),
        *[
            (f"other-{index}", f"Other Member {index}", "MSFT", "house_pdf")
            for index in range(500)
        ],
    ]
    db.conn.executemany(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, owner_code, amount_raw, amount_midpoint, source
        ) VALUES (?, ?, ?, DATE '2023-01-01', DATE '2023-01-02',
                  'Purchase', '', '$1,001 - $15,000', 8000, ?)
        """,
        records,
    )
    fetched_rows = []
    get_skill_history = db.transactions.get_skill_history

    def record_query_rows(*args):
        result = get_skill_history(*args)
        fetched_rows.append(len(result))
        return result

    monkeypatch.setattr(db.transactions, "get_skill_history", record_query_rows)

    try:
        load_skill_table(
            db,
            pd.Timestamp("2024-01-01"),
            {"congress:TARGET MEMBER"},
        )
    finally:
        db.close()

    assert fetched_rows == [1]


def test_weight_lookup_cache_reuses_skill_table(monkeypatch):
    import analyzer.actors as actors

    calls = []
    monkeypatch.setattr(
        actors,
        "load_skill_table",
        lambda db, as_of, actor_ids, **_kwargs: calls.append((as_of, actor_ids))
        or pd.DataFrame(columns=["actor_id", "n_matured", "mean_excess_pct"]),
    )
    candidates = pd.DataFrame(
        [{"actor_id": "congress:ALICE", "kind": "congress", "source": "house_pdf"}]
    )
    cache = {}

    actors.weights_for_candidates("db", candidates, pd.Timestamp("2024-01-01"), skill_table_cache=cache)
    actors.weights_for_candidates("db", candidates, pd.Timestamp("2024-01-01"), skill_table_cache=cache)

    assert len(calls) == 1


def test_filtered_candidates_with_nonconsecutive_index_do_not_crash(monkeypatch):
    import analyzer.actors as actors

    monkeypatch.setattr(
        actors,
        "load_skill_table",
        lambda *_args: pd.DataFrame(columns=["actor_id", "n_matured", "mean_excess_pct"]),
    )
    candidates = pd.DataFrame(
        [
            {"actor_id": "manager:ALICE", "kind": "manager", "source": "13f"},
            {"actor_id": "congress:BOB", "kind": "congress", "source": "house_pdf"},
        ],
        index=[4, 9],
    )

    assert actors.weights_for_candidates(None, candidates, pd.Timestamp("2024-01-01")) == {
        "manager:ALICE": 0.0,
        "congress:BOB": pytest.approx(0.26),
    }


def test_zero_prior_actor_skips_skill_history(monkeypatch):
    import analyzer.actors as actors

    monkeypatch.setattr(
        actors,
        "load_skill_table",
        lambda *_args: pytest.fail("zero-prior 13F actor loaded skill history"),
    )
    candidates = pd.DataFrame(
        [{"actor_id": "manager:ALICE", "kind": "manager", "source": "13f"}]
    )

    assert actors.weights_for_candidates(None, candidates, pd.Timestamp("2024-01-01")) == {
        "manager:ALICE": 0.0
    }
