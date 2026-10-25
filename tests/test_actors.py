import pandas as pd
import pytest

from analyzer.actors import (
    SOURCE_PRIORS,
    actor_id,
    compute_actor_weights,
    compute_skill_table,
    compute_weight,
)
from analyzer.persistence.database import Database
from analyzer.persistence.price_repository import (
    next_nyse_session,
    previous_nyse_session,
)
from analyzer.price_source import (
    _PRICE_BASIS,
    _set_generation_state,
    _store_generation,
    ensure_price_generation_schema,
)


def _with_comparable_generations(prices):
    prices = prices.copy()
    prices["price_generation_id"] = prices["ticker"].map(
        lambda ticker: f"test-generation:{str(ticker).strip().upper()}"
    )
    prices["comparability_status"] = "comparable"
    prices["price_generation_basis"] = _PRICE_BASIS
    return prices


def _set_comparable_states(db, tickers):
    ensure_price_generation_schema(db)
    for ticker in sorted(set(tickers)):
        ticker = str(ticker).strip().upper()
        generation = f"test-generation:{ticker}"
        rows = db.conn.execute(
            "SELECT date, close FROM prices WHERE ticker=? ORDER BY date", [ticker]
        ).fetchdf()
        series = pd.Series(
            rows.close.to_numpy(dtype="float64"), index=pd.to_datetime(rows.date)
        )
        _store_generation(
            db, ticker, generation, series, pd.Series(1.0, index=series.index),
            _PRICE_BASIS,
        )
        _set_generation_state(db, ticker, generation, "comparable")


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

    skill_table = compute_skill_table(
        disclosures, _with_comparable_generations(prices), as_of
    )

    assert skill_table.to_dict("records") == [
        {"actor_id": "congress:ALICE", "n_matured": 2, "mean_excess_pct": pytest.approx(5.0)}
    ]


def test_skill_outcomes_require_comparable_consistent_price_generations(caplog):
    disclosures = pd.DataFrame(
        [
            {
                "actor_id": "congress:SPLIT",
                "ticker": "SPLT",
                "source": "house_pdf",
                "disclosure_date": "2023-01-03",
            },
            *[
                {
                    "actor_id": "congress:COMPARABLE",
                    "ticker": ticker,
                    "source": "house_pdf",
                    "disclosure_date": pd.Timestamp("2023-01-04")
                    + pd.Timedelta(days=index),
                }
                for index, ticker in enumerate(("GOOD1", "GOOD2", "GOOD3"))
            ],
            {
                "actor_id": "congress:MIXED",
                "ticker": "MIXD",
                "source": "house_pdf",
                "disclosure_date": "2023-01-08",
            },
            {
                "actor_id": "congress:MISSING",
                "ticker": "MISS",
                "source": "house_pdf",
                "disclosure_date": "2023-01-09",
            },
        ]
    )
    rows = []

    def add_pair(
        ticker, disclosed, entry_close, exit_close, entry_gen, exit_gen, status="comparable"
    ):
        entry = next_nyse_session(pd.Timestamp(disclosed)).normalize()
        exit_date = previous_nyse_session(
            pd.Timestamp(disclosed) + pd.Timedelta(days=90)
        ).normalize()
        rows.extend(
            [
            (ticker, entry, entry_close, entry_gen, status, _PRICE_BASIS),
            (ticker, exit_date, exit_close, exit_gen, status, _PRICE_BASIS),
            ("SPY", entry, 100.0, "spy-generation", "comparable", _PRICE_BASIS),
            ("SPY", exit_date, 100.0, "spy-generation", "comparable", _PRICE_BASIS),
            ]
        )

    add_pair(
        "SPLT", "2023-01-03", 100.0, 10.0,
        "split-generation", "split-generation", "unknown",
    )
    for index, ticker in enumerate(("GOOD1", "GOOD2", "GOOD3")):
        disclosed = pd.Timestamp("2023-01-04") + pd.Timedelta(days=index)
        add_pair(
            ticker, disclosed, 100.0, 110.0,
            f"generation:{ticker}", f"generation:{ticker}",
        )
    add_pair(
        "MIXD", "2023-01-08", 100.0, 110.0,
        "entry-generation", "exit-generation",
    )
    prices = pd.DataFrame(
        rows,
        columns=[
            "ticker", "date", "close", "price_generation_id",
            "comparability_status", "price_generation_basis",
        ],
    )

    with caplog.at_level("WARNING", logger="analyzer.actors"):
        skill_table = compute_skill_table(
            disclosures, prices, pd.Timestamp("2023-06-01")
        )

    assert skill_table.to_dict("records") == [
        {
            "actor_id": "congress:COMPARABLE",
            "n_matured": 3,
            "mean_excess_pct": pytest.approx(10.0),
        }
    ]
    assert "Dropped 2 matured skill episodes" in caplog.text
    assert "Dropped 1 matured skill episodes with missing total-return endpoint prices" in caplog.text


def test_skill_outcome_uses_adjusted_total_return_closes_for_stock_and_spy():
    disclosed = pd.Timestamp("2023-01-03")
    entry = next_nyse_session(disclosed).normalize()
    exit_date = previous_nyse_session(disclosed + pd.Timedelta(days=90)).normalize()
    # A $1 dividend takes the raw stock from $100 to $99 on the ex-date;
    # Yahoo's adjusted entry close is $99 and the later $108.90 is a 10% total
    # return. SPY's adjusted closes are a 5% total return, so excess is 5 pp.
    disclosures = pd.DataFrame([{
        "actor_id": "congress:DIVIDEND",
        "ticker": "DIV",
        "source": "house_pdf",
        "disclosure_date": disclosed,
    }])
    prices = pd.DataFrame(
        [
            ("DIV", entry, 99.0, "div-generation", "comparable", _PRICE_BASIS),
            ("DIV", exit_date, 108.9, "div-generation", "comparable", _PRICE_BASIS),
            ("SPY", entry, 200.0, "spy-generation", "comparable", _PRICE_BASIS),
            ("SPY", exit_date, 210.0, "spy-generation", "comparable", _PRICE_BASIS),
        ],
        columns=[
            "ticker", "date", "close", "price_generation_id",
            "comparability_status", "price_generation_basis",
        ],
    )

    result = compute_skill_table(disclosures, prices, pd.Timestamp("2023-06-01"))

    assert result.to_dict("records") == [{
        "actor_id": "congress:DIVIDEND",
        "n_matured": 1,
        "mean_excess_pct": pytest.approx(5.0),
    }]


def test_load_skill_table_uses_persisted_comparable_generation_state(tmp_path):
    from analyzer.actors import load_skill_table

    db = Database(tmp_path / "skill-price-generations.duckdb")
    _insert_positive_skill_history(
        db, "Comparable Member", "house_pdf", ("AAA", "BBB", "CCC")
    )
    disclosed = pd.Timestamp("2023-01-05")
    entry = next_nyse_session(disclosed).date()
    exit_date = previous_nyse_session(disclosed + pd.Timedelta(days=90)).date()
    db.conn.execute(
        "INSERT INTO transactions (doc_id, member, ticker, transaction_date, "
        "disclosure_date, transaction_type, owner_code, amount_raw, amount_midpoint, "
        "source, created_at) VALUES ('split-member', 'Split Member', 'SPLT', ?, ?, "
        "'Purchase', '', '$1,001 - $15,000', 8000, 'house_pdf', '2023-01-01')",
        [disclosed.date(), disclosed.date()],
    )
    db.conn.executemany(
        "INSERT INTO prices (ticker, date, close) VALUES (?, ?, ?)",
        [("SPLT", entry, 100.0), ("SPLT", exit_date, 10.0)],
    )
    split_dates = pd.to_datetime([entry, exit_date])
    _store_generation(
        db, "SPLT", "split-unadjusted-generation",
        pd.Series([100.0, 10.0], index=split_dates),
        pd.Series([1.0, 1.0], index=split_dates), _PRICE_BASIS,
    )
    _set_generation_state(
        db, "SPLT", "split-unadjusted-generation", "unknown",
        "split-adjusted-generation", "split overlap diverges",
    )

    try:
        skill_table = load_skill_table(
            db,
            pd.Timestamp("2023-05-01"),
            {"congress:COMPARABLE MEMBER", "congress:SPLIT MEMBER"},
        )
    finally:
        db.close()

    assert skill_table.to_dict("records") == [
        {
            "actor_id": "congress:COMPARABLE MEMBER",
            "n_matured": 3,
            "mean_excess_pct": pytest.approx(10.0),
        }
    ]


def test_cli_and_replay_use_the_same_actor_weight_path():
    from analyzer import cli, replay
    from analyzer.actors import weights_for_candidates

    assert cli.weights_for_candidates is weights_for_candidates
    assert replay._load_dependencies()[2].weights_for_candidates is weights_for_candidates


def test_actor_id_uses_kind_specific_name_normalization():
    assert actor_id("congress", "Dr. Michael T. McCaul Jr.") == "congress:MICHAEL T MCCAUL"
    assert actor_id("congress", "Charles James Fleischmann") != actor_id(
        "congress", "Charles Joseph Fleischmann"
    )
    assert actor_id("officer", "  Jane   DOE ") == "officer:JANE DOE"
    assert actor_id("manager", "  Jane   DOE ") == "manager:JANE DOE"


@pytest.mark.parametrize(
    ("filed_name", "short_name"),
    [
        ("Michael T. McCaul", "Michael McCaul"),
        ("John J. McGuire", "John McGuire"),
        ("Thomas R. Suozzi", "Thomas Suozzi"),
        ("Diana Lynn Harshbarger", "Diana Harshbarger"),
        ("Neal Patrick Dunn MD, FACS", "Neal Patrick Dunn"),
    ],
)
def test_setup_and_replay_candidate_ids_share_deterministic_name_representative(
    filed_name, short_name
):
    from analyzer.member_names import canonicalize_actor_ids, same_member_identity

    filed_id = actor_id("congress", filed_name)
    short_id = actor_id("congress", short_name)
    assert same_member_identity(filed_name, short_name)

    # Production keeps its event spelling. Replay may hold the requested short
    # spelling, while its coalesced event retains the filed ID as provenance.
    production = pd.DataFrame(
        {"actor_id": [filed_id], "_pre_coalesced_actor_id": [filed_id]}
    )
    replay = pd.DataFrame(
        {"actor_id": [short_id, short_id], "_pre_coalesced_actor_id": [None, filed_id]}
    )

    def canonical_ids(frame):
        aliases = frame["_pre_coalesced_actor_id"].dropna().astype(str).tolist()
        return canonicalize_actor_ids(
            frame["actor_id"].astype(str).tolist(), aliases=aliases
        )

    production_ids = canonical_ids(production)
    replay_ids = canonical_ids(replay)
    expected = filed_id
    assert production_ids == [expected]
    assert replay_ids == [expected, expected]

    # The selected representative does not depend on row order or repeated calls.
    forward = canonicalize_actor_ids([filed_id, short_id])
    reverse = canonicalize_actor_ids([short_id, filed_id])[::-1]
    assert forward == reverse
    assert canonicalize_actor_ids([filed_id, short_id]) == forward


def test_canonical_actor_representative_does_not_merge_distinct_full_middles():
    from analyzer.member_names import canonicalize_actor_ids, same_member_identity

    allen = actor_id("congress", "Charles James Allen Smith")
    arthur = actor_id("congress", "Charles James Arthur Smith")
    assert not same_member_identity(
        "Charles James Allen Smith", "Charles James Arthur Smith"
    )

    normalized = canonicalize_actor_ids(
        [allen, arthur], aliases=["congress:CHARLES SMITH"]
    )
    assert normalized == [allen, arthur]
    assert len(set(normalized)) == 2


def test_identity_helpers_have_one_owner_with_legacy_facade_aliases():
    from analyzer import actors, member_names, positions

    assert actors.actor_id is member_names.actor_id
    assert positions.same_actor_identity is member_names.same_actor_identity
    assert positions.same_member_identity is member_names.same_member_identity
    assert positions.dedupe_member_names is member_names.dedupe_member_names


def test_mccaul_skill_matches_exact_name_variant_only(tmp_path):
    from analyzer.actors import weights_for_candidates

    db = Database(tmp_path / "mccaul.duckdb")
    disclosures = [pd.Timestamp("2023-01-03") + pd.Timedelta(days=day) for day in range(3)]
    db.conn.executemany(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, owner_code, amount_raw, amount_midpoint, source, created_at
        ) VALUES (?, ?, ?, ?, ?, 'Purchase', '', '$1,001 - $15,000', 8000, 'house_pdf', '2023-01-01')
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
    _set_comparable_states(db, [ticker for ticker, _ in prices])
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
            transaction_type, owner_code, amount_raw, amount_midpoint, source, created_at
        ) VALUES (?, 'Nydia Velázquez', ?, ?, ?, 'Purchase', '', '$1,001 - $15,000', 8000, 'house_pdf', '2023-01-01')
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
    _set_comparable_states(db, [ticker for ticker, _ in prices])
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
            transaction_type, owner_code, amount_raw, amount_midpoint, source, created_at
        ) VALUES (?, ?, ?, ?, ?, 'Purchase', '', '$1,001 - $15,000', 8000, ?, '2023-01-01')
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
    _set_comparable_states(db, [ticker for ticker, _ in prices])


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
            transaction_type, owner_code, amount_raw, amount_midpoint, source, created_at
        ) VALUES (?, ?, ?, DATE '2023-01-01', DATE '2023-01-02',
                  'Purchase', '', '$1,001 - $15,000', 8000, ?, '2023-01-01')
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


def test_ligature_name_history_uses_full_python_normalized_history(tmp_path):
    from analyzer.actors import load_skill_table

    db = Database(tmp_path / "ligature-member.duckdb")
    disclosures = [
        pd.Timestamp("2023-01-03") + pd.Timedelta(days=index)
        for index in range(6)
    ]
    members = ["John Flower"] * 3 + ["John ﬂower"] * 3
    tickers = ("AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA")
    db.conn.executemany(
        """
        INSERT INTO transactions (
            doc_id, member, ticker, transaction_date, disclosure_date,
            transaction_type, owner_code, amount_raw, amount_midpoint, source, created_at
        ) VALUES (?, ?, ?, ?, ?, 'Purchase', '', '$1,001 - $15,000', 8000, 'house_pdf', '2023-01-01')
        """,
        [
            (f"ligature-{index}", member, ticker, day.date(), day.date())
            for index, (member, ticker, day) in enumerate(
                zip(members, tickers, disclosures)
            )
        ],
    )
    prices = {}
    for index, (ticker, disclosed) in enumerate(zip(tickers, disclosures)):
        entry = next_nyse_session(disclosed).date()
        exit_date = previous_nyse_session(disclosed + pd.Timedelta(days=90)).date()
        prices[(ticker, entry)] = 100.0
        prices[(ticker, exit_date)] = 110.0 if index < 3 else 80.0
        prices[("SPY", entry)] = 100.0
        prices[("SPY", exit_date)] = 100.0
    db.conn.executemany(
        "INSERT INTO prices (ticker, date, close) VALUES (?, ?, ?)",
        [(ticker, day, close) for (ticker, day), close in prices.items()],
    )
    _set_comparable_states(db, [ticker for ticker, _ in prices])
    requested_id = actor_id("congress", "John Flower")
    candidates = pd.DataFrame(
        [{"actor_id": requested_id, "kind": "congress", "source": "house_pdf"}]
    )

    try:
        skill_table = load_skill_table(
            db, pd.Timestamp("2024-05-01"), {requested_id}
        )
    finally:
        db.close()

    assert skill_table.to_dict("records") == [
        {
            "actor_id": requested_id,
            "n_matured": 6,
            "mean_excess_pct": pytest.approx(-5.0),
        }
    ]
    assert compute_actor_weights(candidates, skill_table)[requested_id] == pytest.approx(
        0.26
    )


def test_float_residue_zero_mean_does_not_validate_skill():
    from analyzer.actors import compute_actor_weights

    disclosures = pd.DataFrame(
        [
            {
                "actor_id": "congress:ALICE",
                "ticker": ticker,
                "source": "house_pdf",
                "disclosure_date": pd.Timestamp("2023-01-03") + pd.Timedelta(days=index),
            }
            for index, ticker in enumerate(("AAA", "BBB", "CCC"))
        ]
    )
    prices = []
    for ticker, disclosed, exit_close in zip(
        disclosures["ticker"],
        disclosures["disclosure_date"],
        (110.0, 90.0, 100.0),
    ):
        entry = next_nyse_session(disclosed).normalize()
        exit_date = previous_nyse_session(
            disclosed + pd.Timedelta(days=90)
        ).normalize()
        prices.extend(
            [
                (ticker, entry, 100.0),
                (ticker, exit_date, exit_close),
                ("SPY", entry, 100.0),
                ("SPY", exit_date, 100.0),
            ]
        )
    skill_table = compute_skill_table(
        disclosures,
        _with_comparable_generations(
            pd.DataFrame(prices, columns=["ticker", "date", "close"])
        ),
        pd.Timestamp("2024-05-01"),
    )
    candidates = pd.DataFrame(
        [{"actor_id": "congress:ALICE", "kind": "congress", "source": "house_pdf"}]
    )

    assert abs(skill_table.iloc[0]["mean_excess_pct"]) < 1e-9
    assert compute_actor_weights(candidates, skill_table)["congress:ALICE"] == pytest.approx(
        0.26
    )


def test_infinite_benchmark_price_is_excluded_from_skill():
    from analyzer.actors import compute_actor_weights

    disclosures = pd.DataFrame(
        [
            {
                "actor_id": "congress:ALICE",
                "ticker": ticker,
                "source": "house_pdf",
                "disclosure_date": pd.Timestamp("2023-01-03") + pd.Timedelta(days=index),
            }
            for index, ticker in enumerate(("AAA", "BBB", "CCC", "DDD"))
        ]
    )
    prices = []
    for index, (ticker, disclosed) in enumerate(
        zip(disclosures["ticker"], disclosures["disclosure_date"])
    ):
        entry = next_nyse_session(disclosed).normalize()
        exit_date = previous_nyse_session(
            disclosed + pd.Timedelta(days=90)
        ).normalize()
        prices.extend(
            [
                (ticker, entry, 100.0),
                (ticker, exit_date, 100.0),
                ("SPY", entry, float("inf") if index == 0 else 100.0),
                ("SPY", exit_date, 100.0),
            ]
        )
    skill_table = compute_skill_table(
        disclosures,
        _with_comparable_generations(
            pd.DataFrame(prices, columns=["ticker", "date", "close"])
        ),
        pd.Timestamp("2024-05-01"),
    )
    candidates = pd.DataFrame(
        [{"actor_id": "congress:ALICE", "kind": "congress", "source": "house_pdf"}]
    )

    assert skill_table.to_dict("records") == [
        {"actor_id": "congress:ALICE", "n_matured": 3, "mean_excess_pct": 0.0}
    ]
    assert compute_actor_weights(candidates, skill_table)["congress:ALICE"] == pytest.approx(
        0.26
    )
