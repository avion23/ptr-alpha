import inspect
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

AS_OF = date(2026, 1, 20)
EXPECTED_CANDIDATE_COLUMNS = {
    "ticker",
    "actor_id",
    "kind",
    "source",
    "entry_ref",
    "event_date",
    "disclosure_date",
    "corroboration",
    "as_of",
}
EXPECTED_KINDS = {"congress", "officer", "manager"}


class _QueryResult:
    def __init__(self, frame):
        self.frame = frame

    def fetchdf(self):
        return self.frame.copy()

    def fetchall(self):
        return []

    def fetchone(self):
        if self.frame.empty:
            return None
        return (100.0,)


class _FakeDatabase:
    def __init__(self, frame):
        self.frame = frame
        self.conn = self

    def execute(self, *_args, **_kwargs):
        sql = str(_args[0]).casefold() if _args else ""
        if "from prices" in sql:
            return _QueryResult(
                pd.DataFrame(
                    {
                        "date": [date(2026, 1, 18), date(2026, 1, 20)],
                        "close": [100.0, 100.0],
                    }
                )
            )
        return _QueryResult(self.frame)

    def get_transactions(self, *_args, **_kwargs):
        return self.frame.copy()

    def get_transactions_by_date_range(self, *_args, **_kwargs):
        return self.frame.copy()


class _FakeSource:
    def __init__(self, frame):
        self.db = _FakeDatabase(frame)

    def get_transactions(self, *_args, **_kwargs):
        return self.db.frame.copy()


class _FakePriceSource:
    def __init__(self):
        index = pd.date_range("2025-01-01", periods=900, freq="D")
        self.frame = pd.DataFrame({"ACME": 100.0, "SPY": 100.0}, index=index)

    def get_prices(self, tickers, start=None, end=None):
        frame = self.frame.loc[:, [ticker for ticker in tickers if ticker in self.frame]]
        if start is not None:
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end)]
        return frame.copy()


def _seed(source):
    is_13f = source == "13f"
    return pd.DataFrame(
        [
            {
                "ticker": "ACME",
                "actor_id": "manager:THIEL MACRO LLC" if is_13f else "officer:BURKE JAMES A",
                "member": "Thiel Macro LLC" if is_13f else "Burke James A",
                "kind": "manager" if is_13f else "officer",
                "source": source,
                "entry_ref": "must-not-survive" if is_13f else "fixture-entry",
                "event_date": date(2026, 1, 15),
                "transaction_date": date(2026, 1, 15),
                "disclosure_date": date(2026, 1, 18),
                "as_of": AS_OF,
                "corroboration": False,
                "transaction_type": "Purchase",
                "amount_midpoint": 10_000.0,
                "amount_raw": "$10,000",
                "source_record_id": "fixture-record",
                "source_row_id": "fixture-row",
                "manager_key": "thiel_macro" if is_13f else "burke_james_a",
                "raw_transaction_subtype": "INFERRED_POSITION_INCREASE" if is_13f else "P",
                "shares": 100,
                "total_cost": 10_000.0,
                "cost_basis": 100.0,
                "current_price": 100.0,
                "discount_pct": 0.0,
                "first_buy": date(2026, 1, 15),
                "last_activity": date(2026, 1, 15),
            }
        ]
    )


def _invoke(function, frame, *, price_source=None, source_value=None):
    """Call an optional builder with in-memory inputs matched to its signature."""
    fake_source = _FakeSource(frame)
    fake_prices = price_source or _FakePriceSource()
    values = {}
    for parameter in inspect.signature(function).parameters.values():
        name = parameter.name.casefold()
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in {"as_of", "as_of_date", "cutoff", "cutoff_date"}:
            value = AS_OF
        elif name in {"start", "start_date"}:
            value = date(2026, 1, 1)
        elif name in {"end", "end_date"}:
            value = date(2026, 12, 31)
        elif "weight" in name:
            value = {}
        elif name in {"transaction_source", "source_object"}:
            value = fake_source
        elif name in {"db", "database"}:
            value = fake_source.db
        elif name in {"price_source", "pricesource"}:
            value = fake_prices
        elif name in {"prices", "price_history", "price_data"}:
            value = fake_prices.frame
        elif name == "source" and source_value is not None:
            value = source_value
        elif name in {"horizon", "horizon_days"}:
            value = 30
        elif name in {"top_n", "limit"}:
            value = 5
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            value = frame
        values[parameter.name] = value

    positional = []
    keywords = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name not in values:
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            positional.append(values[parameter.name])
        else:
            keywords[parameter.name] = values[parameter.name]
    return function(*positional, **keywords)


def _present_function(module, name):
    function = getattr(module, name, None)
    if not callable(function):
        pytest.skip(f"{module.__name__}.{name} is not integrated")
    return function


def test_actor_identity_and_source_weight_contract():
    actors = pytest.importorskip("analyzer.actors", reason="actors module is not integrated")

    assert actors.actor_id("congress", "NANCY PELOSI") == "congress:NANCY PELOSI"
    assert actors.actor_id("officer", "Burke James A") == "officer:BURKE JAMES A"
    assert actors.SOURCE_PRIORS["form4"] == 0.60
    assert actors.SOURCE_PRIORS["13f"] is None

    for source, prior in (("form4", 0.60), ("13f", 0.0)):
        initial = actors.compute_weight("congress", 0, 0, source)
        assert 0.0 <= initial <= 1.0
        assert initial == prior
    # 13f never initiates so its weight stays 0 regardless of history.
    assert actors.compute_weight("congress", 200, 200, "13f") == 0.0
    empirical = actors.compute_weight("congress", 200, 200, "form4")
    assert 0.0 <= empirical <= 1.0
    assert abs(empirical - 1.0) < abs(0.60 - 1.0)


@pytest.mark.parametrize(
    ("module_name", "function_name", "source"),
    [
        ("analyzer.positions", "holdings_candidates", "holdings"),
        ("analyzer.pipeline", "eligible_events", "form4"),
        ("analyzer.form4", "candidates_from_sweep", "form4"),
        ("analyzer.form13f", "candidates_from_increases", "13f"),
    ],
)
def test_candidate_builder_frame_contract(module_name, function_name, source):
    module = pytest.importorskip(module_name, reason=f"{module_name} module is not integrated")
    function = _present_function(module, function_name)
    result = _invoke(function, _seed(source), source_value=source)

    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == len(EXPECTED_CANDIDATE_COLUMNS)
    assert set(result.columns) == EXPECTED_CANDIDATE_COLUMNS
    assert result["kind"].isin(EXPECTED_KINDS).all()
    thirteen_f_rows = result.loc[result["source"] == "13f"]
    if source == "13f":
        assert not thirteen_f_rows.empty
    if not thirteen_f_rows.empty:
        assert thirteen_f_rows["corroboration"].eq(True).all()
        assert thirteen_f_rows["entry_ref"].isna().all()


def test_setup_score_contract():
    setups = pytest.importorskip("analyzer.setups", reason="setups module is not integrated")
    score = _present_function(setups, "score")
    corroboration_only = pd.DataFrame(
        [
            {
                "ticker": "ACME",
                "actor_id": "manager:THIEL MACRO LLC",
                "kind": "manager",
                "source": "13f",
                "entry_ref": None,
                "event_date": date(2026, 1, 15),
                "disclosure_date": date(2026, 1, 18),
                "corroboration": True,
                "as_of": AS_OF,
            }
        ]
    )
    result = _invoke(score, corroboration_only)
    assert isinstance(result, tuple) and len(result) == 2
    ranked, _blocked = result
    assert isinstance(ranked, pd.DataFrame)
    assert {"ticker", "score", "actors", "reasons"} <= set(ranked.columns)
    candidate = ranked.loc[ranked["ticker"] == "ACME"]
    assert not candidate.empty
    assert pd.to_numeric(candidate["score"]).eq(0).all()


def test_replay_output_contract():
    replay_module = pytest.importorskip("analyzer.replay", reason="replay module is not integrated")
    replay = _present_function(replay_module, "replay")
    seed = _seed("form4")
    fake_db = _FakeSource(seed).db
    result = replay(
        fake_db,
        ["ACME"],
        [AS_OF],
        members=["Burke James A"],
        horizon_days=30,
    )
    assert isinstance(result, pd.DataFrame)
    assert {"ticker", "as_of", "score", "fwd_return_pct", "excess_pct"} <= set(result.columns)


def test_contract_tests_do_not_name_network_clients_or_endpoints():
    source = Path(__file__).read_text(encoding="utf-8").casefold()
    forbidden = ("ht" + "tp", "url" + "open", "re" + "quests", "y" + "finance")
    assert all(term not in source for term in forbidden)
