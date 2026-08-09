from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from analyzer.member_names import canonical_member_key
from optimize_profit import main
from optimize_profit.metrics import summarize_walk_forward
from optimize_profit.scoring import make_shuffled_scorer, score_constant
from optimize_profit.walk_forward import (
    _build_custom_ranking_dicts,
    _score_candidates,
    run_walk_forward,
)


def _ranking() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "member": ["Nancy P. Pelosi", "John Q. Public", "Alice Example"],
            "purchase_trades": [8, 4, 2],
            "bayes_win_prob": [0.7, 0.5, 0.4],
            "shrunk_alpha": [3.0, 1.0, -1.0],
        }
    )


def _period(date: str, *, status: str = "ready") -> dict:
    return {
        "as_of_ts": pd.Timestamp(date),
        "horizon": 90,
        "status": status,
        "candidate_tickers": {1: ["AAA"]},
    }


def test_custom_ranking_dicts_add_canonical_member_aliases():
    lookups = _build_custom_ranking_dicts(
        _ranking(), lambda frame: dict(zip(frame["member"], [9.0, 2.0, 1.0]))
    )

    assert lookups["alpha"][canonical_member_key("Nancy P. Pelosi")] == 9.0
    assert lookups["trades"][canonical_member_key("John Q. Public")] == 4
    assert lookups["prob"][canonical_member_key("Alice Example")] == 0.4


def test_non_overlapping_schedule_is_enforced():
    periods = {
        "a": _period("2024-01-01"),
        "b": _period("2024-01-31"),
    }

    with pytest.raises(ValueError, match="Overlapping bankroll periods"):
        run_walk_forward(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            periods,
            score_constant,
            top_n=3,
            min_buyers=1,
            allocation="equal",
        )


def test_first_period_loss_anchors_drawdown_and_stop_propagates():
    summary = summarize_walk_forward(
        [{"portfolio_return_pct": -10.0, "spy_return_pct": 0.0, "n_positions": 1}],
        False,
    )
    assert summary["max_drawdown_pct"] == -10.0

    periods = {
        "a": _period("2024-01-01"),
        "b": _period("2024-04-01"),
    }
    first = (
        {
            "as_of_date": pd.Timestamp("2024-01-01").date(),
            "portfolio_return_pct": -10.0,
            "spy_return_pct": 0.0,
            "n_positions": 1,
        },
        {"n_positions": 1},
        [],
    )
    with patch("optimize_profit.walk_forward._run_one_period", return_value=first):
        result = run_walk_forward(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            periods,
            score_constant,
            top_n=3,
            min_buyers=1,
            allocation="equal",
            max_dd_pct=5,
        )

    assert result["stopped_early"] is True
    assert result["n_periods"] == 1
    assert result["max_drawdown_pct"] == -10.0
    assert result["rejection_ledger"][-1]["reason"] == "drawdown_stop_active"


def test_candidate_failures_propagate_instead_of_being_swallowed():
    with patch(
        "optimize_profit.walk_forward.score_ticker_by_buyers",
        side_effect=RuntimeError("canary failure"),
    ):
        with pytest.raises(RuntimeError, match="canary failure"):
            _score_candidates(
                ["AAA"],
                pd.DataFrame({"ticker": ["AAA"]}),
                pd.DataFrame({"signal": [1]}),
                90,
                5.0,
                _ranking(),
                1,
                pd.DataFrame(),
                {"alpha": {}, "trades": {}, "prob": {}},
            )


def test_each_decay_is_passed_to_signal_computation():
    calls = []

    def fake_signals(entries, prices, horizons, decay_lambda):
        calls.append(decay_lambda)
        return pd.DataFrame({"decay": [decay_lambda]})

    def fake_precompute(signals, *args, **kwargs):
        return {"decay": float(signals["decay"].iloc[0])}

    with (
        patch.object(
            main.analysis, "calculate_signal_potential", side_effect=fake_signals
        ),
        patch.object(main, "precompute_walk_forward_data", side_effect=fake_precompute),
    ):
        datasets = main._build_decay_datasets(
            pd.DataFrame({"entry": [1]}),
            pd.DataFrame({"transaction": [1]}),
            pd.DataFrame({"price": [1]}),
        )

    assert calls == list(main.PARAM_GRID["decay_lambda"])
    assert datasets == {decay: {"decay": decay} for decay in calls}


def test_holdout_is_chronologically_disjoint_from_selection():
    periods = {
        date: _period(date) for date in ("2024-04-01", "2024-07-01", "2024-10-01")
    }
    selection = main._subset_periods(
        periods, end=main.HOLDOUT_START, end_inclusive=False
    )
    holdout = main._subset_periods(periods, start=main.HOLDOUT_START)

    assert list(selection) == ["2024-04-01"]
    assert list(holdout) == ["2024-07-01", "2024-10-01"]
    assert set(selection).isdisjoint(holdout)


def test_all_trial_p_values_receive_bh_adjustment():
    p_values = np.array([0.001, 0.02, 0.20, 0.8])
    adjusted = main._bh_adjusted_pvalues(p_values)

    assert len(adjusted) == len(p_values)
    assert adjusted.tolist() == pytest.approx([0.004, 0.04, 0.2666667, 0.8])


def test_constant_and_shuffled_scorers_are_reproducible_canaries():
    ranking = _ranking()
    assert set(score_constant(ranking).values()) == {1.0}

    scorer_a = make_shuffled_scorer(
        lambda frame: dict(zip(frame["member"], frame["shrunk_alpha"])), seed=7
    )
    scorer_b = make_shuffled_scorer(
        lambda frame: dict(zip(frame["member"], frame["shrunk_alpha"])), seed=7
    )
    first = scorer_a(ranking)
    second = scorer_b(ranking)

    assert first == second
    assert sorted(first.values()) == sorted(ranking["shrunk_alpha"].tolist())


def test_failed_robustness_refuses_profit_claim():
    selected = pd.Series({"bh_q_value": 0.5})
    holdout = {
        "n_periods": 2,
        "mean_alpha_pct": -1.0,
        "total_return_pct": 1.0,
        "alpha_sharpe": -0.2,
        "stopped_early": False,
        "period_results": [],
    }
    passive = {"total_return_pct": 2.0}
    constant = {"alpha_sharpe": 0.0}
    nulls = pd.DataFrame({"phase": ["holdout"], "alpha_sharpe": [0.1]})

    robust, reasons = main._assess_holdout_robustness(
        selected, False, holdout, passive, constant, nulls
    )

    assert robust is False
    assert "No selection trial survived Benjamini-Hochberg correction" in reasons
    assert "Frozen strategy did not beat passive SPY on holdout" in reasons
