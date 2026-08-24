import unittest
from dataclasses import FrozenInstanceError
from datetime import date
from typing import Any, cast

import pandas as pd

from analyzer.decision_adapters import (
    BacktestEvidenceAdapter,
    ConsensusForecastAdapter,
    TopNPolicy,
    recommendations_to_forecasts,
)
from analyzer.decision_contracts import (
    Evidence,
    EvidenceReport,
    Forecast,
    MarketSnapshot,
    PortfolioState,
    Provenance,
    TargetPortfolio,
    TargetPosition,
)


AS_OF = date(2025, 1, 2)


def _forecast(
    event_id: str,
    ticker: str,
    score: float | None = None,
    **kwargs,
) -> Forecast:
    return Forecast(
        event_id=event_id,
        ticker=ticker,
        as_of=AS_OF,
        horizon_days=30,
        ranking_score=score,
        **kwargs,
    )


class TestDecisionContracts(unittest.TestCase):
    def test_contracts_are_frozen_and_nested_values_are_copied(self):
        metadata = {"buyers": 2}
        forecast = _forecast("event-1", "AAA", 4.5, metadata=metadata)
        metadata["buyers"] = 99

        self.assertEqual(forecast.metadata, (("buyers", 2),))
        with self.assertRaises(FrozenInstanceError):
            setattr(forecast, "ticker", "BBB")

        prices = {"AAA": 100.0}
        market = MarketSnapshot(AS_OF, cast(Any, prices))
        prices["AAA"] = 1.0
        self.assertEqual(market.prices, (("AAA", 100.0),))
        self.assertEqual(
            PortfolioState(100.0, cast(Any, ["AAA"])).held_tickers, ("AAA",)
        )

    def test_rank_score_never_becomes_expected_return(self):
        recommendations = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "signal_score": [8.25],
                # A legacy caller may carry a diagnostic with this name; the
                # rank adapter must not reinterpret it as a forecast.
                "expected_net_alpha": [123.0],
            }
        )

        (forecast,) = recommendations_to_forecasts(recommendations, AS_OF, 30)

        self.assertEqual(forecast.ranking_score, 8.25)
        self.assertIsNone(forecast.expected_net_alpha)
        self.assertIsNone(forecast.net_alpha_std)
        self.assertIsNone(forecast.probability_positive)

    def test_realized_columns_are_rejected_before_forecast_conversion(self):
        recommendations = pd.DataFrame(
            {"ticker": ["AAA"], "signal_score": [1.0], "bt_return_pct": [5.0]}
        )

        with self.assertRaisesRegex(ValueError, "bt_return_pct"):
            recommendations_to_forecasts(recommendations, AS_OF, 30)

    def test_provenance_availability_cannot_follow_decision_as_of(self):
        future = date(2025, 1, 3)
        with self.assertRaisesRegex(ValueError, "on or before"):
            _forecast(
                "event-future",
                "AAA",
                provenance=Provenance(available_date=future),
            )
        with self.assertRaisesRegex(ValueError, "on or before"):
            Evidence(
                "event-future",
                "AAA",
                AS_OF,
                30,
                provenance=Provenance(available_date=future),
            )

    def test_recommendation_event_ids_are_namespaced_and_unique(self):
        recommendations = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "signal_score": [1.0],
                "event_id": ["legacy-event"],
            }
        )
        (forecast,) = recommendations_to_forecasts(recommendations, AS_OF, 30)
        self.assertTrue(forecast.event_id.startswith("forecast:"))
        self.assertNotEqual(forecast.event_id, "legacy-event")

        duplicate_ids = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "signal_score": [1.0, 0.5],
                "event_id": ["same-event", "same-event"],
            }
        )
        with self.assertRaisesRegex(ValueError, "unique event_ids"):
            recommendations_to_forecasts(duplicate_ids, AS_OF, 30)

    def test_recommendations_and_policy_reject_duplicate_tickers(self):
        recommendations = pd.DataFrame(
            {"ticker": ["AAA", "AAA"], "signal_score": [2.0, 1.0]}
        )
        with self.assertRaisesRegex(ValueError, "duplicate tickers"):
            recommendations_to_forecasts(recommendations, AS_OF, 30)

        with self.assertRaisesRegex(ValueError, "duplicate tickers"):
            TopNPolicy().allocate(
                [_forecast("event-a", "AAA", 2.0), _forecast("event-b", "AAA", 1.0)]
            )

    def test_recommendations_require_one_common_as_of_and_horizon(self):
        mixed_as_of = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "signal_score": [2.0, 1.0],
                "as_of_date": [AS_OF, date(2025, 1, 3)],
            }
        )
        with self.assertRaisesRegex(ValueError, "as_of_date disagrees"):
            recommendations_to_forecasts(mixed_as_of, AS_OF, 30)

        mixed_horizon = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "signal_score": [2.0],
                "horizon_days": [60],
            }
        )
        with self.assertRaisesRegex(ValueError, "horizon disagrees"):
            recommendations_to_forecasts(mixed_horizon, AS_OF, 30)

    def test_top_n_policy_is_rank_only(self):
        ranked = _forecast("ranked", "AAA", 2.0)
        economic_only = _forecast(
            "economic", "BBB", None, expected_net_alpha=999.0
        )

        portfolio = TopNPolicy(top_n=2).allocate([economic_only, ranked])

        self.assertEqual(portfolio.tickers, ("AAA",))
        self.assertEqual(portfolio.positions[0].forecast, ranked)

        cash = TopNPolicy().allocate([economic_only])
        self.assertEqual(cash.positions, ())
        self.assertEqual(cash.cash_weight, 1.0)

    def test_consensus_adapter_preserves_legacy_recommender_and_rank_semantics(self):
        calls = []

        def recommender(signals, transactions, as_of, **kwargs):
            calls.append((signals, transactions, as_of, kwargs))
            return pd.DataFrame({"ticker": ["AAA"], "signal_score": [3.0]})

        adapter = ConsensusForecastAdapter(recommendation_fn=recommender)
        (forecast,) = adapter.forecast(
            pd.DataFrame(), pd.DataFrame(), AS_OF, horizon=30
        )

        self.assertEqual(forecast.ticker, "AAA")
        self.assertEqual(forecast.ranking_score, 3.0)
        self.assertIsNone(forecast.expected_net_alpha)
        self.assertEqual(calls[0][2], pd.Timestamp(AS_OF))
        self.assertEqual(calls[0][3]["scoring_mode"], "consensus")

    def test_evidence_adapter_keeps_realized_values_on_evidence_only(self):
        forecasts = (_forecast("event-a", "AAA", 2.0), _forecast("event-b", "BBB", 1.0))

        def evaluator(recommendations, prices, as_of, horizon, **kwargs):
            # Simulate the production evaluator omitting the first row because
            # no executable outcome was available.
            result = recommendations.iloc[[1]].copy()
            result["bt_return_pct"] = [7.5]
            result["bt_alpha_pct"] = [3.25]
            result["bt_coverage"] = ["complete"]
            result["bt_stale_exit"] = [False]
            result["bt_delisted"] = [False]
            result.attrs["n_no_price"] = 1
            return result

        report = BacktestEvidenceAdapter(evaluator=evaluator).evaluate_report(
            forecasts, pd.DataFrame(), AS_OF, 30
        )

        self.assertEqual([item.event_id for item in report.evidence], ["event-a", "event-b"])
        self.assertEqual(report.evidence[0].coverage, "unavailable")
        self.assertEqual(report.evidence[1].realized_return_pct, 7.5)
        self.assertEqual(report.evidence[1].realized_alpha_pct, 3.25)
        self.assertEqual(report.n_no_price, 1)
        self.assertIsNone(forecasts[1].expected_net_alpha)

    def test_evidence_alignment_fails_closed_without_a_row_key(self):
        forecasts = (_forecast("event-a", "AAA", 2.0), _forecast("event-b", "BBB", 1.0))

        def unsafe_evaluator(recommendations, prices, as_of, horizon, **kwargs):
            return pd.DataFrame(
                {"ticker": ["BBB"], "bt_return_pct": [4.0], "bt_coverage": ["complete"]}
            )

        with self.assertRaisesRegex(ValueError, "safely aligned"):
            BacktestEvidenceAdapter(evaluator=unsafe_evaluator).evaluate_report(
                forecasts, pd.DataFrame(), AS_OF, 30
            )

    def test_realized_decision_frame_is_rejected_by_evidence_adapter(self):
        decision_frame = pd.DataFrame(
            {"ticker": ["AAA"], "signal_score": [1.0], "bt_alpha_pct": [2.0]}
        )

        with self.assertRaisesRegex(ValueError, "bt_alpha_pct"):
            BacktestEvidenceAdapter(evaluator=lambda *args: pd.DataFrame()).evaluate(
                decision_frame, pd.DataFrame(), as_of=AS_OF, horizon_days=30
            )

    def test_evidence_is_immutable_and_rejects_mutable_provenance(self):
        evidence = Evidence("event-a", "AAA", AS_OF, 30, realized_return_pct=2.0)
        with self.assertRaises(FrozenInstanceError):
            setattr(evidence, "coverage", "complete")

        with self.assertRaises(TypeError):
            Forecast("event-a", AS_OF, 30, provenance=cast(Any, {"source": "legacy"}))
        with self.assertRaises(ValueError):
            Forecast("event-a", AS_OF, 30, metadata={"bt_return_pct": 2.0})

    def test_target_position_duplicates_must_match_embedded_forecast(self):
        forecast = _forecast("event-a", "AAA", 2.0, provenance=Provenance(source="filing"))
        with self.assertRaisesRegex(ValueError, "event_id"):
            TargetPosition(
                event_id="wrong",
                ticker="AAA",
                weight=1.0,
                rank=1,
                as_of=AS_OF,
                horizon_days=30,
                forecast=forecast,
                provenance=forecast.provenance,
            )
        with self.assertRaisesRegex(ValueError, "provenance"):
            TargetPosition(
                event_id="event-a",
                ticker="AAA",
                weight=1.0,
                rank=1,
                as_of=AS_OF,
                horizon_days=30,
                forecast=forecast,
            )

    def test_target_portfolio_requires_common_context_and_complete_weights(self):
        first = _forecast("event-a", "AAA", 2.0)
        second = Forecast(
            event_id="event-b",
            ticker="BBB",
            as_of=date(2025, 1, 3),
            horizon_days=60,
            ranking_score=1.0,
        )
        positions = tuple(
            TargetPosition(
                event_id=item.event_id,
                ticker=item.ticker,
                weight=0.5,
                rank=rank,
                as_of=item.as_of,
                horizon_days=item.horizon_days,
                forecast=item,
                provenance=item.provenance,
            )
            for rank, item in enumerate((first, second), start=1)
        )
        with self.assertRaisesRegex(ValueError, "share as_of"):
            TargetPortfolio(AS_OF, positions=positions, cash_weight=0.0)

        position = TargetPosition(
            event_id=first.event_id,
            ticker=first.ticker,
            weight=0.5,
            rank=1,
            as_of=first.as_of,
            horizon_days=first.horizon_days,
            forecast=first,
            provenance=first.provenance,
        )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            TargetPortfolio(AS_OF, positions=(position,), cash_weight=0.0)

        complete = TargetPortfolio(AS_OF, positions=(position,), cash_weight=0.5)
        self.assertEqual(complete.positions[0].weight, 0.5)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            TargetPortfolio(AS_OF, positions=(position,), cash_weight=0.6)

    def test_target_portfolio_rejects_duplicate_tickers_and_event_ids(self):
        first = _forecast("event-a", "AAA", 2.0)
        second_ticker = _forecast("event-b", "AAA", 1.0)
        positions = tuple(
            TargetPosition(
                event_id=item.event_id,
                ticker=item.ticker,
                weight=0.5,
                rank=rank,
                as_of=item.as_of,
                horizon_days=item.horizon_days,
                forecast=item,
                provenance=item.provenance,
            )
            for rank, item in enumerate((first, second_ticker), start=1)
        )
        with self.assertRaisesRegex(ValueError, "duplicate tickers"):
            TargetPortfolio(AS_OF, positions=positions, cash_weight=0.0)

        second_event = _forecast("event-a", "BBB", 1.0)
        positions = tuple(
            TargetPosition(
                event_id=item.event_id,
                ticker=item.ticker,
                weight=0.5,
                rank=rank,
                as_of=item.as_of,
                horizon_days=item.horizon_days,
                forecast=item,
                provenance=item.provenance,
            )
            for rank, item in enumerate((first, second_event), start=1)
        )
        with self.assertRaisesRegex(ValueError, "unique event_ids"):
            TargetPortfolio(AS_OF, positions=positions, cash_weight=0.0)

    def test_policy_rejects_mixed_context_and_market_date_mismatch(self):
        first = _forecast("event-a", "AAA", 2.0)
        second = Forecast(
            event_id="event-b",
            ticker="BBB",
            as_of=AS_OF,
            horizon_days=60,
            ranking_score=1.0,
        )
        with self.assertRaisesRegex(ValueError, "share as_of"):
            TopNPolicy().allocate([first, second])
        with self.assertRaisesRegex(ValueError, "market as_of"):
            TopNPolicy().allocate(
                [first],
                MarketSnapshot(date(2025, 1, 3), cast(Any, {"AAA": 100.0})),
            )

    def test_empty_policy_and_evidence_are_stable(self):
        market = MarketSnapshot(AS_OF, cast(Any, {"AAA": 100.0}))
        empty_portfolio = TopNPolicy().allocate([], market)
        self.assertEqual(empty_portfolio.positions, ())
        self.assertEqual(empty_portfolio.cash_weight, 1.0)

    def test_empty_evidence_is_stable_and_does_not_call_evaluator(self):
        calls = []

        def evaluator(*args, **kwargs):
            calls.append((args, kwargs))
            return pd.DataFrame()

        empty = TargetPortfolio(AS_OF, positions=(), cash_weight=1.0)
        report = BacktestEvidenceAdapter(evaluator=evaluator).evaluate_report(
            empty, pd.DataFrame(), horizon=30
        )
        self.assertEqual(report.evidence, ())
        self.assertEqual(calls, [])
        self.assertEqual(
            BacktestEvidenceAdapter(evaluator=evaluator).evaluate(
                [], pd.DataFrame(), as_of=AS_OF, horizon_days=30
            ),
            (),
        )

    def test_evidence_report_requires_common_as_of(self):
        evidence = Evidence("event-a", "AAA", AS_OF, 30)
        later = Evidence("event-b", "BBB", date(2025, 1, 3), 30)
        with self.assertRaisesRegex(ValueError, "report as_of"):
            EvidenceReport(AS_OF, 30, evidence=(evidence, later))

    def test_evidence_report_requires_common_horizon_and_unique_event_ids(self):
        evidence = Evidence("event-a", "AAA", AS_OF, 30)
        later_horizon = Evidence("event-b", "BBB", AS_OF, 60)
        with self.assertRaisesRegex(ValueError, "report horizon_days"):
            EvidenceReport(AS_OF, 30, evidence=(evidence, later_horizon))

        duplicate = Evidence("event-a", "BBB", AS_OF, 30)
        with self.assertRaisesRegex(ValueError, "unique event_ids"):
            EvidenceReport(AS_OF, 30, evidence=(evidence, duplicate))

    def test_evidence_adapter_requires_requested_context_to_match_forecasts(self):
        forecasts = (_forecast("event-a", "AAA", 2.0),)
        adapter = BacktestEvidenceAdapter(evaluator=lambda *args: pd.DataFrame())
        with self.assertRaisesRegex(ValueError, "as_of_date must match"):
            adapter.evaluate_report(
                forecasts,
                pd.DataFrame(),
                as_of_date=date(2025, 1, 3),
                horizon=30,
            )
        with self.assertRaisesRegex(ValueError, "horizon must match"):
            adapter.evaluate_report(
                forecasts,
                pd.DataFrame(),
                as_of_date=AS_OF,
                horizon=60,
            )

    def test_evidence_adapter_rejects_evaluator_date_or_horizon_drift(self):
        forecasts = (_forecast("event-a", "AAA", 2.0),)

        def date_drifting_evaluator(recommendations, prices, as_of, horizon, **kwargs):
            result = recommendations.iloc[[0]].copy()
            result["bt_as_of"] = [date(2025, 1, 3)]
            return result

        with self.assertRaisesRegex(ValueError, "evidence as_of disagrees"):
            BacktestEvidenceAdapter(evaluator=date_drifting_evaluator).evaluate_report(
                forecasts, pd.DataFrame(), AS_OF, 30
            )

        def horizon_drifting_evaluator(
            recommendations, prices, as_of, horizon, **kwargs
        ):
            result = recommendations.iloc[[0]].copy()
            result["bt_horizon_days"] = [60]
            return result

        with self.assertRaisesRegex(ValueError, "evidence horizon disagrees"):
            BacktestEvidenceAdapter(
                evaluator=horizon_drifting_evaluator
            ).evaluate_report(forecasts, pd.DataFrame(), AS_OF, 30)


if __name__ == "__main__":
    unittest.main()
