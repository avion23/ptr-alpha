"""Memoization layer for backtest subcomputations.

The parameter sweep calls `backtest_recommendations` 648 times with various
parameter combinations, but most subcomputations depend only on a SUBSET of
the 7 sweep parameters. Caching these subcomputations across combinations that
share their actual inputs turns the sweep from O(combos * work) into
O(unique_inputs * work), a 10-100x speedup in practice.

Cache keys use `id()` of the immutable input DataFrames plus scalar params.
This is safe because the sweep holds stable references to its signal caches,
prices, and transactions for the entire sweep lifetime, and these frames are
never mutated in place by any of the cached functions.

The cache is OPT-IN: passing `cache=None` (the default) preserves the original
behavior of every function, leaving the live CLI path unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_of_key(ts: Any) -> str:
    """Hashable iso-string for timestamps/dates."""
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


@dataclass
class BacktestCache:
    """Optional memoization for backtest subcomputations.

    Each dict maps a logical key (input identities + scalar params) to a
    cached result. `stats` tracks hits/misses for introspection.

    Lifetime: a single cache instance should be reused across many
    `backtest_recommendations` calls that share their immutable DataFrame
    inputs. Call `clear()` between unrelated runs.
    """

    # (id_sigs, id_prices, ticker, as_of_iso, horizon) -> float | None
    entry_values: dict[tuple, float | None] = field(default_factory=dict)
    # (id_sigs, id_prices, ticker, as_of_iso, horizon) -> list[np.ndarray]
    ticker_curves: dict[tuple, list] = field(default_factory=dict)
    # (id_sigs, id_prices, as_of_iso, horizon) -> list[np.ndarray]
    global_curves: dict[tuple, list] = field(default_factory=dict)
    # (id_prices, id_tx, ticker, disc_iso, tx_iso, as_of_iso) -> SignalFeatures
    signal_features: dict[tuple, Any] = field(default_factory=dict)
    # (id_sigs, horizon, threshold, bayes_prior, training_lookback, as_of_iso)
    #     -> pd.DataFrame (ranked members) or None
    rank_members: dict[tuple, Any] = field(default_factory=dict)
    # (id_sigs, id_prices, ticker, as_of_iso, horizon, bayes_prior)
    #     -> dict[member, (shrunk_wr, n)]
    ticker_member_perf: dict[tuple, dict] = field(default_factory=dict)
    # (id_tx, lookback_days, as_of_iso, ticker, id_sigs, horizon, threshold,
    #  training_lookback, bayes_prior, min_buyers, uncertainty_lambda)
    #     -> pd.DataFrame (1-row ticker score)
    ticker_scores: dict[tuple, Any] = field(default_factory=dict)

    stats: dict[str, int] = field(default_factory=dict)

    def _hit(self, name: str) -> None:
        self.stats[name + "_hits"] = self.stats.get(name + "_hits", 0) + 1

    def _miss(self, name: str) -> None:
        self.stats[name + "_misses"] = self.stats.get(name + "_misses", 0) + 1

    # -- entry_value ---------------------------------------------------------

    def entry_value_key(
        self, signals_df, prices_df, ticker: str, as_of_date, horizon: int
    ) -> tuple:
        return (
            id(signals_df), id(prices_df), ticker, _as_of_key(as_of_date), horizon,
        )

    def get_entry_value(self, signals_df, prices_df, ticker, as_of, horizon):
        k = self.entry_value_key(signals_df, prices_df, ticker, as_of, horizon)
        if k in self.entry_values:
            self._hit("entry_value")
            return True, self.entry_values[k]
        self._miss("entry_value")
        return False, None

    def set_entry_value(self, signals_df, prices_df, ticker, as_of, horizon, v0):
        k = self.entry_value_key(signals_df, prices_df, ticker, as_of, horizon)
        self.entry_values[k] = v0

    # -- ticker curves (for OU entry value computation) ----------------------

    def ticker_curves_key(
        self, signals_df, prices_df, ticker, as_of, horizon
    ) -> tuple:
        return (
            id(signals_df), id(prices_df), ticker, _as_of_key(as_of), horizon,
        )

    def get_ticker_curves(self, signals_df, prices_df, ticker, as_of, horizon):
        k = self.ticker_curves_key(signals_df, prices_df, ticker, as_of, horizon)
        if k in self.ticker_curves:
            self._hit("ticker_curves")
            return True, self.ticker_curves[k]
        self._miss("ticker_curves")
        return False, None

    def set_ticker_curves(self, signals_df, prices_df, ticker, as_of, horizon, curves):
        k = self.ticker_curves_key(signals_df, prices_df, ticker, as_of, horizon)
        self.ticker_curves[k] = curves

    # -- global curves (OU fallback) -----------------------------------------

    def global_curves_key(self, signals_df, prices_df, as_of, horizon) -> tuple:
        return (id(signals_df), id(prices_df), _as_of_key(as_of), horizon)

    def get_global_curves(self, signals_df, prices_df, as_of, horizon):
        k = self.global_curves_key(signals_df, prices_df, as_of, horizon)
        if k in self.global_curves:
            self._hit("global_curves")
            return True, self.global_curves[k]
        self._miss("global_curves")
        return False, None

    def set_global_curves(self, signals_df, prices_df, as_of, horizon, curves):
        k = self.global_curves_key(signals_df, prices_df, as_of, horizon)
        self.global_curves[k] = curves

    # -- signal_features -----------------------------------------------------

    def signal_features_key(
        self, prices_df, transactions_df, ticker, disc_date, tx_date, as_of
    ) -> tuple:
        tx_key = id(transactions_df)
        return (
            id(prices_df), tx_key, ticker,
            _as_of_key(disc_date),
            _as_of_key(tx_date) if tx_date is not None else None,
            _as_of_key(as_of),
        )

    def get_signal_features(
        self, prices_df, transactions_df, ticker, disc_date, tx_date, as_of
    ):
        k = self.signal_features_key(
            prices_df, transactions_df, ticker, disc_date, tx_date, as_of
        )
        if k in self.signal_features:
            self._hit("signal_features")
            return True, self.signal_features[k]
        self._miss("signal_features")
        return False, None

    def set_signal_features(
        self, prices_df, transactions_df, ticker, disc_date, tx_date, as_of, features
    ):
        k = self.signal_features_key(
            prices_df, transactions_df, ticker, disc_date, tx_date, as_of
        )
        self.signal_features[k] = features

    # -- rank_members --------------------------------------------------------

    def rank_members_key(
        self, signals_df, horizon, threshold, bayes_prior, training_lookback, as_of
    ) -> tuple:
        return (
            id(signals_df), horizon, threshold, bayes_prior,
            training_lookback, _as_of_key(as_of),
        )

    def get_rank_members(
        self, signals_df, horizon, threshold, bayes_prior, training_lookback, as_of
    ):
        k = self.rank_members_key(
            signals_df, horizon, threshold, bayes_prior, training_lookback, as_of
        )
        if k in self.rank_members:
            self._hit("rank_members")
            return True, self.rank_members[k]
        self._miss("rank_members")
        return False, None

    def set_rank_members(
        self, signals_df, horizon, threshold, bayes_prior, training_lookback, as_of,
        rankings,
    ):
        k = self.rank_members_key(
            signals_df, horizon, threshold, bayes_prior, training_lookback, as_of
        )
        self.rank_members[k] = rankings

    # -- ticker_member_perf (used by score_ticker_by_buyers) -----------------

    def ticker_member_perf_key(
        self, signals_df, ticker, horizon, bayes_prior, as_of
    ) -> tuple:
        return (
            id(signals_df), ticker, horizon, bayes_prior, _as_of_key(as_of),
        )

    def get_ticker_member_perf(
        self, signals_df, ticker, horizon, bayes_prior, as_of
    ):
        k = self.ticker_member_perf_key(
            signals_df, ticker, horizon, bayes_prior, as_of
        )
        if k in self.ticker_member_perf:
            self._hit("ticker_member_perf")
            return True, self.ticker_member_perf[k]
        self._miss("ticker_member_perf")
        return False, None

    def set_ticker_member_perf(
        self, signals_df, ticker, horizon, bayes_prior, as_of, perf
    ):
        k = self.ticker_member_perf_key(
            signals_df, ticker, horizon, bayes_prior, as_of
        )
        self.ticker_member_perf[k] = perf

    # -- ticker_scores (full result of score_ticker_by_buyers) ---------------

    def ticker_scores_key(
        self, transactions_df, lookback_days, as_of, ticker,
        signals_df, horizon, threshold, training_lookback, bayes_prior,
        min_buyers, uncertainty_lambda,
    ) -> tuple:
        return (
            id(transactions_df), lookback_days, _as_of_key(as_of), ticker,
            id(signals_df), horizon, threshold, training_lookback, bayes_prior,
            min_buyers, uncertainty_lambda,
        )

    def get_ticker_scores(
        self, transactions_df, lookback_days, as_of, ticker,
        signals_df, horizon, threshold, training_lookback, bayes_prior,
        min_buyers, uncertainty_lambda,
    ):
        k = self.ticker_scores_key(
            transactions_df, lookback_days, as_of, ticker,
            signals_df, horizon, threshold, training_lookback, bayes_prior,
            min_buyers, uncertainty_lambda,
        )
        if k in self.ticker_scores:
            self._hit("ticker_scores")
            return True, self.ticker_scores[k]
        self._miss("ticker_scores")
        return False, None

    def set_ticker_scores(
        self, transactions_df, lookback_days, as_of, ticker,
        signals_df, horizon, threshold, training_lookback, bayes_prior,
        min_buyers, uncertainty_lambda, score,
    ):
        k = self.ticker_scores_key(
            transactions_df, lookback_days, as_of, ticker,
            signals_df, horizon, threshold, training_lookback, bayes_prior,
            min_buyers, uncertainty_lambda,
        )
        # Defensive copy — callers mutate the returned frame (insert, .loc set)
        self.ticker_scores[k] = score.copy() if hasattr(score, "copy") else score

    # -- maintenance ---------------------------------------------------------

    def clear(self) -> None:
        self.entry_values.clear()
        self.ticker_curves.clear()
        self.global_curves.clear()
        self.signal_features.clear()
        self.rank_members.clear()
        self.ticker_member_perf.clear()
        self.ticker_scores.clear()
        self.stats.clear()

    def __len__(self) -> int:
        return (
            len(self.entry_values)
            + len(self.ticker_curves)
            + len(self.global_curves)
            + len(self.signal_features)
            + len(self.rank_members)
            + len(self.ticker_member_perf)
            + len(self.ticker_scores)
        )
