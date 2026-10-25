"""Read-only evaluation of the current, two-month-capped setup rule.

The primary, pre-registered decision rule is: VALIDATED only if the date-
clustered mean 90-day total-return excess of the highest score quintile is
positive and its deterministic 95% date-cluster-bootstrap CI is entirely
above zero. Otherwise report NOT VALIDATED. If fewer than two usable outcome
clusters exist, report NOT VALIDATABLE rather than treating low coverage as an
efficacy result. Secondary ranked-vs-eligible-
unranked and ranked-vs-ineligible contrasts are descriptive; they do not
replace this hurdle or receive multiplicity-adjusted claims.

Default decision dates are quarterly from 2023-01-15 through the latest
quarterly date whose exact 90-day endpoints are available in certified SPY
prices. Candidate inputs use the production ingestion-frame loaders and
production scorer; labels use the production evaluator fed only served,
comparable adjusted-close generations. Outcomes are future labels, never
scoring inputs. All cohort rows, including unavailable labels, are retained.

Power statement: use every supported quarterly cohort in that range; no
minimum economically meaningful effect was supplied, so no nominal power or
MDE is claimed. The run reports its coverage-limited cluster count. With fewer
than two outcome clusters, variance, a cluster CI, and power are not estimable;
the evaluation is NOT VALIDATABLE.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analyzer import positions
from analyzer.actors import weights_for_candidates
from analyzer.backtest.evaluate import evaluate_backtest
from analyzer.manager_watchlist import watchlist_actor_ids
from analyzer.pipeline import eligible_events
from analyzer.price_source import _PRICE_BASIS, price_generation_schema_exists
from analyzer.positions import (
    coalesce_actor_variants,
    dedupe_member_names,
    gate_closed_events,
    holdings_candidates,
)
from analyzer.persistence.database import Database
from analyzer.persistence.price_repository import next_nyse_session, previous_nyse_session
from analyzer.setups import EVIDENCE_MAX_AGE_MONTHS, score as score_setups

import backtest_assumptions as old_harness

DB_PATH = Path("/home/ubuntu/insider-trading/data/congress.duckdb")
OUTPUT_DIR = Path("/tmp/opencode/reco-20261025/fix-w4")
HORIZON_DAYS = 90
ENTRY_SLIPPAGE_BPS = 10.0
EXIT_SLIPPAGE_BPS = 10.0
ENTRY_TOLERANCE = 1.05
EVENT_HALF_LIFE_DAYS = 90
BOOTSTRAP_SEED = 20261025
BOOTSTRAP_REPS = 10_000
FRAME = "ingestion"
POWER_STATEMENT = (
    "All coverage-supported quarterly cohorts are used. No minimum economically "
    "meaningful effect was supplied; nominal power/MDE is not claimed. Fewer than "
    "two outcome clusters makes variance, cluster CI, and power unestimable."
)
_DATE_FIELDS = ("as_of", "disclosure_date", "event_date", "max_input_date")


def assert_no_future_rows(frame: pd.DataFrame, decision: date) -> None:
    """Fail closed if any date-bearing candidate input is later than decision."""
    cutoff = pd.Timestamp(decision).normalize()
    for column in _DATE_FIELDS:
        if column not in frame:
            continue
        values = pd.to_datetime(frame[column], errors="coerce")
        future = values.notna() & (values.dt.normalize() > cutoff)
        if future.any():
            sample = frame.loc[future, [column]].head(3).to_dict("records")
            raise ValueError(
                f"look-ahead: {column} has {int(future.sum())} rows after "
                f"{decision}; sample={sample}"
            )


def _certified_states(db, tickers: list[str]) -> dict[str, dict]:
    symbols = sorted({str(t).strip().upper() for t in tickers if t} | {"SPY"})
    if not price_generation_schema_exists(db):
        return {
            ticker: {
                "generation": None,
                "comparability": "unknown",
                "basis": None,
                "reason": "price generation metadata unavailable",
            }
            for ticker in symbols
        }
    rows = db.conn.execute(
        "SELECT s.ticker, s.served_generation, s.comparability, s.reason, g.basis "
        "FROM price_generation_state s LEFT JOIN price_generations g "
        "ON g.ticker=s.ticker AND g.generation=s.served_generation "
        "WHERE s.ticker=ANY(?)",
        [symbols],
    ).fetchall()
    states = {
        str(ticker).strip().upper(): {
            "generation": generation,
            "comparability": comparability,
            "basis": basis,
            "reason": reason,
        }
        for ticker, generation, comparability, reason, basis in rows
    }
    for ticker in symbols:
        states.setdefault(
            ticker,
            {
                "generation": None,
                "comparability": "unknown",
                "basis": None,
                "reason": "no served price generation",
            },
        )
    return states


def _certified(state: dict) -> bool:
    return (
        state.get("comparability") == "comparable"
        and state.get("basis") == _PRICE_BASIS
        and isinstance(state.get("generation"), str)
        and bool(state["generation"].strip())
    )


def default_decision_dates(db) -> tuple[list[date], list[dict]]:
    """Use the fullest quarterly span supported by certified SPY endpoints."""
    state = _certified_states(db, ["SPY"])["SPY"]
    if not _certified(state):
        raise RuntimeError(f"SPY has no certified total-return generation: {state}")
    rows = db.conn.execute(
        "SELECT date FROM price_generation_values "
        "WHERE ticker='SPY' AND generation=? AND basis=? ORDER BY date",
        [state["generation"], _PRICE_BASIS],
    ).fetchall()
    spy_dates = {row[0] for row in rows}
    if not spy_dates:
        raise RuntimeError("certified SPY generation has no price dates")
    latest = min(max(spy_dates), date.today())
    decisions, excluded = [], []
    month_index = 0
    while True:
        month = 1 + month_index * 3
        year = 2023 + (month - 1) // 12
        month = (month - 1) % 12 + 1
        decision = date(year, month, 15)
        if decision > latest:
            break
        entry = next_nyse_session(pd.Timestamp(decision)).date()
        exit_day = previous_nyse_session(
            pd.Timestamp(entry) + pd.Timedelta(days=HORIZON_DAYS)
        ).date()
        if decision >= date.today():
            excluded.append({"as_of": str(decision), "reason": "decision_not_historical"})
        elif entry not in spy_dates or exit_day not in spy_dates:
            excluded.append(
                {
                    "as_of": str(decision),
                    "reason": "certified_spy_missing_exact_endpoint",
                    "entry_date": str(entry),
                    "exit_date": str(exit_day),
                }
            )
        else:
            decisions.append(decision)
        month_index += 1
    return decisions, excluded


def _decision_quotes(db, tickers: list[str], decision: date) -> dict:
    symbols = sorted(set(tickers))
    if not symbols:
        return {}
    rows = db.conn.execute(
        "SELECT ticker, date, close FROM prices WHERE ticker=ANY(?) AND date<=? "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC)=1",
        [symbols, decision],
    ).fetchall()
    return {
        str(ticker).strip().upper(): (float(close), day)
        for ticker, day, close in rows
        if close is not None and math.isfinite(float(close)) and float(close) > 0
    }


def _production_candidates(db, decision: date) -> pd.DataFrame:
    """Rebuild the production ingestion-frame candidate set at one cutoff."""
    events_all = eligible_events(db, decision, frame_scope=decision)
    assert_no_future_rows(events_all, decision)
    if events_all.empty:
        return events_all

    rows = []
    holdings_cache: dict[str, pd.DataFrame] = {}

    def holdings_for(member: str) -> pd.DataFrame:
        if member not in holdings_cache:
            holdings_cache[member] = holdings_candidates(
                db, member, decision, enable_13f_corroboration=False
            )
            assert_no_future_rows(holdings_cache[member], decision)
        return holdings_cache[member]

    for symbol in sorted(events_all["ticker"].astype(str).str.upper().unique()):
        events = events_all.loc[
            events_all["ticker"].astype(str).str.upper() == symbol
        ].copy()
        kinds = {}
        for raw_actor in events["actor_id"].dropna():
            text = str(raw_actor)
            kind, _, name = text.partition(":")
            if name:
                kinds.setdefault(name, kind)
        frames, congress_frames = [], []
        for member in dedupe_member_names(sorted(kinds)):
            lots = holdings_for(member)
            if lots.empty:
                continue
            lots = lots.loc[lots["ticker"].astype(str).str.upper() == symbol].copy()
            if not lots.empty:
                frames.append(lots)
                congress_frames.append(lots.loc[lots["kind"] == "congress"].copy())
        positions_frame = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        events = coalesce_actor_variants(events, positions_frame)
        events = gate_closed_events(
            events, positions_frame, enable_13f_corroboration=False
        )
        candidates = pd.concat(
            [part for part in congress_frames + [events] if not part.empty],
            ignore_index=True,
        )
        if not candidates.empty:
            candidates["ticker"] = candidates["ticker"].astype(str).str.upper()
            assert_no_future_rows(candidates, decision)
            rows.append(candidates)
    if not rows:
        return pd.DataFrame(columns=events_all.columns)
    result = pd.concat(rows, ignore_index=True)
    assert_no_future_rows(result, decision)
    return result


def _label_outcomes(db, tickers: list[str], decision: date) -> dict[str, dict]:
    """Label candidate tickers with production math over certified generations."""
    symbols = sorted({str(t).strip().upper() for t in tickers if t})
    if not symbols:
        return {}
    states = _certified_states(db, symbols)
    spy_state = states["SPY"]
    entry = next_nyse_session(pd.Timestamp(decision)).date()
    exit_day = previous_nyse_session(
        pd.Timestamp(entry) + pd.Timedelta(days=HORIZON_DAYS)
    ).date()
    labels = {}
    evaluable = []
    for ticker in symbols:
        state = states[ticker]
        bad = []
        if not _certified(state):
            bad.append(
                f"{ticker}:generation_not_certified"
                f"(comparability={state.get('comparability')},basis={state.get('basis')})"
            )
        if not _certified(spy_state):
            bad.append(
                "SPY:generation_not_certified"
                f"(comparability={spy_state.get('comparability')},basis={spy_state.get('basis')})"
            )
        if bad:
            labels[ticker] = {
                "fill_date": None,
                "fill_px": None,
                "exit_date": None,
                "fwd": None,
                "spy": None,
                "excess": None,
                "outcome_status": "uncertifiable",
                "exclusion_reason": "; ".join(bad),
                "price_generation_id": state.get("generation"),
                "spy_generation_id": spy_state.get("generation"),
                "price_basis": state.get("basis"),
                "spy_price_basis": spy_state.get("basis"),
            }
        else:
            evaluable.append(ticker)

    if not evaluable:
        return labels
    prices = db.conn.execute(
        "SELECT v.ticker, v.date, v.close FROM price_generation_values v "
        "JOIN price_generation_state s ON s.ticker=v.ticker "
        "AND s.served_generation=v.generation AND s.comparability='comparable' "
        "JOIN price_generations g ON g.ticker=v.ticker AND g.generation=v.generation "
        "AND g.basis=? WHERE v.ticker=ANY(?) AND v.date BETWEEN ? AND ? "
        "ORDER BY v.date, v.ticker",
        [_PRICE_BASIS, [*evaluable, "SPY"], entry, exit_day],
    ).fetchdf()
    matrix = (
        prices.pivot(index="date", columns="ticker", values="close").sort_index()
        if not prices.empty
        else pd.DataFrame()
    )
    evaluated = evaluate_backtest(
        pd.DataFrame({"ticker": evaluable, "instrument_type": "stock"}),
        matrix,
        pd.Timestamp(decision),
        HORIZON_DAYS,
        entry_slippage_bps=ENTRY_SLIPPAGE_BPS,
        exit_slippage_bps=EXIT_SLIPPAGE_BPS,
    )
    for row in evaluated.itertuples(index=False):
        ticker = str(row.ticker).strip().upper()
        complete = row.bt_coverage == "complete" and pd.notna(row.bt_alpha_pct)
        labels[ticker] = {
            "fill_date": str(row.bt_entry_date) if pd.notna(row.bt_entry_date) else None,
            "fill_px": (
                float(row.bt_entry_price)
                if pd.notna(row.bt_entry_price)
                else None
            ),
            "exit_date": str(row.bt_exit_date) if complete else None,
            "fwd": float(row.bt_return_pct) if complete else None,
            "spy": float(row.bt_spy_return_pct) if complete else None,
            "excess": float(row.bt_alpha_pct) if complete else None,
            "outcome_status": "available" if complete else "outcome_missing",
            "exclusion_reason": (
                None
                if complete
                else str(row.bt_unavailable_reason or "production_evaluator_unavailable")
            ),
            "price_generation_id": states[ticker].get("generation"),
            "spy_generation_id": spy_state.get("generation"),
            "price_basis": states[ticker].get("basis"),
            "spy_price_basis": spy_state.get("basis"),
        }
    return labels


def _valid_refs(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame.get("entry_ref"), errors="coerce")
    return values.notna() & np.isfinite(values) & values.gt(0)


def _current_rule_refs(frame: pd.DataFrame, decision: date) -> pd.Series:
    """Match references the production scorer can actually consume."""
    disclosures = pd.to_datetime(frame["disclosure_date"], errors="coerce")
    cutoff = pd.Timestamp(decision) - pd.DateOffset(months=EVIDENCE_MAX_AGE_MONTHS)
    blocked = frame.get("blocked_reason", pd.Series(None, index=frame.index)).map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    corroboration = frame.get("corroboration", pd.Series(False, index=frame.index))
    return (
        _valid_refs(frame)
        & disclosures.notna()
        & disclosures.ge(cutoff)
        & ~blocked
        & ~corroboration.fillna(False).eq(True)
    )


def _missing_current_reference_reason(frame: pd.DataFrame, decision: date) -> str:
    valid = _valid_refs(frame)
    if not valid.any():
        return "no_valid_entry_reference"
    dates = pd.to_datetime(frame.loc[valid, "disclosure_date"], errors="coerce")
    cutoff = pd.Timestamp(decision) - pd.DateOffset(months=EVIDENCE_MAX_AGE_MONTHS)
    if not (dates.notna() & dates.ge(cutoff)).any():
        return "references_outside_two_month_cap"
    blocked = frame.loc[valid].get("blocked_reason", pd.Series(None, index=frame.index[valid]))
    reasons = sorted({
        value.strip() for value in blocked.dropna().astype(str) if value.strip()
    })
    if reasons:
        return "blocked_reference:" + "|".join(reasons)
    corroboration = frame.loc[valid].get(
        "corroboration", pd.Series(False, index=frame.index[valid])
    ).fillna(False).eq(True)
    if corroboration.all():
        return "corroboration_only"
    return "no_current_rule_reference"


def _score_quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["score_quintile"] = None
    ranked = panel.loc[panel["ranked"] & panel["score"].notna()]
    for _, indices in ranked.groupby("as_of", sort=True).groups.items():
        values = pd.to_numeric(panel.loc[indices, "score"], errors="coerce")
        if len(values) == 1:
            panel.loc[indices, "score_quintile"] = "Q5"
            continue
        if values.nunique() < 2:
            panel.loc[indices, "score_quintile"] = "Q3"
            continue
        pct = (values.rank(method="average") - 1) / (len(values) - 1)
        bins = np.minimum(5, np.floor(pct * 5).astype(int) + 1)
        panel.loc[indices, "score_quintile"] = [f"Q{value}" for value in bins]
    return panel


def _cohort_rows(
    candidates: pd.DataFrame,
    ranked: pd.DataFrame,
    blocked: pd.DataFrame,
    labels: dict[str, dict],
    quotes: dict,
    decision: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_by_ticker = (
        ranked.set_index("ticker")["score"].to_dict() if not ranked.empty else {}
    )
    ranked_tickers = set(score_by_ticker)
    blocked_reason = {}
    if not blocked.empty:
        for (ticker, actor), rows in blocked.groupby(["ticker", "actor_id"]):
            blocked_reason[(str(ticker), str(actor))] = "; ".join(
                sorted(set(rows["reason"].dropna().astype(str)))
            )

    ticker_rows = []
    actor_rows = []
    for ticker, group in candidates.groupby("ticker", sort=True):
        ticker = str(ticker).upper()
        valid = _current_rule_refs(group, decision)
        any_valid = _valid_refs(group)
        max_ref = float(pd.to_numeric(group.loc[valid, "entry_ref"]).max()) if valid.any() else None
        any_max_ref = (
            float(pd.to_numeric(group.loc[any_valid, "entry_ref"]).max())
            if any_valid.any() else None
        )
        outcome = labels.get(ticker, {
            "fill_date": None, "fill_px": None, "exit_date": None,
            "fwd": None, "spy": None, "excess": None,
            "outcome_status": "outcome_missing", "exclusion_reason": "label_not_returned",
            "price_generation_id": None, "spy_generation_id": None,
            "price_basis": None, "spy_price_basis": None,
        })
        ticker_block_reasons = set(
            blocked.loc[blocked["ticker"].astype(str) == ticker, "reason"].dropna().astype(str)
        ) if not blocked.empty else set()
        quote_unusable_reason = (
            "decision_quote_missing" if ticker not in quotes
            else "stale_decision_quote" if "stale quote" in ticker_block_reasons
            else "future_decision_quote" if "future quote" in ticker_block_reasons
            else None
        )
        eligible = (
            bool(outcome["fill_px"] <= ENTRY_TOLERANCE * max_ref)
            if (
                outcome.get("fill_px") is not None
                and max_ref is not None
                and quote_unusable_reason is None
            )
            else None
        )
        is_ranked = ticker in ranked_tickers
        if is_ranked and eligible is False:
            cohort_group = "ranked_ineligible"
        elif is_ranked and eligible is True:
            cohort_group = "ranked_eligible"
        elif is_ranked:
            cohort_group = "ranked_eligibility_unknown"
        elif eligible is True:
            cohort_group = "eligible_not_ranked"
        elif eligible is False:
            cohort_group = "ineligible"
        else:
            cohort_group = "eligibility_unknown"

        disclosure_dates = pd.to_datetime(group.get("disclosure_date"), errors="coerce")
        event_dates = pd.to_datetime(group.get("event_date"), errors="coerce")
        latest_input = pd.concat([disclosure_dates, event_dates]).max()
        ticker_row = {
            "as_of": str(decision),
            "ticker": ticker,
            "ranked": is_ranked,
            "score": float(score_by_ticker[ticker]) if is_ranked else None,
            "score_quintile": None,
            "entry_ref_max": max_ref,
            "entry_ref_any_max": any_max_ref,
            "fill_date": outcome.get("fill_date"),
            "fill_px": outcome.get("fill_px"),
            "entry_eligible": eligible,
            "cohort_group": cohort_group,
            "outcome_status": outcome["outcome_status"],
            "exclusion_reason": outcome.get("exclusion_reason"),
            "fwd": outcome.get("fwd"),
            "spy": outcome.get("spy"),
            "excess": outcome.get("excess"),
            "price_generation_id": outcome.get("price_generation_id"),
            "spy_generation_id": outcome.get("spy_generation_id"),
            "price_basis": outcome.get("price_basis"),
            "spy_price_basis": outcome.get("spy_price_basis"),
            "frame": FRAME,
            "horizon_days": HORIZON_DAYS,
            "entry_slippage_bps": ENTRY_SLIPPAGE_BPS,
            "exit_slippage_bps": EXIT_SLIPPAGE_BPS,
            "decision_quote_date": str(quotes[ticker][1]) if ticker in quotes else None,
            "decision_quote_px": float(quotes[ticker][0]) if ticker in quotes else None,
            "decision_quote_status": quote_unusable_reason or "available",
            "n_actors": int(group["actor_id"].nunique()),
            "max_input_date": str(latest_input.date()) if pd.notna(latest_input) else None,
        }
        ticker_rows.append(ticker_row)

        for actor, actor_group in group.groupby("actor_id", sort=True):
            actor_valid = _current_rule_refs(actor_group, decision)
            actor_any_valid = _valid_refs(actor_group)
            actor_ref = (
                float(pd.to_numeric(actor_group.loc[actor_valid, "entry_ref"]).max())
                if actor_valid.any()
                else None
            )
            actor_any_ref = (
                float(pd.to_numeric(actor_group.loc[actor_any_valid, "entry_ref"]).max())
                if actor_any_valid.any()
                else None
            )
            dates = pd.to_datetime(actor_group["disclosure_date"], errors="coerce").dropna()
            actor_events = pd.to_datetime(actor_group["event_date"], errors="coerce").dropna()
            max_input = max([*dates.tolist(), *actor_events.tolist()], default=pd.NaT)
            block_reasons = set(
                actor_group.get("blocked_reason", pd.Series(dtype=object))
                .dropna().astype(str)
            )
            score_reason = blocked_reason.get((ticker, str(actor)))
            drop_reasons = []
            if quote_unusable_reason:
                drop_reasons.append(quote_unusable_reason)
            if actor_ref is None:
                drop_reasons.append(
                    score_reason or _missing_current_reference_reason(actor_group, decision)
                )
            actor_rows.append(
                {
                    **ticker_row,
                    "actor_id": str(actor),
                    "kind": str(actor_group["kind"].iloc[0]),
                    "actor_entry_ref": actor_ref,
                    "actor_entry_ref_any": actor_any_ref,
                    "entry_reference_status": "valid" if actor_ref is not None else "missing",
                    "n_current_rule_references": int(actor_valid.sum()),
                    "lag_days": (
                        int((pd.Timestamp(decision) - dates).dt.days.min())
                        if not dates.empty else None
                    ),
                    "max_input_disclosure_date": str(dates.max().date()) if not dates.empty else None,
                    "max_input_event_date": str(actor_events.max().date()) if not actor_events.empty else None,
                    "max_input_date": str(max_input.date()) if pd.notna(max_input) else None,
                    "candidate_rows": int(len(actor_group)),
                    "candidate_sources": "|".join(sorted(set(actor_group["source"].astype(str)))),
                    "candidate_source_record_ids": "|".join(
                        sorted(set(actor_group.get("source_record_id", pd.Series(dtype=object)).dropna().astype(str)))
                    ),
                    "candidate_source_row_ids": "|".join(
                        sorted(set(actor_group.get("source_row_id", pd.Series(dtype=object)).dropna().astype(str)))
                    ),
                    "input_block_reasons": "|".join(sorted(block_reasons)) or None,
                    "score_block_reason": score_reason,
                    "drop_unquotable_reason": "|".join(drop_reasons) or None,
                }
            )

    ticker_panel = pd.DataFrame(ticker_rows)
    if not ticker_panel.empty:
        ticker_panel = _score_quintiles(ticker_panel)
        quintiles = ticker_panel.set_index("ticker")["score_quintile"].to_dict()
        for row in actor_rows:
            row["score_quintile"] = quintiles.get(row["ticker"])
    actor_panel = pd.DataFrame(actor_rows)
    if not actor_panel.empty:
        assert_no_future_rows(actor_panel, decision)
    return actor_panel, ticker_panel


def _mean_ci(values: list[float], *, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[samples].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _cluster_stats(frame: pd.DataFrame, *, outcome="excess") -> dict:
    values = frame[["as_of", outcome]].copy()
    values[outcome] = pd.to_numeric(values[outcome], errors="coerce")
    values = values.dropna(subset=[outcome])
    cluster_means = values.groupby("as_of", sort=True)[outcome].mean()
    count = len(cluster_means)
    mean = float(cluster_means.mean()) if count else None
    pooled = float(values[outcome].mean()) if len(values) else None
    t_stat = None
    if count >= 2:
        std = float(cluster_means.std(ddof=1))
        se = std / math.sqrt(count)
        t_stat = (
            float(mean / se) if se > 0 else (0.0 if mean == 0 else math.copysign(math.inf, mean))
        )
    low, high = _mean_ci(cluster_means.to_list())
    return {
        "n_ticker_dates": int(len(values)),
        "clusters": int(count),
        "mean_pooled_pct": pooled,
        "mean_cluster_pct": mean,
        "date_clustered_t": t_stat,
        "ci95_low_pct": low,
        "ci95_high_pct": high,
    }


def _selected(frame: pd.DataFrame, mask: pd.Series, name: str) -> pd.DataFrame:
    result = frame.loc[mask.fillna(False)].copy()
    result["_group"] = name
    return result


def summarize(panel: pd.DataFrame, n_decisions: int) -> pd.DataFrame:
    """Summarize date-clustered group means, contrasts, and score quintiles."""
    if panel.empty:
        return pd.DataFrame()
    ranked = panel["ranked"].fillna(False).astype(bool)
    eligible = panel["entry_eligible"].eq(True)
    ineligible = panel["entry_eligible"].eq(False)
    groups = {
        "ranked": _selected(panel, ranked, "ranked"),
        "eligible_not_ranked": _selected(panel, ~ranked & eligible, "eligible_not_ranked"),
        # Actual-fill ineligible is intentionally independent of ranked status;
        # report ranked_ineligible overlap in the cohort rows.
        "ineligible": _selected(panel, ineligible, "ineligible"),
    }
    rows = []
    for name, frame in groups.items():
        rows.append(
            {
                "metric": "group_mean",
                "group_a": name,
                "group_b": "",
                "n_decision_dates": n_decisions,
                **_cluster_stats(frame),
            }
        )
    for quintile in [f"Q{i}" for i in range(1, 6)]:
        frame = panel.loc[panel["score_quintile"] == quintile]
        if not frame.empty:
            rows.append(
                {
                    "metric": "score_quintile_mean",
                    "group_a": quintile,
                    "group_b": "",
                    "n_decision_dates": n_decisions,
                    **_cluster_stats(frame),
                }
            )

    for baseline in ("eligible_not_ranked", "ineligible"):
        left = groups["ranked"]
        right = groups[baseline]
        left_by_date = left.dropna(subset=["excess"]).groupby("as_of")["excess"].mean()
        right_by_date = right.dropna(subset=["excess"]).groupby("as_of")["excess"].mean()
        common = left_by_date.index.intersection(right_by_date.index)
        difference = (left_by_date.loc[common] - right_by_date.loc[common]).astype(float)
        low, high = _mean_ci(difference.to_list(), seed=BOOTSTRAP_SEED + len(rows))
        count = len(difference)
        effect = float(difference.mean()) if count else None
        se = float(difference.std(ddof=1) / math.sqrt(count)) if count >= 2 else None
        t_stat = (
            float(effect / se) if se and se > 0
            else (0.0 if effect == 0 else math.copysign(math.inf, effect))
            if effect is not None and count >= 2
            else None
        )
        rows.append(
            {
                "metric": "ranked_minus_baseline",
                "group_a": "ranked",
                "group_b": baseline,
                "n_decision_dates": n_decisions,
                "n_ticker_dates": int(len(left.dropna(subset=["excess"]))),
                "clusters": count,
                "mean_pooled_pct": (
                    float(left.dropna(subset=["excess"])["excess"].mean())
                    - float(right.dropna(subset=["excess"])["excess"].mean())
                    if not left.dropna(subset=["excess"]).empty
                    and not right.dropna(subset=["excess"]).empty else None
                ),
                "mean_cluster_pct": effect,
                "date_clustered_t": t_stat,
                "ci95_low_pct": low,
                "ci95_high_pct": high,
            }
        )

    top = panel.loc[panel["score_quintile"] == "Q5"]
    top_stats = _cluster_stats(top)
    rows.append(
        {
            "metric": "primary_top_quintile_vs_zero",
            "group_a": "Q5",
            "group_b": "zero",
            "n_decision_dates": n_decisions,
            **top_stats,
        }
    )
    return pd.DataFrame(rows)


def evaluation_verdict(primary_row: dict) -> str:
    clusters = int(primary_row.get("clusters") or 0)
    if clusters < 2:
        return (
            "NOT VALIDATABLE with available certified history: "
            f"{clusters} usable cluster(s)"
        )
    mean = primary_row.get("mean_cluster_pct")
    low = primary_row.get("ci95_low_pct")
    if (
        mean is not None and low is not None
        and pd.notna(mean) and pd.notna(low)
        and float(mean) > 0 and float(low) > 0
    ):
        return "PASS"
    return "NOT VALIDATED"


def build_drop_report(cohorts: pd.DataFrame, excluded_dates: list[dict] | None = None) -> pd.DataFrame:
    """Count retained-but-excluded ticker/actor inputs by explicit reason."""
    records = []
    for row in cohorts.itertuples(index=False):
        base = {
            "as_of": row.as_of,
            "cohort_group": row.cohort_group,
            "ticker": row.ticker,
            "actor_id": row.actor_id,
        }
        if isinstance(row.drop_unquotable_reason, str) and row.drop_unquotable_reason:
            records.extend(
                {**base, "drop_class": "unquotable", "reason": reason}
                for reason in row.drop_unquotable_reason.split("|")
                if reason
            )
        if row.outcome_status == "uncertifiable":
            records.append({**base, "drop_class": "uncertifiable", "reason": str(row.exclusion_reason)})
        elif row.outcome_status == "outcome_missing":
            records.append({**base, "drop_class": "outcome_missing", "reason": str(row.exclusion_reason)})
    grouped = defaultdict(lambda: {"n_ticker_actor": 0, "tickers": set()})
    for item in records:
        key = (item["as_of"], item["cohort_group"], item["drop_class"], item["reason"])
        grouped[key]["n_ticker_actor"] += 1
        grouped[key]["tickers"].add(item["ticker"])
    rows = [
        {
            "as_of": key[0],
            "cohort_group": key[1],
            "frame": FRAME,
            "drop_class": key[2],
            "reason": key[3],
            "n_ticker_actor": values["n_ticker_actor"],
            "n_tickers": len(values["tickers"]),
        }
        for key, values in sorted(grouped.items())
    ]
    for item in excluded_dates or []:
        rows.append(
            {
                "as_of": item.get("as_of"),
                "cohort_group": "all",
                "frame": FRAME,
                "drop_class": "date_excluded",
                "reason": item["reason"],
                "n_ticker_actor": 0,
                "n_tickers": 0,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["as_of", "cohort_group", "frame", "drop_class", "reason", "n_ticker_actor", "n_tickers"],
    )


def evaluate_synthetic_cohort(
    candidates: pd.DataFrame,
    weights: dict[str, float],
    quotes: dict,
    prices: pd.DataFrame,
    decision: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small end-to-end seam shared by tests: production score + outcome math."""
    assert_no_future_rows(candidates, decision)
    ranked, blocked = score_setups(
        candidates,
        weights,
        quotes,
        entry_tol=ENTRY_TOLERANCE,
        event_half_life_days=EVENT_HALF_LIFE_DAYS,
        watchlist=watchlist_actor_ids(),
        enable_13f_corroboration=False,
    )
    tickers = sorted(candidates["ticker"].astype(str).str.upper().unique())
    evaluated = evaluate_backtest(
        pd.DataFrame({"ticker": tickers, "instrument_type": "stock"}),
        prices,
        pd.Timestamp(decision),
        HORIZON_DAYS,
        entry_slippage_bps=ENTRY_SLIPPAGE_BPS,
        exit_slippage_bps=EXIT_SLIPPAGE_BPS,
    )
    labels = {}
    for row in evaluated.itertuples(index=False):
        complete = row.bt_coverage == "complete"
        labels[str(row.ticker)] = {
            "fill_date": str(row.bt_entry_date) if pd.notna(row.bt_entry_date) else None,
            "fill_px": float(row.bt_entry_price) if pd.notna(row.bt_entry_price) else None,
            "excess": float(row.bt_alpha_pct) if complete else None,
            "fwd": float(row.bt_return_pct) if complete else None,
            "spy": float(row.bt_spy_return_pct) if complete else None,
            "exit_date": str(row.bt_exit_date) if complete else None,
            "outcome_status": "available" if complete else "outcome_missing",
            "exclusion_reason": None if complete else row.bt_unavailable_reason,
            "price_generation_id": "synthetic-certified",
            "spy_generation_id": "synthetic-certified",
            "price_basis": "synthetic-certified",
            "spy_price_basis": "synthetic-certified",
        }
    return _cohort_rows(candidates, ranked, blocked, labels, quotes, decision)


def _date_list(raw: list[str] | None, db) -> tuple[list[date], list[dict]]:
    if raw is None:
        return default_decision_dates(db)
    parsed = [date.fromisoformat(value) for value in raw]
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("--dates must contain one or more unique ISO dates")
    return sorted(parsed), []


def run(db_path: Path, dates: list[date], output_dir: Path, excluded_dates=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db = Database(db_path, read_only=True)
    if not db.is_read_only:
        raise RuntimeError("evaluation DB did not open read-only")
    cohort_frames = []
    ticker_frames = []
    no_candidate_dates = []
    try:
        for decision in dates:
            print(f"evaluate {decision} ({FRAME} frame)", flush=True)
            with db.canonical_as_of(decision), old_harness.cached_position_inputs():
                candidates = _production_candidates(db, decision)
                if candidates.empty:
                    no_candidate_dates.append({"as_of": str(decision), "reason": "no_ingestion_candidates"})
                    print("  no candidates", flush=True)
                    continue
                quotes = _decision_quotes(db, sorted(candidates["ticker"].astype(str).unique()), decision)
                weights = weights_for_candidates(db, candidates, decision)
                ranked, blocked = score_setups(
                    candidates,
                    weights,
                    quotes,
                    entry_tol=ENTRY_TOLERANCE,
                    event_half_life_days=EVENT_HALF_LIFE_DAYS,
                    watchlist=watchlist_actor_ids(),
                    enable_13f_corroboration=False,
                )
                symbols = sorted(candidates["ticker"].astype(str).str.upper().unique())
                labels = _label_outcomes(db, symbols, decision)
                actors, tickers = _cohort_rows(
                    candidates, ranked, blocked, labels, quotes, decision
                )
                cohort_frames.append(actors)
                ticker_frames.append(tickers)
                print(
                    f"  candidates={len(candidates)} ticker_actor_rows={len(actors)} "
                    f"ranked_tickers={int(tickers['ranked'].sum())}",
                    flush=True,
                )
    finally:
        db.close()

    cohorts = pd.concat(cohort_frames, ignore_index=True) if cohort_frames else pd.DataFrame()
    ticker_panel = pd.concat(ticker_frames, ignore_index=True) if ticker_frames else pd.DataFrame()
    if not cohorts.empty:
        # A score quintile is assigned independently within each decision.
        assert cohorts.groupby(["as_of", "ticker", "actor_id"]).size().max() == 1
    summary = summarize(ticker_panel, len(dates)) if not ticker_panel.empty else pd.DataFrame()
    if not summary.empty:
        summary["frame"] = FRAME
        summary["horizon_days"] = HORIZON_DAYS
        summary["price_basis"] = _PRICE_BASIS
        summary["entry_slippage_bps"] = ENTRY_SLIPPAGE_BPS
        summary["exit_slippage_bps"] = EXIT_SLIPPAGE_BPS
    excluded = [*(excluded_dates or []), *no_candidate_dates]
    drops = build_drop_report(cohorts, excluded)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(output_dir / "cohorts.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    drops.to_csv(output_dir / "drop-report.csv", index=False)
    return cohorts, summary, drops


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--dates", nargs="*", help="explicit ISO decision dates; default uses certified SPY coverage")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    db = Database(args.db, read_only=True)
    try:
        dates, excluded = _date_list(args.dates, db)
        spy_state = _certified_states(db, ["SPY"])["SPY"]
    finally:
        db.close()
    cohorts, summary, drops = run(args.db, dates, args.output_dir, excluded)
    primary = summary.loc[summary["metric"] == "primary_top_quintile_vs_zero"] if not summary.empty else pd.DataFrame()
    primary_row = primary.iloc[0].to_dict() if not primary.empty else {}
    verdict = evaluation_verdict(primary_row)
    if not summary.empty:
        summary["evaluation_verdict"] = verdict
        summary.to_csv(args.output_dir / "summary.csv", index=False)
    observed_dates = sorted(set(cohorts.get("as_of", pd.Series(dtype=str)).dropna().astype(str)))
    outcome_dates = sorted(
        set(cohorts.loc[cohorts.get("excess", pd.Series(dtype=float)).notna(), "as_of"].astype(str))
    ) if not cohorts.empty else []
    print(json.dumps({
        "decision_dates": [str(d) for d in dates],
        "requested_date_count": len(dates),
        "candidate_date_clusters": len(observed_dates),
        "candidate_dates": observed_dates,
        "outcome_date_clusters": len(outcome_dates),
        "outcome_dates": outcome_dates,
        "cohort_ticker_actor_rows": len(cohorts),
        "summary_rows": len(summary),
        "primary": primary_row,
        "verdict": verdict,
        "power_statement": POWER_STATEMENT,
        "certified_spy_generation": spy_state,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
