"""Chronological backtest of the system's load-bearing assumptions.

Phase 1 (skill): label members using only pre-2023 disclosures whose 90-day
outcomes matured before each decision date. Skilled = mean excess > +3pp with
>=5 unique episodes. Hyperactive = point-in-time breadth > 150.

Phase 2 (cohorts): for each decision date, score every candidate-rich ticker
with the production pipeline (eligible_events + holdings + score), enter at
the NEXT NYSE session close after the decision date (executable, not the
decision close), and measure 90d forward excess vs SPY from the same entry.

Usage:
  python3 scripts/backtest_assumptions.py skill
  python3 scripts/backtest_assumptions.py run 2023-01-15
  python3 scripts/backtest_assumptions.py analyze
  python3 scripts/backtest_assumptions.py full
Results land in data/backtest_assumptions/.
"""

import json
import math
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy import stats

from analyzer import positions
from analyzer.actors import actor_id, compute_weight
from analyzer.database import Database
from analyzer.manager_watchlist import watchlist_actor_ids
from analyzer.member_names import canonical_member_key
from analyzer.pipeline import eligible_events
from analyzer.positions import (
    coalesce_actor_variants,
    dedupe_member_names,
    gate_closed_events,
    holdings_candidates,
)
from analyzer.price_repository import previous_nyse_session
from analyzer.setups import score as score_setups

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "backtest_assumptions"
DB_PATH = Path("/home/ubuntu/insider-trading/data/congress.duckdb")
HORIZON = 90
MAX_ENTRY_GAP_DAYS = 4

DECISIONS = [
    date(2023, 1, 15),
    date(2023, 7, 15),
    date(2024, 1, 15),
    date(2024, 7, 15),
    date(2025, 1, 15),
    date(2025, 7, 15),
    date(2026, 1, 15),
    date(2026, 7, 1),
]


def next_session_close(db, ticker: str, after: date):
    row = db.conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? AND date > ?"
        " AND close IS NOT NULL ORDER BY date LIMIT 1",
        [ticker, after],
    ).fetchone()
    if not row or (row[0] - after).days > MAX_ENTRY_GAP_DAYS:
        return None
    try:
        price = float(row[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return (row[0], price)


def outcome_exit_date(entry_date: date) -> date:
    return previous_nyse_session(entry_date + timedelta(days=HORIZON)).date()


def fwd_excess(db, ticker: str, entry_date: date, entry_px: float):
    end = outcome_exit_date(entry_date)
    if end <= entry_date:
        return None, None, None
    out = db.conn.execute(
        "SELECT close FROM prices WHERE ticker = ? AND date = ?",
        [ticker, end],
    ).fetchone()
    spy_in = db.conn.execute(
        "SELECT close FROM prices WHERE ticker = 'SPY' AND date <= ?"
        " ORDER BY date DESC LIMIT 1",
        [entry_date],
    ).fetchone()
    spy_out = db.conn.execute(
        "SELECT close FROM prices WHERE ticker = 'SPY' AND date = ?", [end]
    ).fetchone()
    if not out or not spy_in or not spy_out:
        return None, None, None
    try:
        out_px, spy_in_px, spy_out_px = map(
            float, (out[0], spy_in[0], spy_out[0])
        )
    except (TypeError, ValueError):
        return None, None, None
    if any(
        not math.isfinite(price) or price <= 0
        for price in (entry_px, out_px, spy_in_px, spy_out_px)
    ):
        return None, None, None
    ret = (out_px / entry_px - 1) * 100
    spy = (spy_out_px / spy_in_px - 1) * 100
    return ret, spy, ret - spy


def breadth_by_actor(db, decision: date) -> dict[str, int]:
    rows = db.conn.execute(
        "SELECT DISTINCT source, member, ticker FROM canonical_transactions"
        " WHERE disclosure_date <= ? AND member IS NOT NULL AND ticker IS NOT NULL",
        [decision],
    ).fetchall()
    kinds = {
        "house_pdf": "congress",
        "gemini_ocr": "congress",
        "senate_efd": "congress",
        "form4": "officer",
        "13f": "manager",
    }
    symbols: dict[str, set[str]] = {}
    for source, member, ticker in rows:
        kind = kinds.get(source)
        if not kind or not isinstance(member, str) or not isinstance(ticker, str):
            continue
        normalized = canonical_member_key(member) if kind == "congress" else member
        actor = actor_id(kind, normalized)
        symbols.setdefault(actor, set()).add(ticker.upper())
    return {actor: len(tickers) for actor, tickers in symbols.items()}


@contextmanager
def cached_position_inputs():
    load_trades = positions.load_member_trades
    load_prices = positions.load_price_history
    trades_cache = {}
    prices_cache = {}

    def cached_trades(db, member: str, *, include_unpromoted: bool = False):
        key = (id(db), member, include_unpromoted)
        if key not in trades_cache:
            trades_cache[key] = load_trades(
                db, member, include_unpromoted=include_unpromoted
            )
        return trades_cache[key]

    def cached_prices(db, ticker: str):
        key = (id(db), ticker)
        if key not in prices_cache:
            prices_cache[key] = load_prices(db, ticker)
        return prices_cache[key]

    positions.load_member_trades = cached_trades
    positions.load_price_history = cached_prices
    try:
        yield
    finally:
        positions.load_member_trades = load_trades
        positions.load_price_history = load_prices


def cmd_skill():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(DB_PATH, read_only=True)
    try:
        ev = eligible_events(db, date(2022, 12, 31))
        ev = ev[pd.to_datetime(ev["disclosure_date"]) < pd.Timestamp("2023-01-01")]
        ev = ev.drop_duplicates(["actor_id", "ticker", "disclosure_date"])
        ev = ev.loc[ev["actor_id"].astype(str).str.startswith("congress:")]
        episodes = []
        for r in ev.itertuples():
            disclosure = pd.to_datetime(r.disclosure_date).date()
            entry = next_session_close(db, str(r.ticker), disclosure)
            if not entry:
                continue
            exit_date = outcome_exit_date(entry[0])
            _, _, excess = fwd_excess(db, str(r.ticker), entry[0], entry[1])
            if excess is not None:
                episodes.append(
                    {
                        "actor": str(r.actor_id),
                        "ticker": str(r.ticker),
                        "disclosure_date": disclosure,
                        "exit_date": exit_date,
                        "excess": excess,
                    }
                )
        print("pre-2023 unique events:", len(ev), "priced outcomes:", len(episodes), flush=True)
        by_decision = {}
        for decision in DECISIONS:
            matured = [item for item in episodes if item["exit_date"] < decision]
            stats_by_actor = {}
            for item in matured:
                stats_by_actor.setdefault(item["actor"], []).append(item["excess"])
            stats_rows = [
                {
                    "actor": actor,
                    "n": len(rets),
                    "mean_excess": sum(rets) / len(rets),
                }
                for actor, rets in sorted(stats_by_actor.items())
                if len(rets) >= 5
            ]
            labels = {
                item["actor"]
                for item in stats_rows
                if item["mean_excess"] > 3.0
            }
            breadth = breadth_by_actor(db, decision)
            by_decision[str(decision)] = {
                "labels": sorted(labels),
                "stats": stats_rows,
                "breadth": breadth,
            }
        with (OUT_DIR / "skill.json").open("w") as output:
            json.dump({"by_decision": by_decision}, output)
        for decision, values in by_decision.items():
            print(decision, "skilled members:", len(values["labels"]), flush=True)
    finally:
        db.close()


def cmd_run(decision: date, db=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    owns_db = db is None
    if owns_db:
        db = Database(DB_PATH, read_only=True)
    try:
        skill = json.loads((OUT_DIR / "skill.json").read_text())
        point_in_time = skill["by_decision"][str(decision)]
        skilled = set(point_in_time["labels"])
        breadth = point_in_time["breadth"]
        hyper = {actor for actor, count in breadth.items() if count > 150}
        events_all = eligible_events(db, decision)
        tickers = sorted(
            events_all["ticker"].astype("string").str.upper().unique()
        )
        # Candidate-rich universe: >=2 distinct buyer events ever disclosed.
        keep = []
        for t in tickers:
            sub = events_all.loc[
                events_all["ticker"].astype("string").str.upper() == t
            ]
            if sub["actor_id"].nunique() >= 2:
                keep.append(t)
        print(decision, "tickers:", len(keep), flush=True)
        holdings_cache: dict = {}

        def holdings_for(member: str):
            if member not in holdings_cache:
                holdings_cache[member] = holdings_candidates(
                    db, member, decision
                )
            return holdings_cache[member]

        watchlist = watchlist_actor_ids()
        rows = []
        for i, symbol in enumerate(keep):
            events = events_all.loc[
                events_all["ticker"].astype("string").str.upper() == symbol
            ].copy()
            kinds: dict[str, str] = {}
            for actor_id in events["actor_id"].dropna():
                text = str(actor_id)
                if ":" not in text:
                    continue
                kind, _, name = text.partition(":")
                if name:
                    kinds.setdefault(name, kind)
            frames, congress_frames = [], []
            for m in dedupe_member_names(sorted(kinds)):
                frame = holdings_for(m)
                if frame.empty:
                    continue
                frame = frame.loc[
                    frame["ticker"].astype("string").str.upper() == symbol
                ].copy()
                if not frame.empty:
                    frames.append(frame)
                    congress_frames.append(
                        frame.loc[frame["kind"] == "congress"].copy()
                    )
            pos = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame()
            )
            events = coalesce_actor_variants(events, pos)
            events = gate_closed_events(events, pos)
            cands = pd.concat(
                [f for f in congress_frames + [events] if not f.empty],
                ignore_index=True,
            )
            if cands.empty:
                continue
            q = next_session_close(db, symbol, decision)
            if not q:
                continue
            entry_date, entry_px = q
            weights = {
                str(r.actor_id): compute_weight(
                    str(r.kind), 0, 0.0, str(r.source)
                )
                for r in cands.itertuples()
            }
            ranked, blocked = score_setups(
                cands, weights, {symbol: (entry_px, entry_date)},
                watchlist=watchlist,
            )
            ret, spy, exc = fwd_excess(db, symbol, entry_date, entry_px)
            if exc is None:
                continue
            refs = pd.to_numeric(cands["entry_ref"], errors="coerce")
            valid_refs = refs.notna() & np.isfinite(refs) & (refs > 0)
            init = cands.loc[valid_refs].copy()
            actor_ids = set(init["actor_id"].astype(str))
            kinds_hit = set(init["kind"].astype(str))
            has_13f = bool((cands["source"] == "13f").any())
            lags = (
                pd.to_datetime(decision)
                - pd.to_datetime(init["disclosure_date"])
            ).dt.days.dropna()
            fresh_frame = init.loc[lags.between(0, 30)] if not lags.empty else init.iloc[0:0]
            fresh_actors = set(fresh_frame["actor_id"].astype(str))
            entry_eligible = bool((entry_px <= 1.05 * refs.loc[valid_refs]).any())
            rows.append(
                {
                    "ticker": symbol,
                    "as_of": str(decision),
                    "ranked": bool(not ranked.empty),
                    "score": float(ranked["score"].max())
                    if not ranked.empty
                    else 0.0,
                    "n_actors": len(actor_ids),
                    "has_officer": "officer" in kinds_hit,
                    "has_13f": has_13f,
                    "fresh": bool(fresh_actors),
                    "fresh_skilled": bool(fresh_actors & skilled),
                    "min_lag": int(lags.min()) if not lags.empty else None,
                    "entry_px": entry_px,
                    "entry_eligibility": "eligible" if entry_eligible else "ineligible",
                    "entry_eligible": entry_eligible,
                    "fwd": ret,
                    "spy": spy,
                    "excess": exc,
                    "skilled_present": bool(actor_ids & skilled),
                    "hyper_present": bool(actor_ids & hyper),
                    "hyper_only": bool(
                        actor_ids and actor_ids <= hyper
                    ),
                }
            )
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(keep)}", flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / f"cohorts_{decision}.csv", index=False)
        print("wrote", len(rows), "rows", flush=True)
    finally:
        if owns_db:
            db.close()


def clustered_t(frame: pd.DataFrame, group: str) -> float | None:
    values = frame[["excess", "as_of", group]].dropna()
    if values[group].nunique() != 2 or values["as_of"].nunique() < 2:
        return None
    y = values["excess"].to_numpy(dtype=float)
    treatment = values[group].astype(bool).to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(values)), treatment))
    n, k = design.shape
    if n <= k:
        return None
    bread = np.linalg.inv(design.T @ design)
    residuals = y - design @ (bread @ design.T @ y)
    meat = np.zeros((k, k))
    for _, indices in values.groupby("as_of").groups.items():
        positions = values.index.get_indexer(indices)
        cluster_score = design[positions].T @ residuals[positions]
        meat += np.outer(cluster_score, cluster_score)
    clusters = values["as_of"].nunique()
    covariance = (clusters / (clusters - 1)) * ((n - 1) / (n - k)) * bread @ meat @ bread
    variance = covariance[1, 1]
    if not math.isfinite(variance) or variance <= 0:
        return None
    standard_error = math.sqrt(variance)
    return float((bread @ design.T @ y)[1] / standard_error)


def compare_groups(
    frame: pd.DataFrame, name: str, group: str, label: str, other: str
) -> dict:
    values = frame[["excess", "as_of", group]].dropna()
    positive = values.loc[values[group].astype(bool), "excess"].astype(float)
    negative = values.loc[~values[group].astype(bool), "excess"].astype(float)
    welch = (
        stats.ttest_ind(positive, negative, equal_var=False).statistic
        if len(positive) and len(negative)
        else float("nan")
    )
    return {
        "comparison": name,
        "group_a": label,
        "n_a": len(positive),
        "mean_a": float(positive.mean()) if len(positive) else float("nan"),
        "group_b": other,
        "n_b": len(negative),
        "mean_b": float(negative.mean()) if len(negative) else float("nan"),
        "welch_t": float(welch),
        "date_clustered_t": clustered_t(values, group),
    }


def cmd_analyze():
    files = [OUT_DIR / f"cohorts_{decision}.csv" for decision in DECISIONS]
    frames = [pd.read_csv(path) for path in files if path.exists()]
    if len(frames) != len(DECISIONS):
        raise SystemExit("missing cohort CSVs; run all decision dates first")
    observations = pd.concat(frames, ignore_index=True)
    eligible = observations.loc[observations["entry_eligible"]]
    comparisons = [
        compare_groups(
            observations, "skilled vs not", "skilled_present", "skilled", "not skilled"
        ),
        compare_groups(
            observations,
            "fresh+skilled same actor vs rest",
            "fresh_skilled",
            "fresh+skilled",
            "rest",
        ),
        compare_groups(
            eligible,
            "fresh+skilled same actor among eligible",
            "fresh_skilled",
            "fresh+skilled",
            "other eligible",
        ),
        compare_groups(
            observations,
            "eligible vs ineligible",
            "entry_eligible",
            "eligible",
            "ineligible",
        ),
    ]
    summary = pd.DataFrame(comparisons)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def cmd_full():
    cmd_skill()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(DB_PATH, read_only=True)
    try:
        with cached_position_inputs():
            for decision in DECISIONS:
                cmd_run(decision, db)
    finally:
        db.close()
    cmd_analyze()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "skill":
        cmd_skill()
    elif mode == "run":
        with cached_position_inputs():
            cmd_run(date.fromisoformat(sys.argv[2]))
    elif mode == "analyze":
        cmd_analyze()
    elif mode == "full":
        cmd_full()
    else:
        raise SystemExit("usage: skill | run YYYY-MM-DD | analyze | full")
