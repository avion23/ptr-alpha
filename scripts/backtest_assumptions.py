"""Chronological backtest of the system's load-bearing assumptions.

Phase 1 (skill): label members using ONLY pre-2023 disclosures -> forward
90d excess vs SPY. Skilled = mean excess > +3pp with >=5 disclosures.
Hyperactive = breadth (distinct tickers, all history) > 150. No returns from
2023+ are used for labeling: out-of-sample by construction.

Phase 2 (cohorts): for each decision date, score every candidate-rich ticker
with the production pipeline (eligible_events + holdings + score), enter at
the NEXT NYSE session close after the decision date (executable, not the
decision close), and measure 90d forward excess vs SPY from the same entry.

Usage:
  python3 scripts/backtest_assumptions.py skill
  python3 scripts/backtest_assumptions.py run 2023-01-15
  python3 scripts/backtest_assumptions.py analyze
Results land in data/backtest_assumptions/.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd

from analyzer.actors import compute_weight
from analyzer.database import Database
from analyzer.manager_watchlist import watchlist_actor_ids
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
HORIZON = 90

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
    if not row or not row[1]:
        return None
    return (row[0], float(row[1]))


def fwd_excess(db, ticker: str, entry_date: date, entry_px: float):
    end = previous_nyse_session(entry_date + timedelta(days=HORIZON)).date()
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
    if not out or not out[0] or not spy_in or not spy_out[0]:
        return None, None, None
    ret = (float(out[0]) / entry_px - 1) * 100
    spy = (float(spy_out[0]) / float(spy_in[0]) - 1) * 100
    return ret, spy, ret - spy


def cmd_skill():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(DATA_DIR / "congress.duckdb", read_only=True)
    try:
        ev = eligible_events(db, date(2022, 12, 31))
        ev = ev[pd.to_datetime(ev["disclosure_date"]) < pd.Timestamp("2023-01-01")]
        print("pre-2023 events:", len(ev), flush=True)
        stats = []
        for actor, grp in ev.groupby("actor_id"):
            if not str(actor).startswith("congress:"):
                continue
            rets = []
            for r in grp.itertuples():
                disc = pd.to_datetime(r.disclosure_date).date()
                q = next_session_close(db, str(r.ticker), disc)
                if not q:
                    continue
                ret, _, exc = fwd_excess(db, str(r.ticker), q[0], q[1])
                if exc is not None:
                    rets.append(exc)
            if len(rets) >= 5:
                stats.append(
                    {
                        "actor": actor,
                        "n": len(rets),
                        "mean_excess": sum(rets) / len(rets),
                    }
                )
        labels = {}
        for s in stats:
            if s["mean_excess"] > 3.0:
                labels[s["actor"]] = "skilled"
        # Breadth from full history (mechanical, no returns used).
        tx = db.conn.execute(
            "SELECT member, COUNT(DISTINCT ticker) FROM canonical_transactions"
            " GROUP BY 1"
        ).fetchall()
        breadth = {m: n for m, n in tx}
        json.dump(
            {"labels": labels, "stats": stats, "breadth": breadth},
            open(OUT_DIR / "skill.json", "w"),
        )
        skilled = sorted(labels)
        print("skilled members:", len(skilled), flush=True)
        for a in skilled[:30]:
            print("  ", a)
    finally:
        db.close()


def cmd_run(decision: date):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = Database(DATA_DIR / "congress.duckdb", read_only=True)
    try:
        skill = json.load(open(OUT_DIR / "skill.json"))
        skilled = set(skill["labels"])
        breadth = skill["breadth"]
        hyper = {
            "congress:" + m
            for m, n in breadth.items()
            if n and n > 150
        }
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
            init = cands.loc[cands["entry_ref"].notna()]
            actor_ids = set(init["actor_id"].astype(str))
            kinds_hit = set(init["kind"].astype(str))
            has_13f = bool((cands["source"] == "13f").any())
            lags = (
                pd.to_datetime(decision)
                - pd.to_datetime(init["disclosure_date"])
            ).dt.days.dropna()
            if lags.empty:
                continue
            fresh = bool((lags <= 90).any())
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
                    "fresh": fresh,
                    "min_lag": int(lags.min()),
                    "entry_px": entry_px,
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
        db.close()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "skill":
        cmd_skill()
    elif mode == "run":
        cmd_run(date.fromisoformat(sys.argv[2]))
    else:
        raise SystemExit("usage: skill | run YYYY-MM-DD")
