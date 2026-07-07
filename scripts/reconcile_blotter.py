#!/usr/bin/env python3
"""Reconcile a Thinkorswim brokerage blotter against the congressional PTR
database and current market prices.

Usage:
    PYTHONPATH=src python3 scripts/reconcile_blotter.py <csv_path>

Outputs a text table to stdout, a plain-language summary, and writes a
structured JSON result to scripts/blotter_reconcile_output.json.

IMPORTANT (honesty note):
    This script does NOT invoke the program's real recommendation engine
    (src/analyzer/signals / the `analyze` backtest). The flag produced here,
    `congressional_buy_within_60d`, is a FACTUAL PROXY: it is "Y" when at least
    one congressional Purchase appears in the raw transactions table within 60
    days before the user's earliest Buy-To-Open for that ticker. It is NOT a
    verdict from the program's conviction/quality-filter engine. Treat it as a
    coincidence proxy only.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Resolve the congressional duckdb. The main repo lives one level above the
# worktree; the file may also be present locally under data/.
def _find_db() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "congress.duckdb",
        PROJECT_ROOT.parent / "data" / "congress.duckdb",
        Path.cwd() / "data" / "congress.duckdb",
    ]
    for c in candidates:
        if c.exists():
            return c
    # last resort: search upward from cwd
    p = Path.cwd()
    for _ in range(6):
        maybe = p / "data" / "congress.duckdb"
        if maybe.exists():
            return maybe
        p = p.parent
    raise SystemExit("Could not locate data/congress.duckdb")


DB_PATH = _find_db()

# transaction_type values treated as congressional BUYS (purchase-type).
BUY_TYPES = {"purchase"}

# Windows (days) before the user's buy date in which a congressional buy counts.
# 60 is the headline "recommendation-consistency" proxy window; the others are
# reported alongside for sensitivity analysis (issue H1).
MATCH_WINDOW_DAYS = 60
REPORT_WINDOWS = [30, 45, 60, 90]


def _this_year() -> int:
    return datetime.now().year


def parse_price(raw: str) -> float:
    """Strip a trailing non-numeric suffix like ' db' / ' cr' and parse float."""
    s = raw.strip()
    m = re.match(r"^-?\d+(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"cannot parse price from {raw!r}")
    return float(m.group(0))


def parse_description(raw: str):
    """Return (share_qty:int, side:str) from e.g. '100 BTO' / '-10 STC'.

    The share quantity is returned as the RAW signed magnitude from the
    description text. The caller is responsible for normalizing the sign by
    side (see normalize_qty). We never trust the description sign for position
    direction.
    """
    s = raw.strip()
    m = re.match(r"^(-?\d+)\s+(BTO|STC|STO|BTC)\b", s, re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot parse description {raw!r}")
    qty = int(m.group(1))
    side = m.group(2).upper()
    return qty, side


def normalize_qty(side: str, qty: int) -> int:
    """Normalize the signed share quantity explicitly by side (issue M3).

    BTO opens a long  -> positive
    STO opens a short -> negative
    STC closes a long -> magnitude (sign ignored); treated as a long close
    BTC closes a short-> magnitude (sign ignored); treated as a short close
    """
    mag = abs(qty)
    if side == "BTO":
        return mag
    if side == "STO":
        return -mag
    # STC / BTC: closing legs; the sign from the description is ignored.
    return mag


def parse_fill_date(raw: str) -> date:
    """Parse '6/22, 4:30p' -> date(THIS_YEAR, 6, 22)."""
    s = raw.strip()
    m = re.match(r"(\d{1,2})/(\d{1,2})", s)
    if not m:
        raise ValueError(f"cannot parse fill date from {raw!r}")
    month, day = int(m.group(1)), int(m.group(2))
    return date(_this_year(), month, day)


def get_current_price(ticker: str):
    """Fetch most recent close via yfinance. Returns (price, price_date) or (None, None)."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return None, None
        last = hist.iloc[-1]
        price = float(last["Close"])
        price_date = pd.Timestamp(last.name).date()
        return price, price_date
    except Exception as e:  # noqa: BLE001 - network/parse can fail broadly
        return None, None


def get_congressional(ticker: str):
    """Return congressional transactions for ticker, de-duplicated (issue M4).

    The raw `transactions` table contains known phantom duplicate-row groups
    (see scripts/purge_phantom_rows.py). We de-dup on
    (member, ticker, transaction_date, transaction_type, disclosure_date) so
    that n_buys and match tests are not inflated.
    """
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT DISTINCT ON (
                    member, ticker, transaction_date, transaction_type, disclosure_date
                )
                    member, ticker, transaction_date, disclosure_date,
                    transaction_type, owner_code, amount_raw
                FROM transactions
                WHERE LOWER(ticker) = LOWER(?)
                ORDER BY member, ticker, transaction_date,
                         transaction_type, disclosure_date
            ) sub
            ORDER BY COALESCE(transaction_date, disclosure_date) DESC
            """,
            [ticker],
        ).fetchall()
    finally:
        conn.close()
    return rows


def _eff_date(row) -> date | None:
    """Effective date = transaction_date if present else disclosure_date."""
    tx, disc = row[2], row[3]
    if tx is not None:
        return tx if isinstance(tx, date) else None
    if disc is not None:
        return disc if isinstance(disc, date) else None
    return None


def _expected_match_prob(n_buys: int, span_days: float, window_days: int) -> float:
    """A-priori probability of >=1 congressional buy in a random W-day window.

    Modeled as a Poisson process: buys occur at rate n_buys / span_days over the
    history of the ticker's data. P(at least one buy in W days) =
    1 - exp(-rate * W). This is the "by chance" base rate an actual hit should
    be judged against (issue H2).
    """
    if n_buys <= 0 or span_days <= 0 or window_days <= 0:
        return 0.0
    rate = n_buys / span_days  # buys per day
    return 1.0 - math.exp(-rate * window_days)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: reconcile_blotter.py <csv_path>", file=sys.stderr)
        return 2

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"error: {csv_path} not found", file=sys.stderr)
        return 1

    print(f"DB:        {DB_PATH}")
    print(f"Blotter:   {csv_path}")
    print(f"Proxy win: congressional Purchase within {MATCH_WINDOW_DAYS} days "
          f"before user BTO (INCLUSIVE of both endpoints)")
    print("=" * 80)
    print("HONESTY NOTE: 'congressional_buy_within_60d' is a FACTUAL PROXY "
          "(raw congressional")
    print("buy timing), NOT the program's recommendation engine output. The real "
          "engine")
    print("(src/analyzer/signals backtest / conviction+quality filter) was NOT "
          "invoked.")
    print("=" * 80)

    # --- parse blotter -------------------------------------------------------
    fills = []  # dict per row
    parse_failures = []  # rows we could not parse (issue M2)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, start=1):
            if not row.get("Symbol"):
                continue
            try:
                sym = row["Symbol"].strip()
                if not sym:
                    raise ValueError("empty Symbol")
                price = parse_price(row["Price"])
                raw_qty, side = parse_description(row["Description"])
                qty = normalize_qty(side, raw_qty)
                fdate = parse_fill_date(row["Time"])
            except Exception as exc:  # noqa: BLE001 - keep going, record failure
                parse_failures.append(
                    {
                        "line": i,
                        "symbol": (row.get("Symbol") or "").strip() or None,
                        "description": row.get("Description"),
                        "time": row.get("Time"),
                        "error": str(exc),
                    }
                )
                continue
            fills.append(
                {
                    "symbol": sym,
                    "price": price,
                    "qty": qty,
                    "side": side,
                    "date": fdate,
                    "order": row.get("Order #", "").strip(),
                }
            )

    symbols = []
    seen = set()
    for fl in fills:
        if fl["symbol"] not in seen:
            seen.add(fl["symbol"])
            symbols.append(fl["symbol"])

    # --- per-symbol analysis -------------------------------------------------
    results = []
    warnings_list = []  # orphan/over-close STC/BTC warnings (issue M1)
    price_failures = []

    for sym in symbols:
        sym_fills = [fl for fl in fills if fl["symbol"] == sym]
        # chronological order for FIFO
        sym_fills_sorted = sorted(sym_fills, key=lambda x: x["date"])

        # FIFO position tracking with explicit sign handling (issue M3).
        lots = []  # (price, qty) open lots; qty>0 long, qty<0 short
        realized = 0.0
        buy_dates = []
        for fl in sym_fills_sorted:
            if fl["side"] == "BTO":
                lots.append([fl["price"], fl["qty"]])  # qty already normalized +
                buy_dates.append(fl["date"])
            elif fl["side"] == "STO":
                lots.append([fl["price"], fl["qty"]])  # qty already normalized -
            elif fl["side"] == "STC":
                # close a long lot; magnitude only, ignore description sign
                remaining = fl["qty"]
                while remaining > 0 and lots and lots[0][1] > 0:
                    lot = lots[0]
                    matched = min(remaining, lot[1])
                    realized += (fl["price"] - lot[0]) * matched
                    lot[1] -= matched
                    remaining -= matched
                    if lot[1] == 0:
                        lots.pop(0)
                if remaining > 0:
                    msg = (
                        f"{sym}: STC of {fl['qty']} sh on {fl['date']} "
                        f"exceeded open long lots by {remaining} sh "
                        f"(orphan/over-close sell discarded)"
                    )
                    warnings_list.append(msg)
            elif fl["side"] == "BTC":
                # close a short lot; magnitude only, ignore description sign
                remaining = fl["qty"]
                while remaining > 0 and lots and lots[0][1] < 0:
                    lot = lots[0]
                    matched = min(remaining, -lot[1])
                    realized += (lot[0] - fl["price"]) * matched
                    lot[1] += matched
                    remaining -= matched
                    if lot[1] == 0:
                        lots.pop(0)
                if remaining > 0:
                    msg = (
                        f"{sym}: BTC of {fl['qty']} sh on {fl['date']} "
                        f"exceeded open short lots by {remaining} sh "
                        f"(orphan/over-close buy discarded)"
                    )
                    warnings_list.append(msg)
            else:
                # Defensive: unexpected side should not reach here (parser
                # restricts to BTO/STC/STO/BTC), but never crash on it.
                warnings_list.append(
                    f"{sym}: unexpected side {fl['side']!r} on {fl['date']} "
                    f"(skipped)"
                )

        open_qty = sum(q for _, q in lots)
        open_entry = (
            sum(p * q for p, q in lots) / open_qty if open_qty > 0 else 0.0
        )

        # --- market price ----------------------------------------------------
        price, price_date = get_current_price(sym)
        if price is None:
            price_failures.append(sym)

        # --- congressional (de-duplicated) ----------------------------------
        rows = get_congressional(sym)
        total_tx = len(rows)
        buys = [r for r in rows if str(r[4]).strip().lower() in BUY_TYPES]
        n_buys = len(buys)
        buy_eff_dates = [d for d in (_eff_date(r) for r in buys) if d is not None]

        # data span for base-rate (issue H2): min->max effective date over ALL
        # transactions for this ticker.
        all_eff = [d for d in (_eff_date(r) for r in rows) if d is not None]
        if len(all_eff) >= 2:
            span_days = (max(all_eff) - min(all_eff)).days
        else:
            span_days = 0.0

        # earliest user buy date for this symbol = reference for match window
        ref_buy_date = min(buy_dates) if buy_dates else None

        # --- multi-window counts + closest buy (issue H1) -------------------
        window_counts = {w: 0 for w in REPORT_WINDOWS}
        closest = None  # (delta_days, detail)
        if ref_buy_date is not None:
            for r in buys:
                ed = _eff_date(r)
                if ed is None:
                    continue
                delta = (ref_buy_date - ed).days  # >=0 means buy is before BTO
                for w in REPORT_WINDOWS:
                    # inclusive of both endpoints: 0 <= delta <= w
                    if 0 <= delta <= w:
                        window_counts[w] += 1
                # closest buy = smallest non-negative delta
                if delta >= 0 and (closest is None or delta < closest[0]):
                    closest = (
                        delta,
                        {
                            "member": r[0],
                            "transaction_type": r[4],
                            "transaction_date": str(r[2]) if r[2] else None,
                            "disclosure_date": str(r[3]) if r[3] else None,
                            "eff_date": str(ed),
                        },
                    )

        expected_prob_by_window = {
            w: round(_expected_match_prob(n_buys, span_days, w), 4)
            for w in REPORT_WINDOWS
        }

        match_details = []
        if ref_buy_date is not None:
            lo = ref_buy_date - timedelta(days=MATCH_WINDOW_DAYS)
            for r in buys:
                ed = _eff_date(r)
                if ed is None:
                    continue
                if lo <= ed <= ref_buy_date:
                    match_details.append(
                        {
                            "member": r[0],
                            "transaction_type": r[4],
                            "transaction_date": str(r[2]) if r[2] else None,
                            "disclosure_date": str(r[3]) if r[3] else None,
                            "eff_date": str(ed),
                        }
                    )

        # --- P&L -------------------------------------------------------------
        if open_qty > 0:
            # open / unrealized position (long)
            current = price
            if current is None:
                pnl_unit = None
                dollar = None
                note = "OPEN (unrealized); market price unavailable"
            else:
                pnl_unit = current - open_entry
                dollar = pnl_unit * open_qty
                note = "open, unrealized"
            qty_shown = open_qty
            entry = open_entry
            current_shown = current
        elif open_qty < 0:
            # open short
            current = price
            if current is None:
                pnl_unit = None
                dollar = None
                note = "OPEN short (unrealized); market price unavailable"
            else:
                pnl_unit = open_entry - current
                dollar = pnl_unit * (-open_qty)
                note = "open short, unrealized"
            qty_shown = open_qty
            entry = open_entry
            current_shown = current
        else:
            # fully closed (or flat) -> report realized
            stc = [fl for fl in sym_fills_sorted if fl["side"] in ("STC", "BTC")]
            if stc:
                total_s = sum(fl["price"] * abs(fl["qty"]) for fl in stc)
                total_q = sum(abs(fl["qty"]) for fl in stc)
                sell_avg = total_s / total_q
            else:
                sell_avg = None
            bto = [fl for fl in sym_fills_sorted if fl["side"] in ("BTO", "STO")]
            entry = (
                sum(fl["price"] * abs(fl["qty"]) for fl in bto)
                / sum(abs(fl["qty"]) for fl in bto)
                if bto
                else 0.0
            )
            qty_shown = sum(abs(fl["qty"]) for fl in bto)
            current_shown = sell_avg
            if realized != 0 or qty_shown != 0:
                pnl_unit = (realized / qty_shown) if qty_shown else 0.0
                dollar = realized
                note = "CLOSED, realized"
            else:
                pnl_unit = None
                dollar = None
                note = "flat / no position"

        proxy_flag = "Y" if match_details else "N"
        if total_tx == 0:
            note_extra = f"{sym}: ZERO congressional transactions ever"
        elif n_buys == 0:
            note_extra = f"{sym}: only congressional SALE(S), no Purchase tracked"
        else:
            note_extra = ""

        if note_extra:
            note = (note + "; " + note_extra).strip("; ")

        results.append(
            {
                "symbol": sym,
                "qty": qty_shown,
                "entry": round(entry, 4) if entry is not None else None,
                "current": round(current_shown, 4) if current_shown is not None else None,
                "pnl_per_unit": round(pnl_unit, 4) if pnl_unit is not None else None,
                "dollar_pnl": round(dollar, 2) if dollar is not None else None,
                "congressional_total_tx": total_tx,
                "congressional_buys": n_buys,
                "window_counts": window_counts,
                "closest_buy_delta_days": closest[0] if closest else None,
                "closest_buy": closest[1] if closest else None,
                "expected_match_prob_by_window": expected_prob_by_window,
                "congressional_buy_within_60d": proxy_flag,
                "recommended_by_engine": None,
                "match_details": match_details,
                "note": note,
            }
        )

    # --- render table --------------------------------------------------------
    header = [
        "Symbol",
        "Qty",
        "Entry",
        "Current",
        "P&L/u",
        "$P&L",
        "CgBuys",
        "W30",
        "W45",
        "W60",
        "W90",
        "Δd",
        "Exp@60",
        "Proxy60",
        "Note",
    ]
    rows_str = []
    for r in results:
        rows_str.append(
            [
                r["symbol"],
                f"{r['qty']}",
                f"{r['entry']:.2f}" if r["entry"] is not None else "-",
                f"{r['current']:.2f}" if r["current"] is not None else "-",
                f"{r['pnl_per_unit']:+.2f}" if r["pnl_per_unit"] is not None else "-",
                f"{r['dollar_pnl']:+.2f}" if r["dollar_pnl"] is not None else "-",
                f"{r['congressional_buys']}",
                f"{r['window_counts'][30]}",
                f"{r['window_counts'][45]}",
                f"{r['window_counts'][60]}",
                f"{r['window_counts'][90]}",
                f"{r['closest_buy_delta_days']}" if r["closest_buy_delta_days"] is not None else "-",
                f"{r['expected_match_prob_by_window'][60]:.2f}",
                r["congressional_buy_within_60d"],
                r["note"],
            ]
        )

    widths = [len(h) for h in header]
    for row in rows_str:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print()
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows_str:
        print(fmt(row))

    # --- market price detail -------------------------------------------------
    print()
    print("Market prices used (yfinance, auto_adjust=False, period=1d):")
    for sym in symbols:
        price, price_date = get_current_price(sym)
        if price is None:
            print(f"  {sym}: UNAVAILABLE")
        else:
            print(f"  {sym}: {price:.2f} @ {price_date}")

    # --- summary -------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("RECOMMENDATION LABEL USED: 'congressional_buy_within_60d' — a FACTUAL "
          "PROXY.")
    print("It is 'Y' when >=1 congressional Purchase occurred within 60 "
          "INCLUSIVE days")
    print("before the user's earliest BTO. It is NOT a program recommendation; "
          "the real")
    print("engine (signals backtest / conviction+quality filter) was NOT "
          "invoked here.")
    print("'Exp@60' = expected match probability by chance (Poisson base-rate), "
          "judge a")
    print("hit against its prior, not in isolation. W30/W45/W60/W90 = "
          "congressional buy")
    print("counts within that many inclusive days before the BTO. Δd = day-delta "
          "to the")
    print("closest congressional buy (0 = same day as BTO).")
    print("-" * 80)
    for r in results:
        if r["congressional_buy_within_60d"] == "Y":
            status = (
                f"PROXY HIT (congressional Purchase within {MATCH_WINDOW_DAYS}d "
                f"before user BTO); exp-by-chance={r['expected_match_prob_by_window'][60]:.2f}"
            )
        elif r["congressional_buys"] > 0:
            status = "no proxy hit (congressional buys exist but outside 60d window)"
        elif r["congressional_total_tx"] == 0:
            status = "no proxy hit (ZERO congressional transactions for ticker, ever)"
        else:
            status = "no proxy hit (only congressional sales, no purchases)"
        print(f"- {r['symbol']}: {status}")
        if r["match_details"]:
            for m in r["match_details"]:
                print(
                    f"    -> {m['member']} {m['transaction_type']} eff {m['eff_date']} "
                    f"(disc {m['disclosure_date']})"
                )
        if r["closest_buy"]:
            print(
                f"    closest buy: {r['closest_buy_delta_days']}d before BTO "
                f"({r['closest_buy']['member']} eff {r['closest_buy']['eff_date']})"
            )

    print()
    if price_failures:
        print("Market price failures / unavailable: " + ", ".join(price_failures))
    else:
        print("All market prices fetched successfully via yfinance.")

    print()
    if warnings_list:
        print("WARNINGS (orphan/over-close / unexpected side):")
        for w in warnings_list:
            print("  ! " + w)
    else:
        print("No orphan/over-close sell warnings.")

    print()
    if parse_failures:
        print("PARSE FAILURES (rows skipped, run continued):")
        for pf in parse_failures:
            print(
                f"  ! line {pf['line']}: sym={pf['symbol']} "
                f"desc={pf['description']!r} time={pf['time']!r} -> {pf['error']}"
            )
    else:
        print("No unparseable rows.")

    # --- write JSON ----------------------------------------------------------
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(DB_PATH),
        "match_window_days": MATCH_WINDOW_DAYS,
        "report_windows": REPORT_WINDOWS,
        "recommendation_label": "congressional_buy_within_60d",
        "recommendation_note": (
            "FACTUAL PROXY only: 'Y' if >=1 congressional Purchase within the "
            f"{MATCH_WINDOW_DAYS}-day inclusive window before the user's earliest "
            "BTO. NOT the program's recommendation engine output; the real engine "
            "(src/analyzer/signals backtest / conviction+quality filter) was NOT "
            "invoked."
        ),
        "price_failures": price_failures,
        "warnings": warnings_list,
        "parse_failures": parse_failures,
        "table": results,
    }
    out_path = SCRIPT_DIR / "blotter_reconcile_output.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote structured output -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
