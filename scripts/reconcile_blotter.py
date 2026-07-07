#!/usr/bin/env python3
"""Reconcile a Thinkorswim brokerage blotter against the congressional PTR
database and current market prices.

Usage:
    PYTHONPATH=src python3 scripts/reconcile_blotter.py <csv_path>

Outputs a text table to stdout, a plain-language summary, and writes a
structured JSON result to scripts/blotter_reconcile_output.json.
"""
from __future__ import annotations

import csv
import json
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

# Window (days) before the user's buy date in which a congressional buy counts
# as "recommendation-consistent".
MATCH_WINDOW_DAYS = 60


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
    """Return (share_delta:int, side:str) from e.g. '100 BTO' / '-10 STC'."""
    s = raw.strip()
    m = re.match(r"^(-?\d+)\s+(BTO|STC|STO|BTC)\b", s, re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot parse description {raw!r}")
    qty = int(m.group(1))
    side = m.group(2).upper()
    return qty, side


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
        price_date = hist.index[-1].to_pydatetime().date()
        return price, price_date
    except Exception as e:  # noqa: BLE001 - network/parse can fail broadly
        return None, None


def get_congressional(ticker: str):
    """Return all transactions (any type) for ticker, case-insensitive."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT member, ticker, transaction_date, disclosure_date,
                   transaction_type, owner_code, amount_raw
            FROM transactions
            WHERE LOWER(ticker) = LOWER(?)
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
    print(f"Match win: {MATCH_WINDOW_DAYS} days before user buy")
    print("=" * 72)

    # --- parse blotter -------------------------------------------------------
    fills = []  # dict per row
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not row.get("Symbol"):
                continue
            sym = row["Symbol"].strip()
            price = parse_price(row["Price"])
            qty, side = parse_description(row["Description"])
            fdate = parse_fill_date(row["Time"])
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
    failures = []

    for sym in symbols:
        sym_fills = [fl for fl in fills if fl["symbol"] == sym]
        # chronological order for FIFO
        sym_fills_sorted = sorted(sym_fills, key=lambda x: x["date"])

        # FIFO position tracking
        lots = []  # (price, qty) open long lots
        realized = 0.0
        buy_dates = []
        for fl in sym_fills_sorted:
            if fl["side"] in ("BTO", "STO"):
                # BTO = buy to open (long). STO would be sell to open (short) - treat as negative.
                if fl["side"] == "BTO":
                    lots.append([fl["price"], fl["qty"]])
                    buy_dates.append(fl["date"])
                else:
                    lots.append([fl["price"], -fl["qty"]])
            elif fl["side"] in ("STC", "BTC"):
                # STC = sell to close (reduce long). BTC = buy to close (reduce short).
                remaining = abs(fl["qty"])  # positive magnitude to close
                if fl["side"] == "STC":
                    while remaining > 0 and lots and lots[0][1] > 0:
                        lot = lots[0]
                        matched = min(remaining, lot[1])
                        realized += (fl["price"] - lot[0]) * matched
                        lot[1] -= matched
                        remaining -= matched
                        if lot[1] == 0:
                            lots.pop(0)
                else:  # BTC reduces a short lot
                    while remaining > 0 and lots and lots[0][1] < 0:
                        lot = lots[0]
                        matched = min(remaining, -lot[1])
                        realized += (lot[0] - fl["price"]) * matched
                        lot[1] += matched
                        remaining -= matched
                        if lot[1] == 0:
                            lots.pop(0)

        open_qty = sum(q for _, q in lots)
        open_entry = (
            sum(p * q for p, q in lots) / open_qty if open_qty > 0 else 0.0
        )

        # --- market price ----------------------------------------------------
        price, price_date = get_current_price(sym)
        if price is None:
            failures.append(sym)

        # --- congressional ---------------------------------------------------
        rows = get_congressional(sym)
        total_tx = len(rows)
        buys = [r for r in rows if str(r[4]).strip().lower() in BUY_TYPES]
        n_buys = len(buys)

        # earliest user buy date for this symbol = reference for match window
        ref_buy_date = min(buy_dates) if buy_dates else None

        matches = []
        if ref_buy_date is not None:
            lo = ref_buy_date - timedelta(days=MATCH_WINDOW_DAYS)
            for r in buys:
                ed = _eff_date(r)
                if ed is None:
                    continue
                if lo <= ed <= ref_buy_date:
                    matches.append(
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
            # open / unrealized position
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
        else:
            # fully closed (or flat) -> report realized
            # current shown = avg sell price (from STC fills)
            stc = [fl for fl in sym_fills_sorted if fl["side"] in ("STC", "BTC")]
            if stc:
                total_s = sum(fl["price"] * abs(fl["qty"]) for fl in stc)
                total_q = sum(abs(fl["qty"]) for fl in stc)
                sell_avg = total_s / total_q
            else:
                sell_avg = None
            bto = [fl for fl in sym_fills_sorted if fl["side"] in ("BTO", "STO")]
            entry = sum(fl["price"] * abs(fl["qty"]) for fl in bto) / sum(abs(fl["qty"]) for fl in bto) if bto else 0.0
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

        matches_program = "Y" if matches else "N"
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
                "matches_program": matches_program,
                "match_details": matches,
                "note": note,
            }
        )

    # --- render table --------------------------------------------------------
    header = [
        "Symbol",
        "Qty",
        "Entry",
        "Current",
        "P&L/unit",
        "$P&L",
        "CgBuys",
        "Match?",
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
                r["matches_program"],
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
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for r in results:
        if r["matches_program"] == "Y":
            status = f"MATCHES program (congressional buy within {MATCH_WINDOW_DAYS}d before user buy)"
        elif r["congressional_buys"] > 0:
            status = "does NOT match program (congressional buys exist but outside 60d window)"
        elif r["congressional_total_tx"] == 0:
            status = "does NOT match program (ZERO congressional transactions for ticker, ever)"
        else:
            status = "does NOT match program (only congressional sales, no purchases)"
        print(f"- {r['symbol']}: {status}")
        if r["match_details"]:
            for m in r["match_details"]:
                print(
                    f"    -> {m['member']} {m['transaction_type']} eff {m['eff_date']} "
                    f"(disc {m['disclosure_date']})"
                )

    print()
    if failures:
        print("yfinance price failures / unavailable: " + ", ".join(failures))
    else:
        print("All market prices fetched successfully via yfinance.")

    # --- write JSON ----------------------------------------------------------
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(DB_PATH),
        "match_window_days": MATCH_WINDOW_DAYS,
        "price_failures": failures,
        "table": results,
    }
    out_path = SCRIPT_DIR / "blotter_reconcile_output.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote structured output -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
