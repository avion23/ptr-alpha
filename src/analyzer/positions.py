"""Open-position tracker for skilled members.

Closed-window scoring judges each filing once and never revisits it. This
module answers the follow-up question instead: what do the best members
currently hold, what did it cost them, and which positions are on sale.

Congressional filings carry dollar bands, not share counts, so positions are
reconstructed from implied shares (amount_midpoint / entry close, where entry
is the first close on or after the transaction date). Sells relieve lots
FIFO. Option exercises file as Purchases and therefore already count as
accumulation; exchanges are ignored and reported.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from analyzer.parsing.cells import _extract_amount_midpoint

logger = logging.getLogger(__name__)

_BUY_TYPES = frozenset({"purchase"})
_SELL_TYPES = frozenset({"sale", "sale full", "sale partial", "partial sale"})


class PositionsError(Exception):
    """Raised when positions cannot be reconstructed safely."""


def _entry_close(prices: pd.DataFrame, day: date) -> float | None:
    """First close on or after the transaction date."""
    closes = prices.loc[prices.index >= pd.Timestamp(day), "close"]
    closes = closes[closes.notna()]
    if closes.empty:
        return None
    price = float(closes.iloc[0])
    return price if price > 0 else None


def _current_close(prices: pd.DataFrame, as_of: date) -> float | None:
    """Last close on or before the as-of date."""
    closes = prices.loc[prices.index <= pd.Timestamp(as_of), "close"]
    closes = closes[closes.notna()]
    if closes.empty:
        return None
    price = float(closes.iloc[-1])
    return price if price > 0 else None


def build_positions(
    trades: pd.DataFrame,
    price_history: dict[str, pd.DataFrame],
    *,
    as_of: date,
) -> pd.DataFrame:
    """Reconstruct open lots per (member, ticker) with FIFO sale relief.

    `trades` needs member, ticker, transaction_type, transaction_date,
    amount_midpoint. `price_history` maps ticker -> frame indexed by date
    with a close column. Returns one row per open position with cost basis,
    current price, and discount_pct (negative = below member cost).
    """
    required = {
        "member",
        "ticker",
        "transaction_type",
        "transaction_date",
    }
    missing = required - set(trades.columns)
    if missing:
        raise PositionsError(f"trades missing columns: {sorted(missing)}")
    has_midpoint = "amount_midpoint" in trades.columns
    has_raw = "amount_raw" in trades.columns
    if not has_midpoint and not has_raw:
        raise PositionsError("trades need amount_midpoint or amount_raw for sizing")

    lots: dict[tuple[str, str], list] = {}
    first_buy: dict[tuple[str, str], date] = {}
    last_activity: dict[tuple[str, str], date] = {}
    floor_sized: dict[tuple[str, str], int] = {}
    skipped = 0
    ignored_exchanges = 0

    def _size(row) -> tuple[float | None, bool]:
        """Dollar size with floor fallback for open-ended bands.

        Returns (size, sized_by_floor). A '$100,001 -' band parses to its
        $100,001 floor: a conservative lower bound, flagged as such.
        """
        midpoint = row.amount_midpoint if has_midpoint else None
        if midpoint is not None and not pd.isna(midpoint) and float(midpoint) > 0:
            return float(midpoint), False
        raw = row.amount_raw if has_raw else None
        if raw is None or pd.isna(raw):
            return None, False
        _, floor = _extract_amount_midpoint(str(raw))
        if floor is None or floor <= 0:
            return None, False
        return float(floor), True

    ordered = trades.sort_values("transaction_date", kind="mergesort")
    for row in ordered.itertuples():
        if pd.isna(row.member) or pd.isna(row.ticker):
            skipped += 1
            continue
        key = (str(row.member), str(row.ticker))
        day = pd.Timestamp(row.transaction_date).date()
        kind = str(row.transaction_type or "").strip().lower()
        last_activity[key] = day

        history = price_history.get(str(row.ticker))
        if history is None or history.empty:
            skipped += 1
            continue

        if kind in _BUY_TYPES:
            entry = _entry_close(history, day)
            size, by_floor = _size(row)
            if entry is None or size is None:
                skipped += 1
                continue
            shares = size / entry
            lots.setdefault(key, []).append([shares, size])
            if by_floor:
                floor_sized[key] = floor_sized.get(key, 0) + 1
            first_buy.setdefault(key, day)
        elif kind in _SELL_TYPES:
            entry = _entry_close(history, day)
            size, _ = _size(row)
            if entry is None or size is None:
                skipped += 1
                continue
            to_relieve = size / entry
            queue = lots.setdefault(key, [])
            while to_relieve > 0 and queue:
                lot_shares, lot_cost = queue[0]
                if lot_shares <= to_relieve:
                    to_relieve -= lot_shares
                    queue.pop(0)
                else:
                    fraction = to_relieve / lot_shares
                    queue[0] = [lot_shares - to_relieve, lot_cost * (1 - fraction)]
                    to_relieve = 0
            first_buy.setdefault(key, day)
        else:
            ignored_exchanges += 1

    records = []
    for key, queue in lots.items():
        shares = sum(lot[0] for lot in queue)
        cost = sum(lot[1] for lot in queue)
        if shares <= 0 or cost <= 0:
            continue
        history = price_history[key[1]]
        current = _current_close(history, as_of)
        if current is None:
            skipped += 1
            continue
        basis = cost / shares
        records.append(
            {
                "member": key[0],
                "ticker": key[1],
                "shares": shares,
                "total_cost": cost,
                "cost_basis": basis,
                "current_price": current,
                "discount_pct": (current / basis - 1) * 100,
                "first_buy": first_buy[key],
                "last_activity": last_activity[key],
                "floor_sized_lots": floor_sized.get(key, 0),
            }
        )

    logger.info(
        "Built %d open positions (%d skipped rows, %d exchanges ignored)",
        len(records),
        skipped,
        ignored_exchanges,
    )
    columns = [
        "member",
        "ticker",
        "shares",
        "total_cost",
        "cost_basis",
        "current_price",
        "discount_pct",
        "first_buy",
        "last_activity",
        "floor_sized_lots",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        "discount_pct", kind="mergesort"
    )


def discount_alerts(positions: pd.DataFrame, *, threshold_pct: float = -10.0) -> pd.DataFrame:
    """Positions trading at or below the discount threshold (negative pct)."""
    if positions.empty:
        return positions.copy()
    return positions.loc[positions["discount_pct"] <= threshold_pct].copy()


def load_member_trades(db, member: str, *, include_unpromoted: bool = False) -> pd.DataFrame:
    """All canonical rows for one member across every year and source.

    With include_unpromoted, raw filings from not-yet-promoted House
    generations are added (deduped to one row per economic event, preferring
    closed bands). Those rows are UNVALIDATED: same filing may parse
    differently across generations, so amounts are approximate. Canonical
    rows always win; the flag only widens recency, never scoring.
    """
    canonical = db.conn.execute(
        """
        SELECT member, ticker, transaction_type, transaction_date,
               disclosure_date, amount_midpoint, amount_raw, source
        FROM canonical_transactions
        WHERE member = ?
        """,
        [member],
    ).fetchdf()
    if not include_unpromoted:
        return canonical.sort_values("transaction_date", kind="mergesort")

    raw = db.conn.execute(
        """
        SELECT member, ticker, transaction_type, transaction_date,
               disclosure_date, amount_midpoint, amount_raw, source
        FROM (
            SELECT t.*,
                ROW_NUMBER() OVER (
                    PARTITION BY member, ticker, transaction_date,
                        transaction_type, disclosure_date
                    ORDER BY amount_midpoint DESC NULLS LAST
                ) AS rn
            FROM transactions t
            WHERE member = ?
              AND NOT EXISTS (
                  SELECT 1 FROM canonical_transactions c
                  WHERE c.source IS NOT DISTINCT FROM t.source
                    AND c.source_record_id IS NOT DISTINCT FROM t.source_record_id
                    AND c.source_row_id IS NOT DISTINCT FROM t.source_row_id
                    AND c.ingestion_generation IS NOT DISTINCT FROM t.ingestion_generation
              )
        )
        WHERE rn = 1
        """,
        [member],
    ).fetchdf()
    combined = pd.concat([canonical, raw], ignore_index=True)
    logger.warning(
        "Including %d unvalidated unpromoted rows for %s alongside %d canonical",
        len(raw),
        member,
        len(canonical),
    )
    return combined.sort_values("transaction_date", kind="mergesort")


def load_price_history(db, ticker: str) -> pd.DataFrame:
    """Full close history for one ticker, indexed by date."""
    frame = db.conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
        [ticker],
    ).fetchdf()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()
