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

from analyzer.member_names import canonical_member_key
from analyzer.parsing.cells import _extract_amount_midpoint

logger = logging.getLogger(__name__)

_BUY_TYPES = frozenset({"purchase"})
_SELL_TYPES = frozenset({"sale", "sale full", "sale partial", "partial sale"})
_OPTION_MARKERS = frozenset({"option", "call", "put", "stock option"})


class PositionsError(Exception):
    """Raised when positions cannot be reconstructed safely."""


def _is_option_row(row, has_instrument: bool) -> bool:
    """Positive option evidence only; NULL/unknown instruments stay equity."""
    if has_instrument:
        instrument = row.instrument_type
        if instrument is not None and not pd.isna(instrument):
            if str(instrument).strip().lower() in _OPTION_MARKERS:
                return True
    return False


def entry_close(
    prices: pd.DataFrame, day: date, as_of: date | None = None
) -> float | None:
    """First close on or after the transaction date.

    Point-in-time: when as_of is given, closes after it are unknown, so a
    transaction whose entry bar does not exist yet yields None.
    """
    frame = prices
    if as_of is not None:
        frame = prices.loc[prices.index <= pd.Timestamp(as_of)]
    closes = frame.loc[frame.index >= pd.Timestamp(day), "close"]
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
    has_instrument = "instrument_type" in trades.columns
    has_disclosure = "disclosure_date" in trades.columns
    if not has_midpoint and not has_raw:
        raise PositionsError("trades need amount_midpoint or amount_raw for sizing")

    lots: dict[tuple[str, str], list] = {}
    first_buy: dict[tuple[str, str], date] = {}
    last_activity: dict[tuple[str, str], date] = {}
    floor_sized: dict[tuple[str, str], int] = {}
    member_identities: dict[str, list[str]] = {}
    member_keys: dict[str, str] = {}
    member_labels: dict[str, str] = {}
    skipped = 0
    skipped_options = 0
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
    if "amends_source_record_id" in trades.columns:
        superseded = {
            str(value)
            for value in trades["amends_source_record_id"].dropna()
            if str(value).strip()
        }
    else:
        superseded = set()
    if superseded and "source_record_id" in trades.columns:
        mask = trades["source_record_id"].astype("string").isin(superseded)
        dropped = int(mask.sum())
        if dropped:
            logger.info("Dropping %d rows superseded by amendments", dropped)
            ordered = ordered.loc[~mask].copy()
    for row in ordered.itertuples():
        if pd.isna(row.member) or pd.isna(row.ticker):
            skipped += 1
            continue
        member = str(row.member)
        canonical = canonical_member_key(member)
        member_key = member_keys.get(canonical)
        if member_key is None:
            # ponytail: scan distinct aliases; index by surname if this grows costly.
            member_key = next(
                (
                    identity
                    for identity, variants in member_identities.items()
                    if all(
                        _same_member_variant(canonical, variant)
                        for variant in variants
                    )
                ),
                canonical,
            )
            member_keys[canonical] = member_key
            member_identities.setdefault(member_key, []).append(canonical)
            member_labels.setdefault(member_key, member)
        key = (member_key, str(row.ticker))
        day = pd.Timestamp(row.transaction_date).date()
        kind = str(row.transaction_type or "").strip().lower()
        if _is_option_row(row, has_instrument):
            skipped_options += 1
            continue
        disclosed = row.disclosure_date if has_disclosure else None
        disclosure_day = (
            pd.Timestamp(disclosed).date()
            if disclosed is not None and not pd.isna(disclosed)
            else day
        )
        if disclosure_day > as_of or day > as_of:
            skipped += 1
            continue
        last_activity[key] = day

        history = price_history.get(str(row.ticker))
        if history is None or history.empty:
            skipped += 1
            continue

        if kind in _BUY_TYPES:
            entry = entry_close(history, day, as_of)
            size, by_floor = _size(row)
            if entry is None or size is None:
                skipped += 1
                continue
            shares = size / entry
            source = row.source if "source" in trades.columns else None
            lots.setdefault(key, []).append([shares, size, disclosure_day, source])
            if by_floor:
                floor_sized[key] = floor_sized.get(key, 0) + 1
            first_buy.setdefault(key, day)
        elif kind in _SELL_TYPES:
            if kind == "sale full":
                # A full sale closes the position regardless of the filed
                # dollar amount: estimated-amount FIFO can otherwise leave
                # phantom residual shares behind.
                lots[key] = []
                first_buy.setdefault(key, day)
                continue
            entry = entry_close(history, day, as_of)
            size, _ = _size(row)
            if entry is None or size is None:
                skipped += 1
                continue
            to_relieve = size / entry
            queue = lots.setdefault(key, [])
            while to_relieve > 0 and queue:
                lot_shares, lot_cost, lot_disclosure, lot_source = queue[0]
                if lot_shares <= to_relieve:
                    to_relieve -= lot_shares
                    queue.pop(0)
                else:
                    fraction = to_relieve / lot_shares
                    queue[0] = [
                        lot_shares - to_relieve,
                        lot_cost * (1 - fraction),
                        lot_disclosure,
                        lot_source,
                    ]
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
                "member": member_labels[key[0]],
                "ticker": key[1],
                "shares": shares,
                "total_cost": cost,
                "cost_basis": basis,
                "current_price": current,
                "discount_pct": (current / basis - 1) * 100,
                "first_buy": first_buy[key],
                "last_activity": last_activity[key],
                "disclosure_date": max(lot[2] for lot in queue),
                "source": queue[-1][3],
                "floor_sized_lots": floor_sized.get(key, 0),
            }
        )

    logger.info(
        "Built %d open positions (%d skipped rows, %d option rows, %d exchanges ignored)",
        len(records),
        skipped,
        skipped_options,
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
        "disclosure_date",
        "source",
        "floor_sized_lots",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        "discount_pct", kind="mergesort"
    )


def holdings_candidates(
    db, member: str, as_of: date, *, include_unpromoted: bool = False
) -> pd.DataFrame:
    """Return open positions as uncorroborated, unexpired setup candidates.

    Lots count only when their disclosure date (or transaction date when
    missing) is no later than ``as_of``. Position valuation uses the last close
    on or before ``as_of``; future closes cannot enter a replay. Positions have
    no age expiry and remain candidates until sales close their lots.
    """
    columns = [
        "ticker",
        "actor_id",
        "kind",
        "source",
        "entry_ref",
        "event_date",
        "disclosure_date",
        "corroboration",
        "position_evidence",
        "as_of",
    ]
    source_kinds = {
        "house_pdf": "congress",
        "gemini_ocr": "congress",
        "senate_efd": "congress",
        "form4": "officer",
        "13f": "manager",
    }

    try:
        from analyzer.actors import actor_id
    except ModuleNotFoundError as exc:
        if exc.name != "analyzer.actors":
            raise

        # analyzer.actors is not present in older installs; keep its stable format.
        def _actor_id(kind, key):
            return f"{kind}:{key}"

    else:
        _actor_id = actor_id

    trades = load_member_trades(db, member, include_unpromoted=include_unpromoted)
    if trades.empty:
        return pd.DataFrame(columns=columns)

    trades = trades.loc[trades["source"].isin(source_kinds)].copy()
    if trades.empty:
        return pd.DataFrame(columns=columns)

    histories = {
        str(ticker): load_price_history(db, str(ticker))
        for ticker in trades["ticker"].dropna().unique()
    }
    records = []
    for actor_kind in ("congress", "officer", "manager"):
        kind_trades = trades.loc[
            trades["source"].map(source_kinds).eq(actor_kind)
        ]
        if kind_trades.empty:
            continue
        positions = build_positions(kind_trades, histories, as_of=as_of)
        for position in positions.itertuples(index=False):
            source = position.source
            records.append(
                {
                    "ticker": position.ticker,
                    "actor_id": _actor_id(actor_kind, member),
                    "kind": actor_kind,
                    "source": source,
                    "entry_ref": (
                        None if source == "13f" else float(position.cost_basis)
                    ),
                    "event_date": position.last_activity,
                    "disclosure_date": position.disclosure_date,
                    "corroboration": False,
                    "position_evidence": True,
                    "as_of": as_of,
                }
            )

    result = pd.DataFrame(records, columns=columns)
    if not result.empty:
        result["entry_ref"] = pd.Series(
            [record["entry_ref"] for record in records], dtype=object
        )
    return result


def discount_alerts(positions: pd.DataFrame, *, threshold_pct: float = -10.0) -> pd.DataFrame:
    """Positions trading at or below the discount threshold (negative pct)."""
    if positions.empty:
        return positions.copy()
    return positions.loc[positions["discount_pct"] <= threshold_pct].copy()


_NICKNAMES = {
    "bob": "robert", "bill": "william", "jim": "james", "mike": "michael",
    "tom": "thomas", "dave": "david", "dan": "daniel", "steve": "steven",
    "joe": "joseph", "matt": "matthew", "nick": "nicholas", "alex": "alexander",
    "chris": "christopher", "pat": "patrick", "tim": "timothy",
    "jeff": "jeffrey", "greg": "gregory", "ron": "ronald", "ken": "kenneth",
    "larry": "lawrence", "rick": "richard", "chuck": "charles", "ro": "rohit",
}


def _same_person(first: str, candidate: str) -> bool:
    """Last-name-gated nickname equivalence for filed name variants."""
    if first == candidate:
        return True
    for short, full in _NICKNAMES.items():
        if {first, candidate} == {short, full}:
            return True
    return False


def _same_member_variant(first: str, candidate: str) -> bool:
    """Match filed variants without joining conflicting middle initials."""
    first_tokens = first.split()
    candidate_tokens = candidate.split()
    if len(first_tokens) < 2 or len(candidate_tokens) < 2:
        return first == candidate
    if first_tokens[-1] != candidate_tokens[-1]:
        return False
    if not _same_person(first_tokens[0].lower(), candidate_tokens[0].lower()):
        return False
    first_middle = first_tokens[1][0] if len(first_tokens) > 2 else None
    candidate_middle = candidate_tokens[1][0] if len(candidate_tokens) > 2 else None
    return (
        first_middle is None
        or candidate_middle is None
        or first_middle == candidate_middle
    )


def _member_variants(db, member: str) -> list[str]:
    """All filed name variants for the member.

    Filers appear as 'Charles J. Fleischmann', 'Charles J. "Chuck"
    Fleischmann', and 'Charles J Fleischmann' across filings; users type
    'Chuck Fleischmann'. Match on canonical key first, then on
    last-name plus nickname-equivalent first name. An exact match on any
    one variant previously dropped the others' positions silently. Middle
    initials must agree when both names provide one.
    """
    if not isinstance(member, str) or not member.strip():
        return [member]
    wanted = canonical_member_key(member)
    try:
        names = db.conn.execute("SELECT DISTINCT member FROM transactions").fetchall()
    except Exception:
        names = []
    try:
        names += db.conn.execute(
            "SELECT DISTINCT member FROM canonical_transactions"
        ).fetchall()
    except Exception:
        pass
    seen = []
    for row in names:
        name = row[0]
        if name in seen:
            continue
        key = canonical_member_key(name)
        if key == wanted:
            seen.append(name)
        elif _same_member_variant(key, wanted):
            seen.append(name)
    return seen or [member]


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
               disclosure_date, amount_midpoint, amount_raw, source,
               instrument_type, source_record_id, amends_source_record_id
        FROM canonical_transactions
        WHERE member IN (SELECT UNNEST(?))
        """,
        [_member_variants(db, member)],
    ).fetchdf()
    if not include_unpromoted:
        return canonical.sort_values("transaction_date", kind="mergesort")

    raw = db.conn.execute(
        """
        SELECT member, ticker, transaction_type, transaction_date,
               disclosure_date, amount_midpoint, amount_raw, source,
               instrument_type, source_record_id, amends_source_record_id
        FROM (
            SELECT t.*,
                ROW_NUMBER() OVER (
                    PARTITION BY member, ticker, transaction_date,
                        transaction_type, disclosure_date
                    ORDER BY amount_midpoint DESC NULLS LAST
                ) AS rn
            FROM transactions t
            WHERE t.member IN (SELECT UNNEST(?))
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
        [_member_variants(db, member)],
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
