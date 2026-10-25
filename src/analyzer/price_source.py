"""YFinance-backed price fetcher with cache merge."""

import logging
import hashlib
import math
import re
import time
from collections import Counter
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from analyzer.price_index import normalize_price_index
from analyzer.persistence.database import Database
from analyzer.exceptions import DataSourceError
from analyzer.settings import Settings
from analyzer.ticker_resolver import TickerResolver

logger = logging.getLogger(__name__)

_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")
_PRICE_OVERLAP_TOL = 0.001  # 0.1% relative disagreement means another generation.
_PRICE_BASIS = "yfinance-adjusted-close-v1"
# Across 6,776 independent-fetch pairs, the maximum per-date normalized adjusted
# close disagreement is 8.59014e-7 for SPY; round up to 1 ppm. The maximum
# normalized factor disagreement is 1.9797463e-6; round up to 2 ppm for local
# adjustment-factor checks (also above AAPL's measured ~1.59 ppm daily factor noise).
_PRICE_CERTIFICATION_REL_TOL = 1e-6
_PRICE_FACTOR_CERTIFICATION_REL_TOL = 2e-6


# ── YFinancePriceSource: yfinance-backed price fetcher with cache merge ──


class YFinancePriceSource:
    def __init__(
        self, settings: Settings, read_only: bool = False, db: Database | None = None
    ):
        self.settings = settings
        self.data_dir = Path(settings.data.data_dir)
        self._owns_db = db is None
        self.db = (
            db
            if db is not None
            else Database(self.data_dir / "congress.duckdb", read_only=read_only)
        )
        if not self.db.is_read_only:
            ensure_price_generation_schema(self.db)

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def get_prices(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        if len(tickers) == 0:
            raise DataSourceError("No tickers provided for price fetching")

        clean_tickers = _clean_tickers(tickers)
        all_tickers = sorted(
            list(
                set(
                    t
                    for t in _expand_rename_aliases(clean_tickers)
                    if _VALID_TICKER_RE.match(str(t))
                )
                | {"SPY"}
            )
        )

        raw_to_yf, yf_to_raw = _resolve_tickers(all_tickers)
        cached_prices = self.db.get_prices(all_tickers, start, end)
        if not cached_prices.empty:
            logger.info(
                f"Loaded cached prices: {len(cached_prices.columns)} tickers, "
                f"{len(cached_prices)} dates"
            )

        missing_tickers, missing_dates = self.db.get_missing_price_data(
            all_tickers,
            start,
            end,
        )

        if not missing_tickers and not missing_dates:
            logger.info(f"Using fully cached prices for {len(all_tickers)} tickers")
            return _validate_and_log_prices(cached_prices, all_tickers)

        return self._fetch_and_merge_prices(
            all_tickers,
            raw_to_yf,
            yf_to_raw,
            cached_prices,
            start,
            end,
            missing_tickers,
            missing_dates,
        )

    def _fetch_and_merge_prices(
        self,
        all_tickers,
        raw_to_yf,
        yf_to_raw,
        cached_prices,
        start,
        end,
        missing_tickers,
        missing_dates,
    ) -> pd.DataFrame:
        """Fetch missing data from yfinance and merge with the cache.

        A ticker that needs any data is re-fetched over the **full** analysis
        window. yfinance ``auto_adjust`` prices are only self-consistent within
        a single download; a narrow gap repair would mix a stale pre-action
        cached basis with the post-action basis after a split. Re-fetching the
        full window keeps every ticker on one adjustment basis. Because
        ``get_missing`` uses the NYSE trading-day calendar, complete tickers are
        never flagged, so this full-window path runs only for genuinely
        incomplete tickers.
        """
        fetch_tickers = missing_tickers if missing_tickers else all_tickers
        fetch_resolved = sorted({raw_to_yf.get(t, t) for t in fetch_tickers})

        logger.info(
            f"Fetching price data for {len(fetch_resolved)} tickers using yfinance"
        )

        data = self._download_yfinance(fetch_resolved, start, end)
        if data.empty:
            if not cached_prices.empty:
                logger.warning("yfinance failed, using cached data")
                return _validate_and_log_prices(cached_prices, all_tickers)
            raise DataSourceError(
                "No price data could be fetched from yfinance. Data source may be blocked or down."
            )

        raw_prices = self._extract_close_prices(data, fetch_resolved)
        adjusted_prices = self._extract_field_prices(
            data, "Adj Close", fetch_resolved, required=False
        )
        adjustment_factors = None
        if adjusted_prices is not None:
            adjustment_factors = adjusted_prices.divide(raw_prices).replace(
                [np.inf, -np.inf], np.nan
            )
            new_prices = adjusted_prices
        else:
            # Legacy test doubles and provider responses without Adj Close are
            # retained as price observations, but cannot normalize filed refs.
            new_prices = raw_prices
        if adjustment_factors is None:
            adjustment_factors = pd.DataFrame(np.nan, index=raw_prices.index, columns=raw_prices.columns)
        adjustment_factors = self._normalize_price_index(adjustment_factors)
        new_prices = self._normalize_price_index(new_prices)
        new_prices = _clip_price_window(new_prices, start, end)
        adjustment_factors = _clip_price_window(adjustment_factors, start, end)
        new_prices = new_prices.apply(pd.to_numeric, errors="coerce")
        invalid_mask = new_prices.notna() & (
            ~np.isfinite(new_prices) | new_prices.le(0)
        )
        invalid = invalid_mask.sum().sum()
        new_prices = new_prices.mask(invalid_mask)
        if invalid:
            logger.warning(
                "Rejected %d non-finite or non-positive fetched prices", invalid
            )
        new_prices = new_prices.dropna(axis=1, how="all")
        new_prices = _as_frame(new_prices)
        new_prices = self._rename_yf_columns(new_prices, raw_to_yf)
        adjustment_factors = self._rename_yf_columns(
            adjustment_factors, raw_to_yf
        ).reindex(index=new_prices.index, columns=new_prices.columns)

        generation_states = {}
        if self.db.is_read_only:
            logger.info(
                f"Read-only mode: merging {len(new_prices.columns)} fetched tickers with cache"
            )
            prices, generation_states = _merge_price_generation(
                self.db, new_prices, adjustment_factors, _PRICE_BASIS,
                persist=False, cached_prices=cached_prices,
            )
        else:
            prices, generation_states = _merge_price_generation(
                self.db, new_prices, adjustment_factors, _PRICE_BASIS,
                persist=True, cached_prices=cached_prices,
            )
            logger.info(f"Cached {len(new_prices.columns)} ticker generations")

        result = _validate_and_log_prices(prices, all_tickers)
        result.attrs["price_generation_state"] = generation_states
        return result

    def _download_yfinance(self, fetch_resolved: list[str], start, end) -> pd.DataFrame:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # yfinance treats ``end`` as exclusive while this public API
                # and the repository treat it as inclusive.
                download_end = pd.Timestamp(end) + timedelta(days=1)
                data = yf.download(
                    fetch_resolved,
                    start=start,
                    end=download_end,
                    progress=False,
                    threads=True,
                    auto_adjust=False,
                )
                if not isinstance(data, pd.DataFrame):
                    raise DataSourceError("yfinance returned no downloadable frame")
                return data
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        f"yfinance request failed (attempt {attempt + 1}/{max_retries}: {e}), "
                        f"retrying in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"yfinance request failed after {max_retries} attempts ({e}), "
                        "falling back to cached data"
                    )
                    return pd.DataFrame()
        raise DataSourceError("yfinance download retries exhausted")

    @staticmethod
    def _extract_close_prices(
        data: pd.DataFrame, fetch_resolved: list[str]
    ) -> pd.DataFrame:
        """Normalize yfinance Close output for one or many symbols."""
        return YFinancePriceSource._extract_field_prices(
            data, "Close", fetch_resolved, required=True
        )

    @staticmethod
    def _extract_field_prices(
        data: pd.DataFrame, field: str, fetch_resolved: list[str], *, required: bool
    ) -> pd.DataFrame | None:
        try:
            close = data[field]
        except KeyError as exc:
            if not required:
                return None
            raise DataSourceError(
                f"yfinance response did not contain {field} prices"
            ) from exc
        if isinstance(close, pd.Series):
            if len(fetch_resolved) != 1:
                raise DataSourceError("yfinance returned unlabeled multi-symbol Close prices")
            return close.to_frame(fetch_resolved[0])
        if not isinstance(close, pd.DataFrame):
            raise DataSourceError(f"Unsupported yfinance {field} response shape")
        # Labeled provider symbols establish identity; never relabel an
        # unexpected security just because only one symbol was requested.
        return close.loc[:, close.columns.isin(fetch_resolved)].copy()

    @staticmethod
    def _normalize_price_index(prices: pd.DataFrame) -> pd.DataFrame:
        return normalize_price_index(
            prices,
            invalid_error=DataSourceError,
            duplicate_error=DataSourceError,
            duplicate_message="yfinance returned duplicate calendar dates",
        )

    def _rename_yf_columns(
        self, new_prices: pd.DataFrame, raw_to_yf: dict
    ) -> pd.DataFrame:
        """Expose provider prices under every requested market-symbol alias.

        Data providers commonly retain a renamed security's full history only
        under its current symbol. The decision layer still needs the symbol that
        was tradable when the filing became public. Materialize both column
        names from the one provider series; row-level temporal resolution still
        decides which market symbol is eligible on a given disclosure date.
        """
        result = new_prices.copy()
        for raw, provider_symbol in raw_to_yf.items():
            if provider_symbol in new_prices.columns:
                result[raw] = new_prices[provider_symbol]
        return result


def ensure_price_generation_schema(db: Database) -> None:
    """Create additive, versioned price evidence tables (never rewrite prices)."""
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS price_generations (
            ticker VARCHAR,
            generation VARCHAR,
            basis VARCHAR,
            value_hash VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, generation)
        )
    """)
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS price_generation_values (
            ticker VARCHAR,
            generation VARCHAR,
            date DATE,
            close DOUBLE,
            adjustment_factor DOUBLE,
            basis VARCHAR,
            PRIMARY KEY (ticker, generation, date)
        )
    """)
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS price_generation_state (
            ticker VARCHAR PRIMARY KEY,
            served_generation VARCHAR,
            comparability VARCHAR,
            alternate_generation VARCHAR,
            reason VARCHAR
        )
    """)


def price_generation_schema_exists(db) -> bool:
    count = db.conn.execute(
        "SELECT COUNT(*) FROM duckdb_tables() "
        "WHERE table_name='price_generation_state'"
    ).fetchone()[0]
    return count > 0


def canonical_price_hash(
    ticker: str, prices: pd.Series, basis: str, generation: str
) -> str:
    """Hash date/close/basis/generation tuples in a canonical order."""
    digest = hashlib.sha256()
    for day, close in prices.dropna().sort_index().items():
        digest.update(
            f"{ticker}\t{pd.Timestamp(day).date().isoformat()}\t"
            f"{float(close).hex()}\t{basis}\t{generation}\n".encode()
        )
    return digest.hexdigest()


def legacy_price_generation_state(ticker: str, prices: pd.Series) -> dict:
    """Legacy (pre-metadata) state for one ticker from its loaded prices.

    This is the fallback half of :func:`price_generation_state`: when no state
    row exists, the served generation is the hash of the ticker's full price
    history under the legacy basis. ``prices`` must be the same rows the
    per-ticker helper would read (ticker's history, date-ordered).
    """
    ticker = str(ticker).strip().upper()
    if prices.empty:
        return {
            "generation_id": None, "comparability_status": "unknown",
            "alternate_generation": None,
            "reason": "price comparability unknown: generation metadata unavailable",
        }
    legacy_seed = canonical_price_hash(ticker, prices, "legacy-unverified", "legacy")
    return {
        "generation_id": f"legacy:{legacy_seed[:24]}",
        "comparability_status": "unknown",
        "alternate_generation": None,
        "reason": "price comparability unknown: persisted prices predate generation metadata",
    }


def price_generation_state(db, ticker: str, *, as_of=None) -> dict:
    """Return explicit served-generation metadata, including legacy uncertainty.

    ``as_of`` bounds the legacy fallback hash to the decision's point-in-time
    price history: later market data must never rewrite a historical
    candidate's generation id (determinism under post-cutoff ingestion).
    """
    ticker = str(ticker).strip().upper()
    if price_generation_schema_exists(db):
        row = db.conn.execute(
            "SELECT served_generation, comparability, alternate_generation, reason "
            "FROM price_generation_state WHERE ticker=?", [ticker]
        ).fetchone()
        if isinstance(row, (tuple, list)) and len(row) == 4:
            return {
                "generation_id": row[0], "comparability_status": row[1],
                "alternate_generation": row[2], "reason": row[3],
            }
    if as_of is None:
        values = db.conn.execute(
            "SELECT date, close FROM prices WHERE ticker=? ORDER BY date", [ticker]
        ).fetchdf()
    else:
        values = db.conn.execute(
            "SELECT date, close FROM prices WHERE ticker=? AND date <= ? "
            "ORDER BY date",
            [ticker, pd.Timestamp(as_of).date()],
        ).fetchdf()
    series = (
        pd.Series(values.close.to_numpy(), index=pd.to_datetime(values.date))
        if not values.empty
        else pd.Series(dtype=float)
    )
    return legacy_price_generation_state(ticker, series)


def price_generation_states(db, *, schema_exists: bool | None = None) -> dict[str, dict]:
    """Every persisted served-generation state, keyed by exact stored ticker.

    Batched :func:`price_generation_state`: one schema check and one query for
    the whole batch instead of a per-ticker round trip. Tickers without a state
    row are absent — callers pair this with :func:`legacy_price_generation_state`
    for exactly the fallback the per-ticker helper performs. Normalize lookup
    arguments, not stored keys: distinct SQL identities must remain distinct.
    """
    if schema_exists is None:
        schema_exists = price_generation_schema_exists(db)
    if not schema_exists:
        return {}
    rows = db.conn.execute(
        "SELECT ticker, served_generation, comparability, "
        "alternate_generation, reason FROM price_generation_state"
    ).fetchall()
    return {
        ticker: {
            "generation_id": served, "comparability_status": comparability,
            "alternate_generation": alternate, "reason": reason,
        }
        for ticker, served, comparability, alternate, reason in rows
    }


def price_generation_basis(db, ticker: str, generation: str | None) -> str:
    if generation and price_generation_schema_exists(db):
        row = db.conn.execute(
            "SELECT basis FROM price_generations WHERE ticker=? AND generation=?",
            [str(ticker).strip().upper(), generation],
        ).fetchone()
        if isinstance(row, (tuple, list)) and row and row[0]:
            return str(row[0])
    return "legacy-unverified"


def price_generation_basis_map(
    db, *, schema_exists: bool | None = None
) -> dict[tuple[str, str], str]:
    """Every stored generation basis, keyed ``(ticker, generation)``.

    Batched :func:`price_generation_basis`: one query for the whole batch.
    Pairs absent from the map resolve to ``legacy-unverified``, which is
    exactly the per-ticker helper's fallback for a missing header row.
    """
    if schema_exists is None:
        schema_exists = price_generation_schema_exists(db)
    if not schema_exists:
        return {}
    rows = db.conn.execute(
        "SELECT ticker, generation, basis FROM price_generations"
    ).fetchall()
    return {
        (ticker, generation): str(basis)
        for ticker, generation, basis in rows
        if basis
    }


def price_adjustment_factor(db, ticker: str, day, generation: str | None = None) -> float | None:
    """Find the latest known adjusted/raw factor on or before a filing date."""
    if not price_generation_schema_exists(db):
        return None
    if generation is None:
        generation = price_generation_state(db, ticker).get("generation_id")
    if not generation:
        return None
    row = db.conn.execute(
        "SELECT adjustment_factor FROM price_generation_values "
        "WHERE ticker=? AND generation=? AND date<=? "
        "AND adjustment_factor IS NOT NULL ORDER BY date DESC LIMIT 1",
        [str(ticker).strip().upper(), generation, pd.Timestamp(day).date()],
    ).fetchone()
    if not row:
        return None
    factor = float(row[0])
    return factor if math.isfinite(factor) and factor > 0 else None


def _generation_id(ticker: str, prices: pd.Series, factors: pd.Series, basis: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{ticker}\t{basis}\n".encode())
    for day, close in prices.dropna().sort_index().items():
        factor = factors.get(day, np.nan)
        factor_text = float(factor).hex() if pd.notna(factor) else "unknown"
        digest.update(
            f"{pd.Timestamp(day).date().isoformat()}\t{float(close).hex()}\t"
            f"{factor_text}\n".encode()
        )
    return f"yahoo:{digest.hexdigest()[:24]}"


def _store_generation(db, ticker, generation, series, factors, basis) -> str:
    ensure_price_generation_schema(db)
    value_hash = canonical_price_hash(ticker, series, basis, generation)
    db.conn.execute(
        "INSERT INTO price_generations(ticker,generation,basis,value_hash) "
        "VALUES (?,?,?,?) ON CONFLICT(ticker,generation) DO UPDATE "
        "SET value_hash=excluded.value_hash,basis=excluded.basis",
        [ticker, generation, basis, value_hash],
    )
    rows = []
    for day, close in series.dropna().sort_index().items():
        factor = factors.get(day, np.nan)
        rows.append((ticker, generation, pd.Timestamp(day).date(), float(close),
                     float(factor) if pd.notna(factor) else None, basis))
    if rows:
        # One set-based insert avoids per-row autocommits on disk-backed stores.
        db.conn.execute(
            "INSERT INTO price_generation_values "
            "SELECT unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), "
            "unnest(?::DATE[]), unnest(?::DOUBLE[]), "
            "unnest(?::DOUBLE[]), unnest(?::VARCHAR[]) "
            "ON CONFLICT(ticker,generation,date) DO UPDATE SET "
            "close=excluded.close, adjustment_factor=excluded.adjustment_factor, "
            "basis=excluded.basis", [list(column) for column in zip(*rows)],
        )
    return value_hash


def store_generations(db, entries) -> list[str]:
    """Batched :func:`_store_generation` for many (ticker, generation) writes.

    Persists exactly the rows sequential per-ticker calls would: one schema
    ensure, one batch of header upserts and one set-based values insert. When
    several entries address the same (ticker, generation) identity the later
    entry wins, which is what repeating the per-ticker call would leave in the
    tables. This low-level helper does not manage transactions; callers own
    the transaction around a coordinated refresh write.
    """
    entries = list(entries)
    if not entries:
        return []
    headers: dict = {}
    values: dict = {}
    for ticker, generation, series, factors, basis in entries:
        value_hash = canonical_price_hash(ticker, series, basis, generation)
        headers[(str(ticker), str(generation))] = (
            ticker, generation, basis, value_hash
        )
        for day, close in series.dropna().sort_index().items():
            factor = factors.get(day, np.nan)
            values[(str(ticker), str(generation), pd.Timestamp(day).date())] = (
                ticker, generation, pd.Timestamp(day).date(), float(close),
                float(factor) if pd.notna(factor) else None, basis,
            )
    ensure_price_generation_schema(db)
    if headers:
        db.conn.executemany(
            "INSERT INTO price_generations(ticker,generation,basis,value_hash) "
            "VALUES (?,?,?,?) ON CONFLICT(ticker,generation) DO UPDATE "
            "SET value_hash=excluded.value_hash,basis=excluded.basis",
            list(headers.values()),
        )
    if values:
        db.conn.execute(
            "INSERT INTO price_generation_values "
            "SELECT unnest(?::VARCHAR[]), unnest(?::VARCHAR[]), "
            "unnest(?::DATE[]), unnest(?::DOUBLE[]), "
            "unnest(?::DOUBLE[]), unnest(?::VARCHAR[]) "
            "ON CONFLICT(ticker,generation,date) DO UPDATE SET "
            "close=excluded.close, adjustment_factor=excluded.adjustment_factor, "
            "basis=excluded.basis", [list(column) for column in zip(*values.values())],
        )
    return [header[3] for header in headers.values()]


def _set_generation_state(db, ticker, generation, status, alternate=None, reason=None):
    db.conn.execute(
        "INSERT INTO price_generation_state VALUES (?,?,?,?,?) "
        "ON CONFLICT(ticker) DO UPDATE SET served_generation=excluded.served_generation, "
        "comparability=excluded.comparability, "
        "alternate_generation=excluded.alternate_generation, reason=excluded.reason",
        [ticker, generation, status, alternate, reason],
    )


def certify_adjusted_price_generations(
    generations: dict[str, pd.DataFrame], dividends: pd.DataFrame
) -> dict:
    """Certify independently fetched adjusted series against each other/actions.

    A price generation is scale-free for return labels: multiplying every close
    in a generation by one constant changes no return. Fetch comparisons
    therefore normalize on their latest shared session and compare all shared
    total-return ratios. Two shared dates are the mathematical minimum needed
    to compare one return interval.

    For dividend event d, Yahoo's adjustment is
    ``factor[t] = prod(1 - cash_d / raw_close_before_ex_date_d)`` for d > t.
    The prior session close is used because the cash distribution is removed
    from the pre-ex-date share price. ``raw = adjusted_close / adjustment_factor``
    recovers the raw close associated with that generation. A non-dividend
    factor step is accepted only when it is a rational split ratio and it
    cancels a raw split-like move without leaving a split-sized adjusted move.
    """
    if len(generations) < 2:
        return {
            "certified": False,
            "generation_id": max(generations, default=None),
            "reason": f"only {len(generations)} independent adjusted fetches; at least 2 required",
            "stats": {"fetch_count": len(generations)},
        }

    normalized: dict[str, pd.DataFrame] = {}
    for generation, source in generations.items():
        if not {"date", "close", "adjustment_factor"}.issubset(source.columns):
            return {
                "certified": False, "generation_id": None,
                "reason": f"{generation}: missing date/close/adjustment_factor evidence",
                "stats": {"fetch_count": len(generations)},
            }
        frame = source[["date", "close", "adjustment_factor"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["adjustment_factor"] = pd.to_numeric(
            frame["adjustment_factor"], errors="coerce"
        )
        frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
        if (
            frame.empty
            or frame.index.has_duplicates
            or not np.isfinite(frame[["close", "adjustment_factor"]].to_numpy()).all()
            or (frame[["close", "adjustment_factor"]] <= 0).any().any()
        ):
            return {
                "certified": False, "generation_id": None,
                "reason": f"{generation}: duplicate dates or invalid close/adjustment factor",
                "stats": {"fetch_count": len(generations)},
            }
        normalized[str(generation)] = frame

    selected = max(
        normalized,
        key=lambda generation: (
            normalized[generation].index[-1],
            len(normalized[generation]),
            generation,
        ),
    )
    pair_overlaps: list[int] = []
    max_fetch_error = 0.0
    max_factor_fetch_error = 0.0
    for index, generation_a in enumerate(sorted(normalized)):
        for generation_b in sorted(normalized)[index + 1:]:
            a, b = normalized[generation_a], normalized[generation_b]
            overlap = a.index.intersection(b.index)
            pair_overlaps.append(len(overlap))
            if len(overlap) < 2:
                return {
                    "certified": False, "generation_id": selected,
                    "reason": (
                        f"independent fetches {generation_a} and {generation_b} share "
                        f"{len(overlap)} dates; at least 2 required to compare returns"
                    ),
                    "stats": {"fetch_count": len(normalized), "min_pair_overlap": min(pair_overlaps)},
                }
            a_ratio = a.loc[overlap, "close"] / a.loc[overlap[-1], "close"]
            b_ratio = b.loc[overlap, "close"] / b.loc[overlap[-1], "close"]
            error = float(((a_ratio / b_ratio) - 1.0).abs().max())
            max_fetch_error = max(max_fetch_error, error)
            if error > _PRICE_CERTIFICATION_REL_TOL:
                return {
                    "certified": False, "generation_id": selected,
                    "reason": (
                        f"independent fetches disagree: {generation_a}/{generation_b} "
                        f"max normalized relative difference {error:.8g} exceeds "
                        f"{_PRICE_CERTIFICATION_REL_TOL:.1g}"
                    ),
                    "stats": {
                        "fetch_count": len(normalized),
                        "min_pair_overlap": min(pair_overlaps),
                        "max_normalized_fetch_error": max_fetch_error,
                        "max_normalized_factor_error": max_factor_fetch_error,
                    },
                }
            a_factor = (
                a.loc[overlap, "adjustment_factor"]
                / a.loc[overlap[-1], "adjustment_factor"]
            )
            b_factor = (
                b.loc[overlap, "adjustment_factor"]
                / b.loc[overlap[-1], "adjustment_factor"]
            )
            factor_error = float(((a_factor / b_factor) - 1.0).abs().max())
            max_factor_fetch_error = max(max_factor_fetch_error, factor_error)
            if factor_error > _PRICE_FACTOR_CERTIFICATION_REL_TOL:
                return {
                    "certified": False, "generation_id": selected,
                    "reason": (
                        f"independent adjustment factors disagree: {generation_a}/{generation_b} "
                        f"max normalized relative difference {factor_error:.8g} exceeds "
                        f"{_PRICE_FACTOR_CERTIFICATION_REL_TOL:.1g}"
                    ),
                    "stats": {
                        "fetch_count": len(normalized),
                        "min_pair_overlap": min(pair_overlaps),
                        "max_normalized_fetch_error": max_fetch_error,
                        "max_normalized_factor_error": max_factor_fetch_error,
                    },
                }

    actions = dividends.copy()
    if not {"ex_date", "cash_amount"}.issubset(actions.columns):
        return {
            "certified": False, "generation_id": selected,
            "reason": "dividend evidence is missing ex_date/cash_amount",
            "stats": {"fetch_count": len(normalized), "min_pair_overlap": min(pair_overlaps)},
        }
    actions["ex_date"] = pd.to_datetime(actions["ex_date"], errors="coerce").dt.normalize()
    actions["cash_amount"] = pd.to_numeric(actions["cash_amount"], errors="coerce")
    if actions["ex_date"].duplicated().any():
        return {
            "certified": False, "generation_id": selected,
            "reason": "dividend evidence has duplicate ex-dates",
            "stats": {"fetch_count": len(normalized), "min_pair_overlap": min(pair_overlaps)},
        }
    max_dividend_error = 0.0
    explained_splits = 0
    for generation, frame in normalized.items():
        dates = frame.index
        close = frame["close"].to_numpy(dtype="float64")
        factors = frame["adjustment_factor"].to_numpy(dtype="float64")
        raw = close / factors
        in_range = actions.loc[
            actions["ex_date"].between(dates[0], dates[-1], inclusive="both")
        ].sort_values("ex_date")
        expected_dividend_step = np.ones(len(frame), dtype="float64")
        for action in in_range.itertuples(index=False):
            position = dates.get_indexer([action.ex_date])[0]
            if position <= 0:
                return {
                    "certified": False, "generation_id": selected,
                    "reason": f"{generation}: dividend {action.ex_date.date()} lacks ex-date/prior-session prices",
                    "stats": {
                        "fetch_count": len(normalized),
                        "min_pair_overlap": min(pair_overlaps),
                        "max_normalized_fetch_error": max_fetch_error,
                        "max_normalized_factor_error": max_factor_fetch_error,
                    },
                }
            amount = float(action.cash_amount)
            event_factor = 1.0 - amount / raw[position - 1]
            if not math.isfinite(event_factor) or event_factor <= 0:
                return {
                    "certified": False, "generation_id": selected,
                    "reason": f"{generation}: invalid dividend reconstruction on {action.ex_date.date()}",
                    "stats": {"fetch_count": len(normalized), "min_pair_overlap": min(pair_overlaps)},
                }
            # Adj Close's factor rises on the ex-date by the inverse of the
            # dividend haircut applied to the previous session's raw close.
            expected_dividend_step[position] = 1.0 / event_factor

        # Verify each closed-session transition locally. This avoids summing
        # tiny cash-amount/quote-rounding differences over years: on a dividend
        # date compare factor[d]/factor[prev] with 1/(1-D/raw_prev); otherwise
        # the factor ratio must be 1 unless a split is independently explained.
        factor_changes = factors[1:] / factors[:-1]
        raw_changes = raw[1:] / raw[:-1]
        close_changes = close[1:] / close[:-1]
        split_positions: set[int] = set()
        for offset, observed_factor_change in enumerate(factor_changes):
            transition = offset + 1
            component_change = (
                observed_factor_change / expected_dividend_step[transition]
            )
            local_error = abs(float(component_change) - 1.0)
            if local_error <= _PRICE_FACTOR_CERTIFICATION_REL_TOL:
                max_dividend_error = max(max_dividend_error, local_error)
                continue
            split_ratio = Fraction(float(component_change)).limit_denominator(100)
            rational_ratio = float(split_ratio)
            if (
                split_ratio.numerator == split_ratio.denominator
                or abs(rational_ratio / component_change - 1.0)
                > _PRICE_FACTOR_CERTIFICATION_REL_TOL
                or not 0.5 <= raw_changes[offset] * component_change <= 2.0
                or not 0.5 <= close_changes[offset] <= 2.0
            ):
                return {
                    "certified": False, "generation_id": selected,
                    "reason": f"{generation}: unexplained adjustment-factor step on {dates[transition].date()}",
                    "stats": {
                        "fetch_count": len(normalized),
                        "min_pair_overlap": min(pair_overlaps),
                        "max_normalized_fetch_error": max_fetch_error,
                        "max_normalized_factor_error": max_factor_fetch_error,
                    },
                }
            explained_splits += 1
            split_positions.add(transition)

        # A raw split-sized jump without a corresponding accepted factor step
        # is not explained by the served total-return series.
        for offset, raw_change in enumerate(raw_changes):
            if (raw_change <= 0.5 or raw_change >= 2.0) and (offset + 1) not in split_positions:
                return {
                    "certified": False, "generation_id": selected,
                    "reason": f"{generation}: raw split-like jump lacks factor evidence on {dates[offset + 1].date()}",
                    "stats": {
                        "fetch_count": len(normalized),
                        "min_pair_overlap": min(pair_overlaps),
                        "max_normalized_fetch_error": max_fetch_error,
                        "max_normalized_factor_error": max_factor_fetch_error,
                        "max_dividend_reconstruction_error": max_dividend_error,
                    },
                }

    stats = {
        "fetch_count": len(normalized),
        "min_pair_overlap": min(pair_overlaps),
        "max_normalized_fetch_error": max_fetch_error,
        "max_normalized_factor_error": max_factor_fetch_error,
        "max_dividend_reconstruction_error": max_dividend_error,
        "explained_split_steps": explained_splits,
    }
    reason = (
        "certified yfinance-adjusted-close-v1: "
        f"{stats['fetch_count']} independent fetches, min shared dates "
        f"{stats['min_pair_overlap']}, max normalized fetch error "
        f"{max_fetch_error:.8g}, factor fetch error {max_factor_fetch_error:.8g}, "
        f"local dividend residual {max_dividend_error:.8g}, "
        f"explained split steps {explained_splits}"
    )
    return {"certified": True, "generation_id": selected, "reason": reason, "stats": stats}


def publish_price_generation_certification(db, ticker: str, certification: dict, old_state: dict):
    """Publish certification via the same persisted generation-state owner."""
    if certification.get("certified"):
        generation = certification["generation_id"]
        _set_generation_state(
            db, ticker, generation, "comparable", None, certification["reason"]
        )
        return {
            "generation_id": generation,
            "comparability_status": "comparable",
            "alternate_generation": None,
            "reason": certification["reason"],
        }
    served = old_state.get("generation_id")
    alternate = certification.get("generation_id")
    reason = f"price comparability unknown: adjusted-generation certification failed: {certification.get('reason', 'no evidence')}"
    _set_generation_state(db, ticker, served, "unknown", alternate, reason)
    return {
        "generation_id": served,
        "comparability_status": "unknown",
        "alternate_generation": alternate,
        "reason": reason,
    }


def set_generation_states(db, rows) -> None:
    """Batched :func:`_set_generation_state`: one prepared statement batch.

    This low-level helper does not manage transactions; callers own the
    transaction around a coordinated refresh write.
    """
    rows = list(rows)
    if not rows:
        return
    db.conn.executemany(
        "INSERT INTO price_generation_state VALUES (?,?,?,?,?) "
        "ON CONFLICT(ticker) DO UPDATE SET served_generation=excluded.served_generation, "
        "comparability=excluded.comparability, "
        "alternate_generation=excluded.alternate_generation, reason=excluded.reason",
        rows,
    )


def _all_ticker_values(db, ticker):
    rows = db.conn.execute(
        "SELECT date, close FROM prices WHERE ticker=? ORDER BY date", [ticker]
    ).fetchdf()
    if rows.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    series = pd.Series(rows.close.to_numpy(dtype=float), index=pd.to_datetime(rows.date))
    factors = pd.Series(np.nan, index=series.index, dtype=float)
    if price_generation_schema_exists(db):
        state = price_generation_state(db, ticker)
        generation = state.get("generation_id")
        if generation:
            known = db.conn.execute(
                "SELECT date, adjustment_factor FROM price_generation_values "
                "WHERE ticker=? AND generation=?", [ticker, generation]
            ).fetchdf()
            if not known.empty:
                known.index = pd.to_datetime(known.date)
                factors = pd.to_numeric(known.adjustment_factor, errors="coerce").reindex(series.index)
    return series, factors


def _all_ticker_values_batch(db, tickers, states, *, schema_exists=None):
    """Batched :func:`_all_ticker_values` for many tickers in O(1) queries.

    One history query and one batched factor read replace the per-ticker
    round trips (schema check, state lookup, history reload, factor reload).
    ``states`` is a pre-read :func:`price_generation_states` mapping keyed by
    stored ticker; lookup arguments are normalized like the SQL reader.
    Tickers absent from it fall back to
    :func:`legacy_price_generation_state` over exactly the table rows the
    per-ticker helper would re-read, so every generation id is unchanged.
    """
    tickers = [str(ticker) for ticker in tickers]
    result = {
        ticker: (pd.Series(dtype=float), pd.Series(dtype=float))
        for ticker in tickers
    }
    if not result:
        return result
    if schema_exists is None:
        schema_exists = price_generation_schema_exists(db)
    rows = db.conn.execute(
        "SELECT ticker, date, close FROM prices "
        "WHERE ticker IN (SELECT UNNEST(?)) ORDER BY ticker, date",
        [tickers],
    ).fetchdf()
    series_map = {}
    if not rows.empty:
        for ticker, frame in rows.groupby("ticker", sort=False):
            series_map[str(ticker)] = pd.Series(
                frame.close.to_numpy(dtype=float), index=pd.to_datetime(frame.date)
            )
    wanted_factors = {}
    for ticker in tickers:
        series = series_map.get(ticker, pd.Series(dtype=float))
        state = states.get(str(ticker).strip().upper()) or legacy_price_generation_state(ticker, series)
        result[ticker] = (series, pd.Series(np.nan, index=series.index, dtype=float))
        generation = state.get("generation_id")
        if generation:
            wanted_factors[ticker] = generation
    known = price_generation_factors(db, wanted_factors, schema_exists=schema_exists)
    for ticker in tickers:
        raw = known.get(ticker)
        if raw is None or raw.empty:
            continue
        series, _ = result[ticker]
        result[ticker] = (
            series,
            pd.to_numeric(raw, errors="coerce").reindex(series.index),
        )
    return result


def price_generation_factors(
    db, wanted: dict[str, str], *, schema_exists: bool | None = None
) -> dict[str, pd.Series]:
    """Numeric adjustment factors for the given ``(ticker, generation)`` pairs.

    Batched replacement for the per-ticker reload inside
    :func:`_all_ticker_values`: one schema check and one query for the whole
    batch. Returned series are indexed by date but deliberately NOT reindexed
    onto a price index, so the caller can align them onto whichever history it
    holds (pre- or post-insert). Non-numeric legacy values coerce to NaN, as in
    the per-ticker reader. Tickers with no wanted generation or no stored
    rows get an empty series — reindexing or ``.get`` on it yields the same
    all-NaN factors the per-ticker read produces.
    """
    factors = {str(ticker): pd.Series(dtype=float) for ticker in wanted}
    if schema_exists is None:
        schema_exists = price_generation_schema_exists(db)
    if not wanted or not schema_exists:
        return factors
    con = db.conn
    con.execute(
        "CREATE OR REPLACE TEMP TABLE _price_gen_want "
        "(ticker VARCHAR, generation VARCHAR)"
    )
    con.executemany(
        "INSERT INTO _price_gen_want VALUES (?,?)",
        sorted((str(ticker), str(generation)) for ticker, generation in wanted.items()),
    )
    rows = con.execute(
        "SELECT v.ticker, v.date, v.adjustment_factor "
        "FROM price_generation_values v "
        "JOIN _price_gen_want w ON w.ticker = v.ticker AND w.generation = v.generation"
    ).fetchdf()
    if rows.empty:
        return factors
    for ticker, frame in rows.groupby("ticker", sort=False):
        factors[str(ticker)] = pd.Series(
            pd.to_numeric(frame.adjustment_factor, errors="coerce").to_numpy(),
            index=pd.to_datetime(frame.date),
            name="adjustment_factor",
        )
    return factors


def _merge_price_generation(db, new_prices, new_factors, basis, *, persist, cached_prices):
    """Merge only numerically compatible series; divergent values stay versioned.

    Canonical evidence is gathered in a constant number of batched reads (one
    schema check, one state read, one history read, one factor read) before
    any decision; the per-ticker comparison logic and every persisted row are
    unchanged relative to the per-ticker reads this replaces. Writes retain
    their original per-ticker order, including partial persistence on failure
    when the caller does not own a transaction.
    """
    output = (
        cached_prices.copy()
        if not cached_prices.empty
        else pd.DataFrame(index=new_prices.index)
    )
    states = {}
    incoming_map = {}
    incoming_factor_map = {}
    for ticker in new_prices.columns:
        incoming = new_prices[ticker].dropna().astype(float)
        if incoming.empty:
            continue
        incoming_map[ticker] = incoming
        incoming_factor_map[ticker] = (
            new_factors[ticker].reindex(incoming.index) if ticker in new_factors else pd.Series(np.nan, index=incoming.index)
        )
    # Batched evidence: O(1) queries for the whole batch instead of O(T)
    # round trips through _all_ticker_values and price_generation_state.
    schema_ready = price_generation_schema_exists(db)
    state_map = price_generation_states(db, schema_exists=schema_ready)
    old_values = _all_ticker_values_batch(
        db, incoming_map.keys(), state_map, schema_exists=schema_ready
    )
    basis_map = None
    for ticker in incoming_map:
        incoming = incoming_map[ticker]
        factors = incoming_factor_map[ticker]
        candidate_generation = _generation_id(ticker, incoming, factors, basis)
        table_series, old_factors = old_values.get(
            ticker, (pd.Series(dtype=float), pd.Series(dtype=float))
        )
        old_all = table_series
        old = old_all
        if old.empty and ticker in cached_prices:
            old = cached_prices[ticker].dropna().astype(float)
            old_all = old
            old_factors = pd.Series(np.nan, index=old.index, dtype=float)
        overlap = old.index.intersection(incoming.index)
        mismatch = False
        max_relative = 0.0
        rel = None
        if len(overlap):
            denom = incoming.reindex(overlap).abs().replace(0, np.nan)
            rel = (old.reindex(overlap) - incoming.reindex(overlap)).abs() / denom
            max_relative = float(rel.max()) if rel.notna().any() else 0.0
            mismatch = bool((rel > _PRICE_OVERLAP_TOL).any())

        if mismatch or (not old.empty and not len(overlap)):
            reason = (
                f"price comparability unknown: {int((rel > _PRICE_OVERLAP_TOL).sum())} "
                f"overlaps exceed {_PRICE_OVERLAP_TOL:.1%} (max {max_relative:.5%})"
                if mismatch else "price comparability unknown: no overlapping prices"
            )
            # The batched state snapshot equals the per-ticker re-read this
            # branch used to issue: nothing has written this ticker's state or
            # price history between the batch read and here.
            served = (
                state_map.get(str(ticker).strip().upper())
                or legacy_price_generation_state(ticker, table_series)
            ).get("generation_id")
            if persist:
                if not old_all.empty and served:
                    if basis_map is None:
                        basis_map = price_generation_basis_map(
                            db, schema_exists=schema_ready
                        )
                    old_basis = basis_map.get(
                        (str(ticker).strip().upper(), str(served)),
                        "legacy-unverified",
                    )
                    _store_generation(db, ticker, served, old_all, old_factors, old_basis)
                    _store_generation(db, ticker, candidate_generation, incoming, factors, basis)
                    _set_generation_state(db, ticker, served, "unknown", candidate_generation, reason)
            states[ticker] = {
                "generation_id": served, "comparability_status": "unknown",
                "alternate_generation": candidate_generation, "reason": reason,
                "overlap_count": len(overlap), "max_relative_divergence": max_relative,
            }
            continue

        merged = old_all.combine_first(incoming)
        merged.update(incoming)
        merged_factors = old_factors.reindex(merged.index)
        merged_factors.update(factors)
        generation = _generation_id(ticker, merged, merged_factors, basis)
        if persist:
            db.upsert_prices(new_prices[[ticker]])
            _store_generation(db, ticker, generation, merged, merged_factors, basis)
            _set_generation_state(db, ticker, generation, "comparable")
        if ticker not in output.columns:
            output[ticker] = np.nan
        # Incoming dates may extend the existing index (first merge for this
        # ticker into a cached frame): grow the frame before .loc assignment.
        output = output.reindex(output.index.union(incoming.index))
        output.loc[incoming.index, ticker] = incoming
        states[ticker] = {
            "generation_id": generation, "comparability_status": "comparable",
            "alternate_generation": None, "reason": None,
            "overlap_count": len(overlap), "max_relative_divergence": max_relative,
        }
    return output.sort_index(), states


# ── Helpers ──


def _clean_tickers(tickers: list[str]) -> list[str]:
    """Normalize symbols and filter empty values and parser-token quarantines."""
    resolver = TickerResolver()
    clean: list[str] = []
    quarantined: list[str] = []
    for ticker in tickers:
        if isinstance(ticker, bool):
            continue
        normalized = str(ticker).strip().upper() if ticker is not None else ""
        if not normalized or normalized == "NAN":
            continue
        if resolver.resolve(normalized).status == "quarantined":
            quarantined.append(normalized)
            continue
        clean.append(normalized)
    if quarantined:
        logger.debug(
            "Excluded %d quarantined ticker tokens from price fetch: %s",
            len(quarantined),
            ", ".join(sorted(set(quarantined))),
        )
    return clean


def _clip_price_window(
    prices: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """Keep fetched observations inside the inclusive requested window."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts.tz is not None:
        start_ts = start_ts.tz_localize(None)
    if end_ts.tz is not None:
        end_ts = end_ts.tz_localize(None)
    return prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]


def _expand_rename_aliases(tickers: list[str]) -> set[str]:
    """Expand rename aliases into both temporal price symbols.

    A price acquisition window spans many disclosure dates, so a rename alias
    can require both its pre-rename and post-rename market symbols. Fetch both
    series and let per-row entry-price resolution choose the symbol tradable at
    public disclosure time.
    """
    resolver = TickerResolver()
    expanded: set[str] = set()
    for t in tickers:
        normalized = str(t).strip().upper()
        if normalized in resolver.RENAME_MAP:
            new_symbol, _ = resolver.RENAME_MAP[normalized]
            expanded.add(t)
            expanded.add(new_symbol)
        else:
            expanded.add(t)
    return expanded


def _resolve_tickers(all_tickers: list[str]) -> tuple[dict, dict]:
    """Build raw-market-symbol -> provider-symbol mappings.

    Rename aliases use the current provider symbol because Yahoo no longer
    serves historical data under symbols such as FB or SQ. This is a provider
    addressing choice only; TickerResolver still controls the tradable symbol
    at each disclosure date.
    """
    resolver = TickerResolver()
    resolutions = resolver.resolve_batch(all_tickers)
    raw_to_yf: dict[str, str] = {}
    for raw, resolution in resolutions.items():
        normalized = str(raw).strip().upper()
        if normalized in resolver.RENAME_MAP:
            raw_to_yf[raw] = resolver.RENAME_MAP[normalized][0]
        else:
            raw_to_yf[raw] = resolution.price_symbol
    yf_to_raw: dict[str, str] = {}
    for raw, sym in raw_to_yf.items():
        if sym not in yf_to_raw:
            yf_to_raw[sym] = raw
    status_counts = Counter(r.status for r in resolutions.values())
    logger.debug("Ticker resolution summary: %s", dict(sorted(status_counts.items())))
    for resolution in resolutions.values():
        if resolution.status not in {"unverified", "class_share"}:
            logger.debug("Ticker resolution: %s", resolution.notes)
    return raw_to_yf, yf_to_raw


def _validate_and_log_prices(
    prices: pd.DataFrame, all_tickers: list[str]
) -> pd.DataFrame:
    """Quarantine invalid observations and report unresolved tickers."""
    prices = _as_frame(prices.apply(pd.to_numeric, errors="coerce"))
    invalid_mask = prices.notna() & (~np.isfinite(prices) | prices.le(0))
    if invalid_mask.any().any():
        logger.warning(
            "Quarantined %d invalid cached price observations",
            invalid_mask.sum().sum(),
        )
        prices = prices.mask(invalid_mask)
    prices = prices.dropna(axis=1, how="all")

    failed_tickers = sorted(set(all_tickers) - set(prices.columns))
    success_count = len([t for t in all_tickers if t in prices.columns])
    success_rate = success_count / len(all_tickers)

    if failed_tickers:
        logger.warning(
            f"Failed to fetch price data for {len(failed_tickers)} tickers: "
            f"{', '.join(failed_tickers[:10])}"
            f"{'...' if len(failed_tickers) > 10 else ''}"
        )

    logger.info(
        f"Successfully fetched prices for {success_count}/{len(all_tickers)} "
        f"tickers ({success_rate * 100:.1f}% success)"
    )
    return _select_columns(prices, all_tickers)


def _select_columns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Column-select in ticker order and drop all-empty columns."""
    available = [t for t in tickers if t in prices.columns]
    return _as_frame(prices[available]).dropna(axis=1, how="all")


def _as_frame(value: object) -> pd.DataFrame:
    """Narrow pandas ``apply`` results that stubs type as a union."""
    if not isinstance(value, pd.DataFrame):
        raise DataSourceError("unexpected price-table shape")
    return value


def fetch_dividend_events(
    tickers: list[str], *, sleep_s: float = 0.5
) -> pd.DataFrame:
    """Cash dividend ex-dates per ticker from provider corporate actions.

    Full history per ticker (ex-dates years back still contaminate old cost
    bases at replay). One request per ticker, paced; per-ticker failures are
    swallowed and logged — a missing row means unknown, never clean, and the
    scoring gate only fires on known events. Never raises for fetch issues.
    """
    rows = []
    for ticker in sorted({str(t).strip().upper() for t in tickers if str(t).strip()}):
        try:
            actions = yf.Ticker(ticker).actions
            if actions is None or actions.empty or "Dividends" not in actions.columns:
                continue
            pays = actions.loc[actions["Dividends"].fillna(0) > 0, "Dividends"]
            for ex_date, cash in pays.items():
                day = pd.to_datetime(ex_date, errors="coerce")
                amount = float(cash)
                if pd.isna(day) or not math.isfinite(amount) or amount <= 0:
                    continue
                rows.append(
                    {"ticker": ticker, "ex_date": day.date(), "cash_amount": amount}
                )
        except Exception as exc:  # noqa: BLE001 - per-ticker fetch isolation
            logger.warning("Dividend fetch failed for %s: %s", ticker, exc)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return pd.DataFrame(rows, columns=["ticker", "ex_date", "cash_amount"])


def upsert_dividends(db: Database, events: pd.DataFrame) -> int:
    """Persist dividend events idempotently; returns rows present afterwards."""
    if events.empty:
        return 0
    clean = events.dropna(subset=["ticker", "ex_date"]).copy()
    clean["ticker"] = clean["ticker"].astype(str).str.strip().str.upper()
    clean["ex_date"] = pd.to_datetime(clean["ex_date"], errors="coerce").dt.date
    clean = clean.dropna(subset=["ex_date"]).drop_duplicates(["ticker", "ex_date"])
    if clean.empty:
        return 0
    db.conn.execute("CREATE TEMP TABLE _div_stage AS SELECT * FROM clean")
    try:
        db.conn.execute("""
            INSERT INTO dividends (ticker, ex_date, cash_amount)
            SELECT ticker, ex_date, cash_amount FROM _div_stage
            ON CONFLICT DO NOTHING
        """)
    finally:
        db.conn.execute("DROP TABLE IF EXISTS _div_stage")
    return db.conn.execute("SELECT COUNT(*) FROM dividends").fetchone()[0]
