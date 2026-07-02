from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from analyzer import analysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MatchedControlResult:
    as_of_date: date
    treatment_ticker: str
    treatment_alpha: float
    control_tickers: list[str]
    control_alphas: list[float]
    control_mean_alpha: float
    excess_alpha: float  # treatment - control_mean
    n_controls: int


# ---------------------------------------------------------------------------
# Characteristic computation
# ---------------------------------------------------------------------------

def _compute_realized_volatility(
    prices_df: pd.DataFrame, ticker: str, as_of_date: pd.Timestamp, window: int = 20,
) -> float | None:
    """Realized 20-day volatility (std of daily returns, NOT annualized)."""
    if ticker not in prices_df.columns:
        return None
    series = prices_df[ticker].dropna()
    cutoff = pd.Timestamp(as_of_date)
    window_series = series[series.index <= cutoff].tail(window + 1)
    if len(window_series) < 10:
        return None
    daily_returns = window_series.pct_change().dropna()
    if len(daily_returns) < 5:
        return None
    return float(daily_returns.std())


def _compute_max_drawdown(
    prices_df: pd.DataFrame, ticker: str, as_of_date: pd.Timestamp, lookback: int = 60,
) -> float | None:
    """Max drawdown from all-time-high over the last `lookback` days (as a fraction)."""
    if ticker not in prices_df.columns:
        return None
    series = prices_df[ticker].dropna()
    cutoff = pd.Timestamp(as_of_date)
    window = series[series.index <= cutoff].tail(lookback + 1)
    if len(window) < 10:
        return None
    running_max = window.cummax()
    drawdowns = (window - running_max) / running_max
    return float(drawdowns.min())


def _load_sector_and_cap_data(
    tickers: list[str],
) -> dict[str, dict[str, object]]:
    """Fetch sector and market cap tiers from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return {}

    def fetch(ticker: str) -> tuple[str, dict[str, object]]:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            mcap = info.get("marketCap", 0)
            return ticker, {"sector": sector, "market_cap": mcap}
        except Exception:
            return ticker, {"sector": "Unknown", "market_cap": 0}

    results: dict[str, dict[str, object]] = {}
    filtered = [t for t in tickers if t not in ("SPY", "SP500")]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch, t): t for t in filtered}
        for future in as_completed(futures):
            ticker, data = future.result()
            results[ticker] = data
    return results


def _market_cap_tier(market_cap: float) -> str:
    if market_cap >= 200e9:
        return "mega"
    if market_cap >= 10e9:
        return "large"
    if market_cap >= 2e9:
        return "mid"
    return "small"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def find_matched_controls(
    ticker: str,
    as_of_date: date,
    all_tickers: list[str],
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    n_controls: int = 10,
    match_sector: bool = True,
    match_volatility: bool = True,
    match_drawdown: bool = True,
    sector_data: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    """Find control tickers matched on sector/volatility/drawdown characteristics."""
    as_of_ts = pd.Timestamp(as_of_date)

    candidates = [t for t in all_tickers if t != ticker and t in prices_df.columns]
    if not candidates:
        return []

    # Treatment characteristics
    treat_vol = _compute_realized_volatility(prices_df, ticker, as_of_ts) if match_volatility else None
    treat_dd = _compute_max_drawdown(prices_df, ticker, as_of_ts) if match_drawdown else None

    # Sector data
    if sector_data is None:
        needed = [ticker] + candidates
        sector_data = _load_sector_and_cap_data(needed)

    treat_sector = sector_data.get(ticker, {}).get("sector", "Unknown")
    treat_cap = _market_cap_tier(sector_data.get(ticker, {}).get("market_cap", 0))

    # Score each candidate
    scored: list[tuple[float, str]] = []
    for cand in candidates:
        cand_info = sector_data.get(cand, {})
        cand_sector = cand_info.get("sector", "Unknown")
        cand_cap = _market_cap_tier(cand_info.get("market_cap", 0))

        # Sector match: hard filter first, relaxed later
        sector_match = (cand_sector == treat_sector)
        if match_sector and not sector_match:
            # Allow adjacent (same cap tier) as fallback
            if cand_cap != treat_cap:
                continue

        # Cap tier: soft requirement
        if cand_cap != treat_cap:
            continue

        # Volatility match
        if match_volatility and treat_vol is not None:
            cand_vol = _compute_realized_volatility(prices_df, cand, as_of_ts)
            if cand_vol is None:
                continue
            if abs(cand_vol - treat_vol) > 0.5 * treat_vol:
                continue

        # Drawdown match
        if match_drawdown and treat_dd is not None:
            cand_dd = _compute_max_drawdown(prices_df, cand, as_of_ts)
            if cand_dd is None:
                continue
            if abs(cand_dd - treat_dd) > 0.10:
                continue

        # Distance score (lower = better match)
        score = 0.0
        if treat_vol is not None:
            cand_vol_val = _compute_realized_volatility(prices_df, cand, as_of_ts) or 0.0
            score += abs(cand_vol_val - treat_vol)
        if treat_dd is not None:
            cand_dd_val = _compute_max_drawdown(prices_df, cand, as_of_ts) or 0.0
            score += abs(cand_dd_val - treat_dd) * 10  # weight drawdown
        if not sector_match:
            score += 0.1  # penalty for sector mismatch

        scored.append((score, cand))

    scored.sort(key=lambda x: x[0])
    return [t for _, t in scored[:n_controls]]


# ---------------------------------------------------------------------------
# Alpha computation helpers
# ---------------------------------------------------------------------------

def _forward_alpha(
    prices_df: pd.DataFrame,
    ticker: str,
    as_of_date: pd.Timestamp,
    horizon: int,
    spy_start: float | None = None,
) -> float | None:
    """Compute forward alpha (return - SPY return) for a ticker."""
    from analyzer.analysis import _price_at_or_before, _price_on_or_before

    entry = _price_at_or_before(prices_df, ticker, as_of_date, max_staleness_days=30)
    exit_price = _price_on_or_before(
        prices_df, ticker, as_of_date + pd.Timedelta(days=horizon)
    )
    if not entry or not exit_price:
        return None

    return_pct = (exit_price / entry - 1) * 100

    if spy_start is None:
        spy_start = _price_at_or_before(prices_df, "SPY", as_of_date, max_staleness_days=30)
    spy_exit = _price_on_or_before(
        prices_df, "SPY", as_of_date + pd.Timedelta(days=horizon)
    )
    if spy_start and spy_exit:
        spy_return = (spy_exit / spy_start - 1) * 100
    else:
        spy_return = 0.0

    return return_pct - spy_return


# ---------------------------------------------------------------------------
# Matched-control backtest
# ---------------------------------------------------------------------------

def run_matched_control_backtest(
    signals_df: pd.DataFrame,
    all_tx: pd.DataFrame,
    prices_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    horizon: int = 120,
    top_n: int = 5,
    frequency_days: int = 14,
    n_controls: int = 10,
    min_buyers: int = 2,
    lookback_days: int = 60,
    threshold: float = 5.0,
    training_lookback_days: int = 365,
) -> pd.DataFrame:
    """Run backtest with matched controls for each recommendation.

    For each as-of date, get top-N congressional recommendations and find
    matched non-congressional controls. Compute excess alpha = treatment - mean(control).
    """
    as_of_dates = pd.date_range(start_date, end_date, freq=f"{frequency_days}D")

    all_tickers = prices_df.columns.tolist()

    # Pre-fetch sector/cap data for all tickers to avoid repeated yfinance calls
    logger.info("Pre-fetching sector and market cap data for all tickers")
    sector_data = _load_sector_and_cap_data(all_tickers)

    rows: list[dict] = []
    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)

        # Get recommendations using existing backtest logic
        recs = analysis.backtest_recommendations(
            signals_df, all_tx, as_of_ts,
            horizon=horizon,
            lookback_days=lookback_days,
            min_buyers=min_buyers,
            top_n=top_n,
            threshold=threshold,
            prices_df=prices_df,
            training_lookback_days=training_lookback_days,
        )

        if recs.empty:
            continue

        spy_start = analysis._price_at_or_before(prices_df, "SPY", as_of_ts, max_staleness_days=30)

        for _, rec in recs.iterrows():
            ticker = rec["ticker"]
            rank = rec.get("rank", 0)

            # Treatment alpha
            treat_alpha = _forward_alpha(prices_df, ticker, as_of_ts, horizon, spy_start)
            if treat_alpha is None:
                continue

            # Find matched controls
            controls = find_matched_controls(
                ticker, as_of_ts.date(), all_tickers, prices_df, signals_df,
                n_controls=n_controls, sector_data=sector_data,
            )

            # Compute control alphas
            control_alphas = []
            for ctrl in controls:
                alpha = _forward_alpha(prices_df, ctrl, as_of_ts, horizon, spy_start)
                if alpha is not None:
                    control_alphas.append(alpha)

            if not control_alphas:
                continue

            control_mean = float(np.mean(control_alphas))
            excess = treat_alpha - control_mean

            # Get characteristic values for the treatment ticker
            vol = _compute_realized_volatility(prices_df, ticker, as_of_ts)
            dd = _compute_max_drawdown(prices_df, ticker, as_of_ts)
            sector = sector_data.get(ticker, {}).get("sector", "Unknown")

            rows.append({
                "as_of_date": as_of_ts.date(),
                "ticker": ticker,
                "rank": rank,
                "alpha": round(treat_alpha, 2),
                "excess_alpha": round(excess, 2),
                "control_mean_alpha": round(control_mean, 2),
                "n_controls": len(control_alphas),
                "sector": sector,
                "volatility": round(vol, 4) if vol is not None else None,
                "drawdown": round(dd, 4) if dd is not None else None,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_matched_control_summary(results: pd.DataFrame) -> None:
    """Print summary statistics for matched-control backtest results."""
    if results.empty:
        print("\n=== Matched Control Backtest: No results ===")
        return

    valid = results.dropna(subset=["excess_alpha"])
    if valid.empty:
        print("\n=== Matched Control Backtest: No valid excess alpha results ===")
        return

    n = len(valid)
    mean_excess = valid["excess_alpha"].mean()
    std_excess = valid["excess_alpha"].std(ddof=1) if n > 1 else 0.0

    # Bootstrap CI for mean excess alpha
    rng = np.random.default_rng(42)
    n_boot = 5000
    boot_means = np.empty(n_boot)
    values = valid["excess_alpha"].values
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    pct_positive = (valid["excess_alpha"] > 0).mean() * 100

    print(f"\n{'=' * 60}")
    print("=== Matched-Control Backtest Summary ===")
    print(f"{'=' * 60}")
    print(f"N recommendations evaluated: {n}")
    print(f"Mean excess alpha:           {mean_excess:+.2f}%")
    print(f"95% CI (bootstrap):          [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")
    print(f"%% positive excess alpha:    {pct_positive:.1f}%")
    print(f"Mean raw treatment alpha:    {valid['alpha'].mean():+.2f}%")
    print(f"Mean control group alpha:    {valid['control_mean_alpha'].mean():+.2f}%")
    print(f"Median excess alpha:         {valid['excess_alpha'].median():+.2f}%")
    print(f"Std excess alpha:            {std_excess:.2f}%")

    # By sector
    if "sector" in valid.columns and valid["sector"].nunique() > 1:
        print("\n--- By Sector ---")
        sector_summary = valid.groupby("sector").agg(
            n=("excess_alpha", "size"),
            mean_excess=("excess_alpha", "mean"),
            pct_positive=("excess_alpha", lambda x: (x > 0).mean() * 100),
        ).sort_values("mean_excess", ascending=False)
        print(sector_summary.to_string())

    # By rank
    if "rank" in valid.columns and valid["rank"].nunique() > 1:
        print("\n--- By Rank ---")
        rank_summary = valid.groupby("rank").agg(
            n=("excess_alpha", "size"),
            mean_excess=("excess_alpha", "mean"),
            mean_alpha=("alpha", "mean"),
        ).sort_index()
        print(rank_summary.to_string())
