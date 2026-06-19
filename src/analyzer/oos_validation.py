"""Out-of-sample validation: train 2022-2023, test 2024-2025.

Usage:
    uv run python -m analyzer.oos_validation
"""
from __future__ import annotations

import sys
sys.argv = ["ptr-alpha"]

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

from analyzer.database import Database
from analyzer import analysis
from analyzer.pipeline import BacktestParams


def run_backtest_split(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    entry_prices: pd.DataFrame,
    start_date: date,
    end_date: date,
    horizon: int,
    min_buyers: int = 2,
    top_n: int = 5,
    label: str = "",
) -> dict:
    """Run backtest for a date range and return summary metrics."""
    params = BacktestParams(
        start_date=start_date, end_date=end_date,
        horizon=horizon, lookback_days=60, training_lookback_days=365,
        min_buyers=min_buyers, top_n=top_n, threshold=5.0, frequency_days=30,
    )

    signals = analysis.calculate_signal_potential(entry_prices, prices, [horizon])
    as_of_dates = pd.date_range(start_date, end_date, freq="30D")

    all_results = []
    for as_of in as_of_dates:
        as_of_ts = pd.Timestamp(as_of)
        try:
            recs = analysis.backtest_recommendations(
                signals, all_tx, as_of_ts,
                horizon=horizon, lookback_days=60, min_buyers=min_buyers,
                top_n=top_n, threshold=5.0, prices_df=prices,
                training_lookback_days=365,
            )
            if recs.empty:
                continue
            ev = analysis.evaluate_backtest(recs, prices, as_of_ts, horizon)
            ev = ev.dropna(subset=["bt_return_pct"])
            ev.insert(0, "as_of_date", as_of_ts.date())
            all_results.append(ev)
        except Exception:
            continue

    if not all_results:
        return {"label": label, "N": 0, "alpha": 0, "slope": 0, "t": 0, "win%": 0, "r1": 0, "r5": 0}

    v = pd.concat(all_results, ignore_index=True).dropna(subset=["bt_alpha_pct"])
    ra = v.groupby("rank")["bt_alpha_pct"].mean()
    r1 = ra.get(1, 0)
    r5 = ra.get(5, 0)
    t = float(v["bt_alpha_pct"].mean() / (v["bt_alpha_pct"].std() / np.sqrt(len(v)))) if len(v) > 1 and v["bt_alpha_pct"].std() > 0 else 0

    return {
        "label": label,
        "N": len(v),
        "alpha": round(float(v["bt_alpha_pct"].mean()), 2),
        "r1": round(float(r1), 2),
        "r5": round(float(r5), 2),
        "slope": round(float(r1 - r5), 2),
        "t": round(t, 2),
        "win%": round(100 * (v["bt_alpha_pct"] > 0).mean(), 0),
    }


def main():
    db = Database(Path("data") / "congress.duckdb", read_only=False)
    tx_start = pd.Timestamp("2021-10-07")
    tx_end = pd.Timestamp("2025-06-30")
    all_tx = db.get_transactions_by_date_range(tx_start, tx_end)
    all_tickers = sorted(set(all_tx["ticker"].unique().tolist()) | {"SPY"})
    prices = db.get_prices(all_tickers, tx_start, pd.Timestamp("2025-06-30") + pd.Timedelta(days=130))
    entry_prices = db.get_entry_prices(all_tickers, tx_start, pd.Timestamp("2025-06-30") + pd.Timedelta(days=130))

    print("Out-of-Sample Validation")
    print("=" * 70)
    print(f"Data: {len(all_tx)} tx, {prices.shape[1]} tickers")
    print()

    for h in [60, 90]:
        # Full period
        full = run_backtest_split(
            all_tx, prices, entry_prices,
            date(2022, 1, 1), date(2025, 6, 30), h,
            label=f"Full 2022-2025 (h={h})",
        )

        # In-sample (train)
        is_result = run_backtest_split(
            all_tx, prices, entry_prices,
            date(2022, 1, 1), date(2023, 12, 31), h,
            label=f"In-sample 2022-2023 (h={h})",
        )

        # Out-of-sample (test)
        oos_result = run_backtest_split(
            all_tx, prices, entry_prices,
            date(2024, 1, 1), date(2025, 6, 30), h,
            label=f"Out-of-sample 2024-2025 (h={h})",
        )

        print(f"--- Horizon {h} ---")
        for r in [full, is_result, oos_result]:
            print(f"  {r['label']:35s} N={r['N']:3d} alpha={r['alpha']:+5.1f}% slope={r['slope']:+5.1f}% t={r['t']:+5.2f} win={r['win%']:.0f}%")

        # Decay ratio
        if is_result["N"] > 0 and oos_result["N"] > 0:
            decay = oos_result["alpha"] / is_result["alpha"] if is_result["alpha"] != 0 else 0
            print(f"  OOS/IS ratio: {decay:.2f}")
        print()

    db.conn.close()


if __name__ == "__main__":
    main()
