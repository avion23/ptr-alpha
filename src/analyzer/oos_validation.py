"""Out-of-sample validation: train 2022-2023, test 2024-2025.

Walk-forward + split OOS validation for congressional trading signal strategy.

Usage:
    uv run python -m analyzer.oos_validation
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer import analysis
from analyzer.database import Database


# Best config from parameter sweeps
BEST_CONFIG = dict(
    horizon=60,
    frequency_days=30,
    min_buyers=2,
    top_n=5,
    lookback_days=60,
    training_lookback_days=365,
    threshold=5.0,
)


# ── Per-period backtest + summary metrics ────────────────────────────────

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
        return _empty_backtest_summary(label)

    v = pd.concat(all_results, ignore_index=True).dropna(subset=["bt_alpha_pct"])
    return _summarize_backtest_results(label, v)


def _empty_backtest_summary(label: str) -> dict:
    return {"label": label, "N": 0, "alpha": 0, "slope": 0, "t": 0, "win%": 0, "r1": 0, "r5": 0}


def _summarize_backtest_results(label: str, v: pd.DataFrame) -> dict:
    """Aggregate per-recommendation results into IS/OOS summary metrics."""
    ra = v.groupby("rank")["bt_alpha_pct"].mean()
    r1 = ra.get(1, 0)
    r5 = ra.get(5, 0)
    t = _t_stat(v["bt_alpha_pct"])

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


def _t_stat(alpha: pd.Series) -> float:
    """t-statistic of mean alpha over its standard error. 0 if undefined."""
    if len(alpha) > 1 and alpha.std() > 0:
        return float(alpha.mean() / (alpha.std() / np.sqrt(len(alpha))))
    return 0.0


def _degradation_ratio(is_alpha: float, oos_alpha: float) -> float:
    """OOS/IS ratio. >1 = alpha grew, 0.5-1 = healthy decay, <0.5 = degraded."""
    if is_alpha == 0:
        return 0.0
    return round(oos_alpha / is_alpha, 2)


def _print_fold(result: dict) -> None:
    print(f"  {result['label']:45s} N={result['N']:3d} alpha={result['alpha']:+6.1f}% "
          f"slope={result['slope']:+6.1f}% t={result['t']:+6.2f} win={result['win%']:.0f}%")


# ── Split OOS (IS 2022-2023, OOS 2024-2025) ───────────────────────────────

def run_split_oos(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    entry_prices: pd.DataFrame,
    h: int,
) -> dict:
    """Single train/test split: IS=2022-2023, OOS=2024-2025."""
    full = run_backtest_split(
        all_tx, prices, entry_prices,
        date(2022, 1, 1), date(2025, 6, 30), h,
        label=f"Full 2022-2025 (h={h})",
    )
    is_result = run_backtest_split(
        all_tx, prices, entry_prices,
        date(2022, 1, 1), date(2023, 12, 31), h,
        label=f"IS 2022-2023 (h={h})",
    )
    oos_result = run_backtest_split(
        all_tx, prices, entry_prices,
        date(2024, 1, 1), date(2025, 6, 30), h,
        label=f"OOS 2024-2025 (h={h})",
    )
    ratio = _degradation_ratio(is_result["alpha"], oos_result["alpha"])

    print(f"\n--- Split OOS (h={h}) ---")
    for r in [full, is_result, oos_result]:
        _print_fold(r)
    print(f"  OOS/IS ratio: {ratio:.2f}")

    return {
        "full": full,
        "is": is_result,
        "oos": oos_result,
        "oos_is_ratio": ratio,
    }


# ── Walk-forward (3 expanding-window folds) ───────────────────────────

def run_walk_forward(
    all_tx: pd.DataFrame,
    prices: pd.DataFrame,
    entry_prices: pd.DataFrame,
    h: int,
) -> dict:
    """Walk-forward validation across 3 expanding-window folds."""
    folds = _build_folds()

    print(f"\n{'='*70}")
    print(f"Walk-Forward Validation (h={h}, freq=30, mb=2, top_n=5)")
    print(f"{'='*70}")

    fold_results = []
    for fold in folds:
        fold_results.append(_run_one_fold(all_tx, prices, entry_prices, h, fold))

    return {"folds": fold_results}


def _build_folds() -> list[dict]:
    return [
        {"train_start": "2022-01-01", "train_end": "2022-12-31",
         "test_start": "2023-01-01", "test_end": "2023-12-31",
         "label": "Fold 1: train 2022, test 2023"},
        {"train_start": "2022-01-01", "train_end": "2023-12-31",
         "test_start": "2024-01-01", "test_end": "2024-12-31",
         "label": "Fold 2: train 2022-23, test 2024"},
        {"train_start": "2022-01-01", "train_end": "2024-12-31",
         "test_start": "2025-01-01", "test_end": "2025-06-30",
         "label": "Fold 3: train 2022-24, test 2025-H1"},
    ]


def _run_one_fold(all_tx, prices, entry_prices, h, fold) -> dict:
    """Run one walk-forward fold (IS + OOS) and return the fold record."""
    is_result = run_backtest_split(
        all_tx, prices, entry_prices,
        date.fromisoformat(fold["train_start"]),
        date.fromisoformat(fold["train_end"]),
        h,
        label=f"Train {fold['train_start'][:4]}-{fold['train_end'][:4]}",
    )
    oos_result = run_backtest_split(
        all_tx, prices, entry_prices,
        date.fromisoformat(fold["test_start"]),
        date.fromisoformat(fold["test_end"]),
        h,
        label=f"Test  {fold['test_start'][:4]}-{fold['test_end'][:4]}",
    )
    ratio = _degradation_ratio(is_result["alpha"], oos_result["alpha"])

    print(f"\n  {fold['label']}")
    _print_fold({**is_result, "label": is_result["label"]})
    _print_fold({**oos_result, "label": oos_result["label"]})
    print(f"  OOS/IS ratio: {ratio:.2f}")

    return {
        "fold": fold["label"],
        "is": is_result,
        "oos": oos_result,
        "oos_is_ratio": ratio,
    }


# ── Data loading + summary printing + output serialization ─────────────

def _load_data(db: Database):
    """Load all transactions, prices, and entry_prices for the OOS period."""
    tx_start = pd.Timestamp("2021-10-07")
    tx_end = pd.Timestamp("2025-06-30")
    price_end = tx_end + pd.Timedelta(days=130)
    all_tx = db.get_transactions_by_date_range(tx_start, tx_end)
    all_tickers = sorted(set(all_tx["ticker"].dropna().astype(str).unique().tolist()) | {"SPY"})
    prices = db.get_prices(all_tickers, tx_start, price_end)
    entry_prices = db.get_entry_prices(all_tickers, tx_start, price_end)
    return all_tx, prices, entry_prices


def _print_header(all_tx: pd.DataFrame, prices: pd.DataFrame) -> None:
    print("Out-of-Sample Validation")
    print("=" * 70)
    print(f"Data: {len(all_tx)} tx, {prices.shape[1]} tickers")
    print(f"Config: h={BEST_CONFIG['horizon']}, freq={BEST_CONFIG['frequency_days']}, "
          f"mb={BEST_CONFIG['min_buyers']}, top_n={BEST_CONFIG['top_n']}")
    print()


def _print_summary(split_result: dict, wf_result: dict) -> None:
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    is_r = split_result["is"]
    oos_r = split_result["oos"]
    print("\n  Split OOS (IS=2022-23, OOS=2024-25):")
    print(f"    IS:  alpha={is_r['alpha']:+.1f}%  slope={is_r['slope']:+.1f}%  "
          f"win={is_r['win%']:.0f}%  N={is_r['N']}")
    print(f"    OOS: alpha={oos_r['alpha']:+.1f}%  slope={oos_r['slope']:+.1f}%  "
          f"win={oos_r['win%']:.0f}%  N={oos_r['N']}")
    print(f"    Ratio: {split_result['oos_is_ratio']:.2f}")

    wf_alphas = [f["oos"]["alpha"] for f in wf_result["folds"]]
    wf_slopes = [f["oos"]["slope"] for f in wf_result["folds"]]
    wf_ratios = [f["oos_is_ratio"] for f in wf_result["folds"]]

    print("\n  Walk-Forward (3 folds):")
    for fr in wf_result["folds"]:
        o = fr["oos"]
        print(f"    {fr['fold']:45s} alpha={o['alpha']:+6.1f}% "
              f"slope={o['slope']:+6.1f}% ratio={fr['oos_is_ratio']:.2f}")
    avg_ratio = round(np.mean(wf_ratios), 2)
    print(f"    {'Average':45s} alpha={np.mean(wf_alphas):+6.1f}% "
          f"slope={np.mean(wf_slopes):+6.1f}% ratio={avg_ratio:.2f}")

    _print_robustness(wf_alphas, wf_slopes, avg_ratio)


def _print_robustness(wf_alphas: list[float], wf_slopes: list[float], avg_ratio: float) -> None:
    positive_oos = sum(1 for a in wf_alphas if a > 0)
    slope_robust = sum(1 for s in wf_slopes if s > 0)
    print("\n  Robustness:")
    print(f"    Positive OOS alpha in {positive_oos}/3 folds")
    print(f"    Positive OOS slope (r1>r5) in {slope_robust}/3 folds")
    print(f"    Avg OOS/IS ratio: {avg_ratio:.2f}")
    if positive_oos >= 2 and avg_ratio > 0.3:
        print("    -> SIGNAL IS ROBUST (alpha survives OOS)")
    elif positive_oos >= 1 and avg_ratio > 0.1:
        print("    -> SIGNAL IS PARTIALLY ROBUST (some alpha survives)")
    else:
        print("    -> SIGNAL IS NOT ROBUST (alpha degrades OOS)")


def _build_output_dict(
    all_tx: pd.DataFrame, prices: pd.DataFrame,
    split_result: dict, wf_result: dict,
) -> dict:
    """Build the JSON-serializable output dict."""
    wf_alphas = [f["oos"]["alpha"] for f in wf_result["folds"]]
    wf_slopes = [f["oos"]["slope"] for f in wf_result["folds"]]
    wf_ratios = [f["oos_is_ratio"] for f in wf_result["folds"]]
    return {
        "config": BEST_CONFIG,
        "data": {"tx_count": len(all_tx), "ticker_count": prices.shape[1]},
        "split_oos": {
            "is": split_result["is"],
            "oos": split_result["oos"],
            "full": split_result["full"],
            "oos_is_ratio": split_result["oos_is_ratio"],
        },
        "walk_forward": wf_result,
        "summary": {
            "avg_oos_alpha": round(float(np.mean(wf_alphas)), 2),
            "avg_oos_slope": round(float(np.mean(wf_slopes)), 2),
            "avg_oos_is_ratio": round(float(np.mean(wf_ratios)), 2),
            "positive_oos_alpha_folds": sum(1 for a in wf_alphas if a > 0),
            "positive_oos_slope_folds": sum(1 for s in wf_slopes if s > 0),
        },
    }


def main():
    db = Database(Path("data") / "congress.duckdb", read_only=False)
    all_tx, prices, entry_prices = _load_data(db)
    _print_header(all_tx, prices)

    h = BEST_CONFIG["horizon"]
    split_result = run_split_oos(all_tx, prices, entry_prices, h)
    wf_result = run_walk_forward(all_tx, prices, entry_prices, h)
    _print_summary(split_result, wf_result)

    output = _build_output_dict(all_tx, prices, split_result, wf_result)
    out_path = Path("data") / "oos_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results written to {out_path}")

    db.conn.close()


if __name__ == "__main__":
    main()
