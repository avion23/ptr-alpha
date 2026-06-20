"""Walk-forward analysis of congress member profitability.

Determines which metrics actually predict future alpha and optimizes
position sizing parameters.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.argv = ["ptr-alpha"]

from analyzer.database import Database
from analyzer.signals import calculate_signal_potential
from analyzer.member_ranking import rank_members, _rank_members_impl
from analyzer.exceptions import AnalysisError

# ── Configuration ──────────────────────────────────────────────────────────
HORIZON = 60
DECAY_LAMBDA = 0.005
TRAIN_WINDOW_DAYS = 180  # 6 months training
TEST_WINDOW_DAYS = 180   # 6 months test
MIN_MEMBERS_FOR_CORR = 10

METRICS_TO_TEST = [
    "shrunk_alpha",
    "bayes_win_prob",
    "conviction_score",
    "purchase_trades",
    "prob_up_given_buy",
    "sharpe_ratio",
    "avg_spy_alpha_pct",
]

TOP_N_VALUES = [1, 2, 3, 5, 10, 15]
MIN_BUYERS_VALUES = [1, 2, 3, 4, 5]

# ── Data Loading ───────────────────────────────────────────────────────────
print("Loading data...")
t0 = time.time()

db = Database("data/congress.duckdb")
tx_start = pd.Timestamp("2021-10-07")
tx_end = pd.Timestamp("2025-06-30")
all_tx = db.get_transactions_by_date_range(tx_start, tx_end)

price_end = pd.Timestamp("2025-06-30") + pd.Timedelta(days=130)
all_tickers = sorted(
    set(t for t in all_tx["ticker"].dropna().unique() if isinstance(t, str)) | {"SPY"}
)
prices = db.get_prices(all_tickers, tx_start, price_end)
entry_prices = db.get_entry_prices(all_tickers, tx_start, price_end)

print(f"  Data loaded in {time.time()-t0:.1f}s")
print(f"  Transactions: {len(all_tx)}, Tickers: {len(all_tickers)}")

# ── Compute signals ────────────────────────────────────────────────────────
print("Computing signals...")
t1 = time.time()

sigs = calculate_signal_potential(entry_prices, prices, [HORIZON], decay_lambda=DECAY_LAMBDA)
print(f"  Signals computed in {time.time()-t1:.1f}s")
print(f"  Total signals: {len(sigs)}")

# ── Walk-forward: Collect per-period metrics ───────────────────────────────
print("\nRunning walk-forward analysis...")
t2 = time.time()

# Get all unique disclosure dates to determine windows
disc_dates = sigs["disclosure_date"].dropna().unique()
disc_dates = np.sort(disc_dates)
min_date = pd.Timestamp(disc_dates.min())
max_date = pd.Timestamp(disc_dates.max())

print(f"  Date range: {min_date.date()} to {max_date.date()}")

# Generate rolling windows
windows = []
start = min_date
while start + pd.Timedelta(days=TRAIN_WINDOW_DAYS + TEST_WINDOW_DAYS) <= max_date:
    train_end = start + pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    test_end = train_end + pd.Timedelta(days=TEST_WINDOW_DAYS)
    windows.append({
        "train_start": start,
        "train_end": train_end,
        "test_start": train_end,
        "test_end": test_end,
    })
    # Slide by 3 months for overlapping windows
    start += pd.Timedelta(days=90)

print(f"  Windows to evaluate: {len(windows)}")

# ── Per-window analysis ────────────────────────────────────────────────────
all_window_results = []

for wi, w in enumerate(windows):
    # Filter signals to training period
    train_sigs = sigs[
        (sigs["disclosure_date"] >= w["train_start"])
        & (sigs["disclosure_date"] < w["train_end"])
    ].copy()

    # Filter signals to test period
    test_sigs = sigs[
        (sigs["disclosure_date"] >= w["test_start"])
        & (sigs["disclosure_date"] < w["test_end"])
    ].copy()

    if train_sigs.empty or test_sigs.empty:
        continue

    # Rank members using training data
    try:
        train_rankings = rank_members(train_sigs, HORIZON, threshold=5.0)
    except AnalysisError:
        continue

    if train_rankings.empty or len(train_rankings) < MIN_MEMBERS_FOR_CORR:
        continue

    # For each member in training set, compute their actual test-period alpha
    test_purchases = test_sigs[test_sigs["signal_type"] == "Purchase"].copy()
    if test_purchases.empty:
        continue

    # Compute per-member test alpha (mean spy_alpha_pct)
    test_alpha = (
        test_purchases.groupby("member")["spy_alpha_pct"]
        .agg(["mean", "count", "std"])
        .reset_index()
        .rename(columns={"mean": "test_alpha", "count": "test_trades", "std": "test_std"})
    )

    # Merge training metrics with test alpha
    merged = pd.merge(
        train_rankings[["member"] + METRICS_TO_TEST],
        test_alpha,
        on="member",
        how="inner",
    )

    # Require at least a few test trades for meaningful alpha
    merged = merged[merged["test_trades"] >= 2]

    if len(merged) < MIN_MEMBERS_FOR_CORR:
        continue

    merged["window"] = wi
    merged["train_start"] = w["train_start"]
    merged["test_start"] = w["test_start"]
    all_window_results.append(merged)

    if (wi + 1) % 5 == 0:
        print(f"  Window {wi+1}/{len(windows)}: {len(merged)} members with test data")

wf_time = time.time() - t2
print(f"  Walk-forward completed in {wf_time:.1f}s")

if not all_window_results:
    print("ERROR: No valid windows found!")
    sys.exit(1)

all_wf = pd.concat(all_window_results, ignore_index=True)
print(f"\nTotal window-period observations: {len(all_wf)}")

# ── 1. Spearman Correlations ──────────────────────────────────────────────
print("\n" + "="*60)
print("1. SPEARMAN CORRELATIONS: Training Metric Rank vs Test Alpha")
print("="*60)

correlations = {}
for metric in METRICS_TO_TEST:
    # For each window, compute correlation, then average
    window_corrs = []
    for wi in all_wf["window"].unique():
        subset = all_wf[all_wf["window"] == wi].copy()
        if len(subset) < MIN_MEMBERS_FOR_CORR:
            continue
        # Rank by training metric (higher = better)
        subset["train_rank"] = subset[metric].rank(ascending=False)
        subset["test_rank"] = subset["test_alpha"].rank(ascending=False)

        # Spearman correlation
        corr, pval = stats.spearmanr(subset["train_rank"], subset["test_rank"])
        if not np.isnan(corr):
            window_corrs.append({"corr": corr, "pval": pval})

    if window_corrs:
        avg_corr = np.mean([c["corr"] for c in window_corrs])
        med_corr = np.median([c["corr"] for c in window_corrs])
        sig_count = sum(1 for c in window_corrs if c["pval"] < 0.05)
        correlations[metric] = {
            "mean_spearman": round(float(avg_corr), 4),
            "median_spearman": round(float(med_corr), 4),
            "std_spearman": round(float(np.std([c["corr"] for c in window_corrs])), 4),
            "pct_significant": round(sig_count / len(window_corrs) * 100, 1),
            "n_windows": len(window_corrs),
        }
        print(f"  {metric:25s}: mean={avg_corr:+.4f}, median={med_corr:+.4f}, "
              f"sig={sig_count}/{len(window_corrs)} ({sig_count/len(window_corrs)*100:.0f}%)")
    else:
        correlations[metric] = {"mean_spearman": 0.0, "n_windows": 0}
        print(f"  {metric:25s}: insufficient data")

# ── 2. Tier Analysis ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. TIER ANALYSIS: Top 10% vs Bottom 10% by Training Metric")
print("="*60)

tier_results = {}
for metric in METRICS_TO_TEST:
    top_alphas = []
    bottom_alphas = []
    for wi in all_wf["window"].unique():
        subset = all_wf[all_wf["window"] == wi].copy()
        if len(subset) < 20:  # Need enough members for meaningful tiers
            continue

        n_top = max(1, int(len(subset) * 0.10))
        n_bottom = max(1, int(len(subset) * 0.10))

        sorted_by_metric = subset.sort_values(metric, ascending=False)
        top_tier = sorted_by_metric.head(n_top)
        bottom_tier = sorted_by_metric.tail(n_bottom)

        top_alphas.extend(top_tier["test_alpha"].tolist())
        bottom_alphas.extend(bottom_tier["test_alpha"].tolist())

    if top_alphas and bottom_alphas:
        top_mean = float(np.mean(top_alphas))
        bottom_mean = float(np.mean(bottom_alphas))
        lift = top_mean - bottom_mean
        # Statistical test
        t_stat, p_val = stats.ttest_ind(top_alphas, bottom_alphas)

        tier_results[metric] = {
            "top_10pct_mean_alpha": round(top_mean, 4),
            "bottom_10pct_mean_alpha": round(bottom_mean, 4),
            "alpha_lift": round(lift, 4),
            "lift_p_value": round(float(p_val), 6),
            "n_observations": len(top_alphas),
            "top_outperforms": top_mean > bottom_mean,
        }
        direction = "OUTPERFORMS" if top_mean > bottom_mean else "UNDERPERFORMS"
        print(f"  {metric:25s}: top10={top_mean:+.4f}% bot10={bottom_mean:+.4f}% "
              f"lift={lift:+.4f}% ({direction}, p={p_val:.4f})")
    else:
        tier_results[metric] = {"alpha_lift": 0.0, "n_observations": 0}
        print(f"  {metric:25s}: insufficient data")

# ── 3. Trade Count Threshold ──────────────────────────────────────────────
print("\n" + "="*60)
print("3. MIN TRADE COUNT FOR RELIABILITY")
print("="*60)

trade_count_analysis = {}
for min_trades in [2, 3, 5, 8, 10, 15, 20]:
    subset = all_wf[all_wf["purchase_trades"] >= min_trades]
    if len(subset) < MIN_MEMBERS_FOR_CORR:
        trade_count_analysis[min_trades] = {"n_members": len(subset), "mean_test_alpha": 0.0}
        continue

    corr, pval = stats.spearmanr(subset["shrunk_alpha"], subset["test_alpha"])
    mean_alpha = float(subset["test_alpha"].mean())
    trade_count_analysis[min_trades] = {
        "n_members": int(len(subset)),
        "mean_test_alpha": round(mean_alpha, 4),
        "shrunk_alpha_corr": round(float(corr), 4) if not np.isnan(corr) else 0.0,
        "corr_p_value": round(float(pval), 6) if not np.isnan(pval) else 1.0,
    }
    print(f"  min_trades={min_trades:2d}: n={len(subset):5d}, "
          f"mean_alpha={mean_alpha:+.4f}%, "
          f"corr(shrunk_alpha, test)={corr:+.4f} (p={pval:.4f})")

# ── 4. Position Sizing Grid Search ────────────────────────────────────────
print("\n" + "="*60)
print("4. OPTIMAL POSITION SIZING (top_n × min_buyers grid)")
print("="*60)

# For grid search, use a simulated walk-forward:
# For each window, pick top_n tickers by signal score with min_buyers constraint
# and measure their actual test returns

position_results = []

for top_n in TOP_N_VALUES:
    for min_buyers in MIN_BUYERS_VALUES:
        window_returns = []
        window_wins = 0
        window_total = 0

        for wi, w in enumerate(windows):
            # Training data for member rankings
            train_sigs = sigs[
                (sigs["disclosure_date"] >= w["train_start"])
                & (sigs["disclosure_date"] < w["train_end"])
            ].copy()

            # Test data for evaluating picks
            test_sigs = sigs[
                (sigs["disclosure_date"] >= w["test_start"])
                & (sigs["disclosure_date"] < w["test_end"])
            ].copy()

            if train_sigs.empty or test_sigs.empty:
                continue

            # Get member rankings from training
            try:
                train_rankings = rank_members(train_sigs, HORIZON, threshold=5.0)
            except AnalysisError:
                continue

            if train_rankings.empty:
                continue

            # Get test purchases
            test_purchases = test_sigs[
                test_sigs["signal_type"] == "Purchase"
            ].copy()
            if test_purchases.empty:
                continue

            # For each test-period ticker, compute buyer score
            ticker_scores = []
            for ticker, t_grp in test_purchases.groupby("ticker"):
                buyers = t_grp["member"].unique()
                if len(buyers) < min_buyers:
                    continue

                # Score buyers based on training rankings
                buyer_scores = []
                for b in buyers:
                    match = train_rankings[train_rankings["member"] == b]
                    if not match.empty and "shrunk_alpha" in match.columns:
                        buyer_scores.append(float(match["shrunk_alpha"].iloc[0]))

                if not buyer_scores:
                    continue

                avg_score = np.mean(buyer_scores)
                best_score = max(buyer_scores)
                # Weighted score: average buyer quality × buyer count bonus
                count_bonus = np.log1p(len(buyers))
                composite = avg_score * count_bonus

                ticker_scores.append({
                    "ticker": ticker,
                    "n_buyers": len(buyers),
                    "avg_buyer_alpha": avg_score,
                    "best_buyer_alpha": best_score,
                    "composite_score": composite,
                    "test_returns": t_grp["spy_alpha_pct"].dropna().tolist(),
                })

            if not ticker_scores:
                continue

            # Rank tickers and pick top_n
            score_df = pd.DataFrame(ticker_scores)
            score_df = score_df.sort_values("composite_score", ascending=False).head(top_n)

            for _, row in score_df.iterrows():
                if row["test_returns"]:
                    avg_ret = float(np.mean(row["test_returns"]))
                    window_returns.append(avg_ret)
                    window_total += 1
                    if avg_ret > 0:
                        window_wins += 1

        if window_returns:
            avg_ret = float(np.mean(window_returns))
            std_ret = float(np.std(window_returns)) if len(window_returns) > 1 else 0.0
            sharpe = avg_ret / std_ret if std_ret > 0 else 0.0
            win_rate = window_wins / window_total * 100 if window_total > 0 else 0
            max_dd = float(np.min(window_returns)) if window_returns else 0.0

            position_results.append({
                "top_n": top_n,
                "min_buyers": min_buyers,
                "total_picks": window_total,
                "avg_spy_alpha_pct": round(avg_ret, 4),
                "std_spy_alpha_pct": round(std_ret, 4),
                "sharpe_proxy": round(float(sharpe), 4),
                "win_rate_pct": round(win_rate, 1),
                "worst_pick_alpha": round(max_dd, 4),
            })

# Print grid results
print(f"\n  {'top_n':>5s} {'min_b':>5s} {'picks':>6s} {'avg_α%':>8s} {'sharpe':>8s} {'win%':>6s}")
print("  " + "-" * 45)
for r in position_results:
    print(f"  {r['top_n']:5d} {r['min_buyers']:5d} {r['total_picks']:6d} "
          f"{r['avg_spy_alpha_pct']:+8.4f} {r['sharpe_proxy']:8.4f} {r['win_rate_pct']:6.1f}")

# ── 5. Best Combined Metric ───────────────────────────────────────────────
print("\n" + "="*60)
print("5. BEST COMBINED PREDICTOR (multi-metric)")
print("="*60)

# Try a few combined scores
combined_results = {}
for wi in all_wf["window"].unique():
    subset = all_wf[all_wf["window"] == wi].copy()
    if len(subset) < MIN_MEMBERS_FOR_CORR:
        continue

    # Combined score: weighted blend
    subset["combined_v1"] = (
        subset["shrunk_alpha"].rank(pct=True) * 0.4
        + subset["bayes_win_prob"].rank(pct=True) * 0.2
        + subset["conviction_score"].rank(pct=True) * 0.2
        + subset["sharpe_ratio"].rank(pct=True) * 0.2
    )
    subset["combined_v2"] = (
        subset["prob_up_given_buy"].rank(pct=True) * 0.3
        + subset["conviction_score"].rank(pct=True) * 0.3
        + subset["purchase_trades"].rank(pct=True) * 0.2
        + subset["shrunk_alpha"].rank(pct=True) * 0.2
    )
    # Simple: just trades × win prob (conviction proxy)
    subset["trades_x_winprob"] = subset["purchase_trades"] * subset["prob_up_given_buy"]

    for combo_name in ["combined_v1", "combined_v2", "trades_x_winprob"]:
        corr, _ = stats.spearmanr(subset[combo_name], subset["test_alpha"])
        if combo_name not in combined_results:
            combined_results[combo_name] = []
        if not np.isnan(corr):
            combined_results[combo_name].append(corr)

print("  Combined metric correlations with test alpha:")
for name, corrs in combined_results.items():
    if corrs:
        avg = np.mean(corrs)
        print(f"    {name:25s}: mean_spearman={avg:+.4f}")

# ── Build output JSON ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("WRITING RESULTS")
print("="*60)

output = {
    "analysis_config": {
        "horizon": HORIZON,
        "train_window_days": TRAIN_WINDOW_DAYS,
        "test_window_days": TEST_WINDOW_DAYS,
        "decay_lambda": DECAY_LAMBDA,
        "total_transactions": len(all_tx),
        "total_tickers": len(all_tickers),
        "total_signals": len(sigs),
        "n_windows": len(windows),
        "valid_windows_analyzed": len(set(all_wf["window"])),
        "total_window_observations": len(all_wf),
    },
    "spearman_correlations": correlations,
    "tier_analysis": tier_results,
    "trade_count_reliability": {
        str(k): v for k, v in trade_count_analysis.items()
    },
    "position_sizing_grid": position_results,
    "combined_metrics": {
        name: {
            "mean_spearman": round(float(np.mean(corrs)), 4),
            "std_spearman": round(float(np.std(corrs)), 4) if len(corrs) > 1 else 0.0,
            "n_windows": len(corrs),
        }
        for name, corrs in combined_results.items()
    },
    "recommendations": {},
}

# Generate recommendations
best_metric = max(correlations.items(), key=lambda x: abs(x[1]["mean_spearman"]))
best_grid = max(position_results, key=lambda x: x["sharpe_proxy"]) if position_results else {}

# Find best combined
best_combined = None
if combined_results:
    best_combined_name = max(
        combined_results.items(),
        key=lambda x: abs(np.mean(x[1])) if x[1] else 0,
    )
    best_combined = {
        "name": best_combined_name[0],
        "mean_spearman": round(float(np.mean(best_combined_name[1])), 4),
    }

# Best tier metric
best_tier = max(
    tier_results.items(),
    key=lambda x: abs(x[1].get("alpha_lift", 0))
) if tier_results else (None, {})

output["recommendations"] = {
    "best_single_predictor": {
        "metric": best_metric[0],
        "mean_spearman": best_metric[1]["mean_spearman"],
        "interpretation": (
            "Strong positive" if best_metric[1]["mean_spearman"] > 0.2
            else "Moderate positive" if best_metric[1]["mean_spearman"] > 0.05
            else "Weak/no" if best_metric[1]["mean_spearman"] > -0.05
            else "Negative (avoid)"
        ),
    },
    "best_combined_predictor": best_combined,
    "best_tier_metric": {
        "metric": best_tier[0],
        "alpha_lift": best_tier[1].get("alpha_lift", 0) if best_tier[0] else 0,
    },
    "optimal_position_sizing": best_grid,
    "key_findings": [],
}

# Generate key findings
findings = []

# Finding 1: Which metrics predict persistence
sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]["mean_spearman"]), reverse=True)
findings.append(
    f"Best single predictor of future alpha: '{sorted_corrs[0][0]}' "
    f"(Spearman={sorted_corrs[0][1]['mean_spearman']:+.4f})"
)

# Finding 2: Trade count matters
if trade_count_analysis:
    trade_counts = sorted(trade_count_analysis.items(), key=lambda x: x[1].get("shrunk_alpha_corr", 0), reverse=True)
    if trade_counts:
        findings.append(
            f"Trade count threshold: correlation improves at "
            f"min_trades={trade_counts[0][0]} "
            f"(corr={trade_counts[0][1].get('shrunk_alpha_corr', 0):+.4f})"
        )

# Finding 3: Position sizing
if best_grid:
    findings.append(
        f"Optimal position sizing: top_n={best_grid.get('top_n', 'N/A')}, "
        f"min_buyers={best_grid.get('min_buyers', 'N/A')} "
        f"(sharpe_proxy={best_grid.get('sharpe_proxy', 0):.4f})"
    )

# Finding 4: Tier analysis
if best_tier[0]:
    findings.append(
        f"Best tier separation: '{best_tier[0]}' "
        f"(top10% vs bottom10% alpha lift={best_tier[1].get('alpha_lift', 0):+.4f}%)"
    )

# Finding 5: Combined metric
if best_combined:
    findings.append(
        f"Best combined predictor: '{best_combined['name']}' "
        f"(Spearman={best_combined['mean_spearman']:+.4f})"
    )

output["recommendations"]["key_findings"] = findings

# Write output
output_path = Path("data/member_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Convert numpy types for JSON serialization
def np_convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=np_convert)

print(f"\nResults written to: {output_path}")

# ── Summary Printout ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)

print("\nTop predictors of future member alpha (Spearman correlation):")
for name, data in sorted_corrs[:5]:
    print(f"  {name:25s}: {data['mean_spearman']:+.4f}")

print("\nKey findings:")
for i, finding in enumerate(findings, 1):
    print(f"  {i}. {finding}")

print(f"\nTotal analysis time: {time.time()-t0:.1f}s")
