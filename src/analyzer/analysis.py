from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType

DECAY_LAMBDA = 0.05
POSITION_SIZE_BASELINE = 15000.0
MAX_DISCLOSURE_METADATA_ADJUSTMENT = 0.05


def bayesian_win_probability(wins: int, losses: int, market_prior: float = 0.55) -> float:
    alpha = market_prior * 20
    beta = (1 - market_prior) * 20
    return (alpha + wins) / (alpha + beta + wins + losses)


def _get_horizon_data(
    signals_df: pd.DataFrame, horizon: int, transaction_type: str | None = None
) -> pd.DataFrame:
    data = signals_df[signals_df["horizon_days"] == horizon]
    if transaction_type is not None:
        data = data[data["signal_type"] == transaction_type]
    return data


def _compute_dynamic_prior(signals_df: pd.DataFrame, horizon: int) -> float:
    horizon_signals = _get_horizon_data(signals_df, horizon)
    if horizon_signals.empty:
        return 0.50
    up_prob = (horizon_signals["decayed_return_pct"] > 0).mean()
    return max(up_prob, 0.50)


def _size_score_factor(trades: pd.DataFrame) -> float:
    if "amount_midpoint" not in trades.columns:
        return 1.0
    amount = trades["amount_midpoint"].dropna()
    if amount.empty:
        return 1.0
    average_amount = max(float(amount.mean()), 1.0)
    adjustment = np.log10(average_amount / POSITION_SIZE_BASELINE) * 0.025
    adjustment = float(np.clip(adjustment, -MAX_DISCLOSURE_METADATA_ADJUSTMENT, MAX_DISCLOSURE_METADATA_ADJUSTMENT))
    return 1.0 + adjustment


def _owner_score_factor(trades: pd.DataFrame) -> float:
    if "owner_code" not in trades.columns:
        return 1.0
    owner_codes = trades["owner_code"].fillna("").astype(str).str.upper()
    if owner_codes.empty:
        return 1.0
    dependent_child_ratio = (owner_codes == "DC").mean()
    return 1.0 - dependent_child_ratio * MAX_DISCLOSURE_METADATA_ADJUSTMENT


def _get_top_signals(signals_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    top_data = _get_horizon_data(signals_df, horizon, TransactionType.PURCHASE.value)
    if top_data.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    return top_data.nlargest(top_n, "spy_alpha_pct")[
        ["member", "ticker", "disclosure_date", "spy_alpha_pct", "peak_potential_pct"]
    ]


def _get_member_signals(
    signals_df: pd.DataFrame, member: str, horizon: int = 90, top_n: int = 5
) -> pd.DataFrame:
    member_data = _get_horizon_data(signals_df, horizon)
    member_data = member_data[member_data["member"] == member]

    if member_data.empty:
        raise AnalysisError(f"No signals found for member {member} at horizon {horizon}")

    purchases = member_data[member_data["signal_type"] == TransactionType.PURCHASE.value]
    if purchases.empty:
        raise AnalysisError(f"No purchase signals for member {member} at horizon {horizon}")

    return purchases.nlargest(top_n, "spy_alpha_pct")[
        ["ticker", "disclosure_date", "spy_alpha_pct", "peak_potential_pct"]
    ]


def calculate_signal_potential(
    entry_prices_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: list[int] = [30, 60, 90, 180],
    decay_lambda: float = DECAY_LAMBDA,
) -> pd.DataFrame:
    if entry_prices_df.empty:
        raise AnalysisError("Empty entry prices dataframe")
    if prices_df.empty:
        raise AnalysisError("Empty prices dataframe")

    required_cols = {"member", "ticker", "disclosure_date", "transaction_type", "entry_price"}
    if not required_cols.issubset(entry_prices_df.columns):
        raise AnalysisError(f"Missing columns in entry_prices: {required_cols - set(entry_prices_df.columns)}")

    prices_long = prices_df.stack().reset_index(name="price")
    prices_long.columns = ["price_date", "ticker", "price"]

    spy_prices = prices_long[prices_long["ticker"] == "SPY"][["price_date", "price"]].rename(
        columns={"price": "spy_price"}
    )
    prices_long = prices_long[prices_long["ticker"] != "SPY"].copy()

    signals = entry_prices_df.copy()

    if signals.empty:
        raise AnalysisError("No valid price matches found for transactions")

    signals = signals.assign(horizon_days=[horizons] * len(signals)).explode("horizon_days").reset_index(drop=True)
    signals["horizon_days"] = signals["horizon_days"].astype("int32")
    signals["window_end"] = signals["disclosure_date"] + pd.to_timedelta(signals["horizon_days"], unit="D")
    signals["signal_id"] = range(len(signals))

    merged = signals.merge(prices_long, on="ticker", suffixes=("", "_price"))
    window_mask = (merged["price_date"] >= merged["disclosure_date"]) & (merged["price_date"] <= merged["window_end"])
    windowed = merged[window_mask].copy()

    if windowed.empty:
        raise AnalysisError("No price data found in signal windows")

    windowed["days_from_disclosure"] = (windowed["price_date"] - windowed["disclosure_date"]).dt.days
    windowed["decay_factor"] = np.exp(-decay_lambda * windowed["days_from_disclosure"])
    windowed["weighted_return"] = (windowed["price"] / windowed["entry_price"] - 1) * windowed["decay_factor"]

    if not spy_prices.empty:
        spy_merged = windowed.merge(spy_prices, on="price_date", how="left")
        spy_merged["spy_entry_price"] = spy_merged.groupby("signal_id")["spy_price"].transform("first")
        windowed["spy_return"] = spy_merged["spy_price"] / spy_merged["spy_entry_price"] - 1
    else:
        windowed["spy_return"] = 0.0

    windowed["weighted_spy_return"] = windowed["spy_return"] * windowed["decay_factor"]

    agg = windowed.groupby("signal_id").agg(
        peak_price=("price", "max"),
        trough_price=("price", "min"),
        decayed_return=("weighted_return", "sum"),
        spy_cumulative=("weighted_spy_return", "sum"),
        entry_price_first=("entry_price", "first"),
        last_price=("price", "last")
    )
    agg["total_return"] = (agg["last_price"] / agg["entry_price_first"] - 1)
    agg = agg.reset_index()

    final = signals.merge(
        agg[["signal_id", "peak_price", "trough_price", "decayed_return", "spy_cumulative", "total_return"]],
        on="signal_id", how="left"
    )

    is_purchase = final["transaction_type"] == TransactionType.PURCHASE.value
    purchase_mask = is_purchase & (final["entry_price"] != 0)
    sale_mask = ~is_purchase & (final["trough_price"].notna()) & (final["trough_price"] != 0)

    peak_potential = np.zeros(len(final))
    peak_potential[purchase_mask.values] = (
        (final.loc[purchase_mask.values, "peak_price"] / final.loc[purchase_mask.values, "entry_price"] - 1) * 100
    ).values
    peak_potential[sale_mask.values] = (
        (final.loc[sale_mask.values, "entry_price"] / final.loc[sale_mask.values, "trough_price"] - 1) * 100
    ).values

    optional_columns = [column for column in ["owner_code", "amount_midpoint"] if column in final.columns]
    result_columns = [
        "member", "ticker", "disclosure_date", "signal_type", "horizon_days", "entry_price",
        "peak_potential_pct", "decayed_return_pct", "spy_alpha_pct", "total_return_pct", *optional_columns,
    ]

    return final.assign(
        signal_type=final["transaction_type"],
        peak_potential_pct=peak_potential,
        decayed_return_pct=final["decayed_return"].fillna(0).values * 100,
        spy_alpha_pct=(final["decayed_return"].fillna(0) - final["spy_cumulative"].fillna(0)).values * 100,
        total_return_pct=final["total_return"].fillna(0).values * 100,
    )[result_columns]


def _compute_member_stats(
    member: str, grp: pd.DataFrame, market_prior: float, threshold: float | None = None
) -> dict | None:
    rets = grp["decayed_return_pct"].dropna().values
    if len(rets) == 0:
        return None

    median_ret = float(np.median(rets))
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0

    wins = int((rets > 0).sum())
    losses = int(len(rets) - wins)
    p_up = wins / len(rets)
    bayes_win_prob = bayesian_win_probability(wins, losses, market_prior)
    bayes_factor = bayes_win_prob / market_prior

    spy_alpha_vals = grp["spy_alpha_pct"].dropna().values
    avg_spy_alpha = float(np.mean(spy_alpha_vals)) if len(spy_alpha_vals) > 0 else 0.0

    stats = {
        "member": member,
        "median_return_pct": round(median_ret, 2),
        "mean_return_pct": round(mean_ret, 2),
        "trades": len(rets),
        "sharpe_ratio": round(sharpe, 3),
        "prob_up": round(p_up, 3),
        "bayes_win_prob": round(bayes_win_prob, 3),
        "bayes_factor": round(bayes_factor, 3),
        "avg_spy_alpha_pct": round(avg_spy_alpha, 2),
    }
    if threshold is not None:
        stats["hit_rate_pct"] = round((grp["peak_potential_pct"] > threshold).mean() * 100, 2)
    return stats


def rank_members(signal_df: pd.DataFrame, horizon: int = 90, threshold: float = 5.0) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    purchases = signal_df[signal_df["signal_type"] == TransactionType.PURCHASE.value]
    if purchases.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    market_prior = _compute_dynamic_prior(signal_df, horizon)
    member_stats = []
    for member, purchase_grp in purchases.groupby("member"):
        row = _compute_member_stats(member, purchase_grp, market_prior, threshold)
        if row is not None:
            member_stats.append(row)

    result = pd.DataFrame(member_stats)
    if result.empty:
        return result

    return result.rename(columns={
        "mean_return_pct": "avg_peak_return_pct",
        "median_return_pct": "median_peak_return_pct",
        "trades": "purchase_trades",
        "prob_up": "prob_up_given_buy",
    }).sort_values("avg_spy_alpha_pct", ascending=False)


def get_top_signals(signal_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_top_signals(signal_df, horizon, top_n)


def get_member_signals(signal_df: pd.DataFrame, member: str, horizon: int = 90, top_n: int = 5) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_member_signals(signal_df, member, horizon, top_n)


def get_analysis_table(
    signals_df: pd.DataFrame,
    member_filter: str | None,
    show_signals: bool,
    horizon: int,
    top_n: int | None,
    threshold: float,
) -> pd.DataFrame:
    if member_filter:
        return _get_member_signals(signals_df, member_filter, horizon, top_n or 5)
    if show_signals:
        return _get_top_signals(signals_df, horizon, top_n or 15)
    return rank_members(signals_df, horizon, threshold)


def _get_ticker_purchases(
    ticker: str,
    transactions_df: pd.DataFrame,
) -> pd.DataFrame:
    return transactions_df[
        (transactions_df["ticker"] == ticker)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    ]


def score_ticker_by_buyers(
    ticker: str,
    transactions_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    horizon: int = 90,
    threshold: float = 5.0,
    member_rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if signals_df.empty:
        raise AnalysisError("Empty signal dataframe")
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")

    if member_rankings is None:
        member_rankings = rank_members(signals_df, horizon, threshold)

    ticker_trades = _get_ticker_purchases(ticker, transactions_df)

    if ticker_trades.empty:
        return pd.DataFrame({
            "ticker": [ticker],
            "num_buyers": [0],
            "signal_score": [0.0]
        })

    buyers = ticker_trades["member"].unique()
    buyer_stats = member_rankings[member_rankings["member"].isin(buyers)].sort_values(
        "avg_spy_alpha_pct", ascending=False
    )

    if buyer_stats.empty:
        return pd.DataFrame({
            "ticker": [ticker],
            "num_buyers": [len(buyers)],
            "buyers": [", ".join(buyers[:3])],
            "signal_score": [0.0]
        })

    avg_rank = buyer_stats["avg_spy_alpha_pct"].mean()
    max_rank = buyer_stats["avg_spy_alpha_pct"].max()
    total_trades = buyer_stats["purchase_trades"].sum()
    rated_buyers = len(buyer_stats)
    base_signal_score = rated_buyers * avg_rank
    size_factor = _size_score_factor(ticker_trades)
    owner_factor = _owner_score_factor(ticker_trades)
    signal_score = base_signal_score * size_factor * owner_factor

    top_buyers = buyer_stats["member"].head(3).tolist()
    buyer_label = f"Top {len(top_buyers)} of {len(buyers)}" if len(buyers) > 3 else f"{len(buyers)}"

    return pd.DataFrame({
        "ticker": [ticker],
        "num_buyers": [len(buyers)],
        "buyer_label": [buyer_label],
        "buyers": [", ".join(top_buyers)],
        "avg_buyer_performance": [round(avg_rank, 2)],
        "best_buyer_performance": [round(max_rank, 2)],
        "total_buyer_trades": [int(total_trades)],
        "base_signal_score": [round(base_signal_score, 2)],
        "size_factor": [round(size_factor, 3)],
        "owner_factor": [round(owner_factor, 3)],
        "signal_score": [round(signal_score, 2)]
    })


def get_ticker_buyers_with_rankings(
    ticker: str,
    transactions_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    horizon: int = 90,
    threshold: float = 5.0,
) -> pd.DataFrame:
    if signals_df.empty:
        raise AnalysisError("Empty signal dataframe")
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")

    member_rankings = rank_members(signals_df, horizon, threshold)

    ticker_trades = _get_ticker_purchases(ticker, transactions_df)

    if ticker_trades.empty:
        raise AnalysisError(f"No purchases found for {ticker}")

    buyers_with_dates = ticker_trades.groupby("member").agg({
        "transaction_date": list,
        "disclosure_date": list
    }).reset_index()

    result = pd.merge(
        buyers_with_dates,
        member_rankings[["member", "avg_spy_alpha_pct", "hit_rate_pct", "purchase_trades"]],
        on="member",
        how="left"
    )

    result = result.sort_values("avg_spy_alpha_pct", ascending=False, na_position="last")
    result["num_purchases"] = result["transaction_date"].apply(len)

    return result[["member", "num_purchases", "transaction_date", "disclosure_date",
                   "avg_spy_alpha_pct", "hit_rate_pct", "purchase_trades"]]


def rank_sales(signal_df: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    sales = signal_df[signal_df["signal_type"] == TransactionType.SALE.value]
    if sales.empty:
        raise AnalysisError(f"No sale signals found for horizon {horizon}")

    market_prior = _compute_dynamic_prior(signal_df, horizon)
    member_stats = []
    for member, sale_grp in sales.groupby("member"):
        row = _compute_member_stats(member, sale_grp, market_prior)
        if row is not None:
            member_stats.append(row)

    result = pd.DataFrame(member_stats)
    if result.empty:
        return result

    return result.rename(columns={
        "mean_return_pct": "avg_loss_avoided_pct",
        "median_return_pct": "median_loss_avoided_pct",
        "trades": "sale_trades",
        "prob_up": "prob_up_given_sell",
    }).sort_values("avg_spy_alpha_pct", ascending=False)
