from __future__ import annotations

from math import exp, lgamma, log

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType

DECAY_LAMBDA = 0.05
POSITION_SIZE_BASELINE = 10000.0
MAX_DISCLOSURE_METADATA_ADJUSTMENT = 0.15
BAYES_PRIOR_STRENGTH = 20.0
BUYER_RECENCY_DECAY = 0.03


def _compute_ticker_entry_value(
    ticker: str,
    signals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
    rho: float = 0.000137,
) -> float | None:
    """Compute OU entry value V(0) for a ticker from historical return curves.

    Falls back to global prior (average across all tickers) if ticker has
    no prior disclosure history.
    """
    from analyzer.return_process import compute_entry_value, fit_ou

    ticker_col = ticker in prices_df.columns if hasattr(prices_df, 'columns') else False

    # Get all prior fully-elapsed purchase signals for this ticker
    eligible = signals_df[
        (signals_df["ticker"] == ticker)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= as_of_date - pd.Timedelta(days=horizon))
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ].copy()

    curves = []
    if ticker_col and not eligible.empty:
        ticker_prices = prices_df[ticker].dropna()
        for _, row in eligible.iterrows():
            disc_date = row["disclosure_date"]
            entry_price = row.get("entry_price")
            if not entry_price or entry_price <= 0:
                continue
            end_date = disc_date + pd.Timedelta(days=horizon)
            window = ticker_prices[
                (ticker_prices.index >= disc_date) & (ticker_prices.index <= end_date)
            ]
            if len(window) < 3:
                continue
            curve = (window.values / entry_price) - 1.0
            curves.append(curve)

    if curves:
        v0, _mu, _theta = compute_entry_value(curves, rho=rho)
        return v0

    # Fallback: global prior from historical curves (filtered by cutoff to avoid look-ahead)
    all_purchases = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= as_of_date - pd.Timedelta(days=horizon))
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ].copy()

    if all_purchases.empty or not ticker_col:
        return None

    ticker_prices = prices_df[ticker].dropna()
    global_curves = []
    for _, row in all_purchases.iterrows():
        disc_date = row["disclosure_date"]
        entry_price = row.get("entry_price")
        tkr = row["ticker"]
        if not entry_price or entry_price <= 0 or tkr not in prices_df.columns:
            continue
        tkr_prices = prices_df[tkr].dropna()
        end_date = disc_date + pd.Timedelta(days=horizon)
        window = tkr_prices[
            (tkr_prices.index >= disc_date) & (tkr_prices.index <= end_date)
        ]
        if len(window) < 3:
            continue
        curve = (window.values / entry_price) - 1.0
        global_curves.append(curve)

    if not global_curves:
        return None

    v0, _mu, _theta = compute_entry_value(global_curves, rho=rho)
    return v0
BUYER_CONVERGENCE_WEIGHT = 0.30
MIN_ENTRY_PRICE = 3.0
CONVICTION_WEIGHT_ALPHA = 0.6
CONVICTION_WEIGHT_REALIZED = 0.4
TICKER_PERF_MIN_TRADES = 3


def bayesian_win_probability(wins: int, losses: int, market_prior: float = 0.55) -> float:
    alpha = market_prior * BAYES_PRIOR_STRENGTH
    beta = (1 - market_prior) * BAYES_PRIOR_STRENGTH
    return (alpha + wins) / (alpha + beta + wins + losses)


def bayes_factor_against_market(wins: int, losses: int, market_prior: float = 0.55) -> float:
    observations = wins + losses
    if observations == 0:
        return 1.0
    market_prior = float(np.clip(market_prior, 1e-6, 1 - 1e-6))
    alpha = market_prior * BAYES_PRIOR_STRENGTH
    beta = (1 - market_prior) * BAYES_PRIOR_STRENGTH
    log_marginal = (
        lgamma(alpha + wins)
        + lgamma(beta + losses)
        - lgamma(alpha + beta + observations)
        - lgamma(alpha)
        - lgamma(beta)
        + lgamma(alpha + beta)
    )
    log_market = wins * log(market_prior) + losses * log(1 - market_prior)
    return exp(float(np.clip(log_marginal - log_market, -50, 50)))


def _apply_quality_filter(signals_df: pd.DataFrame) -> pd.DataFrame:
    if "entry_price" not in signals_df.columns:
        return signals_df
    return signals_df[signals_df["entry_price"] >= MIN_ENTRY_PRICE].copy()


def _conviction_score(trades: pd.DataFrame) -> float:
    trade_count = len(trades)
    if trade_count == 0:
        return 0.0
    count_score = min(trade_count / 10.0, 1.0)
    has_amounts = "amount_midpoint" in trades.columns and trades["amount_midpoint"].notna().any()
    size_score = 1.0
    if has_amounts:
        avg_amount = trades["amount_midpoint"].dropna().mean()
        size_score = min(avg_amount / 50000.0, 1.0)
    return count_score * 0.6 + size_score * 0.4


def _get_horizon_data(
    signals_df: pd.DataFrame, horizon: int, transaction_type: str | None = None
) -> pd.DataFrame:
    data = signals_df[signals_df["horizon_days"] == horizon]
    if transaction_type is not None:
        data = data[data["signal_type"] == transaction_type]
    return data


def _compute_dynamic_prior(signals_df: pd.DataFrame, horizon: int) -> float:
    horizon_signals = _get_horizon_data(signals_df, horizon, TransactionType.PURCHASE.value)
    if horizon_signals.empty:
        return 0.50
    up_prob = (horizon_signals["decayed_return_pct"] > 0).mean()
    return float(np.clip(up_prob, 0.10, 0.90))


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


def _compute_ticker_member_performance(
    signals_df: pd.DataFrame, ticker: str, horizon: int
) -> dict[str, tuple[float, int]]:
    """Per-member win rate on a specific ticker from historical signals.

    Returns {member: (win_rate, trade_count)} for members with >= TICKER_PERF_MIN_TRADES trades.
    """
    if signals_df.empty or "ticker" not in signals_df.columns:
        return {}

    purchases = signals_df[
        (signals_df["ticker"] == ticker)
        & (signals_df["horizon_days"] == horizon)
        & (signals_df["signal_type"] == TransactionType.PURCHASE.value)
    ]
    if purchases.empty:
        return {}

    result: dict[str, tuple[float, int]] = {}
    for member, grp in purchases.groupby("member"):
        returns = grp["decayed_return_pct"].dropna()
        if len(returns) < TICKER_PERF_MIN_TRADES:
            continue
        win_rate = float((returns > 0).mean())
        result[member] = (win_rate, len(returns))
    return result


def _get_top_signals(signals_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    top_data = _get_horizon_data(signals_df, horizon, TransactionType.PURCHASE.value)
    if top_data.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    top_data = _apply_quality_filter(top_data)
    if top_data.empty:
        raise AnalysisError(f"No signals survived quality filter (min price ${MIN_ENTRY_PRICE})")

    top_data = top_data.copy()
    top_data["signal_score"] = (
        top_data["total_spy_alpha_pct"].fillna(0) * CONVICTION_WEIGHT_ALPHA
        + top_data["total_return_pct"].fillna(0) * CONVICTION_WEIGHT_REALIZED
    )

    return top_data.nlargest(top_n, "signal_score")[
        ["member", "ticker", "disclosure_date", "spy_alpha_pct", "peak_potential_pct",
         "total_return_pct", "total_spy_alpha_pct", "signal_score"]
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

    purchases = _apply_quality_filter(purchases)
    if purchases.empty:
        raise AnalysisError(f"No signals survived quality filter for {member}")

    purchases = purchases.copy()
    purchases["signal_score"] = (
        purchases["total_spy_alpha_pct"].fillna(0) * CONVICTION_WEIGHT_ALPHA
        + purchases["total_return_pct"].fillna(0) * CONVICTION_WEIGHT_REALIZED
    )

    return purchases.nlargest(top_n, "signal_score")[
        ["ticker", "disclosure_date", "spy_alpha_pct", "peak_potential_pct", "total_return_pct", "total_spy_alpha_pct", "signal_score"]
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

    first_price_idx = windowed.groupby("signal_id")["price_date"].idxmin()
    disclosure_prices = windowed.loc[first_price_idx].set_index("signal_id")["price"]
    windowed["disclosure_baseline"] = windowed["signal_id"].map(disclosure_prices)

    windowed["days_from_disclosure"] = (windowed["price_date"] - windowed["disclosure_date"]).dt.days
    windowed["decay_factor"] = np.exp(-decay_lambda * windowed["days_from_disclosure"])
    windowed["weighted_return"] = (windowed["price"] / windowed["disclosure_baseline"] - 1) * windowed["decay_factor"]

    if not spy_prices.empty:
        windowed = windowed.merge(spy_prices, on="price_date", how="left")
        first_spy_idx = windowed.dropna(subset=["spy_price"]).groupby("signal_id")["price_date"].idxmin()
        spy_entry_prices = windowed.loc[first_spy_idx].set_index("signal_id")["spy_price"]
        windowed["spy_return"] = windowed["spy_price"] / windowed["signal_id"].map(spy_entry_prices) - 1
    else:
        windowed["spy_return"] = 0.0

    windowed["weighted_spy_return"] = windowed["spy_return"] * windowed["decay_factor"]
    windowed["spy_decay_factor"] = windowed["decay_factor"].where(windowed["spy_return"].notna())

    agg = windowed.groupby("signal_id").agg(
        peak_price=("price", "max"),
        trough_price=("price", "min"),
        decayed_return=("weighted_return", "sum"),
        spy_cumulative=("weighted_spy_return", lambda values: values.sum(min_count=1)),
        spy_weight_sum=("spy_decay_factor", lambda values: values.sum(min_count=1)),
        weight_sum=("decay_factor", "sum"),
        disclosure_price_first=("disclosure_baseline", "first"),
        last_price=("price", "last")
    )
    agg["decayed_return"] = agg["decayed_return"] / agg["weight_sum"]
    agg["spy_cumulative"] = agg["spy_cumulative"] / agg["spy_weight_sum"]
    agg["total_return"] = (agg["last_price"] / agg["disclosure_price_first"] - 1)
    agg = agg.reset_index()

    final = signals.merge(
        agg[["signal_id", "peak_price", "trough_price", "decayed_return", "spy_cumulative", "total_return", "disclosure_price_first"]],
        on="signal_id", how="left"
    )

    is_purchase = final["transaction_type"] == TransactionType.PURCHASE.value
    has_disclosure_price = final["disclosure_price_first"].notna() & (final["disclosure_price_first"] != 0)
    purchase_mask = is_purchase & has_disclosure_price
    sale_mask = ~is_purchase & (final["trough_price"].notna()) & (final["trough_price"] != 0)

    peak_potential = np.zeros(len(final))
    peak_potential[purchase_mask.values] = (
        (final.loc[purchase_mask.values, "peak_price"] / final.loc[purchase_mask.values, "disclosure_price_first"] - 1) * 100
    ).values
    peak_potential[sale_mask.values] = (
        (final.loc[sale_mask.values, "entry_price"] / final.loc[sale_mask.values, "trough_price"] - 1) * 100
    ).values

    optional_columns = [column for column in ["owner_code", "amount_midpoint"] if column in final.columns]
    result_columns = [
        "member", "ticker", "disclosure_date", "signal_type", "horizon_days", "entry_price",
        "peak_potential_pct", "decayed_return_pct", "spy_alpha_pct", "total_return_pct",
        "total_spy_alpha_pct", *optional_columns,
    ]

    return final.assign(
        signal_type=final["transaction_type"],
        peak_potential_pct=peak_potential,
        decayed_return_pct=final["decayed_return"].values * 100,
        spy_alpha_pct=(final["decayed_return"] - final["spy_cumulative"]).values * 100,
        total_return_pct=final["total_return"].values * 100,
        total_spy_alpha_pct=(final["total_return"] - final["spy_cumulative"]).values * 100,
    )[result_columns]


def _assign_episode_ids(group_sorted: pd.DataFrame, max_gap_days: int) -> np.ndarray:
    dates = pd.to_datetime(group_sorted["disclosure_date"])
    if len(dates) <= 1:
        return np.zeros(len(dates), dtype=np.int64)
    gaps = dates.diff().dt.days.fillna(0).astype(int)
    return (gaps > max_gap_days).cumsum().values.astype(np.int64)


def _collapse_to_episodes(signals_df: pd.DataFrame, max_gap_days: int = 14) -> pd.DataFrame:
    if signals_df.empty:
        return signals_df

    group_cols = ["member", "ticker", "horizon_days", "signal_type"]
    if not all(c in signals_df.columns for c in group_cols):
        return signals_df

    if "disclosure_date" not in signals_df.columns:
        return signals_df

    df = signals_df.copy()
    df = df.sort_values(group_cols + ["disclosure_date"]).reset_index(drop=True)

    episode_ids = np.zeros(len(df), dtype=np.int64)
    for _, group in df.groupby(group_cols, sort=False):
        loc = group.index
        sorted_group = group.sort_values("disclosure_date")
        episode_ids[loc] = _assign_episode_ids(sorted_group, max_gap_days)
    df["_episode_id"] = episode_ids

    if "amount_midpoint" in df.columns:
        df["_weight"] = df["amount_midpoint"].fillna(1.0)
    else:
        df["_weight"] = 1.0

    avg_cols = ["decayed_return_pct", "spy_alpha_pct", "total_return_pct", "total_spy_alpha_pct", "peak_potential_pct"]
    existing_avg_cols = [c for c in avg_cols if c in df.columns]

    for col in existing_avg_cols:
        non_nan = df[col].notna()
        df[f"_wp_{col}"] = np.where(non_nan, df[col] * df["_weight"], 0.0)
        df[f"_ws_{col}"] = np.where(non_nan, df["_weight"], 0.0)

    episode_key = group_cols + ["_episode_id"]

    agg_dict = {
        "episode_count": ("_weight", "count"),
        "_weight_sum": ("_weight", "sum"),
    }

    for col, func in {"disclosure_date": "min", "entry_price": "first", "amount_midpoint": "sum"}.items():
        if col in df.columns:
            agg_dict[col] = (col, func)

    if "owner_code" in df.columns:
        agg_dict["owner_code"] = ("owner_code", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])

    for col in existing_avg_cols:
        agg_dict[col] = (f"_wp_{col}", "sum")
        agg_dict[f"_ws_{col}"] = (f"_ws_{col}", "sum")

    collapsed = df.groupby(episode_key, sort=False).agg(**agg_dict).reset_index()

    for col in existing_avg_cols:
        ws = collapsed[f"_ws_{col}"]
        collapsed[col] = np.where(ws > 0, collapsed[col] / ws, np.nan)
        collapsed = collapsed.drop(columns=[f"_ws_{col}"])

    collapsed = collapsed.drop(columns=["_weight_sum", "_episode_id"])

    orig_cols = [c for c in signals_df.columns if c in collapsed.columns]
    collapsed = collapsed[orig_cols + ["episode_count"]]

    return collapsed


def _compute_member_stats(
    member: str,
    grp: pd.DataFrame,
    market_prior: float,
    threshold: float | None = None,
    invert_returns: bool = False,
) -> dict | None:
    rets = grp["decayed_return_pct"].dropna().values
    if len(rets) == 0:
        return None
    if invert_returns:
        rets = -rets

    median_ret = float(np.median(rets))
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0

    wins = int((rets > 0).sum())
    losses = int(len(rets) - wins)
    p_up = wins / len(rets)
    bayes_win_prob = bayesian_win_probability(wins, losses, market_prior)
    posterior_lift = bayes_win_prob / market_prior
    bayes_factor = bayes_factor_against_market(wins, losses, market_prior)

    spy_alpha_vals = grp["spy_alpha_pct"].dropna().values
    if invert_returns:
        spy_alpha_vals = -spy_alpha_vals
    avg_spy_alpha = float(np.mean(spy_alpha_vals)) if len(spy_alpha_vals) > 0 else 0.0

    total_spy_alpha_vals = grp["total_spy_alpha_pct"].dropna().values if "total_spy_alpha_pct" in grp.columns else np.array([])
    if invert_returns:
        total_spy_alpha_vals = -total_spy_alpha_vals
    avg_total_spy_alpha = float(np.mean(total_spy_alpha_vals)) if len(total_spy_alpha_vals) > 0 else avg_spy_alpha

    stats = {
        "member": member,
        "median_return_pct": round(median_ret, 2),
        "mean_return_pct": round(mean_ret, 2),
        "trades": len(rets),
        "sharpe_ratio": round(sharpe, 3),
        "prob_up": round(p_up, 3),
        "bayes_win_prob": round(bayes_win_prob, 3),
        "bayes_factor": round(bayes_factor, 3),
        "posterior_lift": round(posterior_lift, 3),
        "avg_spy_alpha_pct": round(avg_spy_alpha, 2),
        "avg_total_spy_alpha_pct": round(avg_total_spy_alpha, 2),
    }
    if threshold is not None:
        stats["peak_hit_rate_pct"] = round((grp["peak_potential_pct"] > threshold).mean() * 100, 2)
        if "total_return_pct" in grp.columns:
            stats["realized_hit_rate_pct"] = round((grp["total_return_pct"] > 0).mean() * 100, 2)
    return stats


def rank_members(signal_df: pd.DataFrame, horizon: int = 90, threshold: float = 5.0) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    purchases = _get_horizon_data(signal_df, horizon, TransactionType.PURCHASE.value)
    if purchases.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    purchases = _apply_quality_filter(purchases)
    if purchases.empty:
        raise AnalysisError(f"No signals survived quality filter (min price ${MIN_ENTRY_PRICE})")

    purchases = _collapse_to_episodes(purchases)

    market_prior = _compute_dynamic_prior(signal_df, horizon)
    member_stats = []
    for member, purchase_grp in purchases.groupby("member"):
        row = _compute_member_stats(member, purchase_grp, market_prior, threshold)
        if row is not None:
            conviction = _conviction_score(purchase_grp)
            row["conviction_score"] = round(conviction, 3)
            member_stats.append(row)

    result = pd.DataFrame(member_stats)
    if result.empty:
        return result

    return result.rename(columns={
        "mean_return_pct": "avg_decay_return_pct",
        "median_return_pct": "median_decay_return_pct",
        "trades": "purchase_trades",
        "prob_up": "prob_up_given_buy",
    }).sort_values("avg_total_spy_alpha_pct", ascending=False)


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
    min_buyers: int = 2,
    ticker_perf_signals: pd.DataFrame | None = None,
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

    min_trades = ticker_trades["member"].nunique()
    if min_trades < min_buyers:
        return pd.DataFrame({
            "ticker": [ticker],
            "num_buyers": [min_trades],
            "signal_score": [0.0],
            "note": [f"Below minimum buyer threshold ({min_buyers})"]
        })

    buyers = ticker_trades["member"].unique()
    buyer_stats = member_rankings[member_rankings["member"].isin(buyers)].sort_values(
        "avg_spy_alpha_pct", ascending=False
    )

    if buyer_stats.empty:
        fallback_score = 0.0
        fallback_source = "none"
        perf_signals = ticker_perf_signals if ticker_perf_signals is not None else signals_df
        if not perf_signals.empty and "ticker" in perf_signals.columns:
            ticker_hist = perf_signals[
                (perf_signals["ticker"] == ticker)
                & (perf_signals["signal_type"] == TransactionType.PURCHASE.value)
                & (perf_signals["total_spy_alpha_pct"].notna())
            ]
            if len(ticker_hist) >= 2:
                fallback_score = float(ticker_hist["total_spy_alpha_pct"].mean())
                fallback_source = f"ticker_hist({len(ticker_hist)})"

        return pd.DataFrame({
            "ticker": [ticker],
            "num_buyers": [len(buyers)],
            "buyers": [", ".join(buyers[:3])],
            "signal_score": [round(fallback_score, 2)],
            "fallback_source": [fallback_source],
        })

    best_rank = buyer_stats["avg_spy_alpha_pct"].max()
    total_trades = buyer_stats["purchase_trades"].sum()
    rated_buyers = len(buyer_stats)
    confidence_weights = np.sqrt(buyer_stats["purchase_trades"].clip(lower=1).astype(float).values)
    if "bayes_win_prob" in buyer_stats.columns:
        confidence_weights *= buyer_stats["bayes_win_prob"].fillna(0.55).astype(float).values
    if "disclosure_date" in ticker_trades.columns:
        rated_ticker_trades = ticker_trades[ticker_trades["member"].isin(buyer_stats["member"])]
        latest_disclosure = rated_ticker_trades["disclosure_date"].max()
        member_disclosures = rated_ticker_trades.groupby("member")["disclosure_date"].max()
        days_since = (latest_disclosure - member_disclosures.reindex(buyer_stats["member"])).dt.days.fillna(0).clip(lower=0)
        confidence_weights *= np.exp(-BUYER_RECENCY_DECAY * days_since.values)
    confidence_weight_sum = confidence_weights.sum()
    quality_adjusted_avg = (
        (buyer_stats["avg_spy_alpha_pct"].values * confidence_weights).sum() / confidence_weight_sum
        if confidence_weight_sum > 0
        else 0
    )

    base_signal_score = quality_adjusted_avg
    size_factor = _size_score_factor(ticker_trades)
    owner_factor = _owner_score_factor(ticker_trades)

    ticker_perf = _compute_ticker_member_performance(
        ticker_perf_signals if ticker_perf_signals is not None else signals_df,
        ticker, horizon,
    )
    if ticker_perf:
        member_trade_counts = ticker_trades.groupby("member").size()
        weighted_sum = 0.0
        weight_total = 0.0
        for member in buyer_stats["member"]:
            if member in ticker_perf:
                win_rate, _ = ticker_perf[member]
                n = member_trade_counts.get(member, 1)
                weighted_sum += win_rate * n
                weight_total += n
        ticker_perf_factor = weighted_sum / weight_total if weight_total > 0 else 1.0
    else:
        ticker_perf_factor = 1.0

    signal_score = base_signal_score * size_factor * owner_factor * ticker_perf_factor

    top_buyers = buyer_stats["member"].head(3).tolist()
    buyer_label = f"Top {len(top_buyers)} of {len(buyers)}" if len(buyers) > 3 else f"{len(buyers)}"

    return pd.DataFrame({
        "ticker": [ticker],
        "num_buyers": [len(buyers)],
        "rated_buyers": [rated_buyers],
        "buyer_label": [buyer_label],
        "buyers": [", ".join(top_buyers)],
        "avg_buyer_performance": [round(quality_adjusted_avg, 2)],
        "best_buyer_performance": [round(best_rank, 2)],
        "total_buyer_trades": [int(total_trades)],
        "convergence_factor": [1.0],
        "ticker_perf_factor": [round(ticker_perf_factor, 3)],
        "base_signal_score": [round(base_signal_score, 2)],
        "size_factor": [round(size_factor, 3)],
        "owner_factor": [round(owner_factor, 3)],
        "signal_score": [round(signal_score, 2)],
        "fallback_source": ["member_ranked"],
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
        member_rankings[["member", "avg_spy_alpha_pct", "peak_hit_rate_pct", "purchase_trades"]],
        on="member",
        how="left"
    )

    result = result.sort_values("avg_spy_alpha_pct", ascending=False, na_position="last")
    result["num_purchases"] = result["transaction_date"].apply(len)

    return result[["member", "num_purchases", "transaction_date", "disclosure_date",
                   "avg_spy_alpha_pct", "peak_hit_rate_pct", "purchase_trades"]]


def rank_sales(signal_df: pd.DataFrame, horizon: int = 90) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    sales = _get_horizon_data(signal_df, horizon, TransactionType.SALE.value)
    if sales.empty:
        raise AnalysisError(f"No sale signals found for horizon {horizon}")

    sales = _collapse_to_episodes(sales)

    market_prior = _compute_dynamic_prior(signal_df, horizon)
    member_stats = []
    for member, sale_grp in sales.groupby("member"):
        row = _compute_member_stats(member, sale_grp, market_prior, invert_returns=True)
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


def _price_at_or_before(
    prices_df: pd.DataFrame, ticker: str, target_date: pd.Timestamp
) -> float | None:
    if ticker not in prices_df.columns:
        return None
    series = prices_df[ticker].dropna()
    eligible = series[series.index <= pd.Timestamp(target_date)]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def _price_at_or_near(
    prices_df: pd.DataFrame, ticker: str, target_date: pd.Timestamp,
    tolerance_days: int = 7,
) -> float | None:
    if ticker not in prices_df.columns:
        return None
    series = prices_df[ticker].dropna()
    target = pd.Timestamp(target_date)
    lower = target - pd.Timedelta(days=tolerance_days)
    upper = target + pd.Timedelta(days=tolerance_days)
    window = series[(series.index >= lower) & (series.index <= upper)]
    if window.empty:
        return None
    distances = np.abs(window.index - target)
    return float(window.iloc[distances.argmin()])


def _price_on_or_before(
    prices_df: pd.DataFrame, ticker: str, target_date: pd.Timestamp,
    max_staleness_days: int = 5,
) -> float | None:
    if ticker not in prices_df.columns:
        return None
    series = prices_df[ticker].dropna()
    target = pd.Timestamp(target_date)
    eligible = series[series.index <= target]
    if eligible.empty:
        return None
    price_date = eligible.index[-1]
    staleness = (target - price_date).days
    if staleness > max_staleness_days:
        return None
    return float(eligible.iloc[-1])


def backtest_recommendations(
    signals_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int = 90,
    lookback_days: int = 60,
    min_buyers: int = 2,
    top_n: int = 10,
    threshold: float = 5.0,
    prices_df: pd.DataFrame | None = None,
    training_lookback_days: int | None = None,
) -> pd.DataFrame:
    elapsed_cutoff = as_of_date - pd.Timedelta(days=horizon)
    training = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= elapsed_cutoff)
    ].copy()

    if training_lookback_days is not None:
        training_start = as_of_date - pd.Timedelta(days=training_lookback_days)
        training = training[training["disclosure_date"] >= training_start]

    member_rankings = None
    if not training.empty:
        try:
            member_rankings = rank_members(training, horizon, threshold)
        except AnalysisError:
            member_rankings = None

    if member_rankings is None or member_rankings.empty:
        return pd.DataFrame()

    lookback_start = as_of_date - pd.Timedelta(days=lookback_days)
    recent_mask = (
        (transactions_df["disclosure_date"] >= lookback_start)
        & (transactions_df["disclosure_date"] <= as_of_date)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    )
    recent_trades = transactions_df[recent_mask].copy()

    if recent_trades.empty:
        return pd.DataFrame()

    buyer_counts = recent_trades.groupby("ticker")["member"].nunique()
    candidate_tickers = buyer_counts[buyer_counts >= min_buyers].index.tolist()

    if not candidate_tickers:
        return pd.DataFrame()

    ticker_perf_signals = signals_df[
        (signals_df["horizon_days"] == horizon)
        & (signals_df["disclosure_date"] <= as_of_date)
    ].copy()

    scores = []
    for ticker in candidate_tickers:
        try:
            score = score_ticker_by_buyers(
                ticker, recent_trades, training, horizon, threshold,
                member_rankings, min_buyers,
                ticker_perf_signals=ticker_perf_signals,
            )
            # Compute OU entry value V(0) if prices available
            if prices_df is not None:
                v0 = _compute_ticker_entry_value(
                    ticker, signals_df, prices_df, as_of_date, horizon,
                )
                score["ou_entry_value"] = round(v0, 4) if v0 is not None else None
            scores.append(score)
        except AnalysisError:
            continue

    if not scores:
        return pd.DataFrame()

    result = pd.concat(scores, ignore_index=True)
    result = result.sort_values("signal_score", ascending=False).head(top_n).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def evaluate_backtest(
    recommendations: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    exit_date = as_of_date + pd.Timedelta(days=horizon)

    spy_start = _price_at_or_before(prices_df, "SPY", as_of_date)
    spy_end = _price_on_or_before(prices_df, "SPY", exit_date)
    if not spy_start or not spy_end:
        raise AnalysisError(
            f"SPY price not available for backtest period "
            f"(as_of={as_of_date.date()}, exit={exit_date.date()})"
        )
    spy_return_pct = (spy_end / spy_start - 1) * 100

    rows = []
    for _, rec in recommendations.iterrows():
        ticker = rec["ticker"]
        entry = _price_at_or_before(prices_df, ticker, as_of_date)
        exit_price = _price_on_or_before(prices_df, ticker, exit_date)
        if not entry:
            raise AnalysisError(
                f"No price for {ticker} at/as_of {as_of_date.date()} "
                f"— cannot backtest"
            )
        if not exit_price:
            raise AnalysisError(
                f"No price for {ticker} on/before exit {exit_date.date()} "
                f"(as_of={as_of_date.date()}) — cannot backtest"
            )
        return_pct = (exit_price / entry - 1) * 100
        alpha_pct = return_pct - spy_return_pct
        rows.append({
            "ticker": ticker,
            "bt_entry_price": round(entry, 2),
            "bt_exit_price": round(exit_price, 2),
            "bt_return_pct": round(return_pct, 2),
            "bt_spy_return_pct": round(spy_return_pct, 2),
            "bt_alpha_pct": round(alpha_pct, 2),
        })

    eval_df = pd.DataFrame(rows)
    return recommendations.merge(eval_df, on="ticker", how="left")


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    valid = results.dropna(subset=["bt_return_pct"])
    if valid.empty:
        return pd.DataFrame()

    by_rank = []
    for rank, grp in valid.groupby("rank"):
        by_rank.append({
            "rank": rank,
            "count": len(grp),
            "win_rate_pct": round((grp["bt_return_pct"] > 0).mean() * 100, 1),
            "avg_return_pct": round(grp["bt_return_pct"].mean(), 2),
            "avg_alpha_pct": round(grp["bt_alpha_pct"].mean(), 2) if "bt_alpha_pct" in grp.columns else None,
        })

    summary = pd.DataFrame(by_rank)

    overall = {
        "rank": "ALL",
        "count": len(valid),
        "win_rate_pct": round((valid["bt_return_pct"] > 0).mean() * 100, 1),
        "avg_return_pct": round(valid["bt_return_pct"].mean(), 2),
        "avg_alpha_pct": round(valid["bt_alpha_pct"].mean(), 2) if "bt_alpha_pct" in valid.columns else None,
    }
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return summary

