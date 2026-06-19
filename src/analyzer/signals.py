"""Signal generation and calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer.exceptions import AnalysisError
from analyzer.models import TransactionType
from analyzer.ticker_resolver import TickerResolver

DECAY_LAMBDA = 0.005
POSITION_SIZE_BASELINE = 10000.0
MAX_DISCLOSURE_METADATA_ADJUSTMENT = 0.15
BAYES_PRIOR_STRENGTH = 20.0
BUYER_RECENCY_DECAY = 0.03
TICKER_PERF_MIN_TRADES = 3

MIN_ENTRY_PRICE = 3.0
# Use pure SPY alpha as signal score — avoids double-counting stock return
CONVICTION_WEIGHT_ALPHA = 1.0
CONVICTION_WEIGHT_REALIZED = 0.0


def _price_at_or_before(
    prices_df: pd.DataFrame,
    ticker: str,
    target_date: pd.Timestamp,
    max_staleness_days: int | None = None,
) -> float | None:
    if ticker not in prices_df.columns:
        return None
    series = prices_df[ticker].dropna()
    target = pd.Timestamp(target_date)
    eligible = series[series.index <= target]
    if eligible.empty:
        return None
    price_date = eligible.index[-1]
    if max_staleness_days is not None and (target - price_date).days > max_staleness_days:
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


def _get_horizon_data(
    signals_df: pd.DataFrame, horizon: int, transaction_type: str | None = None
) -> pd.DataFrame:
    data = signals_df[signals_df["horizon_days"] == horizon]
    if transaction_type is not None:
        data = data[data["signal_type"] == transaction_type]
    return data


def _apply_quality_filter(signals_df: pd.DataFrame) -> pd.DataFrame:
    if "entry_price" not in signals_df.columns:
        return signals_df
    return signals_df[signals_df["entry_price"] >= MIN_ENTRY_PRICE].copy()


def _compute_dynamic_prior(signals_df: pd.DataFrame, horizon: int) -> float:
    horizon_signals = _get_horizon_data(signals_df, horizon, TransactionType.PURCHASE.value)
    if horizon_signals.empty:
        return 0.50
    up_prob = (horizon_signals["decayed_return_pct"] > 0).mean()
    return float(np.clip(up_prob, 0.10, 0.90))


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

    # Resolve tickers so the merge matches prices_df columns
    resolver = TickerResolver()
    price_tickers = set(prices_long["ticker"].unique())
    raw_tickers = signals["ticker"].unique()
    for raw in raw_tickers:
        if raw not in price_tickers:
            resolved = resolver.resolve(raw)
            if resolved.price_symbol in price_tickers:
                signals.loc[signals["ticker"] == raw, "ticker"] = resolved.price_symbol

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

    # Incremental log-returns instead of cumulative returns
    # This avoids double-counting: each day's contribution is independent
    windowed = windowed.sort_values(["signal_id", "price_date"])
    windowed["prev_price"] = windowed.groupby("signal_id")["price"].shift(1)
    windowed["prev_days"] = windowed.groupby("signal_id")["days_from_disclosure"].shift(1)
    # First observation per signal has no prior price — use 0 return
    is_first = windowed["prev_price"].isna() | (windowed["prev_price"] == 0)
    windowed["daily_log_return"] = np.where(
        is_first, 0.0,
        np.log(windowed["price"] / windowed["prev_price"])
    )
    # Adjust decay: weight by the midpoint between consecutive days
    windowed["mid_decay"] = np.exp(-decay_lambda * (windowed["days_from_disclosure"] + windowed["prev_days"].fillna(0)) / 2)
    windowed["weighted_return"] = windowed["daily_log_return"] * windowed["mid_decay"]

    if not spy_prices.empty:
        windowed = windowed.merge(spy_prices, on="price_date", how="left")
        # Incremental SPY log-returns
        windowed["prev_spy"] = windowed.groupby("signal_id")["spy_price"].shift(1)
        windowed["spy_daily_return"] = np.where(
            windowed["prev_spy"].isna() | (windowed["prev_spy"] == 0), 0.0,
            np.log(windowed["spy_price"] / windowed["prev_spy"])
        )
        windowed["weighted_spy_return"] = windowed["spy_daily_return"] * windowed["mid_decay"]
        windowed["spy_decay_factor"] = windowed["mid_decay"].where(windowed["spy_daily_return"].notna())
    else:
        windowed["weighted_spy_return"] = 0.0
        windowed["spy_price"] = np.nan
        windowed["spy_decay_factor"] = windowed["decay_factor"]

    agg = windowed.groupby("signal_id").agg(
        peak_price=("price", "max"),
        trough_price=("price", "min"),
        decayed_return=("weighted_return", "sum"),
        spy_cumulative=("weighted_spy_return", lambda values: values.sum(min_count=1)),
        spy_weight_sum=("spy_decay_factor", lambda values: values.sum(min_count=1)),
        weight_sum=("decay_factor", "sum"),
        disclosure_price_first=("disclosure_baseline", "first"),
        last_price=("price", "last"),
        spy_first_price=("spy_price", lambda v: v.dropna().iloc[0] if not v.dropna().empty else np.nan),
        spy_last_price=("spy_price", lambda v: v.dropna().iloc[-1] if not v.dropna().empty else np.nan),
    )
    # Normalize by weight sum for proper decay-weighted average
    agg["decayed_return"] = agg["decayed_return"] / agg["weight_sum"]
    agg["spy_cumulative"] = np.where(
        agg["spy_weight_sum"] > 0,
        agg["spy_cumulative"] / agg["spy_weight_sum"],
        0.0,
    )
    agg["total_return"] = (agg["last_price"] / agg["disclosure_price_first"] - 1)
    agg["actual_spy_return"] = np.where(
        agg["spy_first_price"].notna() & (agg["spy_first_price"] != 0),
        agg["spy_last_price"] / agg["spy_first_price"] - 1,
        0.0,
    )
    agg = agg.reset_index()

    final = signals.merge(
        agg[["signal_id", "peak_price", "trough_price", "decayed_return", "spy_cumulative", "actual_spy_return", "total_return", "disclosure_price_first"]],
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
        "total_spy_alpha_pct", "decayed_spy_return_pct", *optional_columns,
    ]

    return final.assign(
        signal_type=final["transaction_type"],
        peak_potential_pct=peak_potential,
        decayed_return_pct=final["decayed_return"].values * 100,
        spy_alpha_pct=(final["decayed_return"] - final["spy_cumulative"]).values * 100,
        total_return_pct=final["total_return"].values * 100,
        total_spy_alpha_pct=(final["total_return"] - final["actual_spy_return"]).values * 100,
        decayed_spy_return_pct=final["spy_cumulative"].values * 100,
    )[result_columns]


def get_top_signals(signal_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_top_signals(signal_df, horizon, top_n)


def get_member_signals(signal_df: pd.DataFrame, member: str, horizon: int = 90, top_n: int = 5) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signals dataframe")
    return _get_member_signals(signal_df, member, horizon, top_n)
