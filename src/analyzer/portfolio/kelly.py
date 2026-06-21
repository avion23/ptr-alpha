"""Kelly criterion math: f* = (p*b - q) / b, half-Kelly wrapper, payout ratio.

Public API (re-exported from `analyzer.portfolio`):
  - KellyConfig           position-sizing parameters
  - kelly_fraction, half_kelly
  - compute_payout_ratio
  - build_kelly_portfolio  per-ticker weight assignment
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KellyConfig:
    """Configuration for Kelly portfolio construction."""
    capital: float = 100_000.0
    max_ticker_pct: float = 0.20       # max 20% per ticker
    max_member_pct: float = 0.05       # max 5% per member
    total_exposure_pct: float = 1.00    # max 100% invested
    use_half_kelly: bool = True         # safer half-Kelly by default
    crash_guard: bool = True            # reduce by crash_prob
    min_kelly_floor: float = 0.01       # minimum fraction to include
    default_win_rate: float = 0.575     # from sweep results
    default_avg_win: float = 0.015      # avg winning trade return (1.5%)
    default_avg_loss: float = 0.012     # avg losing trade return (1.2%)


def kelly_fraction(p: float, b: float) -> float:
    """Compute full Kelly fraction: f* = (p*b - q) / b.

    Args:
        p: win probability (0, 1)
        b: payout ratio (avg_win / avg_loss), must be > 0

    Returns:
        Kelly fraction (may be negative = don't bet)
    """
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    q = 1.0 - p
    f = (p * b - q) / b
    return max(f, 0.0)


def half_kelly(p: float, b: float) -> float:
    """Half-Kelly fraction: f*/2 (safer, less volatile)."""
    return kelly_fraction(p, b) / 2.0


def compute_payout_ratio(avg_win: float, avg_loss: float) -> float:
    """Payout ratio b = avg_win / avg_loss. Must have avg_loss > 0."""
    if avg_loss <= 0:
        return 0.0
    return avg_win / avg_loss


def build_kelly_portfolio(
    recommendations: pd.DataFrame,
    config: KellyConfig | None = None,
) -> pd.DataFrame:
    """Build portfolio from backtest recommendations using Kelly sizing.

    Args:
        recommendations: DataFrame from backtest_recommendations() with columns
            ticker, signal_score, crash_prob, member (optional).
        config: KellyConfig with sizing parameters.

    Returns:
        DataFrame with columns: ticker, member, weight, kelly_fraction,
            signal_score, crash_prob, position_value.
    """
    if config is None:
        config = KellyConfig()

    if recommendations.empty:
        return _empty_portfolio()

    df = _prepare_recommendations(recommendations, config)
    avg_win, avg_loss = _estimate_win_loss(df, config)
    payout_ratio = compute_payout_ratio(avg_win, avg_loss)

    df["kelly_fraction"] = _compute_kelly_per_row(df, payout_ratio, config)
    if config.crash_guard:
        df["kelly_fraction"] = df["kelly_fraction"] * (1.0 - df["crash_prob"].clip(0, 0.95))

    df = df[df["kelly_fraction"] > config.min_kelly_floor].copy()
    if df.empty:
        return _empty_portfolio()

    df = _apply_signal_score_weights(df)
    if df.empty:
        return _empty_portfolio()

    df = _apply_risk_constraints(df, config)
    df["position_value"] = df["weight"] * config.capital

    return _select_output_columns(df)


def _prepare_recommendations(recommendations: pd.DataFrame, config: KellyConfig) -> pd.DataFrame:
    """Copy and fill in default values for member/crash_prob/win_rate columns."""
    df = recommendations.copy()

    if "member" not in df.columns:
        df["member"] = "unknown"
    if "crash_prob" not in df.columns:
        df["crash_prob"] = 0.0

    if "win_rate" in df.columns:
        df["_win_rate"] = df["win_rate"].clip(0.01, 0.99)
    else:
        df["_win_rate"] = config.default_win_rate
    return df


def _estimate_win_loss(df: pd.DataFrame, config: KellyConfig) -> tuple[float, float]:
    """Pull avg winning/losing return from the recommendations DataFrame.

    Prefers `avg_return_pct` (already-summarized), falls back to per-trade
    `bt_return_pct`, then to the config defaults (converted to percent).
    """
    if "avg_return_pct" in df.columns:
        avg_win = df.loc[df["avg_return_pct"] > 0, "avg_return_pct"].mean()
        avg_loss = abs(df.loc[df["avg_return_pct"] < 0, "avg_return_pct"].mean())
    elif "bt_return_pct" in df.columns:
        avg_win = df.loc[df["bt_return_pct"] > 0, "bt_return_pct"].mean()
        avg_loss = abs(df.loc[df["bt_return_pct"] < 0, "bt_return_pct"].mean())
    else:
        return config.default_avg_win * 100, config.default_avg_loss * 100

    if not avg_win or avg_win <= 0:
        avg_win = config.default_avg_win * 100
    if not avg_loss or avg_loss <= 0:
        avg_loss = config.default_avg_loss * 100
    return float(avg_win), float(avg_loss)


def _compute_kelly_per_row(df: pd.DataFrame, payout_ratio: float, config: KellyConfig) -> pd.Series:
    """Compute Kelly fraction per row (vectorized: loop once with .apply)."""
    def _kelly(row):
        p = row["_win_rate"]
        return (
            half_kelly(p, payout_ratio)
            if config.use_half_kelly
            else kelly_fraction(p, payout_ratio)
        )
    return df.apply(_kelly, axis=1)


def _apply_signal_score_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Initial weight = signal_score * kelly_fraction, normalized to 1.0."""
    raw_weight = df["signal_score"] * df["kelly_fraction"]
    total_raw = raw_weight.sum()
    if total_raw <= 0:
        return pd.DataFrame()
    df["weight"] = raw_weight / total_raw
    return df


def _apply_risk_constraints(df: pd.DataFrame, config: KellyConfig) -> pd.DataFrame:
    """Enforce per-ticker and per-member position limits.

    Two-phase approach:
    1. Clip + normalize until stable (distributes freed weight).
    2. Final hard-clip pass (no normalization) to guarantee caps are met.
       Total weight may end up < 1.0 (cash buffer).
    """
    for _ in range(20):
        prev = df["weight"].copy()
        _normalize_and_clip(df, config)
        if np.allclose(df["weight"].values, prev.values, atol=1e-8):
            break
    _hard_clip_final(df, config)
    _scale_to_exposure(df, config)
    return df


def _normalize_and_clip(df: pd.DataFrame, config: KellyConfig) -> None:
    """One pass: normalize, per-ticker cap, per-member cap (in-place)."""
    total = df["weight"].sum()
    if total > 0:
        df["weight"] = df["weight"] / total
    df["weight"] = df["weight"].clip(upper=config.max_ticker_pct)
    if "member" in df.columns:
        for _member, grp in df.groupby("member"):
            total_member = grp["weight"].sum()
            if total_member > config.max_member_pct:
                scale = config.max_member_pct / total_member
                df.loc[grp.index, "weight"] = df.loc[grp.index, "weight"] * scale


def _hard_clip_final(df: pd.DataFrame, config: KellyConfig) -> None:
    """Final pass: hard-clip without re-normalizing (caps guaranteed)."""
    df["weight"] = df["weight"].clip(upper=config.max_ticker_pct)
    if "member" in df.columns:
        for _member, grp in df.groupby("member"):
            total_member = grp["weight"].sum()
            if total_member > config.max_member_pct:
                scale = config.max_member_pct / total_member
                df.loc[grp.index, "weight"] = df.loc[grp.index, "weight"] * scale


def _scale_to_exposure(df: pd.DataFrame, config: KellyConfig) -> None:
    """Scale total weight down to max_exposure if it's over."""
    total_weight = df["weight"].sum()
    if total_weight > config.total_exposure_pct:
        df["weight"] = df["weight"] * (config.total_exposure_pct / total_weight)


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Project to the public output schema and sort by dollar position."""
    out_cols = [
        "ticker", "member", "weight", "kelly_fraction", "signal_score",
        "crash_prob", "position_value",
    ]
    available = [c for c in out_cols if c in df.columns]
    return df[available].sort_values("position_value", ascending=False).reset_index(drop=True)


def _empty_portfolio() -> pd.DataFrame:
    """Return empty DataFrame with expected columns."""
    return pd.DataFrame(columns=[
        "ticker", "member", "weight", "kelly_fraction",
        "signal_score", "crash_prob", "position_value",
    ])
