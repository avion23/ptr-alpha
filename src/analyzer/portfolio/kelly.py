"""Kelly position sizing with explicit, historical outcome inputs.

A Kelly fraction is an absolute fraction of current bankroll.  This module never
normalizes surviving positions back to 100% exposure: unused bankroll stays cash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_SIZING_COLUMNS = frozenset(
    {"ticker", "member", "signal_score", "win_rate", "avg_win_pct", "avg_loss_pct"}
)


@dataclass(frozen=True, slots=True)
class KellyConfig:
    """Kelly bankroll and risk constraints.

    ``crash_guard`` is off by default because backtest recommendations already
    include their crash penalty in ``signal_score``.  Enable it only for raw,
    unadjusted scores.
    """

    capital: float = 100_000.0
    max_ticker_pct: float = 0.20
    max_member_pct: float = 0.05
    total_exposure_pct: float = 1.00
    use_half_kelly: bool = True
    crash_guard: bool = False
    min_kelly_floor: float = 0.01


def kelly_fraction(p: float, b: float) -> float:
    """Return full Kelly fraction ``(p*b - (1-p)) / b``."""
    if not np.isfinite(p) or not np.isfinite(b) or b <= 0 or p <= 0 or p >= 1:
        return 0.0
    q = 1.0 - p
    return max((p * b - q) / b, 0.0)


def half_kelly(p: float, b: float) -> float:
    """Return half of the full Kelly fraction."""
    return kelly_fraction(p, b) / 2.0


def compute_payout_ratio(avg_win: float, avg_loss: float) -> float:
    """Return positive average win divided by positive loss magnitude."""
    if not np.isfinite(avg_win) or not np.isfinite(avg_loss):
        return 0.0
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    return avg_win / avg_loss


def build_kelly_portfolio(
    recommendations: pd.DataFrame,
    config: KellyConfig | None = None,
) -> pd.DataFrame:
    """Size recommendations from explicit historical outcome estimates.

    Required columns are ``ticker``, ``member``, ``signal_score``, ``win_rate``,
    ``avg_win_pct``, and ``avg_loss_pct``.  ``avg_loss_pct`` is a positive loss
    magnitude.  Missing, placeholder, or non-finite inputs abstain rather than
    substituting optimistic defaults.

    Each valid row starts at its absolute full- or half-Kelly bankroll fraction.
    Group and exposure caps only reduce those fractions; they never redistribute
    freed exposure.
    """
    config = config or KellyConfig()
    _validate_config(config)
    df = _validated_recommendations(recommendations)
    if df.empty:
        return _empty_portfolio()

    df["kelly_fraction"] = df.apply(
        lambda row: _row_kelly_fraction(row, config), axis=1
    )
    df = df[df["kelly_fraction"] > config.min_kelly_floor].copy()
    if df.empty:
        return _empty_portfolio()

    # Kelly is already a bankroll fraction.  Signal score ranks opportunities;
    # it must not inflate or renormalize their absolute risk budget.
    df["weight"] = df["kelly_fraction"]
    _scale_group_cap(df, "ticker", config.max_ticker_pct)
    _scale_group_cap(df, "member", config.max_member_pct)
    _scale_total_exposure(df, config.total_exposure_pct)
    df["position_value"] = df["weight"] * config.capital
    return _select_output_columns(df)


def _validate_config(config: KellyConfig) -> None:
    if not np.isfinite(config.capital) or config.capital <= 0:
        raise ValueError("capital must be positive and finite")
    for name in ("max_ticker_pct", "max_member_pct", "total_exposure_pct"):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0 or value > 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if not np.isfinite(config.min_kelly_floor) or config.min_kelly_floor < 0:
        raise ValueError("min_kelly_floor must be non-negative and finite")


def _validated_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty or not REQUIRED_SIZING_COLUMNS.issubset(recommendations.columns):
        return pd.DataFrame()

    df = recommendations.copy()
    numeric = ["signal_score", "win_rate", "avg_win_pct", "avg_loss_pct"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    ticker = df["ticker"].astype("string").str.strip()
    member = df["member"].astype("string").str.strip()
    placeholder_member = member.str.lower().isin({"unknown", "none", "nan"})
    finite = np.isfinite(df[numeric]).all(axis=1)
    valid = (
        ticker.notna()
        & ticker.ne("")
        & member.notna()
        & member.ne("")
        & ~placeholder_member
        & finite
        & df["signal_score"].gt(0)
        & df["win_rate"].gt(0)
        & df["win_rate"].lt(1)
        & df["avg_win_pct"].gt(0)
        & df["avg_loss_pct"].gt(0)
    )
    df = df.loc[valid].copy()
    if df.empty:
        return df
    df["ticker"] = ticker.loc[valid]
    df["member"] = member.loc[valid]
    if "crash_prob" not in df.columns:
        df["crash_prob"] = 0.0
    else:
        df["crash_prob"] = pd.to_numeric(df["crash_prob"], errors="coerce").fillna(0.0)
    return df


def _row_kelly_fraction(row: pd.Series, config: KellyConfig) -> float:
    payout = compute_payout_ratio(row["avg_win_pct"], row["avg_loss_pct"])
    fraction = (
        half_kelly(row["win_rate"], payout)
        if config.use_half_kelly
        else kelly_fraction(row["win_rate"], payout)
    )
    if config.crash_guard:
        fraction *= 1.0 - float(np.clip(row["crash_prob"], 0.0, 0.95))
    return fraction


def _scale_group_cap(df: pd.DataFrame, column: str, cap: float) -> None:
    totals = df.groupby(column, sort=False)["weight"].transform("sum")
    scale = np.minimum(1.0, cap / totals)
    df["weight"] *= scale


def _scale_total_exposure(df: pd.DataFrame, cap: float) -> None:
    total = float(df["weight"].sum())
    if total > cap:
        df["weight"] *= cap / total


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker", "member", "weight", "kelly_fraction",
        "signal_score", "crash_prob", "win_rate",
        "avg_win_pct", "avg_loss_pct", "position_value",
    ]
    return (
        df[columns]
        .sort_values(["position_value", "signal_score"], ascending=False)
        .reset_index(drop=True)
    )


def _empty_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker", "member", "weight", "kelly_fraction",
            "signal_score", "crash_prob", "win_rate",
            "avg_win_pct", "avg_loss_pct", "position_value",
        ]
    )
