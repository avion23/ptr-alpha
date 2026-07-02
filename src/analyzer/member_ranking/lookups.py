"""Lookup and dict helpers for member rankings.

O(1) data structures derived from member_rankings to replace repeated
DataFrame linear scans in hot paths (per-ticker scoring loops, buyer
lookups).

`get_ticker_buyers_with_rankings` joins per-ticker buyer history with the
member_rankings table for diagnostic / display purposes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyzer._memo import df_memoize
from analyzer.exceptions import AnalysisError
from analyzer.member_names import canonical_member_key
from analyzer.models import TransactionType


@df_memoize(copy=False)
def _get_ticker_purchases(ticker: str, transactions_df: pd.DataFrame) -> pd.DataFrame:
    return transactions_df[
        (transactions_df["ticker"] == ticker)
        & (transactions_df["transaction_type"] == TransactionType.PURCHASE.value)
    ]


def _lookup_buyer_bayes_win_prob(
    member: str,
    member_rankings: pd.DataFrame | None,
) -> float | None:
    """Fetch a member's Bayesian win probability from member_rankings.

    Returns None when rankings are missing, the column is absent, the member
    is unrated, or the value is NaN.

    Tries exact match first, then canonical-key match to handle name variants
    (e.g. 'MICHAEL MCCAUL' vs 'MICHAEL T. MCCAUL').
    """
    if member_rankings is None or member_rankings.empty:
        return None
    if "bayes_win_prob" not in member_rankings.columns:
        return None
    row = member_rankings.loc[member_rankings["member"] == member]
    if row.empty:
        # Fix 7: fall back to canonical key match for name variants
        canon = canonical_member_key(member)
        row = member_rankings.loc[
            member_rankings["member"].apply(canonical_member_key) == canon
        ]
    if row.empty:
        return None
    val = row["bayes_win_prob"].iloc[0]
    return float(val) if pd.notna(val) else None


def _build_buyer_bayes_dict(member_rankings: pd.DataFrame | None) -> dict[str, float]:
    """Precompute {member: bayes_win_prob} dict for O(1) lookups.

    Replaces repeated linear scans of member_rankings DataFrame.

    Fix 7: also adds canonical-key entries so lookups work regardless of which
    name variant a transaction uses (e.g. 'MICHAEL MCCAUL' and 'MICHAEL T. MCCAUL'
    both resolve to the same ranking row).
    """
    if member_rankings is None or member_rankings.empty:
        return {}
    if "bayes_win_prob" not in member_rankings.columns:
        return {}
    # Vectorized — no iterrows
    valid = member_rankings["bayes_win_prob"].notna()
    subset = member_rankings.loc[valid, ["member", "bayes_win_prob"]]
    result: dict[str, float] = dict(zip(subset["member"], subset["bayes_win_prob"].astype(float)))
    # Add canonical-key aliases so any name variant hits the same entry
    aliases: dict[str, float] = {}
    for name, val in result.items():
        key = canonical_member_key(name)
        if key not in result and key not in aliases:
            aliases[key] = val
    result.update(aliases)
    return result


def _build_ranking_dicts(
    member_rankings: pd.DataFrame | None,
    scoring_mode: str = "shrunk_alpha",
) -> dict:
    """Pre-build O(1) lookup dicts from member_rankings DataFrame.

    Returns {"alpha": {member: float}, "trades": {member: int}, "prob": {member: float}}.
    Avoids repeated DataFrame linear scans in the per-ticker scoring loop.

    scoring_mode controls how member scores are computed:
      - "shrunk_alpha": Bayesian-shrunk historical SPY alpha (default)
      - "consistency": prob_up * log(1 + trades) — continuous, differentiable
      - "bayesian_quality": bayes_win_prob * shrunk_alpha
      - "trade_frequency": log(1 + trades)
    """
    if member_rankings is None or member_rankings.empty:
        return {"alpha": {}, "trades": {}, "prob": {}, "has_shrunk": False}

    has_shrunk = "shrunk_alpha" in member_rankings.columns
    alpha_col = "shrunk_alpha" if has_shrunk else "avg_spy_alpha_pct"

    cols = ["member", alpha_col, "purchase_trades"]
    if "bayes_win_prob" in member_rankings.columns:
        cols.append("bayes_win_prob")
    if "prob_up_given_buy" in member_rankings.columns:
        cols.append("prob_up_given_buy")
    valid = member_rankings[cols].dropna(subset=["member"])

    alpha = _compute_alpha_for_scoring_mode(valid, alpha_col, scoring_mode)
    trades_dict = dict(zip(valid["member"], valid["purchase_trades"].fillna(0).astype(int)))
    prob = (
        dict(zip(valid["member"], valid["bayes_win_prob"].fillna(0.5).astype(float)))
        if "bayes_win_prob" in valid.columns
        else {}
    )

    # Fix 7: add canonical-key aliases to all lookup dicts so any name variant
    # (e.g. 'MICHAEL MCCAUL' vs 'MICHAEL T. MCCAUL') resolves to the same entry.
    # Collision note: when two genuinely different members collapse to the same
    # canonical key (rare but possible), the last writer wins for the alias entry.
    # Exact original-name keys always take precedence (the `if canonical_member_key(k) not in d`
    # guard prevents overwriting them); collisions only affect fallback lookups for
    # members not in the rankings under their exact name.
    def _add_canonical_aliases(d: dict) -> dict:
        aliases = {canonical_member_key(k): v for k, v in d.items() if canonical_member_key(k) not in d}
        d.update(aliases)
        return d

    _add_canonical_aliases(alpha)
    _add_canonical_aliases(trades_dict)
    _add_canonical_aliases(prob)

    return {"alpha": alpha, "trades": trades_dict, "prob": prob, "has_shrunk": has_shrunk}


def _compute_alpha_for_scoring_mode(valid: pd.DataFrame, alpha_col: str, scoring_mode: str) -> dict:
    if scoring_mode == "consistency":
        prob_up = valid["prob_up_given_buy"].fillna(0.5).values if "prob_up_given_buy" in valid.columns else np.full(len(valid), 0.5)
        trades = valid["purchase_trades"].fillna(0).values.astype(float)
        alpha_values = prob_up * np.log1p(trades)
    elif scoring_mode == "bayesian_quality":
        bayes = valid["bayes_win_prob"].fillna(0.5).values if "bayes_win_prob" in valid.columns else np.full(len(valid), 0.5)
        raw_alpha = valid[alpha_col].fillna(0.0).values.astype(float)
        alpha_values = bayes * raw_alpha
    elif scoring_mode == "trade_frequency":
        trades = valid["purchase_trades"].fillna(0).values.astype(float)
        alpha_values = np.log1p(trades)
    else:  # "shrunk_alpha" (default)
        alpha_values = valid[alpha_col].astype(float).values
    return dict(zip(valid["member"], alpha_values))


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

    from analyzer.member_ranking.ranking import rank_members

    member_rankings = rank_members(signals_df, horizon, threshold)
    ticker_trades = _get_ticker_purchases(ticker, transactions_df)
    if ticker_trades.empty:
        raise AnalysisError(f"No purchases found for {ticker}")

    buyers_with_dates = ticker_trades.groupby("member").agg({
        "transaction_date": list,
        "disclosure_date": list
    }).reset_index()

    ranking_cols = ["member", "avg_spy_alpha_pct", "purchase_trades"]
    if "peak_hit_rate_pct" in member_rankings.columns:
        ranking_cols.append("peak_hit_rate_pct")
    result = pd.merge(
        buyers_with_dates,
        member_rankings[ranking_cols],
        on="member",
        how="left"
    )
    result = result.sort_values("avg_spy_alpha_pct", ascending=False, na_position="last")
    result["num_purchases"] = result["transaction_date"].apply(len)
    return_cols = ["member", "num_purchases", "transaction_date", "disclosure_date",
                   "avg_spy_alpha_pct", "purchase_trades"]
    if "peak_hit_rate_pct" in result.columns:
        return_cols.insert(4, "peak_hit_rate_pct")
    return result[return_cols]
