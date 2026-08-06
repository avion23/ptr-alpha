import pandas as pd

from analyzer.member_ranking.buyer_scoring import score_ticker_by_buyers


def test_final_result_emits_unrounded_solo_adjusted_score():
    transactions = pd.DataFrame(
        {
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2024-05-14"]),
            "transaction_type": ["Purchase"],
        }
    )
    signals = pd.DataFrame({"member": ["Alice"]})
    rankings = pd.DataFrame(
        {
            "member": ["Alice"],
            "avg_spy_alpha_pct": [1.2345],
            "purchase_trades": [5],
            "bayes_win_prob": [0.9],
        }
    )

    result = score_ticker_by_buyers(
        "AAPL",
        transactions,
        signals,
        member_rankings=rankings,
        min_buyers=1,
        solo_buyer_skill_threshold=0.6,
        solo_buyer_penalty=0.8,
    )
    row = result.iloc[0]

    assert bool(row["solo_buyer"])
    assert row["signal_score_raw"] == 0.9876
    assert row["signal_score"] == 0.99
