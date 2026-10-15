from datetime import date

import pandas as pd

from analyzer.positions import build_positions
from analyzer.setups import score


def test_same_day_fifo_decision_is_byte_identical_across_five_input_orders():
    # Old date-only stable sorting alternates between a closed position and
    # a new $200 lot depending on the database's arbitrary row order.
    trades = pd.DataFrame([
        dict(member="M", ticker="T", transaction_type=kind,
             transaction_date=date(2026, 1, day), disclosure_date=date(2026, 1, day),
             amount_midpoint=amount, source="house_pdf", source_record_id=filing)
        for kind, day, amount, filing in [
            ("Purchase", 2, 1000., "a"),
            ("Purchase", 5, 2000., "c"),
            ("Sale (Full)", 5, 1000., "b"),
        ]
    ])
    history = {"T": pd.DataFrame({"close": [100., 200., 200.]},
                                 index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))}
    outputs = []
    for seed in range(5):
        positions = build_positions(trades.sample(frac=1, random_state=seed), history,
                                    as_of=date(2026, 1, 6))
        assert len(positions) == 1
        candidates = pd.DataFrame([dict(
            ticker="T", actor_id="congress:M", kind="congress", source="house_pdf",
            entry_ref=positions.iloc[0].cost_basis, as_of=date(2026, 1, 6),
            disclosure_date=date(2026, 1, 5), corroboration=False, position_evidence=True,
        )])
        ranked, blocked = score(candidates, {"congress:M": 1.}, {"T": 200.})
        assert not ranked.empty
        outputs.append((positions.to_csv(index=False) + ranked.to_csv(index=False)
                        + blocked.to_csv(index=False)).encode())
    assert len(set(outputs)) == 1
