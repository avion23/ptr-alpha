import numpy as np
import pandas as pd

from analyzer.analysis import calculate_signal_potential


def test_ticker_gap_does_not_use_stale_last_price_for_incomplete_horizon():
    disclosure = pd.Timestamp.now().normalize() - pd.Timedelta(days=120)
    horizon_ends = [
        disclosure + pd.Timedelta(days=80),
        disclosure + pd.Timedelta(days=90),
    ]

    price_dates = pd.DatetimeIndex(
        [
            disclosure,
            horizon_ends[0],
            horizon_ends[1],
            disclosure + pd.Timedelta(days=95),
        ]
    )
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, np.nan, 200.0],
            "SPY": [400.0, 400.0, 400.0, 400.0],
        },
        index=price_dates,
    )
    entries = pd.DataFrame(
        {
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": [disclosure],
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        }
    )

    result = calculate_signal_potential(entries, prices, horizons=[80, 90])
    assert len(result) == 2
    by_horizon = result.set_index("horizon_days")

    assert bool(by_horizon.loc[80, "window_complete"])
    assert np.isclose(by_horizon.loc[80, "total_return_pct"], 10.0)
    assert np.isnan(by_horizon.loc[90, "total_return_pct"])
    for column in (
        "peak_potential_pct",
        "decayed_return_pct",
        "spy_alpha_pct",
        "total_spy_alpha_pct",
        "decayed_spy_return_pct",
    ):
        assert np.isnan(by_horizon.loc[90, column]), column
    assert not bool(by_horizon.loc[90, "window_complete"])


def test_quote_before_disclosure_does_not_complete_empty_window():
    disclosure = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    price_dates = pd.DatetimeIndex(
        [
            disclosure - pd.Timedelta(days=1),
            disclosure,
            disclosure + pd.Timedelta(days=1),
        ]
    )
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, np.nan, np.nan],
            "MSFT": [np.nan, 200.0, 210.0],
            "SPY": [400.0, 401.0, 402.0],
        },
        index=price_dates,
    )
    entries = pd.DataFrame(
        {
            "member": ["Alice", "Bob"],
            "ticker": ["AAPL", "MSFT"],
            "disclosure_date": [disclosure, disclosure],
            "transaction_type": ["Purchase", "Purchase"],
            "entry_price": [100.0, 200.0],
        }
    )

    result = calculate_signal_potential(entries, prices, horizons=[1]).set_index(
        "ticker"
    )

    assert not bool(result.loc["AAPL", "window_complete"])
    for column in (
        "peak_potential_pct",
        "decayed_return_pct",
        "spy_alpha_pct",
        "total_return_pct",
        "total_spy_alpha_pct",
        "decayed_spy_return_pct",
    ):
        assert np.isnan(result.loc["AAPL", column]), column
    assert bool(result.loc["MSFT", "window_complete"])
    assert np.isclose(result.loc["MSFT", "total_return_pct"], 5.0)
