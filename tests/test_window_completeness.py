import numpy as np
import pandas as pd

from analyzer.analysis import calculate_signal_potential


def test_ticker_gap_does_not_use_stale_last_price_for_incomplete_horizon():
    disclosure = pd.Timestamp.now().normalize() - pd.Timedelta(days=120)
    dates = pd.date_range(disclosure, disclosure + pd.Timedelta(days=95), freq="D")
    entry_date = disclosure + pd.Timedelta(days=1)
    complete_exit = entry_date + pd.Timedelta(days=80)
    aapl = pd.Series(np.nan, index=dates)
    aapl.loc[entry_date:complete_exit] = np.linspace(
        100.0, 110.0, len(aapl.loc[entry_date:complete_exit])
    )
    prices = pd.DataFrame({"AAPL": aapl, "SPY": 400.0}, index=dates)
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
    by_horizon = result.set_index("horizon_days")

    assert bool(by_horizon.loc[80, "window_complete"])
    assert np.isclose(by_horizon.loc[80, "total_return_pct"], 10.0)
    assert not bool(by_horizon.loc[90, "window_complete"])
    for column in (
        "peak_potential_pct",
        "decayed_return_pct",
        "spy_alpha_pct",
        "total_return_pct",
        "total_spy_alpha_pct",
        "decayed_spy_return_pct",
    ):
        assert np.isnan(by_horizon.loc[90, column]), column


def test_quote_before_disclosure_does_not_complete_empty_window():
    disclosure = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    price_dates = pd.date_range(
        disclosure - pd.Timedelta(days=1), disclosure + pd.Timedelta(days=2), freq="D"
    )
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, np.nan, np.nan, np.nan],
            "MSFT": [np.nan, 200.0, 210.0, 220.0],
            "SPY": [400.0, 401.0, 402.0, 403.0],
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
    assert np.isclose(result.loc["MSFT", "total_return_pct"], (220 / 210 - 1) * 100)


def test_future_prices_do_not_populate_immature_labels():
    disclosure = pd.Timestamp.now().normalize() - pd.Timedelta(days=10)
    dates = pd.date_range(disclosure, disclosure + pd.Timedelta(days=40), freq="D")
    prices = pd.DataFrame({"AAPL": np.arange(len(dates)) + 100.0, "SPY": 400.0}, index=dates)
    entries = pd.DataFrame(
        {
            "member": ["Alice"],
            "ticker": ["AAPL"],
            "disclosure_date": [disclosure],
            "transaction_type": ["Purchase"],
            "entry_price": [100.0],
        }
    )

    row = calculate_signal_potential(entries, prices, horizons=[30]).iloc[0]

    assert not bool(row["window_complete"])
    for column in (
        "peak_potential_pct",
        "decayed_return_pct",
        "spy_alpha_pct",
        "total_return_pct",
        "total_spy_alpha_pct",
        "decayed_spy_return_pct",
    ):
        assert np.isnan(row[column]), column


def test_first_quote_25_days_after_disclosure_is_not_a_shortened_label():
    disclosure = pd.Timestamp("2024-01-01")
    dates = pd.DatetimeIndex([disclosure, disclosure + pd.Timedelta(days=25), disclosure + pd.Timedelta(days=55)])
    prices = pd.DataFrame(
        {"AAPL": [np.nan, 200.0, 220.0], "SPY": [400.0, 400.0, 400.0]},
        index=dates,
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

    row = calculate_signal_potential(entries, prices, horizons=[30]).iloc[0]

    assert not bool(row["window_complete"])
    assert np.isnan(row["total_return_pct"])


def test_missing_spy_on_security_endpoint_makes_label_unavailable():
    disclosure = pd.Timestamp("2024-01-01")
    dates = pd.date_range(disclosure, disclosure + pd.Timedelta(days=4), freq="D")
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0], "SPY": [400.0, np.nan, 402.0, 403.0, 404.0]},
        index=dates,
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

    row = calculate_signal_potential(entries, prices, horizons=[2]).iloc[0]

    assert not bool(row["window_complete"])
    assert np.isnan(row["total_spy_alpha_pct"])
