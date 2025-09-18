import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_signal_potential(transactions_df, prices_df, horizons=[30, 60, 90, 180]):
    results = []

    for _, row in transactions_df.iterrows():
        ticker = row['ticker']
        disclosure_date = row['disclosure_date']
        is_purchase = row['transaction_type'] == 'Purchase'

        if ticker not in prices_df.columns:
            continue

        px_series = prices_df[ticker].dropna()
        if px_series.empty:
            continue

        entry_price = px_series.asof(disclosure_date)
        if pd.isna(entry_price):
            continue

        for horizon in horizons:
            window_end = disclosure_date + timedelta(days=horizon)
            price_window = px_series.loc[disclosure_date:window_end]

            if price_window.empty:
                continue

            if is_purchase:
                peak_price = price_window.max()
                peak_potential = (peak_price / entry_price - 1) * 100
            else:
                trough_price = price_window.min()
                peak_potential = (entry_price / trough_price - 1) * 100

            results.append({
                'member': row['member'],
                'ticker': ticker,
                'disclosure_date': disclosure_date,
                'signal_type': 'Purchase' if is_purchase else 'Sale',
                'horizon_days': horizon,
                'entry_price': entry_price,
                'peak_potential_pct': peak_potential
            })

    return pd.DataFrame(results)

def rank_members(signal_df, horizon=90, threshold=5.0):
    if signal_df.empty:
        return pd.DataFrame()

    analysis_df = signal_df[signal_df['horizon_days'] == horizon]
    if analysis_df.empty:
        return pd.DataFrame()

    stats = analysis_df.groupby(['member', 'signal_type']).agg(
        avg_peak=('peak_potential_pct', 'mean'),
        median_peak=('peak_potential_pct', 'median'),
        hit_rate=('peak_potential_pct', lambda x: (x > threshold).mean() * 100),
        trades=('ticker', 'count')
    )

    pivoted = stats.unstack('signal_type', fill_value=0)
    pivoted.columns = [f"{stat}_{signal}" for stat, signal in pivoted.columns]

    column_map = {
        'avg_peak_Purchase': 'avg_peak_return_pct',
        'median_peak_Purchase': 'median_peak_return_pct',
        'hit_rate_Purchase': 'hit_rate_pct',
        'trades_Purchase': 'purchase_trades',
        'avg_peak_Sale': 'avg_loss_avoided_pct',
        'trades_Sale': 'sale_trades'
    }

    for old_col, new_col in column_map.items():
        if old_col in pivoted.columns:
            pivoted[new_col] = pivoted[old_col]

    keep_cols = list(column_map.values())
    available_cols = [col for col in keep_cols if col in pivoted.columns]

    result = pivoted[available_cols].round(2)
    result.reset_index(inplace=True)
    return result.sort_values('avg_peak_return_pct', ascending=False) if 'avg_peak_return_pct' in result.columns else result

def get_horizon_performance(signal_df, threshold=5.0):
    purchase_data = signal_df[signal_df['signal_type'] == 'Purchase']
    if purchase_data.empty:
        return pd.DataFrame()

    return purchase_data.groupby('horizon_days')['peak_potential_pct'].agg([
        ('avg_peak_pct', 'mean'),
        ('hit_rate_pct', lambda x: (x > threshold).mean() * 100)
    ]).round(2)

def get_top_signals(signal_df, horizon=90, top_n=15):
    if signal_df.empty:
        return pd.DataFrame()

    top_data = signal_df[
        (signal_df['horizon_days'] == horizon) &
        (signal_df['signal_type'] == 'Purchase')
    ]

    if top_data.empty:
        return pd.DataFrame()

    return top_data.nlargest(top_n, 'peak_potential_pct')[
        ['member', 'ticker', 'disclosure_date', 'peak_potential_pct']
    ]

def get_member_signals(signal_df, member, horizon=90, top_n=5):
    if signal_df.empty:
        return pd.DataFrame()

    member_data = signal_df[
        (signal_df['member'] == member) &
        (signal_df['horizon_days'] == horizon) &
        (signal_df['signal_type'] == 'Purchase')
    ]

    if member_data.empty:
        return pd.DataFrame()

    return member_data.nlargest(top_n, 'peak_potential_pct')[
        ['ticker', 'disclosure_date', 'peak_potential_pct']
    ]