import pandas as pd
import numpy as np
from datetime import timedelta
from .exceptions import AnalysisError

def calculate_signal_potential(transactions_df, prices_df, horizons=[30, 60, 90, 180]):
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")
    if prices_df.empty:
        raise AnalysisError("Empty prices dataframe")

    required_cols = {'member', 'ticker', 'disclosure_date', 'transaction_type'}
    if not required_cols.issubset(transactions_df.columns):
        raise AnalysisError(f"Missing columns in transactions: {required_cols - set(transactions_df.columns)}")

    prices_long = prices_df.stack().reset_index(name='price')
    prices_long.columns = ['date', 'ticker', 'price']

    trans_sorted = transactions_df.sort_values('disclosure_date').reset_index(drop=True)
    prices_sorted = prices_long.sort_values('date')

    signals = pd.merge_asof(
        trans_sorted, prices_sorted,
        left_on='disclosure_date', right_on='date', by='ticker'
    ).dropna(subset=['price']).rename(columns={'price': 'entry_price'})

    if signals.empty:
        raise AnalysisError("No valid price matches found for transactions")

    signals = signals.assign(horizon_days=[horizons] * len(signals)).explode('horizon_days').reset_index(drop=True)
    signals['horizon_days'] = signals['horizon_days'].astype('int32')
    signals['window_end'] = signals['disclosure_date'] + pd.to_timedelta(signals['horizon_days'], unit='D')
    signals['signal_idx'] = signals.index

    merged = pd.merge(
        signals,
        prices_long,
        on='ticker',
        suffixes=('', '_price')
    )

    in_window = merged[
        (merged['date'] >= merged['disclosure_date']) &
        (merged['date'] <= merged['window_end'])
    ]

    if in_window.empty:
        raise AnalysisError("No price data found within signal windows")

    extrema = in_window.groupby('signal_idx').agg(
        peak_price=('price', 'max'),
        trough_price=('price', 'min')
    )

    final = pd.merge(signals, extrema, left_on='signal_idx', right_index=True).dropna(subset=['peak_price', 'trough_price'])

    if final.empty:
        raise AnalysisError("No valid signals calculated after price analysis")

    is_purchase = final['transaction_type'] == 'Purchase'

    peak_potential = np.where(
        is_purchase,
        (final['peak_price'] / final['entry_price'] - 1) * 100,
        (final['entry_price'] / final['trough_price'] - 1) * 100
    )

    return final.assign(
        signal_type=final['transaction_type'],
        peak_potential_pct=peak_potential
    )[['member', 'ticker', 'disclosure_date', 'signal_type', 'horizon_days', 'entry_price', 'peak_potential_pct']]

def rank_members(signal_df, horizon=90, threshold=5.0):
    if signal_df.empty:
        raise AnalysisError("Empty signal dataframe")

    analysis_df = signal_df[signal_df['horizon_days'] == horizon]
    if analysis_df.empty:
        raise AnalysisError(f"No signals found for horizon {horizon}")

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

    keep_cols = [col for col in column_map.values() if col in pivoted.columns]
    result = pivoted[keep_cols].round(2).reset_index()

    if 'avg_peak_return_pct' in result.columns:
        return result.sort_values('avg_peak_return_pct', ascending=False)
    return result

def get_horizon_performance(signal_df, threshold=5.0):
    if signal_df.empty:
        raise AnalysisError("Empty signal dataframe")

    purchase_data = signal_df[signal_df['signal_type'] == 'Purchase']
    if purchase_data.empty:
        raise AnalysisError("No purchase signals found")

    return purchase_data.groupby('horizon_days')['peak_potential_pct'].agg([
        ('avg_peak_pct', 'mean'),
        ('hit_rate_pct', lambda x: (x > threshold).mean() * 100)
    ]).round(2)

def get_top_signals(signal_df, horizon=90, top_n=15):
    if signal_df.empty:
        raise AnalysisError("Empty signal dataframe")

    top_data = signal_df[
        (signal_df['horizon_days'] == horizon) &
        (signal_df['signal_type'] == 'Purchase')
    ]

    if top_data.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    return top_data.nlargest(top_n, 'peak_potential_pct')[
        ['member', 'ticker', 'disclosure_date', 'peak_potential_pct']
    ]

def get_member_signals(signal_df, member, horizon=90, top_n=5):
    if signal_df.empty:
        raise AnalysisError("Empty signal dataframe")

    member_data = signal_df[
        (signal_df['member'] == member) &
        (signal_df['horizon_days'] == horizon) &
        (signal_df['signal_type'] == 'Purchase')
    ]

    if member_data.empty:
        raise AnalysisError(f"No purchase signals found for member {member} at horizon {horizon}")

    return member_data.nlargest(top_n, 'peak_potential_pct')[
        ['ticker', 'disclosure_date', 'peak_potential_pct']
    ]

def get_analysis_table(signals_df, member_filter, show_signals, horizon, top_n, threshold):
    if member_filter:
        return get_member_signals(signals_df, member_filter, horizon, top_n)
    if show_signals:
        return get_top_signals(signals_df, horizon, top_n)
    return rank_members(signals_df, horizon, threshold)