import pandas as pd
import numpy as np
from analyzer.models import TransactionType
from analyzer.exceptions import AnalysisError

def calculate_signal_potential(
    transactions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: list[int] = [30, 60, 90, 180],
) -> pd.DataFrame:
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
    signals['signal_id'] = range(len(signals))
    signals = signals.drop(columns=['date'], errors='ignore')

    merged = pd.merge(signals, prices_long, on='ticker')
    window_mask = (merged['date'] >= merged['disclosure_date']) & (merged['date'] <= merged['window_end'])
    windowed = merged[window_mask]

    extrema = windowed.groupby('signal_id', as_index=False).agg(
        peak_price=('price', 'max'),
        trough_price=('price', 'min')
    )

    final = pd.merge(signals, extrema, on='signal_id', how='inner').drop(columns='signal_id')

    if final.empty:
        raise AnalysisError("No valid signals calculated after price analysis")

    is_purchase = final['transaction_type'] == TransactionType.PURCHASE.value

    peak_potential = np.where(
        is_purchase,
        (final['peak_price'] / final['entry_price'] - 1) * 100,
        (final['entry_price'] / final['trough_price'] - 1) * 100
    )

    return final.assign(
        signal_type=final['transaction_type'],
        peak_potential_pct=peak_potential
    )[['member', 'ticker', 'disclosure_date', 'signal_type', 'horizon_days', 'entry_price', 'peak_potential_pct']]

def rank_members(signal_df: pd.DataFrame, horizon: int = 90, threshold: float = 5.0) -> pd.DataFrame:
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

def get_horizon_performance(signal_df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signal dataframe")

    purchase_data = signal_df[signal_df['signal_type'] == TransactionType.PURCHASE.value]
    if purchase_data.empty:
        raise AnalysisError("No purchase signals found")

    return purchase_data.groupby('horizon_days')['peak_potential_pct'].agg([
        ('avg_peak_pct', 'mean'),
        ('hit_rate_pct', lambda x: (x > threshold).mean() * 100)
    ]).round(2)

def get_top_signals(signal_df: pd.DataFrame, horizon: int = 90, top_n: int = 15) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signal dataframe")

    top_data = signal_df[
        (signal_df['horizon_days'] == horizon) &
        (signal_df['signal_type'] == TransactionType.PURCHASE.value)
    ]

    if top_data.empty:
        raise AnalysisError(f"No purchase signals found for horizon {horizon}")

    return top_data.nlargest(top_n, 'peak_potential_pct')[
        ['member', 'ticker', 'disclosure_date', 'peak_potential_pct']
    ]

def get_member_signals(signal_df: pd.DataFrame, member: str, horizon: int = 90, top_n: int = 5) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signal dataframe")

    member_data = signal_df[
        (signal_df['member'] == member) &
        (signal_df['horizon_days'] == horizon) &
        (signal_df['signal_type'] == TransactionType.PURCHASE.value)
    ]

    if member_data.empty:
        raise AnalysisError(f"No purchase signals found for member {member} at horizon {horizon}")

    return member_data.nlargest(top_n, 'peak_potential_pct')[
        ['ticker', 'disclosure_date', 'peak_potential_pct']
    ]

def get_analysis_table(
    signals_df: pd.DataFrame,
    member_filter: str | None,
    show_signals: bool,
    horizon: int,
    top_n: int | None,
    threshold: float,
) -> pd.DataFrame:
    if member_filter:
        return get_member_signals(signals_df, member_filter, horizon, top_n)
    if show_signals:
        return get_top_signals(signals_df, horizon, top_n)
    return rank_members(signals_df, horizon, threshold)

def score_ticker_by_buyers(
    ticker: str,
    transactions_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    horizon: int = 90,
    threshold: float = 5.0,
    member_rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if signals_df.empty:
        raise AnalysisError("Empty signal dataframe")
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")

    if member_rankings is None:
        member_rankings = rank_members(signals_df, horizon, threshold)

    ticker_trades = transactions_df[
        (transactions_df['ticker'] == ticker) &
        (transactions_df['transaction_type'] == TransactionType.PURCHASE.value)
    ]

    if ticker_trades.empty:
        return pd.DataFrame({
            'ticker': [ticker],
            'num_buyers': [0],
            'signal_score': [0.0]
        })

    buyers = ticker_trades['member'].unique()
    buyer_stats = member_rankings[member_rankings['member'].isin(buyers)].sort_values('avg_peak_return_pct', ascending=False)

    if buyer_stats.empty:
        return pd.DataFrame({
            'ticker': [ticker],
            'num_buyers': [len(buyers)],
            'buyers': [', '.join(buyers[:3])],
            'signal_score': [0.0]
        })

    avg_rank = buyer_stats['avg_peak_return_pct'].mean()
    max_rank = buyer_stats['avg_peak_return_pct'].max()
    total_trades = buyer_stats['purchase_trades'].sum()
    signal_score = len(buyers) * avg_rank

    top_buyers = buyer_stats['member'].head(3).tolist()
    buyer_label = f"Top {len(top_buyers)} of {len(buyers)}" if len(buyers) > 3 else f"{len(buyers)}"

    return pd.DataFrame({
        'ticker': [ticker],
        'num_buyers': [len(buyers)],
        'buyer_label': [buyer_label],
        'buyers': [', '.join(top_buyers)],
        'avg_buyer_performance': [round(avg_rank, 2)],
        'best_buyer_performance': [round(max_rank, 2)],
        'total_buyer_trades': [int(total_trades)],
        'signal_score': [round(signal_score, 2)]
    })

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

    member_rankings = rank_members(signals_df, horizon, threshold)

    ticker_trades = transactions_df[
        (transactions_df['ticker'] == ticker) &
        (transactions_df['transaction_type'] == TransactionType.PURCHASE.value)
    ]

    if ticker_trades.empty:
        raise AnalysisError(f"No purchases found for {ticker}")

    buyers_with_dates = ticker_trades.groupby('member').agg({
        'transaction_date': list,
        'disclosure_date': list
    }).reset_index()

    result = pd.merge(
        buyers_with_dates,
        member_rankings[['member', 'avg_peak_return_pct', 'hit_rate_pct', 'purchase_trades']],
        on='member',
        how='left'
    )

    result = result.sort_values('avg_peak_return_pct', ascending=False, na_position='last')
    result['num_purchases'] = result['transaction_date'].apply(len)

    return result[['member', 'num_purchases', 'transaction_date', 'disclosure_date',
                   'avg_peak_return_pct', 'hit_rate_pct', 'purchase_trades']]