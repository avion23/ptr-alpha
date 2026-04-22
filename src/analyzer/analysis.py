import pandas as pd
import numpy as np
from analyzer.models import TransactionType
from analyzer.exceptions import AnalysisError

DECAY_LAMBDA = 0.05


def calculate_signal_potential(
    transactions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: list[int] = [30, 60, 90, 180],
    decay_lambda: float = DECAY_LAMBDA,
) -> pd.DataFrame:
    if transactions_df.empty:
        raise AnalysisError("Empty transactions dataframe")
    if prices_df.empty:
        raise AnalysisError("Empty prices dataframe")

    required_cols = {'member', 'ticker', 'disclosure_date', 'transaction_type'}
    if not required_cols.issubset(transactions_df.columns):
        raise AnalysisError(f"Missing columns in transactions: {required_cols - set(transactions_df.columns)}")

    prices_long = prices_df.stack().reset_index(name='price')
    prices_long.columns = ['price_date', 'ticker', 'price']

    spy_prices = prices_long[prices_long['ticker'] == 'SPY'][['price_date', 'price']].rename(columns={'price': 'spy_price'})
    prices_long = prices_long[prices_long['ticker'] != 'SPY'].copy()

    trans_sorted = transactions_df.sort_values('disclosure_date').reset_index(drop=True)
    prices_sorted = prices_long.sort_values('price_date')

    signals = pd.merge_asof(
        trans_sorted, prices_sorted,
        left_on='disclosure_date', right_on='price_date', by='ticker'
    ).dropna(subset=['price']).rename(columns={'price': 'entry_price', 'price_date': 'disclosure_price_date'})

    if signals.empty:
        raise AnalysisError("No valid price matches found for transactions")

    signals = signals.assign(horizon_days=[horizons] * len(signals)).explode('horizon_days').reset_index(drop=True)
    signals['horizon_days'] = signals['horizon_days'].astype('int32')
    signals['window_end'] = signals['disclosure_date'] + pd.to_timedelta(signals['horizon_days'], unit='D')
    signals['signal_id'] = range(len(signals))
    if 'price_date' in signals.columns:
        signals = signals.drop(columns=['price_date'])

    merged = signals.merge(prices_long, on='ticker', suffixes=('', '_price'))
    window_mask = (merged['price_date'] >= merged['disclosure_date']) & (merged['price_date'] <= merged['window_end'])
    windowed = merged[window_mask].copy()

    if windowed.empty:
        raise AnalysisError("No price data found in signal windows")

    windowed['days_from_disclosure'] = (windowed['price_date'] - windowed['disclosure_date']).dt.days
    windowed['decay_factor'] = np.exp(-decay_lambda * windowed['days_from_disclosure'])
    windowed['weighted_return'] = (windowed['price'] / windowed['entry_price'] - 1) * windowed['decay_factor']

    if not spy_prices.empty:
        spy_merged = windowed.merge(spy_prices, on='price_date', how='left')
        spy_merged['spy_entry_price'] = spy_merged.groupby('signal_id')['spy_price'].transform('first')
        windowed['spy_return'] = spy_merged['spy_price'] / spy_merged['spy_entry_price'] - 1
    else:
        windowed['spy_return'] = 0.0

    agg = windowed.groupby('signal_id').agg(
        peak_price=('price', 'max'),
        trough_price=('price', 'min'),
        decayed_return=('weighted_return', 'max'),
        spy_cumulative=('spy_return', 'max'),
        entry_price_first=('entry_price', 'first'),
        last_price=('price', 'last')
    )
    agg['total_return'] = (agg['last_price'] / agg['entry_price_first'] - 1)
    agg = agg.reset_index()

    final = signals.merge(agg[['signal_id', 'peak_price', 'trough_price', 'decayed_return', 'spy_cumulative', 'total_return']], on='signal_id', how='left')

    is_purchase = final['transaction_type'] == TransactionType.PURCHASE.value
    purchase_mask = is_purchase & (final['entry_price'] != 0)
    sale_mask = ~is_purchase & (final['trough_price'] != 0)

    peak_potential = np.zeros(len(final))
    peak_potential[purchase_mask.values] = (
        (final.loc[purchase_mask.values, 'peak_price'] / final.loc[purchase_mask.values, 'entry_price'] - 1) * 100
    ).values
    peak_potential[sale_mask.values] = (
        (final.loc[sale_mask.values, 'entry_price'] / final.loc[sale_mask.values, 'trough_price'] - 1) * 100
    ).values

    return final.assign(
        signal_type=final['transaction_type'],
        peak_potential_pct=peak_potential,
        decayed_return_pct=final['decayed_return'].fillna(0).values * 100,
        spy_alpha_pct=(final['decayed_return'].fillna(0) - final['spy_cumulative'].fillna(0)).values * 100,
        total_return_pct=final['total_return'].fillna(0).values * 100,
    )[['member', 'ticker', 'disclosure_date', 'signal_type', 'horizon_days', 'entry_price',
       'peak_potential_pct', 'decayed_return_pct', 'spy_alpha_pct', 'total_return_pct']]

def rank_members(signal_df: pd.DataFrame, horizon: int = 90, threshold: float = 5.0) -> pd.DataFrame:
    if signal_df.empty:
        raise AnalysisError("Empty signal dataframe")

    analysis_df = signal_df[signal_df['horizon_days'] == horizon]
    if analysis_df.empty:
        raise AnalysisError(f"No signals found for horizon {horizon}")

    purchases = analysis_df[analysis_df['signal_type'] == TransactionType.PURCHASE.value]

    MARKET_BASE_RATE = 0.50

    member_stats = []
    for member, grp in purchases.groupby('member'):
        rets = grp['decayed_return_pct'].values
        if len(rets) == 0:
            continue
        hit_rate = (grp['peak_potential_pct'] > threshold).mean() * 100
        median_ret = np.median(rets)
        mean_ret = np.mean(rets)
        std_ret = np.std(rets) if len(rets) > 1 else 0.0
        sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0

        p_up_given_buy = (rets > 0).sum() / len(rets)
        bayes_factor = p_up_given_buy / MARKET_BASE_RATE

        spy_alpha_vals = grp['spy_alpha_pct'].dropna().values
        avg_spy_alpha = np.mean(spy_alpha_vals) if len(spy_alpha_vals) > 0 else 0.0

        member_stats.append({
            'member': member,
            'avg_peak_return_pct': round(mean_ret, 2),
            'median_peak_return_pct': round(median_ret, 2),
            'hit_rate_pct': round(hit_rate, 2),
            'purchase_trades': len(rets),
            'avg_loss_avoided_pct': round(grp[grp['signal_type'] == TransactionType.SALE.value]['peak_potential_pct'].mean() if len(grp[grp['signal_type'] == TransactionType.SALE.value]) > 0 else 0, 2),
            'sale_trades': len(grp[grp['signal_type'] == TransactionType.SALE.value]),
            'sharpe_ratio': round(sharpe, 3),
            'prob_up_given_buy': round(p_up_given_buy, 3),
            'bayes_factor': round(bayes_factor, 3),
            'avg_spy_alpha_pct': round(avg_spy_alpha, 2),
        })

    result = pd.DataFrame(member_stats)

    if result.empty:
        return result

    return result.sort_values('median_peak_return_pct', ascending=False)

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