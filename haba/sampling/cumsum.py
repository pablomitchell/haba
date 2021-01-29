"""
Sampling events from a time series
"""

import pandas as pd

from haba.util.tseries import get_volatility


def get_events(ser, threshold, kind='relative'):
    """
    Sample events from a time series using a
    symmetric cumulative sum filter

    Parameters
    ----------
    ser : pandas.Series
        index is a pandas.DatatimeIndex
    threshold : float
        sample once the cumulative sum (in either direction) has
        surpassed this threshold
    kind : str, defaults to 'relative'
        kind = ['absolute', 'relative']
        if 'absolute' then threshold must be specified in units of
        absolute change of the underlying data in 'ser', otherwise
        it must specified in units of relative change:
            'absolute' ==> ser1 - ser0
            'relative' ==> ser1/ser0 - 1

    Returns
    -------
    events : pandas.DatetimeIndex

    """
    assert kind in ['relative', 'absolute']

    events = []
    sum_neg = sum_pos = 0

    if kind == 'relative':
        ser_chg = ser.fillna(method='ffill').pct_change().dropna()
    else:
        ser_chg = ser.fillna(method='ffill').diff().dropna()

    for dt in ser_chg.index:
        sum_neg = min(0, sum_neg + ser_chg.loc[dt])

        if threshold < abs(sum_neg):
            sum_neg = 0;
            events.append(dt)
            continue

        sum_pos = max(0, sum_pos + ser_chg.loc[dt])

        if threshold < abs(sum_pos):
            sum_pos = 0;
            events.append(dt)
            continue

    return pd.DatetimeIndex(events)


if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt
    from haba.util.tseries import generate_prices

    start = '1990-01-01'
    end = '2020-12-31'

    sharpe = 0.25
    drift = 0.07
    volatility = drift / sharpe
    d = 0.7

    prices_ser = generate_prices(start, end, drift, volatility)
    dt_idx = get_events(prices_ser, 3 * volatility / math.sqrt(260))
    prices_ser_sampled = prices_ser.loc[dt_idx]

    sample_freq_in_days = len(prices_ser)//len(dt_idx)

    prices_ser.plot()
    prices_ser_sampled.plot(linestyle='None', marker='.')
    plt.title(f'sample every {sample_freq_in_days} days')
    plt.show()
