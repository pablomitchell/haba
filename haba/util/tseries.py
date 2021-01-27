"""
Part of haba
"""

import numpy as np
import pandas as pd


business_days_per_year = 260


def generate_prices(start, end, drift, volatility, initial_price=100.0):
    """
    Generate a synthetic return series, and geometrically cumulate
    them to produce a daily price series

    Parameters
    ----------
    start : string or date
        start date given in %Y-%m-%d format
    end : string or date
        end date given in %Y-%m-%d format
    drift : float
        constant drift (mean) of the process
    volatility : float
        constant volatility (std) of the process
    initial_price : float, default 10.0
        beginning price of the series

    Returns
    -------
    prices : pandas.Series
         has pandas.DatetimeIndex with freq='B'

    """
    assert 0 < initial_price

    dt = 1.0 / business_days_per_year  # time step
    dates = pd.bdate_range(start, end)
    noise = np.random.randn(len(dates))
    rets = drift * dt + noise * volatility * np.sqrt(dt)
    rets_ser = pd.Series(rets, index=dates)
    prices_ser = initial_price * (1 + rets_ser).cumprod()
    prices_ser.name = 'price'

    return prices_ser


def get_volatility(ser, window=65, name=None):
    """
    Compute volatility of 'ser'

    Parameters
    ----------
    ser : pandas.Series
       assumes pandas.DatetimeIndex with freq='B'
    window : int
       time horizon expressed in the same units as the data in 'ser'
    name : str
       name to give the resulting time series

    Returns
    -------
    vol : pandas.Series

    """
    if name is None:
        name = f'vol_{window}'

        if ser.name:
            name = f'{ser.name}_{name}'

    vol = np.log(ser).diff().ewm(min_periods=window, span=window).std()
    vol.name = name

    return vol


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    start = '1990-01-01'
    end = '2020-12-31'

    drift = 0.0
    volatility = 0.15

    ser = generate_prices(start, end, drift, volatility)
    vol = np.sqrt(business_days_per_year)*get_volatility(ser)

    ser.plot(legend=True)
    vol.plot(secondary_y=True, legend=True)
    plt.title(f'actual={volatility:0.2f}  vs  empirical={vol.mean():0.2f}')
    plt.show()

