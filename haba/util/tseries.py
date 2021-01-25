"""
Part of haba
"""

import numpy as np
import pandas as pd


def generate_prices(start, end, drift, volatility, initial_price=10.0):
    """
    Generate synthetic return series (modeled after Brownian motion)
    and geometrically cumulate then to produce a synthetic prices series

    Parameters
    ----------
    start : string or date
        start date given in %Y-%m-%d format
    end : string or date
        end date given in %Y-%m-%d format
    drift : float
        constant drift (mean) of the Brownian motion
    volatility : float
        constant volatility (std) of the Brownian motion
    initial_price : float, default 10.0
        beginning price of the series

    Returns
    -------
    prices : pandas.Series

    """
    dt = 1.0 / 260  # time step 1 biz day
    dates = pd.bdate_range(start, end)
    noise = np.random.randn(len(dates))
    rets = drift * dt + noise * volatility * np.sqrt(dt)
    rets_ser = pd.Series(rets, index=dates)
    prices_ser = initial_price * (1 + rets_ser).cumprod()
    return prices_ser


