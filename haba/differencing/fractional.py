"""
Fractional differencing of features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')



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
    dt = 1.0 / 260  # time step

    dates = pd.bdate_range(start, end)
    noise = np.random.randn(len(dates))
    rets = drift * dt + noise * volatility * np.sqrt(dt)
    rets_ser = pd.Series(rets, index=dates)
    prices_ser = initial_price * (1 + rets_ser).cumprod()

    return prices_ser


def get_weights(d, size):
    """
    Generate weights used to fractionally difference time series

    Parameters
    ----------
    d : float
    size : int

    Returns
    -------
    weights : numpy.array

    """
    weights = [1.0]

    for ii in range(1, size):
        weight = -weights[-1] * (d - ii + 1) / ii
        weights.append(weight)

    weights = np.array(weights[::-1])

    return weights


def plot_weights(d_lo, d_hi, n_plots, size):
    """
    Plot weights used for fractionally differencing

    Parameters
    ----------
    d_range : sequence
    n_plots : int
    size : int

    """
    w = {}

    for d in np.linspace(d_lo, d_hi, n_plots):
        w[f'{d:0.2f}'] = get_weights(d, size)

    df_weights = pd.DataFrame(w, index=range(size, 0, -1))
    ax = df_weights.plot()
    ax.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    start = '1998-01-01'
    end = '2015-12-31'

    drift = 0.05
    volatility = 0.17

    ser = generate_prices(start, end, drift, volatility)
    ser.plot()
    plt.show()




