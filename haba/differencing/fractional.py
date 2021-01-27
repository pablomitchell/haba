"""
Fractional differencing of a time series
"""

import numpy as np
import pandas as pd

__all__ = [
    'difference',
]


def _get_weights(d, eps=1.0e-05):
    """
    Generate fixed window weights used to fractionally
    difference time series

    Parameters
    ----------
    d : float
    eps : float

    Returns
    -------
    weights : numpy.array

    """
    weights = [1.0]
    k = 1

    while True:
        weight = -weights[-1] * (d - k + 1) / k

        if abs(weight) < eps:
            break

        weights.append(weight)
        k += 1

    weights = np.array(weights[::-1])

    return weights


def _plot_weights(d_lo, d_hi, n_plots, size):
    """
    Plot weights used for fractionally differencing

    Parameters
    ----------
    d_range : sequence
    n_plots : int
    size : int

    """
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-bright')

    w = {}

    for d in np.linspace(d_lo, d_hi, n_plots):
        w[f'{d:0.2f}'] = get_weights(d, size)

    df_weights = pd.DataFrame(w, index=range(size, 0, -1))
    ax = df_weights.plot()
    ax.legend(loc='upper right')
    plt.show()


def fractional_difference(ser, d, eps=1.0e-05):
    """
    Take the fractional difference of time series

    Parameters
    ----------
    ser : pandas.Series
    d : float
        order of the differencing scheme where `d = 1`
        means ordinary first order differencing
    eps : float
        determines the size of the smallest weight
        in the fixed width differencing scheme

    Returns
    -------
    ser_diff : pandas.Series

    """
    weights = _get_weights(d, eps)
    width = len(weights)
    return (ser
            .fillna(method='ffill')
            .rolling(width)
            .apply(lambda x: np.dot(weights, x))
            )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from haba.util.tseries import generate_prices

    start = '1990-01-01'
    end = '2020-12-31'

    drift = 0.05
    volatility = 0.17
    p0 = 100
    d = 0.7

    ser = generate_prices(start, end, drift, volatility, initial_price=p0)
    diff_ser_05 = fractional_difference(ser, d)
    diff_ser_05_rave = diff_ser_05.rolling(260).mean()

    ser.plot()
    diff_ser_05.plot(secondary_y=True)
    diff_ser_05_rave.plot(secondary_y=True)
    plt.title(f'Correlation {ser.corr(diff_ser_05):0.1f}')
    plt.show()
