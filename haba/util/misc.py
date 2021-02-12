"""
Miscellaneous utilities
"""

import numpy as np
import pandas as pd


def desc(ser):
    """
    Like pandas.Series.describe but without percentiles

    Parameters
    ----------
    ser : pandas.Series

    Returns
    -------
    desc : pandas.Series
        index has elements ['count', 'mean', 'median',
                            'std', 'min', 'max']

    """
    return pd.Series({
        'count': ser.count(),
        'mean': ser.mean(),
        'median': ser.median(),
        'std': ser.std(),
        'min': ser.min(),
        'max': ser.max(),
    })


def sign(x):
    """
    Return the sign of the input

    Parameters
    ----------
    x : numerical data
        could be pandas.Series, numpy.array, float, etc

    Return
    ------
    x_signed : signed numerical output

    """
    return np.sign(np.nan_to_num(x)).astype(int)
