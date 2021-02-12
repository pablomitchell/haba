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

def sign(ser):
    """
    Given a pandas.Series return identically sized series
    with elements assigned integer signs for each element
    in input

    Parameters
    ----------
    ser : pandas.Series

    Return
    ------
    ser_signed : pandas.Series

    """
    return np.sign(ser.fillna(0)).astype(int)
