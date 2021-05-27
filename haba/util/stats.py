"""
Statistical tools
"""

import numpy as np
from numba import njit, prange
from scipy import stats as sps


def normalize(arr, zmax=6, zerr=9, method='uniform'):
    """
    First computes robust standard deviation (rstd), sets any
    data exceeding

        +/- zerr * rstd

    to NaN, and then clips data inside

        +/- zmax * rstd

    Finally, the data is normalized using 'method'

    Parameters
    ----------
    arr : numpy array or pandas object
        object holding data to be normalized
    zmax: int, default 6
        z-score value used to clip data
    zerr: int, default 9
        z-score value used to identify errors in data
    method : str, default 'uniform'
        method used to normalize data:  ('uniform', 'normal')

    Returns
    -------
    normalized : numpy array

    """
    rstd = 1.4826 * np.nanmedian(np.abs(arr - np.nanmedian(arr, axis=0)), axis=0)
    cleaned = np.where(np.abs(arr) < zerr * rstd, arr, np.full_like(arr, np.nan))
    clipped = np.clip(cleaned, -zmax * rstd, +zmax * rstd)

    if method == 'uniform':
        smallest = np.nanmin(clipped, axis=0)
        biggest = np.nanmax(clipped, axis=0)
        return (clipped - smallest) / (biggest - smallest)
    elif method == 'normal':
        mean = np.nanmean(clipped, axis=0)
        std = np.nanstd(clipped, axis=0)
        return (clipped - mean) / std
    else:
        msg = f'Invalid method: {method}'
        raise Exception(msg)


@njit
def _resample(arr, n_resamples):
    counter = 0
    sample_mean = np.mean(arr)

    for _ in prange(n_resamples):
        resampled = np.random.choice(arr, replace=True, size=arr.size)
        resampled_mean = np.mean(resampled)

        if resampled_mean < sample_mean:
            counter += 1

    p_value = counter / n_resamples

    return p_value


def resample(seq, accuracy=0.1, confidence=0.05):
    epsilon = confidence / 2
    p_value = 1 - epsilon
    degrees_of_freedom = len(seq) - 1
    quantile = sps.t.ppf(p_value, degrees_of_freedom)
    odds = (1 - epsilon) / epsilon
    n_resamples = round(odds * (quantile / accuracy) ** 2)
    return _resample(seq, n_resamples)



if __name__ == '__main__':
    import pandas as pd

    obj = pd.DataFrame(np.random.randn(20, 3), columns=list('abc'))
    obj.loc[10, 'a'] = 10
    # obj = pd.Series(np.random.randn(20))
    # obj.loc[10] = 10
    obj_normed = normalize(obj, 6, 9, method='normal')

    print(obj.to_string())
    print('-'*45)
    print(obj_normed)
