"""
Statistical tools
"""

import numpy as np
from numba import njit, prange
from scipy import stats as sps


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
    import time
    size = 260

    print('running...')
    t0 = time.time()

    for _ in range(size*20*):
        resample(np.random.rand(size))

    t1 = time.time()
    print(f'done after: {t1 - t0:.2f} seconds')