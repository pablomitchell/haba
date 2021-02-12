"""
Trend scanning labeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
from scipy.stats import t as tdist


plt.style.use('seaborn')


class TrendScanning(object):

    def __init__(self, prices, low, high, fractional_difference=False):
        """
        Identify and label persistent trends in a price time series

        Parameters
        ----------
        prices : pandas.Series
            series of close price data with pandas.DatetimeIndex
        span : sequence of ints, (lo, hi)
            specifies the look forward periods used to scan for a trend
            'lo' and 'hi' should be large enough to allow for computation
            of t-stats resulting from linear regression
        fractional_difference : bool
            use fraction differencing when computing returns rather
            than regular differencing -- defaults to False
        """
        self.prices = prices
        self.lo = low
        self.hi = high
        self.fractional_difference = fractional_difference

        self.labels = None
        self.weights = None

    @staticmethod
    def _bin(ser):
        ser_scaled = 6 * ser - 3
        return (ser_scaled
                .where(1 < ser_scaled.abs(), 0)
                .clip(-1, 1).astype(int)
                )

    @staticmethod
    def _sgn(ser):
        return np.sign(ser).astype(int)

    @staticmethod
    @numba.jit
    def _compute_tstat(y_arr):
        # fast t-stat computation
        covar = lambda x, y: ((x - x.mean()) * (y - y.mean())).mean()

        try:
            y = y_arr[~np.isnan(y_arr)]
            n = len(y)
            x = np.arange(n)
            sxx = covar(x, x)
            syy = covar(y, y)
            sxy = covar(x, y)
            ssr = (sxx * syy - sxy * sxy) / sxx
            b = sxy / sxx
            return np.sqrt((n - 2) * sxx / ssr) * b
        except:
            return np.nan

    def _make_label(self, ser):
        t_stats = (ser
                  .expanding(min_periods=self.lo)
                  .apply(self._compute_tstat, raw=True)
                  )
        idx = t_stats.abs().idxmax()
        t_stat = t_stats[idx]
        degrees_freedom = t_stats.index.get_loc(idx) - 1
        return tdist.cdf(t_stat, degrees_freedom)

    def make_labels(self):
        """
        Populates 'labels' dataframe with the following columns:
            end:  end date of the trend scanning period
            t_stat : student distribution statistic of the trend
                slope coefficient
            label:  label associated with the sign of the trend
                -1 : down trend
                +1 : up trend
        and where 'labels' shares the same index as 'prices'.
        """
        roll = (self.prices
                .fillna(method='ffill', limit=1)
                .dropna()
                .rolling(window=self.hi, min_periods=self.hi)
                )
        labels = roll.apply(self._make_label).to_frame('p_val').dropna()
        labels['label'] = self._bin(labels['p_val'])
        labels['end'] = labels.index.copy()
        labels.index -= pd.offsets.BDay(self.hi - 1)
        labels.index.name = 'start'
        self.labels = labels.get(['end', 'p_val', 'label']).dropna()

    def make_meta_labels(self, side):
        """
        Given a time-series of 'side' predictions, populates 'labels'
        dataframe with the additional columns:
            side:  the direction of the bet given by the primary model
            meta_label:  label indicating whether 'side' is correct
                0 : incorrect
                1 : correct

        Parameters
        ----------
        side : pandas.Series
            time-series provided by a primary model giving a signal that
            indicates the side and magnitude of the bet
        """
        if self.labels is None:
            self.make_labels()

        self.labels, side_aligned = \
            self.labels.align(side, axis=0, join='left')
        self.labels['side'] = self._sgn(side_aligned)
        self.labels['meta_label'] = (
                self.labels['label'] == self.labels['side']
        ).astype(int)

    def describe(self):
        pass

    def plot_labels(self):
        plt.close('all')

        fig, axes = plt.subplots(2, 1, dpi=200, figsize=(11, 8), sharex='all')

        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        else:
            axes = [axes]

        idx = ts.labels.index

        axes[0].scatter(idx, prices.loc[idx].values, c=ts.labels['label'], cmap='viridis')
        axes[0].title.set_text('Labels')

        axes[1].scatter(idx, prices.loc[idx].values, c=ts.labels['p_val'], cmap='viridis')
        axes[1].title.set_text('P-Vals')

        plt.show()

    def plot_weights(self):
        """
        Plot histogram of sample weights
        """
        assert self.weights is not None


if __name__ == '__main__':
    from haba.util.tseries import generate_prices

    start = '1990-01-01'
    end = '2020-12-31'
    drift = 0.06
    volatility = 0.25
    prices = generate_prices(start, end, drift, volatility)
    # dates = pd.bdate_range(start=start, end=end)
    # prices = pd.Series(np.random.normal(0, 0.1, len(dates)), index=dates).cumsum()
    # prices += np.sin(np.linspace(0, 10, len(dates)))

    # import cProfile, pstats
    # pfile = 'trend_scanning.profile'
    # cProfile.run(
    #     'TrendScanning(prices, low=10, high=21)'
    #     '.make_meta_labels(side=prices)',
    #     pfile
    # )
    # s = pstats.Stats(pfile)
    # s.strip_dirs()
    # s.sort_stats('cumtime').print_stats(15)
    # exit()

    import time
    t0 = time.time()
    ts = TrendScanning(prices, low=5, high=15)
    ts.make_meta_labels(side=prices)
    t1 = time.time()
    ts.plot_labels()
    print(f'{t1 - t0:0.4f} seconds')
    print(ts.labels.tail(25).to_string(float_format='{:.6f}'.format))


