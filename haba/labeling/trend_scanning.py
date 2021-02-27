"""
Trend scanning labeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from haba.util import misc
from haba.util import tseries

plt.style.use('seaborn')


class TrendScanning(object):

    def __init__(self, prices, low, high):
        """
        Identify and label persistent trends in a price time series

        Parameters
        ----------
        prices : pandas.Series
            series of close price data with pandas.DatetimeIndex
        low, high : int
            specifies the look forward periods used to scan for a trend
            'low' and 'high' should be large enough to allow for computation
            of t-stats resulting from linear regression

        """
        assert not prices.isnull().any(), 'prices not allowed to have NAs'
        assert 2 < low
        assert low < high
        assert high < len(prices)

        self.prices = prices
        self.lo = low
        self.hi = high

        self.labels = None
        self.weights = None

    def __repr__(self):
        msg = str()
        show_these_dfs = [
            'prices',
            'labels',
            'weights'
        ]
        divider = '-' * 45

        for name in show_these_dfs:
            df = getattr(self, name)

            if df is None:
                continue

            msg += (
                f'{name.upper()} \n'
                f'{df.head().to_string()} \n'
                f'\t...\n'
                f'{df.tail().to_string()} \n'
                f'{divider} \n'
            )

        return msg

    def _make_weights(self):
        t_abs = self.labels['t'].abs()
        self.weights = pd.Series(
            t_abs / t_abs.sum(),
            index=self.labels.index,
        )

    def _make_label(self, ser):
        exp = ser.expanding(min_periods=self.lo)
        ts = exp.apply(tseries.numba_tstat, raw=True)
        return ts[ts.abs().idxmax()]

    def make_labels(self):
        """
        Populates 'labels' dataframe with the following columns:
            t:  student distribution t-stat of trend slope coefficient
            label:  label associated with the trend direction
                -1 : down trend
                +1 : up trend
        and where 'labels' shares the same index as 'prices'

        """
        roll = self.prices.rolling(window=self.hi, min_periods=self.hi)

        labels = roll.apply(self._make_label).to_frame('t').dropna()
        labels['end'] = labels.index.date
        labels['label'] = misc.sign(labels['t'])
        labels.index -= pd.offsets.BDay(self.hi - 1)
        labels.index.name = 'start'

        self.labels = labels.get(['end', 't', 'label'])
        self._make_weights()

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
        self.labels['side'] = misc.sign(side_aligned)
        self.labels['meta_label'] = (
                self.labels['label'] == self.labels['side']
        ).astype(int)

    def describe(self):
        """
        Return string showing label statistics we care
        about including:
            label
                count - number of events/labels
                down - number of -1 down trends
                up - number of +1 up trends
        """
        if self.labels is None:
            err = 'labels empty: nothing to describe'
            raise AttributeError(err)

        divider = '-' * 45

        label_freq, _ = np.histogram(self.labels['label'], 2)
        label_freq = [label_freq.sum()] + list(label_freq)
        label_desc = pd.Series(label_freq, ['count', 'down', 'up'])

        msg = (
            f'LABEL \n'
            f'{label_desc}'
        )

        return msg

    def plot_labels(self):
        """
        Plot the prices and color each data point depending
        on label and t-static value
        """
        plt.close('all')

        fig, axes = plt.subplots(2, 1, dpi=200, figsize=(11, 8), sharex='all')

        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        else:
            axes = [axes]

        labels, prices = self.labels.align(self.prices, axis=0, join='inner')

        axes[0].scatter(prices.index, prices.values, c=labels['label'], cmap='viridis')
        axes[0].title.set_text('labels')

        axes[1].scatter(prices.index, prices.values, c=labels['t'], cmap='viridis')
        axes[1].title.set_text('t-stat')

        plt.show()

    def plot_weights(self):
        """
        Plot histogram of sample weights
        """
        assert self.weights is not None

        n_bins = int(np.sqrt(len(self.weights)))
        n_per_bin = len(self.weights) / n_bins
        kw = {
            'linestyle': 'dashed',
            'linewidth': 1,
        }

        plt.close('all')
        plt.hist(self.weights, bins=n_bins, alpha=0.75)
        plt.axhline(n_per_bin, color='r', label='equal', **kw)
        plt.axvline(self.weights.mean(), color='g', label='mean', **kw)
        plt.legend()
        plt.title('weights')
        plt.show()


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
    ts = TrendScanning(prices, low=21, high=65)
    # ts.make_labels()
    ts.make_meta_labels(side=prices)
    t1 = time.time()
    print(f'{t1 - t0:0.4f} seconds')
    print(ts)
    print(ts.describe())
    ts.plot_labels()
    ts.plot_weights()



