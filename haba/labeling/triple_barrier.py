"""
Triple barrier labeling
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from haba.differencing.fractional import Fractional
from haba.util import misc

plt.style.use('seaborn')


class TripleBarrier(object):

    def __init__(self, prices, span, scale, holding_period,
                 fractional_difference=False, sample_method=None):
        """
        Parameters
        ----------
        prices : pandas.Series
            series of close price data with pandas.DatetimeIndex
        span : int
            specifies the span in days to use when computing the
            exponentially weighted volatility
        scale : dict of floats
            used to scale volatility dependent parameters in units
            of daily volatility
                scale['events'] - value used to scale the size of
                    the events threshold in units of daily volatility
                scale['bottom'] - value used to scale the size of
                    bottom barrier in units of daily volatility
                scale['top'] - value used to scale the size of top
                    barrier in units of daily volatility
        holding_period : int
            the maximum holding period in days defining the vertical
            barrier
        fractional_difference : bool
            use fraction differencing when computing returns rather
            than regular differencing -- defaults to False
        sample_method : str, default 'uniqueness'
            the method used to construct sample weights
               possible method are:  ['returns', 'uniqueness']
        """
        self.prices = prices
        self.span = span
        self.scale = scale
        self.holding_period = holding_period
        self.fractional_difference = fractional_difference
        self.sample_method = sample_method

        if self.sample_method is None:
            self.sample_method = 'uniqueness'

        self.returns = self._get_returns()
        self.volatility = self._get_volatility()
        self.events = self._get_events()
        self.barriers = self._get_barriers()

        self.labels = None

        # sample weight related objects
        self.ind_matrix = None
        self.ret_matrix = None
        self.weights = None

    def __repr__(self):
        msg = str()
        show_these_dfs = [
            'prices',
            'volatility',
            'barriers',
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

    @property
    def sample_methods(self):
        return [
            'returns',
            'uniqueness',
        ]

    def _barrier_to_label(self, barrier):
        return {
            'top': +1,
            'vertical': 0,
            'bottom': -1,
        }.get(barrier)

    def _get_horizontal_barriers(self):
        # profit taking (top) and stop loss (bottom) barriers
        vol = self.volatility.loc[self.events]

        return (
            -self.scale['bottom'] * vol,
            +self.scale['top'] * vol,
        )

    def _get_vertical_barriers(self):
        # Generate a pandas.Series defining the maximum holding
        # period (vertical barriers) where the index defines the
        # start date and the series value the end date
        offset = pd.offsets.BDay(self.holding_period)
        idx = self.prices.index.searchsorted(self.events + offset)
        idx = idx[idx < len(self.prices)]

        return pd.Series(
            self.prices.index[idx],
            index=self.events[:len(idx)],
        )

    def _get_barriers(self):
        # Create triple barrier dataframe
        bottom, top = self._get_horizontal_barriers()
        vertical = self._get_vertical_barriers()
        data = {
            'bottom': bottom,
            'vertical': vertical,
            'top': top,
        }
        barriers = pd.concat(data, axis=1).dropna()
        self.events = barriers.index

        return barriers

    def _get_events(self):
        # Sample events using a symmetric cumulative sum filter
        events = []
        sum_neg = sum_pos = 0

        for dt in self.volatility.index:
            vol = self.volatility.loc[dt] * self.scale['events']
            ret = self.returns.loc[dt]

            sum_neg = min(0, sum_neg + ret)

            if vol < abs(sum_neg):
                sum_neg = 0
                events.append(dt)
                continue

            sum_pos = max(0, sum_pos + ret)

            if vol < abs(sum_pos):
                sum_pos = 0
                events.append(dt)
                continue

        return pd.DatetimeIndex(events)

    def _get_returns(self):
        rets = np.log(self.prices).diff()

        if self.fractional_difference:
            # err = 'fractional differencing not implemented yet'
            # raise NotImplementedError(err)
            rets = Fractional(rets.cumsum()).difference()

        return rets

    def _get_volatility(self):
        return self.returns.ewm(
            min_periods=self.span,
            span=self.span,
        ).std().dropna()

    def _make_weights(self):
        assert self.sample_method in self.sample_methods

        if self.sample_method == 'uniqueness':
            top = self.ind_matrix
        elif self.sample_method == 'returns':
            top = self.ret_matrix.abs()

        weight = top.div(self.ind_matrix.sum(axis=0), axis=1).mean(axis=1)
        weight = weight.div(weight.sum())

        self.weights = weight

    def make_labels(self):
        """
        Populates 'labels' dataframe with the following columns:
            touch:  date of the first barrier touch
            vertical:  date of the vertical barrier
            days:  elapsed days from event date to first touch
            sign:  sign of the resulting return
            label:  label associated with the barrier touched
                -1 : bottom barrier
                 0 : vertical barrier
                +1 : top barrier
        and where 'labels' shares the same index as 'events'.
        """
        labels = {}

        # work around since pandas.DataFrame.loc can be slow
        shape = len(self.events), len(self.volatility.index)
        ind_matrix = np.zeros(shape)
        ret_matrix = np.zeros(shape)

        for start, end in self.barriers['vertical'].iteritems():
            rets = np.log(self.prices.loc[start:end]).diff().fillna(0)
            cum_rets = rets.cumsum()

            # populating numpy array then instantiating a
            # dataframe is an order of magnitude faster
            # than using .loc directly on a the dataframe
            idx0 = self.events.get_loc(start)
            col0 = self.volatility.index.get_loc(start)
            col1 = self.volatility.index.get_loc(end) + 1
            ind_matrix[idx0, col0:col1] = 1
            ret_matrix[idx0, col0:col1] = rets.values

            top = self.barriers.at[start, 'top']
            bottom = self.barriers.at[start, 'bottom']
            touches = {
                'top': cum_rets[cum_rets > top].index.min(),
                'vertical': end,
                'bottom': cum_rets[cum_rets < bottom].index.min(),
            }
            touches = pd.Series(touches)

            idx = touches.argmin()
            date = touches[idx]
            barrier = touches.index[idx]
            labels[start] = {
                'touch': date,
                'end': end,
                'days': pd.bdate_range(start, date).size - 1,
                'sign': misc.sign(cum_rets.loc[date]),
                'label': self._barrier_to_label(barrier),
            }

        self.labels = pd.DataFrame.from_dict(labels, orient='index')
        self.labels.index.name = 'start'

        self.ind_matrix = pd.DataFrame(
            ind_matrix,
            index=self.events,
            columns=self.volatility.index,
        )
        self.ret_matrix = pd.DataFrame(
            ret_matrix,
            index=self.events,
            columns=self.volatility.index,
        )
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
                bottom - number of -1 touches
                vertical - number of 0 touches
                top - number of +1 touches
            days (trading)
                count - number of events/labels
                mean - average
                median - median
                std - standard deviation
                min - minimum
                max - maximum
        """
        if self.labels is None:
            err = 'labels empty: nothing to describe'
            raise AttributeError(err)

        divider = '-' * 45

        label_freq, _ = np.histogram(self.labels['label'], 3)
        label_freq = [label_freq.sum()] + list(label_freq)
        label_desc = pd.Series(label_freq, ['count', 'bottom', 'vertical', 'top'])
        days_desc = misc.desc(self.labels['days']).astype(int)

        msg = (
            f'LABEL \n'
            f'{label_desc} \n'
            f'{divider} \n'
            f'DAYS \n'
            f'{days_desc} \n'
            f'{divider} \n'
        )

        return msg

    def plot_labels(self, n_samples=None):
        """
        Plot the volatility and prices with triple barriers
        super-imposed over the prices.  For longer time
        horizons, limit the number of triple barriers shown
        on the plot by selecting a value for `n_sample'.

        Parameters
        ----------
        n_samples : int
            number of triple barriers to sample and place
            on the plot, must be less than the number of
            events
        """
        plt.close('all')

        fig, axes = plt.subplots(2, 1, dpi=200, figsize=(11, 8), sharex='all')

        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        else:
            axes = [axes]

        axes[0].plot(self.prices.index, self.prices)
        axes[0].title.set_text('prices')

        ann_vol = 100 * np.sqrt(260) * self.volatility
        axes[1].plot(ann_vol.index, ann_vol)
        axes[1].title.set_text('volatility')

        events = self.events.to_series()

        if n_samples:
            assert n_samples <= len(self.events)
            events = self.events.to_series().sample(n_samples)

        kwargs = {
            'linewidth': 2,
            'color': 'red',
            'fill': False,
        }

        for dt in events:
            barriers = self.barriers.loc[dt]
            price = self.prices.loc[dt]

            start = mdates.date2num(dt)
            end = mdates.date2num(barriers.vertical)
            xy = (start, price)
            width = end - start

            rect_top = Rectangle(xy, width, price * barriers.top, **kwargs)
            rect_bottom = Rectangle(xy, width, price * barriers.bottom, **kwargs)
            axes[0].add_patch(rect_top)
            axes[0].add_patch(rect_bottom)

        fig.tight_layout()
        plt.show()

    def plot_weights(self):
        """
        Plot histogram of sample weights
        """
        assert self.weights is not None

        plt.close('all')
        plt.hist(self.weights, bins='sqrt', alpha=0.75)
        plt.show()


if __name__ == '__main__':
    from haba.util.tseries import generate_prices

    start = '1990-01-01'
    end = '2020-12-31'
    drift = 0.0
    volatility = 0.16
    prices = generate_prices(start, end, drift, volatility)

    span = 65
    scale = {
        'events': 3,
        'top': 3,
        'bottom': 3,
    }
    holding_period = 15
    sample_method = 'returns'
    fractional_difference = True
    plot = False

    # import cProfile, pstats
    # pfile = 'triple_barrier.profile'
    # cProfile.run(
    #     'TripleBarrier(prices, span, scale, holding_period,'
    #     'fractional_difference=fractional_difference,'
    #     'sample_method=sample_method)'
    #     '.make_meta_labels(side=prices)',
    #     pfile
    # )
    # s = pstats.Stats(pfile)
    # s.strip_dirs()
    # s.sort_stats('cumtime').print_stats(15)
    # exit()

    import time
    t0 = time.time()
    tb = TripleBarrier(
        prices, span, scale, holding_period,
        fractional_difference=fractional_difference,
        sample_method=sample_method,
    )
    tb.make_meta_labels(side=prices)
    t1 = time.time()
    tb.plot_labels(n_samples=len(tb.events)//10)
    tb.plot_weights()
    print(f'{t1 - t0:0.4f} seconds')