"""
Triple barrier labeling
"""

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.style.use('seaborn')


class TripleBarrier(object):

    def __init__(self, prices, span, scale, holding_period):
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

        """
        self.prices = prices
        self.scale = scale
        self.holding_period = holding_period

        self.returns = self._get_returns()
        self.volatility = self._get_volatility()
        self.events = self._get_events()
        self.barriers = self._get_barriers()

        self.labels = None

    def __repr__(self):
        msg = str()
        show_these_dfs = [
            'prices',
            'volatility',
            'barriers',
        ]
        divider = '-' * 45

        if self.labels is not None:
            show_these_dfs.append('labels')

        for name in show_these_dfs:
            df = getattr(self, name)
            msg += (
                f'{name.upper()} \n'
                f'{df.head()} \n'
                f'\t...\n'
                f'{df.tail()} \n'
                f'{divider} \n'
            )

        return msg

    @staticmethod
    def _sgn(ser):
        return np.sign(ser).astype(int)

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
            self.scale['top'] * vol,
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
            'top': top,
            'vertical': vertical,
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
            sum_pos = max(0, sum_pos + ret)

            if vol < abs(sum_neg):
                sum_neg = 0
                events.append(dt)
                continue

            if vol < abs(sum_pos):
                sum_pos = 0
                events.append(dt)
                continue

        return pd.DatetimeIndex(events)

    def _get_returns(self):
        return np.log(self.prices).diff()

    def _get_volatility(self):
        return self.returns.ewm(min_periods=span, span=span).std().dropna()

    @staticmethod
    def _desc(ser):
        return pd.Series({
            'count': ser.count(),
            'mean': ser.mean(),
            'median': ser.median(),
            'std': ser.std(),
            'min': ser.min(),
            'max': ser.max(),
        })

    def describe(self):
        if self.labels is None:
            err = 'labels empty: nothing to describe'
            raise AttributeError(err)

        msg = str()

        label_freq, _ = np.histogram(self.labels['label'], 3)
        label_freq = [label_freq.sum()] + list(label_freq)
        label_desc = pd.Series(label_freq, ['count', 'bottom', 'vertical', 'top'])
        days_desc = self._desc(self.labels['days']).astype(int)

        divider = '-'*45

        msg += (
            f'LABEL \n'
            f'{label_desc} \n'
            f'{divider} \n'
            f'DAYS \n'
            f'{days_desc} \n'
            f'{divider} \n'
        )

        return msg

    def make_labels(self):
        labels = {}

        for start, end in self.barriers['vertical'].iteritems():
            rets = np.log(self.prices.loc[start:end]).diff().cumsum()
            top = self.barriers.at[start, 'top']
            bottom = self.barriers.at[start, 'bottom']

            touches = {
                'top': rets[rets > top].index.min(),
                'vertical': end,
                'bottom': rets[rets < bottom].index.min(),
            }
            touches = pd.Series(touches)

            idx = touches.argmin()
            date = touches[idx]
            barrier = touches.index[idx]
            label_ = self._barrier_to_label(barrier)

            labels[start] = {
                'touch': date,
                'vertical': end,
                'days': pd.bdate_range(start, date).size,
                'ret_sgn': self._sgn(rets.loc[date]),
                'label': label_,
            }

        self.labels = pd.DataFrame.from_dict(labels, orient='index')

    def make_meta_labels(self, side):
        if self.labels is None:
            self.make_labels()

        self.labels, side_aligned = self.labels.align(side, axis=0, join='inner')
        self.labels['side'] = self._sgn(side_aligned)
        self.labels['meta_label'] = \
            self._sgn(self.labels['label'] * self.labels['side'])

    def plot(self, n_samples=None):
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

        ann_vol = np.sqrt(260) * self.volatility
        ave_vol = ann_vol.mean()

        axes[0].plot(ann_vol.index, ann_vol)
        axes[0].title.set_text(f'Volatility (mean {ave_vol:0.2f})')

        axes[1].plot(self.prices.index, self.prices)
        axes[1].title.set_text('Prices')

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
            axes[1].add_patch(rect_top)
            axes[1].add_patch(rect_bottom)

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    from haba.util.tseries import generate_prices

    start = '1990-01-01'
    end = '2020-12-31'
    drift = 0.07
    volatility = 0.15
    prices = generate_prices(start, end, drift, volatility)

    span = 130
    scale = {
        'events': 3,
        'top': 3,
        'bottom': 3,
    }
    holding_period = 15
    tb = TripleBarrier(
        prices,
        span=span,
        scale=scale,
        holding_period=holding_period,
    )
    print(tb)

    # n_samples = None
    # n_samples = (len(prices) - span) // span
    # tb.plot(n_samples=n_samples)

    tb.make_labels()
    print(tb)

    side = tb.returns.ewm(min_periods=65, span=65).mean().dropna()
    tb.make_meta_labels(side)
    print(tb)

    print(tb.describe())




