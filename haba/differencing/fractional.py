from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


plt.style.use('seaborn')


class Fractional(object):

    def __init__(self, ser, cumulate=False):
        """
        Compute time-series fractional difference
            https://en.wikipedia.org/wiki/Autoregressive_fractionally_integrated_moving_average

        Parameters
        ----------
        ser : pandas.Series
            time-series to difference
        cumulate : bool
            whether or not to arithmetically cumualte the series
            prior to differencing

        """
        self.cumulate = cumulate
        self.ser = ser

    @property
    def ser(self):
        return self._ser

    @ser.setter
    def ser(self, s):
        self._ser = s

        if self.cumulate:
            self._ser = self._ser.cumsum()

    def _get_weights(self, order, size):
        w = pd.Series(np.arange(size))

        # binomial coefficient generating function
        gfunc = lambda k: (-1 ** k) * (order - k + 1) / k

        w[1:] = w[1:].apply(gfunc).cumprod()
        w[0] = 1

        return w[::-1]

    def difference(self, order=0.6, size=65):
        """
        Performs the differencing

        Parameters
        ----------
        order : float
            the fractional order of the difference,
            must be in the open unit interval: (0, 1)
        size : int
            the number of coefficients to keep in the
            fixed size differencing equation, must
            be in the closed interval: [2, len(ser)]

        Returns
        -------
        fractional_difference : pandas.Series

        """
        order = float(order)
        size = int(size)

        assert (0 < order) and (order < 1)
        assert (2 <= size) and (size <= len(self.ser))

        weights = self._get_weights(order, size)

        return (self.ser
            .fillna(method='ffill')
            .rolling(size)
            .apply(lambda x: np.einsum('i,i', weights, x))
            .dropna()
        )

    @staticmethod
    def _rank(ser):
        kwargs = {
            'ascending': True,
            'na_option': 'top',
            'pct': True,
        }
        return ser.rank(**kwargs)

    def test(self, order_range, size_range):
        """
        Test a range of order/size fractional differences then
        rank the results by a composite weighting of correlation
        (memory) an ADF (stationarity).  Prints a sorted dataframe
        and plots the top ranked case.

        Parameters
        ----------
        order_range : seq
            sequence of floats providing a range of 'order' values
        size_range : seq
            sequence of ints providing a range of 'size' values

        """
        columns = ['order', 'size', 'obs', 'corr', 'adf', 'adf_crit', 'p_val']
        out = pd.DataFrame(columns=columns)
        grid = product(order_range, size_range)

        for ii, (order, size) in enumerate(grid):
            ser_diff = self.difference(order, size)
            adf = adfuller(ser_diff, maxlag=1, regression='c', autolag=None)
            corr = self.ser.corr(ser_diff)
            out.loc[ii] = [order, size, adf[3], corr, adf[0], adf[4]['1%'], adf[1]]
            print(ii)

        out.query('(adf < adf_crit) and (0.5 < corr)', inplace=True)
        out['rank_adf'] = self._rank(-out['adf'])
        out['rank_corr'] = self._rank(out['corr'])
        out['rank'] = self._rank(out['rank_adf'] + out['rank_corr'])
        out.sort_values('rank', inplace=True)

        print(out.to_string(float_format='{:0.4f}'.format))

        order = out['order'].iloc[-1]
        size = out['size'].iloc[-1]
        ser_diff = self.difference(order=order, size=size)
        corr = self.ser.corr(ser_diff)
        self.ser.plot()
        ser_diff.plot(secondary_y=True)
        plt.title(f'order: {order:0.2f}, size: {size:0.0f}, corr: {corr:0.2f}')
        plt.show()
