"""
information driven bars
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from haba.util import misc


class BaseBars(ABC):

    def __init__(self, prices, other=None):
        assert not prices.isnull().any(), 'prices not allowed to have NAs'

        if other is None:
            other = pd.Series(1.0, index=prices.index)

        assert not other.isnull().any(), 'prices not allowed to have NAs'

        rets = np.log(prices).diff()
        ticks = misc.sign(rets)

        self.bars = pd.DataFrame({
            'price': prices,
            'other': other,
            'return': rets,
            'tick': ticks,
            # 'length': 0,
            # 'number': 0,
        })

    def __repr__(self):
        msg = str()
        show_these_dfs = [
            'bars',
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

    def _make_bar(self, arr):
        """
        Parameters
        ----------
        arr : numpy.ndarray

        Returns
        -------
        length : int
            the length of the bar


        """
        return 100

    def make_bars(self):
        index = 0
        indices = []

        while True:
            arr = self.bars.iloc[index:].values
            index += self._make_bar(arr)

            if len(self.bars) <= index:
                break

            indices.append(index)

        indices = np.array(indices)
        dates = self.bars.index[indices]

        self.bars['length'] = pd.Series(
            np.diff(indices, prepend=0),
            index=dates,
        )
        self.bars['number'] = pd.Series(
            np.cumsum(np.ones_like(indices)),
            index=dates,
        )






class ImbalanceBars(BaseBars):
    pass


class RunBars(BaseBars):
    pass


if __name__ == '__main__':
    from haba.util.tseries import generate_prices

    start = '1990-01-01'
    end = '2020-12-31'
    drift = 0.0
    volatility = 0.16
    prices = generate_prices(start, end, drift, volatility)

    bars = RunBars(prices)
    print(bars)

    bars.make_bars()