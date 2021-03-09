"""
information driven bars
"""
from abc import ABC, abstractmethod

import numpy as np
import numba
import pandas as pd

from haba.util import misc


@numba.njit
def numba_ewa(arr, span):
    decay = (span - 1.0) / (span + 1.0)
    periods = len(arr)
    numerator = denominator = 0.0

    for t in range(periods):
        coefficient = decay ** float(t)
        numerator += coefficient * arr[periods - t - 1]
        denominator += coefficient

    return numerator / denominator

def ewa_fast(seq, span):
    arr = np.array(seq)
    return numba_ewa(arr, span)

def ewa_slow(seq, span):
    return pd.Series(seq).ewm(span=span).mean().values[-1]

ewa = ewa_fast


class BaseBars(ABC):

    def __init__(self, prices, other=None):
        assert not prices.isnull().any(), 'prices not allowed to have NAs'

        if other is None:
            other = pd.Series(1.0, index=prices.index)

        assert not other.isnull().any(), 'other not allowed to have NAs'

        rets = np.log(prices).diff()
        ticks = misc.sign(rets)

        self.data = pd.DataFrame({
            'price': prices,
            'other': other,
            'return': rets,
            'tick': ticks,
        })
        self.bars = None

    def __repr__(self):
        msg = str()
        show_these_dfs = [
            'data',
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

    @abstractmethod
    def make_bars(self, exp_nticks=25, nbars=2):
        pass


class ImbalanceBars(BaseBars):

    def make_bars(self, exp_nticks=10, nbars=1):
        """
        Parameters
        ----------
        exp_nticks : int
        nbars : int

        """
        exp_theta = None
        theta = 0
        nticks = 0
        tick_index = []
        imb_up_list = []
        imb_down_list = []
        bars = []

        for date, row in self.data.iloc[1:].iterrows():
            nticks += 1

            if 0 <= row['tick']:
                imb_up_list.append(row['other'])
                imb_down_list.append(0)
            else:
                imb_up_list.append(0)
                imb_down_list.append(row['other'])

            theta = sum(imb_up_list[-nticks:]) - sum(imb_down_list[-nticks:])

            if len(tick_index) == 0 and exp_nticks > len(imb_up_list):
                # first update
                exp_imb_up = ewa(imb_up_list, nbars * exp_nticks)
                exp_imb_down = ewa(imb_down_list, nbars * exp_nticks)
                exp_imb = exp_imb_up - exp_imb_down
                exp_theta = exp_nticks * exp_imb

                print(
                    f'nticks        = {nticks} \n'
                    f'exp_nticks    = {exp_nticks} \n'
                    f'exp_imb       = {exp_imb} \n'
                    f'theta         = {theta} \n'
                    f'exp_theta     = {exp_theta} \n'
                )

            if exp_theta is None:
                continue

            if abs(theta) >= abs(exp_theta):
                # store
                tick_index.append(nticks)
                bars.append(row)
                # update
                exp_nticks = ewa(tick_index, nbars)
                exp_imb_up = ewa(imb_up_list, nbars * exp_nticks)
                exp_imb_down = ewa(imb_down_list, nbars * exp_nticks)
                exp_imb = exp_imb_up - exp_imb_down
                exp_theta = exp_nticks * exp_imb
                # reset
                theta = 0
                nticks = 0

        bars = pd.DataFrame(bars)
        bars['tick_index'] = tick_index
        self.bars = bars



class RunBars(BaseBars):

    def make_bars(self, exp_nticks=20, nbars=2):
        """
        Parameters
        ----------
        exp_nticks : int
        nbars : int

        """
        exp_theta = None
        theta = 0
        nticks = 0
        tick_index = []
        run_up_list = []
        run_down_list = []
        bars = []

        for date, row in self.data.iloc[1:].iterrows():
            nticks += 1

            if 0 <= row['tick']:
                run_up_list.append(row['other'])
                run_down_list.append(0)
            else:
                run_up_list.append(0)
                run_down_list.append(row['other'])

            theta = max(sum(run_up_list[-nticks:]), sum(run_down_list[-nticks:]))

            if len(tick_index) == 0 and exp_nticks > len(run_up_list):
                # first update
                exp_run_up = ewa(run_up_list, nbars * exp_nticks)
                exp_run_down = ewa(run_down_list, nbars * exp_nticks)
                exp_run = max(exp_run_up, exp_run_down)
                exp_theta = exp_nticks * exp_run

            if exp_theta is None:
                continue

            if theta >= exp_theta:
                # store
                tick_index.append(nticks)
                bars.append(row)
                # update
                exp_nticks = ewa(tick_index, nbars)
                exp_run_up = ewa(run_up_list, nbars * exp_nticks)
                exp_run_down = ewa(run_down_list, nbars * exp_nticks)
                exp_run = max(exp_run_up, exp_run_down)
                exp_theta = exp_nticks * exp_run
                # reset
                theta = 0
                nticks = 0

        bars = pd.DataFrame(bars)
        bars['tick_index'] = tick_index
        self.bars = bars



if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from haba.util.tseries import generate_prices

    start = '1990-01-01'
    end = '2020-12-31'
    drift = 0.10
    volatility = 0.18
    prices = generate_prices(start, end, drift, volatility)
    volume = pd.Series(
        np.random.randint(40_000_000, 100_000_000, len(prices)),
        index=prices.index,
    )
    dollars = prices * volume
    other = dollars
    # other = None

    t0 = time.time()
    ibs = ImbalanceBars(prices, other)
    ibs.make_bars(exp_nticks=20, nbars=2.7)
    t1 = time.time()
    print(f'elapsed: {t1 - t0:0.2f} seconds')
    ibs.bars['tick_index'].plot(title='ibs')
    plt.show()
    print(ibs.bars.describe())

    exit()

    t2 = time.time()
    rbs = RunBars(prices, other)
    rbs.make_bars(exp_nticks=20, nbars=2)
    t3 = time.time()
    print(f'elapsed: {t3 - t2:0.2f} seconds')
    rbs.bars['tick_index'].plot(title='rbs')
    plt.show()
    print(rbs.bars.describe())



