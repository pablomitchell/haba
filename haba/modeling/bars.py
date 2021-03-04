"""
information driven bars
"""
from abc import ABC, abstractmethod

import numpy as np
import numba
import pandas as pd

from haba.util import misc


def ewa(seq, window):
    return pd.Series(seq).ewm(window).mean().values[-1]


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

    def make_bars_XX(self, num_prev_bars=5, expected_num_ticks_init=252):
        expected_num_ticks = expected_num_ticks_init
        expected_imbalance = None
        cum_theta = num_ticks = 0
        imbalance_array = []
        imbalance_bars = []
        bar_length_array = []

        for date, row in self.data.iloc[1:].iterrows():
            num_ticks += 1
            imbalance = row['tick'] * row['other']
            imbalance_array.append(imbalance)
            cum_theta += imbalance

            print(
                f'\n'
                f'date               = {date} \n'
                f'num_ticks          = {num_ticks} \n'
                f'imbalance          = {imbalance} \n'
                f'abs_cum_theta      = {abs(cum_theta)}'
            )

            if len(imbalance_bars) == 0 and len(imbalance_array) >= expected_num_ticks_init:
                expected_imbalance = ewa(imbalance_array, window=expected_num_ticks_init)

            if expected_imbalance is None:
                continue

            if abs(cum_theta) >= expected_num_ticks * abs(expected_imbalance):
                imbalance_bars.append(row)
                bar_length_array.append(num_ticks)
                cum_theta = num_ticks = 0
                expected_num_ticks = ewa(bar_length_array, window=num_prev_bars)
                expected_imbalance = ewa(imbalance_array, window=num_prev_bars * expected_num_ticks)

            print(
                f'expected_theta     = {abs(expected_num_ticks * expected_imbalance)} \n'
                f'expected_imbalance = {expected_imbalance} \n'
                f'expected_num_ticks = {expected_num_ticks} \n'
            )
            input('ENTER')

    def make_bars(self, exp_nticks=21, nbars=5):
        nticks = 0
        tick_index = []

        run_up_list = []
        run_down_list = []

        theta = 0
        exp_theta = None

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
                # first calculation of expected values
                exp_run_up = ewa(run_up_list, nbars * exp_nticks)
                exp_run_down = ewa(run_down_list, nbars * exp_nticks)
                exp_run = max(exp_run_up, exp_run_down)
                exp_theta = exp_nticks * exp_run
                print(
                    f'nticks       = {nticks} \n'
                    f'exp_nticks   = {exp_nticks} \n'
                    f'exp_run_up   = {exp_run_up} \n'
                    f'exp_run_down = {exp_run_down} \n'
                    f'exp_run      = {exp_run} \n'
                    f'theta        = {theta} \n'
                    f'exp_theta    = {exp_theta} \n'
                )

            if exp_theta is None:
                continue

            if theta >= exp_theta:
                tick_index.append(nticks)
                bars.append(row)
                # update expected values
                exp_nticks = ewa(tick_index, nbars)
                exp_run_up = ewa(run_up_list, nbars * exp_nticks)
                exp_run_down = ewa(run_down_list, nbars * exp_nticks)
                exp_run = max(exp_run_up, exp_run_down)
                exp_theta = exp_nticks * exp_run
                # print(
                #     f'nticks       = {nticks} \n'
                #     f'exp_nticks   = {exp_nticks} \n'
                #     f'exp_run_up   = {exp_run_up} \n'
                #     f'exp_run_down = {exp_run_down} \n'
                #     f'exp_run      = {exp_run} \n'
                #     f'theta        = {theta} \n'
                #     f'exp_theta    = {exp_theta} \n'
                # )
                # input('ENTER')
                # reset
                theta = 0
                nticks = 0

        bars = pd.DataFrame(bars)
        bars['tick_index'] = tick_index

        n = 20
        print(bars.head(n=n).to_string())
        print(bars.tail(n=n).to_string())
        print(np.mean(tick_index))



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
    # prices.to_csv('foo.csv')
    # exit()

    bars = RunBars(prices)
    bars.make_bars(exp_nticks=25, nbars=1)

    # import matplotlib.pyplot as plt
    # bars.data.tick.cumsum().plot()
    # plt.show()
    # exit()



