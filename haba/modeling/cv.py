"""
cross-validation framework
"""

import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import _deprecate_positional_args


class PurgedKFold(_BaseKFold):
    """
    Extend KFold to work with labels/features that span intervals. The
    train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training examples
    in between.
    """
    def __init__(self, n_splits=3, times=None, pct_embargo=0):

        if not isinstance(times, pd.Series):
            raise ValueError('times must be type pandas.Series')

        super().__init__(n_splits, shuffle=False, random_state=None)

        self.times = times
        self.pct_embargo = pct_embargo


if __name__ == '__main__':
    pk = PurgedKFold(times=pd.Series())
    print(pk)