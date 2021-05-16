# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


from .utils import check_X_y

import numpy as np
from scipy import stats


def T_test(X, y):
    X, y = check_X_y(X, y)

    n_sample, n_feature = X.shape

    p_values = []
    for i in range(n_feature):
        x1, x2 = X[:, i][y == 0], X[:, i][y == 1]
        if np.var(X[:, i]) == 0:
            p_value = 1
        else:
            _, p_value = stats.ttest_ind(x1, x2)
        p_values.append(p_value)

    return -np.array(p_values)

